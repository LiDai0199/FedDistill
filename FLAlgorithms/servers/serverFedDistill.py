import time

from FLAlgorithms.users.userFedDistill import UserFedDistill
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data, aggregate_user_test_data
import numpy as np

class FedDistill(Server):
    """
    实现了FedDistill算法，创建了一个FedDistill服务器，该服务器可以训练用户模型，聚合用户模型，评估用户模型，发送用户模型等。
    初始化服务器时，服务器会读取数据，创建用户，并将训练和测试数据分配给用户。
    """
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args.dataset)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.slow_start = 20
        self.share_model = 'FL' in self.algorithm
        self.pretrain = 'pretrain' in self.algorithm.lower()
        self.user_logits = None
        self.init_ensemble_configs()
        self.init_loss_fn()
        self.init_ensemble_configs()
        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data, label_info =read_user_data(i, data, dataset=args.dataset, count_labels=True)
            self.total_train_samples+=len(train_data)
            self.total_test_samples += len(test_data)
            id, train, test=read_user_data(i, data, dataset=args.dataset)
            user=UserFedDistill(
                args, id, model, train_data, test_data, self.unique_labels, use_adam=False)
            self.users.append(user)
        print("Loading testing data.")
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

    def train(self, args):
        #### pretraining ####
        if self.pretrain:
            ## before training ##
            for iter in range(self.num_pretrain_iters):
                print("\n\n-------------Pretrain iteration number: ", iter, " -------------\n\n")
                for user in self.users:
                    user.train(iter, personalized=True, lr_decay=True)
                self.evaluate(selected=False, save=False)
            ## after training ##
            if self.share_model:
                self.aggregate_parameters()
            self.aggregate_logits(selected=False) # aggregate label-wise logit vector

        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            if self.share_model:
                self.send_parameters(mode=self.mode)# broadcast averaged prediction model
            self.evaluate()  # evaluate global model performance
            self.send_logits()  # send global logits if have any
            random_chosen_id = np.random.choice(self.user_idxs)
            self.timestamp = time.time()  # log user-training start time
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                chosen = user_id == random_chosen_id
                user.train(
                    glob_iter,
                    personalized=True, lr_decay=True, count_labels=True, verbose=chosen)
            curr_timestamp = time.time()  # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            if self.share_model:
                self.aggregate_parameters()
            self.aggregate_logits()  # aggregate label-wise logit vector
            self.evaluate_personalized_model()

            self.save_results(args)
        self.save_model()

    def aggregate_logits(self, selected=True):
        """
        创建全局共享logits
        """
        user_logits = 0
        users = self.selected_users if selected else self.users  # 聚合那些已被选中参与当前训练轮次的用户或者聚合所有用户
        for user in users:
            user_logits += user.logit_tracker.avg()  # 获取每个用户的平均逻辑向量
        logits_data_size = user_logits.numel() * 4
        self.metrics['communication_overhead_upload'].append(logits_data_size )
        self.user_logits = user_logits / len(users)

    def send_logits(self):
        """
        将全局共享logits发送给所有用户
        """
        if self.user_logits == None: return
        for user in self.selected_users:
            user.global_logits = self.user_logits.clone().detach()
