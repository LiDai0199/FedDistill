import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer

class LogitTracker():
    def __init__(self, unique_labels):
        self.unique_labels = unique_labels
        self.labels = [i for i in range(unique_labels)]
        self.label_counts = torch.ones(unique_labels) # avoid division by zero error
        self.logit_sums = torch.zeros((unique_labels,unique_labels) )

    def update(self, logits, Y):
        """
        update logit tracker.
        :param logits: shape = n_sampls * logit-dimension
        :param Y: shape = n_samples
        :return: nothing
        """
        batch_unique_labels, batch_labels_counts = Y.unique(dim=0, return_counts=True)
        self.label_counts[batch_unique_labels] += batch_labels_counts
        # expand label dimension to be n_samples X logit_dimension
        labels = Y.view(Y.size(0), 1).expand(-1, logits.size(1))
        logit_sums_ = torch.zeros((self.unique_labels, self.unique_labels) )
        logit_sums_.scatter_add_(0, labels, logits)
        self.logit_sums += logit_sums_


    def avg(self):
        res = self.logit_sums / self.label_counts.float().unsqueeze(1)
        return res


class UserFedDistill(User):
    """
    Track and average logit vectors for each label, and share it with server/other users.
    """
    def __init__(self, args, id, model, train_data, test_data, unique_labels, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

        self.init_loss_fn()
        self.unique_labels = unique_labels
        self.label_counts = {}
        self.logit_tracker = LogitTracker(self.unique_labels)
        self.global_logits = None
        self.reg_alpha = 1  # 这是一个权重因子，用于平衡train_loss和reg_loss之间的贡献。调整这个参数可以控制正则化项对总损失的影响程度，进而影响模型训练的焦点。

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=True, lr_decay=True, count_labels=True, verbose=True):
        self.clean_up_counts()
        self.model.train()
        REG_LOSS, TRAIN_LOSS = 0, 0
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for i in range(self.K):
                result =self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'], result['y']
                if count_labels:
                    self.update_label_counts(result['labels'], result['counts'])
                self.optimizer.zero_grad()
                result=self.model(X, logit=True)
                output, logit = result['output'], result['logit']
                self.logit_tracker.update(logit, y)
                if self.global_logits != None:
                    ### get desired logit for each sample
                    train_loss = self.loss(output, y)  # 模型预测输出与真实标签之间的差异, 它反映了模型在训练数据上的性能
                    target_p = F.softmax(self.global_logits[y,:], dim=1)  # 老师模型对于输入数据的预测概率分布
                    reg_loss = self.ensemble_loss(output, target_p)  # 模型预测的概率分布与老师模型的预测概率分布之间的差异

                    # 累加损失，可以在一轮训练结束后获得总的训练损失和正则化损失, 可以通过计算平均损失（即总损失除以批次数量或样本数量）来监控模型的训练进度，判断模型是否正在学习，以及是否需要调整训练参数（如学习率、正则化系数等）。
                    REG_LOSS += reg_loss
                    TRAIN_LOSS += train_loss

                    # 正则化损失 reg_loss = 模型预测输出与老师模型输出之间的差异
                    # 训练损失 train_loss = 模型预测输出与真实标签之间的差异
                    loss = train_loss + self.reg_alpha * reg_loss  # 训练过程中的总损失，它将传统的训练损失（train_loss）与正则化损失（reg_loss）结合起来
                else:
                    loss=self.loss(output, y)
                loss.backward()
                self.optimizer.step()#self.local_model)
            # local-model <=== self.model
            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        if lr_decay:
            self.lr_scheduler.step(glob_iter)
        if self.global_logits != None and verbose:
            REG_LOSS = REG_LOSS.detach().numpy() / (self.local_epochs * self.K)
            TRAIN_LOSS = TRAIN_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info = "Train loss {:.2f}, Regularization loss {:.2f}".format(REG_LOSS, TRAIN_LOSS)
            print(info)



