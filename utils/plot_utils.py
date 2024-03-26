import matplotlib.pyplot as plt
import h5py
import numpy as np
from utils.model_utils import get_log_path, METRICS
import seaborn as sns
import matplotlib.colors as mcolors
import os

COLORS = list(mcolors.TABLEAU_COLORS)
MARKERS = ["o", "v", "s", "*", "x", "P"]
ALGO_INDEX = {"FedAvg": 0, "FedDistill": 1, "FedDistill-FL": 2}

plt.rcParams.update({'font.size': 14})
n_seeds = 1


def load_results(args, algorithm, seed):
    alg = get_log_path(args, algorithm, seed, args.gen_batch_size)
    hf = h5py.File("./{}/{}.h5".format(args.result_path, alg), 'r')  # 使用h5py库打开包含实验结果的.h5文件。
    metrics = {}
    for key in METRICS:
        metrics[key] = np.array(hf.get(key)[:])  # 对于列表中的每个指标名称，从.h5文件中加载相应的数据，并将其转换为numpy数组。
    return metrics


def get_label_name(name):
    name = name.split("_")[0]
    if 'Distill' in name:
        if '-FL' in name:
            name = 'FedDistill' + r'$^+$'
        else:
            name = 'FedDistill'
    elif 'FedDF' in name:
        name = 'FedFusion'
    elif 'FedEnsemble' in name:
        name = 'Ensemble'
    elif 'FedAvg' in name:
        name = 'FedAvg'
    return name


def plot_by_type(args, metrics, algorithm):
    plot_type = args.plot_type
    if plot_type == "loss":
        plot_losses(metrics, algorithm, 5, args.times)
    elif plot_type == "communication_overhead":
        plot_communication_overhead(metrics, algorithm)
    elif plot_type == "training_time":
        plot_training_time(metrics, algorithm)


def plot_metric_comparison(args, algorithms, ylabel):
    """通用函数来绘制不同算法的指定指标比较图。"""
    n_seeds = args.times
    dataset_ = args.dataset.split('-')
    algo_labels = ", ".join([get_label_name(algo) for algo in algorithms])
    plot_type = args.plot_type
    sub_dir = f"figs/{dataset_[0]}/{dataset_[2]}"
    os.makedirs(sub_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for algo in algorithms:
        metrics = [load_results(args, algo, seed) for seed in range(n_seeds)]
        plot_by_type(args, metrics, algo)

    plt.xlabel('Communication Rounds', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if len(algorithms) > 1:
        plt.title(f'Comparison of {algo_labels} ({plot_type})', fontsize=16)
    else:
        plt.title(f'{algo_labels} ({plot_type})', fontsize=16)

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    fig_save_path = os.path.join(sub_dir, f"{dataset_[0]}-{dataset_[2]}-{algo_labels.replace(', ', '_')}-{plot_type}.png")
    plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0.5, format='png', dpi=400)
    print(f'File saved to {fig_save_path}')
    plt.show()


def plot_losses(metrics, algo_name, TOP_N, n_seeds):
    # 计算精度
    final_acc_avg = sum(metrics[seed]['glob_acc'][-1] for seed in range(n_seeds)) / n_seeds  # 计算最终精度

    # 计算顶部精度
    top_accs = np.concatenate([np.sort(metrics[seed]['glob_acc'])[-TOP_N:] for seed in
                               range(n_seeds)])  # 从每个种子的精度结果中选取最高的TOP_N个精度值，然后将这些值合并为一个数组。
    acc_avg = np.mean(top_accs)
    acc_std = np.std(top_accs)  # 计算这些顶部精度值的标准偏差
    info = 'Algorithm: {:<10s}, Accuracy (top {:d} average) = {:.2f} %, deviation = {:.2f}'.format(algo_name, TOP_N, acc_avg * 100, acc_std * 100)
    print(info)

    # 合并所有种子的loss数组，用于绘图

    all_losses = np.concatenate([metrics[seed]['glob_loss'] for seed in range(n_seeds)])
    num_of_rounds = len(all_losses) // n_seeds  # 确定轮数，用于绘图中的x轴。
    x_axis_data = np.array(
        list(range(num_of_rounds)) * n_seeds) + 1  # 储存x轴数据（communication round）的数组。eg.[1,2,3,1,2,3,...]
    algo_label = f"{algo_name}: Final Acc Average= {final_acc_avg:.2%}"

    # Plot loss curve
    sns.lineplot(
        x=x_axis_data,  # 由range(length)生成的轮次（Epoch）编号，每个编号重复n_seeds次以匹配所有种子的数据点
        y=all_losses,  # 所有种子的精度值
        legend='brief',
        color=COLORS[ALGO_INDEX[algo_name]],
        label=algo_label,
    )
    # plt.plot(x_axis_data, all_losses, label=algo_label)


def plot_communication_overhead(metrics, algo_name):
    overhead_upload = np.concatenate(
        [metrics[seed]['communication_overhead_upload'] for seed in
         range(n_seeds)])  # 选取所有种子的通讯成本，然后将这些值合并为一个数组。

    overhead_avg = np.mean(overhead_upload)
    overhead_std = np.std(overhead_upload)
    info = 'Algorithm: {:<10s}, Average overhead = {:.1f} bytes, deviation = {:.1f}'.format(algo_name, overhead_avg,
                                                                                            overhead_std)
    print(info)

    # 合并所有种子的overhead数组，用于绘图
    num_of_rounds = len(overhead_upload) // n_seeds  # 确定轮数，用于绘图中的x轴。
    x_axis_data = np.array(
        list(range(num_of_rounds)) * n_seeds) + 1  # 储存x轴数据（communication round）的数组。eg.[1,2,3,1,2,3,...]

    # Plot overhead curve
    algo_label = f"{algo_name}: Average overhead = {overhead_avg} bytes"
    # Plot overhead curve
    sns.lineplot(
        x=x_axis_data,  # 由range(length)生成的轮次（Epoch）编号，每个编号重复n_seeds次以匹配所有种子的数据点
        y=overhead_upload,  # 所有种子的overhead
        legend='brief',
        color=COLORS[ALGO_INDEX[algo_name]],
        label=algo_label,
    )
    # plt.plot(x_axis_data, overhead_upload, label=algo_label)


def plot_training_time(metrics, algo_name):

    # 计算所有训练时间
    all_training_time = np.concatenate(
        [metrics[seed]['user_train_time'] for seed in
         range(n_seeds)])  # 选取所有种子的通讯成本，然后将这些值合并为一个数组。

    training_time_avg = np.mean(all_training_time)
    training_time_std = np.std(all_training_time)
    info = 'Algorithm: {:<10s}, Average training time = {:.2f}s, deviation = {:.1f}'.format(algo_name, training_time_avg,
                                                                                           training_time_std)
    print(info)

    # 合并所有种子的training time数组，用于绘图
    num_of_rounds = len(all_training_time) // n_seeds  # 确定轮数，用于绘图中的x轴。
    x_axis_data = np.array(
        list(range(num_of_rounds)) * n_seeds) + 1  # 储存x轴数据（communication round）的数组。eg.[1,2,3,1,2,3,...]

    algo_label = f"{algo_name}: Average training time = {training_time_avg:.2f}s"

    # Plot training time curve
    sns.lineplot(
        x=x_axis_data,  # 由range(length)生成的轮次（Epoch）编号，每个编号重复n_seeds次以匹配所有种子的数据点
        y=all_training_time,  # 所有种子的训练时间
        legend='brief',
        color=COLORS[ALGO_INDEX[algo_name]],
        label=algo_label,
    )
    # plt.plot(x_axis_data, all_training_time, label=algo_label)


# def plot_results(args, algorithms):
#     n_seeds = args.times  # 获取重复实验的次数
#     dataset_ = args.dataset.split('-')
#     sub_dir = dataset_[0] + "/" + dataset_[2] # e.g. Mnist/ratio0.5
#     os.makedirs("figs/{}".format(sub_dir), exist_ok=True)  # e.g. figs/Mnist/ratio0.5 使用系统命令创建存放图像的目录，mkdir -p确保如果目录不存在则创建它，且不会因为目录已存在而报错。
#     plt.figure(1, figsize=(5, 5))  # 创建一个绘图窗口，指定窗口编号为1，大小为5x5英寸。
#     TOP_N = 5  # 指定在计算平均精度时考虑的顶部精度值数量
#     max_acc = 0  # 初始化记录最高精度的变量
#     for i, algorithm in enumerate(algorithms):
#         algo_name = get_label_name(algorithm)
#
#         ######### plot test accuracy ############
#         metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]  # 为每个随机种子加载算法的测试结果
#         all_curves = np.concatenate([metrics[seed]['glob_acc'] for seed in range(n_seeds)])  # 将所有种子的全局精度（glob_acc）合并成一个数组，用于绘图。
#         top_accs = np.concatenate([np.sort(metrics[seed]['glob_acc'])[-TOP_N:] for seed in range(n_seeds)] )  # 从每个种子的精度结果中选取最高的TOP_N个精度值，然后将这些值合并为一个数组。
#         acc_avg = np.mean(top_accs)  # 计算这些顶部精度值的平均值
#         acc_std = np.std(top_accs)  # 计算这些顶部精度值的标准偏差
#         info = 'Algorithm: {:<10s}, Accuracy = {:.2f} %, deviation = {:.2f}'.format(algo_name, acc_avg * 100, acc_std * 100)
#         print(info)
#         length = len(all_curves) // n_seeds  # 确定每个种子的精度曲线长度，用于绘图中的x轴。
#
#         # 使用seaborn的lineplot函数绘制精度曲线
#         sns.lineplot(
#             x=np.array(list(range(length)) * n_seeds) + 1,  # 由range(length)生成的轮次（Epoch）编号，每个编号重复n_seeds次以匹配所有种子的数据点
#             y=all_curves.astype(float),  # 所有种子的精度值
#             legend='brief',
#             color=COLORS[i],
#             label=algo_name,
#             ci="sd",
#         )
#
#     plt.gcf()  # 用于获取当前的图形
#     plt.grid()  # 使图表中添加网格线
#     plt.title(dataset_[0] + ' Test Accuracy')
#     plt.xlabel('Epoch')
#     max_acc = np.max([max_acc, np.max(all_curves) ]) + 4e-2  # 更新max_acc变量为当前算法的所有曲线中的最大精度值，并稍微增加（+ 4e-2），以确保图表的上边界留有一定的空间。
#
#     # 根据args.min_acc的值来决定Y轴的最小显示范围
#     if args.min_acc < 0:  # 表示未设置具体的最小精度阈值
#         alpha = 0.7
#         min_acc = np.max(all_curves) * alpha + np.min(all_curves) * (1-alpha)  # 计算所有曲线的最大和最小精度值，并按照比例（alpha）混合它们来作为Y轴的最小值。
#     else:
#         min_acc = args.min_acc
#     plt.ylim(min_acc, max_acc)  # 置Y轴的显示范围为计算或指定的最小值和调整后的最大值。
#     fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[2] + '.png')
#     plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0, format='png', dpi=400)
#     print('file saved to {}'.format(fig_save_path))