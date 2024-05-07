import random
import time
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
from colorama import init, Fore, Back, Style
import numpy as np
from tqdm import tqdm
# from data_unit.utils import blind_other_gpus
from pre_dataset import MyDataset
# from munkres import Munkres
from sklearn import metrics
from tensorboardX import SummaryWriter
import os
import argparse
from ruamel.yaml import YAML
from termcolor import cprint
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import logging
import csv
from sklearn.metrics import r2_score
from fastdtw import fastdtw
import torchvision.models as models


def get_args_key(args):
    return "-".join([args.model_name, args.dataset_name, args.custom_key])


def get_args(model_name, dataset_class, dataset_name, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = yaml_path or os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser(description='Parser for Supervised Graph Attention Networks')
    # Basics
    parser.add_argument("--m", default="", type=str, help="Memo")
    parser.add_argument("--num-gpus-total", default=1, type=int)
    parser.add_argument("--num-gpus-to-use", default=1, type=int)
    # parser.add_argument("--black-list", default=None, type=int, nargs="+")
    parser.add_argument("--black-list", default=None, type=int)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--task-type", default="", type=str)
    parser.add_argument("--perf-type", default="accuracy", type=str)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--verbose", default=2)
    parser.add_argument("--save-plot", default=False)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--name", default="GAT")

    # Dataset
    # parser.add_argument('--data-root', default="~/graph-data", metavar='DIR', help='path to dataset')
    parser.add_argument('--data-root', default="./cora", metavar='DIR', help='path to dataset')
    parser.add_argument("--dataset-class", default=dataset_class)
    parser.add_argument("--dataset-name", default=dataset_name)
    parser.add_argument("--data-sampling-size", default=None, type=int, nargs="+")
    parser.add_argument("--data-sampling-num-hops", default=None, type=int)
    parser.add_argument("--data-num-splits", default=1, type=int)
    parser.add_argument("--data-sampler", default=None, type=str)

    # Training
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--loss", default=None, type=str)
    parser.add_argument("--l1-lambda", default=0., type=float)
    parser.add_argument("--l2-lambda", default=0., type=float)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--use-bn", default=False, type=bool)
    parser.add_argument("--perf-task-for-val", default="Node", type=str)  # Node or Link

    # Early stop
    parser.add_argument("--use-early-stop", default=False, type=bool)
    parser.add_argument("--early-stop-patience", default=-1, type=int)
    parser.add_argument("--early-stop-queue-length", default=100, type=int)
    parser.add_argument("--early-stop-threshold-loss", default=-1.0, type=float)
    parser.add_argument("--early-stop-threshold-perf", default=-1.0, type=float)

    # Pretraining
    parser.add_argument("--usepretraining", default=False, type=bool)
    parser.add_argument("--total-pretraining-epoch", default=0, type=int)
    parser.add_argument("--pretraining-noise-ratio", default=0.0, type=float)

    # Baseline
    parser.add_argument("--is-link-gnn", default=False, type=bool)
    parser.add_argument("--link-lambda", default=0., type=float)

    # Test
    parser.add_argument("--val-interval", default=10)

    parser.add_argument('--dateset', type=str, default='Cora', help='')
    parser.add_argument('--UsingLiner', type=bool, default=True, help='')
    parser.add_argument('--useNewA', type=bool, default=True, help='')
    parser.add_argument('--NewATop', type=int, default=0, help='')
    parser.add_argument('--usingact', type=bool, default=True, help='')
    parser.add_argument('--notation', type=str, default=None, help='')

    # Experiment specific parameters loaded from .yamls
    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset_name or args.dataset_class, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


def get_important_args(_args: argparse.Namespace) -> dict:
    important_args = [
        "lr",
        "batch_size",
        "data_sampling_num_hops",
        "data_sampling_size",
        "data_sampler",
        "data_num_splits",
        "to_undirected_at_neg",
        "num_hidden_features",
        "num_layers",
        "use_bn",
        "l1_lambda",
        "l2_lambda",
        "dropout",
        "is_super_gat",
        "is_link-gnn",
        "attention_type",
        "logit_temperature",
        "use_pretraining",
        "total_pretraining_epoch",
        "pretraining_noise_ratio",
        "neg_sample_ratio",
        "edge_sampling_ratio",
        "use_early_stop",
    ]
    ret = {}
    for ia_key in important_args:
        if ia_key in _args.__dict__:
            ret[ia_key] = _args.__getattribute__(ia_key)
    return ret


def save_args(model_dir_path: str, _args: argparse.Namespace):
    if not os.path.isdir(model_dir_path):
        raise NotADirectoryError("Cannot save arguments, there's no {}".format(model_dir_path))
    with open(os.path.join(model_dir_path, "args.txt"), "w") as arg_file:
        for k, v in sorted(_args.__dict__.items()):
            arg_file.write("{}: {}\n".format(k, v))


def pprint_args(_args: argparse.Namespace):
    cprint("Args PPRINT: {}".format(get_args_key(_args)), "yellow")
    for k, v in sorted(_args.__dict__.items()):
        print("\t- {}: {}".format(k, v))


def pdebug_args(_args: argparse.Namespace, logger):
    logger.debug("Args LOGGING-PDEBUG: {}".format(get_args_key(_args)))
    for k, v in sorted(_args.__dict__.items()):
        logger.debug("\t- {}: {}".format(k, v))

init(autoreset=True)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, predicted, target):
        mse = nn.MSELoss()(predicted, target)
        rmse = torch.sqrt(mse)
        return rmse

# Define the modified ResNet50 model
class ModifiedResNet50(nn.Module):
    def __init__(self, input_channels):
        super(ModifiedResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # Modify the first convolution layer to adapt to the input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.resnet(x)

# Create an instance of the modified ResNet50 model
modified_resnet = ModifiedResNet50(input_channels=4)  # Assuming input data has 4 channels

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_size):
        super(ContrastiveModel, self).__init__()
        self.resnet = modified_resnet
        self.fc = nn.Linear(1000,11264) #输入特征是1000 输出特征是11264
        self.layer_norm_x = nn.LayerNorm(5)#具有 5 个特征的输入数据

        # self.resnet.fc = nn.Identity()  # Remove the classifier layer
        # self.fc1 = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        # print(x.shape) [1408, 5]
        x = self.layer_norm_x(x)
        features = self.resnet(x)
        # print(features.shape)  #torch.Size([32, 1000])
        embedded = self.fc(features)
        return embedded
    

class neg_ContrastiveModel(nn.Module):
    def __init__(self, embedding_size):
        super(neg_ContrastiveModel, self).__init__()
        self.resnet = modified_resnet
        self.fc = nn.Linear(1000,100)

        # self.resnet.fc = nn.Identity()  # Remove the classifier layer
        # self.fc1 = nn.Linear(num_features, embedding_size)

    def forward(self, x):
        # print(x.shape) [1408, 5]
        features = self.resnet(x)
        # print(features.shape)   #torch.Size([32, 1000])
        #全连接层（fully connected layer）假设 features 的形状是 (batch_size, input_size)，其中 input_size 是输入特征的数量。self.fc 的作用是将输入的特征进行线性变换，使得输出的形状为 (batch_size, output_size)，其中 output_size 是全连接层的输出特征数量。
        embedded = self.fc(features)
        return embedded

# Define Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pos_pairs, neg_pairs):
        loss_pos = torch.mean(torch.square(pos_pairs))  # Loss for positive pairs
        loss_neg = torch.mean(torch.clamp(self.margin - neg_pairs, min=0))  # Loss for negative pairs
        loss = loss_pos + loss_neg
        return loss
    
# Compute similarity matrix
def compute_similarity_matrix(samples):
    normalized_samples = torch.nn.functional.normalize(samples, p=2, dim=2)  # Normalize samples
    similarity_matrix = torch.matmul(normalized_samples, normalized_samples.transpose(1, 2))  # Compute similarity matrix
    return similarity_matrix

# 这个是下游任务
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        # print(ft_in)  #256
        # print(nb_classes)  #2
        # GAT的
        self.layer_norm_x = nn.LayerNorm(256)
        self.fc = nn.Linear(ft_in, 128)
        self.act = nn.Sigmoid()
        self.mlp = nn.Linear(128, nb_classes)
        # GCN的
        # self.fc = nn.Linear(256, 2)

    def forward(self, seq):
        # print(seq.shape)  #torch.Size([32, 4, 11, 256])
        # ret = self.mlp(seq)
        seq = seq.reshape(32, 4, 11, 256)
        ret = self.layer_norm_x(seq)  #([32, 4, 11, 256])
        ret = self.fc(ret)
        ret = self.act(ret)
        ret = self.mlp(ret)
        # print(ret.shape)  #([32, 4, 11, 2])
        return ret  
    
class SimCLR_test(nn.Module):
    def __init__(self, embedding_size):
        super(SimCLR_test, self).__init__()
        self.pos_model = ContrastiveModel(embedding_size)
        self.neg_model = neg_ContrastiveModel(embedding_size)
        self.layer_norm_x = nn.LayerNorm(5)#具有 5 个特征的输入数据
        self.fc = nn.Linear(1000,11264)
        
    def forward(self, seq_a):
        #获取正负样本
        h_p_0 = self.pos_model(seq_a)
        h_n_0 = self.neg_model(seq_a)
        # print("h_a",h_a.shape)  #([32, 4, 11, 256])
        return h_p_0,h_n_0
    
    def embed(self, seq_a):
        x = self.layer_norm_x(seq_a)
        features = self.pos_model(x)
        # print("llllllllllllllllllll",features.shape)  #torch.Size([32, 1000])
        # embedded = self.fc(features)
        return features.detach()

def run_GCN(args, gpu_id=None, exp_name=None, number=0, return_model=False, return_time_series=False):
    embedding_size = 128
    final_loss_tp = []
    final_mse = []
    writename = "-" + exp_name[4:] + "_" + str(number)
    logname = os.path.join(exp_name, str(number) + "_" + str(args.seed) + ".txt")
    logfile = open(logname, 'a')
    writer = SummaryWriter(comment=writename)
    final_acc = 0
    best_acc = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    dataset_kwargs = {}

    # labels = np.load("./features/feature_L_sc.npy",allow_pickle=True)
    labels = np.load("./LL_sc_200.npy",allow_pickle=True)
    # labels = torch.tensor(labels).to(running_device)
    labels = [arr for arr in labels]
    # 使用 torch.stack 进行堆叠，保持原来的四维形状
    labels = torch.stack(labels, dim=0).to(running_device)
    # print("lable",labels.shape) #([27964, 8, 11, 2])  #lable torch.Size([5422, 8, 11, 2])
    # print(labels.dtype)  #torch.float32

    # labels_rc = np.load("./features/feature_L_rc.npy",allow_pickle=True)
    labels_rc = np.load("./LL_rc_200.npy",allow_pickle=True)
    # labels = torch.tensor(labels).to(running_device)
    labels_rc = [arr[:,:8,:] for arr in labels_rc]
    # 使用 torch.stack 进行堆叠，保持原来的四维形状
    labels_rc = torch.stack(labels_rc, dim=0).to(running_device)
    # print("labels_rc",labels_rc.shape) #(27964, 8, 5, 2)  #labels_rc torch.Size([5422, 8, 10, 2])
    # print(labels_rc.dtype)  #torch.float32

    # a_m = np.load("./features/feature_A_sc.npy",allow_pickle=True)
    a_m = np.load("./AA_sc_200.npy",allow_pickle=True)
    # a_m = torch.tensor(a_m).to(running_device)
    a_m = [arr for arr in a_m]
    # 使用 torch.stack 进行堆叠，保持原来的四维形状
    a_m = torch.stack(a_m, dim=0).to(running_device)
    # print("a_m",a_m.shape) #([27964, 4, 11, 11])  #a_m torch.Size([5422, 4, 11, 11])
    # print(a_m.dtype)  #torch.float32

    a_m_rc = np.load("./AA_rc_200.npy",allow_pickle=True)
    # a_m_rc = torch.tensor(a_m_rc).to(running_device)
    for i in range(len(a_m_rc)):
        # print(a_m_rc[i].shape[1])
        if a_m_rc[i].shape[1] != 10:
            padnum = 10 - a_m_rc[i].shape[1]
            # 创建一个[4, 10, 10]大小的张量，用于填充
            padded_tensor = F.pad(a_m_rc[i], (0, padnum, 0, padnum))  # 在第二个和第三个维度上各填充5个零
            # print(padded_tensor.shape)
            # 将填充后的张量替换原来的数据中的下标为2的张量
            a_m_rc[i] = padded_tensor

    a_m_rc = [arr[:,:8,:8] for arr in a_m_rc]
    # 使用 torch.stack 进行堆叠，保持原来的四维形状
    a_m_rc = torch.stack(a_m_rc, dim=0).to(running_device)
    # print("a_m_rc",a_m_rc.shape) #([27964, 4, 5, 5])  #a_m_rc torch.Size([5422, 4, 10, 10])
    # print(a_m_rc.dtype)  #torch.float32

    data = np.load("./XX_sc_200.npy",allow_pickle=True)
    # data = torch.tensor(data).to(running_device)
    data = [arr for arr in data]
    # 使用 torch.stack 进行堆叠，保持原来的四维形状
    data = torch.stack(data, dim=0).to(running_device)
    # print("data",data.shape) #([27964, 4, 11, 5])  #data torch.Size([5422, 4, 11, 5])
    # print(data.dtype)  #torch.float32

    data_rc = np.load("./XX_rc_200.npy",allow_pickle=True)
    # data_rc = torch.tensor(data_rc).to(running_device)
    data_rc = [arr[:,:8,:] for arr in data_rc]
    # 使用 torch.stack 进行堆叠，保持原来的四维形状
    data_rc = torch.stack(data_rc, dim=0).to(running_device)
    # print("data_rc",data_rc.shape) #([27964, 4, 5, 5])  #data_rc torch.Size([5422, 4, 10, 5])
    # print(data_rc.dtype)  #torch.float32
    
    # 创建单位矩阵
    I_sc = torch.eye(11).to(running_device)  # 生成一个 11x11 的单位矩阵
    I_rc = torch.eye(8).to(running_device)

    my_dataset = MyDataset(data, labels, a_m,data_rc,labels_rc,a_m_rc)
    train_data, val_data, test_data = my_dataset.split(seed=42)
    batch_size = 32
    shuffle = True
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle,drop_last=True)
    
    nb_nodes = 11
    nb_feature = 5
    nb_classes = 2

    model = SimCLR_test(embedding_size)
    my_lr = 0.005
    optimiser = torch.optim.Adam(model.parameters(), lr=my_lr, weight_decay=0.01)
    # contrastive_criterion = nn.TripletMarginLoss()
    contrastive_criterion = ContrastiveLoss(margin=1.0)
    model.to(running_device)
    

    # 自己设定的my_margin，相当于论文中的alpha


    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):

        # 模型开始训练###############################################################################
        for x_sc, lable, A_I_nomal,x_rc, lable_rc, A_I_nomal_rc in train_dataloader:
            model.train()
            optimiser.zero_grad()
            lbl_z = torch.tensor([0.]).to(running_device)
            # print(lable.shape)  #([32, 8, 11, 2])
            # print("Batch Data Shape:", x_sc.shape)  #[32, 4, 11, 5])
            # print("Batch Labels Shape:", lable_rc.shape)  #([32, 8, 5, 2])
            # print("Batch Adjacency Matrix Shape:", A_I_nomal.shape)  #([32, 4, 11, 11])
            # print("Batch Adjacency Matrix Shape:", A_I_nomal_rc.shape)  #([32, 4, 5, 5])
            # break
            # 归一化标签
            train_lbls = lable[:,:4,:,:].cpu()
            train_lbls_reshaped = train_lbls.reshape(-1, 2)
            train_lbls_scaler = MinMaxScaler()
            nor_train_lbls_reshaped = train_lbls_scaler.fit_transform(train_lbls_reshaped)
            # 将归一化后的数据重新调整为原始形状
            normalized_train_lbls = nor_train_lbls_reshaped.reshape(batch_size,4,11,2)
            normalized_train_lbls = torch.tensor(normalized_train_lbls).to(running_device)
            
            positive_emb,negative_emb = model(x_sc)

            positive_embeddings = positive_emb.view(batch_size, 11,-1)
            # print(positive_embeddings.shape)# torch.Size([32, 11, 1024])
            negative_embeddings = negative_emb.view(batch_size, 5,-1)  # shape: [batch_size, num_negative_samples, embedding_size]

            # Compute positive_pairs and negative_pairs
            positive_pairs = compute_similarity_matrix(positive_embeddings)
            # print("??????????",positive_pairs.shape)  #torch.Size([32, 11, 11])
            negative_pairs = compute_similarity_matrix(negative_embeddings)
            loss = contrastive_criterion(positive_pairs, negative_pairs)
            loss.backward()
            optimiser.step()
            # print(f"Contrastive Learning Epoch {epoch + 1}/{500}, Loss: {loss }")
         
            if epoch % 50 == 0:
                model.eval()
                # 这里是测试吧？？？    
                # x_sc, x_sc, x_rc, A_I_nomal,A_I_nomal_rc, I_sc,I_rc
                h_a = model.embed(x_sc)
                train_embs = h_a
                # print("nnnnnnnnnnnnnnnnnnnnnnnnnnnn",train_embs.shape)  #torch.Size([32, 11264])
                test_embs = h_a
                pre_mse = []
                pre_r2 = []
                pre_distance = []
                pre_adeu = []
                pre_ADE = []
                pre_FDE = []
                # accs_small = []
                # xent = nn.CrossEntropyLoss()
                # xent = nn.MSELoss()
                xent = RMSELoss()
                for _ in range(2):
                    log = LogReg(256, nb_classes)
                    # opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=args.wd)
                    opt = torch.optim.Adam(log.parameters(), lr=1e-3, weight_decay=0.01)
                    log.to(running_device)
                    # print(args.num1)  #100
                    for _ in range(100):
                        log.train()
                        opt.zero_grad()
                        # 这里是下游任务,下游任务是轨迹预测
                        logits = log(train_embs.reshape(32,4,11,256)) #32,4,11,2                  
                        loss = xent(logits, normalized_train_lbls.float())
                        final_loss_tp.append(loss.item())
                        loss.backward()
                        opt.step()
                    preds = log(test_embs)
                    # preds = torch.argmax(logits, dim=1)
                    # print(logits)
                    # print(test_lbls)
                    # 均方误差mse
                    mse = torch.mean((normalized_train_lbls.float() - preds)**2)
                    # print("mse",mse)  #0.011897600084845425
                    # R平方
                    # 如果张量在 GPU 上，使用 .cpu() 复制到主机内存
                    y_true_tensor = normalized_train_lbls.cpu().detach().numpy()
                    y_pred_tensor = preds.cpu().detach().numpy()
                    # print(y_true_tensor.shape)  #(32, 4, 11, 2)
                    # print(y_pred_tensor.shape)  #(32, 4, 11, 2)
                    # 计算 R 平方
                    r2 = torch.tensor(r2_score(y_true_tensor.reshape(-1,2), y_pred_tensor.reshape(-1,2)))
                    # print("r2",r2)  # 0.10584236491212823
                    # # 使用DTW计算轨迹匹配
                    distance, path = fastdtw(y_true_tensor.reshape(-1,2), y_pred_tensor.reshape(-1,2))
                    # print("distance",distance)  # 66.90789626542885
                    dtw = torch.tensor(distance)
                    # 标准差
                    std_dev = torch.std(preds.reshape(-1,2), axis=1).cpu().detach().numpy()
                    std_dev = std_dev[:, np.newaxis]
                    # print(std_dev.shape)
                    # 计算每个时间点上的 ADEu
                    adeu_per_point = np.abs(y_pred_tensor.reshape(-1,2) - y_true_tensor.reshape(-1,2)) / std_dev
                    # 计算平均 ADEu
                    average_adeu = torch.tensor(np.mean(adeu_per_point))
                    # print("average_adeu",average_adeu)  #30.51697457102681

                    # 平均位移误差minADE
                    euclidean_distance = torch.norm(normalized_train_lbls  - preds, dim=-1)
                    # print("preds[:1]",preds[:, -1:, :, :].shape)  #preds[:1] torch.Size([1, 4, 11, 2])
                    # print("normalized_train_lbls",normalized_train_lbls.shape)

                    # 最终位移误差minFDE
                    final_euclidean_distance = torch.norm(normalized_train_lbls[:, -1:, :, :] - preds[:, -1:, :, :], dim=-1)
                    average_distance = torch.mean(euclidean_distance)
                    final_euclidean_distance = torch.mean(final_euclidean_distance)

                    pre_mse.append(mse)
                    pre_r2.append(r2)
                    pre_distance.append(dtw)
                    pre_adeu.append(average_adeu)
                    pre_ADE.append(average_distance)
                    pre_FDE.append(final_euclidean_distance)


                final_pre_mse = torch.stack(pre_mse)
                final_pre_r2 = torch.stack(pre_r2)
                final_pre_distance = torch.stack(pre_distance)
                final_pre_adeu = torch.stack(pre_adeu)
                final_pre_ADE = torch.stack(pre_ADE)
                final_pre_FDE = torch.stack(pre_FDE)

                string_2 = Fore.GREEN + " epoch: {},minADE: {:.4f},minFDE: {:.4f} ".format(epoch, average_distance.item(),                                                                    final_euclidean_distance.item())
                tqdm.write(string_2)

                final_mse = torch.mean(final_pre_mse).item()
                final_r2 = torch.mean(final_pre_r2).item()
                final_distance = torch.mean(final_pre_distance).item()
                final_adeu = torch.mean(final_pre_adeu).item()
                final_FDE = torch.mean(final_pre_FDE).item()
                final_acc = torch.mean(final_pre_ADE).item()

                best_acc = min(best_acc, final_acc)


        log_path = f"./training_log/{args.name}/"
        os.makedirs(log_path, exist_ok=True)
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        torch.cuda.empty_cache()
        # save to training log
        log = {'epoch': epoch+1,'batch_size': batch_size, 'lr': my_lr,
                'predictorADE': final_acc, 'predictorFDE': final_FDE, 'predictormse':final_mse,
                'predictorR2':final_r2, 'predictorDistance':final_distance, 'predictorAdeu':final_adeu}

        if epoch == 0:
            with open(f'./training_log/train_log.csv', 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/train_log.csv', 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # save model at the end of epoch
        torch.save(model.state_dict(), f'training_log/model_{epoch+1}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")
    return final_acc, best_acc,final_loss_tp 


def run_with_many_seeds(args, num_seeds, gpu_id=None, name=None, **kwargs):
    results_acc = []
    results_best = []
    results_epoch = []
    loss_GAT_list = []
    loss_tp_list = []
    # print(num_seeds) #10
    for i in range(1):
        cprint("## TRIAL {} ##".format(i), "yellow")
        _args = deepcopy(args)
        _args.seed = _args.seed + random.randint(0, 100)
        acc, best,loss_tp = run_GCN(_args, gpu_id=gpu_id, number=i, exp_name=name, **kwargs)
        # print("loss_tp",loss_tp)
        loss_tp_list.append(loss_tp)
        results_acc.append(torch.as_tensor(acc, dtype=torch.float32))
        results_best.append(torch.as_tensor(best, dtype=torch.float32))
    results_acc = torch.stack(results_acc)
    results_best = torch.stack(results_best)
    # np.savetxt("final_loss_GAT.txt",loss_GAT_list)
    np.savetxt("final_loss_tp.txt",loss_tp_list)
    np.savetxt("acc.txt", results_acc)
    np.savetxt("best_acc.txt", results_best)
    return results_acc.mean().item(),results_best.mean().item()

if __name__ == '__main__':

    num_total_runs = 10
    main_args = get_args(
        model_name="GRLC",
        dataset_class="Planetoid",
        dataset_name="Cora",
        custom_key="classification",
    )
    pprint_args(main_args)
    filePath = "./log"
    exp_ID = 0
    for filename in os.listdir(filePath):
        file_info = filename.split("_")
        file_dataname = file_info[0]
        if file_dataname == main_args.dataset_name:
            exp_ID = max(int(file_info[1]), exp_ID)
    exp_name = main_args.dataset_name + "_" + str(exp_ID + 1)
    exp_name = os.path.join(filePath, exp_name)
    os.makedirs(exp_name)
    if len(main_args.black_list) == main_args.num_gpus_total:
        alloc_gpu = [None]
        cprint("Use CPU", "yellow")
    else:
    #     alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
    #                                  num_gpus_to_use=main_args.num_gpus_to_use,
    #                                  black_list=main_args.black_list)
    #     if not alloc_gpu:
        alloc_gpu = [int(np.random.choice([g for g in range(main_args.num_gpus_total)
                                           if g not in main_args.black_list], 1))]
    cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

    # noinspection PyTypeChecker

    # main_args.ir = 0.01
    # many_seeds_result = run_with_many_seeds(main_args, num_total_runs, gpu_id=alloc_gpu[0], name=exp_name)

    many_seeds_result = run_with_many_seeds(main_args, num_total_runs, gpu_id=0, name=exp_name)
    # print("msr",many_seeds_result)
    # newname = exp_name + "_" + '%.2f' % many_seeds_result
    newname = exp_name + "_%.2f_%.2f" % many_seeds_result

    if main_args.notation:
        newname = newname + "_" + main_args.notation0
    os.rename(exp_name, newname)

    
