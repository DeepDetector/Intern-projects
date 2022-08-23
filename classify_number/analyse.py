import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from dataset.dataset import TestDataset
from model.resnet import resnet50
from model.efficienv2 import efficientnetv2_s
from sklearn.metrics import confusion_matrix
import pandas as pd
import shutil
###############################################################################
#   config setting                                                            #
###############################################################################
np.random.seed(3407)
random.seed(0)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

root_dir = '/raid/AI_lai/share/wb/dataset/generate_data_iou0.1_0.7/'
train_label_path = '/raid/AI_lai/share/wb/dataset/generate_data_iou0.1_0.7/train.csv'
val_label_path = '/raid/AI_lai/share/wb/dataset/generate_data_iou0.1_0.7/validate.csv'
# ckpt = './pkl/bestauc.pkl'
ckpt = './pkl/resnet_class4_sz224_iou0.1_0.7.pkl'
batch_size = 256
epochs = 200
lr = 0.0001
weight_decay = 0.0001
num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def result_img(predict_list, real_list, path_list):
    if len(predict_list) != len(real_list) or len(predict_list) != len(path_list):
        print('size error')
        return
    for i in tqdm(range(len(path_list))):
        if real_list[i]==0:
            if predict_list[i]==1:
                shutil.copy(path_list[i],'./rs_img/0to1/'+path_list[i].split('/')[-1])
        elif real_list[i]==1:
            if predict_list[i]==0:
                shutil.copy(path_list[i],'./rs_img/1to0/'+path_list[i].split('/')[-1])


def main():
    print("===============Config Information===============")
    print("Using {} device.".format(device))
    assert os.path.exists(root_dir), "{} path does not exist.".format(root_dir)
    # Linux系统下可用
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    #  训练集
    train_dataset = TestDataset(train_label_path, root_dir, 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    train_num = len(train_dataset)
    #  验证集
    val_dataset = TestDataset(val_label_path, root_dir, 'val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)
    val_num = len(val_dataset)
    print(train_label_path)
    print("Using {} images for training, {} images for validation.".format(train_num, val_num))

    # 加载模型
    net = resnet50(num_classes=num_classes)
    if ckpt:
        net.load_state_dict(torch.load(ckpt))
        print('Load Net state_dict from pretrain')
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net = net.cuda(device=0)

    # construct an optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)

    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.1) + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    print("========================================================")
    print("======================Star train========================")
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(1):
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        predict_list = []
        real_list = []
        path_list = []
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            all_y = torch.tensor([]).to(device)
            all_label = torch.tensor([]).to(device)
            # for step, (Xv, yv,img) in enumerate(val_bar):
            for step, (Xv, yv, path) in enumerate(val_bar):
                Xv, yv = Xv.cuda(device=0), yv.cuda(device=0)
                outputs = net(Xv)
                label = torch.softmax(outputs, dim=1)
                predict_y = torch.max(label, dim=1)[1]
                predict_list = predict_list + predict_y.cpu().numpy().tolist()
                real_list = real_list + yv.cpu().numpy().tolist()
                path_list = path_list + path
                acc += torch.eq(predict_y, yv.to(device)).sum().item()
                val_bar.desc = "Valid [{}/{}]".format(epoch + 1, epochs)
                # print(predict_y)
        
        c_matrix = confusion_matrix(real_list,predict_list)
        val_accurate = acc / val_num
        print('[epoch %d] val_accuracy: %.4f' %(epoch + 1, val_accurate))
        print(c_matrix)
        # result_img(predict_list, real_list, path_list)
        
    print('Finished Training')


if __name__ == '__main__':
    main()
