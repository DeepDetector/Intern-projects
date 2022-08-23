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
from dataset.dataset import MyDataset
from model.resnet import resnet50,resnet101
from model.efficienv2 import efficientnetv2_s
from sklearn.metrics import confusion_matrix
###############################################################################
#   config setting                                                            #
###############################################################################
np.random.seed(3407)
random.seed(0)
torch.manual_seed(3407)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

root_dir = '/raid/AI_lai/share/wb/dataset/clean_data_iou0.1_0.3/'
train_label_path = '/raid/AI_lai/share/wb/dataset/clean_data_iou0.1_0.3/train.csv'
val_label_path = '/raid/AI_lai/share/wb/dataset/clean_data_iou0.1_0.3/validate.csv'
# ckpt = './pkl/bestauc.pkl'
ckpt = ''
batch_size = 128
epochs = 500
lr = 0.0001
weight_decay = 0.0001
num_classes = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("===============Config Information===============")
    print("Using {} device.".format(device))
    assert os.path.exists(root_dir), "{} path does not exist.".format(root_dir)
    # Linux系统下可用
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    #  训练集
    train_dataset = MyDataset(train_label_path, root_dir, 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    train_num = len(train_dataset)
    #  验证集
    val_dataset = MyDataset(val_label_path, root_dir, 'val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_num = len(val_dataset)
    print(train_label_path)
    print("Using {} images for training, {} images for validation.".format(train_num, val_num))

    # 加载模型
    net = resnet101(num_classes=num_classes)
    if ckpt:
        net.load_state_dict(torch.load(ckpt))
        print('Load Net state_dict from pretrain')
    #net = torch.nn.DataParallel(net, device_ids=[0, 1,2,3])
    net = net.cuda(device=0)

    # construct an optimizer
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=0.0001)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.1) + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))
    print("========================================================")
    print("======================Star train========================")
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_acc = 0.0
        train_bar = tqdm(train_loader)
        for step, (X, y) in enumerate(train_bar):
            # X, y = X.to(device), y.to(device)
            X, y = X.cuda(device=0), y.cuda(device=0)
            optimizer.zero_grad()
            # 正向传播
            logits = net(X)

            # print(logits.shape)
            label = torch.softmax(logits, dim=1)
            label = torch.max(label, dim=1)[1]
            # print(label,y)
            train_acc += torch.eq(label, y).sum().item()
            regular_loss = 0
            # l2 正则化
            for n, p in net.named_modules():
                # if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                if isinstance(p, nn.Conv2d):
                    # regular_loss += torch.sum(torch.abs(p.weight))
                    regular_loss += (p.weight**2).sum()
                elif isinstance(p, nn.Linear):
                    regular_loss += (p.weight**2).sum()
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "Train [{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            all_y = torch.tensor([]).to(device)
            all_label = torch.tensor([]).to(device)
            # for step, (Xv, yv,img) in enumerate(val_bar):
            for step, (Xv, yv) in enumerate(val_bar):
                # Xv, yv = Xv.to(device), yv.to(device)
                Xv, yv = Xv.cuda(device=0), yv.cuda(device=0)
                outputs = net(Xv)
                label = torch.softmax(outputs, dim=1)
                predict_y = torch.max(label, dim=1)[1]
                # print(predict_y[0])
                # vision(heat[0], img[0])
                acc += torch.eq(predict_y, yv.to(device)).sum().item()
                val_bar.desc = "Valid [{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  train_acc: %.4f  val_accuracy: %.4f' %
              (epoch + 1, running_loss / train_steps, train_acc / train_num, val_accurate))

        # save best model
        if val_accurate > best_acc:
            best_acc = val_accurate
            print('=============Best acc============')
            torch.save(net.state_dict(), './pkl/resnet101_class4_sz224_clean_iou0.1_0.3' + '.pkl')

        scheduler.step()

    print('Finished Training')


if __name__ == '__main__':
    main()
