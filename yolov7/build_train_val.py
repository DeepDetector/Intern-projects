# *coding:utf-8 *
import os
import random
from tqdm import tqdm
trainval_percent = 0.1  # 可自行进行调节(设置训练和测试的比例是8：2)
train_percent = 1
ftest = open('./20211212_train_ln/ImageSets/Main/test2.txt', 'w')
ftrain = open('./20211212_train_ln/ImageSets/Main/train2.txt', 'w')
#txtsavepath = './dataset1/ImageSets/Main'
xmlfilepath = './20211212_train_ln/labels'

total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
 
# ftrainval = open('ImageSets/Main/trainval.txt', 'w')

# fval = open('ImageSets/Main/val.txt', 'w')
 
for i in tqdm(list):
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        # ftrainval.write(name)
        if i in train:
            ftest.write(name)
        # else:
        # fval.write(name)
    else:
        ftrain.write(name)

# ftrainval.close()
ftrain.close()
# fval.close()
ftest.close()