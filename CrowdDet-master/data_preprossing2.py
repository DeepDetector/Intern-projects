import numpy as np
import os
import glob
from tqdm import tqdm
import xml.etree.cElementTree as ET
from itertools import combinations
import cv2 as cv

savaname = 'annotations_tarin.odgt'
rootdir = '/raid/AI_lai/share/wb/code/CrowdDet-master'
def generatedata():
    targetdirs = [
    '/raid/AI_lai/CYJ/data/sku_detection/20211206_YB_background_train/train/',
    '/raid/AI_lai/CYJ/data/sku_detection/20211207_online_train/train/',
    '/raid/AI_lai/CYJ/data/sku_detection/20211209_base_background_train/train/',
    '/raid/AI_lai/CYJ/data/sku_detection/20211209_base_train/train',
    # '/raid/AI_lai/CYJ/data/sku_detection/20211209_single_origin_train/train',
    '/raid/AI_lai/CYJ/data/sku_detection/20211210_outsourcing_labeled_train/train'
    ]
    file_name = open(savaname, "w") # 保存为odgt
    for dir in targetdirs:
        # dir = dir + 'train'
        xmlfiles = os.listdir(dir + '/Annotations')
        
        for xmlfile in tqdm(xmlfiles):
            xmlpath = os.path.join(dir, 'Annotations', xmlfile)
            imgpath = os.path.join(dir,'JPEGImages',xmlfile[:-4])
            # print(img.shape)
            tree = ET.parse(xmlpath)
            root = tree.getroot()

            # 读取图片大小
            # h = root.find('size').find('height').text
            # w = root.find('size').find('width').text
            # 读取所有anchor
            objects = root.findall('object')
            num = len(objects)
            if num==0:
                continue
            index = 0
            file_name.write('{{"ID": "{0}", "gtboxes": ['.format(imgpath))
            flag = ''
            for box in objects:
                xmin = int(box.find('bndbox').find('xmin').text)
                ymin = int(box.find('bndbox').find('ymin').text)
                xmax = int(box.find('bndbox').find('xmax').text)
                ymax = int(box.find('bndbox').find('ymax').text)
                w = xmax-xmin
                h = ymax-ymin
                classname = box.find('name').text
                flag = flag + '{"tag": "something", "fbox": '+'[{0}, {1}, {2}, {3}]'.format(xmin,ymin,w,h)+'},'
            flag = flag[:-1]
            file_name.write(flag)
            file_name.write(']}' + '\n') # 每个图片另起一行
    file_name.close()        
                    

# 将xml格式转为odgt格式
if __name__ == '__main__':
    generatedata()
