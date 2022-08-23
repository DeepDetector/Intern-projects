# *coding:utf-8 *
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
 
sets = ['train', 'test']
classes = ["something" ]
 
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
 
 
def convert_annotation(image_id):
 
    in_file = open('./dataset1/Annotations/%s.xml' % (image_id))  # 修改路径（最好使用绝对路径）
    img_file = cv2.imread('./dataset1/images/%s.jpg' % (image_id))
    out_file = open('./dataset1/labels/%s.txt' % (image_id), 'w+')  # 修改路径（最好使用绝对路径）
    tree = ET.parse(in_file)
    root = tree.getroot()
    # size = root.find('size')
    assert img_file is not None
    size = img_file.shape[0:-1]
    h = int(size[0])
    w = int(size[1])
 
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes : # or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text))
        
        ZIP_ONE = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in ZIP_ONE]) + '\n')
 
 
wd = getcwd()
for image_set in sets:
    if not os.path.exists('./20211212_train_ln/labels'):
        os.makedirs('./20211212_train_ln/labels/')
    image_ids = open('/raid/huayan_nfs/share/wb/code/yolov7-main/20211212_train_ln/ImageSets/Main/%s.txt' % (image_set)).read().split()  # 修改路径（最好使用绝对路径）
    list_file = open('/raid/huayan_nfs/share/wb/code/yolov7-main/20211212_train_ln/%s.txt' % (image_set), 'w+')  # 修改路径（最好使用绝对路径）
    # print(image_ids)
    for image_id in image_ids:
        print('==')
        try:
            print(image_id)
            list_file.write('20211212_train_ln/images/%s.jpg\n' % (image_id))  # 修改路径（最好使用绝对路径）
            #convert_annotation(image_id)
        except:
            print('error img:', image_id)
    list_file.close()