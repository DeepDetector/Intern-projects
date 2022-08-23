import numpy as np
import os
import glob
from tqdm import tqdm
import xml.etree.cElementTree as ET
from itertools import combinations
import cv2 as cv

savedir = '/raid/huayan_nfs/share/data/detection/generate_data_iou0.1/'
thr = 0.1 # 设定iou分割阈值

def iou(box1, box2):
    left = max(box1[0], box2[0])
    bottom = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    top = min(box1[3], box2[3])
    cross_box = [0, 0, 0, 0]
    iouarea = 0
    iouscore = 0
    #print('box1:',box1,'box1:',box2)
    #print(right ,left , top , bottom)
    if (right > left) and (top > bottom):
        iouarea = (right - left) * (top - bottom)
        cross_box = [left, top, right, bottom]
        s1 = int((box1[2]-box1[0])*(box1[3]-box1[1]))
        s2 = int((box2[2]-box2[0])*(box2[3]-box2[1]))
        iouscore = iouarea/(s1+s2-iouarea)
        #print('s1:',s1,'s2:',s2,'iouarea:',iouarea,'iou:',iou)
    #else:
        #s1 = int((box1[2]-box1[0])*(box1[3]-box1[1]))
        #s2 = int((box2[2]-box2[0])*(box2[3]-box2[1]))
        #iou = iouarea/(s1+s2-iouarea)
    return iouscore, cross_box


def issaveImg(c_minx, c_miny, c_maxx, c_maxy, img):
    save = True
    number = len(c_minx)
    # print(c_minx, c_miny, c_maxx, c_maxy)
    if number == 1:
        save = True
    elif number == 2:
        iouarea, cross_box = iou([c_minx[0], c_miny[0], c_maxx[0], c_maxy[0]],
                                 [c_minx[1], c_miny[1], c_maxx[1], c_maxy[1]])
        if iouarea <= 0:
            save = False
    elif number > 2:
        indexs = [i for i in range(number)]
        for c in combinations(indexs, 2):
            # print(c)
            iouarea, cross_box = iou([c_minx[c[0]], c_miny[c[0]], c_maxx[c[0]], c_maxy[c[0]]],
                                     [c_minx[c[1]], c_miny[c[1]], c_maxx[c[1]], c_maxy[c[1]]])
            if iouarea <= 0:
                save = False
                break
    # print(iouarea)
    # print('---------')
    # showimg(c_minx, c_miny, c_maxx, c_maxy, [0, 0, 0, 0], img)
    return save

def is2anchorcross(box1,box2):
    minx1 = int(box1.find('bndbox').find('xmin').text)
    miny1 = int(box1.find('bndbox').find('ymin').text)
    maxx1 = int(box1.find('bndbox').find('xmax').text)
    maxy1 = int(box1.find('bndbox').find('ymax').text)
    
    minx2 = int(box2.find('bndbox').find('xmin').text)
    miny2 = int(box2.find('bndbox').find('ymin').text)
    maxx2 = int(box2.find('bndbox').find('xmax').text)
    maxy2 = int(box2.find('bndbox').find('ymax').text)
    iouscore, cross_box = iou([minx1, miny1, maxx1, maxy1],[minx2,miny2, maxx2, maxy2])
    #return iouscore
    if iouscore>thr:
        #print('cross:',iouscore)
        return True
    else:
        #print('not cross:',iouscore)
        return False
        
def save_subimg(img,path,box):
    new_minx = int(box.find('bndbox').find('xmin').text)
    new_miny = int(box.find('bndbox').find('ymin').text)
    new_maxx = int(box.find('bndbox').find('xmax').text)
    new_maxy = int(box.find('bndbox').find('ymax').text)
    new_img = img[new_miny:new_maxy, new_minx:new_maxx]
    cv.imwrite(path, new_img)

def showimg(c_minx, c_miny, c_maxx, c_maxy, new_box, img):
    n = len(c_minx)
    for i in range(n):
        cv.rectangle(img, (c_minx[i], c_miny[i]), (c_maxx[i], c_maxy[i]), (255, 255, 0),2)
    cv.rectangle(img, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (255, 255, 255))
    cv.imshow('image', img)
    cv.waitKey(2000)
    return


def generatedata(targetdir):
    targetdir = targetdir + 'train'
    xmlfiles = os.listdir(targetdir + '/Annotations')
    for xmlfile in tqdm(xmlfiles):
        xmlpath = os.path.join(targetdir, 'Annotations', xmlfile)
        imgpath = os.path.join(targetdir, 'JPEGImages', xmlfile[:-4] + '.jpg')
        img = cv.imread(imgpath)
        # print(img.shape)
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        # 读取所有anchor
        objects = root.findall('object')
        num = len(objects)
        index = 0
        
        # 一个框框的数据，没有交际为class0，有交集为class1
        # 如果图片只有一个框框，直接保存
        if num == 1:
            for box in objects:
                imgsavepath = os.path.join(savedir, 'class0', xmlfile[:-4] + '_' + str(index) + '.jpg')
                save_subimg(img,imgsavepath,box)
                index += 1
        # 如果图片有多个框框，判断框框是否有交集
        else:
            for i in range(num):
                isclass0 = True
                for j in range(num):
                    if j==i:
                        continue
                    if is2anchorcross(objects[i],objects[j]):
                        isclass0 = False
                        break
                if isclass0:
                    #print('save to class0')
                    imgsavepath = os.path.join(savedir, 'class0', xmlfile[:-4] + '_' + str(index) + '.jpg')
                    save_subimg(img,imgsavepath,objects[i])
                else:
                    #print('save to class1')
                    imgsavepath = os.path.join(savedir, 'class1', xmlfile[:-4] + '_' + str(index) + '.jpg')
                    save_subimg(img,imgsavepath,objects[i])
                index += 1
        # 生成两个有交集的框框的数据，class1
        for i in range(1,num):
            for c in combinations(objects, i + 1):
                c_minx = []
                c_miny = []
                c_maxx = []
                c_maxy = []
                for box in c:
                    c_minx.append(int(box.find('bndbox').find('xmin').text))
                    c_miny.append(int(box.find('bndbox').find('ymin').text))
                    c_maxx.append(int(box.find('bndbox').find('xmax').text))
                    c_maxy.append(int(box.find('bndbox').find('ymax').text))
                new_minx = min(c_minx)
                new_miny = min(c_miny)
                new_maxx = max(c_maxx)
                new_maxy = max(c_maxy)
                # showimg(c_minx, c_miny, c_maxx, c_maxy, [new_minx, new_miny, new_maxx, new_maxy], img)
                if issaveImg(c_minx, c_miny, c_maxx, c_maxy,img):

                    new_img = img[new_miny:new_maxy, new_minx:new_maxx]
                    imgsavepath = os.path.join(savedir, 'class' + str(i), xmlfile[:-4] + '_' + str(index) + '.jpg')
                    # print(imgsavepath)
                    cv.imwrite(imgsavepath, new_img)
                    index += 1
                    

# 生成分类数据，一个框如果和别的框没有交集为0，否则为1
if __name__ == '__main__':
    targetdir = ['/raid/huayan_nfs/share/data/detection/online_data/20211207_online_train/','/raid/huayan_nfs/share/data/detection/online_data/20211213_RJ_multi_sku_train/','/raid/huayan_nfs/share/data/detection/online_data/20211216_online_train/','/raid/huayan_nfs/share/data/detection/online_data/20220210_online_train/','/raid/huayan_nfs/share/data/detection/online_data/20220428_outsourcing_labeled_refine_train/']
    for i in range(10):
            if not os.path.exists(savedir+'class'+str(i)):
                os.makedirs(savedir+'class'+str(i))
    for dir in targetdir:
        generatedata(dir)
