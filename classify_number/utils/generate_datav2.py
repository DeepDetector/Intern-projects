import numpy as np
import os
import glob
from tqdm import tqdm
import xml.etree.cElementTree as ET
from itertools import combinations
import cv2 as cv


def iou(box1, box2):
    left = max(box1[0], box2[0])
    bottom = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    top = min(box1[3], box2[3])
    cross_box = [0, 0, 0, 0]
    iouarea = 0
    if right > left and top > bottom:
        iouarea = (right - left) * (top - bottom)
        cross_box = [left, top, right, bottom]
    return iouarea, cross_box


def saveImg(c_minx, c_miny, c_maxx, c_maxy, img):
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


def showimg(c_minx, c_miny, c_maxx, c_maxy, new_box, img):
    n = len(c_minx)
    for i in range(n):
        cv.rectangle(img, (c_minx[i], c_miny[i]), (c_maxx[i], c_maxy[i]), (255, 255, 0),2)
    cv.rectangle(img, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (255, 255, 255))
    cv.imshow('image', img)
    cv.waitKey(2000)
    return


def generatedata(targetdir):
    xmlfiles = os.listdir(targetdir + '/Annotations')
    for xmlfile in tqdm(xmlfiles):
        xmlpath = os.path.join(targetdir, 'Annotations', xmlfile)
        imgpath = os.path.join(targetdir, 'JPEGImages', xmlfile[:-4] + '.jpg')
        img = cv.imread(imgpath)
        # print(img.shape)
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        objects = root.findall('object')
        num = len(objects)
        # print('-------')
        index = 0
        for i in range(num):
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
                if saveImg(c_minx, c_miny, c_maxx, c_maxy,img):
                    new_img = img[new_miny:new_maxy, new_minx:new_maxx]
                    imgsavepath = os.path.join(targetdir, 'class' + str(i), xmlfile[:-4] + '_' + str(index) + '.jpg')
                    # print(imgsavepath)
                    cv.imwrite(imgsavepath, new_img)
                    index += 1
                    


if __name__ == '__main__':
    dirs = ['C:/Users/Woo/Desktop/train2/']
    for dir in dirs:
        for i in range(10):
            if not os.path.exists(dir + 'class' + str(i)):
                os.makedirs(dir + 'class' + str(i))
        generatedata(dir)
