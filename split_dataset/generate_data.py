import numpy as np
import os
import glob
from tqdm import tqdm
import xml.etree.cElementTree as ET
from itertools import combinations
import cv2 as cv

savedir = '/raid/huayan_nfs/share/data/detection/generate_data/'

def generatedata(targetdir):
    targetdir = targetdir + 'train'
    xmlfiles = os.listdir(targetdir+'/Annotations')
    for xmlfile in tqdm(xmlfiles):
        xmlpath = os.path.join(targetdir, 'Annotations',xmlfile)
        imgpath = os.path.join(targetdir,'JPEGImages',xmlfile[:-4]+'.jpg')
        img = cv.imread(imgpath)
        # print(img.shape)
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        objects = root.findall('object')
        num = len(objects)
        # print('-------')
        index = 0
        for i in range(num):
            for c in combinations(objects,i+1):
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

                new_img = img[new_miny:new_maxy,new_minx:new_maxx]
                imgsavepath = os.path.join(savedir,'class'+str(i),xmlfile[:-4]+'_'+str(index)+'.jpg')
                # print(imgsavepath)
                cv.imwrite(imgsavepath,new_img)
                index += 1
                # print(new_minx,new_miny,new_maxx,type(new_maxy))

        # print(objects, len(objects))


if __name__ == '__main__':
    targetdir = ['/raid/huayan_nfs/share/data/detection/online_data/20211207_online_train/','/raid/huayan_nfs/share/data/detection/online_data/20211213_RJ_multi_sku_train/','/raid/huayan_nfs/share/data/detection/online_data/20211216_online_train/','/raid/huayan_nfs/share/data/detection/online_data/20220210_online_train/','/raid/huayan_nfs/share/data/detection/online_data/20220428_outsourcing_labeled_refine_train/']
    for i in range(10):
            if not os.path.exists(savedir+'class'+str(i)):
                os.makedirs(savedir+'class'+str(i))
    for dir in targetdir:
        generatedata(dir)
