import os
from tqdm import tqdm

if __name__ == '__main__':
    rootdir = '/raid/AI_lai/share/wb/code/mmrotate-main/data/custom_dataset/trainval/'
    imgfiles = os.listdir(rootdir+'images')
    for imgname in tqdm(imgfiles):
        filename = imgname[:-4]
        imgpath = os.path.join(rootdir,'images',imgname)
        xmlpath = os.path.join(rootdir,'annfiles',imgname[:-4]+'.txt')
        if not os.path.exists(xmlpath):
            #os.remove(imgpath)
            print(imgpath)
            print(xmlpath)
            
    annfiles = os.listdir(rootdir+'annfiles')
    for annname in tqdm(annfiles):
        filename = annname[:-4]
        imgpath = os.path.join(rootdir,'images',annname[:-4]+'.png')
        xmlpath = os.path.join(rootdir,'annfiles',annname)
        if not os.path.exists(imgpath):
            os.remove(xmlpath)
            print(imgpath)
            print(xmlpath)