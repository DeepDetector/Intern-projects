import pandas as pd
from tqdm import tqdm
import os
import random
import cv2 as cv
# v2版本根据图片大小进行二次筛选
if __name__ == '__main__':
    classdir = ['class0', 'class1', 'class2', 'class3']
    train = {'path': [], 'label': []}
    validate = {'path': [], 'label': []}
    targetdir = '/raid/AI_lai/share/wb/dataset/generate_data_iou0.1_0.3/'
    for i in range(4):
        dirpath = targetdir + classdir[i]
        files = os.listdir(dirpath)
        random.shuffle(files)
        num = len(files)
        for k in tqdm(range(int(0.8 * num))):
            img = cv.imread(targetdir+classdir[i] + '/' + files[k])
            h = img.shape[0]
            w = img.shape[1]
            if h*w < 150 or h<8 or w<8:
                continue
            train['label'] = train['label'] + [str(i)]
            train['path'] = train['path'] + [classdir[i] + '/' + files[k]]
            
        for k in tqdm(range(int(0.8 * num), num)):
            img = cv.imread(targetdir+classdir[i] + '/' + files[k])
            h = img.shape[0]
            w = img.shape[1]
            if h*w < 150 or h<8 or w<8:
                continue
            validate['label'] = validate['label'] + [str(i)]
            validate['path'] = validate['path'] + [classdir[i] + '/' + files[k]]
    train = pd.DataFrame(train, columns=['path', 'label'])
    train.to_csv(targetdir+'train_area150.csv', index=False)
    validate = pd.DataFrame(validate, columns=['path', 'label'])
    validate.to_csv(targetdir+'validate_area150.csv', index=False)
