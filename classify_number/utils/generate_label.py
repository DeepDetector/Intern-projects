import pandas as pd
from tqdm import tqdm
import os
import random

if __name__ == '__main__':
    classdir = ['class0', 'class1', 'class2', 'class3']
    train = {'path': [], 'label': []}
    validate = {'path': [], 'label': []}
    targetdir = '/raid/AI_lai/share/wb/dataset/clean_data_iou0.1_0.3/'
    for i in range(4):
        dirpath = targetdir + classdir[i]
        files = os.listdir(dirpath)
        random.shuffle(files)
        num = len(files)
        for k in tqdm(range(int(0.8 * num))):
            train['label'] = train['label'] + [str(i)]
            train['path'] = train['path'] + [classdir[i] + '/' + files[k]]
        for k in tqdm(range(int(0.8 * num), num)):
            validate['label'] = validate['label'] + [str(i)]
            validate['path'] = validate['path'] + [classdir[i] + '/' + files[k]]
    train = pd.DataFrame(train, columns=['path', 'label'])
    train.to_csv(targetdir+'train.csv', index=False)
    validate = pd.DataFrame(validate, columns=['path', 'label'])
    validate.to_csv(targetdir+'validate.csv', index=False)
