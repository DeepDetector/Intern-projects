import os
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm
sets = ['train', 'test']
classes = ["something" ]

for image_set in sets:
    image_ids = open('/raid/huayan_nfs/share/wb/code/yolov7-main/20211212_train_ln/ImageSets/Main/%s.txt' % (image_set)).read().split()  
    list_file = open('/raid/huayan_nfs/share/wb/code/yolov7-main/20211212_train_ln/%s.txt' % (image_set), 'w+')
    # print(image_ids)
    for image_id in tqdm(image_ids):
        try:
            #print(image_id)
            list_file.write('20211212_train_ln/images/%s.jpg\n' % (image_id))
        except:
            print('error img:', image_id)
    list_file.close()