import os
from tqdm import tqdm
xmldir = '/raid/AI_lai/CYJ/data/sku_detection/20211207_online_test/val/Annotations/'
jpgdir = '/raid/AI_lai/CYJ/data/sku_detection/20211207_online_test/val/JPEGImages_match/'

jpgs = os.listdir(jpgdir)
xmls = os.listdir(xmldir)

for i in range(len(jpgs)):
    jpgs[i] = jpgs[i][:-4]
for i in range(len(xmls)):
    xmls[i] = xmls[i][:-4]
    
for jpg in jpgs:
    if jpg not in xmls:
        print(jpg)
    

