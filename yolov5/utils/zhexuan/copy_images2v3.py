import os 
import shutil
from tqdm import tqdm
from glob import glob

v2_data_root = '/raid/ZZX/projects/source_code_yolo_ib/datasets/v2/images/train/*'
v2_label_root = '/raid/ZZX/projects/source_code_yolo_ib/datasets/v2/labels/train/*'

train_root = '/raid/ZZX/projects/source_code_yolo_ib/datasets/v3/images/train/'
label_root = '/raid/ZZX/projects/source_code_yolo_ib/datasets/v3/labels/train/'

for image_path in glob(train_root + '/*'):
    if 'mask' not in image_path:
        os.remove(image_path)

for ann_path in glob(label_root + '/*'):
    os.remove(ann_path)

for cs in sorted(glob(v2_data_root)):
    if 'mixed' in cs or 'single' in cs: continue
    for c in sorted(glob(cs + '/*.jpg')):
        if 'mask' in c: continue
        filename = c.split('/')[-1]
        assert os.path.isfile(c.replace('images', 'labels').replace('jpg', 'txt')), c.replace('images', 'labels').replace('jpg', 'txt')
        shutil.copyfile(c, train_root + filename)
        shutil.copyfile(c.replace('images', 'labels').replace('jpg', 'txt'), label_root + filename.replace('jpg', 'txt'))
        # lines = open(c.replace('images', 'labels').replace('jpg', 'txt')).read().splitlines()
        # if len(lines)>0:
        #     print(lines)
        # exit()