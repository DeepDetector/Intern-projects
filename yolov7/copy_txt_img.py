import os
import random
from tqdm import tqdm
import shutil
#txtsavepath = './dataset1/ImageSets/Main'
xmldirs = ['20211206_YB_background_train','20211207_online_train','20211209_base_background_train','20211209_base_train','20211209_single_origin_train',
'20211210_outsourcing_labeled_train']
#xmldirs = ['20211206_YB_background_train','20211207_online_train']
#xmlfilepath = './dataset1/Annotations'

for dir in xmldirs:
    xmldir = os.path.join('/raid/huayan_nfs/CYJ/data/detection/',dir,'train/TXT')
    #xmlfilepath = os.readlink(xmlfilepath)
    total_xml = os.listdir(xmldir)
    
    for xml in tqdm(total_xml):
        jpgname = xml[:-4]+'.jpg'
        
        originpath = os.path.join(xmldir,xml)
        targetpath = os.path.join('./20211212_train_ln/labels/',xml)
        shutil.copyfile(originpath,targetpath)
        
        originpath = os.path.join('/raid/huayan_nfs/CYJ/data/detection/',dir,'train/JPEGImages/',jpgname)
        targetpath = os.path.join('./20211212_train_ln/images/',jpgname)
        shutil.copyfile(originpath,targetpath)
