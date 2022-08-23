import os
import random
import shutil
import zipfile


def movefile():
    rootdir = '/raid/AI_lai/share/video_data/'
    files = os.listdir(rootdir)
    random.shuffle(files)
    for i in range(100):
        shutil.copy(rootdir+files[i],'/raid/AI_lai/share/wb/code/source_code_yolo_ib/video/'+files[i])

def zipfiles():
    rootdir = '/raid/AI_lai/share/wb/code/source_code_yolo_ib/video_data/'
    files = os.listdir(rootdir)
    for name in files:
        fz = zipfile.ZipFile(rootdir+name,'r')
        for file in fz.namelist():
            fz.extract(file,rootdir)
        fz.close()
        os.remove(rootdir+name)


def movevideofile():
    rootdir = '/raid/AI_lai/share/wb/code/source_code_yolo_ib/video_data/' 
    targetdir = '/raid/AI_lai/share/wb/code/source_code_yolo_ib/video/'
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    dirs = os.listdir(rootdir)
    for dir in dirs:
        shutil.copy(os.path.join(rootdir,dir,'DR.mp4'),targetdir+dir+'_DR.mp4')
        shutil.copy(os.path.join(rootdir,dir,'TR.mp4'),targetdir+dir+'_TR.mp4')


if __name__ == '__main__':
    # movefile()
    # zipfiles()
    movevideofile()