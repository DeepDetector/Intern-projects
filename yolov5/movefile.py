import os
import random
import shutil
import zipfile


def movefile():
    rootdir = '/raid/AI_lai/share/wb/code/source_code_yolo_ib/save_output_jpg/'
    targetdir = '/raid/AI_lai/share/wb/code/source_code_yolo_ib/save_output_video/'
    files = os.listdir(rootdir)
    for filenamne in files:
        if filenamne.endswith('.mp4'):
            shutil.copy(rootdir+filenamne,targetdir+filenamne)

if __name__ == '__main__':
    movefile()
    