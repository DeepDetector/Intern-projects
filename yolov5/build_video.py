import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
import shutil
def flipimg(img):
    height, width = img.shape[:2]
    matRotate = cv.getRotationMatrix2D((height * 0.5, width * 0.5), -90, 1)
    # print(img.shape)
    img = np.rot90(img,axes=(0,1))
    # print(img.shape)
    img = cv.flip(img, 2)
    # print(img.shape)
    return img

if __name__ == '__main__':
    # check()
    rootdir = './save_output_jpg'
    targetdir = './save_output_video'
    dirs = os.listdir(rootdir)
    if os.path.exists(targetdir):
        shutil.rmtree(targetdir)  # delete output folder
    os.makedirs(targetdir)  # make new output folder
    for dir in dirs:
        dirpath = os.path.join(rootdir,dir)
        imgs = os.listdir(dirpath)
        img0 = cv.imread(os.path.join(dirpath,imgs[0]))
        w = img0.shape[0]
        h = img0.shape[1]
        
        # vid_writer = cv.VideoWriter(targetdir+'/'+dir+'.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (h, w))
        vid_writer = cv.VideoWriter(targetdir+'/'+dir+'.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (360, 640))
        imgs.sort(key = lambda x: int(x[:-4]))
        # print(imgs)
        for imgname in tqdm(imgs):
            img = cv.imread(os.path.join(dirpath,imgname))
            # if img.shape != (360,640,3):
            # if h != 360:
                # print(img.shape)
                # img = flipimg(img)
                # print(os.path.join(dirpath,imgname))
                # print(img.shape)
            vid_writer.write(img)
        vid_writer.release()
