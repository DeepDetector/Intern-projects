import os
from tqdm import tqdm
from shutil import copy
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import math


def rotate(angle, x, y):
    """
    Arc rotation based on origin
    :param angle:
    :param x:
    :param y:
    :return:
    """
    rotatex = math.cos(angle) * x - math.sin(angle) * y
    rotatey = math.cos(angle) * y + math.sin(angle) * x
    return rotatex, rotatey


def xy_rorate(theta, x, y, centerx, centery):
    """
    Rotate for center point
    :param theta:
    :param x:
    :param y:
    :param centerx:
    :param centery:
    :return:
    """
    r_x, r_y = rotate(theta, x - centerx, y - centery)
    return centerx + r_x, centery + r_y


def rec_rotate(x, y, width, height, theta):
    """
    Rotate the center point, input the X, y, width, height and radian of the rectangle,
    and convert it to Quad format
    :param x:
    :param y:
    :param width:
    :param height:
    :param theta:
    :return:
    """
    centerx = x + width / 2
    centery = y + height / 2

    x1, y1 = xy_rorate(theta, x, y, centerx, centery)
    x2, y2 = xy_rorate(theta, x + width, y, centerx, centery)
    x3, y3 = xy_rorate(theta, x, y + height, centerx, centery)
    x4, y4 = xy_rorate(theta, x + width, y + height, centerx, centery)

    return x1, y1, x2, y2, x4, y4, x3, y3


def roxml2txt(rootdir, file):
    xmlpath = os.path.join(rootdir, file)
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    # localImgPath = root.find('path').text
    # for size in root.findall('size'):
    #     height = size.find('height').text
    #     width = size.find('width').text
    #     depth = size.find('depth').text
    fr_line = []
    for object in root.findall('object'):
        name = object.find('name').text
        robndbox = object.find('robndbox')
        if robndbox is None:
            continue
        difficult = object.find('difficult').text
        cx = float(robndbox.find('cx').text)
        cy = float(robndbox.find('cy').text)
        w = float(robndbox.find('w').text)
        h = float(robndbox.find('h').text)
        angle = float(robndbox.find('angle').text)

        x = cx - w / 2
        y = cy - h / 2
        if angle < 1.57:
            theta = round(angle, 6)
        else:
            theta = round(angle - np.pi, 6) 
        x1, y1, x2, y2, x4, y4, x3, y3 = rec_rotate(x, y, w, h, theta)
        x1, y1, x2, y2, x4, y4, x3, y3 = int(x1), int(y1), int(x2), int(y2), int(x4), int(y4), int(x3), int(y3)
        fr_line.append(
            str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' + str(x4) + ' ' + str(y4) + ' ' + str(
                x3) + ' ' + str(y3) + ' ' + name + ' ' + difficult + '\n')
    return fr_line

 
def move(rootdir, targetdir):
    if not os.path.exists(rootdir):
        raise 'rootdir error'
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    if not os.path.exists(os.path.join(targetdir,'images')):
        os.makedirs(os.path.join(targetdir,'images'))
    if not os.path.exists(os.path.join(targetdir, 'annfiles')):
        os.makedirs(os.path.join(targetdir, 'annfiles'))
    files = os.listdir(rootdir)

    for file in tqdm(files):
        if file.endswith(('.jpg','.png','PNG','jpeg')):
            copy(os.path.join(rootdir, file), os.path.join(targetdir, 'images',file[:-4]+'.png'))
        else:
            try:
                txt = roxml2txt(rootdir, file)
                xmlsavepath = os.path.join(targetdir,'annfiles',file[:-4] + '.txt')
                if os.path.exists(xmlsavepath):
                    os.remove(xmlsavepath)
                for line in txt:
                    f2 = open(xmlsavepath, 'a')
                    f2.write(line)
            except TypeError:
                print(file)


if __name__ == '__main__':
    root_dir = '/raid/AI_lai/share/wb/dataset/Rotate_dataset/'
    target_dir = '/raid/AI_lai/share/wb/code/mmrotate-main/data/custom_dataset/test'
    # ['0000','0010','0011','0012','0013']
    for i in ['0002']:
        move(root_dir+i, target_dir)
