import os
import cv2
from utils.output_helper import parse_xml
from shutil import copyfile


if __name__ == '__main__':
    sub_folders = ['ChenYanjie', 'HeWude', 'HuangWei', 'HuWei', 'LinChao',
                   'LinHuixiang', 'LuoPeipei', 'TongYuhong', 'YaoJinfa',
                   'YaoQingyuan', 'ZhangAnguo']
    source_folder = '/raid/LHX/YOLO5-TEST/AG_DATA/'
    target_xml_folder = 'datasets/labels/train/'
    target_img_folder = 'datasets/images/train'

    for sf in sub_folders:
        sf_ = os.path.join(source_folder, sf)

        files = os.listdir(sf_)
        for i, _file in enumerate(files):
            print('{}/{}: {}'.format(i+1, len(files), _file))

            if not _file.lower().endswith('.jpg'):
                continue

            img_file = os.path.join(sf_, _file)
            xml_file = os.path.join(sf_, _file.replace('jpg', 'xml'))

            img = cv2.imread(img_file)
            img_width, img_height = img.shape[1], img.shape[0]

            target_img_file = os.path.join(target_img_folder, _file)
            copyfile(img_file, target_img_file)

            if not os.path.exists(xml_file):
                objects = None
            else:
                objects = parse_xml(xml_file)

            target_xml_file = os.path.join(target_xml_folder, _file.replace('jpg', 'txt'))
            with open(target_xml_file, 'w') as f:
                if objects is not None and len(objects) > 0:
                    for obj in objects:
                        x1, y1, x2, y2 = obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]

                        ratio_x = x1 * 1.0 / img_width
                        ratio_y = y1 * 1.0 / img_height
                        ratio_w = (x2 - x1) * 1.0 / img_width
                        ratio_h = (y2 - y1) * 1.0 / img_height
                        f.write('{} {} {} {} {}\n'.format(0, ratio_x, ratio_y, ratio_w, ratio_h))
                else:
                    f.write('')
