import os
import cv2
from utils.output_helper import parse_xml


if __name__ == '__main__':
    xml_folder = '../datasets/labels/train/'
    img_folder = '../datasets/images/train'

    files = os.listdir(xml_folder)
    for i, _file in enumerate(files):
        print('{}/{}: {}'.format(i+1, len(files), _file))

        if not _file.lower().endswith('.xml'):
            continue
        
        xml_file = os.path.join(xml_folder, _file)
        img_file = os.path.join(img_folder, _file.replace('xml', 'jpg'))

        objects = parse_xml(xml_file)
        img = cv2.imread(img_file)
        img_width, img_height = img.shape[1], img.shape[0]
        x1, y1, x2, y2 = objects['bbox'][0], objects['bbox'][1], objects['bbox'][2], objects['bbox'][3]

        with open(xml_file.replace('xml', 'txt'), 'w') as f:
            ratio_x = x1 * 1.0 / img_width
            ratio_y = y1 * 1.0 / img_height
            ratio_w = (x2 - x1) * 1.0 / img_width
            ratio_h = (y2 - y1) * 1.0 / img_height
            f.write('{} {} {} {} {}\n'.format(0, ratio_x, ratio_y, ratio_w, ratio_h))
