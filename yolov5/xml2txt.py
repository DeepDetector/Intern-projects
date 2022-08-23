import os
import cv2
from utils.output_helper import parse_xml

if __name__ == '__main__':
    xml_folder = 'datasets/labels/train/'
    txt_folder = 'datasets/labels/train'
    img_folder = 'datasets/images/train'

    img_files = os.listdir(img_folder)
    for i, img_file in enumerate(img_files):
        print('{}/{}: {}'.format(i + 1, len(img_files), img_file))

        # if not img_file.lower().endswith('.jpg'):
        #     continue

        xml_file = os.path.join(xml_folder, img_file.replace('.jpg', '.xml'))
        txt_file = os.path.join(xml_folder, img_file.replace('.jpg', '.txt'))
        img_file = os.path.join(img_folder, img_file)

        if os.path.exists(txt_file):
            print('{} exists ......................'.format(txt_file))
            continue

        if os.path.exists(xml_file):
            objects = parse_xml(xml_file)
        else:
            objects = None
                
        img = cv2.imread(img_file)
        img_width, img_height = img.shape[1], img.shape[0]

        with open(txt_file, 'w') as f:
            if objects is not None and len(objects) > 0:
                for obj in objects:
                    x1, y1, x2, y2 = obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]

                    ratio_x = (x1 + x2) * 0.5 / img_width
                    ratio_y = (y1 + y2) * 0.5 / img_height
                    ratio_w = (x2 - x1) * 1.0 / img_width
                    ratio_h = (y2 - y1) * 1.0 / img_height
                    f.write('{} {} {} {} {}\n'.format(0, ratio_x, ratio_y, ratio_w, ratio_h))
            else:
                f.write('')

