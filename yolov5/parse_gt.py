from utils.output_helper import parse_xml
import os


if __name__ == '__main__':
    data_root = '../../AG_DATA/ALL'
    files = os.listdir(data_root)

    output_root = '../../AG_DATA/ALL-GT'

    for i, sfile in enumerate(files):
        print('{}/{} --->'.format(i+1, len(files)))

        if not sfile.lower().endswith('.jpg'):
            continue

        xml_file = sfile.replace('jpg', 'xml')
        xml_file = os.path.join(data_root, xml_file)

        if os.path.exists(xml_file):
            objects = parse_xml(xml_file)
            txt_file = os.path.join(output_root, sfile.replace('jpg', 'txt'))
            with open(txt_file, 'w') as f:
                for obj_struct in objects:
                    print(txt_file, '-->', obj_struct)
                    f.write('{} {} {} {} {}\n'.format(obj_struct['name'], obj_struct['bbox'][0], obj_struct['bbox'][1], obj_struct['bbox'][2], obj_struct['bbox'][3]))
        else:
            txt_file = os.path.join(output_root, sfile.replace('jpg', 'txt'))
            with open(txt_file, 'w') as f:
                f.write('')