from utils.output_helper import parse_xml
import os


if __name__ == '__main__':
    data_root = ''
    files = os.listdir(data_root)

    output_root = ''

    for i, file in enumerate(files):
        print('{}/{} --->'.format(i+1, len(files)))

        if not file.lower().endswith('.jpg'):
            continue

        xml_file = file.replace('jpg', 'xml')
        xml_file = os.path.join(data_root, xml_file)

        if os.path.exists(xml_file):
            objects = parse_xml(xml_file)

            txt_file = os.path.join(output_root, xml_file.replace('xml', 'txt'))
            with open(txt_file, 'w') as f:
                for obj_struct in objects:
                    f.write('{} {} {} {} {}\n'.format(obj_struct['name'], obj_struct['bbox'][0], obj_struct['bbox'][1], obj_struct['bbox'][2], obj_struct['bbox'][3]))