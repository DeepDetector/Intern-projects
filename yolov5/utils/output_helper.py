import os
from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET


def save_annotation_to_xml(xml_file, annotation_boxes, annotation_classes):
    root = Element('annotation')
    folder_ = SubElement(root, 'folder')
    filename_ = SubElement(root, 'filename')
    path_ = SubElement(root, 'path')
    source_ = SubElement(root, 'source')
    database__ = SubElement(source_, 'database')
    size_ = SubElement(root, 'size')
    width__ = SubElement(size_, 'width')
    height__ = SubElement(size_, 'height')
    depth__ = SubElement(size_, 'depth')
    segmented_ = SubElement(root, 'segmented')

    folder_.text = os.path.split(xml_file)[0]
    filename_.text = os.path.split(xml_file)[1]
    path_.text = xml_file[:-4]+'.jpg'
    database__.text = ''
    width__.text = '360'
    height__.text = '640'
    depth__.text = '3'
    segmented_.text = '0'

    if len(annotation_boxes) > 0:
        for box, class_name in zip(annotation_boxes, annotation_classes):
            object_ = SubElement(root, 'object')
            name__ = SubElement(object_, 'name')
            pose__ = SubElement(object_, 'pose')
            truncated__ = SubElement(object_, 'truncated')
            difficult__ = SubElement(object_, 'difficult')
            bndbox__ = SubElement(object_, 'bndbox')
            xmin___ = SubElement(bndbox__, 'xmin')
            ymin___ = SubElement(bndbox__, 'ymin')
            xmax___ = SubElement(bndbox__, 'xmax')
            ymax___ = SubElement(bndbox__, 'ymax')
            name__.text = class_name
            pose__.text = '0'
            truncated__.text = '0'
            difficult__.text = '0'
            xmin___.text = str(box[0])
            ymin___.text = str(box[1])
            xmax___.text = str(box[2])
            ymax___.text = str(box[3])

    tree = ElementTree(root)
    tree.write(xml_file, encoding='utf-8')


def save_annotation_and_score_to_xml(xml_file, annotation_boxes, annotation_classes, annotation_scores):
    root = Element('annotation')
    folder_ = SubElement(root, 'folder')
    filename_ = SubElement(root, 'filename')
    path_ = SubElement(root, 'path')
    source_ = SubElement(root, 'source')
    database__ = SubElement(source_, 'database')
    size_ = SubElement(root, 'size')
    width__ = SubElement(size_, 'width')
    height__ = SubElement(size_, 'height')
    depth__ = SubElement(size_, 'depth')
    segmented_ = SubElement(root, 'segmented')

    folder_.text = os.path.split(xml_file)[0]
    filename_.text = os.path.split(xml_file)[1]
    path_.text = xml_file[:-4]+'.jpg'
    database__.text = ''
    width__.text = '4160'
    height__.text = '8320'
    depth__.text = '3'
    segmented_.text = '0'

    if len(annotation_boxes) > 0:
        for box, class_name, score in zip(annotation_boxes, annotation_classes, annotation_scores):
            object_ = SubElement(root, 'object')
            name__ = SubElement(object_, 'name')
            pose__ = SubElement(object_, 'pose')
            score__ = SubElement(object_, 'score')
            truncated__ = SubElement(object_, 'truncated')
            difficult__ = SubElement(object_, 'difficult')
            bndbox__ = SubElement(object_, 'bndbox')
            xmin___ = SubElement(bndbox__, 'xmin')
            ymin___ = SubElement(bndbox__, 'ymin')
            xmax___ = SubElement(bndbox__, 'xmax')
            ymax___ = SubElement(bndbox__, 'ymax')
            name__.text = class_name
            pose__.text = '0'
            score__.text = str(score)
            truncated__.text = '0'
            difficult__.text = '0'
            xmin___.text = str(box[0])
            ymin___.text = str(box[1])
            xmax___.text = str(box[2])
            ymax___.text = str(box[3])

    tree = ElementTree(root)
    tree.write(xml_file, encoding='utf-8')


# 读取annotation里面的label数据
def parse_xml(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {'name': obj.find('name').text,
                      'pose': obj.find('pose').text,
                      'truncated': int(obj.find('truncated').text),
                      'difficult': int(obj.find('difficult').text)}
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects


