import os
import cv2


def parse_image_bbox(image_name):
    name_tokens = image_name.split('_')
    mixed = name_tokens[2]
    mixed_tokens = mixed.split(')')
    x1, y1, x2, y2 = int(mixed_tokens[1]), int(name_tokens[3]), int(name_tokens[4]), int(name_tokens[5])
    prob = float(name_tokens[7][:-4])

    return (x1, y1, x1 + x2, y1 + y2), prob
