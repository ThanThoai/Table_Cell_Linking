import numpy as np
from typing import *


__all__ = ['LIST_LABEL', "Box", "BoxText"]




import cv2

def get_area_merge(box1, box2):
    x1 = max(box1.xmin, box2.xmin)
    y1 = max(box1.ymin, box2.ymin)
    x2 = min(box1.xmax, box2.xmax)
    y2 = min(box1.ymax, box2.ymax)
    return max(1, x2 - x1) * max(1, y2 - y1)

def area(box):
    return max(1, box.xmax - box.xmin) * max(1, box.ymax - box.ymin)

def is_overlap(box1, box2, threshold = 0.85):
    area_merge = get_area_merge(box1, box2)
    if area_merge / area(box1) >= threshold:
        return True
    return False


def with_line(box1, box2, threshold = 5):
    if abs(box1.xcenter - box2.xcenter) <= threshold:
        return True
    return False





    










 

