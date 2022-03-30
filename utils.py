import numpy as np
from typing import *
import cv2
import json


class Box(object):

    def __init__(self, bbox : List):
        self.xmin = bbox[0]
        self.ymin = bbox[1]
        self.xmax = bbox[2]
        self.ymax = bbox[3]
        self.xcenter = (self.xmin + self.xmax) / 2
        self.ycenter = (self.ymin + self.ymax) / 2

    def get_box(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

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
    if area_merge / min(area(box1), area(box2)) >= threshold:
        return True
    return False


def with_line(box1, box2, threshold = 5):
    if abs(box1.ycenter - box2.ycenter) <= threshold:
        return True
    return False


def get_color():
    color = tuple([int(x) for x in np.random.choice(range(256), size=3)])
    return color

def draw_rectangle(image, box, color):
    box = list(map(int, box))
    image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color = color, thickness = 1)
    return image

def draw_arrow(image, box1, box2, color):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    xcenter1 = int((xmin1 + xmax1) / 2)
    ycenter1 = int((ymin1 + ymax1) / 2)
    xcenter2 = int((xmin2 + xmax2) / 2)
    ycenter2 = int((ymin2 + ymax2) / 2)
    image = cv2.arrowedLine(image, (xcenter1, ycenter1), (xcenter2, ycenter2), color = color, thickness = 1)
    return image

def read_img(filename):
    image = cv2.imread(filename)
    h, w = image.shape[:2]
    return image, h, w

def xywh2xyxy(box):
    x, y, w, h = box
    xmin = x - w / 2
    xmax = x  + w / 2
    ymin = y  - h / 2
    ymax = y + h / 2
    return [xmin, ymin, xmax, ymax]

dict_label = {
    0 : "table",
    1 : "table_column",
    2 : "table_row",
    3 : "table_column_header",
    4 : "table_projected_row_header",
    5 : "table_spanning_cell"
}

def read_file(filename):
    img_path = filename.replace("labels", "images").split(".")[0] + ".jpg"
    _, h_img, w_img = read_img(img_path)

    with open(filename, "r") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip().split(" ")
        label = int(d[0])
        xc = float(d[1]) * w_img
        yc = float(d[2]) * h_img
        wc = float(d[3]) * w_img
        hc = float(d[4]) * h_img
        result.append((Box(xywh2xyxy([xc, yc, wc, hc])), dict_label[label]))
    return result

def getxyxy(box):
    bbox = []
    for b in box:
        bbox += b
    xmin = min(bbox[::2])
    ymin = min(bbox[1::2])
    xmax = max(bbox[::2])
    ymax = max(bbox[1::2])
    return [xmin, ymin, xmax, ymax]

def read_ocr(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    result = {}
    for idx, d in enumerate(data):
        result[idx] = {
            "box" : Box(bbox = d['bbox']),
            "text" : d['text'],
            "id" : idx
        }
    return result


if __name__ == '__main__':

    filename = 'ocr_labels/PMC6233401_table_3_words.json'
    result = read_ocr(filename)
    print(result)

    










 

