from curses import meta
import os
import numpy as np
import cv2
import json
from typing import *

ROOT_IMG = ""
ROOT_LABEL = ""
ROOT_OCR = ""

def read_img(filename):
    image = cv2.imread(filename)
    h, w = image.shape[:2]
    return image, h, w

LIST_LABEL = {
    "table" : 0,
    "table_column" : 1,
    "table_row" : 2,
    "table_column_header" : 3,
    "table_projected_row_header" : 4,
    "table_spanning_cell" : 5,
    "no_object" : 6
}

_LIST_LABEL = {}
for key, value in LIST_LABEL.items():
    _LIST_LABEL[value] = key

class Box(object):

    def __init__(self, bbox : List):
        self.xmin = bbox[0]
        self.ymin = bbox[1]
        self.xmax = bbox[2]
        self.ymax = bbox[3]
        self.xcenter = (self.xmin + self.xmax) / 2
        self.ycenter = (self.ymin + self.ymax) / 2

def xywh2xyxy(box):
    x, y, w, h = box
    xmin = x - w / 2
    xmax = x  + w / 2
    ymin = y  - h / 2
    ymax = y + h / 2
    return [xmin, ymin, xmax, ymax]

def read_file(filename):
    img_path = filename.replace("images", "labels").split(".")[0] + ".jpg"
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
        result.append((Box(xywh2xyxy([xc, yc, wc, hc])), label))
    return result

def read_ocr(filename):
    with open(filename, 'r') as f:
        data = json.load(f)['text']
    result = {}
    for idx, d in data:
        result[idx] = {
            "bbox" : np.array(d['bbox']),
            "text" : d['text'],
            "id" : idx
        }
    return result

def with_line(box1, box2, threshold = 5):
    if abs(box1.xcenter - box2.xcenter) <= threshold:
        return True
    return False


def area_merge(box1, box2):
    x1 = max(box1.xmin, box2.xmin)
    y1 = max(box1.ymin, box2.ymin)
    x2 = min(box1.xmax, box2.xmax)
    y2 = min(box1.ymax, box2.ymax)
    return max(1, x2 - x1) * (1, y2 - y1)

def area(box):
    return max(1, box.xmax - box.xmin) * max(1, box.ymax - box.ymin)

def is_overlap(box1, box2, threshold = 0.85):
    area_merge = area_merge(box1, box2)
    if area_merge / area(box1) >= threshold:
        return True
    return False

def get_box_text_in(box : Box, boxes : Dict):
    result = {}
    for idx in boxes.keys():
        if is_overlap(boxes[idx]['bbox'], box, threshold = 0.9):
            result[idx] = boxes[idx]
    return result


def get_multiline(dict_box : Dict):
    result = {}
    idx = 0
    while len(dict_box) > 0:
        lst_idx = list(dict_box.keys())
        lines = [lst_idx[0]]
        dict_box.pop(lst_idx[0])
        for item in lst_idx[1:]:
            if with_line(dict_box[lst_idx[0]['bbox']], dict_box[item]['bbox']):
                lines.append(item)
        box_lines = [dict_box[i] for i in lines]
        box_lines = sorted(box_lines, key = lambda x : x['bbox'].xmin)
        for item in lines:
            dict_box.pop(item)
        result[f"line-{idx}"] = box_lines
    return result



def get_metadata(boxes : List, boxes_text : Dict):
    metadata = {}
    for box, label in boxes.items():
        if _LIST_LABEL[label] not in metadata:
            metadata[_LIST_LABEL[label]] = []
        
        list_box_text = get_box_text_in(box,  boxes_text)

        metadata[_LIST_LABEL[label]].append(
            {
                "list_box_text" : list_box_text,
                "box_item" : box,
                "list_id" : [i['id'] for i in list_box_text]
            }
        )
    return metadata

def get_header_column(metadata):
    list_box_in_header = metadata['table_column_header']
    multiline = get_multiline(list_box_in_header)
    line = ""
    max_line = 0
    for key in multiline:
        if len(multiline[key]) >= max_line:
            line = key
            max_line = len(multiline[key])
    list_id = [i['id'] for i in multiline[line]]
    return list_id




def create_link_in_row(metadata, matrix):
    list_header = get_header_column(metadata)
    for row in metadata['table_row']:
        box_row = row['box_item']
        flag = False
        for key in ["table_column_header", "table_projected_row_header", "table_spanning_cell"]:
            for item in metadata[key]:
                box_item = item['box_item']
                if is_overlap(box_item, box_row, threshold = 0.5):
                    flag = True
                    break
            if flag:
                break
        if flag:
            continue
        list_id_row = row['list_id']
        for i1 in list_id_row:
            for i2 in list_id_row:
                matrix[i1][i2] = 1
                matrix[i2][i1] = 1
        for col in metadata['table_column']:
            list_id_col = col['list_id']
            idx_head = set(list_id_row) & set(list_id_col)
            idx_tail = set(list_id_col) & set(list_header)
            matrix[idx_head][idx_tail] = 1
    return matrix

def get_box_cell(metadata, boxes_text):
    metadata['cell'] = []
    for col in metadata['table_column']:
        box_col = col['box_item']
        for row in metadata['table_row']:
            box_row = row['box_item']
            box_cell =Box([box_col.xmin, box_row.ymin, box_col.xmax, box_row.ymax])

            text_in_cell = get_box_text_in(box_cell, boxes_text)
            list_id = [i['id'] for i in list_id]
            metadata['cell'].append(
                {
                    "box_item" : box_cell,
                    "list_id" : list_id,
                    "list_box_text" : text_in_cell
                }
            )
    return metadata





def get_column_in_span_cell(metadata):
    list_id_header = get_header_column(metadata)
    for span in metadata['table_spanning_cell']:
        flag = False
        box_span = span['box_item']
        for header in metadata['table_header']:
            box_header = header['box_item']
            if is_overlap(box_span, box_header, threshold = 0.5):
                flag = True
                break
        if flag:
            list_id = []
            multiline = get_multiline(span[''])
            for col in metadata['table_column']:
                box_col = col['box_item']
                if is_overlap(box_span, box_col, threshold = 0.05):
                    list_id_col = col['list_id']
                    id_col = set(list_id_col) & set(list_id_header)
                    list_id.append(id_col)



    

def create_link_in_header(metadata, matrix):
    list_box_in_header = metadata['table_column_header']
    multiline = get_multiline(list_box_in_header)
    if len(multiline) == 1:
        return matrix
    if len(multiline) == 2:
        pass


if __name__ == '__main__':
    pass



        







