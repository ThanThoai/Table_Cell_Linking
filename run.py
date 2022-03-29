import os
import numpy as np
import cv2
import json
from typing import *
import glob
import tqdm

ROOT_IMG = "images_table"
ROOT_LABEL = "labels_table"
ROOT_OCR = "ocrs_table"

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

    def get_box(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def __rstr__(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]


def xywh2xyxy(box):
    x, y, w, h = box
    xmin = x - w / 2
    xmax = x  + w / 2
    ymin = y  - h / 2
    ymax = y + h / 2
    return [xmin, ymin, xmax, ymax]


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
        result.append((Box(xywh2xyxy([xc, yc, wc, hc])), label))
    return result

def read_ocr(filename):
    with open(filename, 'r') as f:
        data = json.load(f)['text']
    result = {}
    for idx, d in enumerate(data):
        result[idx] = {
            "bbox" : Box(bbox = getxyxy(d['bbox'])),
            "text" : d['text'],
            "id" : idx
        }
    return result

def with_line(box1, box2, threshold = 5):
    if abs(box1.xcenter - box2.xcenter) <= threshold:
        return True
    return False

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

def get_box_text_in(box : Box, boxes : Dict):
    result = {}
    for idx in boxes.keys():
        if is_overlap(boxes[idx]['bbox'], box, threshold = 0.9):
            result[idx] = boxes[idx]
    return result


def get_multiline(dict_box : Dict):
    result = {}
    idx = 0
    dict_box = {k : v for k, v in sorted(dict_box.items(), key = lambda  x : x[1]['bbox'].xmin)}
    print(dict_box)
    while len(dict_box) > 0:
        lst_idx = list(map(int, dict_box.keys()))
        lines = [lst_idx[0]]
        for item in lst_idx[1:]:
            if with_line(dict_box[lst_idx[0]]['bbox'], dict_box[item]['bbox']):
                lines.append(item)
        box_lines = [dict_box[i] for i in lines]
        box_lines = sorted(box_lines, key = lambda x : x['bbox'].xmin)
        for item in lines:
            dict_box.pop(item)
        result[f"line-{idx}"] = box_lines
    return result

def get_box_cell(metadata, boxes_text):
    metadata['cell'] = []
    for col in metadata['table_column']:
        box_col = col['box_item']
        for row in metadata['table_row']:
            box_row = row['box_item']
            box_cell =Box([box_col.xmin, box_row.ymin, box_col.xmax, box_row.ymax])

            text_in_cell = get_box_text_in(box_cell, boxes_text)
            list_id = [i['id'] for i in text_in_cell.values()]
            metadata['cell'].append(
                {
                    "box_item" : box_cell,
                    "list_id" : list_id,
                    "list_box_text" : text_in_cell
                }
            )
    return metadata

def get_metadata(boxes : List, boxes_text : Dict):
    metadata = {}
    for box, label in boxes:
        if _LIST_LABEL[label] not in metadata:
            metadata[_LIST_LABEL[label]] = []
        
        list_box_text = get_box_text_in(box,  boxes_text)

        metadata[_LIST_LABEL[label]].append(
            {
                "list_box_text" : list_box_text,
                "box_item" : box,
                "list_id" : [list_box_text[i]['id'] for i in list_box_text.keys()]
            }
        )
    metadata = get_box_cell(metadata, boxes_text)    
    return metadata

def get_header_column(metadata):
    list_box_in_header = metadata['table_column_header'][0]['list_box_text']
    # print(list_box_in_header)
    multiline = get_multiline(list_box_in_header)
    print(multiline)
    line = ""
    max_line = 0
    for key in multiline:
        if len(multiline[key]) >= max_line:
            line = key
            max_line = len(multiline[key])
    list_id = [i['id'] for i in multiline[line]]
    print(list_id)
    return list_id

def create_link_in_row(metadata, matrix):
    list_header = get_header_column(metadata)
    for row in metadata['table_row']:
        box_row = row['box_item']
        flag = False
        for key in ["table_column_header", "table_projected_row_header", "table_spanning_cell"]:
            if key in metadata.keys():
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
            idx_head = list(set(list_id_row) & set(list_id_col))
            idx_tail = list(set(list_id_col) & set(list_header))
            if len(idx_head) == 1 and len(idx_tail) == 1:
                matrix[idx_head[0]][idx_tail[0]] = 1
    return matrix

            
def cluster_box(list_box):
    result = {}
    idx = 0
    list_box = sorted(list_box, key = lambda  x : x['box_item'].xmin)
    while len(list_box) > 0:
        lines = [list_box[0]]
        for item in list_box[1:]:
            if with_line(lines[0]['box_item'], item['box_item']):
                lines.append(item)
        for l in lines:
            list_box.remove(l)
        lines = sorted(lines, key = lambda x : x['box_item'].xmin)
        result[f"cluster-{idx}"] = lines
    return result



def create_link_in_header(metadata, matrix):
    for header in metadata['table_column_header']:
        box_header = header['box_item']
        box_cells = []
        for cell in metadata['cell']:
            box_cell = cell['box_item']

            if is_overlap(box_cell, box_header, threshold = 0.9):
                box_cells.append(cell)

        clusters = cluster_box(box_cells)

        if len(clusters) == 1:
            return matrix
        elif len(clusters) == 2:
            for span in metadata['table_spanning_cell']:
                box_span = span['box_item']
                span_multiline = get_multiline(span['list_box_text'])
                span_idx = int(span_multiline['line-0'][0]['id'])
                if is_overlap(box_span, box_header, threshold = 0.8):
                    for cell in box_cells:
                        if cell['box_item'].xmin >= (span['box_item'].xmin - 5) and cell['box_item'].xmax <= (span['box_item'].xmax + 5):
                            cell_multiline = get_multiline(cell['list_box_text'])
                            cell_idx = int(cell_multiline['line-0'][0]['id'])

                            matrix[span_idx][cell_idx] = 1
            
            return matrix
        else:
            raise("Not supported !!!!!!!")

def create_link_in_cell(metadata, matrix):
    for cell in metadata['cell']:
        cell_multiline = get_multiline(cell['list_box_text'])
        if len(cell_multiline) > 0:
            head_idx = int(cell_multiline['line-0'][0]['id'])
            for key in cell_multiline:
                tail_idx = 0
                if key == "line-0":
                    for c in cell_multiline[key][1:]:
                        tail_idx = c['id']
                else:
                    for c in cell_multiline[key]:
                        tail_idx = c['id']
                matrix[head_idx][tail_idx] = 1
    return matrix

def getxyxy(box):
    bbox = []
    for b in box:
        bbox += b
    xmin = min(bbox[::2])
    ymin = min(bbox[1::2])
    xmax = max(bbox[::2])
    ymax = max(bbox[1::2])
    return [xmin, ymin, xmax, ymax]

def gen_annotations(boxes_ocr, matrix):
    documents = []
    h, w = np.shape(matrix)[:2]
    for ocr in boxes_ocr.values():
        # print(ocr['bbox'])
        temp = {
            "box" : [ocr['bbox'].xmin, ocr['bbox'].ymin, ocr['bbox'].xmax, ocr['bbox'].ymax],
            "text" : ocr['text'],
            "word" : [],
            "id" : ocr['id'],
            "linking" : []
        }
        for i in range(w):
            if matrix[ocr['id']][i] == 1:
                temp['linking'].append([ocr['id'], i])
        documents.append(temp)
    return documents

def draw_rectangle(image, box, color):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[2]), int(box[1]))
    p3 = (int(box[2]), int(box[3]))
    p4 = (int(box[0]), int(box[3]))
    image = cv2.line(image, p1, p2, color = color, thickness = 1)
    image = cv2.line(image, p2, p3, color = color, thickness = 1)
    image = cv2.line(image, p3, p4, color = color, thickness = 1)
    image = cv2.line(image, p1, p4, color = color, thickness = 1)
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

def get_color():
    color = tuple([int(x) for x in np.random.choice(range(256), size=3)])
    return color

def visualize(image, annotation):
    for doc in annotation[12:]:
        color = get_color()
        box_1 = doc['box']
        linking = doc['linking']
        if len(linking) > 0:
            for link in linking:
                i = link[1]
                for doc_2 in annotation:
                    if int(doc_2['id']) == int(i):
                        box_2 = doc_2['box']
                        image = draw_rectangle(image, box_1, color)
                        image = draw_rectangle(image, box_2, color)
                        draw_arrow(image, box_1, box_2, color)
                    break
                break
        break
    return image




                            
def run():
    annotations = {
            "lang" : "vi",
            "info" : {
                "author" : "ThanThoai",
                "version" : 1.0
            },
            "documents" : []
        }
    for idx, jpg in tqdm.tqdm(enumerate(glob.glob(ROOT_IMG + "/*.jpg")[1:2])):
        print(jpg)
        image = cv2.imread(jpg)
        name = jpg.split("/")[-1].split(".")[0]
        ocr_path = os.path.join(ROOT_OCR, name + ".json")
        label_yolo = os.path.join(ROOT_LABEL, name + ".txt")

        boxes_ocr = read_ocr(ocr_path)
        boxes = read_file(label_yolo)

        matrix = np.zeros(shape = (len(boxes_ocr), len(boxes_ocr)))
        metadata = get_metadata(boxes, boxes_ocr)
        matrix = create_link_in_row(metadata, matrix)
        matrix = create_link_in_header(metadata, matrix)
        matrix = create_link_in_cell(metadata, matrix)

        anno = gen_annotations(boxes_ocr, matrix)
        # print(anno)
        image = visualize(image, anno)
        cv2.imwrite(os.path.join("visualize", name + ".jpg"), image)
        temp = {
            "id" : "%05d"%idx,
            "document" : anno,
            "image" : name
        }     
        annotations['documents'].append(temp)
        1/0
    with open("pubtable1m_entity_linking_v1.json", 'w') as f:
        json.dump(annotations, f, ensure_ascii=False)


if __name__ == '__main__':
    run()