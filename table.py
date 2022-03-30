from typing import *
from utils import is_overlap, with_line, Box, read_ocr, read_file, draw_rectangle, get_color, draw_arrow
import os
import numpy as np
import json
import tqdm

ROOT_IMG = "images_table"
ROOT_LABEL = "labels_table"
ROOT_OCR = "ocrs_table"


class Table(object):

    def __init__(self, boxes_text, boxes_element):

        self.boxes_text = boxes_text
        self.boxes_element = boxes_element

        self.label = {
            0 : "table",
            1 : "table_column",
            2 : "table_row",
            3 : "table_column_header",
            4 : "table_projected_row_header",
            5 : "table_spanning_cell"
        }
        self.metadata = self.__get_metadata()
        self.matrix = np.zeros(shape = (len(boxes_text), len(boxes_text)))
    
    def get_box_text_in(self, box):
        result = {}
        for key, value in self.boxes_text.items():
            if is_overlap(value['bbox'], box, threshold = 0.95):
                result[key] = value
        return result

    def get_multiline(self, dict_box : Dict):
        result = {}
        idx = 0
        dict_box = {k : v for k, v in sorted(dict_box.items(), key = lambda  x : x[1]['bbox'].ycenter)}
        for v in dict_box.values():
            result[f'line-{idx}'] = v
            idx += 1
        return result

    def get_box_cell(self, metadata):
        metadata['cell'] = []
        idx = 0
        for col in metadata['table_column']:
            box_col = col['box_item']
            for row in metadata['table_row']:
                box_row = row['box_item']
                box_cell = Box([box_col.xmin, box_row.ymin, box_col.xmax, box_row.ymax])
                text_in_cell = self.get_box_text_in(box_cell)
                lines = self.get_multiline(text_in_cell)
                list_id = [i['id'] for i in text_in_cell.values()]
                metadata['cell'].append(
                    {
                        "box_item" : box_cell,
                        "list_id" : list_id,
                        "list_box_text" : text_in_cell,
                        "lines" : lines,
                        "id" : idx,
                    }
                )
                idx += 1
        return metadata

    def __get_metadata(self):
        metadata = {}
        for box, label in self.boxes_element:
            if self.label[label] not in metadata:
                metadata[self.label[label]] = []
            list_box_text = self.get_box_text_in(box)
            metadata[self.label[label]].append(
                {
                    "list_box_text" : list_box_text,
                    "box_item" : box,
                    "list_id" : list(list_box_text.keys()),
                }
            )
        metadata = self.get_box_cell(metadata)
        for key in metadata.keys():
            if key == 'table' or key == 'cell':
                continue
            for item in metadata[key]:
                item['list_cell'] = []
                box_item = item['box_item']
                for cell in metadata['cell']:
                    box_cell = cell['box_item']
                    if is_overlap(box_cell, box_item, threshold = 0.75):
                        item['list_cell'].append(cell)
        metadata['table_row_in_header'] = []
        metadata['table_row_not_in_header'] = []
        for row in metadata['table_row']:
            row_box = row['box_item']
            flag = False
            for header in metadata['table_column_header']:
                header_box = header['box_item']
                if is_overlap(row_box, header_box, threshold = 0.5):
                    metadata['table_row_in_header'].append(row)
                    flag = True
            if "table_project_row_header" in metadata:
                for projected in metadata['table_projected_row_header']:
                    projected_box = projected['box_item']
                    if is_overlap(row_box, projected_box, threshold = 0.5):
                        flag = True
            if flag:
                continue
            metadata['table_row_not_in_header'].append(row)
        return metadata

    def get_header(self):
        list_id = []
        if len(self.metadata['table_row_in_header']) == 1:
            row = self.metadata['table_row_in_header'][0]
            list_cells = row['list_cell']
            for cell in list_cells:
                lines = cell['lines']
                if len(lines) > 0:
                    list_id.append(lines['line-0']['id'])

            return list_id
        elif len(self.metadata['table_row_in_header']) == 2:
            row_header = sorted(self.metadata['table_row_in_header'], key = lambda x : x['box_item'].ymin)
            box_row = row_header[0]['box_item']
            for span in self.metadata['table_spanning_cell']:
                box_span = span['box_item']
                if is_overlap(box_row, box_span, threshold = 0.9):
                    text_in_span = self.get_box_text_in(box_span)
                    span_lines = self.get_multiline(text_in_span)
                    id_head = span_lines['line-0']['id']
                    for cell in row_header[1]['list_cell']:
                        if cell['box_item'].xmin >= box_span.xmin - 2 and cell['box_item'].xmax <= box_span.xmax + 2:
                            if len(cell['lines']) > 0:
                                list_id.append(cell['lines']['line-0']['id'])
                                self.matrix[cell['lines']['line-0']['id']][id_head] = 1
            return list_id

    def get_link_in_cell(self):
        for cell in self.metadata['cell']:
            if len(cell['lines']) > 1:
                head_idx = cell['lines']['line-0']['id']
                for key in cell['lines']:
                    if key != 'line-0':
                        tail_idx = cell['lines'][key]['id']
                        self.matrix[head_idx][tail_idx] = 1


    def get_link_in_row(self):
        for row in self.metadata['table_row_not_in_header']:
            list_id = []
            for cell in row['list_cell']:
                # print(cell)
                if len(cell['lines']) > 0:
                    list_id.append(cell['lines']['line-0']['id'])
            for i1 in list_id:
                for i2 in list_id:
                    if i1 != i2:
                        self.matrix[i1][i2] = 1
                        self.matrix[i2][i1] = 1

    def get_link_cell_with_header(self):
        list_id_header = self.get_header()
        for col in self.metadata['table_column']:
            lst_col = col['list_id']
            header_choice = list(set(lst_col) & set(list_id_header))
            if len(header_choice) == 1:
                if "table_spanning_cell" in self.metadata:
                    for span in self.metadata['table_spanning_cell']:
                        lst_span = span['list_id']
                        lst_col = list(set(lst_col) - set(lst_span))
                        for c in lst_col:
                            self.matrix[c][header_choice] = 1

            list_cell_col = sorted(col['list_cell'], key=lambda x:x['box_item'].ycenter)
            if len(list_cell_col[0]['lines']) == 0:
                continue
            else:
                head_idx = list_cell_col[0]['lines']['line-0']['id']
                for c_col in list_cell_col[1:]:
                    c_id = list(set(c_col['list_id']) - set(list_id_header))
                    if len(c_id) > 1:
                        tail_idx = c_col['lines']['line-0']['id']
                        self.matrix[tail_idx][head_idx] = 1
                    elif len(c_id) == 1:
                        tail_idx = c_id[0]
                        self.matrix[tail_idx][head_idx] = 1



    def create_link(self):
        self.get_link_in_cell()
        self.get_link_in_row()
        self.get_link_cell_with_header()

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


def visualize(image, annotation):
    for doc in annotation:
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
    return image

def run(name, idx):
    image_path = os.path.join(ROOT_IMG, name + ".jpg")
    label_path = os.path.join(ROOT_LABEL, name + ".txt")
    ocr_path = os.path.join(ROOT_OCR, name + ".json")

    image = cv2.imread(image_path)
    boxes_text = read_ocr(ocr_path)
    boxes_element = read_file(label_path)

    print(boxes_element)

    table = Table(boxes_text, boxes_element)
    table.create_link()
    anno = gen_annotations(boxes_text, table.matrix)
    # print(anno)

    image = visualize(image, anno)
    cv2.imwrite(os.path.join("visualize", name + ".jpg"), image)
    temp = {
        "id" : "%05d"%idx,
        "document" : anno,
        "image" : name
    }     
    return temp



if __name__ == "__main__":
    import cv2
    names = [i.split(".")[0] for i in os.listdir(ROOT_IMG)]

    annotations = {
        "lang" : "vi",
        "info" : {
            "author" : "ThanThoai",
            "version" : 1.0
        },
        "documents" : []
    }

    with open("log_30_3.txt", 'w') as wr:
        for idx, name in tqdm.tqdm(enumerate(names)):
            try:
                anno = run(name, idx)
                annotations['documents'].append(anno)
            except Exception as err:
                wr.write(f"{name}\t{err}")
        # 1/0
    with open("pubtable1m_entity_linking_v1.json", 'w') as f:
        json.dump(annotations, f, ensure_ascii=False)
    
    # image = cv2.imread(image_path)

    # boxes_text = read_ocr(ocr_path)
    # boxes_element = read_file(label_path)

    # table = Table(boxes_text, boxes_element)
    # # print(table.metadata['table_column_header'])

    # id_header = table.get_header()
    # print(id_header)

    # color = get_color()
    # for idx in id_header:
    #     image = draw_rectangle(image, boxes_text[idx]['bbox'].get_box(), color)
        
    # cv2.imwrite("header.jpg", image)

    
