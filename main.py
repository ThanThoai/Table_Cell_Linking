from typing import *
import numpy as np
from utils import is_overlap, with_line, Box, read_ocr, read_file, draw_rectangle, get_color, draw_arrow
import tqdm
import json
import os

ROOT_IMG = "images_table"
ROOT_LABEL = "labels_table"
ROOT_OCR = "ocr_labels"
IDX = 0

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


class RelativeBox(object):

    def __init__(self, relative_position):
        self.row_start = relative_position[0]
        self.row_end = relative_position[1]
        self.col_start = relative_position[2]
        self.col_end = relative_position[3]
        self.num_grid = (self.row_end - self.row_start) * (self.col_end - self.col_start)
        self.num_row = self.row_end - self.row_start
        self.num_col = self.col_end - self.col_start


def is_on_same_line(box_a, box_b, min_y_overlap_ratio = 0.8):
    aymin = box_a.ymin
    bymin = box_b.ymin
    aymax = box_a.ymax
    bymax = box_b.ymax

    if aymin > bymin:
        aymin, bymin = bymin, aymin
        aymax, bymax = bymax, aymax
    
    if bymin <= aymax:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([bymin, bymax, aymax])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (aymax - aymin) * min_y_overlap_ratio
            min_b_overlap = (bymax - bymin) * min_y_overlap_ratio

            return overlap >= min_a_overlap or overlap >= min_b_overlap
        else:
            return True
    return False

def stitch_boxes_into_lines(boxes, max_x_dist = 10, min_y_overlap_ratio=0.8):
    if len(boxes) == 1:
        return [boxes]
    elif len(boxes) == 0:
        return []
    merged_boxes = []
    x_sorted_boxes = sorted(list(boxes.values()), key=lambda x: x['box'].xmin)
    skip_idx = set()

    i = 0

    for i in range(len(x_sorted_boxes)):
        if i in skip_idx:
            continue
        
        rightmost_box_idx = i
        line = [rightmost_box_idx]

        for j in range(i + 1, len(x_sorted_boxes)):
            for j in skip_idx:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'], x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idx.add(j)
                rightmost_box_idx = j
        lines = []
        line_idx = 0
        lines.append([line[0]])

        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = curr_box['box'].xmin - prev_box['box'].xmax
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        for box_group in lines:
            merged_box = {}
            merged_box['text'] = " ".join(
                [x_sorted_boxes[idx]['text'] for idx in box_group]
            )
            xmin, ymin = float('inf'), float('inf')
            xmax, ymax = float('-inf'), float('-inf')

            for idx in box_group:
                xmax = max(x_sorted_boxes[idx]['box'].xmax, xmax)
                xmin = min(x_sorted_boxes[idx]['box'].xmin, xmin)
                ymax = max(x_sorted_boxes[idx]['box'].ymax, ymax)
                ymin = min(x_sorted_boxes[idx]['box'].ymin, ymin)
            
            merged_box['box'] = Box([xmin, ymin, xmax, ymax])
            merged_boxes.append(merged_box)

    return merged_boxes
            


class TextBox(object):

    def __init__(self, texts : List, idx : int):
        self.lines = self.get_line(texts, idx)
        if len(self.lines) == 0:
            self.id = None
            self.relative_id = []
        else:
            self.id = self.lines['line-0']['id']
            self.relative_id = [self.lines[key]['id'] for key in self.lines if key != "line-0"]

    def get_line(self, texts : List, idx : int) -> Dict:
        lines = {}
        global IDX
        if len(texts) == 0:
            return lines
        else:
            # print(texts)
            texts = {k : v for k, v in sorted(texts.items(), key=lambda x : x[1]['box'].xmin)}
            box_lines = []
            set_idx = set()
            for i in texts:
                if i in set_idx:
                    continue
                line = [texts[i]]
                set_idx.add(i)
                for j in texts:
                    if j in set_idx:
                        continue
                    if with_line(texts[i]['box'], texts[j]['box']):
                        line.append(texts[j])
                        set_idx.add(j)
                box_lines.append(line)
            for idx, line in enumerate(box_lines):
                line = sorted(line, key = lambda x : x['box'].xmin)
                xmin = min([i['box'].xmin for i in line])
                xmax = max([i['box'].xmax for i in line])
                ymin = min([i['box'].ymin for i in line])
                ymax = max([i['box'].ymax for i in line])
                text = " ".join([i['text'] for i in line])
                lines[f'line-{idx}'] = {
                    "text" : text,
                    "box" : Box([xmin, ymin, xmax, ymax]),
                    "id" : IDX
                }
                IDX += 1
            return lines

    
class Cell(object):

    def __init__(self, relative_position, boudingbox, text, idx, type, text_id):

        self.RBox = RelativeBox(relative_position)
        self.BBox = Box(boudingbox)
        self.TBox = TextBox(text, text_id)
        self.idx = idx
        self.type = type


class Table(object):

    def __init__(self, boxes_text, boxes_element):
        self.boxes_text = boxes_text
        self.boxes_element = boxes_element
        global IDX
        self.label = {
            0 : "table",
            1 : "table_column",
            2 : "table_row",
            3 : "table_column_header",
            4 : "table_projected_row_header",
            5 : "table_spanning_cell"
        }
        self.metadata, self.header = self.__get_metadata()
        self.matrix = np.zeros(shape = (IDX + 1, IDX + 1))


    def get_box_text_in(self, box):
        result = {}
        for key, value in self.boxes_text.items():
            if is_overlap(value['box'], box, threshold = 0.80):
                result[key] = value
        return result

    def __get_metadata(self):
        xmins = []
        ymins = []
        for box, label in self.boxes_element:
            if label in ['table_row', 'table_column']:
                xmins.append(int(box.xmin))
                ymins.append(int(box.ymin))
                xmins.append(int(box.xmax))
                ymins.append(int(box.ymax))

        cols = { x : idx for idx, x in enumerate(sorted(list(set(xmins))))}
        rows = { y : idx for idx, y in enumerate(sorted(list(set(ymins))))}

        cells = []
        for box_row, label_row in self.boxes_element:
            if label_row == "table_row":
                for box_col, label_col in self.boxes_element:
                    if label_col == "table_column":
                        cells.append(Box([box_col.xmin, box_row.ymin, box_col.xmax, box_row.ymax]))
        idx = 0
        metadata = []
        header = []
        text_id = 0
        for cell in cells:
            flag = True
            for box, label in self.boxes_element:
                if label in ['table_projected_row_header', 'table_spanning_cell']:
                    if is_overlap(box, cell, threshold = 0.85):
                        flag = False
                        break
            if flag:
                re = [rows[int(cell.ymin)], rows[int(cell.ymax)], cols[int(cell.xmin)], cols[int(cell.xmax)]]
                texts = self.get_box_text_in(cell)
                obj = Cell(re, [cell.xmin, cell.ymin, cell.xmax, cell.ymax], texts, idx, 'cell', text_id)
                metadata.append(obj)
                if len(obj.TBox.relative_id) > 0:
                    text_id = max(obj.TBox.relative_id)
                elif obj.TBox.id is not None:
                    text_id = obj.TBox.id
                idx += 1
        
        for box, label in self.boxes_element:
            if label in ['table_projected_row_header', 'table_spanning_cell']:
                re = [rows[int(box.ymin)], rows[int(box.ymax)], cols[int(box.xmin)], cols[int(box.xmax)]]
                texts = self.get_box_text_in(box)
                obj = Cell(re, [box.xmin, box.ymin, box.xmax, box.ymax], texts, idx, label, text_id)
                metadata.append(obj)
                if len(obj.TBox.relative_id) > 0:
                    text_id = max(obj.TBox.relative_id)
                elif obj.TBox.id is not None:
                    text_id = obj.TBox.id
            
            if label in ['table_column_header']:
                re = [rows[int(box.ymin)], rows[int(box.ymax)], cols[int(box.xmin)], cols[int(box.xmax)]]
                header.append(RelativeBox((re)))
            
            idx += 1
        # print(len(metadata))
        return metadata, header


    def get_id_header(self):
        list_header = []
        if len(self.header) == 0:
            return list_header
        elif len(self.header) == 1:
            num_lines = self.header[0].row_end - self.header[0].row_start
            if num_lines == 1:
                for cell in self.metadata:
                    if (cell.RBox.row_start == self.header[0].row_start) and (cell.RBox.row_end == self.header[0].row_end) and cell.type == 'cell' and cell.TBox.id is not None:
                        list_header.append(cell)
                return list_header
            elif num_lines == 2:
                s = 0
                selected = []
                for cell in self.metadata:
                    if cell.type == 'table_spanning_cell' and cell.RBox.row_start == self.header[0].row_start:
                        s += cell.RBox.num_grid
                        selected.append(cell)
                if s == self.header[0].num_grid:
                    for cell in selected:
                        if cell.TBox.id is not None:
                            list_header.append(cell)
                else:
                    for cell in selected:
                        if cell.RBox.num_row == self.header[0].num_row and cell.TBox.id is not None:
                            list_header.append(cell)
                        else:
                            for c in self.metadata:
                                if c.type == 'cell' and c.RBox.row_start == cell.RBox.row_end and c.RBox.col_start >= cell.RBox.col_start and c.RBox.col_end <= cell.RBox.col_end and c.TBox.id is not None:
                                    list_header.append(c)
                                    self.matrix[c.TBox.id][cell.TBox.id] = 1
                return list_header
  
            else:
                raise "Only supported two-one line header"
        else:
            raise "Only supported one header"

    def create_link_in_row(self):
        row_end_header = 0
        if len(self.header) == 1:
            row_end_header = self.header[0].row_end
        for cell_1 in self.metadata:
            if cell_1.type == 'cell' and cell_1.RBox.row_end > row_end_header:
            # if cell_1.type == 'cell':
                for cell_2 in self.metadata:
                    if cell_2.type == 'cell' and cell_1.RBox.row_start == cell_2.RBox.row_start and cell_1.RBox.row_end == cell_2.RBox.row_end:
                        if cell_1.TBox.id is not None and cell_2.TBox.id is not None and cell_1.TBox.id != cell_2.TBox.id:
                            self.matrix[cell_1.TBox.id][cell_2.TBox.id] = 1
                            self.matrix[cell_2.TBox.id][cell_1.TBox.id] = 1

    
    def create_link_cell_with_header(self):
        lst_header = self.get_id_header()
        if len(lst_header) > 0:
            for header in lst_header:
                for cell in self.metadata:
                    if cell.type == 'cell' and cell.RBox.col_start == header.RBox.col_start and cell.RBox.col_end == header.RBox.col_end:
                        if cell.TBox.id is not None:
                            self.matrix[cell.TBox.id][header.TBox.id] = 1

    
    def create_link_cell_to_project(self):
        lst_projected = []
        for cell in self.metadata:
            if cell.type == 'table_projected_row_header':
                lst_projected.append(cell)
        if len(lst_projected):
            lst_projected = sorted(lst_projected, key=lambda x:x.RBox.row_start)
            if len(lst_projected) == 1:
                for cell in self.metadata:
                    if cell.type == 'cell' and cell.RBox.col_start == lst_projected[0].RBox.col_start and cell.RBox.row_start >= lst_projected[0].RBox.row_end and cell.RBox.col_end == 1:
                        if cell.TBox.id is not None:
                            self.matrix[cell.TBox.id][lst_projected[0].TBox.id] = 1
            else:
                for idx, projected in enumerate(lst_projected[:-1]):
                    start = projected.RBox.row_end
                    end = lst_projected[idx + 1].RBox.row_start
                    for cell in self.metadata:
                        if cell.type == 'cell' and cell.RBox.col_start == projected.RBox.col_start and cell.RBox.row_start >= start and cell.RBox.row_end > end and cell.RBox.col_end == 1:
                            if cell.TBox.id is not None:
                                self.matrix[cell.TBox.id][projected.TBox.id] = 1

                for cell in self.metadata:
                    if cell.type == 'cell' and cell.RBox.col_start == lst_projected[-1].RBox.col_start and cell.RBox.row_start >= lst_projected[-1].RBox.row_end and cell.RBox.col_end == 1:
                        if cell.TBox.id is not None:
                            self.matrix[cell.TBox.id][lst_projected[-1].TBox.id] = 1

    def create_link_in_cell(self):
        for cell in self.metadata:
            if len(cell.TBox.lines) >= 2:
                for i in cell.TBox.relative_id:
                    self.matrix[cell.TBox.id][i] = 1
    
    def create_link(self):
        self.create_link_in_row()
        self.create_link_in_cell()
        self.create_link_cell_with_header()
        self.create_link_cell_to_project()

def gen_annotations(table):
    documents = []
    h, w = np.shape(table.matrix)[:2]
    boxes_ocr = []
    for item in table.metadata:
        for i in item.TBox.lines.values():
            boxes_ocr.append(
                {
                    'box' : Box([i['box'].xmin, i['box'].ymin, i['box'].xmax, i['box'].ymax]),
                    'text' : i['text'],
                    'id' : i['id']
                }
            )
    # print(boxes_ocr)
    # 1/0
    for ocr in boxes_ocr:
        temp = {
            "box" : [ocr['box'].xmin, ocr['box'].ymin, ocr['box'].xmax, ocr['box'].ymax],
            "text" : ocr['text'],
            "word" : [],
            "id" : ocr['id'],
            "linking" : []
        }
        for i in range(w):
            if table.matrix[ocr['id']][i] == 1:
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
    # print(name)
    global IDX
    IDX = 0
    image_path = os.path.join(ROOT_IMG, name + ".jpg")
    label_path = os.path.join(ROOT_LABEL, name + ".txt")
    ocr_path = os.path.join(ROOT_OCR, name + "_words.json")

    image = cv2.imread(image_path)
    boxes_text = read_ocr(ocr_path)
    boxes_element = read_file(label_path)
    table = Table(boxes_text, boxes_element)
    table.create_link()
    anno = gen_annotations(table)
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
            # 1/0
            except Exception as err:
                wr.write(f"{name}\t{err}")
                print(err)
        # 1/0
    with open("pubtable1m_entity_linking_v1.json", 'w') as f:
        json.dump(annotations, f, ensure_ascii=False)
    




