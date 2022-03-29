from typing import *
from utils import is_overlap, with_line

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
    
    def get_box_text_in(self, box):
        result = {}
        for key, value in self.boxes_text.keys():
            if is_overlap(value['bbox'], box, threshold = 0.9):
                result[key] = value
        return result

    def get_multiline(self, dict_box : Dict):
        result = {}
        idx = 0
        dict_box = {k : v for k, v in sorted(dict_box.items(), key = lambda  x : x[1]['bbox'].xmin)}
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

    def get_box_cell(self, metadata):
        metadata['cell'] = []
        idx = 0
        for col in metadata['table_column']:
            box_col = col['box_item']
            for row in metadata['table_row']:
                box_row = row['box_item']
                box_cell =Box([box_col.xmin, box_row.ymin, box_col.xmax, box_row.ymax])
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
            list_cells = row['list_cells']
            for cell in list_cells:
                lines = cell['lines']
                list_id.append(lines['line-0'][0]['id'])
    
