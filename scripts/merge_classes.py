
import os
import json
from collections import OrderedDict, defaultdict
from typing import Dict, List


def read_json(json_file: str) -> Dict:
    with open(json_file, 'r') as tmp_json:
        json_dict = json.load(tmp_json)
        return json_dict


def make_json(new_json_dir: str, **kwargs):
    option = kwargs
    file_data = OrderedDict()
    for key, values in option.items():
        file_data[key] = values

    with open(new_json_dir, 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent='\t')


def merge(json_dir: str, json_save_dir: str, merge_classes: List):
    # merge_classes 리스트에 적힌 클래스들을 하나로 합쳐주는 코드
    # category_id 는 전부 새로 부여
    # ex. merge_classes = [0, 52] 이면 0번과 52번 클래스를 합쳐서 1개의 클래스로 만듦

    json_dict = read_json(json_dir)
    annotations, categories = json_dict['annotations'], json_dict['categories']

    new_annotations, new_categories = [], []
    class_mapping_dict = dict()
    cat_id = 0
    merge_id = None
    for cat_info in categories:
        class_id = cat_info['id']
        if class_id in merge_classes:
            if merge_id is None:
                merge_id = cat_id
                cat_id += 1
                cat_info['id'] = merge_id
                new_categories.append(cat_info)
            class_mapping_dict[class_id] = merge_id
        else:
            class_mapping_dict[class_id] = cat_id
            cat_info['id'] = cat_id
            cat_id += 1
            new_categories.append(cat_info)

    for ann_info in annotations:
        class_id = ann_info['category_id']
        ann_info['category_id'] = class_mapping_dict[class_id]
        new_annotations.append(ann_info)

    make_json(json_save_dir, images=json_dict['images'], annotations=new_annotations, categories=new_categories)


if __name__ == '__main__':
    json_dir = '/mnt/dms/mealy_store/1_inspected_data/13th_inspected/13_inspected_add_3.json'
    json_save_dir = '/mnt/dms/mealy_store/1_inspected_data/14th_inspected/from_13_inspected_58_classes.json'
    merge_classes = [0, 52]

    merge(
        json_dir=json_dir,
        json_save_dir=json_save_dir,
        merge_classes=merge_classes
    )
