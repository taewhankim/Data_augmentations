
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


def pick_up(json_dir: str, new_json_dir: str, used_classes_id: List):
    # 기존 json 파일에서 사용할 category_id 에 대한 정보만 골라서 추출 > 새로운 json 생성
    # 사용하지 않는 category_id 가 포함된 이미지와 그 이미지의 annotation 정보는 전부 삭제

    json_dict = read_json(json_dir)
    images, annotations, categories = json_dict['images'], json_dict['annotations'], json_dict['categories']

    new_images, new_annotations, new_categories = [], [], []
    image_id, annotation_id, category_id = 0, 0, 0

    # pick-categories.
    cat_mapping_dict = dict()
    for cat_info in categories:
        class_id = cat_info['id']
        if class_id in used_classes_id:
            cat_mapping_dict[class_id] = category_id
            cat_info['id'] = category_id
            new_categories.append(cat_info)
            category_id += 1

    # pick-images & annotations.
    image_dict = dict()
    image_used_or_not = dict()
    for img_info in images:
        img_id = img_info['id']
        image_dict[img_id] = img_info
        image_used_or_not[img_id] = True

    image_ann_dict = defaultdict(list)
    for ann_info in annotations:
        class_id = ann_info['category_id']
        img_id = ann_info['image_id']
        if class_id not in used_classes_id:
            image_used_or_not[img_id] = False
        image_ann_dict[img_id].append(ann_info)

    for img_id, ann_list in image_ann_dict.items():
        used_tf = image_used_or_not[img_id]
        if not used_tf:
            continue
        else:
            img_info = image_dict[img_id]
            img_info['id'] = image_id
            new_images.append(img_info)
            for ann_info in ann_list:
                ann_info['image_id'] = image_id
                ann_info['id'] = annotation_id
                ann_info['category_id'] = cat_mapping_dict[ann_info['category_id']]
                new_annotations.append(ann_info)
                annotation_id += 1
            image_id += 1

    # create new-json.
    make_json(new_json_dir, images=new_images, annotations=new_annotations, categories=new_categories)


if __name__ == '__main__':
    json_dir = '/mnt/dms/mealy_store/1_inspected_data/13th_inspected/13_inspected_add.json'
    new_json_dir = '/mnt/dms/mealy_store/1_inspected_data/14th_inspected/from_13_inspected_3.json'
    used_classes_id = [
        58, 87, 3, 51, 78, 79, 89, 90, 93, 94, 91, 92, 80, 81, 82, 83, 84, 85, 70, 71, 59, 60, 61, 62, 26, 27, 30,
        31, 32, 40, 41, 42, 43, 44, 68, 69, 72, 73, 74, 34, 35, 36, 17, 18, 86, 66, 67, 46, 47, 48, 77, 11, 64, 45,
        33, 28, 63, 29, 76
    ]

    pick_up(
        json_dir=json_dir,
        new_json_dir=new_json_dir,
        used_classes_id=used_classes_id
    )
