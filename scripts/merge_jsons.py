
import os
import json
from collections import OrderedDict
from typing import Dict, Tuple


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


def merge_jsons_with_same_classes(new_json_dir: Tuple, *args):
    # categories 가 '동일한' json 여러 개를 합치는 메서드
    # img_id 와 ann_id 는 새로 부여

    confirm_len = dict()
    categories, info = [], {}
    new_images, new_annotations = [], []
    img_id, ann_id = 0, 0

    for idx, json_dir in enumerate(args):
        json_dict = read_json(json_dir)
        images, annotations, categories = \
            json_dict['images'], json_dict['annotations'], json_dict['categories']
        confirm_len[idx] = {'images': len(images), 'annotations': len(annotations)}

        img_id_mapping_dict = dict()
        for img_info in images:
            original_img_id = img_info['id']
            img_info['id'] = img_id
            new_images.append(img_info)
            img_id_mapping_dict[original_img_id] = img_id
            img_id += 1

        for ann_info in annotations:
            ann_img_id = ann_info['image_id']
            ann_info['image_id'] = img_id_mapping_dict[ann_img_id]
            ann_info['id'] = ann_id
            new_annotations.append(ann_info)
            ann_id += 1

    # make new-merge json.
    make_json(new_json_dir[0], images=new_images, annotations=new_annotations, categories=categories)

    # confirm-process.
    new_img_len, new_ann_len = len(new_images), len(new_annotations)
    new_json_len = {'images': new_img_len, 'annotations': new_ann_len}
    img_confirm = sum([vals['images'] for vals in confirm_len.values()]) == new_img_len
    ann_confirm = sum([vals['annotations'] for vals in confirm_len.values()]) == new_ann_len
    print(f'original_jsons: {confirm_len} → new_json: {new_json_len}\n'
          f'>> same? images-{img_confirm}, annotations-{ann_confirm}')


def merge_jsons_with_diff_classes(new_json_dir: Tuple, *args):
    # categories 가 '다른' json 여러 개를 합치는 메서드
    # img_id, ann_id, category_id 전부 새로 부여
    # 각 json의 categories 가 전부 다르다는 가정 하에 진행
    # ex. json_1 에는 a,b,c,d 클래스 + json_2 에는 e,f,g 클래스 > 새로운 json 에는 a,b,c,d,e,f,g 클래스

    confirm_len = dict()
    new_images, new_annotations, new_categories = [], [], []
    cat_id, img_id, ann_id = 0, 0, 0

    for idx, json_dir in enumerate(args):
        json_dict = read_json(json_dir)
        images, annotations, categories = \
            json_dict['images'], json_dict['annotations'], json_dict['categories']
        confirm_len[idx] = {'images': len(images), 'annotations': len(annotations)}

        cat_id_mapping_dict = dict()
        for cat_info in categories:
            original_cat_id = cat_info['id']
            cat_info['id'] = cat_id
            cat_id_mapping_dict[original_cat_id] = cat_id
            new_categories.append(cat_info)
            cat_id += 1

        img_id_mapping_dict = dict()
        for img_info in images:
            original_img_id = img_info['id']
            img_info['id'] = img_id
            new_images.append(img_info)
            img_id_mapping_dict[original_img_id] = img_id
            img_id += 1

        for ann_info in annotations:
            ann_img_id = ann_info['image_id']
            ann_class_id = ann_info['category_id']
            ann_info['image_id'] = img_id_mapping_dict[ann_img_id]
            ann_info['id'] = ann_id
            ann_info['category_id'] = cat_id_mapping_dict[ann_class_id]
            new_annotations.append(ann_info)
            ann_id += 1

    # make new-merge json.
    make_json(new_json_dir[0], images=new_images, annotations=new_annotations, categories=new_categories)

    # confirm-process.
    new_img_len, new_ann_len = len(new_images), len(new_annotations)
    new_json_len = {'images': new_img_len, 'annotations': new_ann_len}
    img_confirm = sum([vals['images'] for vals in confirm_len.values()]) == new_img_len
    ann_confirm = sum([vals['annotations'] for vals in confirm_len.values()]) == new_ann_len
    print(f'original_jsons: {confirm_len} → new_json: {new_json_len}\n'
          f'>> same? images-{img_confirm}, annotations-{ann_confirm}')


if __name__ == '__main__':
    new_json_dir = '/mnt/dms/mealy_store/3_run_data/14th_run/annotations/train.json',
    json_one = '/mnt/dms/mealy_store/2_augmented_data/14th_aug/train/augmentation.json'
    json_two = '/mnt/dms/mealy_store/1_splited_data/14th_splited/annotation/train.json'

    merge_jsons_with_same_classes(new_json_dir, json_one, json_two)
    # merge_jsons_with_diff_classes(new_json_dir, json_one, json_two)
