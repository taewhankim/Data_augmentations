
import os
import json
import cv2
import shutil
from tqdm import tqdm
from collections import OrderedDict
import logging.handlers
from collections import defaultdict


def make_dir(*args):
    for dir in args:
        if not os.path.isdir(dir):
            os.mkdir(dir)
            print('make dir ^-^!')


def read_json(json_file):
    with open(json_file, 'r') as tmp_json:
        tmp = json.load(tmp_json)
        images = tmp['images']
        annotations = tmp['annotations']
        categories = tmp['categories']
        info = tmp['info']
        return images, annotations, categories, info


def make_json(new_json_dir, **kwargs):
    option = kwargs
    file_data = OrderedDict()
    for key, values in option.items():
        file_data[key] = values

    with open(new_json_dir, 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent='\t')


def set_log(logger_name, file_dir, handler_ok=True):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    if handler_ok:
        SH = logging.StreamHandler()
        logger.addHandler(SH)
    FH = logging.FileHandler(os.path.join(file_dir))
    logger.addHandler(FH)
    return logger


# mycrowd 산출물인 coco format json 의 images - file_name 에 있는 경로명을 지우는 코드
# ex. /mnt/aistudionas/dms/GIST/3_run_data/test/sample.jpg → sample.jpg
def remove_file_dir(json_dir, new_json_dir):
    images, annotations, categories, info = read_json(json_dir)

    new_images = []
    for img in images:
        file_name = img['file_name']
        new_name = file_name.split('/')[-1]
        img['file_name'] = new_name
        new_images.append(img)
    make_json(new_json_dir, images=new_images, annotations=annotations, categories=categories, info=info)


# 폴더 안에 있는 이미지에 대하여 가짜 annotations 정보를 만들어주는 코드
# cat_info 는 길이 2의 [cat_id, cat_name] 으로 구성된 list
def create_empty_annotation_info(img_dir, cat_info: list, new_json_dir):
    img_list = os.listdir(img_dir)
    new_images, new_annotations = [], []
    img_id, ann_id = 0, 0
    cat_id, cat_name = cat_info[0], cat_info[1]

    for img in img_list:
        img_path = os.path.join(img_dir, img)
        cv_img = cv2.imread(img_path)
        try:
            height, width, _ = cv_img.shape
            tmp_img_dict = {'file_name': img,
                            'width': width,
                            'id': img_id,
                            'height': height}
            new_images.append(tmp_img_dict)

            tmp_ann_dict = {'area': width * height,
                            'category_id': cat_id,
                            'iscrowd': 0,
                            'bbox': [0, 0, width-1, height-1],
                            'segmentation': [0, 0, width-1, height-1],
                            'id': ann_id,
                            'image_id': img_id,
                            'score_all': []}
            new_annotations.append(tmp_ann_dict)

            img_id += 1
            ann_id += 1
        except:
            print(f'wrong image file : {img}')

    new_categories = [{'name': cat_name, 'supercategory': "", 'id': cat_name}]
    info = {"contributor": "AISTUDIO", "year": 120, "date_created": "Mon Nov 02 2020 14:29:23 +0900",
                   "description": "made by AIStudio ATOM DMS", "version": 1, "url": "https://www.aistudio.co.kr"}
    make_json(new_json_dir, images=new_images, annotations=new_annotations, categories=new_categories, info=info)


# json - images 에 있는 파일을 새로운 폴더에 복사하는 메서드
def move_img_by_json(json_dir, img_folder, new_img_folder):
    images, _, _, _ = read_json(json_dir)
    make_dir(new_img_folder)

    for img in tqdm(images):
        file_name = img['file_name']
        ori_file_dir = os.path.join(img_folder, file_name)
        new_file_dir = os.path.join(new_img_folder, file_name)
        shutil.copy(ori_file_dir, new_file_dir)

    file_len = len(os.listdir(new_img_folder))
    print(f'json_img_len: {len(images)}, new_folder_len: {file_len} >> same? {len(images)==file_len}')


# categories id 를 변경하는 메서드
# cat_mapping_dict 에 {원래 id : 바꾸고 싶은 id} 를 넣어야 한다.
def change_categories_id(cat_mapping_dict, json_dir, new_json_dir):
    images, annotations, categories, info = read_json(json_dir)

    for ann in annotations:
        ori_cat_id = ann['category_id']
        ann['category_id'] = cat_mapping_dict[ori_cat_id]
    for cat in categories:
        ori_cat_id = cat['id']
        cat['id'] = cat_mapping_dict[ori_cat_id]

    make_json(new_json_dir, images=images, annotations=annotations, categories=categories, info=info)


# annotations 정보가 없는 images 를 json 에서 제거
def remove_no_info_images(json_dir, new_json_dir):
    images, annotations, categories, info = read_json(json_dir)

    img_list = []
    for ann in annotations:
        img_list.append(ann['image_id'])

    img_list = list(set(img_list))
    new_images = []
    for img in images:
        if img['id'] in img_list:
            new_images.append(img)

    make_json(new_json_dir, images=new_images, annotations=annotations, categories=categories, info=info)


# (categories 가 동일한) json 여러 개를 합치는 메서드
# img_id 와 ann_id 는 새로 부여
def merge_jsons(new_json_dir, **kwargs):
    confirm_len = {}
    new_images, new_annotations = [], []
    img_id, ann_id = 0, 0

    for idx, json in kwargs.items():
        images, annotations, categories, info = read_json(json)
        confirm_len[idx] = {'images': len(images), 'annotations': len(annotations)}

        img_mapping_dict = {}
        for img in images:
            tmp_id = img['id']
            img['id'] = img_id
            img_mapping_dict[tmp_id] = img_id
            new_images.append(img)
            img_id += 1
        for ann in annotations:
            tmp_id = ann['image_id']
            ann['image_id'] = img_mapping_dict[tmp_id]
            ann['id'] = ann_id
            new_annotations.append(ann)
            ann_id += 1

    make_json(new_json_dir, images=new_images, annotations=new_annotations, categories=categories, info=info)

    new_img_len, new_ann_len = len(new_images), len(new_annotations)
    new_json_len = {'images': new_img_len, 'annotations': new_ann_len}
    img_confirm = sum([vals['images'] for vals in confirm_len.values()]) == new_img_len
    ann_confirm = sum([vals['annotations'] for vals in confirm_len.values()]) == new_ann_len
    print(f'original_jsons: {confirm_len} → new_json: {new_json_len}\n'
          f'>> same? images-{img_confirm}, annotations-{ann_confirm}')


# json 에서 원하는 카테고리를 del_cat_id_list 에 담아서 삭제하고 새로운 json 생성
# 카테고리에서 제외 후, images - annotations - categories 에 각각 id 를 0번부터 다시 부여
def minus_categories_json(json_dir, new_json_dir, del_cat_id_list: list):
    images, annotations, categories, info = read_json(json_dir)
    new_images, new_annotations, new_categories = [], [], []
    img_id, ann_id, cat_id = 0, 0, 0
    img_mapping_dict, cat_mapping_dict = {}, {}
    keep_img_list = []
    print(f'rest_categories_num : {len(categories)} - {len(del_cat_id_list)} '
          f'= {len(categories) - len(del_cat_id_list)}')
    print(f'images : {len(images)}, annotations : {len(annotations)}, categories : {len(categories)}')

    # 1. categories 정렬
    for cat in categories:
        ori_cat_id = cat['id']
        if ori_cat_id in del_cat_id_list:
            continue
        cat_mapping_dict[ori_cat_id] = cat_id
        cat['id'] = cat_id
        new_categories.append(cat)
        cat_id += 1

    # 2. annotations 정렬 - 1
    for ann in annotations:
        ori_cat_id = ann['category_id']
        if ori_cat_id in del_cat_id_list:
            continue
        ann['id'] = ann_id
        ann['category_id'] = cat_mapping_dict[ori_cat_id]
        keep_img_list.append(ann['image_id'])
        new_annotations.append(ann)
        ann_id += 1

    # 3. images 정렬
    for img in images:
        ori_img_id = img['id']
        if ori_img_id in keep_img_list:
            img_mapping_dict[ori_img_id] = img_id
            img['id'] = img_id
            new_images.append(img)
            img_id += 1

    # 4. annotations 정렬 - 2
    for new_ann in new_annotations:
        ori_img_id = new_ann['image_id']
        new_ann['image_id'] = img_mapping_dict[ori_img_id]

    make_json(new_json_dir, images=new_images, annotations=new_annotations, categories=new_categories, info=info)
    print(f'new_images : {len(new_images)}, new_annotations : {len(new_annotations)}, '
          f'new_categories : {len(new_categories)}')


# 클래스별 오브젝트 수량이 담긴 딕셔너리를 return
def count_categories(json_dir):
    cat_cnt_dict = defaultdict(list)
    images, annotations, categories, _ = read_json(json_dir)
    cat_mapping_dict = {cat['id']: cat['name'] for cat in categories}

    for ann in annotations:
        try:
            cat_cnt_dict[ann['category_id']].append(ann['image_id'])
        except:
            pass

    results = {}
    for cat_id, cat in cat_cnt_dict.items():
        results[f'{cat_id}_{cat_mapping_dict[cat_id]}'] = len(set(cat))

    print(len(images))
    return results


# segmentation 을 임시로 넣어주는 메서드
# before : [[]] → after : [bbox]
def put_segmentation_info(json_dir, new_json_dir):
    new_annotations = []
    images, annotations, categories, info = read_json(json_dir)
    for ann in annotations:
        bbox = ann['bbox']
        ann['segmentation'] = [bbox]
        new_annotations.append(ann)
    make_json(new_json_dir, images=images, annotations=new_annotations, categories=categories, info=info)


# size 의 단위는 bytes
def get_dir_size(path='.'):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def get_size(path='.'):
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        return get_dir_size(path)


def make_testonly(json_dir, new_json_dir):
    _, _, categories, info = read_json(json_dir)
    make_json(new_json_dir, images=[], categories=categories, info=info)
