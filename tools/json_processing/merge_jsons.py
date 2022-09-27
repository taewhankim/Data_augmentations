
# 1. (category name 이 동일해서 mapping 가능한) json 여러 개를 합치는 코드 > 합치면서 img_id, ann_id, cat_id 전부 새로 부여
# 2. 합치면서 물리적인 파일명도 함께 수정(그러므로 중복된 파일명도 ok) > ex. f'{cat_id}_{cat_name}_{idx}.jpg'
#   → 한 이미지에 여러 카테고리가 있는 경우, 오브젝트 수가 적은 클래스명으로 적음
# 3. 만약 파일명에서 빼고 싶은 카테고리가 있다면 out_from_name list 에 category name 으로 적으면 됨

import os
import shutil
import datetime
from collections import defaultdict
from tqdm import tqdm
from tools.json_processing.utils import read_json, make_dir, make_json


class MergeJson:
    def __init__(self, config: dict):
        one_json_dir = config['one_json_dir']
        one_images_dir = config['one_img_dir']
        two_json_dir = config['two_json_dir']
        two_images_dir = config['two_img_dir']
        new_json_dir = config['new_json_dir']
        self.new_images_dir = config['new_img_dir']

        make_dir(self.new_images_dir)
        self.original_json_info = {}
        _, _, _, info = read_json(one_json_dir)
        self.new_categories, self.cat_mapping_dict, self.new_categories_name, self.new_categories_id = self.set_categories(one=one_json_dir, two=two_json_dir)
        self.new_images, self.img_mapping_dict, self.img_address_dict, self.img_original_dict, self.img_cat_dict = self.set_images(one=one_json_dir, two=two_json_dir)
        self.new_annotations, self.cat_count_dict = self.set_annotations(one=one_json_dir, two=two_json_dir)

        self.out_categories = config['out_from_name'] if 'out_from_name' in config else []
        self.change_images(one=one_images_dir, two=two_images_dir)
        make_json(new_json_dir, images=self.new_images, annotations=self.new_annotations, categories=self.new_categories, info=info)
        print(f'original json - {self.original_json_info}')
        print(f'images: {len(self.new_images)}, annotations: {len(self.new_annotations)}, categories: {len(self.new_categories)}')

    # 1. category name 으로 new_categories 생성
    # cat_mapping_dict : (key) - json_name / (value) - {(key) original_category_id (value) new_category_id}
    # new_categories_name : (key) category_name / (value) category_id
    # new_categories_id : (key) category_id / (value) category_name
    def set_categories(self, **kwargs):
        new_categories = []
        cat_mapping_dict, new_categories_name, new_categories_id = {}, {}, {}

        cat_id = 0
        for json_name, json_dir in kwargs.items():
            _, _, categories, _ = read_json(json_dir)

            tmp_mapping_dict = {}
            for cat in categories:
                cat_name = cat['name']
                if cat_name not in new_categories_name:
                    new_categories_name[cat_name] = cat_id
                    new_categories_id[cat_id] = cat_name

                    tmp_mapping_dict[cat['id']] = cat_id
                    cat['id'] = cat_id
                    new_categories.append(cat)
                    cat_id += 1
                else:
                    new_cat_id = new_categories_name[cat_name]
                    tmp_mapping_dict[cat['id']] = new_cat_id

            cat_mapping_dict[json_name] = tmp_mapping_dict

        return new_categories, cat_mapping_dict, new_categories_name, new_categories_id

    # 2. annotations 를 합칠 때 필요한 초석을 다지기 위해 images 먼저 setting
    # img_mapping_dict : (key) - json_name / (value) - {(key) original_img_id (value) new_img_id}
    # img_address_dict : (key) img_name or tmp_name / (value) json_name
    # img_original_name : (key) tmp_name / (value) original_img_name
    # img_cat_dict : (key) new_img_id / (value) defaultdict(int)
    def set_images(self, **kwargs):
        new_images = []
        img_mapping_dict, img_address_dict, img_original_name = {}, {}, {}
        img_cat_dict = {}

        img_id = 0
        for json_name, json_dir in kwargs.items():
            images, annotations, _, _ = read_json(json_dir)
            self.original_json_info[json_name] = dict(images=len(images), annotations=len(annotations))

            tmp_mapping_dict = {}
            for img in images:
                tmp_img_id = img['id']
                img['id'] = img_id
                tmp_mapping_dict[tmp_img_id] = img_id
                img_cat_dict[img_id] = defaultdict(int)

                if img['file_name'] in img_address_dict:
                    tmp_name = img['file_name'] + str(datetime.datetime.now().strftime("%H%M%S%f"))
                    img_original_name[tmp_name] = img['file_name']
                    img['file_name'] = tmp_name
                img_address_dict[img['file_name']] = json_name
                new_images.append(img)
                img_id += 1

            img_mapping_dict[json_name] = tmp_mapping_dict

        return new_images, img_mapping_dict, img_address_dict, img_original_name, img_cat_dict

    # 3. 새로운 img_id 와 cat_id 를 부여하여 annotations 을 하나로 합치는 메서드
    # cat_count_dict : (key) category_id / (value) count
    def set_annotations(self, **kwargs):
        new_annotations = []
        cat_count_dict, img_cat_dict = defaultdict(int), {}

        ann_id = 0
        for json_name, json_dir in kwargs.items():
            _, annotations, _, _ = read_json(json_dir)

            cat_mapping_dict = self.cat_mapping_dict[json_name]
            img_mapping_dict = self.img_mapping_dict[json_name]
            for ann in annotations:
                new_cat_id = cat_mapping_dict[ann['category_id']]
                new_img_id = img_mapping_dict[ann['image_id']]

                ann['category_id'] = new_cat_id
                ann['image_id'] = new_img_id
                ann['id'] = ann_id
                cat_count_dict[new_cat_id] += 1
                self.img_cat_dict[new_img_id][new_cat_id] += 1
                new_annotations.append(ann)
                ann_id += 1

        return new_annotations, cat_count_dict

    # 4. 물리적 파일명을 바꾸고 > 한 폴더에 합치고 > new_images 생성
    def change_images(self, **kwargs):
        cat_idx = defaultdict(int)

        for img in tqdm(self.new_images):
            tmp_img_name = img['file_name']
            extension = tmp_img_name.split('.')[-1]
            img_address = self.img_address_dict[tmp_img_name]
            if tmp_img_name in self.img_original_dict:
                tmp_img_name = self.img_original_dict[tmp_img_name]
            ori_dir = os.path.join(kwargs[img_address], tmp_img_name)
            tmp_img_cat_dict = self.img_cat_dict[img['id']]

            keep_key, keep_val = -1, -1
            for cat_id, cnt in tmp_img_cat_dict.items():
                if self.new_categories_id[cat_id] in self.out_categories:
                    continue
                else:
                    if cnt > keep_val:
                        keep_key = cat_id
                        keep_val = cnt
                    elif cnt == keep_val:
                        keep_cnt = self.cat_count_dict[keep_key]
                        curr_cnt = self.cat_count_dict[cat_id]
                        # 오브젝트수가 많은 것을 기준으로 이름을 부여하고 싶다면
                        # 부등호만 반대로 수정
                        if keep_cnt > curr_cnt:
                            keep_key = cat_id
                            keep_val = cnt
                    else:
                        continue

            cat_name = self.new_categories_id[keep_key]
            idx = cat_idx[keep_key]
            new_file_name = f'{keep_key}_{cat_name}_{idx}.{extension}'
            img['file_name'] = new_file_name
            new_img_dir = os.path.join(self.new_images_dir, new_file_name)
            shutil.copy(ori_dir, new_img_dir)
            cat_idx[keep_key] += 1


if __name__ == '__main__':
    configs = {
        'one_json_dir': '',
        'one_img_dir': '',
        'two_json_dir': '',
        'two_img_dir': '',

        'new_json_dir': '',
        'new_img_dir': '',
        'out_from_name': []
    }

    MergeJson(configs)
