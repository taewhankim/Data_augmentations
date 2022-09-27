
# 한 개의 json 에서 원하는 오브젝트 수 만큼의 정보만 뽑아서 새로운 json 을 만들고 싶을 때 사용하는 코드
# (결과물) 오브젝트 수에 맞게 뽑은 json + 그 외의 정보들이 담긴 others json
# > 만약 우선적으로 고려해야 할 클래스가 있다면 config 의 small_categories 에 {class_name: need_count} 로 적음

from collections import defaultdict
import random
import os
from tools.json_processing.utils import read_json, make_json, count_categories


class ExportJsons:
    def __init__(self, config: dict):
        main_dir = config['main_dir']
        json_dir = os.path.join(main_dir, config['json_dir'])
        self.new_json_dir = os.path.join(main_dir, config['new_json_dir'])
        self.other_json_dir = os.path.join(main_dir, config['other_json_dir'])

        self.images, self.annotations, self.categories, self.info = read_json(json_dir)
        self.min_standard_count = config['min_standard_count']
        self.small_categories = config['small_categories']

        self.original_cat_cnt_dict = defaultdict(int)
        self.cat_cnt_dict = defaultdict(int)

    # 1. img_mapping_dict : (key) img_id / (value) image information
    def img_mapping(self):
        img_mapping_dict = {}
        for img in self.images:
            img_id = img['id']
            img_mapping_dict[img_id] = img
        return img_mapping_dict

    # 1. img_ann_dict : (key) img_id / (value) [ann, ann, ann ... ]
    # 2. small_categories_id : [img_id, img_id ... ]
    # 3. others_categories_id : [img_id, img_id ... ]
    def ann_mapping(self):
        img_ann_dict = defaultdict(list)
        small_categories_id = []

        random.shuffle(self.annotations)
        for ann in self.annotations:
            cat_id = ann['category_id']
            img_id = ann['image_id']
            self.original_cat_cnt_dict[cat_id] += 1
            img_ann_dict[img_id].append(ann)
            if cat_id in self.small_categories:
                small_categories_id.append(img_id)

        small_categories_id = set(small_categories_id)
        all_categories_id = set([key for key in img_ann_dict.keys()])
        others_categories_id = all_categories_id - small_categories_id

        return img_ann_dict, list(small_categories_id), list(others_categories_id)

    # 만약 categories_id == small_categories_id 라면 check_small=True 로 변경
    def split_json(self, img_mapping_dict, img_ann_dict, categories_id: list, check_small=False):
        new_images, new_annotations = [], []
        rest_categories_id = []

        for img_id in categories_id:
            append_True = False
            tmp_list, tmp_cat = [], []
            ann_list = img_ann_dict[img_id]
            img = img_mapping_dict[img_id]

            for ann in ann_list:
                cat_id = ann['category_id']
                tmp_cat.append(cat_id)
                tmp_list.append(ann)

                if check_small:
                    if cat_id in self.small_categories:
                        if self.cat_cnt_dict[cat_id] <= self.small_categories[cat_id]:
                            append_True = True
                else:
                    if self.cat_cnt_dict[cat_id] <= self.min_standard_count:
                        append_True = True

            if append_True:
                new_images.append(img)
                new_annotations += tmp_list
                for cat in tmp_cat:
                    self.cat_cnt_dict[cat] += 1
                append_True = False
            else:
                rest_categories_id.append(img_id)

        return new_images, new_annotations, rest_categories_id

    # 생성된 images, annotations 등들을 하나로 합쳐주는 단순 메서드 (id 변화 없음)
    def merge_lists(self, *args):
        results = []
        for tmp in args:
            results += tmp
        return results

    # 새로운 json 을 생성하고 남은 것들을 모아 others.json 을 만드는 메서드
    def merge_rest_id(self, img_mapping_dict, img_ann_dict, *args):
        others_images, others_annotations = [], []

        for rest_list in args:
            for img_id in rest_list:
                img = img_mapping_dict[img_id]
                others_images.append(img)
                ann_list = img_ann_dict[img_id]
                for ann in ann_list:
                    others_annotations.append(ann)

        return others_images, others_annotations

    # json 을 나누면서 이것저것 확인하고 싶은 것들을 출력하는 메서드
    # log.txt 로 남기지는 않음 > 나중에 필요하면 남길 수 있게 수정 가능
    def confirm_process(self, new_images, new_annotations, others_images, others_annotations):
        new_images_len = len(new_images)
        new_annotations_len = len(new_annotations)
        others_images_len = len(others_images)
        others_annotations_len = len(others_annotations)
        images_tf = len(self.images) == (new_images_len + others_images_len)
        annotations_tf = len(self.annotations) == (new_annotations_len + others_annotations_len)
        others_cat_cnt_dict = count_categories(self.other_json_dir)

        txt = f'original json - {self.original_cat_cnt_dict}\n' \
              f'new_json - {self.cat_cnt_dict}\n' \
              f'other_json - {others_cat_cnt_dict}\n' \
              f'images - {len(self.images)} = {new_images_len}+{others_images_len} == {images_tf}\n' \
              f'annotations - {len(self.annotations)} = {new_annotations_len}+{others_annotations_len} == {annotations_tf}'

        print(txt)

    # 실제 사용할 때, 필요에 따라서 수정해야 할 부분
    # small_categories 를 빈 {} 로 두고 아래 코드를 그대로 돌려도 문제는 없음... (코드가 깔끔하지 않을뿐)
    def main(self):
        img_mapping_dict = self.img_mapping()
        img_ann_dict, small_id, others_id = self.ann_mapping()
        small_images, small_annotations, rest_small_id = self.split_json(img_mapping_dict, img_ann_dict, small_id, True)
        more_images, more_annotations, rest_other_id = self.split_json(img_mapping_dict, img_ann_dict, others_id)

        new_images = self.merge_lists(small_images, more_images)
        new_annotations = self.merge_lists(small_annotations, more_annotations)
        others_images, others_annotations = self.merge_rest_id(img_mapping_dict, img_ann_dict, rest_small_id, rest_other_id)
        make_json(self.new_json_dir, images=new_images, annotations=new_annotations, categories=self.categories, info=self.info)
        make_json(self.other_json_dir, images=others_images, annotations=others_annotations, categories=self.categories, info=self.info)

        self.confirm_process(new_images, new_annotations, others_images, others_annotations)


if __name__ == '__main__':
    configs = {
        'main_dir': '/mnt/aistudionas/dms/NIA/1_inspected_data/1st_inspected_data/annotations/rest',
        'json_dir': 'rest_imgs_after_train_val_test.json',
        'new_json_dir': 'ttttttt.json',
        'other_json_dir': 'others.json',
        'min_standard_count': 200,
        'small_categories': {}
    }

    EJ = ExportJsons(configs)
    EJ.main()
