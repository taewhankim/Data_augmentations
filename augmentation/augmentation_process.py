
from augmentation.preprocess_tool import PreProcessing
from augmentation.create_augmentation_img import CreateAugmentation
from tools.json_processing.utils import make_json, read_json, set_log

import time
import datetime
import os
import random
import cv2
import numpy as np
from pathlib import Path
from beautifultable import BeautifulTable
from beautifultable import BTColumnCollection
from collections import defaultdict


class TryAugmentation(object):
    def __init__(self, cfg, img_cnt_dict, max_dict, cat_cnt_dict, categories_num):
        # max_or_all : 증식할 때, 오브젝트 수를 어떻게 할 것인지를 결정짓는 요소
        # max = max_dict : 과증식되지 않도록 필요없는 오브젝트는 삭제 → 클래스밸런싱이 완벽하게 맞음
        # (defaults) all = cat_cnt_dict : 과증식되더라도 오브젝트 유지 → 클래스밸런싱은 깨질 수 있음
        self.standard_dict = max_dict if cfg.MAX_OR_ALL == 'MAX' else cat_cnt_dict
        self.img_cnt_dict = img_cnt_dict
        self.categories_num = categories_num

        self.root = cfg.ROOT
        self.apply_cnt = cfg.APPLY_CNT
        self.tool_list = cfg.TOOL_NAME_LIST
        self.post_tool_list = cfg.POST_TOOL_NAME_LIST
        self.range_min = cfg.RANGE_MIN
        self.max = True if cfg.MAX_OR_ALL == "MAX" else False
        self.mask_on = cfg.MASK_ON

        self.original_img_path = os.path.join(self.root, cfg.ORIGINAL_IMAGE)
        self.original_json = os.path.join(self.root, cfg.ORIGINAL_JSON)
        self.augmentation_path = os.path.join(cfg.AUGMENTATION_DIR, os.path.basename(cfg.ORIGINAL_IMAGE))
        self.augmentation_json = os.path.join(self.augmentation_path, 'augmentation.json')

        self.images, self.annotations, self.categories, self.info = read_json(self.original_json)
        Path(self.augmentation_path).mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.logger = set_log(str(int(self.start_time)), os.path.join(self.augmentation_path, 'augmentation.log'))
        self.whole_count = 1
        self.total_need_cnt = sum(self.standard_dict.values())
        self.need_img_cnt = sum(self.img_cnt_dict.values())

        # augmentation 진행하면서 값이 변하는 것들
        self.new_images, self.new_annotations = [], []
        self.cat_log = defaultdict(int)
        self.img_info_dict, self.ann_info_dict = self.extract_information()

    def write_basic_information(self):
        set_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        txt = f'start time : {set_time}\n'\
              f'used function : {self.tool_list}\n' \
              f'range min : {self.range_min}\n\n' \
              f'original images directory : {self.original_img_path}\n' \
              f'original json path : {self.original_json}\n' \
              f'augmentation images directory : {self.augmentation_path}\n' \
              f'augmentation json path : {self.augmentation_json}\n\n' \
              f'--- 생성하고자 하는 총 이미지 수 : {self.need_img_cnt} --- \n' \
              f'--- 생성하고자 하는 총 오브젝트 수 : {self.total_need_cnt} --- \n'

        self.logger.debug(txt)

    def write_final_confirm(self):
        folder_length = len(os.listdir(self.augmentation_path))
        n_images, n_annotations, _, _ = read_json(self.augmentation_json)

        table = BeautifulTable()
        table.column_headers = ["id", "name", "original", "need", "augmentation", "result", "total"]

        # table = BTColumnCollection(table=, default_alignment=, default_padding=_)
        # table.headers = ["id", "name", "original", "need", "augmentation", "result", "total"]

        for cat_info in self.categories:
            if cat_info['id'] not in self.standard_dict:
                continue
            cat_id = cat_info['id']
            cat_name = cat_info['name']

            original = self.categories_num[cat_id]
            need = self.standard_dict[cat_id]
            augmentation = self.cat_log[cat_id]
            result = augmentation - need
            total = original + augmentation

            tmp = [cat_id, cat_name, original, need, augmentation, result, total]
            table.append_row(tmp)
            # table.append(tmp)

        finish_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_time = time.time()
        total_augmentation_time = total_time - self.start_time

        txt = f'\n----- final confirm time -----\n' \
              f'+. original : 원래 object 수량\n' \
              f'+. need : 증식 필요 수량(없는 경우는 0)\n' \
              f'+. augmentation : 증식된 수량\n' \
              f'+. result : augmentation - need >> 음수(부족), 0(완벽), 양수(초과)\n' \
              f'+. total : original + augmentation\n' \
              f'{table}\n\n' \
              f'** 목표 증식량 : {self.need_img_cnt}\naugmentation image cnt : {folder_length}\n' \
              f'augmentation json cnt : {len(n_images)}\n' \
              f'** 목표 증식량 : {self.total_need_cnt}\naugmentation object cnt : {sum(self.cat_log.values())}\n' \
              f'augmentation json cnt : {len(n_annotations)}\n\n' \
              f'finish time : {finish_time}\n' \
              f'total augmentation time : {str(datetime.timedelta(seconds=total_augmentation_time))}\n'

        self.logger.debug(txt)
        self.logger.debug(f'*-*-*- yeah! AUGMENTATION finish ^-^! -*-*-*')

    # img - key : img_id / value : file_name, width, height
    # anno - key : file_name / value : annotation information [{}]
    def extract_information(self):
        img_info_dict, ann_info_dict = {}, defaultdict(list)

        for img in self.images:
            if img['id'] in self.img_cnt_dict:
                img_info_dict[img['id']] = [img['file_name'], img['width'], img['height']]
        for ann in self.annotations:
            key = ann['image_id']
            if key in img_info_dict:
                file_key = img_info_dict[key][0]
                ann_info_dict[file_key].append(ann)

        return img_info_dict, ann_info_dict

    def check_rest_categories(self):
        rest_cat_dict = defaultdict(int)
        for cat_id, count in self.cat_log.items():
            need_cnt = self.standard_dict[cat_id]
            if need_cnt > count:
                rest_cat_dict[cat_id] = need_cnt - count

        return rest_cat_dict

    def count_rest_aug_dict(self, new_img_cat_dict, rest_cat_dict):
        rest_set = set([cat_id for cat_id in rest_cat_dict.keys()])
        img_cnt_dict = defaultdict(int)

        break_tf = False
        while True:
            for img_id, cat_list in new_img_cat_dict.items():
                cat_set = set(cat_list)
                if cat_set - rest_set == cat_set:
                    continue
                else:
                    img_cnt_dict[img_id] += 1
                    for cat_id in cat_list:
                        if cat_id in rest_cat_dict:
                            rest_cat_dict[cat_id] -= 1
                            if rest_cat_dict[cat_id] == 0:
                                del rest_cat_dict[cat_id]

                if len(rest_cat_dict) == 0:
                    break_tf = True
                    break
            if break_tf:
                break

        return img_cnt_dict

    # 실제 augmentation 이 진행되는 main 메소드
    def main_AugmentationProcess(self):
        # settings
        if self.whole_count == 1:
            self.write_basic_information()
        self.logger.debug(f'*-*-*- {self.whole_count}번째 augmentation 시작 -*-*-*')
        file_path = self.augmentation_path if self.whole_count != 1 else self.original_img_path
        new_img_info_dict, new_ann_info_dict = {}, defaultdict(list)
        new_img_cat_dict = defaultdict(list)

        # 1. img_id 와 그에 맞춰 증식할 수(=need_cnt)
        for img_id, need_cnt in self.img_cnt_dict.items():
            continue_tf = False
            img_info = self.img_info_dict[img_id]
            file_name = img_info[0]
            file_img = os.path.join(file_path, file_name)
            tmp_cnt = 0

            # 2. img_id 에 적용할 tool
            break_tf = False
            random.shuffle(self.tool_list)
            for tool_name in self.tool_list:
                if tool_name in file_name:
                    continue
                numpy_img = cv2.imread(file_img)
                preprocess_tool = PreProcessing(need_cnt, numpy_img, self.apply_cnt, tool_name).return_preprocess()
                aug_cnt_list = preprocess_tool()

                # 3. (사용할 ann 정보 골라내서) tool 에 맞춰 1장씩 증식
                for aug_cnt in aug_cnt_list:
                    ann_info = self.ann_info_dict[file_name]
                    tmp_aug_cnt = defaultdict(int)

                    if self.max:
                        new_ann_info = []
                        for ann in ann_info:
                            ann_cat_id = ann['category_id']

                            now_cnt = self.cat_log[ann_cat_id]
                            tmp_aug = tmp_aug_cnt[ann_cat_id]

                            if any([ann_cat_id not in self.standard_dict,
                                    now_cnt + tmp_aug >= self.standard_dict[ann_cat_id]]):
                                ann_info.remove(ann)
                            elif all([self.mask_on and ann['segmentation'] == [[]]]):
                                ann_info.remove(ann)
                            else:
                                tmp_aug_cnt[ann_cat_id] += 1
                                new_ann_info.append(ann)
                        ann_info = new_ann_info

                    if len(ann_info) <= 0:
                        break
                    else:
                        ca = CreateAugmentation(numpy_img, aug_cnt, ann_info, tool_name, img_info, self.mask_on)
                        aug_img, aug_name, cat_list, area_list, box_dict, seg_dict = ca.return_main()

                        img_id_uuid = datetime.datetime.now().strftime("%H%M%S%f")
                        aug_cnt = aug_cnt * 10 if aug_name == 'gamma_' else aug_cnt
                        no_path_name = aug_name + str(aug_cnt) + '__' + file_name

                        tmp_ann_info = []
                        for cat, area, box, seg in zip(cat_list, area_list, box_dict.values(), seg_dict.values()):
                            ann_id_uuid = datetime.datetime.now().strftime("%H%M%S%f")
                            if seg == [[]]:
                                continue
                            else:
                                new_ann = dict(id=ann_id_uuid, image_id=img_id_uuid, category_id=cat,
                                               area=area, bbox=box, iscrowd=0, segmentation=seg)
                                tmp_ann_info.append(new_ann)

                        if len(tmp_ann_info) >= 1:
                            post_tool_list = [0] + self.post_tool_list
                            post_tool_name = random.sample(post_tool_list, 1)[0]
                            if post_tool_name == 0 or post_tool_name in no_path_name:
                                pass
                            else:
                                preprocess_tool = PreProcessing(1, aug_img, self.apply_cnt, post_tool_name).return_preprocess()
                                post_aug_cnt = preprocess_tool()[0]
                                post_img_info = [no_path_name, img_info[1], img_info[2]]
                                ca = CreateAugmentation(aug_img, post_aug_cnt, tmp_ann_info, post_tool_name, post_img_info, self.mask_on)
                                aug_img, aug_name, cat_list, area_list, box_dict, seg_dict = ca.return_main()
                                no_path_name = f'{aug_name}{str(post_aug_cnt)}_{no_path_name}'

                        save_path = os.path.join(self.augmentation_path, no_path_name)
                        cv2.imwrite(save_path, aug_img)

                        new_img_info_dict[f'{img_id}_{tmp_cnt}'] = [no_path_name, img_info[1], img_info[2]]
                        new_img = dict(id=img_id_uuid, file_name=no_path_name, width=img_info[1], height=img_info[2])
                        for tmp, box, seg in zip(tmp_ann_info, box_dict.values(), seg_dict.values()):
                            tmp_cat_id = tmp['category_id']
                            new_ann = dict(id=tmp['id'], image_id=img_id_uuid, category_id=tmp_cat_id,
                                           area=tmp['area'], bbox=box, iscrowd=0, segmentation=seg)
                            self.cat_log[tmp_cat_id] += 1
                            self.new_annotations.append(new_ann)
                            new_ann_info_dict[no_path_name].append(new_ann)
                            new_img_cat_dict[f'{img_id}_{tmp_cnt}'].append(tmp_cat_id)
                        self.new_images.append(new_img)
                        tmp_cnt += 1
                    need_cnt -= 1
                    if need_cnt <= 0:
                        break_tf = True
                        break
                if break_tf == True:
                    continue_tf = True
                    break

            tmp_count = sum(self.cat_log.values())
            self.logger.debug(f'{self.whole_count} - [{tmp_count}/{self.total_need_cnt}] = '
                              f'{np.round((tmp_count/self.total_need_cnt)*100,2)} % 진행 >> '
                              f'{np.round(time.time()-self.start_time, 2)}초 소요')
            if continue_tf == True:
                continue

        rest_cat_dict = self.check_rest_categories()
        if len(rest_cat_dict) == 0:
            self.logger.debug('- finish making IMAGES !')
            make_json(self.augmentation_json, images=self.new_images, annotations=self.new_annotations,
                      categories=self.categories, info=self.info)
            self.logger.debug('-- finish making JSON !')
            self.write_final_confirm()
        else:
            rest_need_dict = self.count_rest_aug_dict(new_img_cat_dict, rest_cat_dict)
            self.img_cnt_dict = rest_need_dict
            self.img_info_dict = new_img_info_dict
            self.ann_info_dict = new_ann_info_dict

            self.whole_count += 1
            self.logger.debug('\n')
            self.main_AugmentationProcess()
