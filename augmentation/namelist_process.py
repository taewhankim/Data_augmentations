
from collections import defaultdict
import os
from tools.json_processing.utils import read_json


class Utils(object):
    @staticmethod
    def count_categories(annotations):
        categories_num = defaultdict(int)
        for anno_info in annotations:
            cat = anno_info['category_id']
            categories_num[cat] += 1

        return categories_num

    @staticmethod
    def count_lack_quantity(categories_num, range_min):
        max_dict = {}
        for key, value in categories_num.items():
            if value < range_min:
                max_dict[int(key)] = range_min - value
            else:
                max_dict[int(key)] = 0
        max_dict = dict(sorted(max_dict.items(), key=lambda x: x[1]))

        return max_dict

    @staticmethod
    def make_id_dict(annotations):
        img_id_dict, cat_id_dict = defaultdict(list), defaultdict(list)
        for ann_dict in annotations:
            img_id = ann_dict['image_id']
            cat_id = ann_dict['category_id']
            img_id_dict[img_id].append(cat_id)
            cat_id_dict[cat_id].append(img_id)
        return img_id_dict, cat_id_dict


class ChooseCategories(object):

    '''

    ** 증식에 사용하고자 하는 images & annotations 골라주는 클래스
        (input-1) cfg(=aug_configs.json)
            → (output-2) json - new_images, new_annotations

    어떤 메소드를 사용할 것인지 cc_num 으로 결정(in aug_configs.json)
        → {1: self.pick_no_use_categories, 2: self.del_no_use_categories}
    - (아무것도 적지 않을 시에는) default : 모든 카테고리 사용
    - 1 : 사용하고 싶지 않은 카테고리가 든 이미지 자체를 제외 + 이미지에 다른 클래스가 남아있어도 제외
    - 2 : 사용하고 싶지 않은 카테고리의 annotations 제외 + 이미지에 다른 클래스가 남아있다면 이미지는 그대로 사용

    '''

    def __init__(self, configs):
        self.configs = configs
        self.original_json = os.path.join(self.configs.ROOT, self.configs.ORIGINAL_JSON)
        self.images, self.annotations, _, _ = read_json(self.original_json)
        self.img_id_dict, self.cat_id_dict = Utils.make_id_dict(self.annotations)

    def return_function(self):
        cc_num = self.configs.CC_NUM
        if cc_num == 0:
            return self.images, self.annotations
        else:
            function_box = {1: self.pick_no_use_categories(), 2: self.del_no_use_categories()}
            return function_box[cc_num]

    # 1 : 사용하고 싶지 않은 카테고리가 든 이미지 자체를 제외(=다른 클래스가 남아있어도 제외)
    def pick_no_use_categories(self):
        out_categories = self.configs.OUT_CATEGORIES
        new_images, new_annotations = [], []

        out_img_list = []
        for img_id, cat_list in self.img_id_dict:
            for cat in cat_list:
                if cat in out_categories:
                    out_img_list.append(img_id)
                    break

        for anno_dict in self.annotations:
            if anno_dict["image_id"] not in out_img_list:
                new_annotations.append(anno_dict)
        for img_dict in self.images:
            if img_dict["id"] not in out_img_list:
                new_images.append(img_dict)

        return new_images, new_annotations

    # 2 : 사용하고 싶지 않은 카테고리의 annotations 제외 + 이미지에 다른 클래스가 남아있다면 이미지는 그대로 사용
    def del_no_use_categories(self):
        out_categories = self.configs.OUT_CATEGORIES
        new_images, new_annotations = [], []

        out_img_list = []
        for img_id, cat_list in self.img_id_dict.items():
            out_len = 0
            for cat in cat_list:
                if cat in out_categories:
                    out_len += 1
            if len(cat_list) == out_len:
                out_img_list.append(img_id)

        for anno_dict in self.annotations:
            if anno_dict['category_id'] not in out_categories:
                new_annotations.append(anno_dict)
        for img_dict in self.images:
            if img_dict["id"] not in out_img_list:
                new_images.append(img_dict)

        return new_images, new_annotations


class ChooseImages(object):

    '''

        ** 증식에 사용하고자 하는
            (input-2) cfg(=aug_configs.json), json - new_images, new_annotations
                → (output-3) augmentation image id dict(=실제 증식에 사용), augmentation category cnt dict(=증식 기준점),
                categories_num(=원본 json 의 각 클래스 별 object 수가 담긴 dict 으로 증식 후, 확인용)

        어떤 메소드를 사용할 것인지 cc_num 으로 결정(in aug_configs.json)
            → {1: self.increase_by_images}
        - (아무것도 적지 않을 시에는) default : object 수를 range_min 에 맞추어 증식하지만 과증식될 수도 있음
        - 1 : 전체 이미지를 똑같이 range_min 개씩 증식

    '''

    def __init__(self, configs):
        self.configs = configs
        self.new_images, self.new_annotations = ChooseCategories(self.configs).return_function()
        self.range_min = self.configs.RANGE_MIN
        self.categories_num = Utils.count_categories(self.new_annotations)

    def return_function(self):
        if self.configs.CI_NUM == 1:
            return self.increase_by_images()
        else:
            return self.increase_by_objects()

    # 1 : 전체 이미지를 똑같이 range_min 개씩 증가 (object 가 아닌 image 를 기준)
    def increase_by_images(self):
        img_cnt_dict, cat_cnt_dict = {}, {}

        for img_dict in self.new_images:
            img_cnt_dict[img_dict["id"]] = self.range_min
        for cat, cnt in self.categories_num.items():
            cat_cnt_dict[cat] = cnt * self.range_min

        return img_cnt_dict, cat_cnt_dict

    # 2 : class 별 object 수가 최소 range_min 이 되도록 최대한 조절
    # >> 1) 필요한 증식 수가 적은 class 부터 선별
    # >> 2) 다른 object 가 섞이지 않은 pure image 부터 선별
    # +. 만약 max_dict 을 return 하여 증식하면 부족한 object 수를 딱 맞출 수 있고 (권장) > 추가 수정이 필요함 (사용금지)
        # 현재 프로세스는 img_cnt_dict 과 cat_cnt_dict 위주로 굴러가기 때문에
        # max_dict 을 사용하면 sum(img_cnt_dict.values()) 가 sum(max_dict.values()) 보다 큰 경우가 발생할 수 있음
    # +. cat_cnt_dict 을 return 하여 증식하면 부족한 object 수 + a 가 될 수 있음
    def increase_by_objects(self):
        max_dict = Utils.count_lack_quantity(self.categories_num, self.range_min)
        img_id_dict, cat_id_dict = Utils.make_id_dict(self.new_annotations)
        img_cnt_dict, cat_cnt_dict = defaultdict(int), defaultdict(int)

        for cat_id, need_cnt in max_dict.items():
            need_cnt -= cat_cnt_dict[cat_id]
            if need_cnt <= 0:
                continue

            pure, not_pure, no_use = [], [], []
            img_list = cat_id_dict[cat_id]
            for img_id in img_list:
                cat_list = img_id_dict[img_id]
                if len(cat_list) == cat_list.count(cat_id):
                    pure.append(img_id)
                else:
                    out_len = 0
                    for cat in cat_list:
                        already_used = cat_cnt_dict[cat]
                        fix_need = max_dict[cat]
                        if already_used >= fix_need:
                            out_len += 1
                            break
                    if out_len == 0:
                        not_pure.append(img_id)
                    else:
                        no_use.append(img_id)

            len_pure = len(pure)
            len_not_pure = len(not_pure)
            tmp_img_list = no_use if len_pure == 0 and len_not_pure == 0 else pure + not_pure

            bk_tf = True
            while bk_tf:
                for img_id in tmp_img_list:
                    cat_list = img_id_dict[img_id]
                    for tmp_cat in cat_list:
                        cat_cnt_dict[tmp_cat] += 1
                        if tmp_cat == cat_id:
                            need_cnt -= 1
                    img_cnt_dict[img_id] += 1

                    if need_cnt <= 0:
                        bk_tf = False
                        break

        return img_cnt_dict, max_dict, cat_cnt_dict, self.categories_num
