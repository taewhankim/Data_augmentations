
# 이미지에 들어있는 오브젝트를 중심으로 crop 해서 새로운 이미지를 만드는 코드
# 오브젝트가 이미지의 크기에 비해 작을 때, 사용할 수 있음
# ex. 만약 한 이미지에 적당한 사이즈의 오브젝트 2개 + 작은 오브젝트 2개가 있다면
# 원본 이미지에는 적당한 사이즈의 오브젝트 2개만 그려지도록 변경
# 작은 오브젝트 2개는 각각의 bbox 를 기준으로 crop 하여 새로운 이미지 1장씩 생성
# >> 1장의 이미지에서 3장이 되는 것

import os
import cv2
import shutil
import numpy as np
import copy
from tqdm import tqdm
from collections import defaultdict
from tools.json_processing.utils import make_dir, read_json, make_json


class CropImages:
    def __init__(self, config: dict):
        main_dir = config['main_dir']
        self.img_dir = os.path.join(main_dir, config['img_dir'])
        self.crop_dir = os.path.join(main_dir, config['crop_dir'])
        json_dir = os.path.join(main_dir, config['json_dir'])
        self.new_json_dir = os.path.join(main_dir, config['new_json_dir'])

        make_dir(self.crop_dir)
        # if config['make_box']:
        #     box_dir = os.path.join(main_dir, config['box_dir'])
        #     make_dir(box_dir)

        self.images, self.annotations, self.categories, self.info = read_json(json_dir)
        self.img_info = self.img_info()
        self.ann_info = self.ann_info()
        self.new_images, self.new_annotations = [], []

        # img_ratio : 이미지 사이즈의 몇 % 이하면 crop 할 것인지 정하는 기준
        # abandon_area : 너무 작아서 사용하지 않을(=버릴) 사이즈의 기준
        self.img_ratio = config['img_ratio']
        self.abandon_area = config['abandon_area']
        self.min_size = config['min_size']

    # img_info : (key) file_name / (value) image information
    def img_info(self):
        img_info = {}
        for img in self.images:
            file_name = img['file_name']
            img_info[file_name] = img

        return img_info

    # ann_info : (key) img_id / (value) annotation information
    def ann_info(self):
        ann_info = defaultdict(list)
        for ann in self.annotations:
            img_id = ann['image_id']
            ann_info[img_id].append(ann)

        return ann_info

    def crop_images(self):
        img_id, ann_id = 0, 0
        area_threshold = int(self.min_size/2)**2

        for file_name, info in tqdm(self.img_info.items()):
            img_w, img_h = info['width'], info['height']
            img_id = info['id']
            ann_list = self.ann_info[img_id]

            tmp_list = []
            cnt = 0

            for ann in ann_list:
                if 'bbox' not in ann:
                    continue

                bbox_area = ann['area']
                if bbox_area < self.abandon_area:
                    continue

                if img_w * img_h * self.img_ratio < bbox_area:
                    tmp_list.append(ann)
                else:
                    img_arr = cv2.imread(os.path.join(self.img_dir, file_name))
                    x, y, w, h = ann['bbox']
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    half_w, half_h = int(w / 2), int(h / 2)
                    center_x, center_y = x + half_w, y + half_h
                    crop_lx, crop_rx = center_x - w, center_x + w
                    crop_uy, crop_dy = center_y - h, center_y + h
                    center_lx_plus = crop_lx * -1 if crop_lx < 0 else 0
                    center_uy_plus = crop_uy * -1 if crop_uy < 0 else 0
                    center_rx_plus = (crop_rx - img_w) * -1 if crop_rx > img_w else 0
                    center_dy_plus = (crop_dy - img_h) * -1 if crop_dy > img_h else 0

                    if any([all([center_lx_plus > 0, center_rx_plus < 0]),
                            all([center_uy_plus > 0, center_dy_plus < 0])]):
                        new_arr = np.zeros(shape=(h*2, w*2, 3))
                        cropped_img = img_arr[y:y+h, x:x+w]
                        new_x, new_y = half_w, half_h
                        new_arr[new_y:new_y+h, new_x:new_x+w] = cropped_img
                        cropped_img = copy.deepcopy(new_arr)
                        # box_img = cv2.rectangle(new_arr, (new_x, new_y), (new_x+w, new_y+h), (0, 255, 0), 3)
                    else:
                        new_center_x = center_x + center_lx_plus + center_rx_plus
                        new_center_y = center_y + center_uy_plus + center_dy_plus
                        new_x, new_y = x-(new_center_x-w), y-(new_center_y-h)
                        new_center_x, new_center_y, new_x, new_y = \
                            int(new_center_x), int(new_center_y), int(new_x), int(new_y)
                        cropped_img = img_arr[new_center_y-h:new_center_y+h, new_center_x-w:new_center_x+w]
                        # box_img = cv2.rectangle(copy.deepcopy(cropped_img), (new_x, new_y),
                        # (new_x+w, new_y+h), (0, 255, 0), 3)

                    if bbox_area < area_threshold:
                        multi = min(int(self.min_size/(2*h)), int(self.min_size/(2*w)))
                        if multi > 1:
                            cropped_img = cv2.resize(cropped_img, dsize=(0, 0),
                                                     fx=multi, fy=multi, interpolation=cv2.INTER_CUBIC)
                            w, h = w*multi, h*multi
                            new_x, new_y = new_x*multi, new_y*multi
                            # box_img = cv2.rectangle(copy.deepcopy(cropped_img), (new_x, new_y),
                            # (new_x+w, new_y+h), (0, 255, 0), 3)

                    save_w, save_h, _ = cropped_img.shape
                    img_name = file_name.split('.')
                    img_name = f'{img_name[0]}_{cnt}.{img_name[1]}'
                    cv2.imwrite(os.path.join(self.crop_dir, img_name), cropped_img)
                    # cv2.imwrite(os.path.join(self.box_dir, img_name), box_img)

                    new_ann_info = {'iscrowd': 0, 'area': w*h, 'bbox': [new_x, new_y, w, h], 'image_id': img_id,
                                    'segmentation': [[new_x, new_y, new_x+w, new_y+h]], 'id': ann_id,
                                    'category_id': ann['category_id'], 'score_all': []}
                    new_img_info = {'file_name': img_name, 'width': save_w, 'id': img_id, 'height': save_h}
                    self.new_images.append(new_img_info)
                    self.new_annotations.append(new_ann_info)
                    img_id += 1
                    ann_id += 1
                    cnt += 1

            if len(tmp_list) != 0:
                new_img_info = {'file_name': file_name, 'width': info['width'], 'id': img_id, 'height': info['height']}
                self.new_images.append(new_img_info)
                shutil.copy(os.path.join(self.img_dir, file_name), os.path.join(self.crop_dir, file_name))
                for tmp_ann in tmp_list:
                    tmp_ann['image_id'] = img_id
                    tmp_ann['id'] = ann_id
                    ann_id += 1
                img_id += 1

        make_json(self.new_json_dir, images=self.new_images, annotations=self.new_annotations,
                  categories=self.categories, info=self.info)


if __name__ == '__main__':
    configs = {
        "main_dir": '',
        "img_dir": '',
        "json_dir": '',
        "crop_dir": '',
        "new_json_dir": '',

        "img_ratio": 0.05,
        "abandon_area": 450,
        "min_size": 360
    }

    CI = CropImages(configs)
    CI.crop_images()
