
import cv2
from collections import defaultdict
import numpy as np
import os
from tools.json_processing.utils import read_json, make_dir


# segmentation map image 생성 : 배경은 검정, 오브젝트는 하양
def fill_poly(new_img_dir, json_dir, file_list):
    make_dir(new_img_dir)
    images, annotations, _, _ = read_json(json_dir)

    img_mapping_dict = {}
    for img in images:
        img_id = img['id']
        img_mapping_dict[img_id] = img

    ann_mapping_dict = defaultdict(list)
    for ann in annotations:
        img_id = ann['image_id']
        ann_mapping_dict[img_id].append(ann)

    idx = 0
    for img_id, ann_info in ann_mapping_dict.items():
        img_info = img_mapping_dict[img_id]
        file_name = img_info['file_name']

        if file_name not in file_list:
            continue
        else:
            width, height = img_info['width'], img_info['height']
            new_img = np.zeros((height, width, 3), np.uint8)
            len_obj = len(ann_info)

            if len_obj != 0:
                for ann in ann_info[:len_obj-1]:
                    segmentation = ann['segmentation']
                    tmp_len = int(len(segmentation[0]) / 2)
                    tmp = []
                    for i in range(tmp_len):
                        x = segmentation[0][i * 2]
                        y = segmentation[0][(i * 2) + 1]
                        tmp.append([x, y])
                    polygon = np.array(tmp, np.int32)
                    new_img = cv2.fillPoly(new_img, [polygon], (255, 255, 255))

                cv2.imwrite(os.path.join(new_img_dir, f'minus_{file_name}'), new_img)
                # cv2.imwrite(os.path.join(new_img_dir, file_name), new_img)
                idx += 1


if __name__ == '__main__':
    json_dir = '/mnt/aistudionas/dms/breadfactory/0_raw_data/2nd_raw_data/2nd_raw_data_20/_annotation/breadfactory_ab_double.json'
    new_img_dir = '/mnt/aistudionas/etc/temp/fillpoly_new'
    file_list = ['0030120200707142927600629.jpg','0030120200707142519857172.jpg', '0030120200707135859725261.jpg',
                 '0030120200707140942358333.jpg', '0030120200707140335491821.jpg', '0030120200707142007065713.jpg',
                 '0030120200707142933242150.jpg', '0030120200707142955339019.jpg']

    fill_poly(new_img_dir, json_dir, file_list)
