
import cv2
import os
import numpy as np
from collections import defaultdict
from tools.json_processing.utils import read_json, make_dir, make_json

# right
K_matrix = np.array(
    [[
        1962.3819193932354,
        0.0,
        926.8655327640137
    ],
    [
        0.0,
        1955.8927039929886,
        498.1908222367694
    ],
    [
        0.0,
        0.0,
        1.0
    ]])

K_matrix_undist = np.array(
    [[
        1667.2661994657194,
        0.0,
        922.468514734786
    ],
    [
        0.0,
        1661.7528743627092,
        499.10655832041937
    ],
    [
        0.0,
        0.0,
        1.0
    ]])

dist_coeffs = np.array(
    [[
        -0.18595132145941573
    ],
    [
        -0.01807638431612279
    ],
    [
        -0.5647096285227255
    ],
    [
        1.1682839674833696
    ]])

# undistorted_func 의 구현
raw_path = '/mnt/aistudionas/dms/breadfactory/0_raw_data/2nd_raw_data/2nd_raw_data_20/f_mix'
json_path = '/mnt/aistudionas/dms/breadfactory/0_raw_data/2nd_raw_data/2nd_raw_data_20/_annotation/breadfactory_f_mix.json'
new_json_path = '/mnt/aistudionas/dms/breadfactory/0_raw_data/2nd_raw_data/2nd_raw_data_20/test_json.json'
conv_path = '/mnt/aistudionas/dms/breadfactory/0_raw_data/2nd_raw_data/2nd_raw_data_20/test'

make_dir(conv_path)
images, annotations, categories, info = read_json(json_path)
new_images, new_annotations = [], []
img_id, ann_id = 0, 0

files = os.listdir(raw_path)
files.sort()

img_mapping_ann = defaultdict(list)
for ann in annotations:
    img_id = ann['image_id']
    img_mapping_ann[img_id].append(ann)

# for f in files:
for img in images:
    file_name = img['file_name']
    img_id = img['id']
    ann_info = img_mapping_ann[img_id]
    object_cnt = len(ann_info)

    file_dir = os.path.join(raw_path, file_name)
    img = cv2.imread(os.path.join(raw_path, file_dir))
    height, width, channel = img.shape

    tmp_img = np.zeros((height, width, (channel + object_cnt)), np.uint8)
    tmp_img[:, :, 0] = img[:, :, 0]
    tmp_img[:, :, 1] = img[:, :, 1]
    tmp_img[:, :, 2] = img[:, :, 2]

    for idx, ann in enumerate(ann_info):
        bbox = ann['bbox']
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
        # tmp_save_img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        # cv2.imwrite(os.path.join(conv_path, f'before_{idx}_{file_name}'), tmp_save_img)
        tmp_img[:, :, (channel + idx)] = cv2.rectangle(np.array(tmp_img[:, :, (channel + idx)]), (x1, y1), (x2, y2), (255, 255, 255), -1)

        # not use segmentation information >> skip
        # segmentation = ann['segmentation']
        # tmp_len = int(len(segmentation[0])/2)
        # for i in range(tmp_len):
        #     x = segmentation[0][i * 2]
        #     y = segmentation[0][(i * 2) + 1]
        #     tmp.append([x, y])
        # polygon = np.array(tmp, np.int32)
        # gray_img = cv2.fillPoly(gray_img, [polygon], 255)

        # tmp_save_img = np.array(tmp_img[:, :, (channel + idx)])
        # cv2.imwrite(os.path.join(conv_path, f'before_{idx}_{file_name}'), tmp_save_img)

    image_size = tmp_img.shape[:2][::-1]
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K_matrix, dist_coeffs, None, K_matrix_undist, image_size, cv2.CV_16SC2)
    img_undistorted = cv2.remap(tmp_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cropped = img_undistorted.copy()
    cropped = img_undistorted[59:1019, 240:1920]
    new_height, new_width, no_channel = cropped.shape

    for idx, ann in enumerate(ann_info):
        tmp_channel = channel + idx
        no_img, contours_info, no_hierachy = cv2.findContours(np.array(cropped[:, :, tmp_channel]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # tmp_save_img = np.array(cropped[:, :, tmp_channel])
        # cv2.imwrite(os.path.join(conv_path, f'{idx}_{file_name}'), tmp_save_img)

        contours_pt = []
        tmp_con = np.array(contours_info)
        tmp_con = tmp_con.flatten().tolist()
        x_list, y_list = [], []
        for idx, tmp in enumerate(tmp_con):
            if idx % 2 == 0:
                x_list.append(tmp)
            else:
                y_list.append(tmp)

        # tmp_len = int((len(tmp_con)) / 2)
        # for i in range(tmp_len):
        #     x = tmp_con[i * 2]
        #     y = tmp_con[(i * 2) + 1]
        #     contours_pt.append([x, y])
        # for con in contours_pt:
        #     x_list.append(con[0])
        #     y_list.append(con[1])

        min_x, max_x = min(x_list), max(x_list)
        min_y, max_y = min(y_list), max(y_list)

        w = 0 if max_x - min_x < 0 else max_x - min_x
        h = 0 if max_y - min_y < 0 else max_y - min_y
        bbox = [min_x, min_y, w, h]
        segmentation = [bbox]

        new_ann = dict(id=ann_id, image_id=img_id, category_id=ann['category_id'],
                      area=w*h, bbox=bbox, iscrowd=0, segmentation=segmentation)
        new_annotations.append(new_ann)
        ann_id += 1

        # new_ann = dict(id=ann_id, image_id=img_id, category_id=ann['category_id'],
        #                area=w * h, bbox=bbox, iscrowd=0, segmentation=segmentation)
        # new_annotations.append(new_ann)
        # ann_id += 1
        # new_img = dict(id=img_id, file_name=f'{idx}_{file_name}', width=new_width, height=new_height)
        # new_images.append(new_img)
        # img_id += 1

    save_img = np.zeros((new_height, new_width, 3), np.uint8)
    save_img[:, :, 0] = cropped[:, :, 0]
    save_img[:, :, 1] = cropped[:, :, 1]
    save_img[:, :, 2] = cropped[:, :, 2]

    new_img = dict(id=img_id, file_name=file_name, width=new_width, height=new_height)
    new_images.append(new_img)
    img_id += 1
    cv2.imwrite(os.path.join(conv_path, file_name), save_img)

make_json(new_json_path, images=new_images, annotations=new_annotations, categories=categories, info=info)
