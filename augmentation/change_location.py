
import numpy as np
import cv2
import sys


class ChangeLocation(object):
    def __init__(self,  mask_on):
        self.mask_on = mask_on

    # [before] get_point + make_tmp_image
    def change_location(self, ann_info, numpy_img):
        location_info = self.get_point(ann_info)
        tmp_img = self.make_tmp_image(numpy_img, location_info)

        return location_info, tmp_img

    # [before] cv2 에 적용할 수 있도록 bbox 또는 segmentation 형태 변환
    def get_point(self, ann_info):
        change_location_info = {}
        self.object_cnt = len(ann_info)

        if self.mask_on:
            # segmentation : [x, y, x, y, ... x, y] -> [[x, y], ..., [x, y]]
            for idx, ann in enumerate(ann_info):
                tmp_location = []
                segmentation = ann['segmentation'][0]
                tmp_len = int(len(segmentation) / 2)
                for i in range(tmp_len):
                    x = segmentation[i * 2]
                    y = segmentation[(i * 2) + 1]
                    tmp_location.append([x, y])
                change_location_info[idx] = tmp_location
        else:
            # bbox : [x, y, width, height] -> [[x1, y1], [x2, y2]]
            for idx, ann in enumerate(ann_info):
                bbox = ann['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                tmp_location = [x1, y1, x2, y2]
                change_location_info[idx] = tmp_location

        return change_location_info

    # [before] augmentation tool 을 적용할 수 있도록 image 형태 변환
    def make_tmp_image(self, numpy_img, location_info):
        object_cnt = len(location_info)
        height, width, channel = numpy_img.shape

        tmp_img = np.zeros((height, width, (channel + object_cnt)), np.uint8)
        tmp_img[:, :, 0] = numpy_img[:, :, 0]
        tmp_img[:, :, 1] = numpy_img[:, :, 1]
        tmp_img[:, :, 2] = numpy_img[:, :, 2]

        if self.mask_on:
            for idx, point_list in location_info.items():
                tmp_img[:, :, (channel + idx)] = \
                    cv2.fillPoly(np.array(tmp_img[:, :, (channel + idx)]), [np.array(point_list, np.int32)], 255)
        else:
            for idx, point_list in location_info.items():
                x1, y1, x2, y2 = point_list[0], point_list[1], point_list[2], point_list[3]
                tmp_img[:, :, (channel + idx)] = \
                    cv2.rectangle(np.array(tmp_img[:, :, (channel + idx)]), (x1, y1), (x2, y2), (255, 255, 255), -1)

        return tmp_img

    # [after] get_contours + make_new_image
    def new_location(self, aug_img, ann_info):
        whole_box, whole_segmentation = self.get_contours(aug_img)
        new_img = self.make_new_image(aug_img)
        area_list = [bbox[2] * bbox[3] for bbox in whole_box.values()]
        cat_list = [ann['category_id'] for ann in ann_info]

        return new_img, whole_box, whole_segmentation, area_list, cat_list

    # [after] deformation 이 적용된 이미지에 맞는 새로운 bbox 와 segmentation
    def get_contours(self, aug_img):
        height, width, channel = aug_img.shape
        plus_channel = channel - 3
        whole_box, whole_segmentation = {}, {}

        for key_idx in range(plus_channel):
            tmp_channel = 3 + key_idx
            # (21-02-04) cv2 버전에 따라 return 되는 값이 2개 혹은 3개...
            contours_info, no_hierachy = \
                cv2.findContours(np.array(aug_img[:, :, tmp_channel]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours_info) > 1:
                max_len = 0
                con_idx = -1
                for idx, info in enumerate(contours_info):
                    if len(info) > max_len:
                        con_idx = idx
                        max_len = len(info)
            else:
                con_idx = -1

            tmp_con = np.array(contours_info[con_idx])
            tmp_con = tmp_con.flatten().tolist()
            x_list, y_list = [], []
            seg_info = []

            for tmp_idx, tmp in enumerate(tmp_con):
                if tmp_idx % 2 == 0:
                    x_list.append(tmp)
                    seg_info.append(tmp)
                else:
                    # seg = [x_list[-1], tmp]
                    y_list.append(tmp)
                    seg_info.append(tmp)

            min_x, max_x = min(x_list), max(x_list)
            min_y, max_y = min(y_list), max(y_list)

            w = 0 if max_x - min_x < 0 else max_x - min_x
            h = 0 if max_y - min_y < 0 else max_y - min_y
            bbox = [min_x, min_y, w, h]
            whole_box[key_idx] = bbox
            whole_segmentation[key_idx] = seg_info if self.mask_on else [bbox]

        return whole_box, whole_segmentation

    # [after] object 수에 맞게 채널을 늘려놓은 이미지를 다시 원상복귀
    def make_new_image(self, aug_img):
        height, width, channel = aug_img.shape
        new_img = np.zeros((height, width, 3), np.uint8)
        new_img[:, :, 0] = aug_img[:, :, 0]
        new_img[:, :, 1] = aug_img[:, :, 1]
        new_img[:, :, 2] = aug_img[:, :, 2]

        return new_img

    def minmax(self, ann_info):
        x_max, y_max = 0, 0
        x_min, y_min = sys.maxsize, sys.maxsize

        for ann in ann_info:
            box_tmp = ann['bbox']
            if box_tmp[0] < x_min:
                x_min = int(box_tmp[0])
            if (box_tmp[0] + box_tmp[2]) > x_max:
                x_max = int(box_tmp[0] + box_tmp[2])
            if box_tmp[1] < y_min:
                y_min = int(box_tmp[1])
            if (box_tmp[1] + box_tmp[3]) > y_max:
                y_max = int(box_tmp[1] + box_tmp[3])

        return x_min, x_max, y_min, y_max
