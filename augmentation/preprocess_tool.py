
import cv2
import random
import numpy as np


class PreProcessing(object):
    # 메서드를 추가할 때, 주의할 점
    # >> 1. 메서드명에 2개의 _ 가 포함되게 설정 + 가운데 글자덩어리가 pre_tool_box 의 key
    # >> 2. output 을 aug_cnt_list(list) 으로 일정하게 맞춰야 함
    def __init__(self, aug_cnt, numpy_img, apply_cnt, tool_name):
        self.aug_cnt = aug_cnt
        self.numpy_img = numpy_img
        self.apply_cnt = apply_cnt
        self.tool_name = tool_name

    def return_preprocess(self):
        pre_tool_box = {"gamma": self.pre_gamma_correction, "cutout": self.pre_random_img,
                        "mirroring": self.pre_mirroring_img, "crop": self.pre_random_img,
                        "rotation": self.pre_rotation_img, "gaussian": self.pre_random_img,
                        "elastic": self.pre_random_img}
        return pre_tool_box[self.tool_name]

    # gamma_correction 의 preprocess
    def pre_gamma_correction(self):
        numpy_img = self.numpy_img.astype('uint8')
        img_yuv = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2YUV)
        y_value = img_yuv[:, :, 0]
        y_mean = np.mean(y_value)

        # 각각 11개, 13개, 11개
        if y_mean < 140:
            min_v = 0.9
            max_v = 2.0
        elif all([y_mean >= 140, y_mean < 200]):
            min_v = 0.7
            max_v = 2.0
        else:
            min_v = 0.5
            max_v = 1.6

        gamma_range = list(np.arange(min_v, max_v, 0.1))
        if 1.0 in gamma_range:
            gamma_range.remove(1.0)

        if self.aug_cnt >= len(gamma_range)*2:
            gam_cnt = len(gamma_range)
        elif all([self.aug_cnt < len(gamma_range)*2, self.aug_cnt >= len(gamma_range)]):
            gam_cnt = random.randint(self.aug_cnt - len(gamma_range), len(gamma_range))
        else:
            gam_cnt = random.randint(1, self.aug_cnt)

        aug_cnt_list = np.round(random.sample(gamma_range, gam_cnt), 2)
        return aug_cnt_list

    # mirroring 의 preprocess
    # len(aug_cnt_list) 의 최대값은 3
    def pre_mirroring_img(self):
        aug_cnt = 3 if self.aug_cnt > 3 else self.aug_cnt
        mirror_num = [-1, 0, 1]
        aug_cnt_list = random.sample(mirror_num, aug_cnt)
        return aug_cnt_list

    # rotation 의 preprocess
    # len(aug_cnt_list) 의 최대값은 351
    def pre_rotation_img(self):
        aug_cnt = self.apply_cnt if self.aug_cnt > self.apply_cnt else self.aug_cnt
        angle = list(range(5, 356))
        aug_cnt_list = random.sample(angle, aug_cnt)
        return aug_cnt_list

    # cutout, crop, gaussian, elastic 의 preprocess
    # len(aug_cnt_list) 의 최대값은 self.apply_cnt(→랜덤하게 바뀌므로 매번 다름)
    def pre_random_img(self):
        aug_cnt = self.apply_cnt if self.aug_cnt > self.apply_cnt else self.aug_cnt
        random_multi = random.uniform(0.1, 1)
        new_aug_cnt = np.round(aug_cnt * random_multi).astype(int)
        if new_aug_cnt == 0:
            new_aug_cnt = 1
        aug_cnt_list = list(range(new_aug_cnt))
        return aug_cnt_list
