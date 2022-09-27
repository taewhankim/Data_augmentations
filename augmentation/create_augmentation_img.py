
import numpy as np
import cv2
import datetime
import random
import copy
from scipy.ndimage.interpolation import map_coordinates

from augmentation.change_location import ChangeLocation
from augmentation.json_tool import MakeJson


class CreateAugmentation(object):
    # 메서드를 추가할 때, 주의할 점
    # >> 1. 메서드명에 1개의 _ 가 포함되게 설정 + 맨 앞의 글자덩어리가 tool_box 의 key
    # >> 2. output 을 augmentation_img 로 일정하게 맞춰야 함
    def __init__(self, numpy_img, aug_cnt, ann_info, tool_name, img_info, mask_on):
        self.numpy_img = numpy_img
        self.aug_num = aug_cnt
        self.ann_info = ann_info
        self.tool_name = tool_name
        self.img_info = img_info
        self.mask_on = mask_on

    # augmentation img 와 new json 정보를 같이 묶어서 return
    def return_main(self):
        tool = self.return_tool()
        aug_name = f'{self.tool_name}_'
        augmentation_img = tool()

        if self.tool_name in ['crop', 'rotation', 'elastic']:
            augmentation_img, box_dict, seg_dict, area_list, cat_list\
                = ChangeLocation(self.mask_on).new_location(augmentation_img, self.ann_info)
        else:
            ma = MakeJson(self.ann_info, self.aug_num, self.img_info, self.tool_name)
            cat_list, area_list, box_dict, seg_dict = ma.json_main()

        return augmentation_img, aug_name, cat_list, area_list, box_dict, seg_dict

    # create_augmentation_img 클래스에 포함된 메서드를 수행하여 augmentation img 를 return
    def return_tool(self):
        tool_box = {"gamma": self.gamma_correction, "cutout": self.cutout_box, "mirroring": self.mirroring_img,
                    "crop": self.crop_img, "rotation": self.rotation_img, "gaussian": self.gaussian_noise,
                    "elastic": self.elastic_deformations}

        return tool_box[self.tool_name]

    # gamma_correction : 이미지의 밝기 조정
    def gamma_correction(self):
        img = self.numpy_img/255.0
        invGamma = 1.0 / self.aug_num
        img = cv2.pow(img, invGamma)

        return np.uint8(img*255)

    # cutout_box : 이미지에 랜덤하게 박스(■)가 그려짐
    # >> box_number_list : 이미지에 뿌려질 박스 수
    # >> cut_length_list : 박스 한 변의 길이
    def cutout_box(self):
        height, width, channel = self.numpy_img.shape
        box_number_list = [30, 40, 50]
        cutout_color_list = [255, 1.2, 1.5, 1.7, 1.9]
        cut_length_list = [5, 10, 15, 20]

        box_number = random.sample(box_number_list, 1)[-1]
        h_range = list(np.arange(0, height))
        w_range = list(np.arange(0, width))
        y = random.sample(h_range, box_number)
        x = random.sample(w_range, box_number)

        numpy_img = self.numpy_img
        for box_y, box_x in zip(y, x):
            cut_length = random.sample(cut_length_list, 1)[-1]
            y1 = box_y - (cut_length // 2)
            y2 = box_y + (cut_length // 2)
            x1 = box_x - (cut_length // 2)
            x2 = box_x + (cut_length // 2)
            cutout_color = random.sample(cutout_color_list, 1)[-1]

            if type(cutout_color) == int:
                numpy_img[y1: y2, x1: x2] = cutout_color
            elif all([cutout_color >= 1.0, cutout_color < 2.0]):
                tmp_mask = numpy_img[y1:y2, x1:x2]
                tmp_mask = tmp_mask / 255
                tmp = 1.0 / cutout_color
                tmp_mask = cv2.pow(tmp_mask, tmp)
                try:
                    tmp_mask *= 255
                    numpy_img[y1: y2, x1: x2] = tmp_mask
                except TypeError:
                    pass
            else:
                tmp_mask = numpy_img[y1: y2, x1: x2]
                tmp_mask = tmp_mask.astype('float64')
                tmp_mask *= cutout_color
                tmp_mask = tmp_mask.astype('uint8')
                numpy_img[y1: y2, x1: x2] = tmp_mask

        return numpy_img

    # mirroring : 이미지를 상하, 좌우, 상하좌우 반전
    def mirroring_img(self):
        mir_img = cv2.flip(self.numpy_img, self.aug_num)

        return mir_img

    # gaussian_noise : 이미지에 noise 를 주는 기법
    # >> 현재는 이미지당 한 장만 생성
    def gaussian_noise(self):
        selection = 0.05
        gau_img = self.numpy_img / 255.0
        noise = np.random.normal(0, 1, (gau_img.shape))
        new_gau = np.clip((gau_img + noise * selection), 0, 1)

        return new_gau * 255.0

    # crop : 이미지를 잘라내어, 잘라낸 이미지를 원본 사이즈로 변경
    # >> percent : 원본 이미지를 몇 % 비율로 잘라낼 것인지 결정
    def crop_img(self):
        height, width, channel = self.numpy_img.shape
        cl = ChangeLocation(self.mask_on)
        location_info, tmp_img = cl.change_location(self.ann_info, self.numpy_img)
        x_min, x_max, y_min, y_max = cl.minmax(self.ann_info)

        percent = 0.9
        # 변경할 이미지 크기 (현재는 원본 이미지의 약 90%로 고정)
        new_w = int(width * percent)
        new_h = int(height * percent)

        # 기준점
        random.seed(datetime.datetime.now())
        x = random.randint(0, int(width - new_w))
        y = random.randint(0, int(height - new_h))
        if x >= x_min:
            if (x_min - 10) < 0:
                x = 0
            else:
                x = x_min - 10
        if y >= y_min:
            if (y_min - 10) < 0:
                y = 0
            else:
                y = y_min - 10

        if (x + new_w) < x_max:
            new_w = (x_max - x) + 10
        if (x + new_w) >= width:
            new_w = width - x

        if (y + new_h) < y_max:
            new_h = (y_max - y) + 10
        if (y + new_h) >= height:
            new_h = height - y

        img = tmp_img[y:(y + new_h), x:(x + new_w)]
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        return img

    def rotation_img(self):
        height, width, channel = self.numpy_img.shape
        cl = ChangeLocation(self.mask_on)
        location_info, tmp_img = cl.change_location(self.ann_info, self.numpy_img)
        object_cnt = len(self.ann_info)

        for i in range(5):
            img = copy.deepcopy(tmp_img)
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), self.aug_num, 1)
            new_img = cv2.warpAffine(img, matrix, (width, height), borderValue=(0, 0, 0))
            # img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

        return new_img

    def elastic_deformations(self):
        height, width, channel = self.numpy_img.shape
        cl = ChangeLocation(self.mask_on)
        location_info, tmp_img = cl.change_location(self.ann_info, self.numpy_img)

        random_state = np.random.RandomState(None)
        shape = tmp_img.shape
        shape_size = shape[:2]

        ela_percent = 30
        alpha_affine = width / ela_percent

        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
             center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(tmp_img, M, shape_size[::-1], borderValue=(0, 0, 0))

        shape_value = (random_state.rand(*shape) * 2 - 1)
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + shape_value, (-1, 1)), np.reshape(x + shape_value, (-1, 1)), np.reshape(z, (-1, 1))

        img = map_coordinates(img, indices, order=1, mode='reflect')
        img = img.reshape(shape)

        return img
