
from augmentation.configs import Configs
from augmentation.namelist_process import ChooseImages
from augmentation.augmentation_process import TryAugmentation


def main():
    cfg = Configs()
    img_cnt_dict, max_dict, cat_cnt_dict, categories_num = ChooseImages(configs=cfg).return_function()
    TryAugmentation(cfg, img_cnt_dict, max_dict, cat_cnt_dict, categories_num).main_AugmentationProcess()


if __name__ == '__main__':
    main()
