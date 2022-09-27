
class Configs(object):
    # setting dir
    ROOT = '/mnt/dms/aistudio_snack/'
    ORIGINAL_JSON = '1_splitted_data/3rd_splitted_data/annotation/val.json'
    ORIGINAL_IMAGE = '1_splitted_data/3rd_splitted_data/val'
    AUGMENTATION_DIR = '/mnt/dms/sunn/augmentation'

    CC_NUM = 0
    CI_NUM = 0
    OUT_CATEGORIES = None

    RANGE_MIN = 30
    APPLY_CNT = 50
    # ["gamma", "cutout", "mirroring", "crop", "rotation", "gaussian", "elastic"]
    TOOL_NAME_LIST = ["gamma", "cutout", "mirroring", "crop", "rotation", "gaussian", "elastic"]
    POST_TOOL_NAME_LIST = []

    MAX_OR_ALL = 'ALL'
    # segmentation 이 [box] 인 데이터는 MASK_ON = False 로 진행해야 함
    # ["crop", "rotation", "elastic"] 에만 적용
    MASK_ON = False
