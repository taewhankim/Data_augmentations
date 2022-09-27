
import os
import json
import shutil
from tqdm import tqdm
from typing import Dict, List


def read_json(json_file: str) -> Dict:
    with open(json_file, 'r') as tmp_json:
        json_dict = json.load(tmp_json)
        return json_dict


def copy(json_dir: str, ori_image_dir: str, new_image_dir: str):
    # json - images 에 있는 이미지만 골라서 특정 폴더로 복사

    os.makedirs(new_image_dir, exist_ok=True)
    json_dict = read_json(json_dir)
    images = json_dict['images']

    for img_info in tqdm(images, total=len(images)):
        file_name = img_info['file_name']
        file_dir = os.path.join(ori_image_dir, file_name)
        new_file_dir = os.path.join(new_image_dir, file_name)
        shutil.copy(file_dir, new_file_dir)


if __name__ == '__main__':
    json_dir = '/mnt/dms/mealy_store/1_inspected_data/14th_inspected/from_13_inspected.json'
    ori_image_dir = '/mnt/dms/mealy_store/1_inspected_data/13th_inspected/images'
    new_image_dir = '/mnt/dms/mealy_store/1_inspected_data/14th_inspected/from_13'

    copy(
        json_dir=json_dir,
        ori_image_dir=ori_image_dir,
        new_image_dir=new_image_dir
    )
