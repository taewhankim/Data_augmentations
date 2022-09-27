
import os
import json
from collections import OrderedDict
from typing import Dict


def read_json(json_file: str) -> Dict:
    with open(json_file, 'r') as tmp_json:
        json_dict = json.load(tmp_json)
        return json_dict


def make_json(new_json_dir: str, **kwargs):
    option = kwargs
    file_data = OrderedDict()
    for key, values in option.items():
        file_data[key] = values

    with open(new_json_dir, 'w', encoding='utf-8') as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent='\t')


def remove(json_dir: str, new_json_dir: str):
    # /mnt/dms/mealy_store/0_raw_data/sample.jpg > sample.jpg

    json_dict = read_json(json_dir)
    images = json_dict['images']

    new_images = []
    for img in images:
        file_name = img['file_name']
        new_file_name = os.path.basename(file_name)
        img['file_name'] = new_file_name
        new_images.append(img)

    # make file-path remove json.
    make_json(
        new_json_dir,
        images=new_images,
        annotations=json_dict['annotations'],
        categories=json_dict['categories'],
        info=json_dict['info']
    )


if __name__ == '__main__':
    json_dir = '/mnt/dms/mealy_store/0_raw_data/14_raw/20220801100627291_coco_lab.json'
    new_json_dir = '/mnt/dms/mealy_store/0_raw_data/14_raw/14_raw.json'

    remove(
        json_dir=json_dir,
        new_json_dir=new_json_dir
    )
