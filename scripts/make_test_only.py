
import os
import json
from collections import OrderedDict
from typing import Dict, Tuple


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


def testonly(json_dir: str, new_json_dir: str):
    json_dict = read_json(json_dir)
    categories = json_dict['categories']
    info = json_dict['info']
    make_json(new_json_dir, images=[], categories=categories, info=info)


if __name__ == '__main__':
    json_dir = '/mnt/dms/mealy_store/3_run_data/14th_run/annotations/test.json'
    new_json_dir = '/mnt/dms/mealy_store/3_run_data/14th_run/annotations/test-only.json'

    testonly(
        json_dir=json_dir,
        new_json_dir=new_json_dir
    )
