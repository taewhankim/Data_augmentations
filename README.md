# Data Augmentation
- aistudio notion page : https://www.notion.so/aistudio/Augmentation-7136628827734bbcb27566718b4e897e
- last updated on 2022-07-26


## Environment
```bash
# create anaconda virtual environment.
conda create -n augmentation_env python=3.8 -y
conda activate augmentation_env

# install libraries for model.
pip install -r requirements.txt
```


## Run
- [configs.py](/augmentation/configs.py)에 적절한 파라미터 값을 입력한 다음에 실행
```bash
python main.py
```
- configs.py에 대한 설명

```{.python}
class Configs(object):
    # setting dir
    ROOT = (str)루트 디렉토리
    ORIGINAL_JSON = (str)원본 JSON 디렉토리
    ORIGINAL_IMAGE = (str)원본 이미지 디렉토리
    AUGMENTATION_DIR = (str)증식될 이미지 및 json이 저장될 경로
 
    # 각 번호에 대한 자세한 설명은 namelist_process.py 참고
    CI_NUM = (int)디폴트(=적절하게 증식될 이미지 선정)는 0, 모든 이미지를 똑같이 증식하려면 1
    CC_NUM = (int)디폴트(=모든 카테고리 사용)는 0, 아니면 1과 2 중에 골라서 기재
    OUT_CATEGORIES = (list)디폴트는 None, CC_NUM이 1 또는 2일 경우에 사용하지 않을 카테고리 id 기재

    RANGE_MIN = (int)증식의 최소 기준 수량
    APPLY_CNT = (int)한 기법당 최대 몇 장까지 허용할 것인지 결정
    # ["gamma", "cutout", "mirroring", "crop", "rotation", "gaussian", "elastic"]
    TOOL_NAME_LIST = (list)사용할 기법들을 list 형식으로 기재
    POST_TOOL_NAME_LIST = (list)증식된 이미지에 랜덤하게 들어갈 기법 기재하고 없으면 []

    MAX_OR_ALL = (str)증식되는 오브젝트 수량을 RANGE_MIN에 딱 맞출거면 MAX 아니면 ALL
    # segmentation 이 [box] 인 데이터는 MASK_ON = False 로 진행해야 함
    # ["crop", "rotation", "elastic"] 에만 적용
    MASK_ON = (boolean)False
```
