## Catboost를 이용한 잡케어 추천 알고리즘 경진대회_테이브팀 :blush:
범주형 변수에 강력한 성능을 보여주는 catboost를 이용하여 코드를 작성해 봤습니다. 
구글 코랩에서 작성하였고, GPU를 이용하여 학습 했습니다.
최종 예측 파일 threshold값을 조정해서 재현율을 끌어올려 LB 점수에서 효과를 봤습니다.

(아래코드는 국경원 요원님의 잡케어 추천 알고리즘 경진대회 [Private 8위 0.66203] | Catboost 코드를 참고하여 수정하였습니다.)
[참조 코드](https://dacon.io/competitions/official/235863/codeshare/3887?page=1&dtype=recent)

## 1.Library & Data Load

### 컴퓨터 환경
* 코랩환경에서 진행했습니다.
 
### 라이브러리 버전
* catboost : 1.0.4
* eli5 : 0.11.0
* optuna : 2.10.0
* numpy : 1.19.5
* pandas : 1.1.5
* sklearn : 1.0.1

### 학습 소요 시간
* optuna : 2시간 30분
* catboost : 5분???(거짓말)

### 실행 프로세스

1. 데이터 전처리
    + eli5_permutation feature importance 기반 데이터 전처리

2. Optuna
  * best parameter 추출

3. Catboost
  * 모델 적합

### 데이터 로드

## 2. Data Preprocess
- 속성_D_코드.csv,속성_L_코드.csv,속성_H_코드.csv를 학습데이터에 추가하기 위해 데이터 병합을 진행했고 Eli5에서 제외한 학습에 방해가 되는 코드는 학습 피쳐에서 제외했습니다.

- Eli5 패키지의 permutation feature importance로 catboostclassifier모델을 진행하는데 있어 제외할 컬럼을 미리 선정했습니다. 또한 cat_feature에 범주형 칼럼리스트를 만들어 학습에 용이하게 했습니다.

```py
import os
import sys
import platform
import random
import math
from typing import List ,Dict, Tuple

import pandas as pd
import numpy as np
 
import sklearn 
from sklearn.model_selection import StratifiedKFold , KFold
from sklearn.metrics import f1_score 

from catboost import Pool,CatBoostClassifier

print(f"- os: {platform.platform()}")
print(f"- python: {sys.version}")
print(f"- pandas: {pd.__version__}")
print(f"- numpy: {np.__version__}")
print(f"- sklearn: {sklearn.__version__}")
```

## 3. 모델링
Optuna로 Random Search를 통해 Catboost 최적의 파라미터를 사용하였습니다.
objective 함수의 param에 파라미터를 넣고, 구간을 넣으면 랜덤한 값으로 학습되며 f1-score값이 반환되는 함수입니다. "trial"에 반복 횟수를 작성하면 됩니다.


Optuna 주석처리로해서


Catboost 특성상 학습이 오래 걸리기 때문에 최적의 파라미터를 찾아 Best_params로 정의하였습니다.
```py
Best Params: {'iterations': 1422, 'objcetive': 'CrossEntropy', 'bootstrap_type': 'Bayesian', 'od_wait': 666, 'learning_rate': 0.9782109291187356, 'reg_lambda': 70.72533306533951, 'random_strength': 47.81900485462368, 'depth': 3, 'min_data_in_leaf': 20, 'leaf_estimation_iterations': 5, 'one_hot_max_size': 1, 'bagging_temperature': 0.07799233624102353}
```


모델링 전개과정 넣깅



## 4.결과값제출
train을 K-fold한 값의 평균을 구하다 보니 예측값의 극단값이 작아질 수 밖에 없습니다.

따라서 threshold를 조정해가며 최적의 threshold : 0.33792를 찾았습니당

```py
cols_drop = ["id","person_prefer_f","person_prefer_g" ,
             "person_prefer_d_3_attribute_d_m_contents_attribute_d_attribute_d_m", "person_prefer_h_3_attribute_h_l"]
```
