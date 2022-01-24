## 2021년 잡케어 추천 알고리즘 경진대회_테이브팀

* 참조 코드
  * [catboost 모델](https://dacon.io/competitions/official/235863/codeshare/3887?page=1&dtype=recent) 


### 컴퓨터 환경
* CPU : 
* GPU : 
* RAM : 
* 운영체제 : 

### 라이브러리 버전
* catboost : 1.0.4
* eli5 : 0.11.0
* optuna : 2.10.0
* numpy : 1.19.5
* pandas : 1.1.5
* sklearn : 1.0.1

### 학습 소요 시간
* optuna : 2시간 30분
* catboost : 5분

### 환경 셋팅 및 실행 프로세스

1. 데이터 전처리
  * eli5_permutation feature importance 기반 데이터 전처리

2. Optuna
  * best parameter 추출

3. Catboost
  * 모델 적합


