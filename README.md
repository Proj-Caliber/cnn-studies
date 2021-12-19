# Waste Recycling Image Segmentation

## 🥼 자율연구 - CNN(AI-Challenge)

### 👷 Rebuilders
* AI-Challenge ([Ashbee Kim](https://github.com/AshbeeKim))
* Plastics Segmentation ([Park jeong yeol](https://github.com/qkrwjdduf159))


### 📋 Guidelines
```
# 우선 작업을 진행할 폴더 생성
$ mkdir <feature name>
$ cd <feature name>

# 방법 1 : organization에서 바로 작업
$ git init
    # 만약 git init을 했을 때, 로컬 브랜치 명이 main, master라면
    $ git branch --move main(or master) <feature branch name>
$ git remote add origin git@github.com:Proj-Caliber/Waste-Recycling-Image-Segmentation.git
    # 만약 remote name을 origin이 아닌 다른 명칭으로 하고 싶다면
    $ git remote rename origin <new name>
$ git pull origin <remote branch name>
$ git push -u origin <remote branch name>
```
이후 진행한 작업은 commit 하고 $ git push origin <remote branch name>

작업이 전부 끝나면, develop 브랜치로 pull request 날리기
    
</br>

### 🐋 권장 서버 환경

* **OS : ubuntu 18.04**
* **CUDA : 11.1.1**
* **Python : ^3.7.7**


### 🏗️ 구조
```
.
├── assets
│   ├── data
│   ├── ...
│   ├── mask
│   └── weights
├── models
│   ├── detection
│   ├── segmentation
│   └── transformer
└── docs
    ├── paper-review
    ├── papers
    └── tutorials
```

### 📂 데이터 접근 경로
* case1
```
# 🧪 ai-challenge
./assets/data
├── train
│   └── metadata.json
│       ├── t3_0001.JPG
│       ├── ...
│       └── t3_0030.JPG
└── taco

# 대회 제공 샘플데이터 부족으로 공개된 데이터를 참고해 가중치 부여(./assets/data)
user@ubuntu-18.04: git fetch https://github.com/pedropro/TACO.git
user@ubuntu-18.04: cat readme.md
```


### 일정

| <h3 align="center">기간</h3> | <h3 align="center">대회 일정 및 진행 과정</h3>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                                                                                               <h3 align="center"> 비 고 </h3> |
| :----------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------: |
|                1주차                | **<h4><bold>30.Sep.Thu</bold></h4>** : <span style="color:gray;">대회 별 세부 문제 정의서 공지(대외비)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                       [paperswithcode :: CNN Overview](https://paperswithcode.com/methods/category/convolutional-neural-networks) </br>🎉🌱📚👷---🚧 |
|                2주차                | **<h4><bold>08.Oct.Fri</bold></h4>** : 온라인 기술워크샵 개최 및 샘플 데이터 공개                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |      [paperswithcode :: Object Dection Models Overview](https://paperswithcode.com/methods/category/object-detection-models) </br>🚧---📚📖📝🚚---🚧 |
|                3주차                | **<h4><bold>Paper Review</bold></h4>**[Youngpyoryu :: RCNN, Fast RCNN, Faster RCNN](https://github.com/Youngpyoryu/Waste-Recycling-Image-Segmentation/tree/master/paper-review)</br>[qkrwjdduf159 :: FPN(Feature Pyramid Network)](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/FPN(Feature%20Pyramid%20Network).ipynb)</br>[AshbeeKim :: YOLOv1](https://github.com/AshbeeKim/AshbeeKim/tree/master/Papers/YOLOv1.md)                                                                                                                                                                                  |                                                                  Customizing Back-bone Network ~</br>🚧---</br>🚀📝✏️📋📝👷🔧</br>---🚧 |
|                4주차                | **<h4><bold>Paper Review</bold></h4>**[qkrwjdduf159 :: LMM(Linear Mixed Model)](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/Linear%20Mixed%20Model.ipynb)</br>[Youngpyoryu :: MMdetection]</br>[qkrwjdduf159 :: Mask RCNN](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/Mask%20RCNN.ipynb) </br>[AshbeeKim :: YOLOv3]                                                                                                                                                                                                                                                                |                                                           Searching Image Sets for pre-train ~</br>🚧---</br>🚗👷✏️📋📝🚙👷🔨</br>---🚧 |
|                5주차                | **<h4><bold>Paper Review</bold></h4>**[AshbeeKim :: Cascade Mask RCNN]</br>[yunyoseob :: DL+RL: Approximation with ANN]</br>[qkrwjdduf159 :: U-Net](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/U-Net.ipynb)                                                                                                                                                                                                                                                                                                                                                                                      |                                                                                                        🚧---</br>🔧✏️📋⌨️🏗👷🔨</br>---🚧 |
|                6주차                | **<h4><bold>Code Review & Develop </bold></h4>** <span><i><s>[AshbeeKim :: YOLOv1v3](https://colab.research.google.com/drive/1Y4Mh9_x5YKbw2dJp0OoXRIlc3tpz-F3m?usp=sharing)</s></i></br><span><i>Pretrain&Tutorial </i><li>[Youngproryu](https://colab.research.google.com/drive/11qEpXYzLDsOyeVkdzb8vnfK1jdPFKCa3?usp=sharing)</li><li>[qkrwjdduf159](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/Model%20tutorial%20code/%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0.ipynb)</li><span><i>create annotations for sample Images</i><li>[yunyoseob]</li><li>[AshbeeKim]</li> |                            [pedropro TACO](https://github.com/pedropro/TACO)</br>💡🗃---</br>🚧---</br>🛠♻️🏷🏗🍱🍻🏷⌨️📌</br>---🚧 |
|                7주차                | **<h4><bold>08.Nov.Mon~09.Nov.Tue</bold></h4>** : 4차 대회 사전 테스트(환경설정)</br>**<h4><bold>10.Nov.Wed~12.Nov.Fri</bold></h4>** : 4차 3단계 대회 개최(변경 전)                                                                                                                                                                                                                                                                                                                                                                                                                              | **<h4><bold>12.Nov.Fri~</bold></h4>** : 4차 3단계 대회 재개(변경 후)</br>🚧---</br>🐳🛠🍎🐧➕:monocle_face:🐳🐛🔨</br>---🚧 |
|                8주차                | **<h4><bold>13.Nov.Sun</bold></h4>** : 4차 3단계 대회 종료 (~2pm)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |                                                                                                              🛂:adhesive_bandage:🚨:test_tube:---➖📝 |



### 규정

* 검출 조건
  * 모든 이미지는 PET+ 물 등 혼합재질, 찌그러지거나 파손된 비정형 생활 폐기물이 포함될 수 있음
  * 한 장의 이미지에는 단수의 폐기물 혹은 복수의 폐기물이 존재
  * 폐기물의 배경은 일상 생활 배경과 같이 고정되어 있지 않고 이미지마다 다를 수 있음
* 문제 데이터 사양
  * 다양한 해상도와 크기의 이미지
  * jpeg 포멧으로 제시
  * 총 6000장의 이미지가 문제로 제시되면, 참가팀은 이를 모두 처리하여 함
  * 복합 재질의 사물인 경우 핵심이 되는 사물만 검출
* 제출 형식 : 추론결과를 해당 트랙별 json형식으로 제출

📝 상세 정의서는 대외비라 공개가 불가합니다.

</br>

### 🚅 참고 문헌 및 Repo

* R-CNN > SPPNet > Fast R-CNN > Faster R-CNN
* MMDetection > Detectron
* Neck > FPN
* YOLO Family
* [Albumentation](https://github.com/albumentations-team/albumentations)
* [MMdetection](https://github.com/open-mmlab/mmdetection)

</br>

---

## LICENSE

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/), and the underlying source code used to format and display that content is licensed under the [MIT license](https://github.com/github/choosealicense.com/blob/gh-pages/LICENSE.md).

![CC BY-SA 4.0](http://i.creativecommons.org/l/by-sa/4.0/88x31.png)

![]()
