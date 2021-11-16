# Waste Recycling Image Segmentation

## AI-Challenge

### <b>서버 환경</b>

* **OS : ubuntu 18.04**
* **CUDA : 11.1.1**
* **Python : 3.7.7**

### 일정

|    <h3 align="center">기간</h3>    | <h3 align="center">대회 일정 및 진행 과정</h3> | <h3 align="center">비 고</h3> |
| :-------------: | :-------------------- | ------: |
|    1주차     | **<h4><bold>30.Sep.Thu</bold></h4>** : <span style="color:gray;">대회 별 세부 문제 정의서 공지(대외비)</span> | [paperswithcode :: CNN Overview](https://paperswithcode.com/methods/category/convolutional-neural-networks)</br>:tada::seedling::books::construction_worker:---:construction: |
|    2주차     | **<h4><bold>08.Oct.Fri</bold></h4>** : 온라인 기술워크샵 개최 및 샘플 데이터 공개 | [paperswithcode :: Object Dection Models Overview](https://paperswithcode.com/methods/category/object-detection-models)</br>:construction:---:books::open_book::pencil::truck:---:construction:|| |
|    3주차     | **<h4><bold>Paper Review</bold></h4>**[Youngpyoryu :: RCNN, Fast RCNN, Faster RCNN](https://github.com/Youngpyoryu/Waste-Recycling-Image-Segmentation/tree/master/paper-review)</br>[qkrwjdduf159 :: FPN(Feature Pyramid Network)](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/FPN(Feature%20Pyramid%20Network).ipynb)</br>[AshbeeKim :: YOLOv1](https://github.com/AshbeeKim/AshbeeKim/tree/master/Papers/YOLOv1.md) | Customizing Back-bone Network ~</br>:construction:---</br>:rocket::memo::pencil2::clipboard::pencil::construction_worker::wrench:</br>---:construction: |
|    4주차     | **<h4><bold>Paper Review</bold></h4>**[qkrwjdduf159 :: LMM(Linear Mixed Model)](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/Linear%20Mixed%20Model.ipynb)</br>[Youngpyoryu :: MMdetection]</br>[qkrwjdduf159 :: Mask RCNN](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/Mask%20RCNN.ipynb)</br>[AshbeeKim :: YOLOv3] | Searching Image Sets for pre-train ~</br>:construction:---</br>:car::construction_worker::pencil2::clipboard::pencil::blue_car::construction_worker::hammer:</br>---:construction: |
|    5주차     | **<h4><bold>Paper Review</bold></h4>**[AshbeeKim :: Cascade Mask RCNN]</br>[yunyoseob :: DL+RL: Approximation with ANN]</br>[qkrwjdduf159 :: U-Net](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/U-Net.ipynb) | :construction:---</br>:wrench::pencil2::clipboard::keyboard::building_construction::construction_worker::hammer:</br>---:construction: |
|    6주차     | **<h4><bold>Code Review & Develop</bold></h4>**<span><i><s>[AshbeeKim :: YOLOv1v3](https://colab.research.google.com/drive/1Y4Mh9_x5YKbw2dJp0OoXRIlc3tpz-F3m?usp=sharing)</s></i></span></br><span><i>Pretrain&Tutorial</i><li>[Youngproryu](https://colab.research.google.com/drive/11qEpXYzLDsOyeVkdzb8vnfK1jdPFKCa3?usp=sharing)</li><li>[qkrwjdduf159](https://github.com/qkrwjdduf159/Model-tutorial/blob/main/Model%20tutorial%20code/%ED%99%95%EC%9D%B8%ED%95%98%EA%B8%B0.ipynb)</li></span><span><i>create annotations for sample Images</i><li>[yunyoseob]</li><li>[AshbeeKim]</li></span> | [pedropro TACO](https://github.com/pedropro/TACO)</br>:bulb::card_file_box:---</br>:construction:---</br>:hammer_and_wrench::recycle::label::building_construction::bento::beers::label::keyboard::pushpin:</br>---:construction: |
|    7주차     | **<h4><bold>08.Nov.Mon~09.Nov.Tue</bold></h4>** : 4차 대회 사전 테스트(환경설정)</br>**<h4><bold>10.Nov.Wed~12.Nov.Fri</bold></h4>** : 4차 3단계 대회 개최(변경 전) | **<h4><bold>12.Nov.Fri~</bold></h4>** : 4차 3단계 대회 재개(변경 후)</br>:construction:---</br>:whale::hammer_and_wrench::apple::penguin::heavy_plus_sign::monocle_face::whale::bug::hammer:</br>---:construction: |
|    8주차     | **<h4><bold>13.Nov.Sun</bold></h4>** : 4차 3단계 대회 종료 (~2pm) | :passport_control::adhesive_bandage::rotating_light::test_tube:---:heavy_minus_sign::pencil: |

🧐 <p style="background:gray;">현재 링크가 걸려 있지 않은 부분은 차차 정리 및 공유가 된다면 업데이트할 예정입니다.</P>

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

## 기본 가이드라인

### Git

* master나 develop 브랜치
  * push 사용하지 않기(pull만 허용)
  * pushed된 코드에 관해 peer-review 남기기
  * pull request 전, 잠재적 충돌을 해결할 것
  * 병합 이후에는 로컬 저장소와 원격 특정 브랜치를 제거할 것
* 공통
  * pull request 전, 특정 브랜치 생성과 코드 규격, 스타일 등을 포함한 모든 것을 확인할 것
  * .gitignore를 활용할 것

[git 프로젝트 가이드라인 참고_1](https://github.com/huggingface/transformers)

[git 프로젝트 가이드라인 참고_2](https://github.com/elsewhencode/project-guidelines)

📝 요청 시 git commands 정리 진행하겠습니다.

### 고려 사항

* 서버 환경에서 구동이 되어야 하기 때문에, 가상 환경 혹은 가상 머신은 상기의 서버 환경과 맞춘 뒤 코드 작성
* 의존성을 잘 고려해서, requirements.txt와 .py 혹은 .ipynb내에 라이브러리 사용 버전에 대한 정보를 기재하기

[project_guideline(env)](https://github.com/Proj-Caliber/Waste-Recycling-Image-Segmentation/blob/master/project_guide.md)

📝 코드 가이드라인 작성자 추천받습니다~

### 시간 복잡도

```python
# 컴파일러나 비동기방식 사용을 통해 연산 시간 단축
# 컴파일러
Pypy
Numba
Cython(CPyhon과 다름)
# 비동기방식 <- 함수
Asyncio
Trio
# SIMD(ARM NEON) or OPENVINO를 사용하는 것도 연산 속도를 높이는 데 도움이 됨
```

## 🥼 자율연구

### 활용 데이터

```console
user@ubuntu-18.04: git fetch https://github.com/pedropro/TACO.git
user@ubuntu-18.04: cat readme.md
```

📝사용 방법은 readme.md에 적혀있습니다.(영어라서 저도 지금 당장 사용법을 정리하기엔 무리가 있습니다.)

### 🚅 참고 문헌 및 Repo

* R-CNN > SPPNet > Fast R-CNN > Faster R-CNN
* MMDetection > Detectron
* Neck > FPN > PANet > DetectorRS > BiFPN > NASFPN > AugFPN
* YOLO Family > SSD > RetinaNet 및 Focal Loss
* Model Scaling > EfficientNet > EfficientDet
* Cascade RCNN > Deformable Convolutional Networks(DCN) > Transformer
* YOLO v4 > M2Det > CornerNet
* [Albumentation](https://github.com/albumentations-team/albumentations)
* [MMdetection](https://github.com/open-mmlab/mmdetection)

---

## LICENSE

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/), and the underlying source code used to format and display that content is licensed under the [MIT license](https://github.com/github/choosealicense.com/blob/gh-pages/LICENSE.md).

![CC BY-SA 4.0](http://i.creativecommons.org/l/by-sa/4.0/88x31.png)

![]()
