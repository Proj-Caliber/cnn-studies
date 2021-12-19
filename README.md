# Waste Recycling Image Segmentation


## 🥼 자율연구 - CNN(Plastic Segmentation)

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


### 데이터 접근 경로
* case2
```
# 🧐 plastics segmentation
./assets/data
├── test
│   ├── annotations
│   │   ├── PE
│   │   ├── PET
│   │   ├── PP
│   │   └── PS
│   └── image
│       ├── PE
│       ├── PET
│       ├── PP
│       └── PS
└── train
    ├── annotation
    │   ├── PE
    │   ├── PET
    │   ├── PP
    │   └── PS
    └── image
        ├── PE
        ├── PET
        ├── PP
        └── PS
```


### ✨ 참고 문헌 및 Repo by Roles

* [YoungpyoRyu](https://github.com/Youngpyoryu)
  * [U-Net++](https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical)
  * [Deep-Net](https://paperswithcode.com/paper/semantic-image-segmentation-with-deep)
* [Ashbee Kim](https://github.com/AshbeeKim)
  * [Mask-RCNN and U-net Ensembled](https://paperswithcode.com/paper/mask-rcnn-and-u-net-ensembled-for-nuclei)
* [Park jeong yeol](https://github.com/qkrwjdduf159)
  * [MMDetection](https://paperswithcode.com/paper/mmdetection-open-mmlab-detection-toolbox-and) [Repository🗄️](https://github.com/open-mmlab/mmdetection)
  * [UNet](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical)
* [Yonje Olivia Choi](https://github.com/oliviachchoi)
  * [Faster R-CNN](https://paperswithcode.com/paper/faster-r-cnn-towards-real-time-object)
  * [Mask R-CNN](https://paperswithcode.com/paper/mask-r-cnn)

🗒️ 모델 구현을 위해 참고한 논문입니다.

💬 이름을 클릭하면, 각자의 프로필을 확인할 수 있습니다.

💬 If you interested in us, click name to check our profiles.


---

## LICENSE

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/), and the underlying source code used to format and display that content is licensed under the [MIT license](https://github.com/github/choosealicense.com/blob/gh-pages/LICENSE.md).

![CC BY-SA 4.0](http://i.creativecommons.org/l/by-sa/4.0/88x31.png)

![]()
