# Waste Recycling Image Segmentation

## 🥼 자율연구 - CNN

### 📖 Notes & Announcements 📢
* 💡 대회 내 데이터 접근 경로 변경으로 인해, 내부 구조 수정했습니다.
* 💡 각 case에 맞게 현재 구조 및 모델 수정 중에 있습니다.
* 🏆 2021 폐플라스틱 객체검출 예측 경진대회 🥉


### 🐋 권장 서버 환경

* **OS : ubuntu 18.04**
* **CUDA : 11.1.1**
* **Python : 3.7.7^**


### 🏗️ 구조
```
.
├── assets
│   ├── data
│   ├── ...
│   ├── mask
│   └── weights
├── config
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


### 👥 Crews

* [YoungpyoRyu](https://github.com/Youngpyoryu)
* [Ashbee Kim](https://github.com/AshbeeKim)
* [Park jeong yeol](https://github.com/qkrwjdduf159)
* [Yun yoseob](https://github.com/yunyoseob)
* [kim-hyun-ho](https://github.com/kim-hyun-ho)
* [Yonje Olivia Choi](https://github.com/oliviachchoi)

💬 이름을 클릭하면, 각자의 프로필을 확인할 수 있습니다.

💬 If you interested in us, click name to check our profiles.

</br>

### 📑 Papers

🗒️ 모델 구현을 위해, 전원 R-CNN, Fast R-CNN, Faster R-CNN은 확인했습니다.

🗒️ 하기의 논문은 현재 고려 중이거나, 검토 중인 논문입니다.

* [Mask-RCNN and U-net Ensembled](https://paperswithcode.com/paper/mask-rcnn-and-u-net-ensembled-for-nuclei)
* [MMDetection](https://paperswithcode.com/paper/mmdetection-open-mmlab-detection-toolbox-and)
* [U-Net++](https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical)
* [Deep-Net](https://paperswithcode.com/paper/semantic-image-segmentation-with-deep)

([🗄️ paper reviews](https://github.com/Proj-Caliber/Waste-Recycling-Image-Segmentation/tree/develop/docs/paper-review) 혹은 [🗄️ tutorials](https://github.com/Proj-Caliber/Waste-Recycling-Image-Segmentation/tree/develop/docs/tutorials) 를 참고하시면 현재까지 해당 프로젝트로 진행했던 부분을 확인할 수 있습니다.)

</br>

---

## 📜 LICENSE

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/), and the underlying source code used to format and display that content is licensed under the [MIT license](https://github.com/github/choosealicense.com/blob/gh-pages/LICENSE.md).

![CC BY-SA 4.0](http://i.creativecommons.org/l/by-sa/4.0/88x31.png)

![]()
