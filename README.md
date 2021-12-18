# Waste Recycling Image Segmentation


## 🥼 자율연구 - Plastic Segmentation

```
.
├── assets
│   ├── data
│   │   ├── test
│   │   │   ├── annotations
│   │   │   └── image
│   │   └── train
│   │       ├── annotation
│   │       └── image
│   └── mask
│       ├── test
│       └── train
├── config
│   ├── detection
│   ├── segmentation
│   └── transformer
└── docs
    ├── paper-review
    ├── papers
    └── tutorials
```

### 데이터 접근 경로

```
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


### 📌 Rules

* 대회 관련 이미지 및 json 파일은 공유가 불가능합니다.

  ```
  # .jpg, .jpeg, 그리고 .json 파일은 .gitignore에 추가했습니다.
  # 만약 커스텀한 .json 파일을 추가하고 싶다면, 아래의 명령어로 업로드가 가능합니다.

  git add {file path or name}.json -f
  git commit
  ```
* master, develop에 push를 지양합니다.
* 새로운 feature 생성 시, feature의 특징을 설명하는 이름을 branch로 생성합니다.

  ```
  # git branch로 생성하는 방법도 있지만, git checkout 사용을 권장합니다.

  git checkout -b {new branch name}
  ```

💡 대회 내 데이터 접근 경로 변경으로 인해, 내부 구조 수정했습니다.

---

## LICENSE

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-sa/4.0/), and the underlying source code used to format and display that content is licensed under the [MIT license](https://github.com/github/choosealicense.com/blob/gh-pages/LICENSE.md).

![CC BY-SA 4.0](http://i.creativecommons.org/l/by-sa/4.0/88x31.png)

![]()
