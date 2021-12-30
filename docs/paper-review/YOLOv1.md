# YOLOv1
---
[//]: <> (large, Large, LARGE, huge, HUGE, https://jsfiddle.net/8ndx694g/)
## Abstract
  Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. 
  Compared to state-of-the-art detection systems, YOLO makes more localization errors but is less likely to predict false positives on background. 
  Finally, YOLO learns very general representations of objects. It outperforms other detection methods, including DPM and R-CNN, when gener- alizing from natural images to other domains like artwork.

<p align="center"><img width="500" alt="Figure 1: The YOLO Detection System" src="https://github.com/AshbeeKim/AshbeeKim/blob/master/.github/image/YOLOv1_1.jpeg"/></p>

<li>resizes the input image to 448 * 448</li>
<li>runs a single convolutional network on the image</li>
<li>thresholds the resulting detections by the model's sonfidence</li>

## 3 points should check
  1. 빠른 처리 속도. 회귀 문제로, 복잡한 파이프라인을 작성할 필요가 없음. 이미지 처리 속도는 초당 45프레임(no batch processing, Titan X GPU). fast version(GPU로 봐야하는지, 최근 YOLO*모델로 해석해야하는지 보류)은 150 fps*를 넘음. 이는 스트리밍 비디오에서 처리 지연 속도는 25ms 미만이란 뜻.

</br><p align="center"><img src="https://render.githubusercontent.com/render/math?math=\huge%20YOLO%7B%5Cgeqq%7D2*mAP"/>of other real-time systems</p></br>

  2. 높은 예측 설득력. (sliding window와 region proposal-based techniques와 다르게) YOLO는 train과 test에서의 전체 이미지를 보기 때문에, 즉시 class별 특성을 통해 문맥상 정보(contextual information)를 인코딩한다. 탐지 방법 중 최고인 Fast R-CNN*(2016년 기준), 전체 이미지(the larger context)를 볼 수 없기 때문에, 이미지의 객체도 배경으로 처리하는(patches) 실수가 있다.

<div align="center">
  <p align="left"><i align="left">the number of background errors,</i></p></br>
  <img src="https://render.githubusercontent.com/render/math?math=\huge%20YOLO%5Cleqq%5Cfrac%7BFast%20R-CNN%7D%7B2%7D"/></div></br>

  3. 일반화 할 수 있는(generalizable) 대표 특성 학습. wide margin으로 DPM이나 R-CNN과 같은 상위 탐지 모델을 능가함. new domains나 unexpected inputs을 적용해도, 일반화 가능성이 높기 때문에 거의 깨지지 않음.

</br>
  YOLO(v1 기준)는 여전히 최신 기술(state-of-the-art) 중 정확도 측면에서 뒤쳐짐. 이미지 내 객체를 빨리 구별이 가능하지만, 특히 작은 객체 등의 몇몇 객체를 정확하게 초점을 맞추는 것(localize)은 어려워 함. training, testing 코드는 오픈소스이며, 여러 pretrained models도 다운 가능하다고 적혀있음.
</br></br>

[their demo project webpage](https://pjreddie.com/darknet/yolo/)
</br>
<p>
* fps : frames per second</br>
* <a href="https://arxiv.org/pdf/1506.02640v5.pdf">YOLO</a> : You Only Look Once:Unified, Real-time Object Detection</br>
* <a href="https://arxiv.org/pdf/1504.08083v2.pdf">Fast R-CNN</a> : Fast Regions with CNN Features</br>
* <a href="https://arxiv.org/pdf/1409.5403v2.pdf">DPM</a> : Deformable parts models</br>
* <a href="https://arxiv.org/pdf/1311.2524v5.pdf">R-CNN</a> : Regions with CNN Features</br>
</p>

## Unified Detection

<p align="center"><img width="500" alt="The Model- YOLOv1" src="https://github.com/AshbeeKim/AshbeeKim/blob/master/.github/image/YOLOv1_2.png"/></p>

* divides the image into an S × S grid
* predicts B bounding boxes for each grid cell
* predicts confidence for those boxes, and C class probabilities <- be encoded as an S × S × (B ∗ 5 + C) tensor

[//]:<> (단일 neural network에서 객체 탐지 요소를 분할, 전체 이미지에서 얻은 특성을 사용해 각각의 bounding box 에측, 한 이미지의 유사도를 구하기 위해 모든 클래스_훈련+검증/실시간_ 아우르는 모든 bounding box를 예측, 높은 정확성을 유지하면서도 훈련 시작부터 실시간 데이터 속도까지 사용가능하도록 설계됨)

[//]: <> (input image를 간격을 S로 해서 나눔. 객체의 중점이 하나의 격자 공간_grid cell_에 있다면, 객체 탐지가 가능함. 각각의 격자 공간은 B bounding boxes와 박스의 신뢰도?_confidence scores stands for?_를 예측. 이 신뢰도는 객체를 박스에 포함하는 모델이 어떻게 설득력을 가지는지와 예측한 박스로 도출한 것을 정확히 할 것인지를 반영)


</br><p align="left"><img src="https://render.githubusercontent.com/render/math?math=\large%20Confidence%3DPr(Object)*IOU_%7Bpred%7D%5E%7Btruth%7D"/> 로 정의한다면,</p>
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\large%20%5Cbegin%7Bcases%7D%5Cmathrm%7BConfidence%7D%3D%5C%3B%5C%3B%5C%3B0%5C%3B%5C%3B%5C%3B%5C%3B%5C%3BIf%5C%3Bobject%5C%3Bexists%5C%3Bin%5C%3Bthat%5C%3Bcell%5C%5C%5C%5C%5Cmathrm%7BConfidence%7D%3DIOU%5C%3B%5C%3Bbetween%5C%3Bthe%5C%3Bpredicted%5C%3Bbox%5C%3Band%5C%3Bthe%5C%3Bground%5C%3Btruth%5Cend%7Bcases%7D"/></p></br>

### 5 predictions for each BBox
- (x, y) : 격자 공간의 가장자리와 관련이 높은 박스의 중점
- (w, h) : 전체 이미지 중 관련도가 높다고 예측된 가로, 세로값
- c(confidence) : <img src="https://render.githubusercontent.com/render/math?math=IOU_%7Bpred%7D%5E%7Btruth%7D">

[//]:<> (각각의 격자 공간은 $\mathrm{Pr}\lgroup{Class_i|Object}\rgroup$_클래스별 조건부 확률_도 예측, 이 확율이 격자 공간이 객체를 포함하는지에 대한 조건이 됨_weights나 feature information이라고 봐도 되는 부분?_. 격자 공간당 클래스별 확률은 하나씩만 예측가능._BBox수랑 무관_. 검증 시 클래스별 조건부확률 * 개별 box 얘상 정확도?)

</br>
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=%5CLARGE%20Pr(Class_i%7CObject)%5C%3B*%5C%3BPr(Object)%5C%3B*%5C%3BIOU_%7Bpred%7D%5E%7Btruth%7D%5C%3B%3D%5C%3BPr(Class_i)%5C%3B*%5C%3BIOU_%7Bpred%7D%5E%7Btruth%7D"/></p></br>

[//]:<> (위의 수식은 박스별 특정 클래스 신뢰도?를 구함. 그 점수는 박스 내 클래스의 확률과 해당 객체와 예측 박스의 적합도로 인코딩됨.)

</br>
<p align="left"><img src="https://render.githubusercontent.com/render/math?math=%5Clarge%7BFor%5C%3Bevaluationg%5C%3BYOLOv1%5C%3Bon%5C%3BPASCAL%5C%3BVOC%2C%7D"></p></br>
<p margin-left="50px">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://render.githubusercontent.com/render/math?math=%5Chuge%20S%20%3D%207%2C"></p>
<p margin-left="50px">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://render.githubusercontent.com/render/math?math=%5Chuge%20B%20%3D%202%2C"></p>
<p margin-left="50px">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://render.githubusercontent.com/render/math?math=%5Chuge%20C%20%3D%2020%5C%3B%5C%3B%5C%3B%5Ctiny%7B(20%5C%3Blabeled%5C%3Bclasses)%7D%2C"></p>
<p margin-left="50px">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<img src="https://render.githubusercontent.com/render/math?math=%5Chuge%20final%5C%3Bprediction%3D7%5C%3B*%5C%3B7%5C%3B*%5C%3B30%5C%3B%5C%3B%5C%3B%5Csmall%7Btensor%7D%0A"></p>

<p align="center"><img width="500" alt="The Architecture" src="https://github.com/AshbeeKim/AshbeeKim/blob/master/.github/image/YOLOv1_3.png"/></p>

[//]:<> (Our detection network has 24 convolutional layers followed by 2 fully connected layers. Alternating 1 × 1 convolutional layers reduce the features space from preceding layers. We pretrain the convolutional layers on the ImageNet classification task at half the resolution_224 × 224 input image_ and then double the resolution for detection.)

</br>
<p>
* IOU : intersection over union
</p>


