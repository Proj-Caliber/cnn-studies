---------------------------------------------------------------------------------
### Computer Vision task
1. Classification
2. Classification + Localization
3. Object Detection
4. Instance Segmentation
---------------------------------------------------------------------------------
<image src = 'https://www.researchgate.net/publication/340681873/figure/fig1/AS:881351528022016@1587141987049/Classification-object-detection-and-instance-segmentation-9.png'>

- classification : Single object에 대해서 object의 클래스를 분류하는 문제
- Classification + Localization : Single object에 대해서 object의 위치를 bounding box(BB)로 찾고 (localization) + 클래스를 분류하는 문제(Classification)
- object detection : 다중 물체에서 각각의 object에 대해 Classification + Localization을 수행하는 것.
- Instance Segmentation : Object Detection과 유사하지만, 다른 점은 object의 위치를 Bounding box(BB)가 아닌 실제  edge로 찾는 것.

---------------------------------------------------------------------------------
object detection에는 1-stage detector,2-stage detector가 있다.

<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F4Fi2X%2FbtqCWbZjit2%2FsN9Ba7jKxiVI0h4S5InzMk%2Fimg.png'>

그림 출처 : https://ganghee-lee.tistory.com/35

2-stage를 한눈에 볼 수 있는 그림.

- 알고리즘 및 네트워크를 통해 물체가 있을만한 영역을 우선 뽑아 내는 과정을 거친다.
- 이 영역을 ROI(region of Interest)라고 한다.


1-stage detector의 대표적인 모델은 YOLO(You Only Look Once) 계열이 있다.

<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fds1hoJ%2FbtqCX8tXMTh%2FJlGldm3aTsGzwiratKhbqK%2Fimg.png'>

- 1-stage는 2-stage와 다르게 ROI 영역을 추출하지 않고, 전체 image에 대해서, convolution network로 classification, localization을 수행한다.
- 속도 측면에서는 장점이 매우 많지만, image에서 여러 물체가 섞여 있다면 정확도를 떨어진다.


### # Rich feature hierarchies for accurate object detection and semantic segmentation(RCNN)

---------------------------------------------------------------------------------

- 성능 : Pascal Voc 2010을 기준으로 53.7%이며, 이미지 한 장에 CPU로는 47초, GPU로는 13초가 걸림.

* MAP(mean average preicison) 참고 : https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

- R-CNN이 등장하기 전 HOG와 Shift를 활용한 object detection 성능은 몇 년 동안 정체 되어 있었습니다.

- HOG과 Shift 참고) https://darkpgmr.tistory.com/116 "https://darkpgmr.tistory.com/116"

<img src = 'https://www.researchgate.net/profile/Neeraj-Bokde/publication/341099304/figure/fig4/AS:888908552359938@1588943722119/RCNN-architecture-17.ppm'>

 - "R-CNN Process"
 1. Image를 입력받는다.
 2. Selective search알고리즘에 의해 regional proposal output 약 2000개를 추출한다. 추출한 region proposal output을 모두 동일 input size로 만들어주기 위해 warp해준다.
 (*왜 동일 input size로 만들어 줄까? : Convolution Layer에는 input size가 고정이지가 않다. ->dataset마다 다름.)
 * 그러나 마지막 FC layer에서의 input size는 고정이므로 Convolution Layer에 대한 output size도 동일해야 함.
 * Convolution layer에 입력에서부터 동일한 input size로 넣어주어서 output size를 동일하게 하는 것.

3. 2000개의 warped image를 각각 CNN모델에 넣는다
4. 각각의 Convolution 결과에 대해 classification을 진행하여 결과를 얻는다.


* 이미지 와핑(image warping)
참고 : https://com24everyday.tistory.com/366

- 이미지의 형태로 바꾸는 것.

위의 과정들은 수행하기 위해 R-CNN은 세 가지 모듈로 나누어 놓음.
1. Region Proposal : "Object가 있을만한 영역"을 찾는 모듈(기존의 Sliding window방식의 비 효율성 극복)
2. Convolution Neural Network : 각각의 영역으로 부터 고정된 크기의 Feature Vector를 뽑아 냄. (고정된 크기의 output을 얻기 위해 warp 작업을 통해 크기를 찌그러뜨려서 동일 input size로 만들고 CNN에 넣는다.)
3. Support Vector Machine : Classification을 위한 선형 지도학습 모델
(* 왜 classifier로 Softmax를 쓰지 않고 SVM을 사용하였을까? : CNN fine-tuning을 위한 학습 데이터가 시기 상 많지 않아서 Softmax를 적용시키면 오히려 성능이 낮아져서 SVM을 사용하였음.)



## <1. Region Proposal(영역 찾기)>
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbgy0Br%2FbtqBaBRS4jL%2F28K3ONkBIQsO8IxKnCCv7k%2Fimg.png'>
<center>R-CNN 구조(1.Region Proposal)</center>

R-CNN에서는 가장 먼저 Region Proposal 단계에서 '물체가 있을 법한 영역'을 찾는다. 이는 기존의 Sliding window 방식의 비효율성을 극복하기 위한 것이다. 

**Sliding window 란?**
- 이미지에서 물체를 찾기 위해 window의 (크기, 비율)을 임의로 마구 바꿔가면서 모든 영역에 대해서 탐색하는 것.
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcMCgBR%2FbtqA6EB3JZX%2F59Ow9gEmwyMxA1swoHk9qK%2Fimg.png'>
<center>  < sliding window> 좌 : 모든 영역에 대해 탐색 / 우 : 크기와 비율을 변형></center>

- 임의의 (크기, 비율)로 모든 탐색하는 것은 너무 느리다. 이를 극복하기 위해 R-CNN에서는 이 비효율성을 극복하기 위해 Selective search 알고리즘을 사용.

**Selective Search**

<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMMlO6%2FbtqA7pEJsfi%2F4fLKHSxIkKJ8tEaFvKQ651%2Fimg.png'>
<center>  Selective Search</center>

1. 색상, 질감, 영역크기 등을 이용해 non-object-based segmentation을 수행한다.

이 작업을 통해 좌측 제일 하단 그림과 같이 많은 small segmented areas들을 얻을 수 있다.

2. Bottom-up 방식으로 small segmented areas들을 합쳐서 더 큰 segmented areas들을 만든다.

* bottom-up 방식이란?
- 이미지의 픽셀 단위부터 조금씩 파악하여 특징을 찾는 방식

* Top-down 
- 이미지 전체를 보고 그 이미지에서 Task에 걸맞는 특징을 찾는 방법.

참고 : https://woosikyang.github.io/Bottom-Up-and-Top-Down-Attention-for-Image-Captioning-and-VQA.html#:~:text=%EC%B2%AB%EB%B2%88%EC%A7%B8%EB%8A%94%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%A0%84%EC%B2%B4%EB%A5%BC,%EC%9D%B4%EB%A5%BC%20Bottom%2Dup%20%EC%9D%B4%EB%9D%BC%EA%B3%A0%20%ED%95%A9%EB%8B%88%EB%8B%A4.

3. (2)작업을 반복하여 최종적으로 2000개의 region proposal을 생성한다.

Selective search알고리즘에 의해 2000개의 region proposal이 생성되면 이들을 모두 CNN에 넣기 전에  
같은 사이즈로 warp시켜야한다. (CNN output 사이즈를 동일하게 만들기 위해 - For FC layer)
--------------------------------

## <2. Convolution Neural Network>
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcVwCdl%2FbtqA9BLoE49%2FTL94t2Kdy745q9pBCYZlq0%2Fimg.png'>
<center>  RCNN 구조(2.CNN)</center>

Warp작업을 통해 region proposal 모두 224x224 크기로 되면 CNN 모델에 넣는다.

여기서 CNN은 AlexNet의 거의 구조를 그대로 가져다 썼다.

최종적으로 CNN을 거쳐 각각의 region proposal로부터 4096-dimentional feature vector를 뽑아내고,

이를 통해 고정길이의 Feature Vector를 만들어낸다.
--------------------------------
## 3.< support Vector Machine>
<img src= 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHTaEr%2FbtqA9BxS2bV%2FkQJvYDyBDzpKY9pwVjegW1%2Fimg.png'> 