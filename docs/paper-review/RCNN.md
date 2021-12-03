---------------------------------------------------------------------------------
### Computer Vision task
1. Classification
2. Classification + Localization
3. Object Detection
4. Instance Segmentation
---------------------------------------------------------------------------------
<p align = "center"> 
<image src = 'https://www.researchgate.net/publication/340681873/figure/fig1/AS:881351528022016@1587141987049/Classification-object-detection-and-instance-segmentation-9.png'>
</p>
- classification : Single object에 대해서 object의 클래스를 분류하는 문제
- Classification + Localization : Single object에 대해서 object의 위치를 bounding box(BB)로 찾고 (localization) + 클래스를 분류하는 문제(Classification)
- object detection : 다중 물체에서 각각의 object에 대해 Classification + Localization을 수행하는 것.
- Instance Segmentation : Object Detection과 유사하지만, 다른 점은 object의 위치를 Bounding box(BB)가 아닌 실제  edge로 찾는 것.



--------------------------------------------------------------------------------
본격적으로 Object Detection 모델에 대해 살펴보도록 하겠습니다. 어떤 논문을, 어떤 순서에 따라 읽어야할지 고민하던 중, hoya님이 작성하신 2014~2019년도까지의  [Object Detection 논문 추천 목록](https://github.com/hoya012/deep_learning_object_detection)을 보게 되었습니다.

<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdsPLzF%2FbtqOhOxs7rM%2FKL50KxObME0vjKpMkoqHuK%2Fimg.png'>

**'빨간색'**은 꼭 봐야하는 논문입니다.

호야님의 순서와

https://paperswithcode.com/methods/category/convolutional-neural-networks

이 목록들을 살펴보면서 앞으로 글을 작성하려고 합니다.


---------------------------------------------------------------------------------
object detection에는 1-stage detector,2-stage detector가 있다.
<p align = "center"> 
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F4Fi2X%2FbtqCWbZjit2%2FsN9Ba7jKxiVI0h4S5InzMk%2Fimg.png'>
그림 출처 : https://ganghee-lee.tistory.com/35
</p>

2-stage를 한눈에 볼 수 있는 그림.

- 알고리즘 및 네트워크를 통해 물체가 있을만한 영역을 우선 뽑아 내는 과정을 거친다.
- 이 영역을 ROI(region of Interest)라고 한다.


1-stage detector의 대표적인 모델은 YOLO(You Only Look Once) 계열이 있다.

<p align = "center"> 
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fds1hoJ%2FbtqCX8tXMTh%2FJlGldm3aTsGzwiratKhbqK%2Fimg.png'>
</p>

- 1-stage는 2-stage와 다르게 ROI 영역을 추출하지 않고, 전체 image에 대해서, convolution network로 classification, localization을 수행한다.
- 속도 측면에서는 장점이 매우 많지만, image에서 여러 물체가 섞여 있다면 정확도를 떨어진다.


### # Rich feature hierarchies for accurate object detection and semantic segmentation(RCNN)

---------------------------------------------------------------------------------


논문의 저자들은 VOC 2012 데이터를 기준으로 기존의 방법들보다 mAP(mean average precision)이 30%이상 향상된 더 간단하고 확장 가능한 detection 알고리즘인 R-CNN을 소개한다.

또한 2가지 인사이트에 대해 설명한다.

1.  localize와 segementation을 위해 bottom-up(상향식)방식의 region proposals에 CNN을 적용했다.
2.  labeled data가 부족할 때, 보조작업(auxiliary task)를 supervised pre-training과 뒤를 이은 domain-specific fine-tuning을 통해 상당한 성능 향상을 이뤘다.

R-CNN이란 이름은 Regions with CNN features로, 그렇게 불리는 이유는 CNN과 Region proposals를 결합했기 때문이다.

R-CNN은 sliding-window 방식과 CNN 구조를 사용한 그 당시 최근에 제안된 OverFeat 보다 좋은 성능을 보이는 것을 확인했다.

- 성능 : Pascal Voc 2010을 기준으로 53.7%이며, 이미지 한 장에 CPU로는 47초, GPU로는 13초가 걸림.

* MAP(mean average preicison) 참고 : https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

- R-CNN이 등장하기 전 HOG와 Shift를 활용한 object detection 성능은 몇 년 동안 정체 되어 있었습니다.

- HOG과 Shift 참고) https://darkpgmr.tistory.com/116 "https://darkpgmr.tistory.com/116"

<p align = "center"> 
<img src = 'https://www.researchgate.net/profile/Neeraj-Bokde/publication/341099304/figure/fig4/AS:888908552359938@1588943722119/RCNN-architecture-17.ppm'>
</p>

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

<p align = "center"> 
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbgy0Br%2FbtqBaBRS4jL%2F28K3ONkBIQsO8IxKnCCv7k%2Fimg.png'>
<center>R-CNN 구조(1.Region Proposal)</center>
</p>

R-CNN에서는 가장 먼저 Region Proposal 단계에서 '물체가 있을 법한 영역'을 찾는다. 이는 기존의 Sliding window 방식의 비효율성을 극복하기 위한 것이다. 

**Sliding window 란?**
- 이미지에서 물체를 찾기 위해 window의 (크기, 비율)을 임의로 마구 바꿔가면서 모든 영역에 대해서 탐색하는 것.
<p align = "center"> 
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcMCgBR%2FbtqA6EB3JZX%2F59Ow9gEmwyMxA1swoHk9qK%2Fimg.png'>
<center>  < sliding window> 좌 : 모든 영역에 대해 탐색 / 우 : 크기와 비율을 변형></center>
</p>

- 임의의 (크기, 비율)로 모든 탐색하는 것은 너무 느리다. 이를 극복하기 위해 R-CNN에서는 이 비효율성을 극복하기 위해 Selective search 알고리즘을 사용.

**Selective Search**
<p align = "center"> 
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FMMlO6%2FbtqA7pEJsfi%2F4fLKHSxIkKJ8tEaFvKQ651%2Fimg.png'>
<center>  Selective Search</center>
</p>

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
<p align = "center"> 
<img src = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcVwCdl%2FbtqA9BLoE49%2FTL94t2Kdy745q9pBCYZlq0%2Fimg.png'>
<center>  RCNN 구조(2.CNN)</center>
</p>

Warp작업을 통해 region proposal 모두 224x224 크기로 되면 CNN 모델에 넣는다.

R-CNN에서는 CNN arichitecture를 AlexNet을 이용했다. 이 때, classification dataset을 이용해 pre-trained된 AlexNet 구조를 이용한다. 이 후 Domain Specific fine-tuning을 통해 CNN을 다시 학습을 시킨다. 이렇게 학습된 CNN은 region proposals 2000개의 각각의 227X227 image를 입력받아 4096-dimensional feature vector를 추출.

**Domain specific fine-tuning**
<p align = "center"> 
<img src = 'https://images.velog.io/images/skhim520/post/98d25442-c210-46e7-9991-19233c8af219/image.png'>
</p>

2000장의 region proposals와 ground-truth box의 **IoU(Intersection of Union)**을 비교하여 IoU가 0.5보다 큰 경우 positive samples, 0.5보다 작은 경우 negative samples로 나눈다. 이렇게 sample을 나눴을 때, ground truth만 positive sample로 정의할 때 보다 30배 많은 학습데이터를 얻을 수 있다. 많은 데이터를 통해 overfitting을 방지한다. Positive sample는 객체가 포함되어 있는 sample을 의미하고, negative sample은 객체가 포함되지 않은 배경 sample을 의미한다. 이렇게 나눈 후 positive sample 32개 + negative sample 96개 = 128개의 이미지로 이루어진 하나의 미니 배치를 만든다.  
이렇게 생성된 배치들을 이용해 fine-tuning을 진행한다. fine-tuning을 하기 위해서 기존의 pre-trained된 AlexNet의 마지막 softmax layer를 수정해서 N+1 way classification을 수행하게 한다. 이때, N은 R-CNN에서 사용하는 dataset의 객체들의 종류의 개수이고, 1을 더해준 이유는 배경인지 판단하기 위해서이다. SGD를 통해 N+1 way classification을 수행하면서 학습된 CNN 구조는 domain-specific fine-tuning을 이룬다.  
마지막의 N+1 way classification을 위해 수정한 softmax layer는 R-CNN 모델 사용시 사용하지 않는다. 왜냐하면 softmax layer는 fine-tuning을 위해 사용한 것이고, 원래 R-CNN에서 CNN 구조의 목표는 4096-dimensional feature vector를 추출하는 것이기 때문이다.

**IOU(Intersection of Union)**
<p align = "center"> 
<img src = 'https://images.velog.io/images/skhim520/post/d3aa6530-bf4a-4678-a1a6-87adf9ae2e63/image.png'> 
</p>




--------------------------------
## <3. support vector Machine>

<p align = "center"> 
<img src= 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHTaEr%2FbtqA9BxS2bV%2FkQJvYDyBDzpKY9pwVjegW1%2Fimg.png'> 
</p>

CNN 모델로부터 feature가 추출되면 Linear SVM을 통해 Classification을 진행한다. 위에서 설명했듯이, Classifier로 Softmax보다 SVM이 더 좋은 성능이 보였기 때문에 SVM을 채택함.
SVM은 CNN으로 부터 추출된 각각의 feature vector들의 점수를 class별로 매기고, 객체인지 아닌지, 객체라면 어떤 객체인지 등을 판별하는 역할을 하는 Classifier이다.

<p align="center">
<img src ='https://images.velog.io/images/skhim520/post/35fef43c-8e70-4541-98c7-051cad84406a/image.png'>
</p>

2000장의 region proposals에서 fine-tuning때와는 다르게 ground truth box만을 positive sample, IoU 값이 0.3보다 작은 것은 negative sample로 지정한다. 이때, IoU값이 0.3보다 큰 경우 무시한다. 이때 0.3은 gird search를 통해 찾은 값이다. 이후는 fine-tuning과 마찬가지로 positive sample 32개 + negative sample 96개 = 128개의 미니배치를 구성한 후 fine-tuning된 AlexNet에 입력하여 4096 dimensional feature vector를 추출한다. 추출된 벡터를 이용해 linear SVMs를 학습한다. SVM은 2진 분류를 수행하므로 분류하려는 객체의 종류만큼 SVM이 필요하다. 학습이 한 차례 끝난 후,  **_hard negative mining_**  기법을 적용하여 재학습을 수행한다.

R-CNN에서는 단순히 N-way softmax layer를 통해 분류를 진행하지 않고, SVMs를 이용해 분류를 한다. 이는 SVM을 사용했을 때 성능이 더 좋기 때문이다. 성능 차이의 이유를 논문의 저자들은 positive sample을 정의할 때 SVM을 학습시킬 때 더 엄밀하게 정의한다는 점과 SVM이 hard negative를 이용해 학습하기 때문이라고 서술했다.

linear SVM에서는 output으로 class와 confidence score를 반환한다.

**hard negative mining**
<p align="center">
<img src ='https://images.velog.io/images/skhim520/post/7bb64312-21f8-44f7-8217-cc72b1a1b892/image.png'>
</p>

이미지에서 사람을 탐지하는 경우 사람은 positive sample이 되고, 그 외의 배경은 negative sample이 된다. 이때, 모델이 bounding box를 배경이라고 예측하고 실제로 배경인 경우 true negative sample라고 한다. 반면에 모델이 사람이라고 예측했지만, 실제로 배경인 경우 false positive sample에 해당한다.

객체 탐지 시, positive sample보다 negative sample이 더 많은 클래스 불균형 때문에 모델은 주로 false positive 오류를 주로 범하게 된다. 이러한 문제를 해결하기 위해 처음 linear SVMs를 학습시킬 때의 false positive sample들을 epoch마다 학습 데이터에 추가하여 학습을 진행한다. 이를 통해 모델이 강건해지고, false positive 오류가 줄어든다.

--------------------------------

### <3.1 Bounding Box Regression>

selective search 알고리즘을 통해 얻은 객체의 위치는 부정확할 수 있다. 이런 문제를 해결하기 위해 객체의 위치를 조절해주는 Bounding box regressor가 있다.

N개의 training pair인  ${(P^{i},G^{i})}_{i=1,{\dots},N}$​  에 대해  $P^{i}  = (P_{x}^{i}​,P_{y}^{i}​,P_{w}^{i}​,P_{h}^{i}​)$는 해당 region에 대한 추정값으로 각각은 region 중심의 x,y좌표와 width와 height를 나타내고, 이에 대응되게  $G^{i}  = (G_{x}^{i}​,G_{y}^{i}​,G_{w}^{i}​,G_{h}^{i}​)$은 해당 region에 대한 ground truth이다.

<p align="Center">
<img src ='https://media.vlpt.us/images/skhim520/post/5d57ac72-364d-470b-96d2-174c7f1db113/image.png'>
</p>

위의 식을 최적의 $w_a$​를 찾고 싶다. (a=x,y,w,h, 별 표기가 어려워 a로 대치)  

${\hat {w​}}_{a}$​: 학습되는 가중치  

${\phi}_{5}​(P^i)$: $P^i$에 해당하는 feature vector. 여기서 feature vector는 fine-tuning된 CNN의 
output  

${\lambda}$: ridge regression을 위한 상수(논문에서는 1000 사용)  

$t_{a}^{i}$​는 밑을 참고

<img src='https://media.vlpt.us/images/skhim520/post/f4f0a7cc-4a6a-4dd7-89a8-5ee691cfa7f2/image.png'>

위의 식을 통해 찾은  $w_{a}$​를 이용해  $d_{a}$​라는 transformation 함수를 정의할 수 있다.  
즉, 다음과 같이 정의할 수 있다.  $d_a (P)=  w_a^{T}​ {\phi}_5 (P)$​

이를 통해서 ground truth  G의 추정값인  ${\hat {G}}$를 다음과 같이 추정할 수 있다.

<img src='https://media.vlpt.us/images/skhim520/post/73114071-825e-4818-b748-e5fc04060f23/image.png'>

위와 같이 추정하는 이유는 hard negative mining을 참고해보면 이해하기 쉬울 것이다.

위와 같은 training pair를 정의할 때, P는 ground truth와 IoU 값이 0.6이상인 경우만 사용한다. 왜냐하면 겹치는 영역이 많이 작을 경우, 학습의 어려움이 존재하기 때문이다.

---------------------------------------

#### Non Maximum Supreesion

R-CNN을 통해 얻게 되는 2000개의 bounding box를 전부 다 표시할 경우우 하나의 객체에 대해 지나치게 많은 bounding box가 겹칠 수 있다. 따라서 가장 적합한 bounding box를 선택하는 Non maximum supression 알고리즘을 적용한다.
<img src = 'https://images.velog.io/images/skhim520/post/4e3627c5-1c63-481f-b635-5e0246b0c687/image.png'>

non maximum supression 알고리즘은 다음과 같다.

1.  bounding box별로 지정한 confidence scroe threshold 이하의 box를 제거한다.
2.  남은 bounding box를 confidence score에 따라 내림차순으로 정렬한다. 그 다음 confidence score가 높은 순의 bounding box부터 다른 box와의 IoU값을 조사하여 IoU threshold 이상인 box를 모두 제거한다.
3.  2의 과정을 반복하여 남아있는 box만 선택한다.

## 단점

1.  이미지 한 장당 2000개의 region proposal을 추출하므로 학습 및 추론의 속도가 느리다.
2.  3가지 모델을 사용하다보니 구조와 학습 과정이 복잡하다. 또한 end-to-end 학습을 수행할 수 없다.



출처 : 
R-CNN 논문([Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524))
https://velog.io/@skhim520/R-CNN-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
https://herbwood.tistory.com/5
https://ganghee-lee.tistory.com/35


RCNN 구현 참고 : https://github.com/rbgirshick/rcnn