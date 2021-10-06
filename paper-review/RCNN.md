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

- 1-stage는 2-stage와 다르게 ROI 영역을 추출하지 않고, 전체 image에 대해서, convolutional network로 classification, localization을 수행한다.
- 속도 측면에서는 장점이 매우 많지만, image에서 여러 물체가 섞여 있다면 정확도를 떨어진다.


---------------------------------------------------------------------------------

논문으로 들어오면,  R-CNN이 등장하기 전 HOG와 Shift를 활용한 object detection 성능은 몇 년 동안 정체 되어 있었습니다.

- HOG과 Shift 참고) https://darkpgmr.tistory.com/116 "https://darkpgmr.tistory.com/116"