# Notes
주석은 보통 5~6줄로 작성

* [decorator](https://docs.python.org/3/glossary.html#term-decorator)
    * classmethod()
    * staticmethod()
* [asynchronous-iterator](https://docs.python.org/3/glossary.html#term-asynchronous-iterator)

```python
__file__    # main.py
__name__    # main
import __future__   # 를 활용한 ML, DL 코드 많이 보임
import setuptools, itertools, functools, operator, collections
import property
```
* property
    * getter
    * setter
    * deleter



1. 데이터 불러와서 처리하는 모듈 : train(+ validation), test(what if...alias validation?)
```python
# google.colab
from google.colab import drive
ROOT = "/"
from glob import glob as gb
```
2. 모델(단순 detection, detect+segs, transforms(+ augmentation)
```python
# from models.model import CaliberM
# model, loss, tensor에서 cuda사용. GPU memory불균형이 생길 수 있음(data parallel)
nn.DataParallel or nn.DistributedDataParallel
```
3. train or eval
    1. 학습만 했을 때, 어떤 기준을 가지고 가중치를 업데이트 할 것인지
    2. U-Net 적용? 미적용?
4. 결과
```python
from models.sendResult import SendResult
```
    1. 추론 후, 결과 json 파일 형태로 return
    2. random한 n개의 추론 결과 시각화(format, gif? png? html?)