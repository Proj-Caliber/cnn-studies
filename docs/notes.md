# Notes

## specific

1. data pipelines :

<small>train(+ validation), test(what if...alias validation?)</small>

- [ ] init
- [ ] update
- [ ] inference
```python
# init/update/inference
```

2. model pipelines 

<small>detection, detect+segs, transforms(+ augmentation</small>

- [ ] train
- [ ] eval
```python
# detection/segmentation
# loss funciton, opencv mask
```

4. UI?GUI(VSL)

```python
from models.sendResult import SendResult
# 추론 후, 결과 json 파일 형태로 return
# random한 n개의 추론 결과 시각화(format, gif? png? html?)
```


## general

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
