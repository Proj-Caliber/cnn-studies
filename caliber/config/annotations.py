import os
import cv2 as cv
import json
# print(os.environ['CUDA_DEVICE_ORDER'])
# print(os.environ['HOME'])
# 도커에 그들이 환경변수를 재설정한 데에는 사용하란 뜻이 있을 것으로 생각됨. os.system만 안 건들면 되는거???
# train_path = 

# 대부분 [xmin, ymin, xmax, ymax]로 작성 -> 이미 한 대회에서 제공한 이미지파일 annotations.json
# .strip(), .lower()

