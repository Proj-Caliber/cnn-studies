# config/model.py
import torch
import torch.nn as nn
import torchvision
from config.sendResult import SendResult

class CaliberM(nn.Module):
    def __init__(self):
        super().__init__()
        

# transforms도 사용해야하는데
# model 결과 받으면, 각 label별 cnt=1이 튜플 형식으로 쌓이게끔 작성