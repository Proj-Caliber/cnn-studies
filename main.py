import sys
import argparse
import torch

# try:
EF = ['parser']
# 1~main.py까지 다 작성 시, 걸리는 구간 확인용으로 assert
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=5, help='')
parser.add_argument('--batch', type=int, default=1, help='') 
args = parser.parse_args()
# parser.add_argument('--lr', type=, default=, help='') 
# data_type? case?가 어떠한가, epoch, batch_size, 
# model -> num_classes, option:init/update/inference, pretrained weight가 있는가.(extension:pt, pth)
# argparse로 처리할 부분과 shell script config로 진행할 부분 고려해보기

EF.append('device')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with torch.cuda.device(0):
    if torch.cuda.device_count() >= 2:
        x = torch.tensor([1., 2.]).to(torch.device('cuda:1'))
        x = torch.tensor([1., 2.]).to(device)

# parallel가능 여부에 따라, logic은 달라야 함.

'''
1. 데이터 불러와서 처리하는 모듈 : train(+ validation), test(what if...alias validation?)
# google.colab
from google.colab import drive
ROOT = "/"
from glob import glob as gb
'''

#     model = CaliberM()

'''
2. 모델(단순 detection, detect+segs, transforms(+ augmentation)
# from models.model import CaliberM
# model, loss, tensor에서 cuda사용. GPU memory불균형이 생길 수 있음(data parallel)
nn.DataParallel or nn.DistributedDataParallel

3. train or eval
3_1. 학습만 했을 때, 어떤 기준을 가지고 가중치를 업데이트 할 것인지
3_2. U-Net 적용? 미적용?

4. 결과
from models.sendResult import SendResult
4_1. 추론 후, 결과 json 파일 형태로 return
4_2. random한 n개의 추론 결과 시각화(format, gif? png? html?)
'''

#     EF.append("sendResult")
#     SendResult().FIN()

# except:
#     print(EF[-1])