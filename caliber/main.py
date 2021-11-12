import os
from glob import glob as gb
import torch
# from config.envinfos import EnvInfos    #, engine
# from config.train import *
# from config.models import * # CaliberM, CaliberOpt
# references/detection/engine.py ... ././utils.py 
# from engine import train_one_epoch, evaluate
# import utils

test_path = '/home/agc2021/dataset'
test_image = gb(f'{test_path}/t3_*.JPG')
assert test_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
# 몇 개가 걸리는지에 따라 병렬로 받을지는 조금 더 판 뒤에 해야함...
# EnvInfos().memCheck()
# model = CaliberM()
# optimizer = CaliberOpt()

# inference는 따로 .json으로 나와야 함.
# from config.resultForm import Result