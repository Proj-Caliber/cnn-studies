# config/model.py
from config.train import CaliberM, CaliberOpt
# from config.send_result import *
model = CaliberM()
optimizer = CaliberOpt()

# model 결과 받으면, 각 label별 cnt=1이 튜플 형식으로 쌓이게끔 작성