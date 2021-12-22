# -*- python-mode -*-
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import collections
# print(os.path.abspath(os.path.dirname(__file__)))
# # /Volumes/GitHub/organization-caliber/recycle_image_project/after_plastic/ai-challenge

parser = argparse.ArgumentParser(description='model params')
parser.add_argument('--mode', type=str, default='init', help="'init', 'update' or 'inference'")
parser.add_argument('--epoch', type=int, default=5, help='')
parser.add_argument('--batch', type=int, default=1, help='') 
# parser.add_argument('--lr', type=, default=, help='') 
args = parser.parse_args()
# data_type? case?가 어떠한가, epoch, batch_size, 
# argparse로 처리할 부분과 shell script config로 진행할 부분 고려해보기

# EF.append('device')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# with torch.cuda.device(0):
#     if torch.cuda.device_count() >= 2:
#         x = torch.tensor([1., 2.]).to(torch.device('cuda:1'))
#         x = torch.tensor([1., 2.]).to(device)

# parallel가능 여부에 따라, logic은 달라야 함.

# model = CaliberM()
# model -> num_classes, option:init/update/inference, pretrained weight가 있는가.(extension:pt, pth)
# num_classes = 5
# num_classes = os.environ["NUM_CLASSES"]   # dockerfile 셋업 끝나면 주석 해제


#     EF.append("sendResult")
#     SendResult().FIN()

# except:
#     print(EF[-1])