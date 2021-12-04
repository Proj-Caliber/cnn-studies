import os
import sys
import torch
from glob import glob as gb
# from config.sendResult import SendResult
# from config.model import CaliberM
if __name__ == "__main__":
    from os.path import join as jn
    from os.path import split as spt
    ROOT = os.getcwd()
    
    from config.annots_cv import annotCats
    
################################# 경로 확인되면 모델 작동되는 코드만 남기고 지워야 함 ###############################

# try:
#     # EF = ["test_path"]
#     # test_path = '/home/agc2021/dataset'
#     # test_image = gb(f'{test_path}/t3_*.JPG')
#     # assert test_image
    
#     # EF.append("asset_path_path")
#     # train_path = '/home/agc2021/assets'
#     # train_image = gb(f'{train_path}/t3_*.JPG')
#     # assert train_image
    
#     EF = ["asset_path"]
#     assets_path = os.path.join(os.getcwd(), 'assets')
#     taco_weights = gb(f'{assets_path}/taco/model*')
#     train_image = gb(f'{assets_path}/train/t3_*.JPG')
#     EF.append("taco")
#     assert taco_weights
#     EF.append("train")
#     assert train_image
    
#     EF.append("torch_device")    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     assert device
    
#     EF.append("torch_gpu")
#     device = torch.device("cuda")
#     assert device
    
#     EF.append("torch_parallel")
#     with torch.cuda.device(0):
#         if torch.cuda.device_count() >= 2:
#             x = torch.tensor([1., 2.]).to(torch.device('cuda:1'))
#             x = torch.tensor([1., 2.]).to(device)
#     assert x
    
#     EF.append("model")
#     extension = 'pth'
#     num_classes = 7 + 1
#     option = 'inference'
    
#     model = CaliberM()
    
#     EF.append("sendResult")
#     SendResult().FIN()
    
# except:
#     print(EF[-1])