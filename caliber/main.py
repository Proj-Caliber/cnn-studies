import os
import sys
import torch
from glob import glob as gb
from config.sendResult import SendResult

try:
    EF = ["test_path"]
    test_path = '/home/agc2021/dataset'
    test_image = gb(f'{test_path}/t3_*.JPG')
    assert test_image
    
    EF.append("asset_path_path")
    train_path = '/home/agc2021/assets'
    train_image = gb(f'{train_path}/t3_*.JPG')
    assert train_image
    
    EF.append("asset_path_os")
    train_path = os.path.join(os.getcwd(), 'assets')
    train_image = gb(f'{train_path}/t3_*.JPG')
    assert train_image
    
    EF.append("torch_device")    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device
    
    EF.append("torch_gpu or parallel")
    with torch.cuda.device(0):
        if torch.cuda.device_count() >= 2:
            x = torch.tensor([1., 2.]).to(torch.device('cuda:1'))
            x = torch.tensor([1., 2.]).to(device)
    assert x
    
    EF.append("model")
    extension = 'pth'
    num_classes = 7 + 1
    option = 'inference'
    
    
    EF.append("sendResult")
    SendResult.FIN()
    
except:
    print(EF[-1])