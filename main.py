import os
import json
import cv2 as cv
from glob import glob
from tqdm import tqdm
'''
vision부분과 engine부분을 어떻게 진행해야 할까?
'''

if __name__ == '__main__':
    from typing import Any, Callable, cast, Dict, List, Optional, Tuple
    # from config.TVS.detection import utils as utils
    # from config.TVS.detection import engine as engine

    # from config.TVS.detection import coco_eval as coco_eval
    # from config.TVS.detection import coco_utils as coco_utils

    # from config.TVS.detection import transforms as transforms
    # from config.dataset import *
    from CaliberDataset import CustomDataset
    print("done!")

    # from config.detection import 
    # from config import model
    ROOT = os.getcwd()
    PATH = os.path.join(ROOT, "data")

    BATCH_SIZE = 1
    NUM_EPOCHS = 5
    MOMENTUM = 0.9
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005
    
    # use our dataset and defined transformations
    train_dataset = CustomDataset(root=PATH, mode = 'train')
    # val_dataset = CustomDataset(root=PATH, mode='val')
    test_dataset = CustomDataset(root=PATH, mode='test')    
    # 학습 모델 load후 .eval(test_dataset)으로 사용 예정

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(train_dataset)).tolist()

    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-3600])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-3600:])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    from torch.utils.data import DataLoader
    # define training and validation data loaders

    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, collate_fn=utils.collate_fn)
    # dl_val = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=utils.collate_fn)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=utils.collate_fn)
    # train 시
    import train
    print("done!")
