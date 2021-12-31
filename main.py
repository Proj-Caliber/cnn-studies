import os
import json
import cv2 as cv
from glob import glob
from tqdm import tqdm
import torch
import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser(
        'Prepare instance segmentation task with Mask R-CNN'
    )
    parser.add_argument('--PATH', help = 'Where is the data?')
    parser.add_argument('--output_dir',
                        help = 'path to save model',
                        default = '../',
                        type = 'str')
    
    # default가 4개인 이유는 대회에서 4개였기 때문에 나중에 수정해도 무방
    parser.add_argument('--NUM_CLASSES', default = 4, type = 'int')
    
    # default가 5인 이유는 일단 확인해보고 나중에 더 추가하도록 유도하기 위해서
    parser.add_argument('--NUM_EPOCH', default = 5, type = int)
    
    parser.add_argument('--BATCH_SIZE', default = 28, type = int)
    return parser

if __name__ == '__main__':
    from typing import Any, Callable, cast, Dict, List, Optional, Tuple
    # from config.TVS.detection import utils as utils
    # from config.TVS.detection import engine as engine

    # from config.TVS.detection import coco_eval as coco_eval
    # from config.TVS.detection import coco_utils as coco_utils

    # from config.TVS.detection import transforms as transforms
    from config.dataset import CustomDataset
    from config.model import CaliberM
    print("done!")

    # from config.detection import 
    # from config import model
    # ROOT = os.getcwd()
    # PATH = os.path.join(ROOT, "Data")

    MOMENTUM = 0.9
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005
    
    parser = get_arg_parser()
    args = parser.parse_args
    
    num_clesses = args.NUM_CLASSES
    num_epoch = args.NUM_EPOCH
    batch_size = args.BATCH_SIZE
    PATH = args.PATH
    
    # use our dataset and defined transformations
    train_dataset = CustomDataset(root=PATH)
    # val_dataset = CustomDataset(root=PATH, mode='val')
    test_dataset = CustomDataset(root=PATH)    
    # 학습 모델 load후 .eval(test_dataset)으로 사용 예정

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(train_dataset)).tolist()

    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-3600])
    # val_dataset = torch.utils.data.Subset(val_dataset, indices[-3600:])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    from torch.utils.data import DataLoader
    # define training and validation data loaders

    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=utils.collate_fn)
    # dl_val = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, collate_fn=utils.collate_fn)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, collate_fn=utils.collate_fn)
    
    
    # train 시
    import config.train
    print("done!")
