# config.detector.py
import os
# from typing import Type, Callable, Union, List, Optional, no_type_check
import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def resnet50_fpn(device=None, num_classes=NUM_CLASSES, save_opt=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features   # 1024
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if save_opt:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), 'assets/Pweights.pt'))
    else:
        torch.save(model, os.path.join(os.getcwd(), 'assets/pretrained.pt'))
    return model