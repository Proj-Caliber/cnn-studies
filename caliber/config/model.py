import os
import torch
import torch.nn as nn
import torchvision
from pretrained import get_finetuned_model
from backbone import add_different_backbone
from detector import resnet50_fpn

class CaliberM(nn.Module):
    def __init__(self, extension='pth', num_classes=2, option=None):
        super().__init__()
        self.pretrained_path = os.path.join(os.getcwd(), f"assets/model.{extension}")
        self.num_classes = num_classes  # 살릴지말지 나중에 결정
        if option:  # train
            self.model = get_finetuned_model(self.num_classes)
        else:   # inference
            self.model = self.get_model()
        
        
    def get_model(self):
        if self.extension=='pth':
            # model 일단 기본 구조를 가져올 모듈부터 생성해야 함
            checkpoint = torch.load(self.pretrained_path)
            
    def Train(self):
        model = self.model
        model.train()
        return model
    
    def Inference(self):
        model = self.model
        model.eval()
        return model