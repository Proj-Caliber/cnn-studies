# config/backbone.py
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

backbone = torch.load('../assets/mobilenet_v2-b0353104.pth')
# model을 불러오고 model.eval()이 들어가는 이유는? 안 들어가는 것도 있는데 왜? 두 가지 방법을 사용함 1. model.eval() 2. model.train()
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision