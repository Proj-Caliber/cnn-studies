# config.detector.py
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cuda')
# pretrained resnet50_fpn
# model = torch.load(os.path.join(os.getcwd(), "assets/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"))
# model.eval()
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 8 # 7 class + background
in_features = model.roi_heads.box_predictor.cls_score.in_features   # 1024
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
torch.save(model.state_dict(), os.path.join(os.getcwd(), 'assets/pretrained.pt'))