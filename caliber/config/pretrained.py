# config.pretrained.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# 7 class + background
def get_finetuned_model(model=None, num_classes=2, pre_path=None, weight_path=None, mask=False):
    if model:
        if weight_path:
            model.load_state_dict(torch.load(weight_path))
        
    else:
        if pre_path:
            model = torch.load('/home/agc2021/assets/fasterrcnn_resnet50_fpn_coco.pth')
        elif pre_path=='init':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        else:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        if weight_path:
            model.load_state_dict(model.load(weight_path))
    in_features = model.roi_heads.box_predictor.cls_score.in_features   # 1024
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if mask:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model