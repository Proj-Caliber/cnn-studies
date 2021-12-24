# models.detection.detector.py
from torch import Tensor
from typing import Type, Callable, Union, List, Optional, no_type_check
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

if __name__ == "__main__":
    def resnet50_fpn(num_classes=num_classes, save_opt=True):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features   # 1024
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        if save_opt:
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'assets/Pweights.pt'))
        else:
            torch.save(model, os.path.join(os.getcwd(), 'assets/pretrained.pt'))
        return model