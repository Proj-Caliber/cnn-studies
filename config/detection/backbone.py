# config/backbone.py
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

if __name__ == "__main__":
    from config.detector.pretrained import get_finetuned_model
def add_different_backbone():
    if __name__ == "__main__":
        model = get_finetuned_model() # 이 부분 한 번 더 확인해보기!!!
        backbone = torch.load('../assets/mobilenet_v2-b0353104.pth')
    else:
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone,
                       num_classes=8,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model