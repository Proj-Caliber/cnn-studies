# config.pretrained.py
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

if __name__ == '__main__':
    # mode == 'update'라면 해당 py 작동
    def get_finetuned_model(model=False, num_classes=None, weight_path=False, mask=False):
        if model:
            if weight_path:
                model.load_state_dict(torch.load(weight_path))
            
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