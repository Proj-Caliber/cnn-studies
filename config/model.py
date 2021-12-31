import os
import torch
import torch.nn as nn
if __name__ == "__main__":
    # engine부분을 어떻게 넣을 것인가?
    from vision.references.detection.engine import train_one_epoch, evaluate
    from config.detection.pretrained import get_finetuned_model
    from config.detection.backbone import add_different_backbone
    from config.detection.detector import resnet50_fpn

    class CaliberM(nn.Module):
        def __init__(self, device=device, extension='pt', num_classes=NUM_CLASSES, option=None):
            super().__init__()
            self.pretrained_path = os.path.join(os.getcwd(), f"assets/taco/model.{extension}")
            self.device = device
            self.num_classes = num_classes
            self.extension = extension
            if option:  # train
                self.model = get_finetuned_model(self.num_classes)
                self.model = self.Train()
            else:   # inference
                self.model = self.get_model()
                self.model = self.Inference()
            
            
        def get_model(self):
            if self.extension=='pt':
                pre_path = '/home/agc2021/assets/fasterrcnn_resnet50_fpn_coco.pth'
                model = get_finetuned_model(num_classes=self.num_classes, pre_path=pre_path, weight_path=self.pretrained_path)
                model.load_state_dict(torch.load(self.pretrained_path), strict=False, map_location=self.device)
            else:
                model = presnet50_fpn(device=self.device, num_classes=self.num_classes)
            return model
                
        def Train(self):
            model = self.model
            model.to(self.device)
            optimizer, lr_scheduler = self.Optim()
            model.train()
            for epoch in range(self.num_epochs):
                train_one_epoch(model, optimizer, dataset, self.device, epoch, print_freq=10)
                lr_scheduler.step()
                evaluate(model, dataset, device=self.device)
            return model
        
        def Inference(self):
            model = self.model
            model.to(self.device)
            model.eval()
            
            for idx in range(len(dataset)):
                img, _ = dataset[idx]
                label_boxes = np.array(dataset[idx][1]["boxes"])
                with torch.no_grad():
                    prediction = model([img])
                image = Image.fromarray(img.mul(255).permute(1, 2,0).byte().numpy())
            return model
        
        def Params(self):
            return [p for p in self.model.parameters() if p.requires_grad]
        
        def Optim(self):
            optimizer = torch.optim.SGD(self.Params(), lr=0.005, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
            return optimizer, lr_scheduler