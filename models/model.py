# models.model.py
import cv2 as cv
import torch.nn as nn
from PIL import Image
from PTV.detection.engine import train_one_epoch, evaluate
from detection.pretrained import get_finetuned_model
from detection.backbone import add_different_backbone
from detection.detector import resnet50_fpn

if __name__ == '__main__':
    base_dir = os.path.join(root, 'assets/data')
    num_classes = 5
    momentum = 0.9
    learning_rate = 0.001
    step_size = 5
    # dockerfile 셋업 끝나면 하단의 주석 해제 후 상기의 config 제거
    weight_decay = 0.0005
    gamma = 0.1
    # base_dir = os.environ["BASE_DIR"]
    # num_classes = os.environ["NUM_CLASSES"]   
    # lr_scheduler = os.environ["LR_SCHEDULER"] # 해당 부분 코드를 제대로 작성한 경우 사용, 아니면 위의 값 사용.
    # step = os.environ["STEP"]

    class CaliberM(nn.Module):
        def __init__(self, device=None, num_classes=num_classes, mode=args.mode, weight_path=False):
            super().__init__()
            self.device = device
            self.num_classes = num_classes
            self.pretrained_path = weight_path if weight_path else None

            if mode:  # train(init, update)
                self.model = get_finetuned_model(self.num_classes)
                self.model = self.Train()
            else:   # eval(inference)
                self.model = self.get_model()
                self.model = self.Inference()            
            
        def get_model(self):
            try:
                model = get_finetuned_model(num_classes=self.num_classes, weight_path=self.pretrained_path)
                model.load_state_dict(torch.load(self.pretrained_path), strict=False, map_location=self.device)
            except:
                print("weight 경로 내 파일을 다시 확인하세요.")
            return model
                
        def Train(self):
            model = self.model
            model.to(self.device)
            optimizer, lr_scheduler = self.Optim()
            model.train()
            for epoch in range(self.num_epochs):
                train_one_epoch(model, optimizer, dataset, device=self.device, epoch, print_freq=10)
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
