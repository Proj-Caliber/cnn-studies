# models.model.py
import cv2 as cv
import torch.nn as nn
from PIL import Image
from PTV.engine import train_one_epoch, evaluate
from detection.pretrained import get_finetuned_model
from detection.backbone import add_different_backbone
from detection.detector import resnet50_fpn

if (__name__ == '__main__') or (__name__ == 'models.model'):
    num_classes = os.environ["NUM_CLASSES"]
    momentum = os.environ["MOMENTUM"]
    learning_rate = os.environ["LEARNING_RATE"]
    weight_decay = os.environ["WEIGHT_DECAY"]
    
    # lr_scheduler = os.environ["LR_SCHEDULER"] # 해당 부분 코드를 제대로 작성한 경우 사용, 아니면 위의 값 사용.
    step_size = os.environ["STEP"]
    gamma = os.environ["GAMMA"]
    
    # parser(main.py)
    mode = os.environ["MODE"]
    epochs = os.environ["EPOCHS"]
    batch_size = os.environ["BATCH_SIZE"]
    weight_path = os.path.join(os.environ["WORKSPACE"], os.environ["SAVE_DIR"])

    class CaliberM(nn.Module):
        def __init__(self, device=None, num_classes=num_classes, mode=mode, weight_path=weight_path):
            super().__init__()
            self.device = device
            self.num_classes = num_classes
            self.weight_path = weight_path if weight_path else None

            if mode == "inference":   # eval(inference)
                self.model = self.get_model()
                self.model = self.Inference()            
            else:  # train(init, update)
                self.model = get_finetuned_model(self.num_classes)
                self.model = self.Train()
            
        def get_model(self):
            try:
                model = get_finetuned_model(num_classes=self.num_classes, weight_path=self.weight_path)
                model.load_state_dict(torch.load(self.pretrained_path), strict=False, map_location=self.device)
            except:
                print("weight 경로 내 파일을 다시 확인하세요.")
            return model
                
        def Train(self):
            model = self.model
            model.to(self.device)
            optimizer, lr_scheduler = self.Optim()
            model.train()
            for epoch in range(epochs):
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
            optimizer = torch.optim.SGD(self.Params(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            return optimizer, lr_scheduler
