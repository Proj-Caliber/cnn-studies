# assets/LoadDataset.py
import os
import cv2 as cv
import json
import torch

class PreTrainDataset:
    def __init__(self, path=os.getcwd(), transforms=None, bbox_opt="minmax"):
        self.path = path
        self.bbox_opt = bbox_opt.lower()
        self.transforms = transforms
        self.train_path = list(sorted(os.path.join(self.path, 'assets')))
        self.img_path = [os.path.join(self.path, 'assets', imgs) for imgs in self.train_path if imgs[:-4]==".JPG"]
        self.wgt_path = [os.path.join(self.path, 'assets', wgts) for wgts in self.train_path if wgts[:-4]==".pth"]
        self.imgs, self.targets = self.cvtImageAnnots()
        
    def __getitem__(self, idx):
        image_id = torch.tensor([idx])
        # img = Image.open()
        img = self.imgs[idx]
        target = self.targets[idx]
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
    def cvtImageAnnots(self):
        imgs = []
        for path in self.img_path:
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            imgs.append(img)
        imgs = torch.stack(imgs)
        
        ann_path = str([os.path.join(self.path, 'assets', ann) for ann in self.train_path if ann[:-4]=="json"])
        with open(ann_path, 'r') as jsf:
            ann_data = json.loads(jsf.read())   # ['fname', 'backgroud', 'label', 'bbox_minmax', 'bbox_minWH', 'bbox_centerWH'] : 'background'는 나중에 활용하기.
        targets = {}
        targets['image_id'] = torch.stack([torch.tensor([idx]) for idx in range(len(ann_data['fname']))])
        targets['boxes'] = torch.as_tensor(ann_data[self.bboxtype()], dtype=torch.float32)
        targets['labels'] = torch.as_tensor(self.label2code(ann_data['label']), dtype=torch.int64)
        targets['area'] = (targets['boxes'][:, 3]) * (targets['boxes'][:, 2]) if self.bbox_opt[-2:]=='wh' \
            else (targets['boxes'][:, 3] - targets['boxes'][:, 1]) * (targets['boxes'][:, 2] - targets['boxes'][:, 0])
        # mask..., dtype=torch.uint8
        return imgs, targets
    
    def bboxtype(self):
        try:
            assert self.bbox_opt in ['minmax', 'minwh', 'centerwh']
            if self.bbox_opt=='minmax':
                bbox_opt = 'bbox_'+self.bbox_opt
            else:
                bbox_opt = 'bbox_'+self.bbox_opt[:-2]+self.bbox_opt[-2:].upper()
            return bbox_opt
        except AssertionError:
            print("Put right BBox option which is one of 'minmax', 'minwh', and 'centerwh'")
        
    def label2code(label_list):
        cvtDict = {1:'paper', 2:'carton', 3:'can', 4:'glass', 5:'pet', 6:'plastic', 7:'plastic bag'}
        revDict = {v:k for k,v in cvtDict.items()}
        codes = []
        for cont in label_list:
            if type(cont)==str:
                codes.append(revDict[cont])
            else:
                subs = []
                for sub in cont:
                    subs.append(revDict[sub])
                codes.append(subs)
        # 언제 뭐가 필요할지 모르니까, 우선은 이렇게만 작성
        return codes