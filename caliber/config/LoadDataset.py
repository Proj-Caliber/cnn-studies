# assets/LoadDataset.py
import os
import cv2 as cv
import json
import torch
# print(os.environ['HOME'])
# 도커에 그들이 환경변수를 재설정한 데에는 사용하란 뜻이 있을 것으로 생각됨. os.system만 안 건들면 되는거???
class PreTrainDataset:
    def __init__(self, path=os.getcwd()):
        self.path = path
        self.train_path = list(sorted(os.path.join(self.path, 'assets')))
        self.img_path = [os.path.join(self.path, 'assets', imgs) for imgs in self.train_path if imgs[:-4]==".JPG"]
        self.wgt_path = [os.path.join(self.path, 'assets', wgts) for wgts in self.train_path if wgts[:-4]==".pth"]
        self.imgs = self.cvtImages()
        
    def __getitem__(self, idx):
        image_id = torch.tensor([idx])
        img = Image.open()
        return img, target
    
    def cvtImages(self):
        ann_path = str([os.path.join(self.path, 'assets', ann) for ann in self.train_path if ann[:-4]=="json"])
        with open(ann_path, 'r') as jsf:
            ann_data = json.loads(jsf.read())
        # ['fname', 'backgroud', 'label', 'bbox_minmax', 'bbox_minWH', 'bbox_centerWH']
        # 1:'paper], 2:'carton', 3:'can', 4:'glass', 5:'pet', 6:'plastic', 7:'plastic bag'
        # json_data['cls_code'] = list(map())
        return imgs