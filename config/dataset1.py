import os
import sys
import functools
import json
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset


## 이 부분은 어떻게 사용하는 부분인가?
__all__ = [
    "CustomDataset",
    "instanceMask",
    eval
]

# 그런데 이 부분이 폐플라스틱 객체검출 데이터에 관해서 작성했지만 오픈소스로는 부족하다고 생각이 된다.
# 그래서 이 부분은 어떻게 진행할지 얘기해 봐야겠다.
if (__name__ == '__main__') or (__name__ == 'config.dataset'):
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, root, transforms):
            self.root = root
            self.transforms = transforms
            mode = self.root.split('/')[-1]
            self.mode = mode
            
            ## image 관련
            PP = list(sorted(os.listdir(os.path.join(root, 'image', 'PP'))))
            PET = list(sorted(os.listdir(os.path.join(root, 'image', 'PET'))))
            PS = list(sorted(os.listdir(os.path.join(root, 'image', 'PS'))))
            PE = list(sorted(os.listdir(os.path.join(root, 'image', 'PE'))))
            
            image = [PP, PET, PS, PE]
            image = [i for i in image]
            
            self.image = image
            
            ## annotation 관련
            if self.mode == 'train':
                PP_annot = list(sorted(os.listdir(os.path.join(root, 'annotation', 'PP'))))
                PET_annot = list(sorted(os.listdir(os.path.join(root, 'annotation', 'PET'))))
                PS_annot = list(sorted(os.listdir(os.path.join(root, 'annotation', 'PS'))))
                PE_annot = list(sorted(os.listdir(os.path.join(root, 'annotation', 'PE'))))
            else:
                PP_annot = list(sorted(os.listdir(os.path.join(root, 'annotations', 'PP'))))
                PET_annot = list(sorted(os.listdir(os.path.join(root, 'annotations', 'PET'))))
                PS_annot = list(sorted(os.listdir(os.path.join(root, 'annotations', 'PS'))))
                PE_annot = list(sorted(os.listdir(os.path.join(root, 'annotations', 'PE'))))
                
            annot = [PP_annot, PET_annot, PS_annot, PE_annot]
            annot = [i for i in annot]
            
            self.annot = annot
            self.categories = {1:'pet', 2:'ps', 3:'pp', 4:'pe'}                
            
        def __getitem__(self, idx):
            if self.mode == 'train':
                img_path = self.image[idx]
                annot_path = self.annot[idx]
                
                with open(annot_path, 'r') as f:
                    annot = json.loads(f.read())
                
                image_id = int(annot_path.split('/')[-1].split('_')[1])
                
                img = Image.open(img_path).convert('RGB')
                
                x = annot['images'][0]['width']
                y = annot['images'][0]['height']
                
                annot = annot['annotations']
                
                mask = []
                boxes = []
                labels = []
                iscrowd = []
                area = []
                masked_image = np.zeros((x, y), dtype = np.uint8)
                
                for i in range(len(annot)):
                    segmentation = annot[i]['segmentation'][0]
                    areas = annot[i]['area']
                    iscrowds = annot[i]['iscrowd']
                    xmin, ymin, width, height = annot[i]['bbox'][0], annot[i]['bbox'][1], annot[i]['bbox'][2], annot[i]['bbox'][3]
                    xmax = xmin + width
                    ymax = ymin + height
                    
                    label = annot[i]['category_id']
                    
                    all_points_x = []
                    all_points_y = []
                    
                    for j in range(len(segmentation)):
                        if j%2 == 0:
                            all_points_x.append(segmentation[j])
                        else:
                            all_points_y.append(segmentation[j])
                            
                    polygon_xy = np.array([(x, y) for (x,y) in zip(all_points_x, all_points_y)])
                    
                    cv2.fillpoly(masked_image, np.uint([polygon_xy]), i+1)
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
                    area.append(areas)
                    iscrowd.append(iscrowds)
                    
                mask = np.array(masked_image)
                obj_ids = np.unique(mask)
                obj_ids = obj_ids[1:]
                masks = mask == obj_ids[:,None, None]
                num_objs = len(obj_ids)
                
                boxes = torch.as_tensor(boxes, dtype = torch.float32)
                labels = torch.ones((num_objs,), dtype = torch.int64)
                mask = torch.as_tensor(masks, dtype = torch.uint8)
                image_id = torch.tensor([image_id])
                area = torch.as_tensor(area, dtype = torch.float32)
                iscrowd = torch.as_tensor(iscrowd, dtype = torch.int32)

                target = {}
                target['boxes'] = boxes
                target['labels'] = labels
                target['masks'] = mask
                target['image_id'] = image_id
                target['area'] = area
                target['iscrowd'] = iscrowd

                img = self.preprocess(img)

                return img, target
            
            if self.mode == 'test':
                img_path = self.image[idx]
                annot_path = self.annot[idx]
                
                target = {}
                
                img = Image.open(img_path).convert('RGB')
                
                image_id = int(annot_path.split('/')[-1].split('_')[1])
                
                try:
                    target['image_id'] = torch.stack([torch.tensor([idx]) for i in range(len(annot))], dtype = torch.int64)
                except:
                    target['image_id'] = torch.tensor([idx], dtype = torch.int)
                    
                return img, target
            
print('done')