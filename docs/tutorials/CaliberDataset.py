import cv2 as cv
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from glob import glob
import numpy as np
import cv2 as cv
from PIL import Image
import json
from torchvision import transforms as T
import torch

# 원래 데이터 intance마다
def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask 

def instanceMask(mask, segmentation):
    import cv2 as cv
    mask = mask.copy()
    pts = np.array(segmentation, dtype = np.uint8).reshape(-1, 2)

    pts = np.stack([pts, pts, pts])
    cv.fillPoly(mask, pts, color=(255, 255, 255))
    return mask


class CustomDataset(Dataset):
    def __init__(self, root, transforms = None, target_transforms = None, mode = 'train'):
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.mode = mode.lower()
        self.infos = self.baseInfos()
        self.imgs, self.annots = self.infos['image_path'], self.infos['annot_path']
        # if self.mode == 'train':
        #     self.imgs, self.masks = blahblah
        self.masks = None

    def __getitem__(self, idx=None):
        '''return tuple(image, target)'''
        image_path = self.imgs[idx]
        # # mask 생성 함수보고 고쳐야 할 듯
        image = Image.open(image_path)

        # C, H, W ==  [H, W, C] >> [N+1, H, W, C] >> [N, H, W]
        # # masks = np.zeros(mask_shape, dtype = np.uint8)  
        # # 굳이 3채널 정보가 필요하지 않다면, (H, W) 만 받아도, instance 공간 할당은 이미 해놓음
        # image = torchvision.transforms(image)    # [C, H, W] 
        # # PIL
        # image = Image.open(image_path).convert("RGB")
        # # image = np.asarray(image)
        # mask = np.zeros(image.shape, dtype=np.uint8)
        
        annot_path = self.annots[idx]
        if self.mode == "train":
            # # train인 경우에만 target['masks'] 처리할 함수가 들어가야 함.
            # mask_shape, target = self.json2annots(annot_path, idx)
            mask, target = self.json2annots(annot_path, idx)
        else:
            target = self.json2annots(annot_path, idx)
        # target : dict

        if self.transforms is not None:
            try:
                data = self.transforms(img, target)
                return data
            except:
                if self.transforms is not None:
                    img = self.transforms(img)
                elif self.target_transforms is not None:
                    target = self.target_transforms(target)
                return img, target
        else:
            # target_transform만 따로 들어오는 경우는 고려하지 않았음.
            img = self.preprocess(image)
            return img, target

    def baseInfos(self):
        baseinfos = {"label" : [], "metainfo_id" : [], "feature" : [], "image_path" : [], "annot_path": []}

        bpath = os.path.join(self.root, self.mode)
        bdirs = sorted(os.listdir(bpath))   # image, annotation(s)
        
        for bdir in tqdm(bdirs):
            for dirs in sorted(os.listdir(os.path.join(bpath, bdir))):
                dpath = os.path.join(bpath, bdir, dirs)
                
                if bdir.lower()=='image':
                    paths = sorted(glob(dpath + "/*.jpg"))
                    baseinfos['image_path'].extend(paths)
                    fnames = [os.path.basename(fname) for fname in paths]
                    labels = list(map(lambda x: (x.split('_')[0]), fnames))
                    baseinfos['label'].extend(labels)
                    metaIds = list(map(lambda x: int(x.split('_')[1]), fnames))
                    baseinfos['metainfo_id'].extend(metaIds)
                    feats = list(map(lambda x: int(x.split('_')[-1][:-4]), fnames))
                    baseinfos['feature'].extend(feats)
                else:
                    paths = sorted(glob(dpath + "/*.json"))
                    baseinfos['annot_path'].extend(paths)
        return baseinfos

    def preprocess(self, img):
        image = img
        import torchvision.transforms as T
        m, s = np.mean(image, axis = (0, 1)), np.std(image, axis = (0, 1))        
        if self.mode == 'train':
            transform = T.Compose([
                                   T.ToTensor(),
                                   T.Normalize(mean = m, std = s),
            ])
            image = transform(image)
        else:
            transform = T.Compose([
                                #    T.Resize(256),
                                   T.ToTensor(),
                                   T.Normalize(mean = m, std = s),
            ])
            image = transform(image)
        return image
    
    def json2annots(self, annot_path, idx):
        # import cv2 as cv
        with open(annot_path, 'r', encoding = 'UTF-8') as f:
            annot = json.loads(f.read())

        n_objects = len(annot['annotations'])
        W, H = int(annot['images'][0]['width']), int(annot['images'][0]['height'])

        if self.mode == 'train':
            # target = {"boxes" : [], "labels" : [], "masks" : [np.zeros((W, H), dtype = np.uint8)], "image_id" : [], "area" : [], "iscrowd" : []}
            target = {"boxes" : [], "labels" : [], "masks" : [], "image_id" : [], "area" : [], "iscrowd" : []}

            # bbox ground bbox
            gxmin, gymin, gxmax, gymax = 0, 0, 0, 0
            masked_image = np.zeros((W,H), dtype = np.uint8)

            for i in range(n_objects):
                bbox = annot['annotations'][i]['bbox']
                xmin, ymin, width, height = bbox[0],bbox[1],bbox[2],bbox[3]
                xmax, ymax = xmin + width, ymin + height
                target['boxes'].append([xmin, ymin, xmax, ymax])
                # ########## just-in-case(ground_bbox) ##########
                # gxin = xmin if xmin > gxmin else gxmin
                # gymin = ymin if ymin > gymin else gymin
                # gmax = xmax if xmax > gxmax else gxmax
                # gmax = ymax if ymax > gymax else gymax
                # target['boxes'].append([xmin, ymin, xmax, ymax])

                label = annot['annotations'][i]['category_id']
                # torch.ones로 하지 않은 이유는 클래스 정보가 들어가야된다고 판단했기 때문
                target['labels'].append([label])

                # try:
                #     assert mask!=None

                all_points_x = []
                all_points_y = []

                for j in range(len(annot['annotations'][i]['segmentation'][0])):
                    if j%2 == 0:
                        all_points_x.append(annot['annotations'][i]['segmentation'][0][j])
                    else:
                        all_points_y.append(annot['annotations'][i]['segmentation'][0][j])

                polygon_xy = np.array([(x,y) for (x,y) in zip(all_points_x, all_points_y)])
                cv.fillPoly(masked_image, np.uint([polygon_xy]), i+1)    # (H, W, C)

                # except:
                    # mask 처리 전, 공간 할당
                    # target['masks'].append(np.zeros((W, H), dtype=np.uint8))

                area = torch.tensor(annot['annotations'][i]['area'], dtype = torch.float32)
                target['area'].append([area])

                iscrowd = torch.tensor(annot['annotations'][i]['iscrowd'], dtype = torch.int64)
                target['iscrowd'].append([iscrowd])
                # ignore = torch.tensor(annot['annotations'][i]['ignore'], dtype = torch.int64)
                # target['ignore'].append([ignore])

            ########################################################## instance id, 필요 시 주석해제 ##########################################################    
            # segmentation에 DataLoader 내에서 어떻게 선언되어 있는지 확인해야 함
            # target['id'] = torch.stack([torch.tensor([i+1]) for i in range(n_objects)]) # shape : (3, 1)
            # ValueError: Expected target boxes to be a tensorof shape [N, 4], got torch.Size([1, 3, 4]).
            ########################################################## type은 맞으나, shape이 맞는지 확신이 없는 부분 ##########################################################
            # torch.Size([3, 2048, 2048])
            # target['boxes'] = torch.as_tensor(target['boxes'], dtype = torch.float32)   
            # shape : (3, 4)
            target['boxes'] = torch.stack([torch.tensor(bbox, dtype = torch.float32) for bbox in target['boxes']])

            # shape : torch.Size([1, 3, 1])
            target['labels'] = torch.as_tensor((target['labels'], ), dtype = torch.int64).squeeze()
            
            mask = np.array(masked_image)
            obj_ids = np.unique(mask)
            # background는 빼야하기 때문에
            obj_ids = obj_ids[1:]
            # 각 mask별로 나눠주기
            masks = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)
            target['masks'] = torch.as_tensor(masks, dtype = torch.uint8)
            
            # image_path[index] -> 이미지 자체의 아이디
            target['image_id'] = torch.stack([torch.tensor([idx]) for i in range(n_objects)])   # shape : (3, 1)
            target['area'] = torch.as_tensor(target['area'], dtype = torch.float32) # shape : (1, 3) >> (3, 1)
            target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype = torch.int64)
            
            return target['masks'], target

        else:
            target = {}
            try:
                target['image_id'] = torch.stack([torch.tensor([idx]) for i in range(n_objects)], dtype = torch.int64)
            except:
                target['image_id'] = torch.tensor([idx], dtype=torch.int)
            return target
        # return (n_objects, W, H), target

    # def createMask(self, segmentation=None):
    #     # if self.pipe != "detection":
    #     # elif self.pipi != "detection":
    #     #     img, background = self.imageNmask(image_path, mask_shape)
    #     # bbox
    #     # segmentation
    #     background = self.masks
    #     pts_x, pts_y = [], []
    #     for i in range(len(segmentation)):
    #         pts_x.append(segmentation[i]) if i%2 == 0 else pts_y.append(segmentation[i])
    #     polygon_xy = np.array([(x, y) for (x, y) in zip(pts_x, pts_y)])
    #     cv2.fillPoly(background, np.uint([polygon_xy]), i)
    #     return background
 

    def _labels2category(self, label):
        category = {1 : "PET", 2 : "PS", 3 : "PP", 4 : "PE"}
        return category(label)

    def __len__(self):
        return len(self.imgs)

    # ########## just-in-case ##########
    # # when we need more edas
    # def additional_infos(self):
    #     with open(self.annots[0], 'r') as f:
    #         annot = json.loads(f.read())
    #     metainfo = annot['metainfo']