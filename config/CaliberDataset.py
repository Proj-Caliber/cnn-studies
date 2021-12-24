import os
import json
import torch
import cv2 as cv
import numpy as np
from torch.utils import data

if (__name__ == '__main__') or (__name__ == 'config.CaliberDataset'):
    import os
    print(__name__)
    from torch.utils.data import Dataset6
    def instanceMask(mask, segmentation):
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
            self.masks = None

        def __getitem__(self, idx=None):
            '''return tuple(image, target)'''
            image_path = self.imgs[idx]
            # # mask 생성 함수보고 고쳐야 할 듯
            image = cv.imread(image_path)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # C, H, W ==  [H, W, C] >> [N+1, H, W, C] >> [N, H, W]
            mask = np.zeros(image.shape, dtype=np.uint8)    
            # # masks = np.zeros(mask_shape, dtype = np.uint8)  
            image = Image.fromarray(image)  # cv to PIL
            
            annot_path = self.annots[idx]
            if self.mode != "test":
                target = self.json2annots(annot_path, idx)
            elif self.mode == "train":
                # # train인 경우에만 target['masks'] 처리할 함수가 들어가야 함.
                mask, target = self.json2annots(annot_path, idx, mask=mask)
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
            if self.mode == "val":
                bpath = os.path.join(self.root, "train")
            else:
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
                                    T.Resize(256),
                                    T.ToTensor(),
                                    T.Normalize(mean = m, std = s),
                ])
                image = transform(image)
            return image
        
        def json2annots(self, annot_path, idx, mask=None):
            # try:
            with open(annot_path, 'r') as f:
                annot = json.loads(f.read())

            n_objects = len(annot['annotations'])
            W, H = int(annot['images'][0]['width']), int(annot['images'][0]['height'])
            
            if self.mode == 'test':
                target = {}
                try:
                    target['image_id'] = torch.stack([torch.tensor([idx]) for i in range(n_objects)], dtype = torch.int64)
                except:
                    target['image_id'] = torch.tensor([idx], dtype=torch.int)
                return target
            # return (n_objects, W, H), target

            else:
                target = {"boxes" : [], "labels" : [], "masks" : [], "image_id" : [], "area" : [], "iscrowd" : []}

                # bbox ground bbox
                gxmin, gymin, gxmax, gymax = 0, 0, 0, 0
                for i in range(n_objects):
                    bbox = annot['annotations'][i]['bbox']
                    xmin, ymin, width, height = bbox[0],bbox[1],bbox[2],bbox[3]
                    xmax, ymax = xmin + width, ymin + height
                    target['boxes'].append([xmin, ymin, xmax, ymax])

                    label = annot['annotations'][i]['category_id']
                    target['labels'].append([label])

                    masks = mask.copy() # (H, W, C)
                    pts = np.array(annot['annotations'][i]['segmentation'], dtype = np.uint8).reshape(-1, 2)    # (num_of_points,2)
                    # org:[C, H, W] >> [H, W, C] >> [N+1, H, W, C] >> [N, H, W]
                    pts = np.stack([pts, pts, pts]).astype(int) # (C, num_of_points. 2)
                    cv.fillPoly(masks, pts, (255, 255, 255))    # (H, W, C)

                    instance = cv.cvtColor(mask, cv.COLOR_RGB2GRAY) # (W, H):instance mase

                    target['masks'].append(instance)

                    area = torch.tensor(annot['annotations'][i]['area'], dtype = torch.float32)
                    target['area'].append([area])

                    iscrowd = torch.tensor(annot['annotations'][i]['iscrowd'], dtype = torch.int64)
                    target['iscrowd'].append([iscrowd])

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
                
                # shape : torch.Size([1, 4, 2048, 2048])
                # target['masks'] = torch.as_tensor(target['masks'], dtype = torch.uint8) # (2, H, W) : (C, H, W) >> detection?
                target['masks'] = torch.stack([torch.tensor(mask, dtype = torch.uint8) for mask in target['masks']])  # [3, 2048, 2048] : (num_of_instance, H, W) >> segmentation?
                # mssk 처리 전, 공간 할당

                ########################################################## below ~~ torch.Size([3, 1]) : 각각의 shape도 type도 맞는 것 같음 ##########################################################
                # image_path[index] -> 이미지 자체의 아이디
                target['image_id'] = torch.stack([torch.tensor([idx]) for i in range(n_objects)])   # shape : (3, 1)
                target['area'] = torch.as_tensor(target['area'], dtype = torch.float32) # shape : (1, 3) >> (3, 1)
                target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype = torch.int64)
                # target['ignore'] = torch.as_tensor(target['ignore'], dtype = torch.int64)
                return target['masks'], target
            # except:
            #     print(idx)
            #     pass
    
        def _labels2category(self, label):
            category = {1 : "PET", 2 : "PS", 3 : "PP", 4 : "PE"}
            return category(label)

        def __len__(self):
            return len(self.imgs)

    dataset = CustomDataset(root=os.getcwd()), mode='train')
    print(dataset.__len__())