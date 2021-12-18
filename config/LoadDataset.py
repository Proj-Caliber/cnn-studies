# config/LoadDataset.py
import os
import cv2 as cv
import json
import torch
import pandas as pd

class CustomPath:
    def __init__(self, path=None, ddir=None, data_opt=None, fdir=None, image_ext="jpg", annotation_ext="json", annot_path=True):
        try:
            self.path = os.getcwd() if path is None else path
        except:
            print("본인이 설치한 폴더 구조에 맞게 파일 경로를 재설정해야 합니다.")
        self.bpath = self.basepath(ddir=ddir, data_opt=data_opt, fdir=fdir)
        self.images = self.imagepath(extension=image_ext)
        self.annots = self.annotpath(extension=annotation_ext, annot_path=annot_path)
    
    def basepath(self, ddir=None, data_opt="Train", fdir=None):
        self.data_opt = data_opt.lower()    # train일 때 model.train()으로 받기 위함
        if ((ddir==None)&(data_opt==None)&(fdir==None)):
            return self.path
        else:
            ddir = [p for p in ddir.split('/') if p != ""]
            data_opt = [p for p in data_opt.split('/') if p != ""]
            
            # 만약 클래스 폴더 별로 모델이 돌아가게 작성한다면 사용될 부분
            if type(fdir)==str:
                fdir = [p for p in fdir.split('/') if p != ""]
                add_dirs = (([].extend(ddir)).extend(data_opt)).extend(fdir)
                add_dirs = list(map(lambda x : "".join(x) if x != None else "" , add_dirs))
                add_dirs = "/".join(add_dirs)
                return os.path.join(self.path, add_dirs)
            
            # 전체 클래스를 한 tensor에 넣을 경우 사용될 부분
            elif type(fdir)==list:
                add_dirs = ([].extend(ddir)).extend(data_opt)
                add_dirs = list(map(lambda x : "".join(x) if x != None, add_dirs))
                add_dirs = "/".join(add_dirs)
                lower_dir = os.listdir(os.path.join(self.path, add_dirs))
                exp_initial = input(f"""\n\t요청한 디렉토리 내 존재하는 최하단 디렉토리는 다음과 같습니다.\n\t\t\t{lower_dir}\n\t\t이 중 제외할 디렐토리의 첫 번째 문자를 입력하세요. : \t""")
                try:
                    exp_initial = exp_initial.lower()
                except:
                    exp_initial = str(exp_initial)
                bdirs = []
                for ldir in lower_dir:
                    if (ldir.lower()).startswith(exp_initial):
                        pass
                    else:
                        bdirs.append(os.path.join(self.path, add_dirs, ldir)) if add_dirs != None else bdirs.append(os.path.join(self.path, ldir))
                return bdirs
            
    def imagepath(self, extension="jpg"):
        from glob import glob
        if type(self.bpath)==str:
            return list(sorted(glob(self.bpath + "/*." + extension)))
        else:
            img_path = []
            for bpath in self.bpath:
                img_path.extend(list(sorted(glob(bpath) + "/*." + extension))))
            return img_path
    
    def annotpath(self, extension="json", annot_path=True):
        from glob import glob
        if annot_path:
            return list(sorted(self.bpath + "/*." + extension))
        else:
            annot_dir = input(f"{self.bpath} 하위 경로 중 annotation 할 파일이 위치한 경로를 작성하세요. : ")
            if annot_dir.startswith('/'):
                annot_dir = annot_dir[1:]
            if annot_dir.endswith('/'):
                annot_dir = annot_dir[:-2]
            annot_dir = os.path.join(self.bpath, annot_dir)
            return list(sorted(annot_dir + "/*." + extension))

class plasticDF:
    '''
    항수 다시 만들기 너무 귀찮지만, 대회에서 너~~~~~~~~무 데이터도 경로도 잘 줘서...
    우선 원래 load하는 부분은 차차 수정할테지만, 급하게 사용해야해서 만듦
    '''
    def __init__(self, data=None):
        self.DF_info = data
    ############### 내일 일 끝내고 다시 작성하기 #################
    
        

class CaliberDataset(CustomPath):
    def __init__(self, path=None, bbox_opt="minmax", transforms=None, ddir=None, data_opt=None, fdir=None, image_ext="jpg", annotation_ext="json", annot_path=True):
        super(CaliberDataset, self).__init__(self, path=path, ddir=ddir, data_opt=data_opt, fdir=fdir, image_ext=image_ext, annotation_ext=annotation_ext, annot_path=annot_path)
        # self.path, self.bpath, self.images, self.annots
        self.bbox_opt = bbox_opt.lower()
        self.transforms = transforms
        # self.wgt_path = [os.path.join(self.path, 'assets', wgts) for wgts in self.train_path if wgts[:-4]==".pth"]
        # 학습한 weights를 받아서 사용하는 부분은 나중에 모델할 때, 추가적으로 작성
        # 
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
        for path in self.images:
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            imgs.append(img)
        imgs = torch.stack(imgs)
        
        with open(self.annots, 'r') as jsf:
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
    
def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    print(filename)
    boxes_array = data[data["filename"]==filename][["xmin", "ymin", "xmax", "ymax"]].values
    return boxes_array

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
                fullPath = fullPath.replace(data_folder, '')
                allFiles.append(fullPath)
    return allFiles