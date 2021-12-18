import cv2 as cv
import torchvision.transforms as T
# import albumentation as A

def get_T(transform=None):
    transform = []
    transform.append(T.ToTensor())
    
    if transform:
        transform.append(T.RandomHoriziontalFlip(0.5))
    return T.Compose(transform)

### 대회엔 transform까지만 사용이 가능할 듯
# def get_A(transform=None, bbox_params=None):
#     '''보통 중간에 위치한 경우가 많았음'''
#     transforms = []
#     transforms.append(A.ToTensor())
    
#     if transform:
#         transforms.append([A.CenterCrop(width=450, height=450),
#                           A.HorizontalFlip(p=0.5),
#                           A.RandomBrightnessContrast(p=0.2)])
#     return T.Compose(transform)

# def save_A():