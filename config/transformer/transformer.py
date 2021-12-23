import cv2 as cv
import torchvision.transforms as T

# 여기서 전처리 부분을 좀 더 강화하는 방법을 생각하자~
def get_T(transform=None):
    transform = []
    transform.append(T.ToTensor())
    
    if transform:
        transform.append(T.RandomHoriziontalFlip(0.5))
    return T.Compose(transform)