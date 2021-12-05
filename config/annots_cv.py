import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class annotCats:
    def __init__(self, image_path=None, label=None, xtickrange=100, ytickrange=100):
        self.colors = self.cvColor()
        self.label = label
        try:
            self.image = cv.imread(image_path)
            self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
            self.H, self.W, self.C = self.image.size
        except:
            print("이미지 경로를 제대로 입력하세요.")
        self.xtickrange = xtickrange
        self.ytickrange = ytickrange
        
    def img2points(self, x, y, pcolor="RED", size=5, option=-1, check=True):
        if check:
            self.updated_img = self.image.copy()
        plt.grid(color="white", linestyle="dashdot", linewidth=.5)
        cv.circle(self.updated_img, (x, y), size, self.colors[pcolor], option)
        plt.xtick([num for num in range(0, self.W+1, self.xtickrange)])
        plt.ytick([num for num in range(0, self.H+1, self.ytickrange)])
        plt.imshow(self.updated_img)
        if not check:
            return self.updated_img
            ################################# 대회 끝나고 다시 정리할 부분 ###############################

    def img2rectangle(self, x1, y1, x2, y2, pcolor="RED", check=True, rect_lwidth=2):
        if check:
            self.updated_img = self.image.copy()
        plt.grid(color="white", linestyle="dashdot", linewidth=.5)
        cv.rectangle(self.updated_img, (x1, y1), (x2, y2), self.colors[pcolor], linewidth=rect_lwidth)
        plt.xtick([num for num in range(0, self.W+1, self.xtickrange)])
        plt.ytick([num for num in range(0, self.H+1, self.ytickrange)])
        plt.imshow(self.updated_img)
        if not check:
            return self.updated_img
            #################################### 대회 끝나고 다시 정리할 부분 ####################################
        
    def cvColor(self):
        # R, G, B
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        GRAY = (125, 125, 125)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        CYAN = (0, 255, 255)
        MAGENTA = (255, 0, 255)
        YELLOW = (255, 255, 0)
        PINK = (238, 130, 238)
        ORANGE = (255, 165, 0)
        MINT = (60, 179, 113)
        LAVENDER = (106, 90, 205)
        IVORY = (240, 240, 240)
        SALMON = (240, 150, 120)

        colors = {"RED":RED, "GREEN":GREEN, "BLUE":BLUE, \
                "MAGENTA":MAGENTA, "CYAN":CYAN, "YELLOW":YELLOW, \
                "WHITE":WHITE, "GRAY":GRAY, "BLACK":BLACK, \
                "PINK":PINK, "ORANGE":ORANGE, "MINT":MINT, \
                "LAVENDER":LAVENDER, "IVORY":IVORY, "SALMON":SALMON}
        return colors
    
# segmentation(polygon)은 따로 작성해야할 것 같음
class segPolygon(annotCats):
    def __init__(self, object_points=None, option="nbytwo"):
        super(segPolygon, self).__init__()
        self.seg_opt = option.lower()
        self.obj_points = self.make_general(object_points=object_points)
        
    def make_general(self, object_points=None):
        '''
        # 대회에서 제공한 segmentation의 형태가 일반적으로 사용되는 방식과 다름
        option = "nbytwo" or "twobyn"
        # 주의 : 다른 json으로 사용할 경우, 다시 뜯어서 작성해야 함
        '''
        try:
            segmt = []
            for obj in range(len(object_points)):
                if self.seg_opt=="nbytwo":
                    segmt.append(np.array(object_points[obj]['segmentation']).reshape(-1,2))
                elif self.seg_opt=="twobyn":
                    segmt.append(np.array(object_points[obj]['segmentation']).reshape(2,-1))
            return segmt     
        except:
            print(f"annotation 처리가 함수와 맞지 않습니다.")

    def points2img(self, pcolor="WHITE"):
        img = self.image.copy()
        for segs in self.obj_points:
            for pts in segs:
                cv.circle(img, (pts[0], pts[1]), 5, self.colors(pcolor.upper()), -1)
        plt.xtick([num for num in range(0, self.W+1, self.xtickrange)])
        plt.ytick([num for num in range(0, self.H+1, self.ytickrange)])
        plt.imshow(img)
        
    def points2mask(self, pcolor="WHITE"):
        mask = np.zeros((self.W, self.H, 3), dtype=np.uint8)
        for segs in self.obj_points:
            for pts in segs:
                cv.circle(mask, (pts[0], pts[1]), 5, self.colors(pcolor.upper()), -1)
        plt.xtick([num for num in range(0, self.W+1, self.xtickrange)])
        plt.ytick([num for num in range(0, self.H+1, self.ytickrange)])
        plt.imshow(mask)
        
    ############################## polygon 관련 추가로 작성해야 하는 부분이 있다면, 추가 예정 ##################################
        
    def __len__(self):
        print(f"현재 이미지 내 object의 수 : {len(self.obj_points)}")
        return len(self.obj_points)
        