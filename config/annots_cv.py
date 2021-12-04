import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

class annotCats:
    def __init__(self, image_path=None, label=None):
        self.colors = self.cvColor()
        self.label = label
        try:
            self.image = Image.open(image_path)
            self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        except:
            print("이미지 경로를 제대로 입력하세요.")
        
    def img2points(self, x, y, pcolor="RED", size=5, option=-1, check=True):
        if check:
            self.updated_img = self.image.copy()
        plt.grid(color=self.colors["WHITE"], linestyle="dashdot", linewidth=.5)
        cv.circle(self.updated_img, (x, y), size, self.colors[pcolor], option)
        plt.imshow(self.updated_img)
        if not check:
            return self.updated_img

    def img2rectangle(self, x1, y1, x2, y2, pcolor="RED", check=True, rect_lwidth=2):
        if check:
            self.updated_img = self.image.copy()
        plt.grid(color=self.colors["WHITE"], linestyle="dashdot", linewidth=.5)
        cv.rectangle(self.updated_img, (x1, y1), (x2, y2), self.colors[pcolor], linewidth=rect_lwidth)
        plt.imshow(self.updated_img)
        if not check:
            return self.updated_img 
        
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
# class segPolygon:
# 