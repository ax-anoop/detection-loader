import cv2 as cv
from PIL import Image, ImageDraw
import torchvision.transforms as T
import numpy as np
import torch 


class Visualize:
    def __init__(self, classes=None):
        self.classes = list(range(20)) if classes is None else classes

    def _convert_box(self, box, format="midpoints", scale=[1,1]):
        '''
        - Box formats: 
            > corners   = [x,y,x,y]
            > midpoints = [x,y,w,h]
        - Scale = [width, height] of image (use if normalized)
        - Convert box from center to corner or vice versa
        '''
        if len(box) > 4:
            box = box[1:]
        if scale != [1,1]:
            box = [box[0]*scale[0], box[1]*scale[1], box[2]*scale[0], box[3]*scale[1]]
        if format == 'midpoints':
            box = [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2]
        elif format == 'corners':
            box = [box[0], box[1], box[2], box[3]]
        return box

    def _draw_box(self, img, box, format, center=False,r=3):
        box = np.array(box)
        box = self._convert_box(box, format, scale=list(img.size))
        cen_x, cen_y = box[0] + (box[2]-box[0])/2, box[1] + (box[3]-box[1])/2
        img1 = ImageDraw.Draw(img)  
        img1.rectangle([box[0],box[1],box[2],box[3]], outline = "yellow", width = 2)
        if center:
            img1.ellipse([(cen_x-r, cen_y-r),(cen_x+r, cen_y+r)], fill=(255,0,0,255))
        return img

    def _grid_to_box(self, box):
        '''
        Transform boxes in grid space to image space
        '''
        print(box)

    def _parse_yolo(self, lbl):
        C = lbl.shape[0]
        boxes = []
        for h in range(C): 
            for w in range(C):
                y = lbl[h,w,:]
                if y.sum() == 0:
                    continue
                classes = int(np.argmax(y[0:len(self.classes)]))
                box = y[len(self.classes)+1:len(self.classes)+5]
                mid_x, mid_y = float((w + box[0])/C), float((h + box[1])/C)
                width, height = float(box[2]/C), float(box[3]/C)
                boxes.append([classes, mid_x, mid_y, width, height])
        return boxes

    def _proc_img(self, img, format, boxes):
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        elif type(img) == torch.Tensor:
            img = T.ToPILImage()(img)
        else:
            print("Image type not supported")
            return
        if boxes is not None and format is not None:
            if format == 'midpoints':
                for box in boxes:
                    img = self._draw_box(img, box, format)
            elif format == 'corners':
                for box in boxes:
                    img = self._draw_box(img, box, format)
            elif format == 'yolo':
                boxes = self._parse_yolo(boxes)
                for box in boxes:
                    img = self._draw_box(img, box, "midpoints", center=True)
        return img
    
    def show_img(self, img, format=None, boxes=None):
        img = self._proc_img(img, format, boxes)
        img.show()
        
    def draw_box(self, img, format=None, boxes=None):
        img = self._proc_img(img, format, boxes)
        return img
    
    def save_img(self, img, name, format=None, boxes=None):
        img = self._proc_img(img, format, boxes)
        img.save(name)
    
    def stack_imgs(self, imgs, horizontal=True, size=416):
        cvimgs = []
        for i in imgs: 
            if type(i) != np.ndarray:
                cvimg = cv.cvtColor(np.array(i), cv.COLOR_RGB2BGR)
                cvimg = cv.resize(cvimg, (size,size))
                cvimgs.append(cvimg)
            else:
                cvimgs.append(i)
        if horizontal:
            stack = np.hstack(cvimgs)
        else:
            stack = np.vstack(cvimgs)
        return stack

    def pil_to_cv(self, img):
        return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
        
        
def main():
    # viz = visualize.Visualize()
    # viz.save_img(img, save_img="b4_transform.png", boxes=boxes, format="midpoints", )
    pass
    
if __name__ == "__main__":
    main()