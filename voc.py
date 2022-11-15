import pandas as pd 
import cv2 as cv 
import os 

import utils 

class VOC():
    def __init__(self, fdir, train=False):
        self.fdir = fdir 
        self.train = train
        self.paths = self._load_paths() #dataframe of img and lbl paths

    def _load_paths(self):
        fpath = self.fdir + "/test.csv"
        if self.train:
            fpath = self.fdir + "/test.csv"
        assert os.path.exists(fpath)
        paths = pd.read_csv(fpath, names=["img","label"],header=None)
        for index, row in paths.iterrows():
            row["img"], row["label"] = self.fdir+"/images/"+row["img"], self.fdir+"/labels/"+row["label"]
        return paths

    def _get_img(self, img_file):
        img = cv.imread(img_file)[:, :, ::-1].copy()
        return img

    def _get_lbl(self, lbl_file):
        boxes, classes = [], []
        lbl = open(lbl_file)
        for line in lbl.readlines():
            line = line.split()
            box = list(map(float, line))
            boxes.append(box[1:])
            classes.append(box[0])
        return boxes, classes 

    def pull_item(self, idx, bformat='xywh'):
        '''
        returns: img, box, class
        box = [x1, y1, w, h]
        img = native shape, no transforms
        '''
        img = self._get_img(self.paths.iloc[[idx]].img.item())
        boxes, classes = self._get_lbl(self.paths.iloc[[idx]].label.item())
        if bformat == 'xyxy':
            boxes = utils.get_xyxy(boxes)
        return img, boxes, classes

if __name__ == '__main__':
    v = VOC("/home/server/Desktop/data/pascal_voc")
    img, boxes, classes = v.pull_item(0)
    print(img.shape, boxes, classes)
    img, boxes, classes = v.pull_item(0, bformat='xyxy')
    print(img.shape, boxes, classes)