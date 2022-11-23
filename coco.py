from pycocotools.coco import COCO

import pandas as pd 
import cv2 as cv
import json
import os

import utils 
import sys, os

# Don't like hidden prints in coco :/
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
class CocoDetection():
    def __init__(self, root, train=False):
        self.train = train
        self.ann_file = root + "/annotations/instances_val2017.json"
        self.root = root + "/images/" + "val2017"
        if self.train:
            self.ann_file = root + "/annotations/instances_train2017.json"
            self.root = root + "/images/" + "train2017"
        with HiddenPrints():
            self.coco = COCO(self.ann_file)
        self.ids = list(self.coco.imgs.keys())
        
    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = self._get_img(os.path.join(self.root, path))
        boxes, classes = self._get_labels(target, img.shape)
        return img, boxes, classes

    def _get_img(self, img_file):
        img = cv.imread(img_file)[:, :, ::-1].copy()
        return img

    def _get_labels(self, targets, ims):
        boxes, classes = [],[]
        for t in targets:
            boxes.append([t["bbox"][0]/ims[0],t["bbox"][1]/ims[1],t["bbox"][2]/ims[0],t["bbox"][3]/ims[1]])
            classes.append(t["category_id"])
        return boxes, classes

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    v = CocoDetection("/home/server/Desktop/data/coco")
    print(len(v))
    img, boxes, classes = v.pull_item(0)
    # print(boxes[0], classes[0])