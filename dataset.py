import os
import torch
try:
    from . import voc 
    from . import coco
except: 
    import voc
    import coco 
'''
- Native VOC format is [x_min, y_min, x_max, y_max]
- Native coco format is [x_min, y_min, w, h]
'''   
class DataSet():
    def __init__(self, fdir, train=False, datasets=["voc"], bformat="xywh", size=416, transform=None, box_transform=None, box_transform_args={}):
        self.fdir = fdir
        self.train = train
        self.size = size
        # Things for the datasets 
        self.lens, self.dsnames = [], []
        self.datasets = self.load_datasets(datasets)
        self.transform = transform
        self.box_transform = box_transform
        self.box_kwargs = box_transform_args
        
    def load_datasets(self, ds):
        datasets = {}
        for d in ds:
            if d == "voc":
                path = os.path.join(self.fdir, "pascal_voc")
                dataset = voc.VOC(path, self.train)
            elif d == "coco":
                path = os.path.join(self.fdir, "coco")
                dataset = coco.CocoDetection(path, self.train)
            else:
                print("dataset not supported")
                raise NotImplementedError
            self.dsnames.append(d)
            self.lens.append(len(dataset))
            datasets[d] = dataset
        return datasets
    
    def __len__(self):
        return int(sum(self.lens))
    
    def _match_idx(self, idx):
        d = 0
        for l in self.lens:
            if idx < l:
                break
            d += 1
        return self.dsnames[d] 
    
    def __getitem__(self, idx):
        dataset_name = self._match_idx(idx)
        img, bbox, cls =  self.datasets[dataset_name].pull_item(idx)
        # print("getting item.", bbox, idx)
        if self.transform:
            augmentations = self.transform(image=img, bboxes=bbox)
            img = augmentations["image"]
            bbox = augmentations["bboxes"]
        transformed_bbox = bbox
        if self.box_transform:
            transformed_bbox = self.box_transform(bbox, cls, **self.box_kwargs)
        # return img, torch.tensor(bbox), transformed_bbox, cls
        return img, transformed_bbox

if __name__ == '__main__':
    dl = DataSet("/home/server/Desktop/data", datasets=["voc","coco"])