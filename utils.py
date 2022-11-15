import numpy as np 

# convert boxes [N X 4], xywh to xyxy 
def get_xyxy(boxes):
    n_box = []
    for box in boxes:
        n_box.append([box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2])
    return n_box 

# convert boxes [N X 4], xyxy to xywh
def get_xywh(boxes):
    n_box = []
    for box in boxes:
        n_box.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
    return n_box