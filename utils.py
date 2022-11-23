import numpy as np 
import torch

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

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

'''
Overall model structure:
    1. There will be 3 output heads 
    2. Each head will have 3 anchors, or 3 outputs [batch, 3, S, S, 6]

Looking at a single head:
    1. We take the detection box, and find the anchor that has the highest IOU
    2. Now for that paticular anchor, we add box into grid matrix & only for that anchor. 

Function: 
    For each box:
        For each head:
            For each anchor:                                                                    #
                - Check IOU with box                                                            #
                    - If MAX IOU (from all anchors), add box to grid matrix                     #
                - If IoU > 0.5, add -1 to the grid matrix                                       # Ignore prediction. Essentially we are saying that, if box already allocated to anchor, but the overlap with another anchor is > 0.5, write -1 so we don't 'punish this anchor' in loss. 
'''
def box_to_anchors(boxes, classes, anchors, SA=[13, 16, 52], ignore_iou_thresh=0.5):
    num_anchors = anchors.shape[0]
    num_anchors_per_scale = num_anchors // 3
    
    targets = [torch.zeros((num_anchors // 3, S, S, 6)) for S in SA]
    for i, box in enumerate(boxes):
        iou_anchors = iou_width_height(torch.tensor(box[2:4]), anchors)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, width, height = box
        class_label = classes[i]
        head_has_anchor = [False] * 3
        
        for anchor_idx in anchor_indices:
            scale_idx = torch.div(anchor_idx, num_anchors_per_scale, rounding_mode='trunc')            
            anchor_on_scale = anchor_idx % num_anchors_per_scale
            S = SA[scale_idx]
            i, j = int(S * y), int(S * x)  # which cell
            # should be 'cell_taken', essentially if anchor 
            cell_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
            # print(scale_idx, cell_taken, head_has_anchor[scale_idx], iou_anchors[anchor_idx])
            if not cell_taken and not head_has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                width_cell, height_cell = (
                    width * S,
                    height * S,
                )  # can be greater than 1 since it's relative to cell
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                head_has_anchor[scale_idx] = True
            elif not cell_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
    return targets
