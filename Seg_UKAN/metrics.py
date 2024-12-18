import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
from metrics import compute_precision
from metrics import compute_recall

def compute_precision(output, target):
    """
    Compute precision for binary segmentation.
    Precision = TP / (TP + FP)
    """
    # Ensure output and target are binary tensors (0 or 1)
    true_positive = ((output == 1) & (target == 1)).sum().item()
    false_positive = ((output == 1) & (target == 0)).sum().item()
    
    precision = true_positive / (true_positive + false_positive + 1e-10)  # Avoid division by zero
    return precision
def compute_recall(output, target):
    """
    Compute precision for binary segmentation.
    Precision = TP / (TP + FP)
    """
    # Ensure output and target are binary tensors (0 or 1)
    true_positive = ((output == 1) & (target == 1)).sum().item()
    false_positive = ((output == 1) & (target == 0)).sum().item()
    
    recall = true_positive / (true_positive + false_positive + 1e-10)  # Avoid division by zero
    return recall



    
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    precision = compute_precision(output, target)  # Call function to get the precision value
    recall = compute_recall(output, target)  # Call function to get the recall value
    f1_ = 2 * (precision * recall) / (precision + recall + 1e-10)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_,f1_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)


    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_,f1_
