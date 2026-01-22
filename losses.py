import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        # Flatten both tensors to (Batch, -1) to calculate intersection per cube
        predict = predict.view(-1)
        target = target.view(-1)
        
        intersection = (predict * target).sum()
        dice = (2. * intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
        
        return 1 - dice
    
class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, weight=10.0):
        super(WeightedDiceLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight

    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        
        # Multiply the intersection by a weight to penalize missing the nodule more heavily
        intersection = (predict * target).sum()
        dice = (2. * self.weight * intersection + self.smooth) / \
               (self.weight * predict.sum() + target.sum() + self.smooth)
        
        return 1 - dice