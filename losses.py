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
    def __init__(self, smooth=1e-6, weight=100.0):
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

#tackles class imbalance by combining focal loss and dice loss
class UnifiedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, lambda_focal=0.5, smooth=1e-6):
        super(UnifiedFocalLoss, self).__init__()
        self.gamma = gamma
        self.lambda_focal = lambda_focal
        self.smooth = smooth

    def forward(self, preds, targets):
        # 1. Focal Component
        preds = torch.clamp(preds, self.smooth, 1.0 - self.smooth)
        bce = - (targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
        focal_weight = torch.pow(torch.abs(targets - preds), self.gamma)
        focal_loss = (focal_weight * bce).mean()

        # 2. Dice Component
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)

        # 3. Combined Unified Loss
        return (self.lambda_focal * focal_loss) + ((1 - self.lambda_focal) * dice_loss)