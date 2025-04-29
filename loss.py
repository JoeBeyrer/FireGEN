import torch
import torch.nn as nn


class GenLoss(nn.Module):
    def __init__(self, alpha=1, lambd=2, pos_weight=None, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_weight = bce_weight
        self.lambd = lambd
        self.alpha = alpha
    
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)  
        smooth = 1e-6  # Avoid division by zero
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1.0 - (2.0 * intersection + smooth) / (union + smooth)

    
    def forward(self, dis_output, gen_mask, target_mask):
        loss = self.bce(dis_output, torch.ones_like(dis_output))

        # Weighted BCE for class imbalance
        bce_loss = self.bce(gen_mask, target_mask)  
        # Dice loss for region overlap
        dice_loss = self.dice_loss(gen_mask, target_mask)  

        total_loss = self.alpha*loss + self.lambd*(self.bce_weight*bce_loss + (1-self.bce_weight)*dice_loss)

        return total_loss
    

class DiscLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, dis_real_output, disc_gen_output):
        real_loss = self.bce(dis_real_output, torch.ones_like(dis_real_output))
        gen_loss = self.bce(disc_gen_output, torch.zeros_like(disc_gen_output))

        return real_loss + gen_loss
