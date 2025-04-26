import torch
import torch.nn as nn


class GenLoss(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambd = lambd
    
    def forward(self, dis_output, gen_mask, target_mask):
        loss = self.bce(dis_output, torch.ones_like(dis_output))
        # Regularized loss necessary to ensure correctness on top of realism
        reg_loss = torch.mean(torch.abs(target_mask - gen_mask))
        total_loss = loss + self.lambd * reg_loss

        return total_loss
    

class DiscLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, dis_real_output, disc_gen_output):
        real_loss = self.bce(dis_real_output, torch.ones_like(dis_real_output))
        gen_loss = self.bce(disc_gen_output, torch.zeros_like(disc_gen_output))

        return real_loss + gen_loss
