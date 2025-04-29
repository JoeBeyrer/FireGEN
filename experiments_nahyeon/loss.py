import torch
import torch.nn as nn

"""class GenLoss(nn.Module):
    def __init__(self, alpha=1.0, lambd=1.0, pos_weight=None, bce_weight=0.5):
        super().__init__()
        # 재구성용 BCE
        self.recon_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # 판별기 로짓에 대한 BCE (–log(1–D))
        self.adv_bce   = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.lambd      = lambd  # adversarial loss weight
        self.alpha      = alpha  # reconstruction vs adversarial balance

    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1.0 - (2.0 * inter + smooth) / (union + smooth)

    def forward(self, gen_logits, disc_pred, target_mask):
        """"""
        gen_logits:  generator 출력 (raw logits before sigmoid)
        disc_pred:   discriminator(gen_logits, x) 의 raw logits
        target_mask: ground-truth mask
        """"""
        # 1) 재구성 손실: BCE + Dice
        bce_recon  = self.recon_bce(gen_logits, target_mask)
        dice_recon = self.dice_loss(gen_logits, target_mask)
        recon_loss = self.bce_weight * bce_recon + (1-self.bce_weight) * dice_recon

        # 2) 적대적 손실: –log(1 – D(x, G(x)))
        #    D 로부터 바로 sigmoid 하지 않은 raw logit 을 받고,
        #    target = 1 으로 설정하면 BCEWithLogitsLoss는
        #    –log(σ(disc_pred)) 를 계산하므로, 여기서는 fake label=1 로 줘야
        #    –log(1 – σ(disc_pred)) 항이 된다.
        #    (torch treats label=1 as positive class → uses logit)
        fake_labels = torch.ones_like(disc_pred)
        adv_loss    = self.adv_bce(disc_pred, fake_labels)

        # 3) 최종 합산
        total_loss = recon_loss + self.lambd * adv_loss
        return total_loss"""


"""class GenLoss(nn.Module):
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

    
    def forward(self, gen_mask, target_mask):
        #loss = self.bce(dis_output, torch.ones_like(dis_output))

        # Weighted BCE for class imbalance
        bce_loss = self.bce(gen_mask, target_mask)  
        # Dice loss for region overlap
        dice_loss = self.dice_loss(gen_mask, target_mask)  

        total_loss = self.bce_weight*bce_loss + (1-self.bce_weight)*dice_loss

        return total_loss"""
    
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
