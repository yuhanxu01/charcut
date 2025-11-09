import torch
import torch.nn.functional as F

def dice_loss(pred, target, eps=1e-6):
    # pred, target in [0,1], shape (N,1,H,W)
    num = 2 * (pred * target).sum(dim=(1,2,3))
    den = (pred + target).sum(dim=(1,2,3)) + eps
    return 1 - (num + eps) / den

def recon_loss_full(image, alphas, colors, eps=1e-6):
    # image: (B,3,H,W), alphas: (B,K,1,H,W) in [0,1], colors: (B,K,3,H,W)
    comp = (alphas * colors).sum(dim=1)  # (B,3,H,W)
    return F.l1_loss(comp, image)
