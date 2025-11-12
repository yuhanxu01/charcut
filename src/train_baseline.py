"""
Simple baseline training script using U-Net for semantic segmentation.
Much simpler than the two-stage Mask R-CNN + Refinement pipeline.

Pipeline:
1. Load 256x256 RGB images
2. Merge all character masks into single semantic segmentation map
3. Train U-Net with BCE + Dice loss
4. No instance matching, no reconstruction loss
"""
import argparse
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
import json

from models.baseline_unet import BaselineUNet


class SemanticDataset(Dataset):
    """
    Convert instance masks to semantic segmentation.
    Returns RGB image and binary segmentation map (character vs background).
    """
    def __init__(self, root_dir):
        ann_path = os.path.join(root_dir, "annotations.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.root = root_dir
        self.images = data["images"]
        self.ann = data["annotations"]

        # Build index
        self.by_image = {}
        for a in self.ann:
            self.by_image.setdefault(a["image_id"], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        img_path = os.path.join(self.root, item["file_name"])
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Create merged semantic mask
        semantic_mask = np.zeros((H, W), dtype=np.float32)
        insts = self.by_image.get(item["id"], [])

        for a in insts:
            mpath = os.path.join(self.root, a["mask_path"])
            m = Image.open(mpath).convert("L")
            m_np = (np.array(m) > 127).astype(np.float32)
            # Union all instance masks
            semantic_mask = np.maximum(semantic_mask, m_np)

        # Convert to tensors
        img_t = torchvision.transforms.functional.pil_to_tensor(img).float() / 255.0
        mask_t = torch.from_numpy(semantic_mask).unsqueeze(0)  # (1, H, W)

        return img_t, mask_t


def dice_loss(pred, target, smooth=1e-6):
    """Dice loss for segmentation"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def combined_loss(pred, target):
    """BCE + Dice loss"""
    bce = nn.functional.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")

    for imgs, masks in pbar:
        imgs = imgs.to(device)
        masks = masks.to(device)

        # Forward
        preds = model(imgs)
        loss = combined_loss(preds, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(dataloader, desc="Validation"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = combined_loss(preds, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    ap = argparse.ArgumentParser(description="Train baseline U-Net for character segmentation")
    ap.add_argument("--data", required=True, help="Path to dataset directory")
    ap.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    ap.add_argument("--batch", type=int, default=4, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--base_filters", type=int, default=32, help="Base number of filters in U-Net")
    ap.add_argument("--out", default="runs/baseline", help="Output directory for checkpoints")
    ap.add_argument("--num_workers", type=int, default=2, help="Number of data loading workers")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    print("Loading dataset...")
    train_ds = SemanticDataset(args.data)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    print(f"Dataset size: {len(train_ds)} images")

    # Model
    print("Creating model...")
    model = BaselineUNet(in_channels=3, base_filters=args.base_filters).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.6f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.out, f"epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(args.out, "best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "train_loss": train_loss,
            }, best_path)
            print(f"Best model saved: {best_path}")

    # Save final model
    final_path = os.path.join(args.out, "final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
    }, final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")


if __name__ == "__main__":
    main()
