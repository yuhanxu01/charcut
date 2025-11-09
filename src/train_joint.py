import argparse, os, torch, torchvision, numpy as np
from torch.utils.data import DataLoader
from dataset import ParagraphDataset
from models.build_maskrcnn import build_maskrcnn
from models.refine_unet import RefineUNet
from models.matcher import hungarian_match
from utils.losses import dice_loss, recon_loss_full
from tqdm import tqdm
from PIL import Image

def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return list(imgs), list(tgts)

def crop_from_image(img, box):
    x0,y0,x1,y1 = [int(v) for v in box]
    x0,y0 = max(0,x0), max(0,y0)
    return img[:, y0:y1, x0:x1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_refine", type=float, default=1e-3)
    ap.add_argument("--ckpt_stageA", default="runs/stageA/last.pt")
    ap.add_argument("--out", default="runs/joint")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = ParagraphDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maskrcnn = build_maskrcnn(num_classes=2, pretrained=False).to(device)
    if os.path.exists(args.ckpt_stageA):
        maskrcnn.load_state_dict(torch.load(args.ckpt_stageA, map_location=device))
    maskrcnn.train()

    refine = RefineUNet(in_ch=4, base=32).to(device)
    opt = torch.optim.AdamW([p for p in maskrcnn.parameters() if p.requires_grad], lr=args.lr)
    opt_ref = torch.optim.AdamW([p for p in refine.parameters() if p.requires_grad], lr=args.lr_refine)

    for ep in range(args.epochs):
        pbar = tqdm(dl, desc=f"ep{ep}")
        for imgs_pil, targets in pbar:
            imgs = [torchvision.transforms.functional.pil_to_tensor(x).float().to(device)/255. for x in imgs_pil]
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k,v in t.items()} for t in targets]

            # Stage A supervised losses
            loss_dict = maskrcnn(imgs, targets)
            loss_A = sum(loss_dict.values())

            # Inference coarse masks for refinement targets
            with torch.no_grad():
                maskrcnn.eval()
                preds = maskrcnn(imgs)
                maskrcnn.train()

            # Build matching and refine losses per image
            loss_alpha = torch.tensor(0.0, device=device)
            loss_dice = torch.tensor(0.0, device=device)
            loss_recon = torch.tensor(0.0, device=device)

            for b in range(len(imgs)):
                pred_masks = preds[b].get("masks", torch.zeros((0,1,1,1), device=device))  # (N,1,H,W)
                pred_boxes = preds[b].get("boxes", torch.zeros((0,4), device=device))
                pm_np = [pred_masks[i,0].detach().cpu().numpy() > 0.5 for i in range(pred_masks.shape[0])]
                gt_masks = targets[b]["masks"].detach().cpu().numpy().astype(bool)
                pairs = hungarian_match(pm_np, gt_masks, max_cost=0.99)

                # full-image tensors
                H, W = imgs[b].shape[1:]
                K = len(pairs)
                if K == 0:
                    continue
                alphas_full = []
                colors_full = []

                for pi, gi in pairs:
                    box = pred_boxes[pi]
                    # crop RGB and coarse mask to same ROI
                    rgb_crop = crop_from_image(imgs[b], box)
                    coarse = crop_from_image(pred_masks[pi], box)
                    if rgb_crop.numel() == 0 or coarse.numel() == 0:
                        continue
                    srgb = torch.nn.functional.interpolate(rgb_crop.unsqueeze(0), size=(64,64), mode="bilinear", align_corners=False)
                    scoarse = torch.nn.functional.interpolate(coarse.unsqueeze(0), size=(64,64), mode="bilinear", align_corners=False)
                    refine_in = torch.cat([srgb, scoarse], dim=1)  # (1,4,64,64)
                    alpha = refine(refine_in)  # (1,1,64,64)
                    # upsample alpha back to ROI size
                    alpha_up = torch.nn.functional.interpolate(alpha, size=(rgb_crop.shape[1], rgb_crop.shape[2]), mode="bilinear", align_corners=False)[0,0]
                    # accumulate losses
                    gt_full = targets[b]["masks"][gi].float().to(device)
                    gt_roi = crop_from_image(gt_full.unsqueeze(0), box)[0:1]  # (1,h,w)
                    gtroi_res = torch.nn.functional.interpolate(gt_roi.unsqueeze(0), size=alpha.shape[-2:], mode="nearest")[0:1]
                    loss_alpha = loss_alpha + torch.nn.functional.l1_loss(alpha, gtroi_res)
                    loss_dice = loss_dice + dice_loss(alpha, gtroi_res)

                    # collect for reconstruction
                    # simple color model: black text on white => use rgb crop directly
                    # compose into full image-sized tensors
                    a_full = torch.zeros((H,W), device=device)
                    c_full = torch.zeros((3,H,W), device=device)
                    x0,y0,x1,y1 = [int(v.item()) for v in box]
                    a_full[y0:y1, x0:x1] = alpha_up
                    c_full[:, y0:y1, x0:x1] = rgb_crop
                    alphas_full.append(a_full)
                    colors_full.append(c_full)

                if alphas_full:
                    A = torch.stack(alphas_full, dim=0)  # (K,H,W)
                    # softmax allocation across instances
                    A = torch.softmax(A, dim=0)
                    A = A.unsqueeze(1)  # (K,1,H,W)
                    C = torch.stack(colors_full, dim=0)  # (K,3,H,W)
                    loss_recon = loss_recon + recon_loss_full(imgs[b].unsqueeze(0), A.unsqueeze(0), C.unsqueeze(0))

            loss = loss_A + 5.0*loss_alpha + 1.0*loss_dice + 3.0*loss_recon
            opt.zero_grad()
            opt_ref.zero_grad()
            loss.backward()
            opt.step()
            opt_ref.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()), A=float(loss_A.detach().cpu()))
        torch.save({
            "maskrcnn": maskrcnn.state_dict(),
            "refine": refine.state_dict()
        }, os.path.join(args.out, f"epoch{ep}.pt"))
    torch.save({
        "maskrcnn": maskrcnn.state_dict(),
        "refine": refine.state_dict()
    }, os.path.join(args.out, "joint_last.pt"))

if __name__ == "__main__":
    main()
