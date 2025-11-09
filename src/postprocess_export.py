import argparse, os, torch, torchvision
from PIL import Image
import numpy as np
from models.build_maskrcnn import build_maskrcnn
from models.refine_unet import RefineUNet
from utils.order_sort import sort_boxes_tblr

def to_tensor(img):
    return torchvision.transforms.functional.pil_to_tensor(img).float()/255.

def crop(img_t, box):
    x0,y0,x1,y1 = [int(v) for v in box]
    return img_t[:, y0:y1, x0:x1]

def export_tile(rgb_crop, alpha, size=16):
    c, h, w = rgb_crop.shape
    alpha = alpha.clamp(0,1)
    rgba = torch.cat([rgb_crop, alpha.unsqueeze(0)], dim=0)  # 4xhxw
    rgba = torch.nn.functional.interpolate(rgba.unsqueeze(0), size=(size,size), mode="bilinear", align_corners=False)[0]
    arr = (rgba.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="dir of RGB images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    maskrcnn = build_maskrcnn(num_classes=2, pretrained=False).to(device).eval()
    refine = RefineUNet(in_ch=4, base=32).to(device).eval()
    if "maskrcnn" in ckpt:
        maskrcnn.load_state_dict(ckpt["maskrcnn"])
        refine.load_state_dict(ckpt["refine"])
    else:
        maskrcnn.load_state_dict(ckpt)

    for fname in sorted(os.listdir(args.input)):
        if not fname.lower().endswith(".png"):
            continue
        img_pil = Image.open(os.path.join(args.input, fname)).convert("RGB")
        img_t = to_tensor(img_pil).to(device)
        with torch.no_grad():
            preds = maskrcnn([img_t])[0]
        boxes = preds["boxes"].detach().cpu().numpy().tolist()
        masks = preds["masks"][:,0]
        order = sort_boxes_tblr([(x0,y0,x1-x0,y1-y0) for x0,y0,x1,y1 in boxes], y_tol=8)
        tiles = []
        for k in order:
            b = preds["boxes"][k]
            m = masks[k].unsqueeze(0)
            rgb = crop(img_t, b)
            if rgb.numel() == 0:
                continue
            coarse = crop(m, b)
            if coarse.numel() == 0:
                continue
            srgb = torch.nn.functional.interpolate(rgb.unsqueeze(0), size=(64,64), mode="bilinear", align_corners=False)
            scoarse = torch.nn.functional.interpolate(coarse.unsqueeze(0), size=(64,64), mode="bilinear", align_corners=False)
            a = refine(torch.cat([srgb, scoarse], dim=1))[0,0]
            a_up = torch.nn.functional.interpolate(a.unsqueeze(0).unsqueeze(0), size=(rgb.shape[1], rgb.shape[2]), mode="bilinear", align_corners=False)[0,0]
            tile = export_tile(rgb, a_up, size=args.size)
            tiles.append(tile)
        # write tiles with order index
        base = os.path.splitext(fname)[0]
        out_dir = os.path.join(args.out, base)
        os.makedirs(out_dir, exist_ok=True)
        for i, t in enumerate(tiles):
            t.save(os.path.join(out_dir, f"{i:03d}.png"))
        print("exported", fname, len(tiles))

if __name__ == "__main__":
    main()
