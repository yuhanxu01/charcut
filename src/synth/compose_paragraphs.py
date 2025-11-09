import argparse, os, json, random, numpy as np
from PIL import Image
from utils.render_utils import random_affine, paste_rgba, add_canvas_noise

def pick_sprite_dir(sprites_root):
    chars = [d for d in os.listdir(sprites_root) if d.startswith("U")]
    ch = random.choice(chars)
    ch_dir = os.path.join(sprites_root, ch)
    files = [os.path.join(ch_dir, f) for f in os.listdir(ch_dir) if f.endswith(".png")]
    return ch, random.choice(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sprites", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_images", type=int, default=1000)
    ap.add_argument("--canvas", type=int, default=256)
    ap.add_argument("--line_height", type=int, default=18)
    ap.add_argument("--p_overlap", type=float, default=0.3)
    args = ap.parse_args()
    img_dir = os.path.join(args.out, "images")
    msk_dir = os.path.join(args.out, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    ann = {"images":[], "annotations":[]}
    ann_id = 1
    for i in range(args.num_images):
        canvas = Image.new("RGBA", (args.canvas, args.canvas), (255,255,255,255))
        x, y = 4, 4
        insts = []
        count = 0
        while y < args.canvas - args.line_height - 4:
            ch, spr_path = pick_sprite_dir(args.sprites)
            spr = Image.open(spr_path).convert("RGBA")
            angle = random.uniform(-6, 6)
            scale = random.uniform(0.9, 1.1)
            spr2 = random_affine(spr, angle, scale)
            if x + spr2.width > args.canvas - 4:
                x = 4
                y += args.line_height + random.randint(-2, 3)
                if y >= args.canvas - args.line_height - 4:
                    break
            dx = random.randint(-2,2)
            dy = random.randint(-2,2)
            pos = (x+dx, y+dy)
            paste_rgba(canvas, spr2, pos)
            # mask for this instance
            mask_full = Image.new("L", (args.canvas, args.canvas), 0)
            alpha = spr2.split()[-1]
            mask_full.paste(alpha, pos)
            bbox = [pos[0], pos[1], spr2.width, spr2.height]
            mpath = os.path.join(msk_dir, f"{i:06d}_{count:03d}.png")
            mask_full.save(mpath)
            insts.append({"bbox": bbox, "mask_path": os.path.relpath(mpath, args.out), "char": ch, "sprite": os.path.relpath(spr_path, args.out)})
            x += int(spr2.width * (0.92 if random.random() < args.p_overlap else 1.0))
            count += 1
        canvas = add_canvas_noise(canvas, sigma=4.0)
        img_path = os.path.join(img_dir, f"{i:06d}.png")
        canvas.convert("RGB").save(img_path)
        ann["images"].append({"file_name": os.path.relpath(img_path, args.out), "id": i, "width": args.canvas, "height": args.canvas})
        for ins in insts:
            ins["image_id"] = i
            ins["id"] = ann_id
            ann_id += 1
            ann["annotations"].append(ins)
    with open(os.path.join(args.out, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    print("done:", args.out)

if __name__ == "__main__":
    main()
