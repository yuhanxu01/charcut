"""
Visualization tools for model predictions.

Creates side-by-side comparisons of:
- Original image
- Ground truth mask
- Predicted mask
- Overlay comparison
"""
import argparse
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm


def create_overlay(img, mask, color=(255, 0, 0), alpha=0.5):
    """Create overlay of mask on image"""
    img_array = np.array(img).copy()
    mask_array = (np.array(mask) > 127).astype(np.uint8)

    # Create colored mask
    overlay = np.zeros_like(img_array)
    overlay[mask_array > 0] = color

    # Blend
    result = cv2.addWeighted(img_array, 1 - alpha, overlay, alpha, 0)
    return Image.fromarray(result)


def visualize_comparison(img_path, gt_mask_path, pred_mask_path, output_path):
    """Create side-by-side comparison visualization"""
    # Load images
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    gt_mask = Image.open(gt_mask_path).convert("L") if os.path.exists(gt_mask_path) else Image.new("L", (W, H), 0)
    pred_mask = Image.open(pred_mask_path).convert("L") if os.path.exists(pred_mask_path) else Image.new("L", (W, H), 0)

    # Create overlays
    gt_overlay = create_overlay(img, gt_mask, color=(0, 255, 0), alpha=0.4)
    pred_overlay = create_overlay(img, pred_mask, color=(255, 0, 0), alpha=0.4)

    # Difference map
    gt_array = (np.array(gt_mask) > 127).astype(np.uint8)
    pred_array = (np.array(pred_mask) > 127).astype(np.uint8)

    # True positive (white), False positive (red), False negative (blue)
    diff = np.zeros((H, W, 3), dtype=np.uint8)
    tp = np.logical_and(gt_array, pred_array)
    fp = np.logical_and(pred_array, ~gt_array.astype(bool))
    fn = np.logical_and(~pred_array.astype(bool), gt_array)

    diff[tp] = [255, 255, 255]  # White - correct
    diff[fp] = [255, 0, 0]      # Red - false positive
    diff[fn] = [0, 0, 255]      # Blue - false negative
    diff_img = Image.fromarray(diff)

    # Create grid
    grid_w = W * 3
    grid_h = H * 2
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    # Place images
    grid.paste(img, (0, 0))
    grid.paste(gt_overlay, (W, 0))
    grid.paste(pred_overlay, (W * 2, 0))

    grid.paste(gt_mask.convert("RGB"), (0, H))
    grid.paste(pred_mask.convert("RGB"), (W, H))
    grid.paste(diff_img, (W * 2, H))

    # Add labels
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    labels = [
        (10, 10, "Original"),
        (W + 10, 10, "GT Overlay"),
        (W * 2 + 10, 10, "Pred Overlay"),
        (10, H + 10, "GT Mask"),
        (W + 10, H + 10, "Pred Mask"),
        (W * 2 + 10, H + 10, "Diff (W:TP R:FP B:FN)")
    ]

    for x, y, text in labels:
        # Draw text with background
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    grid.save(output_path)


def visualize_dataset(img_dir, gt_root, pred_dir, output_dir, max_images=20):
    """Create visualizations for multiple images"""
    os.makedirs(output_dir, exist_ok=True)

    # Load GT annotations
    ann_path = os.path.join(gt_root, "annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    print(f"Creating visualizations (max {max_images} images)...")
    count = 0

    for img_info in tqdm(gt_data["images"][:max_images]):
        img_file = os.path.basename(img_info["file_name"])
        img_path = os.path.join(img_dir, img_file)

        if not os.path.exists(img_path):
            continue

        # Create merged GT mask
        H, W = img_info["height"], img_info["width"]
        gt_mask = np.zeros((H, W), dtype=np.uint8)

        for ann in gt_data["annotations"]:
            if ann["image_id"] == img_info["id"]:
                mask_path = os.path.join(gt_root, ann["mask_path"])
                if os.path.exists(mask_path):
                    m = np.array(Image.open(mask_path).convert("L"))
                    gt_mask = np.maximum(gt_mask, m)

        gt_mask_path = os.path.join(output_dir, f"temp_gt_{img_file}")
        Image.fromarray(gt_mask).save(gt_mask_path)

        # Pred mask path
        pred_mask_path = os.path.join(pred_dir, img_file)

        # Create visualization
        output_path = os.path.join(output_dir, f"comparison_{img_file}")
        visualize_comparison(img_path, gt_mask_path, pred_mask_path, output_path)

        # Clean up temp file
        os.remove(gt_mask_path)
        count += 1

    print(f"Created {count} visualizations in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument("--images", required=True, help="Directory with original images")
    parser.add_argument("--gt", required=True, help="Ground truth dataset root")
    parser.add_argument("--pred", required=True, help="Directory with predictions")
    parser.add_argument("--out", required=True, help="Output directory for visualizations")
    parser.add_argument("--max", type=int, default=20, help="Maximum number of images to visualize")
    args = parser.parse_args()

    visualize_dataset(args.images, args.gt, args.pred, args.out, args.max)


if __name__ == "__main__":
    main()
