"""
Baseline inference script using U-Net semantic segmentation.

Pipeline:
1. Run U-Net to get semantic segmentation
2. Use connected components analysis to separate instances
3. Sort instances top-to-bottom, left-to-right
4. Export as tiles in different formats (rgba, binary, grayscale)
"""
import argparse
import os
import torch
import torchvision
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from models.baseline_unet import BaselineUNet
from utils.order_sort import sort_boxes_tblr


def to_tensor(img):
    """Convert PIL image to tensor"""
    return torchvision.transforms.functional.pil_to_tensor(img).float() / 255.0


def extract_instances(semantic_mask, min_size=5):
    """
    Extract individual character instances from semantic segmentation.

    Args:
        semantic_mask: Binary mask (H, W) with 0/1 values
        min_size: Minimum number of pixels for valid instance

    Returns:
        List of (instance_mask, bbox) tuples
        bbox format: (x, y, w, h)
    """
    # Label connected components
    labeled, num_features = ndimage.label(semantic_mask)

    instances = []
    for i in range(1, num_features + 1):
        instance_mask = (labeled == i).astype(np.uint8)
        pixel_count = instance_mask.sum()

        if pixel_count < min_size:
            continue

        # Get bounding box
        ys, xs = np.where(instance_mask > 0)
        x0, y0 = xs.min(), ys.min()
        x1, y1 = xs.max() + 1, ys.max() + 1
        w, h = x1 - x0, y1 - y0

        instances.append({
            "mask": instance_mask,
            "bbox": (x0, y0, w, h)
        })

    return instances


def export_tile(img_array, mask, bbox, size=16, output_format="rgba"):
    """
    Export single character tile.

    Args:
        img_array: Original RGB image as numpy array (H, W, 3)
        mask: Instance mask (H, W)
        bbox: Bounding box (x, y, w, h)
        size: Output tile size
        output_format: "rgba" | "binary" | "grayscale"

    Returns:
        PIL Image
    """
    x, y, w, h = bbox
    H, W = img_array.shape[:2]

    # Crop image and mask
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)

    if x1 <= x0 or y1 <= y0:
        # Invalid crop, return empty tile
        if output_format == "rgba":
            return Image.new("RGBA", (size, size), (255, 255, 255, 0))
        else:
            return Image.new("L", (size, size), 0)

    crop_img = img_array[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    if output_format == "rgba":
        # Create RGBA tile
        rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        rgba[:, :, :3] = crop_img
        rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)
        tile = Image.fromarray(rgba, "RGBA")
        tile = tile.resize((size, size), Image.BILINEAR)
        return tile

    elif output_format == "binary":
        # Binary mask as palette image
        mask_img = Image.fromarray((crop_mask * 255).astype(np.uint8), "L")
        mask_img = mask_img.resize((size, size), Image.BILINEAR)
        # Binarize after resize
        binary = (np.array(mask_img) > 127).astype(np.uint8) * 255
        img = Image.fromarray(binary, "L")
        return img.convert("P")

    elif output_format == "grayscale":
        # Grayscale mask
        mask_img = Image.fromarray((crop_mask * 255).astype(np.uint8), "L")
        mask_img = mask_img.resize((size, size), Image.BILINEAR)
        return mask_img

    else:
        raise ValueError(f"Unknown output_format: {output_format}")


def main():
    ap = argparse.ArgumentParser(description="Baseline inference for character extraction")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--input", required=True, help="Directory of input RGB images")
    ap.add_argument("--out", required=True, help="Output directory for tiles")
    ap.add_argument("--size", type=int, default=16, help="Output tile size")
    ap.add_argument("--format", type=str, default="rgba", choices=["rgba", "binary", "grayscale"],
                    help="Output format: rgba (4-channel), binary (1-bit palette), grayscale (8-bit)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    ap.add_argument("--min_size", type=int, default=5, help="Minimum instance size in pixels")
    ap.add_argument("--base_filters", type=int, default=32, help="Base filters (must match training)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = BaselineUNet(in_channels=3, base_filters=args.base_filters).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully!")

    # Process images
    image_files = sorted([f for f in os.listdir(args.input) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"Found {len(image_files)} images to process")

    for fname in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(args.input, fname)
        img_pil = Image.open(img_path).convert("RGB")
        img_array = np.array(img_pil)

        # Run inference
        img_t = to_tensor(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_t)[0, 0]  # (H, W)

        # Binarize prediction
        binary_mask = (pred.cpu().numpy() > args.threshold).astype(np.uint8)

        # Extract instances
        instances = extract_instances(binary_mask, min_size=args.min_size)

        if len(instances) == 0:
            print(f"Warning: No instances found in {fname}")
            continue

        # Sort instances
        bboxes = [inst["bbox"] for inst in instances]
        order = sort_boxes_tblr(bboxes, y_tol=8)

        # Export tiles
        base_name = os.path.splitext(fname)[0]
        out_dir = os.path.join(args.out, base_name)
        os.makedirs(out_dir, exist_ok=True)

        for idx, inst_idx in enumerate(order):
            inst = instances[inst_idx]
            tile = export_tile(
                img_array,
                inst["mask"],
                inst["bbox"],
                size=args.size,
                output_format=args.format
            )
            tile_path = os.path.join(out_dir, f"{idx:03d}.png")
            tile.save(tile_path)

        print(f"Exported {len(order)} tiles from {fname}")

    print(f"\nInference complete! Results saved to: {args.out}")


if __name__ == "__main__":
    main()
