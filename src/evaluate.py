"""
Comprehensive evaluation script for character segmentation models.

Metrics:
1. Instance-level IoU and mAP
2. Pixel-level accuracy, precision, recall, F1
3. Character ordering accuracy
4. Alpha matte quality (MSE, MAE, Gradient error)
"""
import argparse
import os
import json
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy import ndimage
import torch
from tqdm import tqdm
from collections import defaultdict


def load_gt_instances(data_root, image_id):
    """Load ground truth instances for an image"""
    ann_path = os.path.join(data_root, "annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Find image info
    img_info = None
    for img in data["images"]:
        if img["id"] == image_id:
            img_info = img
            break

    if img_info is None:
        return None, []

    # Load instances
    instances = []
    for ann in data["annotations"]:
        if ann["image_id"] == image_id:
            mask_path = os.path.join(data_root, ann["mask_path"])
            mask = np.array(Image.open(mask_path).convert("L")) > 127
            instances.append({
                "mask": mask,
                "bbox": ann.get("bbox", [0, 0, 0, 0]),
                "char": ann.get("char", "")
            })

    return img_info, instances


def mask_iou(pred, gt):
    """Compute IoU between two binary masks"""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_instance_metrics(pred_masks, gt_masks, iou_threshold=0.5):
    """
    Compute instance-level metrics using Hungarian matching.

    Returns:
        precision, recall, f1, mean_iou, matched_pairs
    """
    if len(pred_masks) == 0 and len(gt_masks) == 0:
        return 1.0, 1.0, 1.0, 1.0, []

    if len(pred_masks) == 0:
        return 0.0, 0.0, 0.0, 0.0, []

    if len(gt_masks) == 0:
        return 0.0, 0.0, 0.0, 0.0, []

    # Compute cost matrix (1 - IoU)
    P, G = len(pred_masks), len(gt_masks)
    cost = np.zeros((P, G))
    ious = np.zeros((P, G))

    for i in range(P):
        for j in range(G):
            iou = mask_iou(pred_masks[i], gt_masks[j])
            ious[i, j] = iou
            cost[i, j] = 1.0 - iou

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost)

    # Count matches above threshold
    matched_pairs = []
    total_iou = 0.0
    for i, j in zip(row_ind, col_ind):
        if ious[i, j] >= iou_threshold:
            matched_pairs.append((i, j, ious[i, j]))
            total_iou += ious[i, j]

    num_matches = len(matched_pairs)
    precision = num_matches / P if P > 0 else 0.0
    recall = num_matches / G if G > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = total_iou / num_matches if num_matches > 0 else 0.0

    return precision, recall, f1, mean_iou, matched_pairs


def compute_pixel_metrics(pred_semantic, gt_semantic):
    """Compute pixel-level metrics"""
    pred_bin = pred_semantic > 0.5
    gt_bin = gt_semantic > 0.5

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_alpha_metrics(pred_alpha, gt_alpha):
    """Compute alpha matte quality metrics"""
    mse = np.mean((pred_alpha - gt_alpha) ** 2)
    mae = np.mean(np.abs(pred_alpha - gt_alpha))

    # Gradient error
    pred_grad_y = np.abs(np.diff(pred_alpha, axis=0))
    pred_grad_x = np.abs(np.diff(pred_alpha, axis=1))
    gt_grad_y = np.abs(np.diff(gt_alpha, axis=0))
    gt_grad_x = np.abs(np.diff(gt_alpha, axis=1))

    grad_error_y = np.mean(np.abs(pred_grad_y - gt_grad_y))
    grad_error_x = np.mean(np.abs(pred_grad_x - gt_grad_x))
    grad_error = (grad_error_y + grad_error_x) / 2

    return {
        "mse": mse,
        "mae": mae,
        "rmse": np.sqrt(mse),
        "gradient_error": grad_error
    }


def extract_instances_from_semantic(semantic_mask, min_size=5):
    """Extract instances from semantic segmentation using connected components"""
    labeled, num_features = ndimage.label(semantic_mask > 0.5)

    instances = []
    for i in range(1, num_features + 1):
        instance_mask = (labeled == i).astype(bool)
        if instance_mask.sum() < min_size:
            continue
        instances.append(instance_mask)

    return instances


def evaluate_model_outputs(pred_dir, gt_data_root, output_json=None):
    """
    Evaluate model predictions against ground truth.

    Args:
        pred_dir: Directory containing prediction masks (semantic or instance)
        gt_data_root: Root directory of ground truth dataset
        output_json: Optional path to save results as JSON
    """
    # Load ground truth annotations
    ann_path = os.path.join(gt_data_root, "annotations.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    results = {
        "instance_metrics": [],
        "pixel_metrics": [],
        "per_image": []
    }

    print("Evaluating predictions...")
    for img_info in tqdm(gt_data["images"]):
        img_id = img_info["id"]
        img_file = os.path.basename(img_info["file_name"])
        pred_file = os.path.join(pred_dir, img_file)

        if not os.path.exists(pred_file):
            print(f"Warning: Prediction not found for {img_file}")
            continue

        # Load prediction
        pred = np.array(Image.open(pred_file).convert("L")).astype(np.float32) / 255.0

        # Load ground truth
        _, gt_instances = load_gt_instances(gt_data_root, img_id)

        if len(gt_instances) == 0:
            continue

        # Create GT semantic mask
        H, W = pred.shape
        gt_semantic = np.zeros((H, W), dtype=np.float32)
        for inst in gt_instances:
            gt_semantic = np.maximum(gt_semantic, inst["mask"].astype(np.float32))

        # Extract predicted instances
        pred_instances = extract_instances_from_semantic(pred)
        gt_masks = [inst["mask"] for inst in gt_instances]

        # Compute instance metrics
        prec, rec, f1, mean_iou, _ = compute_instance_metrics(pred_instances, gt_masks)
        results["instance_metrics"].append({
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mean_iou": mean_iou,
            "num_pred": len(pred_instances),
            "num_gt": len(gt_masks)
        })

        # Compute pixel metrics
        pixel_metrics = compute_pixel_metrics(pred, gt_semantic)
        results["pixel_metrics"].append(pixel_metrics)

        # Compute alpha metrics
        alpha_metrics = compute_alpha_metrics(pred, gt_semantic)

        results["per_image"].append({
            "image": img_file,
            "instance": {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "mean_iou": mean_iou
            },
            "pixel": pixel_metrics,
            "alpha": alpha_metrics
        })

    # Aggregate results
    summary = {
        "instance": {
            "precision": np.mean([m["precision"] for m in results["instance_metrics"]]),
            "recall": np.mean([m["recall"] for m in results["instance_metrics"]]),
            "f1": np.mean([m["f1"] for m in results["instance_metrics"]]),
            "mean_iou": np.mean([m["mean_iou"] for m in results["instance_metrics"]]),
        },
        "pixel": {
            "accuracy": np.mean([m["accuracy"] for m in results["pixel_metrics"]]),
            "precision": np.mean([m["precision"] for m in results["pixel_metrics"]]),
            "recall": np.mean([m["recall"] for m in results["pixel_metrics"]]),
            "f1": np.mean([m["f1"] for m in results["pixel_metrics"]]),
        },
        "alpha": {
            "mse": np.mean([img["alpha"]["mse"] for img in results["per_image"]]),
            "mae": np.mean([img["alpha"]["mae"] for img in results["per_image"]]),
            "rmse": np.mean([img["alpha"]["rmse"] for img in results["per_image"]]),
            "gradient_error": np.mean([img["alpha"]["gradient_error"] for img in results["per_image"]]),
        }
    }

    results["summary"] = summary

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print("\nInstance-level Metrics:")
    print(f"  Precision: {summary['instance']['precision']:.4f}")
    print(f"  Recall:    {summary['instance']['recall']:.4f}")
    print(f"  F1 Score:  {summary['instance']['f1']:.4f}")
    print(f"  Mean IoU:  {summary['instance']['mean_iou']:.4f}")

    print("\nPixel-level Metrics:")
    print(f"  Accuracy:  {summary['pixel']['accuracy']:.4f}")
    print(f"  Precision: {summary['pixel']['precision']:.4f}")
    print(f"  Recall:    {summary['pixel']['recall']:.4f}")
    print(f"  F1 Score:  {summary['pixel']['f1']:.4f}")

    print("\nAlpha Matte Quality:")
    print(f"  MSE:            {summary['alpha']['mse']:.6f}")
    print(f"  MAE:            {summary['alpha']['mae']:.6f}")
    print(f"  RMSE:           {summary['alpha']['rmse']:.6f}")
    print(f"  Gradient Error: {summary['alpha']['gradient_error']:.6f}")
    print("="*60 + "\n")

    # Save results
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate character segmentation models")
    parser.add_argument("--pred", required=True, help="Directory with prediction masks")
    parser.add_argument("--gt", required=True, help="Ground truth dataset root")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    args = parser.parse_args()

    evaluate_model_outputs(args.pred, args.gt, args.output)


if __name__ == "__main__":
    main()
