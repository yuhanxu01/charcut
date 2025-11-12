# CharCut: Character Instance Matting and Ordering

End-to-end pipeline for extracting per-character RGBA crops from noisy 256x256 text images with overlaps, and outputting 16x16 ordered PNGs.

## Features
- Synthetic data: sprites and 256x256 paragraphs with overlaps.
- COCO-like annotations with per-instance binary mask PNG and bbox.
- **Two training pipelines**:
  - **Baseline**: Simple U-Net semantic segmentation (~1.8M params, fast)
  - **Advanced**: Mask R-CNN + U-Net refinement (~46M params, high quality)
- Hungarian matching across predicted vs GT instances (advanced only).
- Soft pixel allocation + reconstruction loss (advanced only).
- Multiple output formats: RGBA tiles, binary masks (palette mode), grayscale.
- Postprocess to 16x16 tiles sorted top-to-bottom then left-to-right.

## Two Approaches

### Approach A: Simple Baseline ⚡ (Recommended for Quick Start)
- **Model**: Lightweight U-Net for semantic segmentation
- **Parameters**: ~1.8M (26x smaller than advanced pipeline)
- **Training**: Single-stage, BCE + Dice loss
- **Instance separation**: Connected components analysis
- **Speed**: Fast training (~5x faster) and inference
- **Use case**: Quick prototyping, limited compute, simple datasets

### Approach B: Advanced Two-Stage Pipeline 🎯
- **Model**: Mask R-CNN + U-Net refinement
- **Parameters**: ~46M (includes ResNet-50 backbone)
- **Training**: Two-stage with Hungarian matching + reconstruction loss
- **Instance separation**: Instance-aware detection
- **Quality**: Better alpha matting for overlapping characters
- **Use case**: High-quality extraction, complex overlapping scenarios

## Quick Start

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Provide fonts
Place Chinese font TTFs under `assets/fonts/`:
- NotoSansCJK-Regular.otf (or similar CJK font)
- Optional: SimSun.ttf, SimHei.ttf, KaiTi.ttf

### 3) Synthesize a tiny demo dataset
```bash
python src/synth/generate_sprites.py --charset assets/charset_demo.txt --fonts assets/fonts --out data/sprites --size 16
python src/synth/compose_paragraphs.py --sprites data/sprites --out data/demo --num_images 50 --canvas 256
```

### 4a) **Baseline Pipeline** (Fast and Simple)

#### Train
```bash
python src/train_baseline.py --data data/demo --epochs 20 --batch 4 --lr 1e-3
```

#### Inference with different output formats
```bash
# RGBA tiles (default)
python src/infer_baseline.py --ckpt runs/baseline/best.pt --input data/demo/images --out exports/baseline_rgba --format rgba

# Binary masks (palette mode, smaller file size)
python src/infer_baseline.py --ckpt runs/baseline/best.pt --input data/demo/images --out exports/baseline_binary --format binary

# Grayscale alpha mattes
python src/infer_baseline.py --ckpt runs/baseline/best.pt --input data/demo/images --out exports/baseline_gray --format grayscale
```

### 4b) **Advanced Pipeline** (High Quality)

#### Train Stage A (Mask R-CNN warmup)
```bash
python src/train_stageA.py --data data/demo --epochs 2 --batch 2 --lr 1e-4
```

#### Joint finetune with refinement + reconstruction
```bash
python src/train_joint.py --data data/demo --epochs 2 --batch 2 --lr 1e-4 --lr_refine 1e-3
```

#### Export with different output formats
```bash
# RGBA tiles (default)
python src/postprocess_export.py --ckpt runs/joint/joint_last.pt --input data/demo/images --out exports/advanced_rgba --format rgba

# Binary masks (palette mode)
python src/postprocess_export.py --ckpt runs/joint/joint_last.pt --input data/demo/images --out exports/advanced_binary --format binary

# Grayscale alpha mattes
python src/postprocess_export.py --ckpt runs/joint/joint_last.pt --input data/demo/images --out exports/advanced_gray --format grayscale
```

## Comparison: Baseline vs Advanced

| Feature | Baseline | Advanced |
|---------|----------|----------|
| Model size | ~1.8M params | ~46M params |
| Training time (50 imgs, 20 epochs) | ~5 min (GPU) | ~25 min (GPU) |
| Inference speed | Fast | Moderate |
| Memory usage | Low (~2GB) | High (~6GB) |
| Instance separation | Connected components | Mask R-CNN detection |
| Overlap handling | Basic (erosion/dilation) | Advanced (soft allocation) |
| Alpha quality | Good | Excellent |
| Best for | Simple datasets, prototyping | Complex overlaps, production |

## Output Formats

All pipelines support three output formats:

1. **RGBA** (default): 4-channel PNG with RGB + alpha matte
   - Best for: Direct rendering, transparency support
   - File size: Largest

2. **Binary** (palette mode): 1-bit binarized mask as palette PNG
   - Best for: Storage efficiency, simple segmentation
   - File size: Smallest (~10x smaller)
   - Use `--format binary`

3. **Grayscale**: 8-bit alpha matte only
   - Best for: Soft alpha values, further processing
   - File size: Medium
   - Use `--format grayscale`

## Notes
- This repo is minimal. It is designed to run on synthetic data and small GPU.
- Replace demo charset with larger sets for real training.
- **Baseline** is recommended for quick experiments and datasets without heavy overlap.
- **Advanced** pipeline is better for overlapping characters and high-quality alpha matting.
- Reconstruction loss (advanced) needs images saved as RGB; masks are L (0/255).

## Risks and Caveats
- **Baseline**: Connected components may merge touching characters; tune min_size parameter.
- **Advanced**: Overlap allocation depends on reconstruction loss; tune weights per dataset.
- Small sprites make font rasterization sensitive to hinting; use multiple fonts.
- Real handwriting varies; add stronger geometry and noise augmentations.
- Large charset increases IO; prefer LMDB/HDF5 for production.
