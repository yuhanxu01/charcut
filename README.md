# CharCut: Character Instance Matting and Ordering

End-to-end pipeline for extracting per-character RGBA crops from noisy 256x256 text images with overlaps, and outputting 16x16 ordered PNGs.

## Features
- Synthetic data: sprites and 256x256 paragraphs with overlaps.
- COCO-like annotations with per-instance binary mask PNG and bbox.
- Model: class-agnostic Mask R-CNN + lightweight U-Net refinement.
- Hungarian matching across predicted vs GT instances.
- Soft pixel allocation + reconstruction loss.
- Postprocess to 16x16 RGBA tiles sorted top-to-bottom then left-to-right.

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

### 4) Train Stage A (Mask R-CNN warmup)
```bash
python src/train_stageA.py --data data/demo --epochs 2 --batch 2 --lr 1e-4
```

### 5) Joint finetune with refinement + reconstruction
```bash
python src/train_joint.py --data data/demo --epochs 2 --batch 2 --lr 1e-4 --lr_refine 1e-3
```

### 6) Export 16x16 ordered RGBA tiles
```bash
python src/postprocess_export.py --ckpt runs/joint_last.pt --input data/demo/images --out exports/tiles16
```

## Notes
- This repo is minimal. It is designed to run on synthetic data and small GPU.
- Replace demo charset with larger sets for real training.
- Reconstruction loss needs images saved as RGB; masks are L (0/255).

## Risks and caveats
- Overlap allocation depends on reconstruction loss; tune weights per dataset.
- Small sprites make font rasterization sensitive to hinting; use multiple fonts.
- Real handwriting varies; add stronger geometry and noise augmentations.
- Large charset increases IO; prefer LMDB/HDF5 for production.
