#!/usr/bin/env bash
set -e
python src/postprocess_export.py --ckpt runs/joint/joint_last.pt --input data/demo/images --out exports/tiles16
