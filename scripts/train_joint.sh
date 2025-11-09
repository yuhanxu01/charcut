#!/usr/bin/env bash
set -e
python src/train_joint.py --data data/demo --epochs 2 --batch 2 --lr 1e-4 --lr_refine 1e-3
