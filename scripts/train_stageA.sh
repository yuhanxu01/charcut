#!/usr/bin/env bash
set -e
python src/train_stageA.py --data data/demo --epochs 2 --batch 2 --lr 1e-4
