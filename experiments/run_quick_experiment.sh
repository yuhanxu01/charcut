#!/bin/bash
################################################################################
# Quick Experiment Runner - Baseline vs Advanced Comparison
#
# This script runs a minimal comparison between baseline and advanced pipelines
# for quick evaluation (~15-20 minutes on GPU).
#
# Usage:
#   bash experiments/run_quick_experiment.sh
################################################################################

set -e

# Configuration
ROOT_DIR=$(pwd)
EXPERIMENTS_DIR="${ROOT_DIR}/experiments"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"
DATA_DIR="${ROOT_DIR}/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
QUICK_RUN="${RESULTS_DIR}/quick_${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CharCut Quick Experiment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

mkdir -p "${QUICK_RUN}"

# Generate small dataset (100 images)
echo -e "${BLUE}[1/5] Generating test dataset (100 images)...${NC}"
DATASET_DIR="${DATA_DIR}/quick_${TIMESTAMP}"

python src/synth/generate_sprites.py \
    --charset assets/charset_demo.txt \
    --fonts assets/fonts \
    --out "${DATASET_DIR}/sprites" \
    --size 16

python src/synth/compose_paragraphs.py \
    --sprites "${DATASET_DIR}/sprites" \
    --out "${DATASET_DIR}/data" \
    --num_images 100 \
    --canvas 256

echo -e "${GREEN}✓ Dataset generated${NC}"
echo ""

# Train baseline
echo -e "${BLUE}[2/5] Training Baseline U-Net (15 epochs, ~5 min)...${NC}"
python src/train_baseline.py \
    --data "${DATASET_DIR}/data" \
    --epochs 15 \
    --batch 4 \
    --lr 1e-3 \
    --out "${QUICK_RUN}/baseline/checkpoints"

echo -e "${GREEN}✓ Baseline training complete${NC}"
echo ""

# Train advanced
echo -e "${BLUE}[3/5] Training Advanced Pipeline (Stage A: 5 epochs, Joint: 10 epochs, ~10 min)...${NC}"

python src/train_stageA.py \
    --data "${DATASET_DIR}/data" \
    --epochs 5 \
    --batch 2 \
    --lr 1e-4 \
    --out "${QUICK_RUN}/advanced/stageA"

python src/train_joint.py \
    --data "${DATASET_DIR}/data" \
    --epochs 10 \
    --batch 2 \
    --lr 1e-4 \
    --lr_refine 1e-3 \
    --ckpt_stageA "${QUICK_RUN}/advanced/stageA/last.pt" \
    --out "${QUICK_RUN}/advanced/joint"

echo -e "${GREEN}✓ Advanced training complete${NC}"
echo ""

# Evaluate both models
echo -e "${BLUE}[4/5] Evaluating models...${NC}"

# Baseline inference and eval
python src/infer_baseline.py \
    --ckpt "${QUICK_RUN}/baseline/checkpoints/best.pt" \
    --input "${DATASET_DIR}/data/images" \
    --out "${QUICK_RUN}/baseline/predictions" \
    --format grayscale

python src/evaluate.py \
    --pred "${QUICK_RUN}/baseline/predictions" \
    --gt "${DATASET_DIR}/data" \
    --output "${QUICK_RUN}/baseline/metrics.json"

echo -e "${GREEN}✓ Baseline evaluation complete${NC}"

# Note: Advanced pipeline evaluation requires tile merging
echo -e "${YELLOW}⚠ Advanced pipeline full evaluation requires tile merging (skipped)${NC}"
echo ""

# Visualizations
echo -e "${BLUE}[5/5] Creating visualizations...${NC}"

python src/visualize_results.py \
    --images "${DATASET_DIR}/data/images" \
    --gt "${DATASET_DIR}/data" \
    --pred "${QUICK_RUN}/baseline/predictions" \
    --out "${QUICK_RUN}/baseline/visualizations" \
    --max 10

echo -e "${GREEN}✓ Visualizations created${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Quick Experiment Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results: ${QUICK_RUN}"
echo ""
echo "Baseline metrics:"
python3 -c "
import json
with open('${QUICK_RUN}/baseline/metrics.json') as f:
    m = json.load(f)['summary']
    print(f\"  Precision: {m['instance']['precision']:.4f}\")
    print(f\"  Recall:    {m['instance']['recall']:.4f}\")
    print(f\"  F1 Score:  {m['instance']['f1']:.4f}\")
    print(f\"  Mean IoU:  {m['instance']['mean_iou']:.4f}\")
    print(f\"  Pixel Acc: {m['pixel']['accuracy']:.4f}\")
"
echo ""
echo "Visualizations: ${QUICK_RUN}/baseline/visualizations/"
echo ""
