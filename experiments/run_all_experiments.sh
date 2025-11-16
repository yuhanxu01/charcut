#!/bin/bash
################################################################################
# Comprehensive Experiment Runner for CharCut
#
# This script runs all experiments defined in the configuration and generates
# a comprehensive comparison report.
#
# Usage:
#   bash experiments/run_all_experiments.sh [dataset_size]
#
# Arguments:
#   dataset_size: tiny (100 imgs), small (500 imgs), medium (2000 imgs)
#                 Default: small
################################################################################

set -e  # Exit on error

# Configuration
DATASET_SIZE=${1:-small}
ROOT_DIR=$(pwd)
EXPERIMENTS_DIR="${ROOT_DIR}/experiments"
RESULTS_DIR="${EXPERIMENTS_DIR}/results"
DATA_DIR="${ROOT_DIR}/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_RUN="${RESULTS_DIR}/run_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Dataset configurations
declare -A DATASET_CONFIGS
DATASET_CONFIGS[tiny]=100
DATASET_CONFIGS[small]=500
DATASET_CONFIGS[medium]=2000

NUM_IMAGES=${DATASET_CONFIGS[$DATASET_SIZE]}

if [ -z "$NUM_IMAGES" ]; then
    log_error "Invalid dataset size: $DATASET_SIZE"
    log_info "Valid options: tiny, small, medium"
    exit 1
fi

# Create experiment run directory
mkdir -p "${EXPERIMENT_RUN}"
LOG_FILE="${EXPERIMENT_RUN}/experiment.log"

log_info "==================================================================="
log_info "CharCut Comprehensive Experiment Suite"
log_info "==================================================================="
log_info "Dataset size: ${DATASET_SIZE} (${NUM_IMAGES} images)"
log_info "Results directory: ${EXPERIMENT_RUN}"
log_info "Log file: ${LOG_FILE}"
log_info "==================================================================="

# Log to file
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

################################################################################
# Step 1: Generate Dataset
################################################################################

log_info ""
log_info "Step 1: Generating synthetic dataset..."
log_info "-------------------------------------------------------------------"

DATASET_DIR="${DATA_DIR}/exp_${DATASET_SIZE}_${TIMESTAMP}"

# Check if fonts exist
if [ ! -d "${ROOT_DIR}/assets/fonts" ] || [ -z "$(ls -A ${ROOT_DIR}/assets/fonts)" ]; then
    log_error "No fonts found in assets/fonts/"
    log_info "Please add CJK font files (.ttf or .otf) to assets/fonts/"
    log_info "Example: NotoSansCJK-Regular.otf, SimSun.ttf, etc."
    exit 1
fi

# Generate sprites
log_info "Generating character sprites..."
python src/synth/generate_sprites.py \
    --charset assets/charset_demo.txt \
    --fonts assets/fonts \
    --out "${DATASET_DIR}/sprites" \
    --size 16

# Generate paragraph images
log_info "Composing ${NUM_IMAGES} paragraph images..."
python src/synth/compose_paragraphs.py \
    --sprites "${DATASET_DIR}/sprites" \
    --out "${DATASET_DIR}/data" \
    --num_images ${NUM_IMAGES} \
    --canvas 256 \
    --line_height 18 \
    --p_overlap 0.3

log_success "Dataset generated at: ${DATASET_DIR}/data"

################################################################################
# Step 2: Run Baseline Experiments
################################################################################

log_info ""
log_info "Step 2: Running Baseline U-Net experiments..."
log_info "-------------------------------------------------------------------"

# Array of baseline configurations: name:epochs:batch:lr:base_filters
BASELINE_CONFIGS=(
    "default:30:4:1e-3:32"
    "deep:30:4:1e-3:48"
    "shallow:30:4:1e-3:24"
    "lr_high:30:4:5e-3:32"
    "lr_low:30:4:5e-4:32"
    "batch_large:30:8:2e-3:32"
)

for config in "${BASELINE_CONFIGS[@]}"; do
    IFS=':' read -r name epochs batch lr base_filters <<< "$config"

    log_info ""
    log_info "Running baseline experiment: ${name}"
    log_info "  Epochs: ${epochs}, Batch: ${batch}, LR: ${lr}, Filters: ${base_filters}"

    EXP_DIR="${EXPERIMENT_RUN}/baseline_${name}"
    mkdir -p "${EXP_DIR}"

    # Train
    log_info "  Training..."
    python src/train_baseline.py \
        --data "${DATASET_DIR}/data" \
        --epochs ${epochs} \
        --batch ${batch} \
        --lr ${lr} \
        --base_filters ${base_filters} \
        --out "${EXP_DIR}/checkpoints" \
        --num_workers 4 \
        > "${EXP_DIR}/train.log" 2>&1

    if [ $? -eq 0 ]; then
        log_success "  Training completed"
    else
        log_error "  Training failed! Check ${EXP_DIR}/train.log"
        continue
    fi

    # Inference
    log_info "  Running inference..."
    python src/infer_baseline.py \
        --ckpt "${EXP_DIR}/checkpoints/best.pt" \
        --input "${DATASET_DIR}/data/images" \
        --out "${EXP_DIR}/predictions" \
        --format grayscale \
        --threshold 0.5 \
        --min_size 5 \
        --base_filters ${base_filters} \
        > "${EXP_DIR}/infer.log" 2>&1

    log_success "  Inference completed"

    # Evaluate
    log_info "  Evaluating..."
    python src/evaluate.py \
        --pred "${EXP_DIR}/predictions" \
        --gt "${DATASET_DIR}/data" \
        --output "${EXP_DIR}/metrics.json" \
        > "${EXP_DIR}/eval.log" 2>&1

    log_success "  Evaluation completed"

    # Visualize (first 20 images)
    log_info "  Creating visualizations..."
    python src/visualize_results.py \
        --images "${DATASET_DIR}/data/images" \
        --gt "${DATASET_DIR}/data" \
        --pred "${EXP_DIR}/predictions" \
        --out "${EXP_DIR}/visualizations" \
        --max 20 \
        > "${EXP_DIR}/viz.log" 2>&1

    log_success "  Visualizations created"
    log_success "Baseline experiment '${name}' completed!"
done

################################################################################
# Step 3: Run Advanced Pipeline Experiments
################################################################################

log_info ""
log_info "Step 3: Running Advanced two-stage pipeline experiments..."
log_info "-------------------------------------------------------------------"

# Array of advanced configurations: name:stage_a_epochs:joint_epochs:batch:lr:lr_refine
ADVANCED_CONFIGS=(
    "default:10:15:2:1e-4:1e-3"
    "long:20:30:2:1e-4:1e-3"
    "batch_large:10:15:4:2e-4:2e-3"
)

for config in "${ADVANCED_CONFIGS[@]}"; do
    IFS=':' read -r name stage_a_epochs joint_epochs batch lr lr_refine <<< "$config"

    log_info ""
    log_info "Running advanced experiment: ${name}"
    log_info "  Stage A: ${stage_a_epochs} epochs, Joint: ${joint_epochs} epochs"
    log_info "  Batch: ${batch}, LR: ${lr}, Refine LR: ${lr_refine}"

    EXP_DIR="${EXPERIMENT_RUN}/advanced_${name}"
    mkdir -p "${EXP_DIR}"

    # Stage A training
    log_info "  Training Stage A (Mask R-CNN)..."
    python src/train_stageA.py \
        --data "${DATASET_DIR}/data" \
        --epochs ${stage_a_epochs} \
        --batch ${batch} \
        --lr ${lr} \
        --out "${EXP_DIR}/stageA" \
        > "${EXP_DIR}/train_stageA.log" 2>&1

    if [ $? -eq 0 ]; then
        log_success "  Stage A training completed"
    else
        log_error "  Stage A training failed! Check ${EXP_DIR}/train_stageA.log"
        continue
    fi

    # Joint training
    log_info "  Training joint (Mask R-CNN + Refinement)..."
    python src/train_joint.py \
        --data "${DATASET_DIR}/data" \
        --epochs ${joint_epochs} \
        --batch ${batch} \
        --lr ${lr} \
        --lr_refine ${lr_refine} \
        --ckpt_stageA "${EXP_DIR}/stageA/last.pt" \
        --out "${EXP_DIR}/joint" \
        > "${EXP_DIR}/train_joint.log" 2>&1

    if [ $? -eq 0 ]; then
        log_success "  Joint training completed"
    else
        log_error "  Joint training failed! Check ${EXP_DIR}/train_joint.log"
        continue
    fi

    # Inference
    log_info "  Running inference..."
    python src/postprocess_export.py \
        --ckpt "${EXP_DIR}/joint/joint_last.pt" \
        --input "${DATASET_DIR}/data/images" \
        --out "${EXP_DIR}/predictions" \
        --format grayscale \
        --size 256 \
        > "${EXP_DIR}/infer.log" 2>&1

    log_success "  Inference completed"

    # Note: For evaluation, we need to merge tiles back to full image
    # For now, skip detailed evaluation of advanced pipeline
    log_warning "  Advanced pipeline evaluation requires tile merging (skipped for now)"

    log_success "Advanced experiment '${name}' completed!"
done

################################################################################
# Step 4: Generate Comparison Report
################################################################################

log_info ""
log_info "Step 4: Generating comparison report..."
log_info "-------------------------------------------------------------------"

# Create comparison report
REPORT_FILE="${EXPERIMENT_RUN}/REPORT.md"

cat > "${REPORT_FILE}" << EOF
# CharCut Experiment Report

**Date:** $(date)
**Dataset:** ${DATASET_SIZE} (${NUM_IMAGES} images)
**Dataset Path:** ${DATASET_DIR}/data

## Experiment Summary

This report compares different model configurations for character instance segmentation.

---

## Baseline U-Net Experiments

| Experiment | Precision | Recall | F1 | IoU | Pixel Acc | Training Time |
|------------|-----------|--------|----|----|-----------|---------------|
EOF

# Parse results from each baseline experiment
for config in "${BASELINE_CONFIGS[@]}"; do
    IFS=':' read -r name epochs batch lr base_filters <<< "$config"
    EXP_DIR="${EXPERIMENT_RUN}/baseline_${name}"
    METRICS_FILE="${EXP_DIR}/metrics.json"

    if [ -f "${METRICS_FILE}" ]; then
        # Extract metrics using Python
        PRECISION=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_FILE}'))['summary']['instance']['precision']:.4f}\")")
        RECALL=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_FILE}'))['summary']['instance']['recall']:.4f}\")")
        F1=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_FILE}'))['summary']['instance']['f1']:.4f}\")")
        IOU=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_FILE}'))['summary']['instance']['mean_iou']:.4f}\")")
        PIX_ACC=$(python3 -c "import json; print(f\"{json.load(open('${METRICS_FILE}'))['summary']['pixel']['accuracy']:.4f}\")")

        # Get training time from log
        if [ -f "${EXP_DIR}/train.log" ]; then
            TRAIN_TIME="See log"
        else
            TRAIN_TIME="N/A"
        fi

        echo "| baseline_${name} | ${PRECISION} | ${RECALL} | ${F1} | ${IOU} | ${PIX_ACC} | ${TRAIN_TIME} |" >> "${REPORT_FILE}"
    else
        echo "| baseline_${name} | - | - | - | - | - | Failed |" >> "${REPORT_FILE}"
    fi
done

cat >> "${REPORT_FILE}" << EOF

### Configuration Details

EOF

for config in "${BASELINE_CONFIGS[@]}"; do
    IFS=':' read -r name epochs batch lr base_filters <<< "$config"
    cat >> "${REPORT_FILE}" << EOF
**baseline_${name}:**
- Epochs: ${epochs}
- Batch size: ${batch}
- Learning rate: ${lr}
- Base filters: ${base_filters}

EOF
done

cat >> "${REPORT_FILE}" << EOF

---

## Advanced Pipeline Experiments

EOF

for config in "${ADVANCED_CONFIGS[@]}"; do
    IFS=':' read -r name stage_a_epochs joint_epochs batch lr lr_refine <<< "$config"
    EXP_DIR="${EXPERIMENT_RUN}/advanced_${name}"

    cat >> "${REPORT_FILE}" << EOF
**advanced_${name}:**
- Stage A epochs: ${stage_a_epochs}
- Joint epochs: ${joint_epochs}
- Batch size: ${batch}
- Learning rate: ${lr}
- Refinement LR: ${lr_refine}
- Status: $([ -d "${EXP_DIR}/joint" ] && echo "Completed" || echo "Failed")

EOF
done

cat >> "${REPORT_FILE}" << EOF

---

## Best Model Selection

EOF

# Find best baseline model by F1 score
BEST_F1=0
BEST_MODEL=""

for config in "${BASELINE_CONFIGS[@]}"; do
    IFS=':' read -r name epochs batch lr base_filters <<< "$config"
    EXP_DIR="${EXPERIMENT_RUN}/baseline_${name}"
    METRICS_FILE="${EXP_DIR}/metrics.json"

    if [ -f "${METRICS_FILE}" ]; then
        F1=$(python3 -c "import json; print(json.load(open('${METRICS_FILE}'))['summary']['instance']['f1'])")

        # Compare F1 scores
        IS_BETTER=$(python3 -c "print(${F1} > ${BEST_F1})")
        if [ "$IS_BETTER" = "True" ]; then
            BEST_F1=$F1
            BEST_MODEL="baseline_${name}"
        fi
    fi
done

cat >> "${REPORT_FILE}" << EOF
**Best Baseline Model:** ${BEST_MODEL}
- F1 Score: ${BEST_F1}
- Checkpoint: ${EXPERIMENT_RUN}/${BEST_MODEL}/checkpoints/best.pt

## Recommendations

1. **For quick prototyping**: Use \`baseline_default\` configuration
2. **For best accuracy**: Use \`${BEST_MODEL}\` configuration
3. **For production**: Consider advanced pipeline with longer training

## Visualizations

Sample visualizations are available in each experiment's \`visualizations/\` directory.

---

## Files

- Full results: \`${EXPERIMENT_RUN}/\`
- This report: \`${REPORT_FILE}\`
- Experiment log: \`${LOG_FILE}\`

EOF

log_success "Report generated: ${REPORT_FILE}"

################################################################################
# Summary
################################################################################

log_info ""
log_info "==================================================================="
log_info "All experiments completed!"
log_info "==================================================================="
log_info ""
log_info "Results location: ${EXPERIMENT_RUN}"
log_info "Report: ${REPORT_FILE}"
log_info "Log: ${LOG_FILE}"
log_info ""
log_info "Best baseline model: ${BEST_MODEL} (F1: ${BEST_F1})"
log_info ""
log_info "To view the report:"
log_info "  cat ${REPORT_FILE}"
log_info ""
log_info "To use the best model for inference:"
log_info "  python src/infer_baseline.py \\"
log_info "    --ckpt ${EXPERIMENT_RUN}/${BEST_MODEL}/checkpoints/best.pt \\"
log_info "    --input your_images/ \\"
log_info "    --out output/ \\"
log_info "    --format binary"
log_info ""
log_info "==================================================================="
