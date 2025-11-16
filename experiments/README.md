# Experiment Framework

This directory contains scripts for running comprehensive experiments to find the best model pipeline for character segmentation.

## Quick Start

### Option 1: Quick Experiment (~15-20 minutes)

Run a minimal comparison between baseline and advanced pipelines:

```bash
bash experiments/run_quick_experiment.sh
```

This will:
- Generate 100 test images
- Train baseline U-Net (15 epochs)
- Train advanced pipeline (Stage A: 5 epochs, Joint: 10 epochs)
- Evaluate and visualize results

### Option 2: Full Experiment Suite (~2-4 hours)

Run comprehensive experiments with multiple configurations:

```bash
# Small dataset (500 images) - recommended
bash experiments/run_all_experiments.sh small

# Tiny dataset (100 images) - for testing
bash experiments/run_all_experiments.sh tiny

# Medium dataset (2000 images) - for final benchmarking
bash experiments/run_all_experiments.sh medium
```

This will:
- Generate synthetic dataset
- Run 6 baseline configurations with different hyperparameters
- Run 3 advanced pipeline configurations
- Evaluate all models with comprehensive metrics
- Generate visualizations and comparison report

## Experiment Configurations

### Baseline Configurations

1. **baseline_default**: Standard configuration (32 filters, lr=1e-3)
2. **baseline_deep**: More capacity (48 filters)
3. **baseline_shallow**: Less capacity (24 filters)
4. **baseline_lr_high**: Higher learning rate (5e-3)
5. **baseline_lr_low**: Lower learning rate (5e-4)
6. **baseline_batch_large**: Larger batch size (8 vs 4)

### Advanced Configurations

1. **advanced_default**: Standard two-stage pipeline
2. **advanced_long**: Longer training (20+30 epochs)
3. **advanced_batch_large**: Larger batch size (4 vs 2)

## Output Structure

After running experiments, results are organized as:

```
experiments/results/
└── run_YYYYMMDD_HHMMSS/
    ├── baseline_default/
    │   ├── checkpoints/
    │   │   ├── best.pt
    │   │   └── epoch_*.pt
    │   ├── predictions/
    │   ├── visualizations/
    │   ├── metrics.json
    │   ├── train.log
    │   └── eval.log
    ├── baseline_deep/
    │   └── ...
    ├── advanced_default/
    │   └── ...
    ├── REPORT.md          # Comparison report
    └── experiment.log     # Full experiment log
```

## Analyzing Results

### View Comparison Report

```bash
cat experiments/results/run_YYYYMMDD_HHMMSS/REPORT.md
```

### Generate Detailed Analysis

```bash
python experiments/analyze_results.py \
    --run_dir experiments/results/run_YYYYMMDD_HHMMSS \
    --output experiments/results/run_YYYYMMDD_HHMMSS/analysis
```

This creates:
- Comparison charts (bar plots for all metrics)
- Summary tables in Markdown
- Best model recommendations

### View Visualizations

Visualizations show side-by-side comparisons:
- Original image
- Ground truth overlay
- Prediction overlay
- Ground truth mask
- Prediction mask
- Difference map (White=correct, Red=false positive, Blue=false negative)

```bash
# View visualizations
ls experiments/results/run_YYYYMMDD_HHMMSS/baseline_default/visualizations/
```

## Evaluation Metrics

### Instance-Level Metrics
- **Precision**: Fraction of predicted instances that match GT
- **Recall**: Fraction of GT instances that are detected
- **F1 Score**: Harmonic mean of precision and recall
- **Mean IoU**: Average Intersection-over-Union of matched instances

### Pixel-Level Metrics
- **Accuracy**: Overall pixel classification accuracy
- **Precision**: Pixel-wise precision
- **Recall**: Pixel-wise recall
- **F1 Score**: Pixel-wise F1

### Alpha Matte Quality
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Gradient Error**: Edge quality metric

## Manual Evaluation

To evaluate a single model:

```bash
# Run inference
python src/infer_baseline.py \
    --ckpt path/to/model.pt \
    --input data/test/images \
    --out predictions/ \
    --format grayscale

# Evaluate
python src/evaluate.py \
    --pred predictions/ \
    --gt data/test/ \
    --output results.json

# Visualize
python src/visualize_results.py \
    --images data/test/images \
    --gt data/test/ \
    --pred predictions/ \
    --out visualizations/
```

## Tips for Best Results

### For Quick Prototyping
- Use `run_quick_experiment.sh`
- Focus on F1 score for overall performance

### For Production Models
- Use `run_all_experiments.sh medium`
- Consider both accuracy and inference speed
- Review visualizations to understand failure modes

### Hyperparameter Tuning
- Edit `configs/experiments.yaml` to add custom configurations
- Modify `run_all_experiments.sh` to include new experiments
- Focus on:
  - Learning rate (most impactful)
  - Model capacity (filters)
  - Batch size (affects training stability)

## Troubleshooting

### Out of Memory
- Reduce batch size
- Use smaller dataset (tiny instead of small)
- Reduce model capacity (base_filters)

### Poor Results
- Check data quality (visualize generated images)
- Increase training epochs
- Try different learning rates
- Ensure fonts are properly loaded

### Slow Training
- Use smaller dataset first
- Enable GPU (`torch.cuda.is_available()`)
- Increase num_workers for data loading
- Reduce number of experiments

## Advanced Usage

### Custom Experiments

Create your own experiment configuration:

```bash
# Train with custom settings
python src/train_baseline.py \
    --data your_data/ \
    --epochs 50 \
    --batch 8 \
    --lr 2e-3 \
    --base_filters 64 \
    --out runs/custom_experiment/

# Evaluate
python src/evaluate.py \
    --pred runs/custom_experiment/predictions \
    --gt your_data/ \
    --output runs/custom_experiment/metrics.json
```

### Comparing Multiple Runs

```bash
# Analyze and compare multiple experiment runs
python experiments/analyze_results.py \
    --run_dir experiments/results/run_20241116_120000

python experiments/analyze_results.py \
    --run_dir experiments/results/run_20241117_140000

# Compare the generated summary.md files
```

## Files

- `run_all_experiments.sh`: Comprehensive experiment suite
- `run_quick_experiment.sh`: Quick baseline vs advanced comparison
- `analyze_results.py`: Post-experiment analysis and visualization
- `configs/experiments.yaml`: Experiment configurations
- `README.md`: This file

## Next Steps

After finding the best model:

1. **Save the best checkpoint**:
   ```bash
   cp experiments/results/run_*/baseline_default/checkpoints/best.pt models/production_model.pt
   ```

2. **Use for inference**:
   ```bash
   python src/infer_baseline.py \
       --ckpt models/production_model.pt \
       --input new_images/ \
       --out output/ \
       --format binary  # or rgba, grayscale
   ```

3. **Fine-tune on real data**:
   - Prepare your real dataset in COCO format
   - Resume training from best checkpoint
   - Evaluate on held-out test set

4. **Deploy**:
   - Export model to ONNX/TorchScript
   - Optimize with TensorRT (optional)
   - Integrate into your application
