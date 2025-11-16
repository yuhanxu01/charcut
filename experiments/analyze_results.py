"""
Analyze and compare experiment results.

This script reads all metrics.json files from an experiment run directory,
compares the results, and generates charts and detailed reports.
"""
import argparse
import os
import json
import glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_experiment_results(run_dir):
    """Load all experiment results from a run directory"""
    results = {}

    # Find all metrics.json files
    metrics_files = glob.glob(os.path.join(run_dir, "*/metrics.json"))

    for mfile in metrics_files:
        exp_name = os.path.basename(os.path.dirname(mfile))

        with open(mfile, 'r') as f:
            data = json.load(f)

        results[exp_name] = {
            "summary": data.get("summary", {}),
            "path": mfile
        }

    return results


def create_comparison_chart(results, output_dir):
    """Create comparison charts for all experiments"""
    os.makedirs(output_dir, exist_ok=True)

    exp_names = list(results.keys())
    if not exp_names:
        print("No experiments found!")
        return

    # Extract metrics
    metrics_to_plot = [
        ("instance", "precision", "Instance Precision"),
        ("instance", "recall", "Instance Recall"),
        ("instance", "f1", "Instance F1 Score"),
        ("instance", "mean_iou", "Instance Mean IoU"),
        ("pixel", "accuracy", "Pixel Accuracy"),
        ("pixel", "f1", "Pixel F1 Score"),
        ("alpha", "mae", "Alpha MAE (lower is better)"),
    ]

    for category, metric, title in metrics_to_plot:
        values = []
        names = []

        for exp_name in sorted(exp_names):
            try:
                val = results[exp_name]["summary"][category][metric]
                values.append(val)
                names.append(exp_name.replace("baseline_", "b_").replace("advanced_", "a_"))
            except (KeyError, TypeError):
                continue

        if not values:
            continue

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(names, values, color='steelblue', edgecolor='black')

        # Highlight best model
        if "lower is better" not in title:
            best_idx = np.argmax(values)
            bars[best_idx].set_color('green')
        else:
            best_idx = np.argmin(values)
            bars[best_idx].set_color('green')

        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Rotate x labels for readability
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, f'{category}_{metric}.png')
        plt.savefig(chart_path, dpi=150)
        plt.close()

        print(f"  Created chart: {chart_path}")


def create_summary_table(results, output_file):
    """Create a detailed summary table in Markdown"""
    with open(output_file, 'w') as f:
        f.write("# Experiment Results Comparison\n\n")

        # Instance metrics table
        f.write("## Instance-Level Metrics\n\n")
        f.write("| Experiment | Precision | Recall | F1 | Mean IoU |\n")
        f.write("|------------|-----------|--------|----|----------|\n")

        for exp_name in sorted(results.keys()):
            summary = results[exp_name]["summary"]
            inst = summary.get("instance", {})
            f.write(f"| {exp_name} | "
                   f"{inst.get('precision', 0):.4f} | "
                   f"{inst.get('recall', 0):.4f} | "
                   f"{inst.get('f1', 0):.4f} | "
                   f"{inst.get('mean_iou', 0):.4f} |\n")

        # Pixel metrics table
        f.write("\n## Pixel-Level Metrics\n\n")
        f.write("| Experiment | Accuracy | Precision | Recall | F1 |\n")
        f.write("|------------|----------|-----------|--------|----|\ n")

        for exp_name in sorted(results.keys()):
            summary = results[exp_name]["summary"]
            pixel = summary.get("pixel", {})
            f.write(f"| {exp_name} | "
                   f"{pixel.get('accuracy', 0):.4f} | "
                   f"{pixel.get('precision', 0):.4f} | "
                   f"{pixel.get('recall', 0):.4f} | "
                   f"{pixel.get('f1', 0):.4f} |\n")

        # Alpha quality table
        f.write("\n## Alpha Matte Quality\n\n")
        f.write("| Experiment | MSE | MAE | RMSE | Gradient Error |\n")
        f.write("|------------|-----|-----|------|----------------|\n")

        for exp_name in sorted(results.keys()):
            summary = results[exp_name]["summary"]
            alpha = summary.get("alpha", {})
            f.write(f"| {exp_name} | "
                   f"{alpha.get('mse', 0):.6f} | "
                   f"{alpha.get('mae', 0):.6f} | "
                   f"{alpha.get('rmse', 0):.6f} | "
                   f"{alpha.get('gradient_error', 0):.6f} |\n")

        # Find best models
        f.write("\n## Best Models\n\n")

        # Best by F1
        best_f1_exp = max(results.items(),
                         key=lambda x: x[1]["summary"].get("instance", {}).get("f1", 0))
        f.write(f"**Best Instance F1:** {best_f1_exp[0]} "
               f"(F1={best_f1_exp[1]['summary']['instance']['f1']:.4f})\n\n")

        # Best by IoU
        best_iou_exp = max(results.items(),
                          key=lambda x: x[1]["summary"].get("instance", {}).get("mean_iou", 0))
        f.write(f"**Best Mean IoU:** {best_iou_exp[0]} "
               f"(IoU={best_iou_exp[1]['summary']['instance']['mean_iou']:.4f})\n\n")

        # Best pixel accuracy
        best_acc_exp = max(results.items(),
                          key=lambda x: x[1]["summary"].get("pixel", {}).get("accuracy", 0))
        f.write(f"**Best Pixel Accuracy:** {best_acc_exp[0]} "
               f"(Acc={best_acc_exp[1]['summary']['pixel']['accuracy']:.4f})\n\n")

        # Best alpha quality (lowest MAE)
        best_alpha_exp = min(results.items(),
                            key=lambda x: x[1]["summary"].get("alpha", {}).get("mae", float('inf')))
        f.write(f"**Best Alpha Quality:** {best_alpha_exp[0]} "
               f"(MAE={best_alpha_exp[1]['summary']['alpha']['mae']:.6f})\n\n")

    print(f"  Created summary table: {output_file}")


def print_summary(results):
    """Print summary to console"""
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)

    if not results:
        print("No results found!")
        return

    print(f"\nTotal experiments: {len(results)}\n")

    # Sort by F1 score
    sorted_results = sorted(results.items(),
                           key=lambda x: x[1]["summary"].get("instance", {}).get("f1", 0),
                           reverse=True)

    print("Ranking by Instance F1 Score:")
    print("-" * 70)
    for rank, (exp_name, data) in enumerate(sorted_results, 1):
        inst = data["summary"].get("instance", {})
        pixel = data["summary"].get("pixel", {})
        print(f"{rank}. {exp_name}")
        print(f"   Instance F1: {inst.get('f1', 0):.4f}  |  "
              f"Precision: {inst.get('precision', 0):.4f}  |  "
              f"Recall: {inst.get('recall', 0):.4f}")
        print(f"   Mean IoU: {inst.get('mean_iou', 0):.4f}  |  "
              f"Pixel Acc: {pixel.get('accuracy', 0):.4f}")
        print()

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--run_dir", required=True, help="Experiment run directory")
    parser.add_argument("--output", default=None, help="Output directory for analysis")
    args = parser.parse_args()

    # Default output directory
    if args.output is None:
        args.output = os.path.join(args.run_dir, "analysis")

    os.makedirs(args.output, exist_ok=True)

    print(f"Analyzing experiments in: {args.run_dir}")
    print(f"Output directory: {args.output}")

    # Load results
    results = load_experiment_results(args.run_dir)

    if not results:
        print("Error: No experiment results found!")
        print(f"Make sure {args.run_dir} contains subdirectories with metrics.json files")
        return

    print(f"\nFound {len(results)} experiments")

    # Print console summary
    print_summary(results)

    # Create visualizations
    print("\nGenerating charts...")
    create_comparison_chart(results, os.path.join(args.output, "charts"))

    # Create summary table
    print("\nGenerating summary table...")
    create_summary_table(results, os.path.join(args.output, "summary.md"))

    print(f"\nAnalysis complete! Results saved to: {args.output}")
    print(f"  - Charts: {args.output}/charts/")
    print(f"  - Summary: {args.output}/summary.md")


if __name__ == "__main__":
    main()
