#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
HuggingFace Evaluate Benchmark Runner

Simple benchmark comparing baseline GPU models with QAIC-compiled models.
Uses HuggingFace Evaluate library for metric computation.

Usage:
    # Run benchmark with default settings
    python run_benchmark.py --model Qwen/Qwen2-1.5B-Instruct --datasets gsm8k,boolq --num-samples 10
    
    # Run only baseline
    python run_benchmark.py --model Qwen/Qwen2-1.5B-Instruct --datasets gsm8k --skip-qaic
    
    # Run only QAIC
    python run_benchmark.py --model Qwen/Qwen2-1.5B-Instruct --datasets gsm8k --skip-baseline
"""

import argparse
import json
import os
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import local modules
from dataset_loader import DatasetLoader
from metric_handler import MetricHandler

# Try to import QEfficient
try:
    from QEfficient import QEFFAutoModelForCausalLM
    QEFFICIENT_AVAILABLE = True
except ImportError:
    QEFFICIENT_AVAILABLE = False
    print("Warning: QEfficient not available. QAIC evaluation will not work.")

# Visualization constants
VIZ_FIGURE_WIDTH = 16
VIZ_FIGURE_HEIGHT = 6
VIZ_DETAILED_WIDTH = 18
VIZ_BAR_WIDTH = 0.35
VIZ_DPI = 300
VIZ_ALPHA = 0.85
VIZ_EDGE_WIDTH = 1.5
VIZ_GRID_ALPHA = 0.3

# Color scheme
COLOR_BASELINE = '#3498db'
COLOR_QAIC = '#e74c3c'
COLOR_POSITIVE = '#27ae60'
COLOR_NEGATIVE = '#e74c3c'
COLOR_SCATTER = '#9b59b6'


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HF Evaluate Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k",
        help="Comma-separated list of datasets (gsm8k, boolq, hellaswag)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation"
    )
    parser.add_argument(
        "--skip-qaic",
        action="store_true",
        help="Skip QAIC evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for baseline (cuda/cpu)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--baseline-precision",
        type=str,
        choices=["auto", "fp32", "fp16", "bf16"],
        default="fp32",
        help="Precision for baseline model (auto=fp16 on cuda, fp32 on cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (both baseline and QAIC)"
    )
    
    # QAIC-specific arguments
    parser.add_argument(
        "--prefill-seq-len",
        type=int,
        default=32,
        help="Prefill sequence length for QAIC"
    )
    parser.add_argument(
        "--ctx-len",
        type=int,
        default=512,
        help="Context length for QAIC"
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=16,
        help="Number of cores for QAIC"
    )
    parser.add_argument(
        "--device-group",
        type=str,
        default="0",
        help="Device IDs for QAIC (comma-separated)"
    )
    
    return parser.parse_args()


def run_baseline_evaluation(
    model_name: str,
    dataset_name: str,
    examples: List[Dict],
    device: str,
    max_new_tokens: int,
    precision: str = "fp32",
    batch_size: int = 1
) -> List[str]:
    """
    Run baseline evaluation using HuggingFace model on GPU/CPU.
    
    Args:
        model_name: HuggingFace model ID
        dataset_name: Name of dataset
        examples: List of examples from dataset
        device: Device to use (cuda/cpu)
        max_new_tokens: Maximum tokens to generate
        precision: Precision for model (auto/fp32/fp16/bf16)
        batch_size: Batch size for inference
    
    Returns:
        List of predictions
    """
    print(f"\n{'='*60}")
    print(f"Running Baseline Evaluation: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Determine torch dtype based on precision
    if precision == "auto":
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    elif precision == "fp32":
        torch_dtype = torch.float32
    elif precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer (dtype: {torch_dtype})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype
        ).to(device)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set padding side to left for batch generation
        tokenizer.padding_side = "left"
        
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Generate predictions
    predictions = []
    print(f"Generating predictions for {len(examples)} examples...")
    
    try:
        # Process in batches
        for i in tqdm(range(0, len(examples), batch_size), desc="Baseline"):
            batch = examples[i:i + batch_size]
            prompts = [ex["prompt"] for ex in batch]
            
            # Tokenize batch with padding
            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=False
            ).to(device)
            
            # Store input lengths for each example in batch
            input_lengths = inputs.attention_mask.sum(dim=1).tolist()
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # Decode each output in the batch
            for j, (output, input_len) in enumerate(zip(outputs, input_lengths)):
                # Extract only the generated tokens (skip input prompt)
                generated_tokens = output[input_len:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Extract answer based on dataset type
                answer = DatasetLoader.extract_answer_from_generation(
                    generated_text, 
                    batch[j]["type"]
                )
                
                predictions.append(answer)
    finally:
        # Clean up to free memory
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
    
    print(f"Baseline evaluation complete")
    return predictions


def run_qaic_evaluation(
    model_name: str,
    dataset_name: str,
    examples: List[Dict],
    prefill_seq_len: int,
    ctx_len: int,
    num_cores: int,
    device_group: List[int],
    max_new_tokens: int,
    batch_size: int = 1
) -> List[str]:
    """
    Run QAIC evaluation using QEfficient.
    
    Args:
        model_name: HuggingFace model ID
        dataset_name: Name of dataset
        examples: List of examples from dataset
        prefill_seq_len: Prefill sequence length
        ctx_len: Context length
        num_cores: Number of cores
        device_group: List of device IDs
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
    
    Returns:
        List of predictions
    """
    if not QEFFICIENT_AVAILABLE:
        raise ImportError("QEfficient not available. Install it to run QAIC evaluation.")
    
    print(f"\n{'='*60}")
    print(f"Running QAIC Evaluation: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Prefill Seq Len: {prefill_seq_len}")
    print(f"Context Length: {ctx_len}")
    print(f"Num Cores: {num_cores}")
    print(f"Device Group: {device_group}")
    print(f"Batch Size: {batch_size}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and compile QAIC model
    print("Loading QAIC model...")
    model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
    
    print("Compiling model for QAIC...")
    try:
        qpc_path = model.compile(
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_cores=num_cores,
            num_devices=len(device_group),
        )
        print(f"Model compiled to: {qpc_path}")
    except Exception as e:
        print(f"Error compiling model: {e}")
        raise
    
    # Generate predictions
    predictions = []
    print(f"Generating predictions for {len(examples)} examples...")
    
    # Process in batches
    for i in tqdm(range(0, len(examples), batch_size), desc="QAIC"):
        batch = examples[i:i + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        
        try:
            # Generate using QAIC with batch of prompts
            exec_info = model.generate(
                tokenizer=tokenizer,
                prompts=prompts,  # Pass list of prompts for batch processing
                device_id=device_group,
                generation_len=min(max_new_tokens, ctx_len - prefill_seq_len),
            )
            
            # Extract generated texts for the batch
            batch_generated_texts = []
            if hasattr(exec_info, 'generated_texts'):
                generated_texts = exec_info.generated_texts
                
                # Handle different possible output structures
                if isinstance(generated_texts, list):
                    for gen_text in generated_texts:
                        if isinstance(gen_text, list):
                            # If it's a nested list, take the first element
                            text = gen_text[0] if len(gen_text) > 0 else ""
                        else:
                            text = gen_text
                        batch_generated_texts.append(str(text).strip())
                else:
                    # Fallback: single text for whole batch
                    batch_generated_texts = [str(generated_texts).strip()]
            
            # Ensure we have the right number of outputs
            while len(batch_generated_texts) < len(batch):
                batch_generated_texts.append("")
            
            # Extract answers for each example in the batch
            for j, (generated_text, example) in enumerate(zip(batch_generated_texts, batch)):
                answer = DatasetLoader.extract_answer_from_generation(
                    generated_text,
                    example["type"]
                )
                predictions.append(answer)
            
        except Exception as e:
            print(f"Error generating for batch starting at index {i}: {e}")
            # Add empty predictions for failed batch
            for _ in range(len(batch)):
                predictions.append("")
    
    print(f"QAIC evaluation complete")
    return predictions


def compute_and_display_metrics(
    predictions: List[str],
    references: List[str],
    dataset_name: str,
    eval_type: str
) -> Dict:
    """
    Compute metrics and display results.
    
    Args:
        predictions: List of predictions
        references: List of reference answers
        dataset_name: Name of the dataset
        eval_type: Type of evaluation ("Baseline" or "QAIC")
    
    Returns:
        Dictionary containing computed metrics
    """
    metric_handler = MetricHandler(dataset_name)
    metrics = metric_handler.compute(predictions, references)
    
    print(f"\n{eval_type} Results for {dataset_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Correct: {metrics['correct']}/{metrics['total']}")
    
    return metrics


def save_results(results: Dict, output_dir: str, filename: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to: {filepath}")


def create_comparison_report(
    baseline_results: Dict,
    qaic_results: Dict,
    output_dir: str
):
    """Create comparison report and visualization."""
    
    # Create comparison dataframe
    comparison_data = []
    
    for dataset in baseline_results.keys():
        if dataset in qaic_results:
            baseline_acc = baseline_results[dataset]["accuracy"]
            qaic_acc = qaic_results[dataset]["accuracy"]
            diff = qaic_acc - baseline_acc
            
            comparison_data.append({
                "dataset": dataset,
                "baseline_accuracy": baseline_acc,
                "qaic_accuracy": qaic_acc,
                "difference": diff,
                "relative_change_pct": (diff / baseline_acc * 100) if baseline_acc > 0 else 0
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Save CSV
    csv_path = os.path.join(output_dir, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved comparison to: {csv_path}")
    
    # Create visualizations
    if len(df) > 0:
        # Create main comparison graph
        create_accuracy_comparison_graph(df, output_dir)
        
        # Create additional detailed graph
        create_detailed_comparison_graph(df, output_dir)
    
        # Print summary
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Dataset':<15} {'Baseline':<12} {'QAIC':<12} {'Difference':<12} {'Rel %':<10}")
        print("-" * 70)
        for _, row in df.iterrows():
            print(f"{row['dataset']:<15} "
                  f"{row['baseline_accuracy']:<12.4f} "
                  f"{row['qaic_accuracy']:<12.4f} "
                  f"{row['difference']:<+12.4f} "
                  f"{row['relative_change_pct']:<+10.2f}%")
        
        if len(df) > 1:
            print("\n" + "-" * 70)
            print(f"{'Average':<15} "
                  f"{df['baseline_accuracy'].mean():<12.4f} "
                  f"{df['qaic_accuracy'].mean():<12.4f} "
                  f"{df['difference'].mean():<+12.4f} "
                  f"{df['relative_change_pct'].mean():<+10.2f}%")
        print("="*70 + "\n")


def create_accuracy_comparison_graph(df: pd.DataFrame, output_dir: str):
    """
    Create main accuracy comparison graph showing baseline vs QAIC.
    
    Args:
        df: DataFrame with comparison data
        output_dir: Directory to save the graph
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(VIZ_FIGURE_WIDTH, VIZ_FIGURE_HEIGHT))
    
    datasets = df["dataset"].values
    baseline_acc = df["baseline_accuracy"].values
    qaic_acc = df["qaic_accuracy"].values
    
    x = np.arange(len(datasets))
    
    # Plot 1: Side-by-side comparison
    bars1 = ax1.bar(x - VIZ_BAR_WIDTH/2, baseline_acc, VIZ_BAR_WIDTH, 
                    label="Baseline (GPU)", color=COLOR_BASELINE, 
                    alpha=VIZ_ALPHA, edgecolor='black', linewidth=VIZ_EDGE_WIDTH)
    bars2 = ax1.bar(x + VIZ_BAR_WIDTH/2, qaic_acc, VIZ_BAR_WIDTH, 
                    label="QAIC (Cloud AI 100)", color=COLOR_QAIC, 
                    alpha=VIZ_ALPHA, edgecolor='black', linewidth=VIZ_EDGE_WIDTH)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Accuracy Comparison: Baseline vs QAIC', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.set_ylim(0, max(max(baseline_acc), max(qaic_acc)) * 1.15)
    ax1.grid(True, alpha=VIZ_GRID_ALPHA, axis='y', linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Accuracy difference
    differences = df["difference"].values
    colors = [COLOR_POSITIVE if d >= 0 else COLOR_NEGATIVE for d in differences]
    
    bars = ax2.bar(x, differences, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=VIZ_EDGE_WIDTH)
    
    # Add value labels
    for i, (bar, diff, pct) in enumerate(zip(bars, differences, df["relative_change_pct"].values)):
        label_y = diff + (0.005 if diff >= 0 else -0.005)
        ax2.text(i, label_y, f'{diff:+.3f}\n({pct:+.1f}%)',
                ha='center', va='bottom' if diff >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2.5, alpha=0.8)
    ax2.set_ylabel('Accuracy Difference (QAIC - Baseline)', fontsize=13, fontweight='bold')
    ax2.set_title('QAIC Performance Delta\n(Positive = QAIC Better)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=VIZ_GRID_ALPHA, axis='y', linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(viz_path, dpi=VIZ_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved accuracy comparison graph to: {viz_path}")


def create_detailed_comparison_graph(df: pd.DataFrame, output_dir: str):
    """
    Create detailed comparison graph with multiple views.
    
    Args:
        df: DataFrame with comparison data
        output_dir: Directory to save the graph
    """
    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(VIZ_DETAILED_WIDTH, VIZ_FIGURE_HEIGHT))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    datasets = df["dataset"].values
    baseline_acc = df["baseline_accuracy"].values
    qaic_acc = df["qaic_accuracy"].values
    differences = df["difference"].values
    
    # Plot 1: Grouped bar chart (spans 2 rows, 2 columns)
    ax1 = fig.add_subplot(gs[:, :2])
    x = np.arange(len(datasets))
    
    bars1 = ax1.bar(x - VIZ_BAR_WIDTH/2, baseline_acc * 100, VIZ_BAR_WIDTH, 
                    label="Baseline (GPU)", color=COLOR_BASELINE, 
                    alpha=VIZ_ALPHA, edgecolor='black', linewidth=VIZ_EDGE_WIDTH)
    bars2 = ax1.bar(x + VIZ_BAR_WIDTH/2, qaic_acc * 100, VIZ_BAR_WIDTH, 
                    label="QAIC (Cloud AI 100)", color=COLOR_QAIC, 
                    alpha=VIZ_ALPHA, edgecolor='black', linewidth=VIZ_EDGE_WIDTH)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('Baseline vs QAIC Accuracy Comparison', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=11, fontweight='bold', rotation=0)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax1.set_ylim(0, max(max(baseline_acc), max(qaic_acc)) * 115)
    ax1.grid(True, alpha=VIZ_GRID_ALPHA, axis='y', linestyle='--', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Scatter plot showing correlation (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(baseline_acc * 100, qaic_acc * 100, s=150, alpha=0.7, 
                color=COLOR_SCATTER, edgecolors='black', linewidth=2)
    
    # Add diagonal line (perfect correlation)
    min_val = min(min(baseline_acc), min(qaic_acc)) * 100
    max_val = max(max(baseline_acc), max(qaic_acc)) * 100
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Match')
    
    # Add labels for each point
    for i, dataset in enumerate(datasets):
        ax2.annotate(dataset, (baseline_acc[i] * 100, qaic_acc[i] * 100), 
                    fontsize=8, ha='right', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Baseline Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('QAIC Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy Correlation', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=VIZ_GRID_ALPHA, linestyle='--', linewidth=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Relative performance (bottom right)
    ax3 = fig.add_subplot(gs[1, 2])
    colors = [COLOR_POSITIVE if d >= 0 else COLOR_NEGATIVE for d in differences]
    bars = ax3.barh(datasets, df["relative_change_pct"].values, color=colors, 
                    alpha=0.8, edgecolor='black', linewidth=VIZ_EDGE_WIDTH)
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, df["relative_change_pct"].values)):
        width = bar.get_width()
        label_x = width + (0.5 if width >= 0 else -0.5)
        ax3.text(label_x, i, f'{pct:+.1f}%',
                ha='left' if width >= 0 else 'right', va='center', 
                fontsize=9, fontweight='bold')
    
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2.5, alpha=0.8)
    ax3.set_xlabel('Relative Change (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Relative Performance\n(QAIC vs Baseline)', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=VIZ_GRID_ALPHA, axis='x', linestyle='--', linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add overall title
    fig.suptitle('Comprehensive Accuracy Analysis: Baseline vs QAIC', 
                fontsize=16, fontweight='bold', y=0.98)
    
    viz_path = os.path.join(output_dir, "detailed_comparison.png")
    plt.savefig(viz_path, dpi=VIZ_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved detailed comparison graph to: {viz_path}")


def main():
    """Main execution."""
    args = parse_args()
    
    print("="*60)
    print("HuggingFace Evaluate Benchmark Runner")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Output Dir: {args.output_dir}")
    print("="*60)
    
    # Parse datasets and device group once
    datasets = [d.strip() for d in args.datasets.split(',')]
    device_group = [int(x.strip()) for x in args.device_group.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store results
    baseline_results = {}
    qaic_results = {}
    
    # Run evaluations for each dataset with progress bar
    print(f"\n{'='*60}")
    print(f"Processing {len(datasets)} dataset(s)")
    print(f"{'='*60}\n")
    
    for dataset_name in tqdm(datasets, desc="Overall Progress", unit="dataset"):
        print(f"\n{'#'*60}")
        print(f"# Processing Dataset: {dataset_name.upper()}")
        print(f"{'#'*60}")
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}...")
        loader = DatasetLoader(dataset_name, args.num_samples)
        examples = loader.load()
        references = [ex["reference"] for ex in examples]
        print(f"Loaded {len(examples)} examples")
        
        # Run baseline evaluation
        if not args.skip_baseline:
            baseline_predictions = run_baseline_evaluation(
                args.model,
                dataset_name,
                examples,
                args.device,
                args.max_new_tokens,
                precision=args.baseline_precision,
                batch_size=args.batch_size
            )
            
            # Compute and display metrics
            baseline_results[dataset_name] = compute_and_display_metrics(
                baseline_predictions,
                references,
                dataset_name,
                "Baseline"
            )
        
        # Run QAIC evaluation
        if not args.skip_qaic:
            qaic_predictions = run_qaic_evaluation(
                args.model,
                dataset_name,
                examples,
                args.prefill_seq_len,
                args.ctx_len,
                args.num_cores,
                device_group,
                args.max_new_tokens,
                batch_size=args.batch_size
            )
            
            # Compute and display metrics
            qaic_results[dataset_name] = compute_and_display_metrics(
                qaic_predictions,
                references,
                dataset_name,
                "QAIC"
            )
    
    # Save results
    if baseline_results:
        save_results(baseline_results, args.output_dir, "baseline_results.json")
    
    if qaic_results:
        save_results(qaic_results, args.output_dir, "qaic_results.json")
    
    # Create comparison report
    if baseline_results and qaic_results:
        create_comparison_report(baseline_results, qaic_results, args.output_dir)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
