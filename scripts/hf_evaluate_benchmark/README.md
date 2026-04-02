# HuggingFace Evaluate Benchmark Suite

Simple, clean benchmark comparing baseline GPU models with QAIC-compiled models using HuggingFace Evaluate library.

## Overview

This benchmark suite provides a straightforward way to compare accuracy between:
- **Baseline**: HuggingFace models running on GPU/CPU
- **QAIC**: QEfficient-compiled models running on Qualcomm AI Cloud hardware

**Key Features:**
- ✅ Uses HuggingFace Evaluate library (not lm-eval)
- ✅ Direct text generation comparison (no log-likelihood approximations)
- ✅ Simple, clean implementation (~400 lines total)
- ✅ Supports 3 representative datasets
- ✅ Automatic comparison report and visualization

## Supported Datasets

| Dataset | Type | Metric | Description |
|---------|------|--------|-------------|
| **GSM8K** | Generation | Exact Match | Math word problems |
| **BoolQ** | Classification | Accuracy | Yes/No questions |
| **HellaSwag** | Multiple Choice | Accuracy | Common sense reasoning |

## Installation

```bash
# Install required packages
pip install datasets evaluate transformers torch pandas matplotlib tqdm

# For QAIC evaluation, install QEfficient
pip install -e /path/to/QEfficient
```

## Quick Start

### 1. Run Complete Benchmark (Baseline + QAIC)

```bash
cd efficient-transformers/scripts/hf_evaluate_benchmark

python run_benchmark.py \
  --model meta-llama/Llama-3.2-1B \
  --datasets gsm8k,boolq \
  --num-samples 10 \
  --output-dir ./results
```

### 2. Run Only Baseline (GPU)

```bash
python run_benchmark.py \
  --model Qwen/Qwen2-1.5B-Instruct \
  --datasets gsm8k \
  --num-samples 10 \
  --skip-qaic
```

### 3. Run Only QAIC

```bash
python run_benchmark.py \
  --model Qwen/Qwen2-1.5B-Instruct \
  --datasets gsm8k \
  --num-samples 10 \
  --skip-baseline
```

## Command-Line Options

### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model ID | `Qwen/Qwen2-1.5B-Instruct` |
| `--datasets` | Comma-separated datasets | `gsm8k` |
| `--num-samples` | Samples per dataset | `10` |
| `--output-dir` | Output directory | `./results` |
| `--skip-baseline` | Skip baseline evaluation | `False` |
| `--skip-qaic` | Skip QAIC evaluation | `False` |

### Generation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device for baseline | `cuda` (if available) |
| `--max-new-tokens` | Max tokens to generate | `128` |

### QAIC Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prefill-seq-len` | Prefill sequence length | `32` |
| `--ctx-len` | Context length | `512` |
| `--num-cores` | Number of cores | `16` |
| `--device-group` | Device IDs (comma-separated) | `0` |

## Output Structure

```
results/
├── baseline_results.json     # Baseline metrics
├── qaic_results.json         # QAIC metrics
├── comparison.csv            # Side-by-side comparison
└── comparison.png            # Visualization
```

### Example Output Files

**baseline_results.json:**
```json
{
  "gsm8k": {
    "accuracy": 0.7500,
    "correct": 75,
    "total": 100,
    "error_rate": 0.2500
  }
}
```

**comparison.csv:**
```csv
dataset,baseline_accuracy,qaic_accuracy,difference,relative_change_pct
gsm8k,0.7500,0.7300,-0.0200,-2.67
boolq,0.8200,0.8100,-0.0100,-1.22
```

## Usage Examples

### Example 1: Quick Test (10 samples)

```bash
python run_benchmark.py \
  --model Qwen/Qwen2-1.5B-Instruct \
  --datasets gsm8k \
  --num-samples 10
```

### Example 2: Multiple Datasets

```bash
python run_benchmark.py \
  --model Qwen/Qwen2-1.5B-Instruct \
  --datasets gsm8k,boolq,hellaswag \
  --num-samples 50
```

### Example 3: Full Evaluation

```bash
python run_benchmark.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --datasets gsm8k,boolq,hellaswag \
  --num-samples 500 \
  --output-dir ./results_llama3
```

### Example 4: Custom QAIC Configuration

```bash
python run_benchmark.py \
  --model Qwen/Qwen2-1.5B-Instruct \
  --datasets gsm8k \
  --num-samples 100 \
  --prefill-seq-len 64 \
  --ctx-len 1024 \
  --num-cores 16 \
  --device-group 0,1,2,3
```

## Understanding Results

### Accuracy Metrics

- **Baseline Accuracy**: True accuracy using GPU inference
- **QAIC Accuracy**: Accuracy using QAIC-compiled model
- **Difference**: QAIC - Baseline (positive = QAIC better)
- **Relative Change %**: Percentage change from baseline

### Interpretation

**For Generation Tasks (GSM8K):**
- Direct comparison is valid
- Both use same evaluation method (text generation)
- Differences indicate actual accuracy impact

**For Classification Tasks (BoolQ, HellaSwag):**
- Direct comparison is valid
- Both extract answers from generated text
- Differences indicate actual accuracy impact

### Expected Results

Typical accuracy ranges:
- **GSM8K**: 30-80% (depends on model size)
- **BoolQ**: 60-85%
- **HellaSwag**: 40-70%

Small differences (±5%) between baseline and QAIC are normal and may be due to:
- Numerical precision differences
- Generation sampling variations
- Answer extraction heuristics

## Comparison with lm-eval Approach

### Advantages of HF Evaluate

1. ✅ **Direct comparison**: Same evaluation method for baseline and QAIC
2. ✅ **No approximations**: Uses actual text generation, not log-likelihood proxies
3. ✅ **Simpler code**: No complex wrapper classes needed
4. ✅ **More transparent**: Clear what's being measured
5. ✅ **Better for generation tasks**: Designed for text outputs

### Limitations

1. ⚠️ **Manual dataset handling**: Need to implement dataset loaders
2. ⚠️ **Less standardized**: lm-eval provides consistent protocol
3. ⚠️ **Answer extraction**: Relies on heuristics to extract answers from text

## Architecture

### Components

```
hf_evaluate_benchmark/
├── __init__.py              # Package initialization
├── dataset_loader.py        # Load and format datasets
├── metric_handler.py        # Compute metrics with HF Evaluate
├── run_benchmark.py         # Main benchmark script
└── README.md               # This file
```

### Data Flow

```
1. Load Dataset → DatasetLoader
   ↓
2. Generate Predictions → Baseline/QAIC Model
   ↓
3. Extract Answers → Answer Extraction
   ↓
4. Compute Metrics → MetricHandler (HF Evaluate)
   ↓
5. Create Report → Comparison & Visualization
```

## Testing Individual Components

### Test Dataset Loader

```bash
python dataset_loader.py
```

### Test Metric Handler

```bash
python metric_handler.py
```

## Troubleshooting

### Issue: "QEfficient not available"

**Solution**: Install QEfficient or use `--skip-qaic` flag

```bash
pip install -e /path/to/QEfficient
```

### Issue: "CUDA out of memory"

**Solution**: Use CPU or reduce batch size

```bash
python run_benchmark.py --device cpu --num-samples 10
```

### Issue: Low accuracy for all models

**Solution**: Check answer extraction logic in `dataset_loader.py`

The `extract_answer_from_generation()` function uses heuristics to extract answers. You may need to adjust these for your specific model's output format.

### Issue: QAIC compilation fails

**Solution**: Adjust QAIC parameters

```bash
python run_benchmark.py \
  --prefill-seq-len 16 \
  --ctx-len 256 \
  --num-cores 8
```

## Best Practices

1. **Start small**: Use `--num-samples 10` for testing
2. **Test baseline first**: Use `--skip-qaic` to verify baseline works
3. **Check answer extraction**: Review generated outputs to ensure answers are extracted correctly
4. **Use appropriate context length**: Set `--ctx-len` based on your prompts
5. **Monitor memory**: QAIC compilation can be memory-intensive

## Extending the Benchmark

### Adding New Datasets

1. Add dataset loader in `dataset_loader.py`:
```python
def _load_new_dataset(self) -> List[Dict]:
    dataset = load_dataset("dataset_name", split="test")
    # Format examples
    return examples
```

2. Add metric in `metric_handler.py`:
```python
def _get_metric(self):
    if self.dataset_name == "new_dataset":
        return evaluate.load("metric_name")
```

3. Add answer extraction logic in `dataset_loader.py`:
```python
@staticmethod
def extract_answer_from_generation(generated_text: str, dataset_type: str) -> str:
    if dataset_type == "new_type":
        # Extract answer
        return answer
```

## Performance Considerations

- **Baseline**: ~1-2 seconds per example (GPU)
- **QAIC**: ~0.5-1 second per example (after compilation)
- **Compilation**: 5-15 minutes (one-time cost)

## Summary

This benchmark suite provides a simple, clean way to compare baseline and QAIC model accuracy using HuggingFace Evaluate. It's designed to be:

- **Easy to use**: Simple command-line interface
- **Easy to understand**: Clear, well-documented code
- **Easy to extend**: Modular architecture

For questions or issues, refer to the code comments or contact the QEfficient team.
