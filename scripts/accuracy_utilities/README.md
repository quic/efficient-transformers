# Onnx FP16 Out of Range Analyzer

This script at the ONNX level to summarize the activations for a given input and identify the out-of-range operations and layers in the ONNX model.

## Table of Contents

- Requirements
- Installation
- Usage
  - Example
- Arguments
- Output Details

## Requirements

- Python 3.8+
- Required Python packages:
  - `QEfficient`

## Installation

- Install QEfficient

## Usage

To run the script, use the following command:

```bash
python onnx_debug_fp16_tracer.py--model_path /path/to/onnx --inputs {''}
```

## Arguments (Help Section)
```bash
usage: onnx_debug_fp16_tracer.py [-h] --model_path MODEL_PATH --inputs INPUTS

Debugger for ONNX models

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to ONNX model
  --inputs INPUTS       Inputs to Onnx in Dict [str: value]
```

## Output Details
The script logs the following information:

- Dumps the summary of ops crossing the FP16 activations for given inputs
- NPI Yaml file with Ops to run in FP32 to restore Accuracy






