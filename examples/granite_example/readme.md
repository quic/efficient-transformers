# InternVL Inference
This directory contains an example script of how to run inference on Granite-vision-3.2-2b via QEFFAutoModelForCausalLM class.

## Required packages:
- `torch==2.4.1+cpu`


You can install them using pip:
```sh
pip install torch==2.4.1+cpu 
```

To run example script after package installations:
```sh
python granite_vision_inference.py
```

Expected output for given sample inputs in the script:
```sh
The highest scoring model on ChartQA is Granite Vision with a score of 0.87.
```