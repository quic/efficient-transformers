# InternVL Inference
This directory contains an example script of how to run inference on InternVL-1B model via QEFFAutoModelForCausalLM class.

## Required packages:
- `torch==2.7.1+cpu`
- `torchvision==0.22.1+cpu`
- `timm==1.0.14`
- `einops==0.8.1`

You can install them using pip:
```sh
pip install torch==2.7.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu timm==1.0.14 torchvision==0.22.1+cpu einops==0.8.1
```

To run example script after package installations:
```sh
python internvl_inference.py
```

Expected output for given sample inputs in the script:
```sh
The image is a promotional graphic for Microsoft Azure. It features a blue background with a hexagonal pattern on the left side. The hexagons are white and are arranged in a way that suggests a network or connectivity theme. 

On the right side of the image, the Microsoft Azure logo is prominently displayed. The logo consists of the Azure name in white, with the Microsoft logo above it, which includes four colored squares (blue, green, yellow, and red). Below the logo, the word "Azure" is written in large white letters.

Below the logo, there is text that reads:
- "By Dinesh Kumar Wick
```