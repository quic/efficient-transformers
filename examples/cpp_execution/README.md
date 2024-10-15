
# Text Generation using CPP Inference

## Overview
This example contains a single C++ file, an example text generation inferencing python file and a CMakeLists.txt that can be used for compiling.

## Requirements
1. PyBind11
2. Cpp17 and above

## Running Guide
```bash

# Compile the cpp file using the following commands
mkdir build
cd build

cmake ..
make -j 8

# Run the python script to get the generated text
cd ../../../
python -m examples.cpp_execution.text_inference_using_cpp --model_name gpt2 --batch_size 1 --prompt_len 32 --ctx_len 128 --mxfp6 --num_cores 14 --device_group [0] --prompt "My name is" --mos 1 --aic_enable_depth_first

```

## Limitations
1. Currently this is working for Batch size = 1
2. Not supporting DMA Buffer Handling
