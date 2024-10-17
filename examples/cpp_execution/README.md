
# Text Generation using CPP Inference

## Overview
This example demonstrates how to execute a model on AI 100 using Efficient Transformers and C++ APIs. The Efficient Transformers library is utilized for transforming and compiling the model, while the QPC is executed using C++ APIs. It is tested on both x86 and ARM platform.

## Prerequisite
1. PyBind11
2. Cpp17 or above
3. QEfficient [Quick Installation Guide]( https://github.com/quic/efficient-transformers?tab=readme-ov-file#quick-installation)

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

## Future Enhancements
1. Batch size > 1 support currently under evaluation.
2. Chunking
3. DMA Buffer Handling
4. Continuous Batching
5. Handling streamer
