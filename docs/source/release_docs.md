# Efficient Transformer Library - Release 1.20.0

We're thrilled to introduce new model support and features in the Efficient Transformer library, bringing powerful enhancements and expanded capabilities to our platform.

All the below mentioned features and newly onboarded models are supported on Release 1.20.0 branch and mainline.

## Newly Onboarded Models

- **Llama-4-Scout-17B-16E-Instruct**
  - Sample script for Text only (Recommended class for testing is `QEFFAutoModelForImageTextToText`).
  - Sample script for Image + Text (Recommended class for testing is `QEFFAutoModelForImageTextToText`).
  - Single QPC + Dual QPC support (please check the comment section of example script for running single QPC).
  - Added support for chunk attention in Llama4.
  - Continuous batching and multi batch execution is planned for rel#1.21.0 (FR JIRA).
  - With the redefined interface between QEFF and VLLM, we should be able to run the multiple images in single prompt, please follow (example) and see sample completion below.

- **Grok-1**
  - Since architecture for this model is `Grok1ModelForCausalLM`, it can be executed using `QEffAutoModelForCausalLM`.

- **Gemma3**
  - Sample script for Text only (Recommended class for testing is `QEFFAutoModelForImageTextToText`).
  - Sample script for Image + Text.
  - Added support for sliding window.
  - Continuous batching and multi batch execution is planned for rel#1.21.0 (FR JIRA).

- **Granite Vision models**
  - Sample script.

- **Granite MOE models**

## New Features

- Upgraded Transformer version to 4.51.3.

- **SpD, multiprojection heads**
  - Implemented post-attention hidden size projections to speculate tokens ahead of the base model.

- **Compilation support for io_encrypt flag**
  - Added support for Model-IP I/O encryption feature using `qaic-exec` (compile only).
  - Users can now directly pass the `--io-encrypt` flag in both high-level APIs (compile) and command-line APIs (infer and compile).

- **Support for separate prefill and decode compilation**
  - Added support for separate prefill and decode compilation for encoder (vision) and language models. This feature will be utilized for disaggregated serving.

- **New features for Embedding Models**
  - **Flexible Pooling configuration:**
    - User can specify popular pooling strategies via string identifiers or provide custom pooling methods.
    - Enables seamless integration of pooling at the end of the embedding model, offering flexibility for various use cases. Pooling will also run on AI 100 for improved performance.
    - Sample script.
    - Added support for sentence embedding.
    - With pooling added, Efficient-Transformers now enables direct sentence embedding generation on AI 100, improving efficiency and semantic quality for downstream tasks.
  - **Support for compilation with multiple sequence lengths**
    - Users can specify a single or list of `seq_len` values during compilation (example).
    - At generation time, the closest greater or equal `seq_len` graph from the QPC is auto selected for optimal execution.

- **On Device Sampling for CausalLM models**
  - Sampling now runs directly on the QAIC device, reducing host-device communication and boosting inference throughput and scalability.
  - Documentation and Usage guide.

- **SwiftKV model (Snowflake/Llama-3.1-SwiftKV-8B-Instruct)**
  - Added support for both continuous and non-continuous batching execution in SwiftKV.
  - Since architecture for this model is `LlamaSwiftKVForCausalLM`, it can be executed using `QEffAutoModelForCausalLM`.

- **Execution of GGUF models (without quantized weights)**
  - Sample script.

- **Compressed quantization status for FP8 model**
  - Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic · Hugging Face

- **QNN updates**
  - Updated the QNN custom IO generation method for adhering to compiler changes.
  - Added `--target_backend AIC` as default parameter in QNN Converter.

## Fine Tuning

- Added Bert FT support, doc.
- Documentation and a code template to run fine tuning on custom dataset.
- Added `--help` option available for usage of training parameters.
- Added support for gradient checkpointing in the finetuning script.
- Added support for passing device type in torch `GradScaler`.
- Detailed documentation is here.

## Upcoming Models

- Qwen3
- Mistral 3.1

## Upcoming Features

- Compute context length support (FR JIRA planned for 1.21.0).
- Support for passing MDP file to compiler during compilation (planned as bug-fix in 1.20.0).
- Upgrading the ONNX dependency to address a security vulnerability identified in the current version of ONNX:
  - `onnx==1.18.0`, `onnxruntime==1.22`, `onnxscript==0.2.5`, `protobuff==6.31.0` (planned for 1.21.0).
- Support for `-inf` for pad tokens, for optimized softmax handling in compiler (planned for 1.21.0).
