![alt text](docs/image/Cloud_AI_100.png)


---
# Efficient Transformers Library 
---

*Latest news* :fire: <br>
- [06/2025] Added support for Llama4 Multi-Model [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
- [06/2025] Added support for Gemma3 Multi-Modal-Model [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- [06/2025] Added support of model `hpcai-tech/grok-1` [hpcai-tech/grok-1](https://huggingface.co/hpcai-tech/grok-1)
- [03/2025] Added support for swiftkv model [Snowflake/Llama-3.1-SwiftKV-8B-Instruct](https://huggingface.co/Snowflake/Llama-3.1-SwiftKV-8B-Instruct)
- [02/2025] [VLMs support](https://github.com/quic/efficient-transformers/pull/267) added for the models [InternVL-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B), [Llava](https://huggingface.co/llava-hf/llava-1.5-7b-hf) and [Mllama](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- [01/2025] [FP8 models support](https://huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127) Added support for inference of FP8 models.

- [01/2025] Added support for [Ibm-Granite] (https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)
- [11/2024] [finite adapters support](https://github.com/quic/efficient-transformers/pull/153) allows mixed adapter usage for peft models.
- [11/2024] [Speculative decoding TLM](https://github.com/quic/efficient-transformers/pull/119) QEFFAutoModelForCausalLM model can be compiled for returning more than 1 logits during decode for TLM.
- [11/2024] Added support for [Meta-Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), [Meta-Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and [Meta-Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)

<details>
<summary>More</summary>
- [04/2025] [Granite 3.0 and 3.1 Language MOE Models] (https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-base)
- [09/2024] [AWQ](https://arxiv.org/abs/2306.00978)/[GPTQ](https://arxiv.org/abs/2210.17323) 4-bit quantized models are supported <br>
- [09/2024] Now we support [PEFT](https://huggingface.co/docs/peft/index) models
- [01/2025] Added support for [Ibm-Granite] (https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)
- [01/2025] Added support for [Ibm-Granite-Guardian] (https://huggingface.co/ibm-granite/granite-guardian-3.1-8b)
- [09/2024] Added support for [Gemma-2-Family](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)<br>
- [09/2024] Added support for [CodeGemma-Family](https://huggingface.co/collections/google/codegemma-release-66152ac7b683e2667abdee11)
- [09/2024] Added support for [Gemma-Family](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)
- [09/2024] Added support for [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [09/2024] Added support for [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [09/2024] Added support for [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
- [09/2024] Added support for [granite-20b-code-base](https://huggingface.co/ibm-granite/granite-20b-code-base-8k)
- [09/2024] Added support for [granite-20b-code-instruct-8k](https://huggingface.co/ibm-granite/granite-20b-code-instruct-8k)
- [09/2024] Added support for [Starcoder1-15B](https://huggingface.co/bigcode/starcoder)
- [08/2024] Added support for inference optimization technique ```continuous batching```
- [08/2024] Added support for [Jais-adapted-70b](https://huggingface.co/inceptionai/jais-adapted-70b)
- [08/2024] Added support for [Jais-adapted-13b-chat](https://huggingface.co/inceptionai/jais-adapted-13b-chat)
- [08/2024] Added support for [Jais-adapted-7b](https://huggingface.co/inceptionai/jais-adapted-7b)
- [06/2024] Added support for [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b)
- [06/2024] Added support for [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)
- [06/2024] Added support for [StarCoder2-15B](https://huggingface.co/bigcode/starcoder2-15b)
- [06/2024] Added support for [Phi3-Mini-4K-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [06/2024] Added support for [Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1)
- [06/2024] Added support for [Vicuna-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)
- [05/2024] Added support for [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) & [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).
- [04/2024] Initial release of [efficient transformers](https://github.com/quic/efficient-transformers) for seamless inference on pre-trained LLMs.
</details>

# Overview

## Train anywhere, Infer on Qualcomm Cloud AI with a Developer-centric Toolchain

This library provides reimplemented blocks of LLMs which are used to make the models functional and highly performant on Qualcomm Cloud AI 100.
There are several models which can be directly transformed from a pre-trained original form to a deployment ready optimized form.
For other models, there is comprehensive documentation to inspire upon the changes needed and How-To(s).

## Typically for LLMs, the library provides:
1. Reimplemented blocks from Transformers <link> which enable efficient on-device retention of intermediate states.
2. Graph transformations to enable execution of key operations in lower precision
3. Graph transformations to replace some operations to other mathematically equivalent operations
4. Handling for under-flows and overflows in lower precision
5. Patcher modules to map weights of original model's operations to updated model's operations
6. Exporter module to export the model source into a `ONNX` Graph.
7. Sample example applications and demo notebooks
8. Unit test templates. 

**It is mandatory for each Pull Request to include tests such as**:
1. If the PR is for adding support for a model, the tests should include successful execution of the model post changes (the changes included as part of PR) on Pytorch and ONNXRT. Successful exit criteria is MSE between output of original model and updated model.
2. If the PR modifies any common utilities, tests need to be included to execute tests of all models included in the library.


## Quick Installation
```bash

# Create Python virtual env and activate it. (Recommended Python 3.10)
sudo apt install python3.10-venv
python3.10 -m venv qeff_env
source qeff_env/bin/activate
pip install -U pip

# Clone and Install the QEfficient Repo.
pip install git+https://github.com/quic/efficient-transformers

# Or build wheel package using the below command.
pip install build wheel
python -m build --wheel --outdir dist
pip install dist/qefficient-0.0.1.dev0-py3-none-any.whl

``` 

For more details about using ``QEfficient`` via Cloud AI 100 Apps SDK, visit [Linux Installation Guide](https://quic.github.io/efficient-transformers/source/installation.html)


## Documentation

* [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html#)
* [Python API](https://quic.github.io/efficient-transformers/source/hl_api.html)
* [Validated Models](https://quic.github.io/efficient-transformers/source/validate.html)
* [Models coming soon](https://quic.github.io/efficient-transformers/source/validate.html#models-coming-soon)

> Note: More details are here: https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/

## Acknowledgements
Thanks to:
* HuggingFace transformers for work in LLM GenAI modeling implementation
*  ONNX, Pytorch,  ONNXruntime community.

## Support
If you run into any problems with the code, please file Github issues directly to this repo.

## Contributing
This project welcomes contributions and suggestions. Please check the License. Integration with a CLA Bot is underway. 
