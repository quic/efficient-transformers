![alt text](docs/image/Cloud_AI_100.png)


---
# Efficient Transformers Library 
---

*Latest news* :fire: <br>

- [coming soon] Support for more popular [models](https://quic.github.io/efficient-transformers/source/validate.html#models-coming-soon) and inference optimization technique speculative decoding <br>
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

# Create Python virtual env and activate it. (Required Python 3.8)

python3.8 -m venv qeff_env
source qeff_env/bin/activate
pip install -U pip

# Clone and Install the QEfficient Repo.
pip install git+https://github.com/quic/efficient-transformers --extra-index-url https://download.pytorch.org/whl/cpu

# Or build wheel package using the below command.
pip install build wheel
python -m build --wheel --outdir dist
pip install dist/QEfficient-0.0.1.dev0-py3-none-any.whl --extra-index-url https://download.pytorch.org/whl/cpu

``` 

For more details about using ``QEfficient`` via Cloud AI 100 Apps SDK, visit [Linux Installation Guide](https://quic.github.io/efficient-transformers/source/installation.html)


## Documentation

* [Quick Start Guide](https://quic.github.io/efficient-transformers/source/quick_start.html#)
* [Python API](https://quic.github.io/efficient-transformers/source/hl_api.html)
* [Validated Models](https://quic.github.io/efficient-transformers/source/validate.html)
* [Models coming soon](models-coming-soon)

> Note: More details are here: https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Model-Architecture-Support/Large-Language-Models/llm/

## Acknowledgements
Thanks to:
* HuggingFace transformers for work in LLM GenAI modeling implementation
*  ONNX, Pytorch,  ONNXruntime community.

## Support
If you run into any problems with the code, please file Github issues directly to this repo.

## Contributing
This project welcomes contributions and suggestions. Please check the License. Integration with a CLA Bot is underway. 
