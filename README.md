![alt text](docs/image/Cloud_AI_100.png)


---
# Efficient Transformers Library 
---

*Latest news* :fire: <br>

- [coming soon] Support for more popular [models](https://quic.github.io/efficient-transformers/source/validate.html#models-coming-soon) and inference optimization technique speculative decoding <br>
- [11/2024] [finite adapters support](https://github.com/quic/efficient-transformers/pull/153) allows mixed adapter usage for peft models.
- [09/2024] [AWQ](https://arxiv.org/abs/2306.00978)/[GPTQ](https://arxiv.org/abs/2210.17323) 4-bit quantized models are supported
- [09/2024] Now we support [PEFT](https://huggingface.co/docs/peft/index) models
- [09/2024] Added support for [Gemma-2-Family](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- [09/2024] Added support for [CodeGemma-Family](https://huggingface.co/collections/google/codegemma-release-66152ac7b683e2667abdee11)
- [09/2024] Added support for [Gemma-Family](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b)
- [09/2024] Added support for [Meta-Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), [70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [405B](https://huggingface.co/meta-llama/Llama-3.1-405B)
- [09/2024] Added support for [granite-20b-code-base](https://huggingface.co/ibm-granite/granite-20b-code-base-8k), [granite-20b-code-instruct-8k](https://huggingface.co/ibm-granite/granite-20b-code-instruct-8k)
- [08/2024] Added support for inference optimization technique ```continuous batching```
- [08/2024] Added support for [Jais-adapted-70b](https://huggingface.co/inceptionai/jais-adapted-70b), [Jais-adapted-13b-chat](https://huggingface.co/inceptionai/jais-adapted-13b-chat), [Jais-adapted-7b](https://huggingface.co/inceptionai/jais-adapted-7b)


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
pip install dist/QEfficient-0.0.1.dev0-py3-none-any.whl

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
