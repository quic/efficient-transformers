![alt text](image/Cloud_AI_100.png)


# Introduction Qualcomm ``efficient-transformers`` library
 
**Train anywhere, Infer on Qualcomm Cloud AI with a Developer-centric Toolchain**

This library provides reimplemented blocks of LLMs which are used to make the models functional and highly performant on Qualcomm Cloud AI 100.
We support wide range of [models](validated_models) architectures, for easy efficient deployment on Cloud AI 100 cards. Users only need to provide model card from HuggingFace or Path to the local model and the library will take care of transforming model to it's efficient implementation for Cloud AI 100.

For other models, there is comprehensive documentation to inspire upon the changes needed and How-To(s).

**Typically for LLMs, the library provides:**
1. Reimplemented blocks from Transformers which enable efficient on-device retention of intermediate states. read more [here](kv_cache)
2. Graph transformations to enable execution of key operations in lower precision
3. Graph transformations to replace some operations to other mathematically equivalent operations that are efficient/supported on HW backend
4. Handling for underflow and overflows in lower precision
5. Patcher modules to map weights of original model's operations to updated model's operations
6. Exporter module to export the model source into a `ONNX` Graph.
7. Sample example applications and demo notebooks
8. Unit test templates. 

***Latest news*** : <br>

- [coming soon] Support for more popular [models](coming_soon_models) and inference optimization techniques like continuous batching and speculative decoding <br>
* [06/2024] Added support for [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b)
- [06/2024] Added support for [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct)
- [06/2024] Added support for [StarCoder2-15B](https://huggingface.co/bigcode/starcoder2-15b)
- [06/2024] Added support for [Phi3-Mini-4K-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [06/2024] Added support for [Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1)
- [06/2024] Added support for [Vicuna-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)
- [05/2024] Added support for [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) & [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).
- [04/2024] Initial release of [efficient transformers](https://github.com/quic/efficient-transformers) for seamless inference on pre-trained LLMs.