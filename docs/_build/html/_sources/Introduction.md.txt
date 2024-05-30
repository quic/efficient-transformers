![alt text](_static/Cloud_AI_100.png)


---
# Introduciton Qualcomm Transformers Library 

---

*Latest news* :fire: <br>

- [coming soon] support for more popular [models](#models-coming-soon) and inference optimization techniques like continuous batching and speculative decoding <br>
- [04/2024] Initial release of [efficient transformers](https://github.com/quic/efficient-transformers) for seamless inference on pre-trained LLMs.


<span style="font-size: 20px;">**Train anywhere, Infer on Qualcomm Cloud AI with a Developer-centric Toolchain**</span>



This library provides reimplemented blocks of LLMs which are used to make the models functional and highly performant on Qualcomm Cloud AI 100.
There are several models which can be directly transformed from a pre-trained original form to a deployment ready optimized form.
For other models, there is comprehensive documentation to inspire upon the changes needed and How-To(s).

<span style="font-size: 20px;">**Typically for LLMs, the library provides**</span>
1. Reimplemented blocks from Transformers <link> which enable efficient on-device retention of intermediate states.
2. Graph transformations to enable execution of key operations in lower precision
3. Graph transformations to replace some operations to other mathematically equivalent operations
4. Handling for underflows and overflows in lower precision
5. Patcher modules to map weights of original model's operations to updated model's operations
6. Exporter module to export the model source into a ONNX Graph.
7. Sample example applications and demo notebooks
8. Unit test templates. 

<span style="font-size: 20px;">**It is mandatory for each Pull Request to include tests such as**:</span>
1. If the PR is for adding support for a model, the tests should include successful execution of the model post changes (the changes included as part of PR) on Pytorch and ONNXRT. Successful exit criteria is MSE between output of original model and updated model.
2. If the PR modifies any common utilities, tests need to be included to execute tests of all models included in the library.
