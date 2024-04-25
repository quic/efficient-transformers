## Validation of Model before and After KV Cache Optimization

# Run APIs

| API    | <left>params</left> | Usage            |
|------------------|:--------:|:--------------------:|
| run_hf_model_on_pytorch |  <ol><li>model_hf</li></ol>        |<li>Runs HuggingFace model on Pytorch and returns output tokens</li> |
| run_kv_model_on_pytorch |  <ol><li>model_kv</li><li>n_layer</li><li>padding_shape</li></ol>        |<li>Runs KV cache model on PyTorch and returns output tokens</li> |
| run_kv_model_on_ort |  <ol><li>model_path</li><li>n_layer</li><li>padding_shape</li></ol>        |<li>Runs KV cache model on onnxruntime and returns output tokens</li> |
| run_kv_model_on_cloud_ai_100 |  <ol><li>qpc_path</li><li>device_id</li><li>n_layer</li><li>padding_shape</li></ol>        |<li>To be run only when device is available </li> runs compiled qpc on Cloud AI 100 and returns output tokens</li> |

Class InputHandler is called inside these APIs which generates inputs for prefill and decode stage.

These APIs, input_generation, checking available device id and Constants scripts are present inside tests/utils folder.

# Sample Usage :

These APIs are part of ApiRunner class :

      n_layer : number of layers in the model
      padding_shape : shape of past key values to initialize the first iteration key value inputs with zeros.</li>

<li>run_api = ApiRunner(tokenizer, Constants.INPUT_STRING, Constants.PROMPT_LEN, Constants.CTX_LEN) </li>
<li>pytorch_hf_tokens = run_api.run_hf_model_on_pytorch(model_hf) </li>
<li>pytorch_kv_tokens = run_api.run_kv_model_on_pytorch(model_kv, n_layer, padding_shape) </li>
<li>ort_tokens = run_api.run_kv_model_on_ort(onnx_model_path, n_layer, padding_shape) </li>
<li>cloud_ai_100_tokens = run_api.run_kv_model_on_cloud_ai_100(session, n_layer, padding_shape) </li>

Output tokens are compared to validate the model on the required framework level (PyTorch/ONNX/cloud_ai_100).</li>
