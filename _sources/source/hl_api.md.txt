**This page give you an overview about the all the APIs that you might need to integrate the `QEfficient` into your python applications.**

# High Level API

## `QEFFAutoModelForCausalLM`
```{eval-rst}
.. automodule:: QEfficient.transformers.models.modeling_auto
   :inherited-members:
   :undoc-members: 
   :exclude-members: QEffAutoModel,QEFFTransformersBase, run_ort, run_pytorch, get_tokenizer, run_cloud_ai_100, execute
``` 

## `QEffAutoPeftModelForCausalLM`
```{eval-rst}
.. autoclass:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM
   :members:
```

## `export`
```{eval-rst}
.. automodule:: QEfficient.exporter.export_hf_to_cloud_ai_100
   :members:
   :show-inheritance:
   :exclude-members: convert_to_cloud_kvstyle, convert_to_cloud_bertstyle
```
## `compile`
```{eval-rst}
.. automodule:: QEfficient.compile.compile_helper
   :members:
   :show-inheritance:
.. code-block:: bash
   import QEfficient
   base_path, onnx_model_path = QEfficient.export(model_name="gpt2")
   qpc_path = QEfficient.compile(onnx_path=onnx_model_path, qpc_path=os.path.join(base_path, "qpc"), num_cores=14, device_group=[0])
```
## `Execute`
```{eval-rst}
.. automodule:: QEfficient.generation.text_generation_inference
   :members:
   :show-inheritance: 
   :exclude-members:  latency_stats_bertstyle,cloud_ai_100_exec_kv_helper
```