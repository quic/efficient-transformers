**This page give you an overview about the all the APIs that you might need to integrate the `QEfficient` into your python applications.**

# High Level API

## `QEFFAutoModelForCausalLM`
```{eval-rst}
.. autoclass:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM
   :member-order: bysource
   :members:
``` 
(QEFFAutoModel)=
## `QEFFAutoModel`
```{eval-rst}
.. autoclass:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel
   :member-order: bysource
   :members:
``` 
(QEffAutoPeftModelForCausalLM)=
## `QEffAutoPeftModelForCausalLM`
```{eval-rst}
.. autoclass:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM
   :member-order: bysource
   :members:
```
(QEffAutoLoraModelForCausalLM)=
## `QEffAutoLoraModelForCausalLM`
```{eval-rst}
.. autoclass:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM
   :member-order: bysource
   :members:
```

## `export`
```{eval-rst}
.. automodule:: QEfficient.exporter.export_hf_to_cloud_ai_100
   :members:
   :show-inheritance:
   :exclude-members: convert_to_cloud_kvstyle, convert_to_cloud_bertstyle
.. deprecated::
   This function will be deprecated in version 1.19, please use QEFFAutoModelForCausalLM.export instead
```
## `compile`
```{eval-rst}
.. automodule:: QEfficient.compile.compile_helper
   :members:
   :show-inheritance:
.. code-block:: python

   import QEfficient
   base_path, onnx_model_path = QEfficient.export(model_name="gpt2")
   qpc_path = QEfficient.compile(onnx_path=onnx_model_path, qpc_path=os.path.join(base_path, "qpc"), num_cores=14, device_group=[0])
.. deprecated::
   This function will be deprecated in version 1.19, please use QEFFAutoModelForCausalLM.compile instead
```
## `Execute`
```{eval-rst}
.. automodule:: QEfficient.generation.text_generation_inference
   :members:
   :show-inheritance:
   :exclude-members:  latency_stats_bertstyle,cloud_ai_100_exec_kv_helper
```
