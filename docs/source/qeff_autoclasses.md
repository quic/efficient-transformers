# QEfficient Auto Classes

(QEFFAutoModelForCausalLM)=
## `QEFFAutoModelForCausalLM`

```{eval-rst}
.. autoclass:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM
   :noindex:
   :no-members:
   :no-show-inheritance:
```

### High-Level API

```{eval-rst}
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.from_pretrained
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.export
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.compile
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.generate
```

### Low-Level API

```{eval-rst}
.. autoproperty:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.model_name
.. autoproperty:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.get_model_config
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.get_sampling_inputs_and_outputs
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.build_prefill_specialization
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.build_decode_specialization
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCausalLM.check_and_get_num_speculative_tokens
```

---
(QEFFAutoModel)=
## `QEFFAutoModel`

```{eval-rst}
.. autoclass:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel
   :noindex:
   :no-members:
   :no-show-inheritance:
```

### High-Level API

```{eval-rst}
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel.from_pretrained
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel.export
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel.compile
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel.generate
```

### Low-Level API

```{eval-rst}
.. autoproperty:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel.get_model_config
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel.cloud_ai_100_feature_generate
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModel.pytorch_feature_generate
```

---
(QEffAutoPeftModelForCausalLM)=
## `QEffAutoPeftModelForCausalLM`

```{eval-rst}
.. autoclass:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM
   :noindex:
   :no-members:
   :no-show-inheritance:
```

### High-Level API

```{eval-rst}
.. automethod:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.from_pretrained
.. automethod:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.export
.. automethod:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.compile
.. automethod:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.generate
```

### Low-Level API

```{eval-rst}
.. autoproperty:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.model_name
.. autoproperty:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.model_hash
.. autoproperty:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.get_model_config
.. autoproperty:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.active_adapter
.. automethod:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.load_adapter
.. automethod:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.set_adapter
.. automethod:: QEfficient.peft.auto.QEffAutoPeftModelForCausalLM.disable_adapter
```

---
(QEffAutoLoraModelForCausalLM)=
## `QEffAutoLoraModelForCausalLM`

```{eval-rst}
.. autoclass:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM
   :noindex:
   :no-members:
   :no-show-inheritance:
```

### High-Level API

```{eval-rst}
.. automethod:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.from_pretrained
.. automethod:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.export
.. automethod:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.compile
.. automethod:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.generate
```

### Low-Level API

```{eval-rst}
.. autoproperty:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.model_hash
.. autoproperty:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.get_model_config
.. automethod:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.download_adapter
.. automethod:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.load_adapter
.. automethod:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.unload_adapter
.. automethod:: QEfficient.peft.lora.auto.QEffAutoLoraModelForCausalLM.set_adapter
```

---
(QEFFAutoModelForImageTextToText)=
## `QEFFAutoModelForImageTextToText`

```{eval-rst}
.. autoclass:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForImageTextToText
   :noindex:
   :no-members:
   :no-show-inheritance:
```

### High-Level API

```{eval-rst}
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForImageTextToText.from_pretrained
```

---
(QEFFAutoModelForSpeechSeq2Seq)=
## `QEFFAutoModelForSpeechSeq2Seq`

```{eval-rst}
.. autoclass:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSpeechSeq2Seq
   :noindex:
   :no-members:
   :no-show-inheritance:
```

### High-Level API

```{eval-rst}
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSpeechSeq2Seq.from_pretrained
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSpeechSeq2Seq.export
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSpeechSeq2Seq.compile
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSpeechSeq2Seq.generate
```

### Low-Level API

```{eval-rst}
.. autoproperty:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSpeechSeq2Seq.get_model_config
```