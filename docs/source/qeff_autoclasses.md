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

### MDP Compile Options

For `QEFFAutoModelForCausalLM.compile()`, pass `num_devices` for the total device count and `mdp_num_partitions` for the number of pipeline-parallel partitions. Tensor-slice devices per MDP partition (`mdp_ts_num_devices`) are calculated internally as `num_devices // mdp_num_partitions`.

Do not pass `mdp_ts_num_devices` to this public `compile()` API. It is ignored with a warning when passed through `**compiler_options`.

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

---
(QEFFAutoModelForSequenceClassification)=
## `QEFFAutoModelForSequenceClassification`

```{eval-rst}
.. autoclass:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSequenceClassification
   :noindex:
   :no-members:
   :no-show-inheritance:
```

### High-Level API

```{eval-rst}
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSequenceClassification.from_pretrained
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSequenceClassification.export
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSequenceClassification.compile
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForSequenceClassification.generate
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

### MDP Compile Options

`QEFFAutoModelForImageTextToText.from_pretrained(...)` returns a concrete wrapper whose `compile()` API follows the same MDP convention as `QEFFAutoModelForCausalLM`: pass `num_devices` as the total device count and `mdp_num_partitions` as the pipeline-parallel partition count. Tensor-slice devices per MDP partition (`mdp_ts_num_devices`) are calculated internally as `num_devices // mdp_num_partitions`.

Do not pass `mdp_ts_num_devices` to the public auto-model `compile()` API. It is ignored with a warning when passed through `**compiler_options`.

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

(QEFFAutoModelForCTC)=
## `QEFFAutoModelForCTC`


```{eval-rst}
.. autoclass:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCTC
   :noindex:
   :no-members:
   :no-show-inheritance:
```

### High-Level API

```{eval-rst}
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCTC.from_pretrained
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCTC.export
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCTC.compile
.. automethod:: QEfficient.transformers.models.modeling_auto.QEFFAutoModelForCTC.generate
```
