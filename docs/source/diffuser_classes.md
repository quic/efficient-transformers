# Diffuser Classes


## Pipeline API

(QEffTextEncoder)=
### `QEffTextEncoder`

```{eval-rst}
.. autoclass:: QEfficient.diffusers.pipelines.pipeline_module.QEffTextEncoder
   :members:
   :no-show-inheritance:
```

---

(QEffUNet)=
### `QEffUNet`

```{eval-rst}
.. autoclass:: QEfficient.diffusers.pipelines.pipeline_module.QEffUNet
   :members:
   :no-show-inheritance:
```

---

(QEffVAE)=
### `QEffVAE`

```{eval-rst}
.. autoclass:: QEfficient.diffusers.pipelines.pipeline_module.QEffVAE
   :members:
   :no-show-inheritance:
```

---

(QEffFluxTransformerModel)=
### `QEffFluxTransformerModel`

```{eval-rst}
.. autoclass:: QEfficient.diffusers.pipelines.pipeline_module.QEffFluxTransformerModel
   :members:
   :no-show-inheritance:
```

----

(QEffWanUnifiedTransformer)=
### `QEffWanUnifiedTransformer`

```{eval-rst}
.. autoclass:: QEfficient.diffusers.pipelines.pipeline_module.QEffWanUnifiedTransformer
   :members:
   :no-show-inheritance:
```

----


## Model Classes

(QEffWanPipeline)=
### `QEffWanPipeline`

WAN supports two execution architectures:

- `use_unified=True` (default): one unified transformer module.
- `use_unified=False`: separate `transformer_high` and `transformer_low` modules.

First-block-cache is currently supported only for non-unified WAN:

```python
from QEfficient import QEffWanPipeline

pipeline = QEffWanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    use_unified=False,
    enable_first_block_cache=True,
    first_block_cache_downsample_factor=4,
)

output = pipeline(
    prompt="A cat playing in a sunny garden",
    cache_threshold_high=0.1,
    cache_threshold_low=0.065,
)
```

See examples:

- `examples/diffusers/wan/wan_lightning.py`
- `examples/diffusers/wan/wan_lightning_custom.py`
- `examples/diffusers/wan/wan_first_block_cache.py`

```{eval-rst}
.. autoclass:: QEfficient.diffusers.pipelines.wan.pipeline_wan.QEffWanPipeline
   :members:
   :no-show-inheritance:
```

(QEffWanImageToVideoPipeline)=
### `QEffWanImageToVideoPipeline`

```{eval-rst}
.. autoclass:: QEfficient.diffusers.pipelines.wan.pipeline_wan_i2v.QEffWanImageToVideoPipeline
   :members:
   :no-show-inheritance:
```
----

(QEffFluxPipeline)=
### `QEffFluxPipeline`

FLUX supports optional first-block-cache via runtime monkey patching:

```python
from QEfficient import QEffFluxPipeline

pipeline = QEffFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    enable_first_block_cache=True,
    first_block_cache_downsample_factor=4,
)

output = pipeline(
    prompt="A laughing girl",
    cache_threshold=0.1,
)
```

When `enable_first_block_cache=False`, the pipeline follows baseline behavior and ignores `cache_threshold`.

See examples:

- `examples/diffusers/flux/flux_1_schnell.py`
- `examples/diffusers/flux/flux_1_shnell_custom.py`
- `examples/diffusers/flux/flux_1_schnell_first_block_cache.py`

```{eval-rst}
.. autoclass:: QEfficient.diffusers.pipelines.flux.pipeline_flux.QEffFluxPipeline
   :members:
   :no-show-inheritance:
```

----
