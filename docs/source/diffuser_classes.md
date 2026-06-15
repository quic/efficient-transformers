# Diffuser Classes

## Pipeline API

### `QEffTextEncoder` { #QEffTextEncoder }

::: QEfficient.diffusers.pipelines.pipeline_module.QEffTextEncoder
    options:
      members:
        - export
        - compile
        - get_model_config
        - get_onnx_params

---

### `QEffUNet` { #QEffUNet }

::: QEfficient.diffusers.pipelines.pipeline_module.QEffUNet
    options:
      members:
        - export
        - compile
        - get_model_config

---

### `QEffVAE` { #QEffVAE }

::: QEfficient.diffusers.pipelines.pipeline_module.QEffVAE
    options:
      members:
        - export
        - compile
        - get_model_config
        - get_onnx_params
        - get_video_onnx_params

---

### `QEffFluxTransformerModel` { #QEffFluxTransformerModel }

::: QEfficient.diffusers.pipelines.pipeline_module.QEffFluxTransformerModel
    options:
      members:
        - export
        - compile
        - get_model_config
        - get_onnx_params

---

### `QEffWanUnifiedTransformer` { #QEffWanUnifiedTransformer }

::: QEfficient.diffusers.pipelines.pipeline_module.QEffWanUnifiedTransformer
    options:
      members:
        - export
        - compile
        - get_model_config
        - get_onnx_params

---

## Model Classes

### `QEffWanPipeline` { #QEffWanPipeline }

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

::: QEfficient.diffusers.pipelines.wan.pipeline_wan.QEffWanPipeline
    options:
      members:
        - from_pretrained
        - compile
        - generate

---

### `QEffWanImageToVideoPipeline` { #QEffWanImageToVideoPipeline }

::: QEfficient.diffusers.pipelines.wan.pipeline_wan_i2v.QEffWanImageToVideoPipeline
    options:
      members:
        - from_pretrained
        - compile
        - generate

---

### `QEffFluxPipeline` { #QEffFluxPipeline }

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

::: QEfficient.diffusers.pipelines.flux.pipeline_flux.QEffFluxPipeline
    options:
      members:
        - from_pretrained
        - compile
        - generate
