import copy
import functools
import json
import tempfile
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM

MODEL_PATH = Path(
    "/home/huggingface_hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"
)

TS = 1
enable_mla = True
mla_absorption = {"cache_compressed": True, "absorption": True, "online": False}
prefill_seq_len = 1
ctx_len = 128
qaic_config = {"mla_absorption": mla_absorption, "num_kv_heads_repeat": TS}  # No blocking with kv head replication

EXPORT_START = 1
EXPORT_END = 2
LAYERWISE_MODE = "single_qpc"


def _ensure_pretrained_window_attrs():
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_start"):
        transformers.modeling_utils.PreTrainedModel._start = 0
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_end"):
        transformers.modeling_utils.PreTrainedModel._end = 0


def _null_outside_window_layers(model):
    start = int(getattr(transformers.modeling_utils.PreTrainedModel, "_start", 0))
    end = int(getattr(transformers.modeling_utils.PreTrainedModel, "_end", 0))

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return

    print(f"{start} to {end}")
    for idx, _ in enumerate(layers):
        if idx < start or idx >= end:
            layers[idx] = None


def _install_window_patch(model_cls):
    if getattr(model_cls, "_window_patch_installed", False):
        return

    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _null_outside_window_layers(self)

    model_cls.__init__ = patched_init
    model_cls._window_patch_installed = True


def load_text_only_kimi(model_path: Path, num_hidden_layers: int):
    _ensure_pretrained_window_attrs()
    kimi_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

    # Kimi K2.5 is multimodal, so we load only the text stack config.
    text_config = copy.deepcopy(kimi_config.text_config)

    deepseek_cls = get_class_from_dynamic_module("modeling_deepseek.DeepseekV3ForCausalLM", str(model_path))
    _install_window_patch(deepseek_cls)

    checkpoint_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = checkpoint_index["weight_map"]

    allowed_prefixes = [
        "language_model.model.embed_tokens.",
        "language_model.model.norm.",
        "language_model.lm_head.",
    ]
    layer_start = int(getattr(transformers.modeling_utils.PreTrainedModel, "_start", 0))
    layer_end = int(getattr(transformers.modeling_utils.PreTrainedModel, "_end", 0))
    allowed_prefixes.extend(
        [f"language_model.model.layers.{layer_idx}." for layer_idx in range(layer_start, layer_end)]
    )

    required_shards = sorted(
        {
            shard_name
            for checkpoint_key, shard_name in weight_map.items()
            if any(checkpoint_key.startswith(prefix) for prefix in allowed_prefixes)
        }
    )
    filtered_weight_map = {
        checkpoint_key: shard_name
        for checkpoint_key, shard_name in weight_map.items()
        if any(checkpoint_key.startswith(prefix) for prefix in allowed_prefixes)
    }
    if not filtered_weight_map:
        raise RuntimeError("No text-only weights were selected from the Kimi K2.5 checkpoint.")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_model_path = Path(tmpdir)
        (temp_model_path / "config.json").write_text(text_config.to_json_string(use_diff=False))
        (temp_model_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum((model_path / shard_name).stat().st_size for shard_name in required_shards)
                    },
                    "weight_map": filtered_weight_map,
                }
            )
        )
        for shard_name in required_shards:
            (temp_model_path / shard_name).symlink_to(model_path / shard_name)

        # We are loading a task checkpoint into the base text model, so disable the
        # base/task prefix heuristic and let `key_mapping` strip `language_model.`.
        original_base_model_prefix = deepseek_cls.base_model_prefix
        deepseek_cls.base_model_prefix = ""
        try:
            model, loading_info = deepseek_cls.from_pretrained(
                str(temp_model_path),
                config=text_config,
                local_files_only=True,
                key_mapping={r"^language_model\.": ""},
                output_loading_info=True,
            )
        finally:
            deepseek_cls.base_model_prefix = original_base_model_prefix

    unexpected_keys = loading_info["unexpected_keys"]
    missing_keys = loading_info["missing_keys"]
    mismatched_keys = loading_info["mismatched_keys"]
    if unexpected_keys or missing_keys or mismatched_keys:
        raise RuntimeError(
            "Failed to load the text-only Kimi K2.5 checkpoint slice cleanly. "
            f"missing={missing_keys}, unexpected={unexpected_keys}, mismatched={mismatched_keys}"
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    return model, tokenizer


def _build_layer_windows(total_layers: int, start: int, end: int):
    if not (0 <= start < end <= total_layers):
        raise ValueError(
            f"Invalid export window start={start}, end={end} for total_layers={total_layers}. "
            "Expected: 0 <= start < end <= total_layers."
        )

    windows = []
    if start > 0:
        windows.append((0, start))

    step = end - start
    current = start
    while current < total_layers:
        current_end = min(current + step, total_layers)
        windows.append((current, current_end))
        current = current_end

    return windows


def _resolve_export_root(onnx_path: Path) -> Path:
    parts = list(onnx_path.parts)
    if "onnx_layerwise_tmp" in parts:
        marker_idx = parts.index("onnx_layerwise_tmp")
        return Path(*parts[:marker_idx])
    return onnx_path.parent


def main():
    _ensure_pretrained_window_attrs()
    text_config = AutoConfig.from_pretrained(str(MODEL_PATH), trust_remote_code=True).text_config
    total_layers = 3 # getattr(text_config, "num_hidden_layers", None)
    if total_layers is None:
        raise ValueError("Could not resolve `num_hidden_layers` from text_config.")
    windows = _build_layer_windows(total_layers=total_layers, start=EXPORT_START, end=EXPORT_END)
    first_onnx_path = None
    for start, end in windows:
        transformers.modeling_utils.PreTrainedModel._start = start
        transformers.modeling_utils.PreTrainedModel._end = end
        transformers.modeling_utils.PreTrainedModel._total_layers = total_layers
        QEfficient.transformers.models.deepseek_v3.modeling_deepseek.QEffDeepseekV3Model._start = start
        QEfficient.transformers.models.deepseek_v3.modeling_deepseek.QEffDeepseekV3Model._end = end
        QEfficient.transformers.models.deepseek_v3.modeling_deepseek.QEffDeepseekV3Model._total_layers = total_layers
        QEfficient.base.modeling_qeff.QEFFBaseModel._start = start
        QEfficient.base.modeling_qeff.QEFFBaseModel._end = end
        QEfficient.base.modeling_qeff.QEFFBaseModel._total_layers = total_layers
        model, tokenizer = load_text_only_kimi(MODEL_PATH, num_hidden_layers=end - start)
        qeff_model = QEFFAutoModelForCausalLM(
            model, num_kv_heads_repeat=1, qaic_config=qaic_config, torch_dtype=torch.float16
        )
        onnx_path = qeff_model.compile(
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            mxfp6_matmul=True,
            mxint8_kv_cache=False,
            num_devices=TS,
            num_cores=16,
            qaic_config=qaic_config,
            use_onnx_subfunctions=True,
        )
        if first_onnx_path is None:
            first_onnx_path = Path(onnx_path)

    if first_onnx_path is None:
        raise RuntimeError("No ONNX path produced during compilation.")
    export_root = _resolve_export_root(first_onnx_path)

    if LAYERWISE_MODE == "multiple_qpc":
        QEfficient.utils.compile_layerwise(str(export_root))
        QEfficient.utils.inference_pipeline(str(export_root))
    else:
        QEfficient.utils.layerwise_pipeline(str(export_root))


if __name__ == "__main__":
    main()
