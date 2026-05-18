import argparse
import copy
import functools
import inspect
import json
import tempfile
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM
from compile_full_model import run_full_model_compile

def _build_qaic_config(
    mla_absorption_cfg: dict,
    enable_blocking: bool,
    blocking_mode: str,
    num_kv_heads_repeat: int,
    num_kv_blocks: int,
    head_block_size: int,
    par_num_split: Optional[int],
):
    return {
        "mla_absorption": mla_absorption_cfg,
        "enable_blocking": enable_blocking,
        "blocking_mode": blocking_mode,
        "num_kv_heads_repeat": num_kv_heads_repeat,
        "num_kv_blocks": num_kv_blocks,
        "head_block_size": head_block_size,
        "par_num_split": par_num_split,
    }


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


def _build_layer_windows(total_layers: int, window_size: int):
    if total_layers <= 0:
        raise ValueError(f"Invalid total_layers={total_layers}. Expected: total_layers > 0.")
    if window_size <= 0:
        raise ValueError(f"Invalid window_size={window_size}. Expected: window_size > 0.")

    windows = []
    end = total_layers
    while end > 0:
        start = max(0, end - window_size)
        windows.append((start, end))
        end = start

    return windows


def _resolve_export_root(onnx_path: Path) -> Path:
    parts = list(onnx_path.parts)
    if "onnx_layerwise_tmp" in parts:
        marker_idx = parts.index("onnx_layerwise_tmp")
        return Path(*parts[:marker_idx])
    return onnx_path.parent


def _parse_args():
    parser = argparse.ArgumentParser(description="Compile Kimi text-only model layer windows with QEfficient.")
    parser.add_argument("--model_path", dest="model_path", type=Path, required=True, help="Path to the downloaded Kimi model")
    parser.add_argument("--aic_hw_version", dest="aic_hw_version", type=str, default="ai100")
    parser.add_argument("--window_size", dest="window_size", type=int, default=1)
    parser.add_argument("--layerwise_mode", dest="layerwise_mode", type=str, default="single_qpc")
    parser.add_argument("--total_layers", dest="total_layers", type=int, default=None)
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices for compile stages.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for specialization config.")
    parser.add_argument("--seq_len", type=int, default=1, help="Prefill/compile sequence length.")
    parser.add_argument("--ctx_len", type=int, default=128, help="Context length for compile stages.")
    parser.add_argument("--num_cores", type=int, default=16, help="Number of accelerator cores.")
    parser.add_argument("--mxfp6", dest="mxfp6", action="store_true", default=True, help="Enable mxfp6 compile flag (default: True)")
    parser.add_argument("--no-mxfp6", dest="mxfp6", action="store_false", help="Disable mxfp6 compile flag")
    parser.add_argument(
        "--enable_blocking",
        dest="enable_blocking",
        action="store_true",
        default=False,
        help="Enable or disable blocking.",
    )
    parser.add_argument("--blocking_mode", dest="blocking_mode", type=str, default="kv", help="Blocking mode.")
    parser.add_argument(
        "--num_kv_heads_repeat",
        dest="num_kv_heads_repeat",
        type=int,
        default=1,
        help="Number of KV heads to repeat.",
    )
    parser.add_argument(
        "--num_kv_blocks",
        dest="num_kv_blocks",
        type=int,
        default=8,
        help="Number of KV blocks.",
    )
    parser.add_argument(
        "--head_block_size",
        dest="head_block_size",
        type=int,
        default=None,
        help="Head block size.",
    )
    parser.add_argument(
        "--par_num_split",
        dest="par_num_split",
        type=int,
        default=None,
        help="T-dim split per KV block for optimized MLA KV blocking.",
    )
    parser.add_argument(
        "--absorption",
        dest="absorption",
        action="store_true",
        default=False,
        help="Enable or disable MLA absorption.",
    )
    parser.add_argument(
        "--online",
        dest="online",
        action="store_true",
        default=False,
        help="Enable or disable MLA online mode.",
    )
    parser.add_argument(
        "--mxint8_kv_cache",
        dest="mxint8_kv_cache",
        action="store_true",
        default=True,
        help="Enable mxint8 kv-cache compile flag (default: True)",
    )
    parser.add_argument(
        "--no-mxint8_kv_cache",
        dest="mxint8_kv_cache",
        action="store_false",
        help="Disable mxint8 kv-cache compile flag",
    )
    parser.add_argument(
        "--prefill_only",
        dest="prefill_only",
        action="store_true",
        default=False,
        help="Enable or disable MLA online mode.",
    )
    return parser.parse_args()


def main():
    _ensure_pretrained_window_attrs()
    args = _parse_args()
    model_path = args.model_path
    enable_blocking = args.enable_blocking
    blocking_mode = args.blocking_mode
    num_kv_heads_repeat = args.num_kv_heads_repeat
    num_kv_blocks = args.num_kv_blocks
    head_block_size = args.head_block_size
    par_num_split = args.par_num_split
    
    mla_absorption_cfg = {
        "cache_compressed": True,
        "absorption": args.absorption,
        "online": args.online,
    }
    text_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True).text_config
    resolved_total_layers = args.total_layers
    if resolved_total_layers is None:
        resolved_total_layers = getattr(text_config, "num_hidden_layers", None)
    if resolved_total_layers is None:
        raise ValueError("Could not resolve `num_hidden_layers` from text_config.")

    window_size = args.window_size
    layerwise_mode = args.layerwise_mode
    total_layers = resolved_total_layers
    
    windows = _build_layer_windows(total_layers=total_layers, window_size=window_size)
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
        model, tokenizer = load_text_only_kimi(model_path, num_hidden_layers=end - start)
        model.config.num_hidden_layers = total_layers
        
        qaic_config = _build_qaic_config(
            mla_absorption_cfg=mla_absorption_cfg,
            enable_blocking=enable_blocking,
            blocking_mode=blocking_mode,
            num_kv_heads_repeat=num_kv_heads_repeat,
            num_kv_blocks=num_kv_blocks,
            head_block_size=head_block_size,
            par_num_split=par_num_split,
        )
        qeff_model = QEFFAutoModelForCausalLM(
            model, num_kv_heads_repeat=num_kv_heads_repeat, qaic_config=qaic_config, torch_dtype=torch.float16
        )
        onnx_path = qeff_model.compile(
            prefill_seq_len=args.seq_len,
            ctx_len=args.ctx_len,
            mxfp6_matmul=args.mxfp6,
            mxint8_kv_cache=args.mxint8_kv_cache,
            num_devices=args.num_devices,
            num_cores=args.num_cores,
            qaic_config=qaic_config,
            prefill_only=args.prefill_only,
            use_onnx_subfunctions=True,
        )
        if first_onnx_path is None:
            first_onnx_path = Path(onnx_path)

    if first_onnx_path is None:
        raise RuntimeError("No ONNX path produced during compilation.")
    export_root = _resolve_export_root(first_onnx_path)

    if layerwise_mode == "multiple_qpc":
        QEfficient.utils.compile_layerwise(str(export_root))
        QEfficient.utils.inference_pipeline(str(export_root))
    else:
        final_onnx_path = QEfficient.utils.layerwise_pipeline(str(export_root))
        if final_onnx_path is None:
            raise RuntimeError("QEfficient.utils.layerwise_pipeline returned an empty ONNX path.")
        compile_num_layers = total_layers
        compile_kwargs = {
            "onnx_path": Path(final_onnx_path),
            "num_devices": args.num_devices,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "ctx_len": args.ctx_len,
            "num_cores": args.num_cores,
            "num_layers": compile_num_layers,
            "mxfp6": args.mxfp6,
            "mxint8_kv_cache": args.mxint8_kv_cache,
            "aic_version": args.aic_hw_version,
        }
        optional_compile_kwargs = {
            "qaic_config": qaic_config,
            "enable_blocking": enable_blocking,
            "blocking_mode": blocking_mode,
            "num_kv_heads_repeat": num_kv_heads_repeat,
            "num_kv_blocks": num_kv_blocks,
            "head_block_size": head_block_size,
            "par_num_split": par_num_split,
            "cache_compressed": True,
            "absorption": args.absorption,
            "online": args.online,
        }
        compile_signature = inspect.signature(run_full_model_compile)
        for key, value in optional_compile_kwargs.items():
            if key in compile_signature.parameters:
                compile_kwargs[key] = value

        run_full_model_compile(**compile_kwargs)


if __name__ == "__main__":
    main()
