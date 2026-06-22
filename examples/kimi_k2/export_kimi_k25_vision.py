# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import copy
import inspect
import json
import re
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils as hf_import_utils

from QEfficient import QEFFAutoModelForImageTextToText


MODEL_PATH = Path(
    "/home/huggingface_hub/models--moonshotai--Kimi-K2.5/snapshots/4d01dfe0332d63057c186e0b262165819efb6611"
)
NUM_VISION_LAYERS = 2
NUM_TEXT_LAYERS = 2
LOADED_EXPERT_IDS = (0, 1, 2, 3)
NUM_EXPERTS_PER_TOKEN = 2

EXPERT_KEY_PATTERN = re.compile(r"^(language_model\.model\.layers\.\d+\.mlp\.experts\.)(\d+)(\..+)$")


def _ensure_torch_fx_import_compatibility():
    """Backfill `is_torch_fx_available` for remote model code expecting older Transformers APIs."""
    if hasattr(hf_import_utils, "is_torch_fx_available"):
        return

    def _is_torch_fx_available() -> bool:
        if not hf_import_utils.is_torch_available():
            return False
        try:
            import torch.fx  # noqa: F401

            return True
        except Exception:
            return False

    hf_import_utils.is_torch_fx_available = _is_torch_fx_available


def _patch_kimi_tie_weights_compat(kimi_cls):
    tie_signature = inspect.signature(kimi_cls.tie_weights)
    if tuple(tie_signature.parameters) != ("self",):
        return

    def _tie_weights_compat(self, missing_keys=None, recompute_mapping=True):
        lm_tie_weights = getattr(self.language_model, "tie_weights")
        try:
            return lm_tie_weights(missing_keys=missing_keys, recompute_mapping=recompute_mapping)
        except TypeError:
            return lm_tie_weights()

    kimi_cls.tie_weights = _tie_weights_compat


def _patch_deepseek_init_weights_compat(kimi_cls):
    module_prefix, _ = kimi_cls.__module__.rsplit(".", maxsplit=1)
    deepseek_module_name = f"{module_prefix}.modeling_deepseek"
    deepseek_module = sys.modules.get(deepseek_module_name)
    if deepseek_module is None or not hasattr(deepseek_module, "DeepseekV3PreTrainedModel"):
        return

    deepseek_cls = deepseek_module.DeepseekV3PreTrainedModel
    if getattr(deepseek_cls, "_qeff_t55_init_weights_patched", False):
        return

    def _init_weights_compat(self, module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.normal_(mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    deepseek_cls._init_weights = _init_weights_compat
    deepseek_cls._qeff_t55_init_weights_patched = True


def _prepare_config(model_path: Path):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

    # Avoid FA2 dispatch checks in Transformers 5.5.x for this remote code.
    config._attn_implementation = "eager"
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = "eager"
    if hasattr(config, "vision_config"):
        config.vision_config._attn_implementation = "eager"
    return config


def _parse_expert_ids(value: str):
    expert_ids = tuple(int(expert_id) for expert_id in value.split(",") if expert_id.strip())
    if not expert_ids:
        raise argparse.ArgumentTypeError("At least one expert id must be provided.")
    return expert_ids


def _validate_layer_count(name, requested_count, available_count):
    if requested_count < 1:
        raise ValueError(f"{name} must be >= 1, got {requested_count}.")
    if requested_count > available_count:
        raise ValueError(f"{name}={requested_count} exceeds available layers={available_count}.")


def _validate_expert_subset(loaded_expert_ids, num_experts_per_tok, total_experts):
    expert_ids = tuple(loaded_expert_ids)
    if len(expert_ids) != 4:
        raise ValueError(f"Expected exactly 4 routed experts, got {expert_ids!r}.")
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError(f"Expert ids must be unique, got {expert_ids!r}.")
    invalid_ids = [expert_id for expert_id in expert_ids if expert_id < 0 or expert_id >= total_experts]
    if invalid_ids:
        raise ValueError(f"Expert ids {invalid_ids!r} are outside the valid range [0, {total_experts - 1}].")
    if num_experts_per_tok > len(expert_ids):
        raise ValueError(f"num_experts_per_tok={num_experts_per_tok} cannot exceed {len(expert_ids)} loaded experts.")
    return expert_ids


def _remap_checkpoint_key(checkpoint_key, expert_index_map):
    match = EXPERT_KEY_PATTERN.match(checkpoint_key)
    if not match:
        return checkpoint_key

    original_expert_idx = int(match.group(2))
    remapped_expert_idx = expert_index_map.get(original_expert_idx)
    if remapped_expert_idx is None:
        return None
    return f"{match.group(1)}{remapped_expert_idx}{match.group(3)}"


def _is_routed_gate_weight(checkpoint_key):
    return checkpoint_key.endswith(".mlp.gate.weight")


def _is_routed_gate_bias(checkpoint_key):
    return checkpoint_key.endswith(".mlp.gate.e_score_correction_bias")


def _allowed_prefixes(num_vision_layers: int, num_text_layers: int):
    prefixes = [
        "vision_tower.patch_embed.",
        "vision_tower.encoder.final_layernorm.",
        "mm_projector.",
        "language_model.model.embed_tokens.",
        "language_model.model.norm.",
        "language_model.lm_head.",
    ]
    prefixes.extend(f"vision_tower.encoder.blocks.{layer_idx}." for layer_idx in range(num_vision_layers))
    prefixes.extend(f"language_model.model.layers.{layer_idx}." for layer_idx in range(num_text_layers))
    return prefixes


def _build_layer_subset_config(config, num_vision_layers, num_text_layers, loaded_expert_ids, num_experts_per_tok):
    stripped_config = copy.deepcopy(config)
    text_config = stripped_config.text_config
    vision_config = stripped_config.vision_config

    _validate_layer_count("num_text_layers", num_text_layers, text_config.num_hidden_layers)
    _validate_layer_count("num_vision_layers", num_vision_layers, vision_config.vt_num_hidden_layers)

    text_config.num_hidden_layers = num_text_layers
    vision_config.vt_num_hidden_layers = num_vision_layers

    loaded_expert_ids = _validate_expert_subset(
        loaded_expert_ids,
        num_experts_per_tok,
        text_config.n_routed_experts,
    )
    text_config.n_routed_experts = len(loaded_expert_ids)
    text_config.num_experts_per_tok = num_experts_per_tok
    text_config.n_group = 1
    text_config.topk_group = 1
    return stripped_config, loaded_expert_ids


def _materialize_subset_checkpoint(model_path: Path, temp_model_path: Path, weight_map, allowed_prefixes, loaded_expert_ids):
    expert_index_map = {expert_id: remapped_idx for remapped_idx, expert_id in enumerate(loaded_expert_ids)}
    shard_to_entries = {}
    for checkpoint_key, source_shard_name in weight_map.items():
        if not checkpoint_key.startswith(tuple(allowed_prefixes)):
            continue

        remapped_key = _remap_checkpoint_key(checkpoint_key, expert_index_map)
        if remapped_key is None:
            continue
        shard_to_entries.setdefault(source_shard_name, []).append((checkpoint_key, remapped_key))

    filtered_weight_map = {}
    subset_shards = []
    for shard_idx, (source_shard_name, shard_entries) in enumerate(sorted(shard_to_entries.items())):
        tensors = {}
        with safe_open(model_path / source_shard_name, framework="pt", device="cpu") as shard_reader:
            for checkpoint_key, remapped_key in shard_entries:
                tensor = shard_reader.get_tensor(checkpoint_key)
                if _is_routed_gate_weight(checkpoint_key):
                    tensor = tensor[list(loaded_expert_ids), :].contiguous()
                elif _is_routed_gate_bias(checkpoint_key):
                    tensor = tensor[list(loaded_expert_ids)].contiguous()
                tensors[remapped_key] = tensor

        subset_shard_name = f"model-subset-{shard_idx:05d}.safetensors"
        save_file(tensors, str(temp_model_path / subset_shard_name))
        subset_shards.append(subset_shard_name)
        filtered_weight_map.update({remapped_key: subset_shard_name for _, remapped_key in shard_entries})

    return filtered_weight_map, subset_shards


def _load_layer_subset_model(
    *,
    model_path: Path,
    kimi_cls,
    config,
    num_vision_layers: int,
    num_text_layers: int,
    loaded_expert_ids,
    num_experts_per_tok: int,
    local_files_only: bool,
    dtype,
):
    checkpoint_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = checkpoint_index["weight_map"]
    stripped_config, loaded_expert_ids = _build_layer_subset_config(
        config,
        num_vision_layers=num_vision_layers,
        num_text_layers=num_text_layers,
        loaded_expert_ids=loaded_expert_ids,
        num_experts_per_tok=num_experts_per_tok,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_model_path = Path(tmpdir)
        filtered_weight_map, subset_shards = _materialize_subset_checkpoint(
            model_path=model_path,
            temp_model_path=temp_model_path,
            weight_map=weight_map,
            allowed_prefixes=_allowed_prefixes(num_vision_layers, num_text_layers),
            loaded_expert_ids=loaded_expert_ids,
        )
        (temp_model_path / "config.json").write_text(stripped_config.to_json_string(use_diff=False))
        (temp_model_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": sum((temp_model_path / shard_name).stat().st_size for shard_name in subset_shards)},
                    "weight_map": filtered_weight_map,
                }
            )
        )

        model_kwargs = {
            "config": stripped_config,
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "local_files_only": local_files_only,
            "output_loading_info": True,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        original_base_model_prefix = kimi_cls.base_model_prefix
        kimi_cls.base_model_prefix = ""
        try:
            model, loading_info = kimi_cls.from_pretrained(str(temp_model_path), **model_kwargs)
        finally:
            kimi_cls.base_model_prefix = original_base_model_prefix

    unexpected_keys = loading_info["unexpected_keys"]
    missing_keys = loading_info["missing_keys"]
    mismatched_keys = loading_info["mismatched_keys"]
    if unexpected_keys or missing_keys or mismatched_keys:
        raise RuntimeError(
            "Failed to load the stripped Kimi K2.5 checkpoint slice cleanly. "
            f"missing={missing_keys}, unexpected={unexpected_keys}, mismatched={mismatched_keys}"
        )

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Load Kimi K2.5 with runtime compatibility for transformers==5.5.4.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--local-files-only", action="store_true", help="Only read from local cache/files.")
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Load the full model. By default, the script loads a small layer subset for faster startup.",
    )
    parser.add_argument(
        "--num-vision-layers",
        type=int,
        default=NUM_VISION_LAYERS,
        help=(
            "Load only the first N vision layers. "
            f"Defaults to {NUM_VISION_LAYERS} for faster startup. Ignored when --full-model is set."
        ),
    )
    parser.add_argument(
        "--num-text-layers",
        type=int,
        default=NUM_TEXT_LAYERS,
        help=(
            "Load only the first N text layers. "
            f"Defaults to {NUM_TEXT_LAYERS} for faster startup. Ignored when --full-model is set."
        ),
    )
    parser.add_argument("--expert-ids", type=_parse_expert_ids, default=LOADED_EXPERT_IDS)
    parser.add_argument("--num-experts-per-token", type=int, default=NUM_EXPERTS_PER_TOKEN)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="torch_dtype used by from_pretrained.",
    )
    parser.add_argument("--skip-processor", action="store_true", help="Skip loading processor.")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip loading tokenizer.")
    return parser.parse_args()


def _dtype_from_arg(dtype_arg: str):
    if dtype_arg == "auto":
        return None
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main():
    args = parse_args()
    _ensure_torch_fx_import_compatibility()

    config = _prepare_config(args.model_path)
    kimi_cls = get_class_from_dynamic_module(
        "modeling_kimi_k25.KimiK25ForConditionalGeneration",
        str(args.model_path),
    )
    _patch_kimi_tie_weights_compat(kimi_cls)
    _patch_deepseek_init_weights_compat(kimi_cls)

    kimi_module = sys.modules[kimi_cls.__module__]
    if hasattr(kimi_module, "MoonViT3dEncoder"):
        kimi_module.MoonViT3dEncoder.use_deterministic_attn = False

    model_kwargs = {
        "config": config,
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "local_files_only": args.local_files_only,
    }
    dtype = _dtype_from_arg(args.dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if args.full_model:
        model = kimi_cls.from_pretrained(str(args.model_path), **model_kwargs)
    elif args.num_vision_layers is not None and args.num_text_layers is not None:
        model = _load_layer_subset_model(
            model_path=args.model_path,
            kimi_cls=kimi_cls,
            config=config,
            num_vision_layers=args.num_vision_layers,
            num_text_layers=args.num_text_layers,
            loaded_expert_ids=args.expert_ids,
            num_experts_per_tok=args.num_experts_per_token,
            local_files_only=args.local_files_only,
            dtype=dtype,
        )
        print(
            "Loaded layer subset: "
            f"vision={model.config.vision_config.vt_num_hidden_layers}, "
            f"text={model.config.text_config.num_hidden_layers}, "
            f"experts={model.config.text_config.n_routed_experts}"
        )
    else:
        raise ValueError("Pass both --num-vision-layers and --num-text-layers to load a layer subset.")
    print(f"Loaded model: {type(model).__name__}")

    if not args.skip_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_path), trust_remote_code=True, local_files_only=args.local_files_only
        )
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    if not args.skip_processor:
        processor = AutoProcessor.from_pretrained(
            str(args.model_path), trust_remote_code=True, local_files_only=args.local_files_only
        )
        print(f"Processor type: {type(processor).__name__}")
    
    mla_absorption = {"cache_compressed": True, "absorption": False, "online": False}
    qaic_config = {
        "mla_absorption": mla_absorption
    }  # , "enable_blocking": True, "blocking_mode": "par", "par_num_split": 4, "num_kv_blocks": 8}

    qeff_model = QEFFAutoModelForImageTextToText(model)  # , qaic_config=qaic_config)

    skip_vision = False

    if skip_vision:
        ## TEXT-ONLY MODE ##

        ## STEP 3: Compile Model for Text-Only Execution
        # Set skip_vision=True to bypass image processing
        qeff_model.compile(
            qaic_config=qaic_config,
            prefill_seq_len=32,
            ctx_len=1024,
            num_cores=16,
            num_devices=2,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=False,
            skip_vision=True,  # Skip vision encoder for text-only inference
            mos=1,
            num_patches=2400,  # num_patches
            h=30,  # h
            w=80,  # w
            num_image_tokens=600,  # num_image_tokens
        )
        breakpoint()
        ## STEP 4: Prepare Text-Only Input
        # Create a text-only message without any image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Tell me about yourself."},
                ],
            },
        ]

        ## STEP 5: Process Input with Chat Template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        ## STEP 6: Run Text-Only Inference
        streamer = TextStreamer(tokenizer)
        output = qeff_model.generate(inputs=inputs, device_ids=[0, 1], generation_len=10)

        ## STEP 7: Display Results
        print(output.generated_ids)
        print(tokenizer.batch_decode(output.generated_ids))
        print(output)

    else:
        ## VISION + TEXT MODE ##

        ## STEP 3: Compile Model for Vision+Text Execution
        # Do not set skip_vision (defaults to False) to enable image processing
        qeff_model.compile(
            qaic_config=qaic_config,
            prefill_seq_len=1,
            ctx_len=1024,
            num_cores=16,
            num_devices=2,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=False,
            #skip_vision=True,  # Skip vision encoder for text-only inference
            mos=1,
            num_patches=2400,  # num_patches
            h=30,  # h
            w=80,  # w
            num_image_tokens=600,  # num_image_tokens
        )

        breakpoint()
        ## STEP 4: Prepare Image and Text Input
        # Define the image URL to process
        image_url = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png"
        image = Image.open(BytesIO(requests.get(args.image_url).content)).convert("RGB")

        # Create a message with both image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image},
                    {"type": "text", "text": "Can you describe the image in detail."},
                ],
            },
        ]

        ## STEP 5: Process Input with Chat Template
        """inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        """
        inputs = processor(
            messages=messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors="pt",
        )
        # Convert pixel values to float32 for processing
        inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)

        ## STEP 6: Run Vision+Text Inference
        streamer = TextStreamer(tokenizer)
        output = qeff_model.generate(inputs=inputs, device_ids=[0, 1], generation_len=100)

        ## STEP 7: Display Results
        print(output.generated_ids)
        print(tokenizer.batch_decode(output.generated_ids))
        print(output)



if __name__ == "__main__":
    main()
