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
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import import_utils as hf_import_utils

KIMI_K25_MODEL_NAME = "moonshotai/Kimi-K2.5"
DEFAULT_MODEL_PATH = Path(
    "/home/huggingface_hub/models--moonshotai--Kimi-K2.5/snapshots/4d01dfe0332d63057c186e0b262165819efb6611"
)
NUM_VISION_LAYERS = 2
NUM_TEXT_LAYERS = 2
LOADED_EXPERT_IDS = (0, 1, 2, 3)
NUM_EXPERTS_PER_TOKEN = 2

EXPERT_KEY_PATTERN = re.compile(r"^(language_model\.model\.layers\.\d+\.mlp\.experts\.)(\d+)(\..+)$")


def is_kimi_k25(model_name: str) -> bool:
    return model_name == KIMI_K25_MODEL_NAME


def set_deterministic(seed: int):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_model_path(model_name: str = KIMI_K25_MODEL_NAME) -> Path:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return Path(snapshot_download(repo_id=model_name, cache_dir=os.environ.get("HF_HUB_CACHE")))


def ensure_torch_fx_import_compatibility():
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


def patch_kimi_tie_weights_compat(kimi_cls):
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


def patch_deepseek_init_weights_compat(kimi_cls):
    module_prefix, _ = kimi_cls.__module__.rsplit(".", maxsplit=1)
    deepseek_module = sys.modules.get(f"{module_prefix}.modeling_deepseek")
    if deepseek_module is None or not hasattr(deepseek_module, "DeepseekV3PreTrainedModel"):
        return

    deepseek_cls = deepseek_module.DeepseekV3PreTrainedModel
    if (
        getattr(deepseek_cls, "_qeff_kimi_k25_init_weights_patched", False)
        or getattr(deepseek_cls, "_qeff_test_init_weights_patched", False)
        or getattr(deepseek_cls, "_qeff_t55_init_weights_patched", False)
    ):
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
    deepseek_cls._qeff_kimi_k25_init_weights_patched = True
    deepseek_cls._qeff_test_init_weights_patched = True
    deepseek_cls._qeff_t55_init_weights_patched = True


def load_kimi_k25_class(model_path_or_name):
    ensure_torch_fx_import_compatibility()
    kimi_cls = get_class_from_dynamic_module(
        "modeling_kimi_k25.KimiK25ForConditionalGeneration",
        str(model_path_or_name),
    )
    patch_kimi_tie_weights_compat(kimi_cls)
    patch_deepseek_init_weights_compat(kimi_cls)
    return kimi_cls


def patch_kimi_k25_remote_code_compat(config):
    return load_kimi_k25_class(config._name_or_path)


def prepare_config(model_path: Path):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

    config._attn_implementation = "eager"
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = "eager"
    if hasattr(config, "vision_config"):
        config.vision_config._attn_implementation = "eager"
    return config


def get_kimi_k25_test_config(model_name: str, model_config_dict):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config._attn_implementation = "eager"
    config.torch_dtype = torch.float32
    config.dtype = torch.float32
    additional_params = model_config_dict[model_name]["additional_params"]

    for attr, value in additional_params["text_config"].items():
        setattr(config.text_config, attr, value)
    config.text_config._attn_implementation = "eager"
    config.text_config.torch_dtype = torch.float32
    config.text_config.dtype = torch.float32

    for attr, value in additional_params["vision_config"].items():
        setattr(config.vision_config, attr, value)
    config.vision_config._attn_implementation = "eager"
    config.vision_config.torch_dtype = torch.float32
    config.vision_config.dtype = torch.float32

    patch_kimi_k25_remote_code_compat(config)
    return config


def load_kimi_k25_model_from_config(config):
    kimi_cls = patch_kimi_k25_remote_code_compat(config)
    model = kimi_cls._from_config(config)
    torch_dtype = getattr(model.config, "torch_dtype", None)
    if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        model = model.to(torch.float32)
    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(KIMI_K25_MODEL_NAME, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(KIMI_K25_MODEL_NAME, trust_remote_code=True)
    return model, tokenizer, processor


def get_kimi_k25_num_image_tokens(config, grid_thws):
    merge_height, merge_width = config.vision_config.merge_kernel_size
    return int(grid_thws[0, 1].item() // merge_height) * int(grid_thws[0, 2].item() // merge_width)


def parse_expert_ids(value: str):
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


def allowed_prefixes(num_vision_layers: int, num_text_layers: int):
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


def build_layer_subset_config(config, num_vision_layers, num_text_layers, loaded_expert_ids, num_experts_per_tok):
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


def materialize_subset_checkpoint(
    model_path: Path, temp_model_path: Path, weight_map, allowed_weight_prefixes, loaded_expert_ids
):
    expert_index_map = {expert_id: remapped_idx for remapped_idx, expert_id in enumerate(loaded_expert_ids)}
    shard_to_entries = {}
    for checkpoint_key, source_shard_name in weight_map.items():
        if not checkpoint_key.startswith(tuple(allowed_weight_prefixes)):
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


def load_layer_subset_model(
    *,
    model_path: Path,
    kimi_cls,
    config,
    num_vision_layers: int,
    num_text_layers: int,
    loaded_expert_ids,
    num_experts_per_tok: int,
    dtype,
):
    checkpoint_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = checkpoint_index["weight_map"]
    stripped_config, loaded_expert_ids = build_layer_subset_config(
        config,
        num_vision_layers=num_vision_layers,
        num_text_layers=num_text_layers,
        loaded_expert_ids=loaded_expert_ids,
        num_experts_per_tok=num_experts_per_tok,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_model_path = Path(tmpdir)
        filtered_weight_map, subset_shards = materialize_subset_checkpoint(
            model_path=model_path,
            temp_model_path=temp_model_path,
            weight_map=weight_map,
            allowed_weight_prefixes=allowed_prefixes(num_vision_layers, num_text_layers),
            loaded_expert_ids=loaded_expert_ids,
        )
        (temp_model_path / "config.json").write_text(stripped_config.to_json_string(use_diff=False))
        (temp_model_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum((temp_model_path / shard_name).stat().st_size for shard_name in subset_shards)
                    },
                    "weight_map": filtered_weight_map,
                }
            )
        )

        model_kwargs = {
            "config": stripped_config,
            "trust_remote_code": True,
            "attn_implementation": "eager",
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
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    print(f"Loaded model: {type(model).__name__}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Processor type: {type(processor).__name__}")

    return model, tokenizer, processor


def load_kimi_k25_layer_subset_model(
    *,
    model_path: Path | None = None,
    num_vision_layers: int = NUM_VISION_LAYERS,
    num_text_layers: int = NUM_TEXT_LAYERS,
    loaded_expert_ids=LOADED_EXPERT_IDS,
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOKEN,
    dtype=torch.float32,
    seed: int = 1234,
):
    set_deterministic(seed)
    ensure_torch_fx_import_compatibility()
    resolved_model_path = Path(model_path) if model_path is not None else resolve_model_path()
    config = prepare_config(resolved_model_path)
    kimi_cls = load_kimi_k25_class(resolved_model_path)

    model, tokenizer, processor = load_layer_subset_model(
        model_path=resolved_model_path,
        kimi_cls=kimi_cls,
        config=config,
        num_vision_layers=num_vision_layers,
        num_text_layers=num_text_layers,
        loaded_expert_ids=loaded_expert_ids,
        num_experts_per_tok=num_experts_per_tok,
        dtype=dtype,
    )
    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    return model.eval().to("cpu"), tokenizer, processor


@torch.no_grad()
def run_kimi_k25_hf_model_on_pytorch(model, processor, inputs, max_gen_len):
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    grid_thws = inputs["grid_thws"]
    new_tokens = []

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None and hasattr(model.config, "text_config"):
        eos_token_id = getattr(model.config.text_config, "eos_token_id", None)

    for _ in range(max_gen_len):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        new_tokens.append(next_token)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    output_tokens = torch.cat(new_tokens, dim=1).squeeze(0)
    py_output = processor.tokenizer.decode(output_tokens.tolist()).strip()
    print("Original HF Model Outputs (Torch CPU):")
    print("Completion:", repr(py_output))
    return output_tokens


@torch.no_grad()
def run_kimi_k25_hf_model_on_pytorch_CB(model, processor, images, queries, max_gen_len):
    generated_tokens = []

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None and hasattr(model.config, "text_config"):
        eos_token_id = getattr(model.config.text_config, "eos_token_id", None)

    for idx, (image, query) in enumerate(zip(images, queries)):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": image},
                    {"type": "text", "text": query},
                ],
            },
        ]
        inputs = processor(messages=conversation, add_generation_prompt=True, tokenize=False, return_tensors="pt")
        generated_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]
        grid_thws = inputs["grid_thws"]
        new_tokens = []

        for _ in range(max_gen_len):
            outputs = model(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                use_cache=False,
                return_dict=True,
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            new_tokens.append(next_token)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=1,
            )

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        output_tokens = torch.cat(new_tokens, dim=1).squeeze(0)
        py_output = processor.tokenizer.decode(output_tokens.tolist()).strip()
        print(f"Original HF Model Outputs (Torch CPU) for prompt {idx}:")
        print("Query:", repr(query))
        print("Completion:", repr(py_output))
        generated_tokens.append(output_tokens.numpy())

    return generated_tokens
