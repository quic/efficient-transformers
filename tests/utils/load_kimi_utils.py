# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect
import random
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

KIMI_K25_MODEL_NAME = "moonshotai/Kimi-K2.5"


def is_kimi_k25(model_name: str) -> bool:
    return model_name == KIMI_K25_MODEL_NAME


def set_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


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


def patch_dynamic_cache_compat():
    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def _from_legacy_cache(cls, legacy_cache):
            if legacy_cache is None:
                return cls()
            if isinstance(legacy_cache, cls):
                return legacy_cache
            if isinstance(legacy_cache, Cache):
                return cls(ddp_cache_data=[tuple(layer[:2]) for layer in legacy_cache])

            ddp_cache_data = []
            for layer in legacy_cache:
                if layer is None:
                    continue
                if len(layer) < 2:
                    raise ValueError("Each legacy cache layer must provide key/value tensors.")
                ddp_cache_data.append((layer[0], layer[1]))
            return cls(ddp_cache_data=ddp_cache_data)

        DynamicCache.from_legacy_cache = _from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):

        def _to_legacy_cache(self):
            return tuple((layer[0], layer[1]) for layer in self)

        DynamicCache.to_legacy_cache = _to_legacy_cache

    if not hasattr(DynamicCache, "get_max_length"):

        def _get_max_length(self):
            return None

        DynamicCache.get_max_length = _get_max_length


def load_kimi_k25_class(model_path_or_name):
    kimi_cls = get_class_from_dynamic_module(
        "modeling_kimi_k25.KimiK25ForConditionalGeneration",
        str(model_path_or_name),
    )
    patch_kimi_tie_weights_compat(kimi_cls)
    patch_deepseek_init_weights_compat(kimi_cls)
    return kimi_cls


def get_kimi_k25_test_config(model_name: str, model_config_dict, *, seed: int = 42):
    set_deterministic(seed)
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

    load_kimi_k25_class(config._name_or_path)
    return config


def _attach_fake_gptq_weight(linear: torch.nn.Linear, group_size: int = 16):
    weight = linear.weight.detach().to(torch.float32)
    out_features, in_features = weight.shape
    if in_features % group_size != 0 or (in_features // group_size) % 2 != 0:
        raise ValueError(f"Cannot pack shape {tuple(weight.shape)} with group_size={group_size}.")

    weight_blocks = weight.view(out_features, in_features // group_size, group_size)
    scales = weight_blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).eps) / 7
    quantized = (weight_blocks / scales.unsqueeze(-1)).round().add(8).clamp(0, 15).to(torch.uint8)
    dequantized = (quantized.to(torch.int8) - 8).to(torch.float32) * scales.unsqueeze(-1)
    linear.weight.data.copy_(dequantized.view_as(weight).to(linear.weight.dtype))

    quantized = quantized.view(out_features, in_features)
    qweight = quantized[:, 0::2] | (quantized[:, 1::2] << 4)
    zero_points = torch.full((out_features, in_features // group_size), 8, dtype=torch.uint8)
    qzeros = zero_points[:, 0::2] | (zero_points[:, 1::2] << 4)

    linear.bits = 4
    linear.group_size = group_size
    linear.act_order = False
    linear.qweight = torch.nn.Parameter(qweight, requires_grad=False)
    linear.scales = torch.nn.Parameter(scales, requires_grad=False)
    linear.qzeros = torch.nn.Parameter(qzeros, requires_grad=False)
    linear.g_idx = torch.nn.Parameter(torch.arange(in_features) // group_size, requires_grad=False)


def _simulate_kimi_k25_quantized_experts(model):
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV3MoE":
            continue
        for expert in module.experts:
            _attach_fake_gptq_weight(expert.gate_proj)
            _attach_fake_gptq_weight(expert.up_proj)
            _attach_fake_gptq_weight(expert.down_proj)


def load_kimi_k25_model_from_config(config, *, seed: int = 42):
    kimi_cls = load_kimi_k25_class(config._name_or_path)
    set_deterministic(seed)
    model = kimi_cls._from_config(config)
    torch_dtype = getattr(model.config, "torch_dtype", None)
    if torch_dtype == torch.bfloat16 or torch_dtype == torch.float16:
        model = model.to(torch.float32)
    _simulate_kimi_k25_quantized_experts(model)
    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    # Random logits can change argmax after QAIC FP16 conversion. A zero language head keeps
    # token parity deterministic while the test still exercises the complete VLM graph.
    model.language_model.lm_head.weight.data.zero_()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(KIMI_K25_MODEL_NAME, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(KIMI_K25_MODEL_NAME, trust_remote_code=True)
    return model, tokenizer, processor


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
