# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import random

import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from QEfficient.base.pytorch_transforms import CustomOpsTransform, KVCacheTransform, ModuleMapping
from QEfficient.utils.logging_utils import logger


def get_intermediate_outputs(name, saver_dict):
    def hook(model, input, output):
        saver_dict[name] = output
    return hook


@torch.no_grad()
def get_all_kv_cache_transform_intermediate_outputs(llama_model, inputs):
    model_intermediates = dict()
    model_handles = []

    for name, module in llama_model.named_modules():
        if isinstance(module, tuple(KVCacheTransform._module_mapping.keys())) and name!="":
            model_handles.append(module.register_forward_hook(get_intermediate_outputs(name, model_intermediates)))

    model_out = llama_model(**inputs)
    [handle.remove() for handle in model_handles]
    model_intermediates.update({"final_output": model_out})
    return model_intermediates


def compare_original_vs_kv_model_outputs(original_val, kv_val, tolerance=1e-6) -> bool:
    # Base Case
    if isinstance(original_val, DynamicCache):
        # To handle-> KV cache model Cache shape and original model cache shape is different
        return True
    elif original_val is None:
        assert kv_val is None
        return True
    elif isinstance(original_val, torch.Tensor):
        mae = torch.mean(torch.abs(original_val-kv_val))
        if  mae >= tolerance:
            logger.critical(f"MAE={mae} is greater than expected tolerance={tolerance}")
            return False
        return True
    
    # Call recursively if tuple/dict
    elif isinstance(original_val, (tuple, list)):
        for sub_orig_val, sub_kv_val in zip(original_val, kv_val):
            if not compare_original_vs_kv_model_outputs(sub_orig_val, sub_kv_val, tolerance):
                return False
        return True
    elif isinstance(original_val, dict) or hasattr(original_val, "__dict__"):
        for output_name in original_val.keys():    
            sub_orig_val = original_val[output_name]
            sub_k_val = kv_val[output_name]
            if isinstance(sub_orig_val, DynamicCache) or output_name == "past_key_values":
                # To handle mismatch of shapes for past_key_values
                continue
            if output_name=="logits":
                # FIXME: Why is the tolerance need to be so high for logits?
                tolerance= 0.8
            if not compare_original_vs_kv_model_outputs(sub_orig_val, sub_k_val, tolerance):
                return False
        return True
    else:
        raise TypeError(f"got unexpected type inputs {type(original_val)}")


def test_module_mapping_transform():
    with pytest.raises(TypeError):
        ModuleMapping()

    class TestTransform(ModuleMapping):
        _module_mapping = {nn.Linear: nn.Identity}

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.a = nn.Linear(32, 64)
            self.b = nn.Linear(64, 32)

        def forward(self, x):
            x = self.a(x)
            x = self.b(x)
            return x

    model = TestModel()
    x = torch.rand(1, 32)
    y1 = model(x)
    assert torch.any(y1 != x)

    model, transformed = TestTransform.apply(model)
    assert transformed
    y2 = model(x)
    assert torch.all(y2 == x)


@pytest.mark.parametrize("input_size", [random.randint(0, 10)], ids=lambda x: "input_size="+ str(x))
@pytest.mark.parametrize("hidden_size", random.sample([4, 64, 2048, 4096], 2), ids=lambda x: "hidden_size="+ str(x))
@pytest.mark.parametrize("module", CustomOpsTransform._module_mapping.keys(), ids=lambda x: "module=" + x.__name__)
def test_custom_ops_transform(module: nn.Module, hidden_size: int, input_size: int) -> None:
    """Test custom Ops transform individually

    Args:
        module (nn.Module): Pytorch module
        hidden_size (int): hidden_size for RMSNorm operation
        input_size (int): Random inputs shape for testing
    """
    model = module(hidden_size=hidden_size)
    rand_data  = torch.rand(input_size, hidden_size)
    
    original_output = model(rand_data)
    
    model, transformed = CustomOpsTransform.apply(model)
    assert transformed

    transformed_model_output = model(rand_data)
    
    assert not isinstance(model, module)
    assert torch.all(original_output==transformed_model_output)


@pytest.mark.parametrize("hidden_size", [128], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("intermediate_size", [512], ids=lambda x: "intermediate_size=" + str(x))
@pytest.mark.parametrize("vocab_size", [32000], ids=lambda x: "vocab_size=" + str(x))
@pytest.mark.parametrize("num_key_value_heads", [8, 32], ids=lambda x: "num_key_value_heads=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [32], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1], ids=lambda x: "num_hidden_layers=" + str(x))
def test_kv_cache_transform_llama(num_hidden_layers, vocab_size, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, ctx_len) -> None:
    # Create small model
    config = LlamaConfig(vocab_size=vocab_size,
                               hidden_size=hidden_size,
                               intermediate_size=intermediate_size,
                               num_attention_heads=num_attention_heads,
                               num_key_value_heads=num_key_value_heads,
                               num_hidden_layers=num_hidden_layers,
                               use_cache=True)
    hf_model = LlamaForCausalLM(config=config)
    hf_model.eval()

    # Run original model
    input_ids = torch.randint(0,vocab_size, size=(1, 8))
    original_model_outputs = get_all_kv_cache_transform_intermediate_outputs(hf_model, {"input_ids": input_ids})

    # Apply transform
    hf_model, transformed = KVCacheTransform.apply(hf_model)
    assert transformed

    # Prepare KV model inputs
    padding_shape = [1, num_key_value_heads, ctx_len, hidden_size // num_attention_heads]
    past_key_values = []
    for _ in range(num_hidden_layers):
        past_key = torch.zeros((padding_shape), dtype=torch.float32)
        past_value = torch.zeros((padding_shape), dtype=torch.float32)
        pkv = (past_key, past_value)
        past_key_values.append(pkv)

    # Run KV model
    transformed_model_outputs = get_all_kv_cache_transform_intermediate_outputs(hf_model,
                                                                                inputs={"input_ids":input_ids,
                                                                                        "position_ids":torch.Tensor([range(input_ids.shape[1])]).long(),
                                                                                        "past_key_values":tuple(past_key_values)})
    
    assert compare_original_vs_kv_model_outputs(original_model_outputs, transformed_model_outputs, tolerance=1e-6)
