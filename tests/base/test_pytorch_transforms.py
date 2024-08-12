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
from transformers.models.codegen.modeling_codegen import CodeGenConfig, CodeGenForCausalLM
from transformers.models.falcon.modeling_falcon import FalconConfig, FalconForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers.models.gptj.modeling_gptj import GPTJConfig, GPTJForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralConfig, MistralForCausalLM
from transformers.models.mixtral.modeling_mixtral import MixtralConfig, MixtralForCausalLM
from transformers.models.mpt.modeling_mpt import MptConfig, MptForCausalLM
from transformers.models.phi.modeling_phi import PhiConfig, PhiForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
from transformers.models.starcoder2.modeling_starcoder2 import Starcoder2Config, Starcoder2ForCausalLM

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.transformers.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.utils.logging_utils import logger


def compare_original_vs_kv_model_pt_outputs(original_val, kv_val, tolerance=1e-6) -> bool:
    # Base case
    if original_val is None:
        assert kv_val is None
        return True
    elif isinstance(original_val, torch.Tensor):
        mae = torch.mean(torch.abs(original_val - kv_val))
        if mae >= tolerance:
            logger.critical(f"MAE={mae} is greater than expected tolerance={tolerance}")
            return False
        return True

    # Call recursively if tuple/list
    elif isinstance(original_val, (tuple, list)):
        for sub_orig_val, sub_kv_val in zip(original_val, kv_val):
            if not compare_original_vs_kv_model_pt_outputs(sub_orig_val, sub_kv_val, tolerance):
                return False
        return True
    else:
        raise TypeError(f"got unexpected type inputs {type(original_val)}")


def run_kv_cache_transform_and_test(
    hf_model,
    num_hidden_layers,
    vocab_size,
    hidden_size,
    num_attention_heads,
    num_key_value_heads,
    ctx_len,
    input_len,
    logits_tolerance=0.8,
):
    # Run original model
    input_ids = torch.randint(0, vocab_size, size=(1, input_len))
    with torch.inference_mode():
        original_model_outputs = hf_model(input_ids=input_ids, output_hidden_states=True)

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
    with torch.inference_mode():
        transformed_model_outputs = hf_model(
            input_ids=input_ids,
            position_ids=torch.Tensor([range(input_ids.shape[1])]).long(),
            past_key_values=tuple(past_key_values),
            output_hidden_states=True,
        )

    assert original_model_outputs.keys() == transformed_model_outputs.keys()

    # FIXME: Tolerance should not be so high for logits
    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["logits"], transformed_model_outputs["logits"], tolerance=logits_tolerance
    ), "Logits are not matching with tolerance=0.8"
    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["hidden_states"], transformed_model_outputs["hidden_states"], tolerance=1e-6
    )

    # Slice Past key values based on input_len
    pkv = transformed_model_outputs["past_key_values"][0]
    new_pkv = []
    for past_key_value in pkv:
        new_pkv.append(past_key_value[:, :, :input_len, :])
    transformed_model_outputs["past_key_values"] = (tuple(new_pkv),)

    assert compare_original_vs_kv_model_pt_outputs(
        original_model_outputs["past_key_values"], transformed_model_outputs["past_key_values"], tolerance=1e-10
    )


def test_module_mapping_transform():
    with pytest.raises(TypeError):
        ModuleMappingTransform()

    class TestTransform(ModuleMappingTransform):
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


@pytest.mark.parametrize("input_size", [random.randint(0, 10)], ids=lambda x: "input_size=" + str(x))
@pytest.mark.parametrize("hidden_size", random.sample([4, 64, 2048, 4096], 2), ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("module", CustomOpsTransform._module_mapping.keys(), ids=lambda x: "module=" + x.__name__)
def test_custom_ops_transform(module: nn.Module, hidden_size: int, input_size: int) -> None:
    """Test custom Ops transform individually

    Args:
        module (nn.Module): Pytorch module
        hidden_size (int): hidden_size for RMSNorm operation
        input_size (int): Random inputs shape for testing
    """
    model = module(hidden_size=hidden_size)
    rand_data = torch.rand(input_size, hidden_size)

    original_output = model(rand_data)

    model, transformed = CustomOpsTransform.apply(model)
    assert transformed

    transformed_model_output = model(rand_data)

    assert not isinstance(model, module)
    assert torch.all(original_output == transformed_model_output)


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("hidden_size", [128], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("intermediate_size", [512], ids=lambda x: "intermediate_size=" + str(x))
@pytest.mark.parametrize("num_key_value_heads", [8, 32], ids=lambda x: "num_key_value_heads=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [32], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1, 3], ids=lambda x: "num_hidden_layers=" + str(x))
def test_kv_cache_transform_llama(
    num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, ctx_len, input_len
) -> None:
    # Create small model
    config = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        use_cache=True,
    )
    hf_model = LlamaForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers,
        config.vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        ctx_len,
        input_len,
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("n_embd", [192], ids=lambda x: "n_embd=" + str(x))
@pytest.mark.parametrize("n_inner", [512], ids=lambda x: "n_inner=" + str(x))
@pytest.mark.parametrize("n_head", [12], ids=lambda x: "n_head=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("n_layer", [1, 3], ids=lambda x: "n_layer=" + str(x))
def test_kv_cache_transform_gpt2(n_layer, n_embd, n_inner, n_head, ctx_len, input_len) -> None:
    # Create small model
    config = GPT2Config(
        n_embd=n_embd,
        n_inner=n_inner,
        n_head=n_head,
        n_layer=n_layer,
        use_cache=True,
    )
    hf_model = GPT2LMHeadModel(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(hf_model, n_layer, config.vocab_size, n_embd, n_head, n_head, ctx_len, input_len)


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("n_embd", [1024], ids=lambda x: "n_embd=" + str(x))
@pytest.mark.parametrize("n_inner", [2048], ids=lambda x: "n_inner=" + str(x))
@pytest.mark.parametrize("n_head", [16], ids=lambda x: "n_head=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("n_layer", [1, 3], ids=lambda x: "n_layer=" + str(x))
def test_kv_cache_transform_codegen(n_layer, n_embd, n_inner, n_head, ctx_len, input_len) -> None:
    # Create small model
    config = CodeGenConfig(
        n_embd=n_embd,
        n_inner=n_inner,
        n_head=n_head,
        n_layer=n_layer,
        use_cache=True,
    )
    hf_model = CodeGenForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(hf_model, n_layer, config.vocab_size, n_embd, n_head, n_head, ctx_len, input_len)


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("hidden_size", [4544], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1], ids=lambda x: "num_hidden_layers=" + str(x))
@pytest.mark.parametrize("multi_query", [True, False], ids=lambda x: "multi_query=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [71], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
def test_kv_cache_transform_falcon(
    hidden_size, num_hidden_layers, num_attention_heads, ctx_len, input_len, multi_query
) -> None:
    # Create small model
    config = FalconConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        multi_query=multi_query,
        use_cache=True,
        _attn_implementation="eager",
    )
    hf_model = FalconForCausalLM(config=config)
    hf_model.eval()
    # FIXME: Logits Tolerance is too high!!!
    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers,
        config.vocab_size,
        hidden_size,
        num_attention_heads,
        num_attention_heads,
        ctx_len,
        input_len,
        logits_tolerance=1.5,
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("n_embd", [4096], ids=lambda x: "n_embd=" + str(x))
@pytest.mark.parametrize("n_inner", [512], ids=lambda x: "n_inner=" + str(x))
@pytest.mark.parametrize("n_head", [16], ids=lambda x: "n_head=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("n_layer", [1, 3], ids=lambda x: "n_layer=" + str(x))
def test_kv_cache_transform_gptj(n_layer, n_embd, n_inner, n_head, ctx_len, input_len) -> None:
    # Create small model
    config = GPTJConfig(
        n_embd=n_embd,
        n_inner=n_inner,
        n_head=n_head,
        n_layer=n_layer,
        use_cache=True,
    )
    hf_model = GPTJForCausalLM(config=config)
    hf_model.eval()
    if n_layer == 1:
        logits_tolerance = 1.2
    else:
        logits_tolerance = 0.8
    run_kv_cache_transform_and_test(
        hf_model,
        n_layer,
        config.vocab_size,
        n_embd,
        n_head,
        n_head,
        ctx_len,
        input_len,
        logits_tolerance=logits_tolerance,
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("hidden_size", [128], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("intermediate_size", [512], ids=lambda x: "intermediate_size=" + str(x))
@pytest.mark.parametrize("num_key_value_heads", [8, 32], ids=lambda x: "num_key_value_heads=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [32], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1, 3], ids=lambda x: "num_hidden_layers=" + str(x))
def test_kv_cache_transform_mistral(
    num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, ctx_len, input_len
) -> None:
    # Create small model
    config = MistralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        use_cache=True,
    )
    hf_model = MistralForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers,
        config.vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        ctx_len,
        input_len,
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("hidden_size", [128], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("intermediate_size", [512], ids=lambda x: "intermediate_size=" + str(x))
@pytest.mark.parametrize("num_key_value_heads", [8, 32], ids=lambda x: "num_key_value_heads=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [32], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1, 3], ids=lambda x: "num_hidden_layers=" + str(x))
def test_kv_cache_transform_mixtral(
    num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, ctx_len, input_len
) -> None:
    # Create small model
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        use_cache=True,
    )
    hf_model = MixtralForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers,
        config.vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        ctx_len,
        input_len,
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("d_model", [2048], ids=lambda x: "d_model=" + str(x))
@pytest.mark.parametrize("n_heads", [16], ids=lambda x: "n_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("n_layers", [1, 3], ids=lambda x: "n_layers=" + str(x))
def test_kv_cache_transform_mpt(n_layers, d_model, n_heads, ctx_len, input_len) -> None:
    # Create small model
    config = MptConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        use_cache=True,
    )
    hf_model = MptForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(
        hf_model, n_layers, config.vocab_size, d_model, n_heads, n_heads, ctx_len, input_len
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("hidden_size", [128], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("intermediate_size", [512], ids=lambda x: "intermediate_size=" + str(x))
@pytest.mark.parametrize("num_key_value_heads", [8, 32], ids=lambda x: "num_key_value_heads=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [32], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1, 3], ids=lambda x: "num_hidden_layers=" + str(x))
def test_kv_cache_transform_phi(
    num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, ctx_len, input_len
) -> None:
    # Create small model
    config = PhiConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        use_cache=True,
    )
    hf_model = PhiForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers,
        config.vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        ctx_len,
        input_len,
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("hidden_size", [128], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("intermediate_size", [512], ids=lambda x: "intermediate_size=" + str(x))
@pytest.mark.parametrize("num_key_value_heads", [8, 32], ids=lambda x: "num_key_value_heads=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [32], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1, 3], ids=lambda x: "num_hidden_layers=" + str(x))
def test_kv_cache_transform_phi3(
    num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, ctx_len, input_len
) -> None:
    # Create small model
    config = Phi3Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        use_cache=True,
    )
    hf_model = Phi3ForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers,
        config.vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        ctx_len,
        input_len,
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("hidden_size", [128], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("intermediate_size", [512], ids=lambda x: "intermediate_size=" + str(x))
@pytest.mark.parametrize("num_key_value_heads", [8, 32], ids=lambda x: "num_key_value_heads=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [32], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1, 3], ids=lambda x: "num_hidden_layers=" + str(x))
def test_kv_cache_transform_qwen2(
    num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, ctx_len, input_len
) -> None:
    # Create small model
    config = Qwen2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        use_cache=True,
    )
    hf_model = Qwen2ForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers,
        config.vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        ctx_len,
        input_len,
    )


@pytest.mark.parametrize("input_len", [8], ids=lambda x: "input_len=" + str(x))
@pytest.mark.parametrize("hidden_size", [128], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("intermediate_size", [512], ids=lambda x: "intermediate_size=" + str(x))
@pytest.mark.parametrize("num_key_value_heads", [8, 16], ids=lambda x: "num_key_value_heads=" + str(x))
@pytest.mark.parametrize("num_attention_heads", [16], ids=lambda x: "num_attention_heads=" + str(x))
@pytest.mark.parametrize("ctx_len", [32], ids=lambda x: "ctx_len=" + str(x))
@pytest.mark.parametrize("num_hidden_layers", [1, 3], ids=lambda x: "num_hidden_layers=" + str(x))
def test_kv_cache_transform_starcoder2(
    num_hidden_layers, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, ctx_len, input_len
) -> None:
    # Create small model
    config = Starcoder2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=num_hidden_layers,
        use_cache=True,
    )
    hf_model = Starcoder2ForCausalLM(config=config)
    hf_model.eval()
    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers,
        config.vocab_size,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        ctx_len,
        input_len,
    )
