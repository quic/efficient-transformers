# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.customop.matmulnbits import QuantLinearORT
from QEfficient.transformers.pytorch_transforms import CustomOpsTransform, KVCacheTransform
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.quant_transforms import AwqToMatmulNbitsTransform
from QEfficient.utils._utils import get_padding_shape_from_config
from QEfficient.utils.logging_utils import logger

KVCacheTransformTestConfigs = [
    ("llama", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("llama", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("llama", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("llama", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("gpt2", 3, 12, 192, {"n_inner": 512}, 0.8),
    ("gpt2", 1, 12, 192, {"n_inner": 512}, 0.8),
    ("codegen", 1, 16, 1024, {"n_inner": 2048}, 0.8),
    ("codegen", 3, 16, 1024, {"n_inner": 2048}, 0.8),
    ("falcon", 1, 71, 4544, {"multi_query": True}, 1.5),
    ("falcon", 3, 71, 4544, {"multi_query": False}, 1.5),
    ("falcon", 1, 71, 4544, {"multi_query": False}, 1.5),
    ("falcon", 3, 71, 4544, {"multi_query": True}, 1.5),
    ("gptj", 3, 16, 4096, {"n_inner": 512}, 1),
    ("gptj", 1, 16, 4096, {"n_inner": 512}, 1.2),
    ("mistral", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("mistral", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("mistral", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("mistral", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("mixtral", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("mixtral", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("mixtral", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("mixtral", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("mpt", 1, 16, 2048, {}, 0.8),
    ("mpt", 3, 16, 2048, {}, 0.8),
    ("phi", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("phi", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("phi", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("phi", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("phi3", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("phi3", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("phi3", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("phi3", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("qwen2", 1, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("qwen2", 1, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("qwen2", 3, 32, 128, {"num_key_value_heads": 8, "intermediate_size": 512}, 0.8),
    ("qwen2", 3, 32, 128, {"num_key_value_heads": 32, "intermediate_size": 512}, 0.8),
    ("starcoder2", 3, 24, 192, {"num_key_value_heads": 2, "intermediate_size": 512}, 0.8),
    ("starcoder2", 1, 24, 192, {"num_key_value_heads": 2, "intermediate_size": 512}, 0.8),
    ("starcoder2", 3, 24, 192, {"num_key_value_heads": 24, "intermediate_size": 512}, 0.8),
    ("starcoder2", 1, 24, 192, {"num_key_value_heads": 24, "intermediate_size": 512}, 0.8),
]


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
    padding_shape,
    vocab_size,
    input_len,
    logits_tolerance=0.8,
):
    hf_model.eval()
    # Run original model
    input_ids = torch.randint(0, vocab_size, size=(1, input_len))
    with torch.inference_mode():
        original_model_outputs = hf_model(input_ids=input_ids, output_hidden_states=True)

    # Apply transform
    hf_model, transformed = KVCacheTransform.apply(hf_model)
    assert transformed

    # Prepare KV model inputs
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


@pytest.mark.parametrize("input_size", [2, 5], ids=lambda x: "input_size=" + str(x))
@pytest.mark.parametrize("hidden_size", [64, 1024], ids=lambda x: "hidden_size=" + str(x))
@pytest.mark.parametrize("module", CustomOpsTransform._module_mapping.keys(), ids=lambda x: "module=" + x.__name__)
def test_rms_norm_ops_transform(module: torch.nn.Module, hidden_size: int, input_size: int) -> None:
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


@pytest.mark.parametrize(
    "config_class, num_hidden_layers, num_attention_heads, hidden_size, kwargs, logits_tolerance",
    KVCacheTransformTestConfigs,
)
def test_kv_cache_transform(
    config_class, num_hidden_layers, num_attention_heads, hidden_size, kwargs, logits_tolerance
):
    config = AutoConfig.for_model(
        config_class,
        **kwargs,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        use_cache=True,
    )
    hf_model = AutoModelForCausalLM.from_config(config=config, attn_implementation="eager")

    padding_shape = get_padding_shape_from_config(config=config, batch_size=1, seq_len=32)

    run_kv_cache_transform_and_test(
        hf_model,
        num_hidden_layers=num_hidden_layers,
        padding_shape=padding_shape,
        vocab_size=config.vocab_size,
        input_len=8,
        logits_tolerance=logits_tolerance,
    )


@pytest.mark.parametrize("in_features", [2048, 4096])
@pytest.mark.parametrize("out_features", [2048, 4096])
def test_awq_to_matmulnbits_transform(in_features, out_features):
    wqlinear = WQLinear_GEMM(w_bit=4, group_size=128, in_features=in_features, out_features=out_features, bias=False)

    wqlinear.qweight = torch.randint(
        low=-(2**31), high=2**31 - 1, size=(in_features, out_features // 8), dtype=torch.int32
    )
    wqlinear.qzeros = torch.randint(
        low=-(2**31), high=2**31 - 1, size=(in_features // wqlinear.group_size, out_features // 8), dtype=torch.int32
    )
    wqlinear.scales = torch.rand(in_features // wqlinear.group_size, out_features, dtype=torch.float32)

    rand_data = torch.rand(4, in_features)
    old_out = wqlinear(rand_data)
    new_module, transformed = AwqToMatmulNbitsTransform.apply(wqlinear)
    assert transformed
    new_out = new_module(rand_data)
    assert isinstance(new_module, QuantLinearORT)
    compare_original_vs_kv_model_pt_outputs(old_out, new_out, tolerance=1e-8)
