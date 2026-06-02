# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Tests for NSP-blocked MoE prefill dispatch (Qwen3MOE + GPT-OSS).
Uses EXPERT_BLOCKING_NUM_NSP=2 so tests run fast on any num_experts.
Covers: parity, decode export, prefill+chunking export (disagg mode).
"""

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

os.environ.setdefault("EXPERT_BLOCKING_NUM_NSP", "2")

from QEfficient import QEFFAutoModelForCausalLM

MODEL_KWARGS = {"attn_implementation": "eager"}

QWEN3_MOE_CFG = dict(
    max_position_embeddings=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    hidden_size=128,
    intermediate_size=512,
    vocab_size=127,
    num_key_value_heads=2,
)
GPTOSS_CFG = dict(
    max_position_embeddings=256,
    num_hidden_layers=2,
    num_attention_heads=2,
    hidden_size=32,
    intermediate_size=32,
    vocab_size=127,
    num_key_value_heads=2,
)


# ── Qwen3MOE ──────────────────────────────────────────────────────────────────


def test_qwen3moe_blocked_forward_parity():
    from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import (
        QEffPrefillChunkedQwen3MoeSparseMoeBlock,
    )

    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)

    blocks = [
        m
        for _, m in model.named_modules()
        if hasattr(m, "experts") and hasattr(m, "gate") and hasattr(m, "num_experts")
    ]
    assert blocks

    block = blocks[0]
    chunked = QEffPrefillChunkedQwen3MoeSparseMoeBlock.__new__(QEffPrefillChunkedQwen3MoeSparseMoeBlock)
    chunked.__dict__.update(block.__dict__)
    chunked.__class__ = QEffPrefillChunkedQwen3MoeSparseMoeBlock
    chunked.__qeff_init__()

    x = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        orig, _ = chunked.orig_forward(x)
        blocked, _ = chunked.forward(x)

    assert orig.shape == blocked.shape
    assert (orig - blocked).abs().max().item() < 0.1, "Qwen3MOE parity failed"


def test_qwen3moe_decode_export(tmp_path):
    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.export(tmp_path / "decode")
    assert qeff.onnx_path.is_file()


def test_qwen3moe_prefill_chunked_export(tmp_path):
    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.export(tmp_path / "prefill", prefill_only=True, enable_chunking=True)
    assert qeff.onnx_path.is_file()


def test_qwen3moe_prefill_chunked_subfunction_export_contains_cumsum_custom_ops(tmp_path):
    import onnx

    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    onnx_path = qeff.export(
        tmp_path / "prefill-subfunction",
        prefill_only=True,
        enable_chunking=True,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    function_names = {func.name for func in onnx_model.functions}
    used_op_types = {node.op_type for node in onnx_model.graph.node}
    for function_proto in onnx_model.functions:
        used_op_types.update(node.op_type for node in function_proto.node)

    assert "CtxScatter3DInt" in function_names
    assert "CtxScatter3D" in function_names
    assert "CtxGather3D" in function_names
    assert "CtxScatter3DInt" in used_op_types
    assert "CtxScatter3D" in used_op_types
    assert "CtxGather3D" in used_op_types


# ── GPT-OSS ───────────────────────────────────────────────────────────────────


def test_gptoss_blocked_forward_parity():
    from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
        QEffPrefillOnlyChunkedGptOssMLP,
    )
    from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyChunkedTransform

    config = AutoConfig.for_model("gpt_oss", **GPTOSS_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)

    blocks_orig = [m for _, m in model.named_modules() if m.__class__.__name__ == "GptOssMLP"]
    assert blocks_orig

    x = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        orig, _ = blocks_orig[0].forward(x)

    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    PrefillOnlyChunkedTransform.apply(qeff.model)

    blocks_chunked = [m for _, m in qeff.model.named_modules() if isinstance(m, QEffPrefillOnlyChunkedGptOssMLP)]
    assert blocks_chunked

    with torch.no_grad():
        blocked, _ = blocks_chunked[0].forward(x)

    assert orig.shape == blocked.shape
    assert (orig - blocked).abs().max().item() < 0.1, "GPT-OSS parity failed"


def test_gptoss_decode_export(tmp_path):
    config = AutoConfig.for_model("gpt_oss", **GPTOSS_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.export(tmp_path / "decode")
    assert qeff.onnx_path.is_file()


def test_gptoss_prefill_chunked_export(tmp_path):
    config = AutoConfig.for_model("gpt_oss", **GPTOSS_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.export(tmp_path / "prefill", prefill_only=True, enable_chunking=True)
    assert qeff.onnx_path.is_file()


def test_gptoss_prefill_chunked_export_traces_packed_chunks(tmp_path):
    import onnx
    from onnx import numpy_helper

    config = AutoConfig.for_model("gpt_oss", **{**GPTOSS_CFG, "max_position_embeddings": 1024})
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=True)
    onnx_path = qeff.export(
        tmp_path / "prefill-subfunction-512",
        prefill_only=True,
        enable_chunking=True,
        prefill_seq_len=512,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    slice_starts = []
    op_types = []
    for nodes in [onnx_model.graph.node] + [function.node for function in onnx_model.functions]:
        constants = {}
        for node in nodes:
            op_types.append(node.op_type)
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        constants[node.output[0]] = numpy_helper.to_array(attr.t).flatten().tolist()
        for node in nodes:
            if node.op_type == "Slice" and len(node.input) > 1 and node.input[1] in constants:
                slice_starts.append(constants[node.input[1]])

    assert [256] in slice_starts
    assert op_types.count("CtxGather3D") >= 2 * op_types.count("CtxScatter3DInt")
