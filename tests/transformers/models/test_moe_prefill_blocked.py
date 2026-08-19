# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import copy
from collections import Counter

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEFFAutoModelForCausalLM

MODEL_KWARGS = {"attn_implementation": "eager"}

GLM4_MOE_CFG = dict(
    max_position_embeddings=1024,
    num_hidden_layers=2,
    num_attention_heads=4,
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    vocab_size=127,
    num_key_value_heads=2,
    n_routed_experts=4,
    num_experts_per_tok=2,
    first_k_dense_replace=0,
    n_group=1,
    topk_group=1,
    head_dim=16,
)


def test_glm4_moe_blocked_prefill_forward_parity():
    from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import (
        QEffGlm4MoeMoE,
        QEffPrefillChunkedGlm4MoeMoE,
    )

    config = AutoConfig.for_model("glm4_moe", **GLM4_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    block = next(module for module in model.modules() if module.__class__.__name__ == "Glm4MoeMoE")

    qeff_block = copy.deepcopy(block)
    qeff_block.__class__ = QEffGlm4MoeMoE
    qeff_block.__qeff_init__()

    chunked_block = copy.deepcopy(block)
    chunked_block.__class__ = QEffPrefillChunkedGlm4MoeMoE
    chunked_block.__qeff_init__()
    chunked_block.expert_blocking_num_nsp = 2
    chunked_block.expert_blocking_packed_chunk_size = 256

    x = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        orig = qeff_block(x)
        blocked = chunked_block(x)

    assert orig.shape == blocked.shape
    assert torch.allclose(orig, blocked, atol=1e-4, rtol=1e-4)


def test_glm4_moe_decode_export(tmp_path):
    config = AutoConfig.for_model("glm4_moe", **GLM4_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.export(tmp_path / "decode")
    assert qeff.onnx_path.is_file()


def test_glm4_moe_prefill_chunked_subfunction_export_contains_cumsum_custom_ops(tmp_path):
    import onnx

    config = AutoConfig.for_model("glm4_moe", **GLM4_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    onnx_path = qeff.export(
        tmp_path / "prefill-subfunction",
        prefill_only=True,
        prefill_seq_len=512,
        enable_chunking=True,
        num_cores=2,
        moe_prefill_packed_chunk_size=256,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    decoder_functions = [func for func in onnx_model.functions if func.name.startswith("QEffGlm4MoeDecoderLayer")]
    assert len(decoder_functions) == config.num_hidden_layers

    for function_proto in decoder_functions:
        op_counts = Counter(node.op_type for node in function_proto.node)
        assert op_counts["Sin"] == 0
        assert op_counts["Cos"] == 0
        # prefill_seq_len=512 and packed_chunk_size=256 gives two packed chunks.
        # With n_routed_experts=4 and num_cores=2, each layer has two expert slots.
        assert op_counts["CtxGather3D"] == 12
        assert op_counts["CtxScatter3D"] == 4
        assert op_counts["CtxScatter3DInt"] == 2


def test_glm4_moe_kv_blocking_transform_and_prefill_export(tmp_path):
    import onnx

    from QEfficient.blocking.attention_blocking import BlockingMode
    from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import QEffGlm4MoeAttention

    config = AutoConfig.for_model("glm4_moe", **GLM4_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    qeff.transform(
        ctx_len=1024,
        seq_len=512,
        qaic_config={"enable_blocking": True, "blocking_mode": "kv", "num_kv_blocks": 2},
    )

    attn_modules = [module for module in qeff.model.modules() if isinstance(module, QEffGlm4MoeAttention)]
    assert attn_modules
    for attn_module in attn_modules:
        blocking_config = getattr(attn_module, "attn_blocking_config", None)
        assert blocking_config is not None
        assert blocking_config.mode == BlockingMode.KV
        assert blocking_config.num_kv_blocks == 2

    onnx_path = qeff.export(
        tmp_path / "prefill-kv-blocked",
        prefill_only=True,
        prefill_seq_len=512,
        enable_chunking=True,
        use_onnx_subfunctions=True,
        num_cores=2,
        moe_prefill_packed_chunk_size=256,
        offload_pt_weights=False,
    )
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    decoder_functions = [func for func in onnx_model.functions if func.name.startswith("QEffGlm4MoeDecoderLayer")]
    assert len(decoder_functions) == config.num_hidden_layers
    for function_proto in decoder_functions:
        op_counts = Counter(node.op_type for node in function_proto.node)
        assert op_counts["CtxGatherBlockedKV"] == 4


# ── Qwen3MOE ──────────────────────────────────────────────────────────────────


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
        QEffQwen3MoeExperts,
    )

    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)

    blocks = [
        m
        for _, m in model.named_modules()
        if hasattr(m, "experts") and hasattr(m, "gate") and hasattr(m.gate, "num_experts")
    ]
    assert blocks

    block = blocks[0]
    chunked = QEffPrefillChunkedQwen3MoeSparseMoeBlock.__new__(QEffPrefillChunkedQwen3MoeSparseMoeBlock)
    chunked.__dict__.update(block.__dict__)
    chunked.__class__ = QEffPrefillChunkedQwen3MoeSparseMoeBlock
    chunked.experts.__class__ = QEffQwen3MoeExperts
    chunked.experts.__qeff_init__()
    chunked.__qeff_init__()
    x = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        orig, _ = chunked.orig_forward(x)
        chunked.expert_blocking_num_nsp = 2
        chunked.expert_blocking_packed_chunk_size = 256
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
    qeff.export(tmp_path / "prefill", prefill_only=True, enable_chunking=True, num_cores=2)
    assert qeff.onnx_path.is_file()


def test_qwen3moe_disagg_compile_uses_distinct_decode_and_prefill_onnx(tmp_path, monkeypatch):
    import subprocess

    compile_commands = []

    def fake_compile(command, *args, **kwargs):
        compile_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_compile)

    config = AutoConfig.for_model("qwen3_moe", **QWEN3_MOE_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)

    qeff.compile(
        compile_dir=tmp_path / "decode-compile",
        prefill_seq_len=1,
        ctx_len=128,
        num_cores=2,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        offload_pt_weights=False,
        retain_full_kv=True,
    )
    decode_onnx_path = qeff.onnx_path

    qeff.compile(
        compile_dir=tmp_path / "prefill-compile",
        prefill_seq_len=64,
        ctx_len=128,
        num_cores=2,
        moe_prefill_packed_chunk_size=32,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        prefill_only=True,
        enable_chunking=True,
        offload_pt_weights=False,
    )
    prefill_onnx_path = qeff.onnx_path

    compiled_onnx_args = [arg for command in compile_commands for arg in command if str(arg).startswith("-m=")]
    assert len(compiled_onnx_args) == 2
    assert decode_onnx_path != prefill_onnx_path
    assert decode_onnx_path.is_file()
    assert prefill_onnx_path.is_file()
    assert compiled_onnx_args[0] == f"-m={decode_onnx_path}"
    assert compiled_onnx_args[1] == f"-m={prefill_onnx_path}"


def test_qwen3moe_prefill_chunked_subfunction_export_contains_cumsum_custom_ops(tmp_path):
    import onnx
    from onnx import numpy_helper

    config = AutoConfig.for_model("qwen3_moe", **{**QWEN3_MOE_CFG, "max_position_embeddings": 1024})
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)
    onnx_path = qeff.export(
        tmp_path / "prefill-subfunction",
        prefill_only=True,
        enable_chunking=True,
        prefill_seq_len=512,
        num_cores=2,
        moe_prefill_packed_chunk_size=256,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    function_names = {func.name for func in onnx_model.functions}
    used_op_types = {node.op_type for node in onnx_model.graph.node}
    slice_starts = []
    for function_proto in onnx_model.functions:
        constants = {}
        for node in function_proto.node:
            used_op_types.add(node.op_type)
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        constants[node.output[0]] = numpy_helper.to_array(attr.t).flatten().tolist()
        for node in function_proto.node:
            if node.op_type == "Slice" and len(node.input) > 1 and node.input[1] in constants:
                slice_starts.append(constants[node.input[1]])

    assert "CtxScatter3DInt" in function_names
    assert "CtxScatter3D" in function_names
    assert "CtxGather3D" in function_names
    assert "CtxScatter3DInt" in used_op_types
    assert "CtxScatter3D" in used_op_types
    assert "CtxGather3D" in used_op_types
    assert [256] in slice_starts


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
    blocks_chunked[0].expert_blocking_num_nsp = 2
    blocks_chunked[0].expert_blocking_packed_chunk_size = 256

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
    qeff.export(tmp_path / "prefill", prefill_only=True, enable_chunking=True, num_cores=2)
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
        num_cores=2,
        moe_prefill_packed_chunk_size=256,
        use_onnx_subfunctions=True,
        offload_pt_weights=False,
    )

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    op_types = []
    slice_starts = []
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
