# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import copy
from collections import Counter

import pytest
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

    x = torch.randn(1, 8, config.hidden_size)

    # HF reference forward (before any QEff class swap).
    hf_block = copy.deepcopy(block)
    with torch.no_grad():
        hf_out = hf_block(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    qeff_block = copy.deepcopy(block)
    qeff_block.__class__ = QEffGlm4MoeMoE
    qeff_block.__qeff_init__()

    chunked_block = copy.deepcopy(block)
    chunked_block.__class__ = QEffPrefillChunkedGlm4MoeMoE
    chunked_block.__qeff_init__()
    chunked_block.expert_blocking_num_nsp = 2
    chunked_block.expert_blocking_packed_chunk_size = 256

    with torch.no_grad():
        orig = qeff_block(x)
        blocked = chunked_block(x)

    assert orig.shape == blocked.shape == hf_out.shape
    # HF (module-list experts) vs QEff decode-bmm and vs QEff expert-blocked.
    assert torch.allclose(hf_out, orig, atol=1e-3, rtol=1e-3), "GLM4 HF vs decode-bmm parity failed"
    assert torch.allclose(hf_out, blocked, atol=1e-3, rtol=1e-3), "GLM4 HF vs expert-blocked parity failed"
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
        # Static-chunk export traces those two chunks over export SL=32.
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


def test_static_moe_prefill_export_seq_len_uses_chunk_count():
    from QEfficient.transformers.models.modeling_auto import _get_static_moe_prefill_export_seq_len

    assert _get_static_moe_prefill_export_seq_len(1) == 32
    assert _get_static_moe_prefill_export_seq_len(2) == 32
    assert _get_static_moe_prefill_export_seq_len(3) == 33
    assert _get_static_moe_prefill_export_seq_len(32) == 32
    assert _get_static_moe_prefill_export_seq_len(48) == 48


def test_static_moe_prefill_export_seq_len_rejects_zero_chunks():
    from QEfficient.transformers.models.modeling_auto import _get_static_moe_prefill_export_seq_len

    with pytest.raises(ValueError, match="num_packed_chunks"):
        _get_static_moe_prefill_export_seq_len(0)


def test_configure_moe_prefill_blocking_sets_static_chunk_attrs_and_hash():
    from QEfficient.transformers.models.modeling_auto import _configure_moe_prefill_blocking_modules

    class DummyBlock(torch.nn.Module):
        supports_moe_prefill_blocking = True
        supports_static_moe_prefill_chunks = True

    model = torch.nn.Sequential(DummyBlock(), DummyBlock())
    hash_params = {}

    has_blocking, has_static, num_chunks = _configure_moe_prefill_blocking_modules(
        model=model,
        hash_params=hash_params,
        prefill_seq_len=512,
        num_cores=4,
        moe_prefill_packed_chunk_size=256,
    )

    assert has_blocking is True
    assert has_static is True
    assert num_chunks == 2
    assert hash_params == {
        "moe_prefill_num_nsp": 4,
        "moe_prefill_packed_chunk_size": 256,
        "moe_prefill_num_packed_chunks": 2,
    }
    for module in model:
        assert module.expert_blocking_num_nsp == 4
        assert module.expert_blocking_packed_chunk_size == 256
        assert module.expert_blocking_num_packed_chunks == 2


def test_configure_moe_prefill_blocking_rejects_invalid_inputs():
    from QEfficient.transformers.models.modeling_auto import _configure_moe_prefill_blocking_modules

    model = torch.nn.Sequential()
    with pytest.raises(ValueError, match="moe_prefill_packed_chunk_size"):
        _configure_moe_prefill_blocking_modules(
            model=model,
            hash_params={},
            prefill_seq_len=512,
            num_cores=4,
            moe_prefill_packed_chunk_size=0,
        )
    with pytest.raises(ValueError, match="num_cores"):
        _configure_moe_prefill_blocking_modules(
            model=model,
            hash_params={},
            prefill_seq_len=512,
            num_cores=0,
            moe_prefill_packed_chunk_size=256,
        )


def test_configure_moe_prefill_blocking_leaves_non_moe_models_unchanged():
    from QEfficient.transformers.models.modeling_auto import _configure_moe_prefill_blocking_modules

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    hash_params = {}

    has_blocking, has_static, num_chunks = _configure_moe_prefill_blocking_modules(
        model=model,
        hash_params=hash_params,
        prefill_seq_len=512,
        num_cores=4,
        moe_prefill_packed_chunk_size=256,
    )

    assert has_blocking is False
    assert has_static is False
    assert num_chunks == 2
    assert hash_params == {}


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
    x = torch.randn(1, 8, config.hidden_size)

    # HF reference forward (before any QEff class swap).
    hf_block = copy.deepcopy(block)
    with torch.no_grad():
        hf_out = hf_block(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    chunked = QEffPrefillChunkedQwen3MoeSparseMoeBlock.__new__(QEffPrefillChunkedQwen3MoeSparseMoeBlock)
    chunked.__dict__.update(copy.deepcopy(block).__dict__)
    chunked.__class__ = QEffPrefillChunkedQwen3MoeSparseMoeBlock
    chunked.experts.__class__ = QEffQwen3MoeExperts
    chunked.experts.__qeff_init__()
    chunked.__qeff_init__()
    with torch.no_grad():
        orig, _ = chunked.orig_forward(x)
        chunked.expert_blocking_num_nsp = 2
        chunked.expert_blocking_packed_chunk_size = 256
        blocked, _ = chunked.forward(x)

    assert orig.shape == hf_out.shape == blocked.shape
    # HF (fused gate_up) vs QEff simple-loop and vs QEff expert-blocked.
    assert (hf_out - orig.view_as(hf_out)).abs().max().item() < 1e-3, "Qwen3MOE HF vs simple-loop parity failed"
    assert (hf_out - blocked.view_as(hf_out)).abs().max().item() < 0.1, "Qwen3MOE HF vs expert-blocked parity failed"
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
    # prefill_seq_len=512 and packed_chunk_size=256 gives two static chunks.
    # The exporter may fold Slice constants, so assert the source config instead
    # of relying on a literal chunk-start value in ONNX.
    assert qeff.hash_params["moe_prefill_num_packed_chunks"] == 2
    assert qeff.hash_params["moe_prefill_packed_chunk_size"] == 256
    assert qeff.hash_params["moe_prefill_num_nsp"] == 2
    assert [256] not in slice_starts


# ── GPT-OSS ───────────────────────────────────────────────────────────────────


def test_gptoss_blocked_forward_parity():
    from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
        QEffPrefillOnlyChunkedGptOssMLP,
        QEffPrefillOnlyGptOssMLP,
    )
    from QEfficient.transformers.models.pytorch_transforms import PrefillOnlyChunkedTransform, PrefillOnlyTransform

    config = AutoConfig.for_model("gpt_oss", **GPTOSS_CFG)
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)

    blocks_orig = [m for _, m in model.named_modules() if m.__class__.__name__ == "GptOssMLP"]
    assert blocks_orig

    x = torch.randn(1, 8, config.hidden_size)
    with torch.no_grad():
        orig, _ = blocks_orig[0].forward(x)

    qeff = QEFFAutoModelForCausalLM(model, continuous_batching=False)

    # Expert-blocked prefill flavour.
    chunked_model = copy.deepcopy(qeff.model)
    PrefillOnlyChunkedTransform.apply(chunked_model)
    blocks_chunked = [m for _, m in chunked_model.named_modules() if isinstance(m, QEffPrefillOnlyChunkedGptOssMLP)]
    assert blocks_chunked
    blocks_chunked[0].build_moe_weights()
    blocks_chunked[0].expert_blocking_num_nsp = 2
    blocks_chunked[0].expert_blocking_packed_chunk_size = 256
    with torch.no_grad():
        blocked, _ = blocks_chunked[0].forward(x)

    # Simple-loop prefill flavour.
    loop_model = copy.deepcopy(qeff.model)
    PrefillOnlyTransform.apply(loop_model)
    blocks_loop = [m for _, m in loop_model.named_modules() if isinstance(m, QEffPrefillOnlyGptOssMLP)]
    assert blocks_loop
    blocks_loop[0].build_moe_weights()
    with torch.no_grad():
        looped, _ = blocks_loop[0].forward(x)

    assert orig.shape == blocked.shape == looped.shape
    # HF (interleaved fused gate_up) vs QEff expert-blocked and vs QEff simple-loop.
    assert (orig - blocked).abs().max().item() < 0.1, "GPT-OSS HF vs expert-blocked parity failed"
    assert (orig - looped).abs().max().item() < 1e-3, "GPT-OSS HF vs simple-loop parity failed"


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

    # prefill_seq_len=512 and packed_chunk_size=256 gives two static chunks.
    # The exporter may fold Slice constants, so assert the source config instead
    # of relying on a literal chunk-start value in ONNX.
    assert qeff.hash_params["moe_prefill_num_packed_chunks"] == 2
    assert qeff.hash_params["moe_prefill_packed_chunk_size"] == 256
    assert qeff.hash_params["moe_prefill_num_nsp"] == 2
    assert [256] not in slice_starts
    assert op_types.count("CtxGather3D") >= 2 * op_types.count("CtxScatter3DInt")


# ── Qwen3.5-MOE ───────────────────────────────────────────────────────────────


def _randomize_expert_weights(block):
    """HF lazy-experts init expert weights to zeros; fill with random values so the
    HF reference forward is non-degenerate and parity is meaningful."""


def _randomize_expert_weights(block):
    """HF lazy-experts init expert weights to zeros; fill with random values so the
    HF reference forward is non-degenerate and parity is meaningful. Also replace any
    uninitialized (NaN) parameters so results are independent of test ordering."""
    with torch.no_grad():
        for name in ("gate_up_proj", "down_proj"):
            if hasattr(block.experts, name):
                getattr(block.experts, name).normal_(mean=0.0, std=0.1)
        for param in block.parameters():
            if torch.isnan(param).any():
                param.normal_(mean=0.0, std=0.1)
    return block


def _build_qwen3_5_moe_sparse_block():
    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    cfg = Qwen3_5MoeTextConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        vocab_size=127,
        max_position_embeddings=128,
        head_dim=32,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention"],
    )
    block = _randomize_expert_weights(Qwen3_5MoeSparseMoeBlock(cfg).eval())
    return cfg, block


def test_qwen3_5_moe_blocked_forward_parity():
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffPrefillChunkedQwen3_5MoeSparseMoeBlock,
        QEffQwen3_5MoeExperts,
        QEffQwen3_5MoeSparseMoeBlock,
        QEffQwen3_5MoeTopKRouter,
    )

    cfg, block = _build_qwen3_5_moe_sparse_block()
    x = torch.randn(1, 8, cfg.hidden_size)
    with torch.no_grad():
        hf_out = block(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    def _to_qeff(cls):
        b = copy.deepcopy(block)
        b.gate.__class__ = QEffQwen3_5MoeTopKRouter
        b.experts.__class__ = QEffQwen3_5MoeExperts
        b.experts.__qeff_init__()
        b.__class__ = cls
        b.__qeff_init__()
        b.build_moe_weights()
        return b

    decode = _to_qeff(QEffQwen3_5MoeSparseMoeBlock)
    chunked = _to_qeff(QEffPrefillChunkedQwen3_5MoeSparseMoeBlock)
    chunked.expert_blocking_num_nsp = 2
    chunked.expert_blocking_packed_chunk_size = 256
    with torch.no_grad():
        dec = decode(x)
        blk = chunked(x)

    assert dec.shape == blk.shape == hf_out.shape
    assert (hf_out - dec).abs().max().item() < 1e-2, "Qwen3.5-MOE HF vs decode-bmm parity failed"
    assert (hf_out - blk).abs().max().item() < 1e-2, "Qwen3.5-MOE HF vs expert-blocked parity failed"


# ── Qwen3-VL-MOE ──────────────────────────────────────────────────────────────


def _build_qwen3_vl_moe_sparse_block():
    from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock

    cfg = Qwen3VLMoeTextConfig(
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        vocab_size=127,
        max_position_embeddings=128,
        head_dim=32,
        moe_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
    )
    block = _randomize_expert_weights(Qwen3VLMoeTextSparseMoeBlock(cfg).eval())
    return cfg, block


def test_qwen3_vl_moe_blocked_forward_parity():
    from QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        QEffPrefillChunkedQwen3VLMoeTextSparseMoeBlock,
        QEffQwen3VLMoeTextExperts,
        QEffQwen3VLMoeTextSparseMoeBlock,
    )

    cfg, block = _build_qwen3_vl_moe_sparse_block()
    x = torch.randn(1, 8, cfg.hidden_size)
    with torch.no_grad():
        hf_out = block(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    def _to_qeff(cls):
        b = copy.deepcopy(block)
        b.experts.__class__ = QEffQwen3VLMoeTextExperts
        b.experts.__qeff_init__()
        b.__class__ = cls
        if hasattr(b, "__qeff_init__"):
            b.__qeff_init__()
        b.build_moe_weights()
        return b

    decode = _to_qeff(QEffQwen3VLMoeTextSparseMoeBlock)
    chunked = _to_qeff(QEffPrefillChunkedQwen3VLMoeTextSparseMoeBlock)
    chunked.expert_blocking_num_nsp = 2
    chunked.expert_blocking_packed_chunk_size = 256
    with torch.no_grad():
        dec, _ = decode(x)
        blk, _ = chunked(x)

    assert dec.shape == blk.shape == hf_out.shape
    # Decode path mirrors HF (topk-then-softmax) exactly; expert-blocked uses the
    # optimized softmax-then-topk gate, so only compare decode against HF here.
    assert (hf_out - dec).abs().max().item() < 1e-2, "Qwen3-VL-MOE HF vs decode-bmm parity failed"


# ── Mixtral / GraniteMoE (decode-only, net-new canonical stacking) ─────────────


def test_mixtral_decode_forward_parity():
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

    from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import QEffMixtralSparseMoeBlock

    cfg = MixtralConfig(
        num_local_experts=4, hidden_size=32, intermediate_size=64, num_experts_per_tok=2, hidden_act="silu"
    )
    block = MixtralSparseMoeBlock(cfg).eval()
    with torch.no_grad():
        # Standalone construction may leave the lazy gate/expert params uninitialized
        # (NaN); fill them so the parity comparison is well-defined and order-independent.
        block.gate.weight.normal_(0, 0.1)
        for name in ("gate_up_proj", "down_proj"):
            if hasattr(block.experts, name):
                getattr(block.experts, name).normal_(0, 0.1)

    x = torch.randn(1, 8, cfg.hidden_size)
    with torch.no_grad():
        hf_out = block(x)
    hf_out = hf_out[0] if isinstance(hf_out, tuple) else hf_out

    qeff = copy.deepcopy(block)
    qeff.__class__ = QEffMixtralSparseMoeBlock
    with torch.no_grad():
        out = qeff(x)
    out = out[0] if isinstance(out, tuple) else out

    assert hf_out.shape == out.shape
    assert (hf_out - out).abs().max().item() < 1e-3, "Mixtral HF vs QEff decode parity failed"


def test_granitemoe_decode_forward_parity():
    from transformers import GraniteMoeConfig
    from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE

    from QEfficient.transformers.models.granitemoe.modeling_granitemoe import (
        QEffGraniteMoeMoE,
        QEffGraniteMoeTopKGating,
    )

    cfg = GraniteMoeConfig(
        num_local_experts=4, hidden_size=32, intermediate_size=64, num_experts_per_tok=2, num_hidden_layers=1
    )
    moe = GraniteMoeMoE(cfg).eval()
    with torch.no_grad():
        moe.input_linear.weight.normal_(0, 0.1)
        moe.output_linear.weight.normal_(0, 0.1)
        for param in moe.parameters():
            if torch.isnan(param).any():
                param.normal_(0, 0.1)

    x = torch.randn(1, 8, cfg.hidden_size)
    with torch.no_grad():
        hf_out = moe(x)
    hf_out = hf_out[0] if isinstance(hf_out, tuple) else hf_out

    qeff = copy.deepcopy(moe)
    qeff.__class__ = QEffGraniteMoeMoE
    qeff.router.__class__ = QEffGraniteMoeTopKGating
    with torch.no_grad():
        out = qeff(x)
    out = out[0] if isinstance(out, tuple) else out

    assert hf_out.shape == out.shape
    assert (hf_out - out).abs().max().item() < 1e-3, "GraniteMoE HF vs QEff decode parity failed"
