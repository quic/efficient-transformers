# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.blocking.attention_blocking import BlockingMode
from QEfficient.customop.ctx_scatter_gather import CtxGatherFunc3D, CtxGatherFunc3DGeneralized
from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import QEffGlm4MoeAttention

MODEL_KWARGS = {"attn_implementation": "eager"}

GLM4_MOE_CFG = dict(
    max_position_embeddings=1024,
    num_hidden_layers=1,
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


def test_glm4_moe_transform_enables_kv_blocking_on_qeff_attention():
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
        assert attn_module.attn_blocking_config.mode == BlockingMode.KV
        assert attn_module.attn_blocking_config.num_kv_blocks == 2


def test_ctx_gather_3d_generalized_keeps_eager_parity_without_data_shaped_symbolic():
    data = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    ctx_indices = torch.tensor([[0, 3], [torch.iinfo(torch.int32).max, 2]], dtype=torch.int32)

    regular = CtxGatherFunc3D.apply(data, ctx_indices)
    generalized = CtxGatherFunc3DGeneralized.apply(data, ctx_indices)

    assert torch.equal(generalized, regular)
    assert ".setTypeAs(data)" in inspect.getsource(CtxGatherFunc3D.symbolic)
    assert ".setTypeAs(data)" not in inspect.getsource(CtxGatherFunc3DGeneralized.symbolic)
