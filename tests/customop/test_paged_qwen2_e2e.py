# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""End-to-end parity: QEffQwen2ForCausalLM with paged KV vs contiguous KV.

Runs the SAME transformed Qwen2 model twice — once with the contiguous
continuous-batching cache (no block_table) and once with the paged block-pool
cache (+ block_table) — and asserts the output logits are identical. This
exercises the full block_table threading (ForCausalLM -> Model -> DecoderLayer
-> Attention -> past_key_value_update -> paged cache) on the real model.

Requires the full QEfficient import to be available (CPU). Skips otherwise.
"""

import pytest
import torch

pytest.importorskip("QEfficient")

from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform  # noqa: E402


def _build_model():
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    cfg.torch_dtype = torch.float32
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model, _ = KVCacheTransform.apply(model)
    model.eval()
    return model, cfg


def _make_block_table(bsz, max_blocks, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(max_blocks, generator=g) + b * max_blocks for b in range(bsz)]
    return torch.stack(rows, 0).to(torch.int32)


def _zeros_pkv(n_layers, *shape):
    return [(torch.zeros(*shape), torch.zeros(*shape)) for _ in range(n_layers)]


def test_qwen2_paged_logits_match_contiguous():
    model, cfg = _build_model()
    n_layers = cfg.num_hidden_layers
    kvh = cfg.num_key_value_heads
    hd = cfg.hidden_size // cfg.num_attention_heads

    bsz = 2
    page_size, max_blocks = 8, 4
    ctx_len = page_size * max_blocks  # 32
    num_blocks = bsz * max_blocks + 1  # + null block
    block_table = _make_block_table(bsz, max_blocks, seed=3)
    batch_index = torch.arange(bsz).view(bsz, 1).to(torch.int32)

    prompt_len = 12
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, prompt_len))
    position_ids = torch.arange(prompt_len).view(1, -1).expand(bsz, prompt_len).to(torch.int64)
    attn_mask = torch.ones(bsz, ctx_len)  # only its last-dim size matters (target_length)

    with torch.no_grad():
        cont = model(
            input_ids=input_ids,
            position_ids=position_ids,
            batch_index=batch_index,
            attention_mask=attn_mask,
            past_key_values=_zeros_pkv(n_layers, bsz, kvh, ctx_len, hd),
            use_cache=True,
        )
        paged = model(
            input_ids=input_ids,
            position_ids=position_ids,
            batch_index=batch_index,
            block_table=block_table,
            attention_mask=attn_mask,
            past_key_values=_zeros_pkv(n_layers, num_blocks, kvh, page_size, hd),
            use_cache=True,
        )

    assert cont.logits.shape == paged.logits.shape
    assert torch.allclose(cont.logits, paged.logits, atol=1e-4, rtol=1e-4), (
        f"max abs diff = {(cont.logits - paged.logits).abs().max().item()}"
    )


if __name__ == "__main__":
    test_qwen2_paged_logits_match_contiguous()
    print("QWEN2 PAGED E2E PARITY: PASS")
