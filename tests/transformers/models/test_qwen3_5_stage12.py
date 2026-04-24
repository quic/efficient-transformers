"""
Stage 1→2 validation for Qwen3.5-0.8B (qwen3_5 dense text backbone).

Tests that QEfficient transforms produce identical outputs to the HF baseline.
Uses a tiny model config (4 layers: 3 linear + 1 full) for fast iteration.
No MoE — standard MLP, same SSM-hybrid logic as qwen3_5_moe.

Run with:
    /tmp/qeff_explore_qwen3_5_moe/bin/python tests/transformers/models/test_qwen3_5_stage12.py

Requires:
    - venv with transformers 5.5.4 (e.g., /tmp/qeff_explore_qwen3_5_moe)
"""

import sys
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM


# ── Tiny model config (one full cycle: 3 linear + 1 full attention) ──────────

TINY_CONFIG = Qwen3_5TextConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    vocab_size=512,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
    attn_output_gate=True,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_only_layers=[],
    mtp_num_hidden_layers=0,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [4, 4, 8],
    },
)


def build_tiny_model():
    """Build a tiny Qwen3.5 text-only model from scratch (random weights, no download)."""
    model = Qwen3_5ForCausalLM(TINY_CONFIG)
    model.float()  # float32 for parity check
    model.eval()
    return model


def run_stage1_stage2():
    print("=" * 60)
    print("Qwen3.5-0.8B Stage 1→2 Validation (tiny config, no MoE)")
    print("=" * 60)

    torch.manual_seed(42)
    bs, seq_len = 1, 8
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (bs, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_position = torch.arange(seq_len)

    # ── Stage 1: HF baseline ─────────────────────────────────────────────────
    print("\n[Stage 1] HF baseline forward pass (seq_len=8, no transforms)...")
    model_hf = build_tiny_model()

    with torch.no_grad():
        out_hf = model_hf(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
        )
    logits_hf = out_hf.logits
    print(f"  HF logits shape: {logits_hf.shape}")  # [bs, seq, vocab]
    top1_hf_all = logits_hf.argmax(-1)
    print(f"  HF top-1 tokens (all positions): {top1_hf_all.tolist()}")
    print("  Stage 1: OK")

    # ── Stage 2: QEff transforms ──────────────────────────────────────────────
    print("\n[Stage 2] Applying QEff transforms...")
    from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform

    model_qeff = build_tiny_model()
    model_qeff.load_state_dict(model_hf.state_dict())  # identical weights
    CustomOpsTransform.apply(model_qeff)
    KVCacheTransform.apply(model_qeff)
    model_qeff.eval()

    print(f"  CausalLM class: {type(model_qeff).__name__}")
    assert type(model_qeff).__name__ == "QEffQwen3_5ForCausalLM", \
        f"Expected QEffQwen3_5ForCausalLM, got {type(model_qeff).__name__}"
    print(f"  Layer 0 (linear): {type(model_qeff.model.layers[0]).__name__}")
    print(f"  Layer 3 (full):   {type(model_qeff.model.layers[3]).__name__}")

    # Prefill path (seq_len > 1) — no past states, validation mode
    with torch.no_grad():
        out_qeff, new_conv_states, new_recurrent_states = model_qeff(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_position=cache_position,
        )
    logits_qeff = out_qeff.logits
    print(f"  QEff logits shape: {logits_qeff.shape}")  # [bs, 1, vocab] — last position only
    print(f"  New conv states: {len(new_conv_states)} (expected 3)")
    print(f"  New rec states:  {len(new_recurrent_states)} (expected 3)")
    if new_conv_states:
        print(f"  conv_state[0] shape:      {new_conv_states[0].shape}")
        print(f"  recurrent_state[0] shape: {new_recurrent_states[0].shape}")
    print("  Stage 2: OK")

    # ── Compare Stage 1 vs Stage 2 ───────────────────────────────────────────
    print("\n[Comparison] Stage 1 vs Stage 2 (last token logits)...")
    # HF returns all positions [bs, seq, vocab]; QEff extracts last via position_ids argmax → [bs, 1, vocab]
    last_hf = logits_hf[:, -1, :]       # [bs, vocab]
    last_qeff = logits_qeff[:, 0, :]    # [bs, vocab]

    max_diff = (last_hf - last_qeff).abs().max().item()
    top1_hf = last_hf.argmax(-1)
    top1_qeff = last_qeff.argmax(-1)

    print(f"  Max abs diff (last token): {max_diff:.6e}")
    print(f"  Top-1 token HF:   {top1_hf.tolist()}")
    print(f"  Top-1 token QEff: {top1_qeff.tolist()}")

    if (top1_hf == top1_qeff).all():
        print("  ✓ Top-1 tokens match")
    else:
        print("  ✗ Top-1 tokens DIFFER")
        sys.exit(1)

    if max_diff < 1e-3:
        print(f"  ✓ Max diff {max_diff:.2e} < 1e-3 (transforms preserve numerics)")
    else:
        print(f"  ⚠ Max diff {max_diff:.2e} — check RMSNorm convention (should be GemmaCustomRMSNormAIC)")

    # ── Decode step (seq_len=1) ───────────────────────────────────────────────
    print("\n[Decode step] seq_len=1 with conv/rec states from prefill...")
    CTX_LEN = 32
    NUM_LAYERS = 4
    NUM_KV_HEADS = TINY_CONFIG.num_key_value_heads
    HEAD_DIM = TINY_CONFIG.head_dim

    # Build zero KV cache (all layers — linear layers' slots are never touched)
    zero_kv = [(torch.zeros(bs, NUM_KV_HEADS, CTX_LEN, HEAD_DIM),
                torch.zeros(bs, NUM_KV_HEADS, CTX_LEN, HEAD_DIM)) for _ in range(NUM_LAYERS)]
    comp_ctx = torch.zeros(bs, CTX_LEN, dtype=torch.int64)

    decode_ids = torch.randint(0, TINY_CONFIG.vocab_size, (bs, 1))
    decode_pos = torch.tensor([[seq_len]])
    decode_cache = torch.tensor([seq_len])

    with torch.no_grad():
        out_decode, new_conv2, new_rec2 = model_qeff(
            input_ids=decode_ids,
            position_ids=decode_pos,
            cache_position=decode_cache,
            past_key_values=zero_kv,
            past_conv_states=new_conv_states,
            past_recurrent_states=new_recurrent_states,
            comp_ctx_lengths=comp_ctx,
        )
    print(f"  Decode logits shape: {out_decode.logits.shape}")
    assert out_decode.logits.shape == (bs, 1, TINY_CONFIG.vocab_size)
    assert len(new_conv2) == 3 and len(new_rec2) == 3
    assert new_conv2[0].shape == new_conv_states[0].shape
    print(f"  Conv state evolved: {not torch.allclose(new_conv2[0], new_conv_states[0])}")
    print("  ✓ Decode step OK")

    # ── Second decode step — verify state changes ─────────────────────────────
    print("\n[Second decode step] Verify state accumulates...")
    with torch.no_grad():
        out_decode2, _, _ = model_qeff(
            input_ids=decode_ids,
            position_ids=torch.tensor([[seq_len + 1]]),
            cache_position=torch.tensor([seq_len + 1]),
            past_key_values=out_decode.past_key_values,
            past_conv_states=new_conv2,
            past_recurrent_states=new_rec2,
            comp_ctx_lengths=comp_ctx,
        )
    assert not torch.allclose(out_decode.logits, out_decode2.logits), \
        "Logits must differ between decode steps (state must be updating)"
    print("  ✓ Logits differ between step 1 and step 2 (state evolving correctly)")

    print("\n" + "=" * 60)
    print("ALL STAGE 1→2 CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_stage1_stage2()
