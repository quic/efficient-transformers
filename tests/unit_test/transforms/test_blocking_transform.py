# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Unit tests for BlockingAttentionTransform in QEfficient.transformers.models.pytorch_transforms.

Verifies that:
  1. BlockingAttentionTransform attaches attn_blocking_config to QEff attention modules
  2. Returns (model, True) when QEff attention modules are present
  3. Works correctly for all supported model families
  4. Works with different BlockingMode values
  5. Re-applying overrides the previous config

All tests run on CPU only, using tiny in-memory models.
KVCacheTransform must be applied before BlockingAttentionTransform because the
blocking transform matches against QEff attention class types (the *values* of
KVCacheTransform._module_mapping), not the raw HF attention class types (the keys).
"""

import pytest

from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode

VOCAB_SIZE = 500
CTX_LEN = 32


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def make_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return LlamaForCausalLM(cfg).eval()


def make_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return Qwen3ForCausalLM(cfg).eval()


def make_tiny_qwen3_vl():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    text_cfg = Qwen3VLTextConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    vision_cfg = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=64,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    cfg = Qwen3VLConfig(text_config=text_cfg, vision_config=vision_cfg)
    return Qwen3VLForConditionalGeneration(cfg).eval()


def make_tiny_gpt_oss():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    cfg = GptOssConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=64,
        head_dim=32,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=CTX_LEN,
        rope_parameters={"rope_type": "default"},
    )
    return GptOssForCausalLM(cfg).eval()


def make_tiny_gemma():
    from transformers import GemmaConfig, GemmaForCausalLM

    cfg = GemmaConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
    )
    return GemmaForCausalLM(cfg).eval()


def make_tiny_gemma2():
    from transformers import Gemma2Config, Gemma2ForCausalLM

    cfg = Gemma2Config(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
        head_dim=32,
        sliding_window=CTX_LEN,
    )
    return Gemma2ForCausalLM(cfg).eval()


def make_tiny_mistral():
    from transformers import MistralConfig, MistralForCausalLM

    cfg = MistralConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=CTX_LEN,
    )
    return MistralForCausalLM(cfg).eval()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_FACTORIES = [
    (make_tiny_llama, "llama"),
    (make_tiny_qwen3, "qwen3"),
    (make_tiny_qwen3_vl, "qwen3_vl"),
    (make_tiny_gpt_oss, "gpt_oss"),
    (make_tiny_gemma, "gemma"),
    (make_tiny_gemma2, "gemma2"),
    (make_tiny_mistral, "mistral"),
]
_MODEL_IDS = [label for _, label in _MODEL_FACTORIES]


def _qeff_attention_modules(model):
    """Return all modules whose type is in KVCacheTransform supported attention classes."""
    from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

    supported = {
        qeff_cls for qeff_cls in KVCacheTransform._module_mapping.values() if qeff_cls.__name__.endswith("Attention")
    }
    return [m for m in model.modules() if type(m) in supported]


def _blocking_cfg(**kwargs):
    return AttentionBlockingConfig(**kwargs)


# ---------------------------------------------------------------------------
# Tests: Blocking Transform Importability
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingTransformImportability:
    """All quantization transforms must be importable and have correct structure."""

    def test_blocking_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform

        assert BlockingAttentionTransform is not None

    # kv cache transform must happen before blocking transform
    def test_kv_cache_transform_importable(self):
        from QEfficient.transformers.models.pytorch_transforms import KVCacheTransform

        assert KVCacheTransform is not None


# ---------------------------------------------------------------------------
# Tests: basic application per model family
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingTransformApplied:
    """BlockingAttentionTransform must attach attn_blocking_config to all QEff attention modules."""

    @pytest.mark.parametrize("make_model,label", _MODEL_FACTORIES, ids=_MODEL_IDS)
    def test_returns_transformed_true(self, make_model, label):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_model()
        model, _ = KVCacheTransform.apply(model)
        config = _blocking_cfg(mode=BlockingMode.KV, num_kv_blocks=2)
        _, transformed = BlockingAttentionTransform.apply(model, config)
        assert transformed, f"[{label}] BlockingAttentionTransform must return transformed=True"

    @pytest.mark.parametrize("make_model,label", _MODEL_FACTORIES, ids=_MODEL_IDS)
    def test_config_attached_to_all_attn_modules(self, make_model, label):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_model()
        model, _ = KVCacheTransform.apply(model)
        config = _blocking_cfg(mode=BlockingMode.KV, num_kv_blocks=2)
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        attn_mods = _qeff_attention_modules(model)
        assert attn_mods, f"[{label}] no QEff attention modules found after KVCacheTransform"

        for m in attn_mods:
            assert hasattr(m, "attn_blocking_config"), f"[{label}] {type(m).__name__} missing attn_blocking_config"
            assert m.attn_blocking_config is config, (
                f"[{label}] attn_blocking_config must be the same object that was passed in"
            )


# ---------------------------------------------------------------------------
# Tests: blocking modes
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingModes:
    """BlockingAttentionTransform must preserve the BlockingMode in the attached config."""

    @pytest.mark.parametrize(
        "mode",
        [BlockingMode.NONE, BlockingMode.KV, BlockingMode.Q, BlockingMode.H, BlockingMode.QKV],
    )
    def test_blocking_mode_preserved_on_llama(self, mode):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)
        config = AttentionBlockingConfig(mode=mode, head_block_size=8, num_kv_blocks=2, num_q_blocks=2)
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        for m in _qeff_attention_modules(model):
            assert m.attn_blocking_config.mode == mode, f"Expected mode={mode}, got {m.attn_blocking_config.mode}"

    def test_all_config_fields_preserved(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)
        config = AttentionBlockingConfig(
            mode=BlockingMode.KV,
            num_kv_blocks=4,
            num_q_blocks=2,
            head_block_size=8,
            skip_kv=False,
            num_batch_blocks=1,
        )
        model, transformed = BlockingAttentionTransform.apply(model, config)

        assert transformed
        for m in _qeff_attention_modules(model):
            c = m.attn_blocking_config
            assert c.mode == BlockingMode.KV
            assert c.num_kv_blocks == 4
            assert c.num_q_blocks == 2
            assert c.head_block_size == 8
            assert c.skip_kv is False
            assert c.num_batch_blocks == 1


# ---------------------------------------------------------------------------
# Tests: re-application overrides the previous config
# ---------------------------------------------------------------------------


@pytest.mark.transforms
class TestBlockingTransformIdempotent:
    """Applying BlockingAttentionTransform twice must replace the first config with the second."""

    def test_second_apply(self):
        from QEfficient.transformers.models.pytorch_transforms import BlockingAttentionTransform, KVCacheTransform

        model = make_tiny_llama()
        model, _ = KVCacheTransform.apply(model)

        config1 = AttentionBlockingConfig(mode=BlockingMode.KV, num_kv_blocks=2)
        config2 = AttentionBlockingConfig(mode=BlockingMode.Q, num_q_blocks=4)

        model, _ = BlockingAttentionTransform.apply(model, config1)
        model, transformed = BlockingAttentionTransform.apply(model, config2)

        assert transformed, "Reapplication of BlockingAttentionTransform did not succeed"

        for m in _qeff_attention_modules(model):
            assert m.attn_blocking_config is config2, (
                "Second BlockingAttentionTransform.apply must override the first config"
            )
            assert m.attn_blocking_config.mode == BlockingMode.Q
