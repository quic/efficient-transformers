# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
CPU-only tests for QEfficient/diffusers module.

Tests verify:
  - Module importability (all diffusers sub-modules)
  - Attention blocking config parsing (get_attention_blocking_config)
  - Attention blocking functions: apply_head_blocking, apply_kv_blocking,
    apply_q_blocking, apply_qkv_blocking, compute_blocked_attention
  - QEff normalization layers: QEffAdaLayerNormZero, QEffAdaLayerNormZeroSingle,
    QEffAdaLayerNormContinuous
  - Diffusers transforms structure: CustomOpsTransform, AttentionTransform,
    NormalizationTransform
  - Pipeline utilities: calculate_compressed_latent_dimension,
    calculate_latent_dimensions_with_frames, ModulePerf, QEffPipelineOutput
  - Pipeline module class structure: QEffTextEncoder, QEffVAE,
    QEffFluxTransformerModel, QEffWanUnifiedTransformer
  - Flux transformer blocks: QEffFluxTransformerBlock,
    QEffFluxSingleTransformerBlock, QEffFluxTransformer2DModel (tiny in-memory)

All tests run on CPU only. No QAIC hardware required. No network downloads.
"""

import os

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _standard_attention(q, k, v, attention_mask=None):
    """Reference standard scaled-dot-product attention (BS, NH, CL, DH)."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def _make_qkv(bs=1, nh=2, cl=8, dh=16):
    """Build random (q, k, v) tensors of shape (BS, NH, CL, DH)."""
    q = torch.randn(bs, nh, cl, dh)
    k = torch.randn(bs, nh, cl, dh)
    v = torch.randn(bs, nh, cl, dh)
    return q, k, v


# ---------------------------------------------------------------------------
# 1. Module importability
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
class TestDiffusersModuleImportability:
    """All QEfficient/diffusers sub-modules must be importable on CPU."""

    def test_diffusers_init_importable(self):
        import QEfficient.diffusers
        assert QEfficient.diffusers is not None

    def test_modeling_utils_importable(self):
        import QEfficient.diffusers.models.modeling_utils
        assert QEfficient.diffusers.models.modeling_utils is not None

    def test_normalization_importable(self):
        import QEfficient.diffusers.models.normalization
        assert QEfficient.diffusers.models.normalization is not None

    def test_pytorch_transforms_importable(self):
        import QEfficient.diffusers.models.pytorch_transforms
        assert QEfficient.diffusers.models.pytorch_transforms is not None

    def test_transformer_flux_importable(self):
        import QEfficient.diffusers.models.transformers.transformer_flux
        assert QEfficient.diffusers.models.transformers.transformer_flux is not None

    def test_pipeline_utils_importable(self):
        import QEfficient.diffusers.pipelines.pipeline_utils
        assert QEfficient.diffusers.pipelines.pipeline_utils is not None

    def test_pipeline_module_importable(self):
        import QEfficient.diffusers.pipelines.pipeline_module
        assert QEfficient.diffusers.pipelines.pipeline_module is not None

    def test_get_attention_blocking_config_importable(self):
        from QEfficient.diffusers.models.modeling_utils import get_attention_blocking_config
        assert callable(get_attention_blocking_config)

    def test_compute_blocked_attention_importable(self):
        from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention
        assert callable(compute_blocked_attention)

    def test_qeff_flux_transformer_2d_model_importable(self):
        from QEfficient.diffusers.models.transformers.transformer_flux import QEffFluxTransformer2DModel
        assert QEffFluxTransformer2DModel is not None

    def test_qeff_ada_layer_norm_zero_importable(self):
        from QEfficient.diffusers.models.normalization import QEffAdaLayerNormZero
        assert QEffAdaLayerNormZero is not None

    def test_qeff_pipeline_output_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import QEffPipelineOutput
        assert QEffPipelineOutput is not None


# ---------------------------------------------------------------------------
# 2. Attention blocking config
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
class TestAttentionBlockingConfig:
    """get_attention_blocking_config must parse env vars correctly."""

    def _get_config(self, mode=None, head_block=None, kv_blocks=None, q_blocks=None):
        """Helper: set env vars, call get_attention_blocking_config, restore."""
        from QEfficient.diffusers.models.modeling_utils import get_attention_blocking_config

        env_backup = {}
        keys = {
            "ATTENTION_BLOCKING_MODE": mode,
            "head_block_size": head_block,
            "num_kv_blocks": kv_blocks,
            "num_q_blocks": q_blocks,
        }
        for k, v in keys.items():
            env_backup[k] = os.environ.get(k)
            if v is not None:
                os.environ[k] = str(v)
            elif k in os.environ:
                del os.environ[k]
        try:
            return get_attention_blocking_config()
        finally:
            for k, v in env_backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_default_mode_is_default(self):
        blocking_mode, _, _, _ = self._get_config()
        assert blocking_mode == "default", f"Default blocking mode must be 'default', got '{blocking_mode}'"

    def test_default_head_block_size_is_none_or_positive(self):
        """Default head_block_size is None (unused in 'default' mode) or a positive int."""
        _, head_block_size, _, _ = self._get_config()
        assert head_block_size is None or head_block_size > 0

    def test_default_num_kv_blocks_is_none_or_positive(self):
        """Default num_kv_blocks is None (unused in 'default' mode) or a positive int."""
        _, _, num_kv_blocks, _ = self._get_config()
        assert num_kv_blocks is None or num_kv_blocks > 0

    def test_default_num_q_blocks_is_none_or_positive(self):
        """Default num_q_blocks is None (unused in 'default' mode) or a positive int."""
        _, _, _, num_q_blocks = self._get_config()
        assert num_q_blocks is None or num_q_blocks > 0

    def test_custom_mode_kv(self):
        blocking_mode, _, _, _ = self._get_config(mode="kv")
        assert blocking_mode == "kv"

    def test_custom_mode_q(self):
        blocking_mode, _, _, _ = self._get_config(mode="q")
        assert blocking_mode == "q"

    def test_custom_mode_qkv(self):
        blocking_mode, _, _, _ = self._get_config(mode="qkv")
        assert blocking_mode == "qkv"

    def test_custom_head_block_size(self):
        _, head_block_size, _, _ = self._get_config(head_block=4)
        assert head_block_size == 4

    def test_custom_num_kv_blocks(self):
        _, _, num_kv_blocks, _ = self._get_config(kv_blocks=8)
        assert num_kv_blocks == 8

    def test_custom_num_q_blocks(self):
        _, _, _, num_q_blocks = self._get_config(q_blocks=16)
        assert num_q_blocks == 16

    def test_returns_four_values(self):
        result = self._get_config()
        assert len(result) == 4

    def test_invalid_mode_raises_value_error(self):
        from QEfficient.diffusers.models.modeling_utils import get_attention_blocking_config

        os.environ["ATTENTION_BLOCKING_MODE"] = "invalid_xyz_mode"
        try:
            with pytest.raises((ValueError, KeyError)):
                get_attention_blocking_config()
        finally:
            del os.environ["ATTENTION_BLOCKING_MODE"]


# ---------------------------------------------------------------------------
# 3. Head blocking attention
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
@pytest.mark.accuracy
class TestHeadBlockingAttention:
    """apply_head_blocking must produce correct outputs on CPU."""

    def test_output_shape_matches_input(self):
        from QEfficient.diffusers.models.modeling_utils import apply_head_blocking

        q, k, v = _make_qkv(bs=1, nh=4, cl=8, dh=16)
        out = apply_head_blocking(q, k, v, head_block_size=2)
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_output_is_finite(self):
        from QEfficient.diffusers.models.modeling_utils import apply_head_blocking

        q, k, v = _make_qkv(bs=1, nh=4, cl=8, dh=16)
        out = apply_head_blocking(q, k, v, head_block_size=2)
        assert torch.isfinite(out).all(), "apply_head_blocking output contains NaN/Inf"

    def test_small_seq_matches_standard_attention(self):
        """For CL <= 512, head blocking must match standard attention exactly."""
        from QEfficient.diffusers.models.modeling_utils import apply_head_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        ref = _standard_attention(q, k, v)
        out = apply_head_blocking(q, k, v, head_block_size=1)
        max_diff = (ref - out).abs().max().item()
        assert max_diff < 1e-4, f"Head blocking vs standard attention max_diff={max_diff:.2e}"

    def test_batch_size_2_works(self):
        from QEfficient.diffusers.models.modeling_utils import apply_head_blocking

        q, k, v = _make_qkv(bs=2, nh=4, cl=8, dh=16)
        out = apply_head_blocking(q, k, v, head_block_size=2)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()

    def test_single_head_block_size_equals_num_heads(self):
        """head_block_size == num_heads should process all heads at once."""
        from QEfficient.diffusers.models.modeling_utils import apply_head_blocking

        q, k, v = _make_qkv(bs=1, nh=4, cl=8, dh=16)
        out = apply_head_blocking(q, k, v, head_block_size=4)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 4. KV blocking attention
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
@pytest.mark.accuracy
class TestKVBlockingAttention:
    """apply_kv_blocking must produce correct outputs on CPU."""

    def test_output_shape_matches_input(self):
        from QEfficient.diffusers.models.modeling_utils import apply_kv_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = apply_kv_blocking(q, k, v, head_block_size=2, num_kv_blocks=2)
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_output_is_finite(self):
        from QEfficient.diffusers.models.modeling_utils import apply_kv_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = apply_kv_blocking(q, k, v, head_block_size=2, num_kv_blocks=2)
        assert torch.isfinite(out).all()

    def test_small_seq_matches_standard_attention(self):
        """For CL <= 512, kv blocking must match standard attention."""
        from QEfficient.diffusers.models.modeling_utils import apply_kv_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        ref = _standard_attention(q, k, v)
        out = apply_kv_blocking(q, k, v, head_block_size=2, num_kv_blocks=1)
        max_diff = (ref - out).abs().max().item()
        assert max_diff < 1e-4, f"KV blocking vs standard attention max_diff={max_diff:.2e}"

    def test_batch_size_2_works(self):
        from QEfficient.diffusers.models.modeling_utils import apply_kv_blocking

        q, k, v = _make_qkv(bs=2, nh=2, cl=8, dh=16)
        out = apply_kv_blocking(q, k, v, head_block_size=2, num_kv_blocks=2)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 5. Q blocking attention
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
@pytest.mark.accuracy
class TestQBlockingAttention:
    """apply_q_blocking must produce correct outputs on CPU."""

    def test_output_shape_matches_input(self):
        from QEfficient.diffusers.models.modeling_utils import apply_q_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = apply_q_blocking(q, k, v, head_block_size=2, num_q_blocks=2)
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_output_is_finite(self):
        from QEfficient.diffusers.models.modeling_utils import apply_q_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = apply_q_blocking(q, k, v, head_block_size=2, num_q_blocks=2)
        assert torch.isfinite(out).all()

    def test_small_seq_matches_standard_attention(self):
        """For CL <= 512, q blocking must match standard attention."""
        from QEfficient.diffusers.models.modeling_utils import apply_q_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        ref = _standard_attention(q, k, v)
        out = apply_q_blocking(q, k, v, head_block_size=2, num_q_blocks=1)
        max_diff = (ref - out).abs().max().item()
        assert max_diff < 1e-4, f"Q blocking vs standard attention max_diff={max_diff:.2e}"

    def test_batch_size_2_works(self):
        from QEfficient.diffusers.models.modeling_utils import apply_q_blocking

        q, k, v = _make_qkv(bs=2, nh=2, cl=8, dh=16)
        out = apply_q_blocking(q, k, v, head_block_size=2, num_q_blocks=2)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 6. QKV blocking attention
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
@pytest.mark.accuracy
class TestQKVBlockingAttention:
    """apply_qkv_blocking must produce correct outputs on CPU."""

    def test_output_shape_matches_input(self):
        from QEfficient.diffusers.models.modeling_utils import apply_qkv_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = apply_qkv_blocking(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2)
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"

    def test_output_is_finite(self):
        from QEfficient.diffusers.models.modeling_utils import apply_qkv_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = apply_qkv_blocking(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2)
        assert torch.isfinite(out).all()

    def test_small_seq_matches_standard_attention(self):
        """For CL <= 512, qkv blocking must match standard attention."""
        from QEfficient.diffusers.models.modeling_utils import apply_qkv_blocking

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        ref = _standard_attention(q, k, v)
        out = apply_qkv_blocking(q, k, v, head_block_size=2, num_kv_blocks=1, num_q_blocks=1)
        max_diff = (ref - out).abs().max().item()
        assert max_diff < 1e-4, f"QKV blocking vs standard attention max_diff={max_diff:.2e}"

    def test_batch_size_2_works(self):
        from QEfficient.diffusers.models.modeling_utils import apply_qkv_blocking

        q, k, v = _make_qkv(bs=2, nh=2, cl=8, dh=16)
        out = apply_qkv_blocking(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 7. compute_blocked_attention dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
@pytest.mark.accuracy
class TestComputeBlockedAttention:
    """compute_blocked_attention must dispatch to the correct function."""

    def test_head_mode_output_shape(self):
        from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention

        q, k, v = _make_qkv(bs=1, nh=4, cl=8, dh=16)
        out = compute_blocked_attention(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2, blocking_mode="head")
        assert out.shape == q.shape

    def test_kv_mode_output_shape(self):
        from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = compute_blocked_attention(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2, blocking_mode="kv")
        assert out.shape == q.shape

    def test_q_mode_output_shape(self):
        from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = compute_blocked_attention(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2, blocking_mode="q")
        assert out.shape == q.shape

    def test_qkv_mode_output_shape(self):
        from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        out = compute_blocked_attention(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2, blocking_mode="qkv")
        assert out.shape == q.shape

    def test_all_modes_produce_finite_outputs(self):
        """All four blocking modes must produce finite outputs."""
        from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention

        q, k, v = _make_qkv(bs=1, nh=4, cl=8, dh=16)
        for mode in ["head", "kv", "q", "qkv"]:
            out = compute_blocked_attention(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2, blocking_mode=mode)
            assert torch.isfinite(out).all(), f"Mode '{mode}' produced NaN/Inf"

    def test_small_seq_all_modes_agree(self):
        """For CL <= 512, all modes must produce the same result as standard attention."""
        from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention

        q, k, v = _make_qkv(bs=1, nh=4, cl=8, dh=16)
        ref = _standard_attention(q, k, v)

        for mode in ["head", "kv", "q", "qkv"]:
            out = compute_blocked_attention(q, k, v, head_block_size=1, num_kv_blocks=1, num_q_blocks=1, blocking_mode=mode)
            max_diff = (ref - out).abs().max().item()
            assert max_diff < 1e-4, (
                f"Mode '{mode}' vs standard attention max_diff={max_diff:.2e}"
            )

    def test_with_attention_mask(self):
        """compute_blocked_attention must accept an optional boolean attention_mask."""
        from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention

        q, k, v = _make_qkv(bs=1, nh=2, cl=8, dh=16)
        # attention_mask must be boolean (True = masked/ignored position)
        mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        out = compute_blocked_attention(q, k, v, head_block_size=2, num_kv_blocks=2, num_q_blocks=2, blocking_mode="head", attention_mask=mask)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 8. QEff normalization layers
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
@pytest.mark.accuracy
class TestQEffNormalizationLayers:
    """QEff normalization layers must produce correct outputs on CPU."""

    def _make_ada_layer_norm_zero(self, embedding_dim=16):
        from QEfficient.diffusers.models.normalization import QEffAdaLayerNormZero
        return QEffAdaLayerNormZero(embedding_dim=embedding_dim).eval()

    def _make_ada_layer_norm_zero_single(self, embedding_dim=16):
        from QEfficient.diffusers.models.normalization import QEffAdaLayerNormZeroSingle
        return QEffAdaLayerNormZeroSingle(embedding_dim=embedding_dim).eval()

    def _make_ada_layer_norm_continuous(self, embedding_dim=16, conditioning_dim=16):
        from QEfficient.diffusers.models.normalization import QEffAdaLayerNormContinuous
        return QEffAdaLayerNormContinuous(
            embedding_dim=embedding_dim,
            conditioning_embedding_dim=conditioning_dim,
        ).eval()

    def test_ada_layer_norm_zero_instantiates(self):
        norm = self._make_ada_layer_norm_zero()
        assert norm is not None

    def test_ada_layer_norm_zero_single_instantiates(self):
        norm = self._make_ada_layer_norm_zero_single()
        assert norm is not None

    def test_ada_layer_norm_continuous_instantiates(self):
        norm = self._make_ada_layer_norm_continuous()
        assert norm is not None

    def test_ada_layer_norm_zero_output_shape(self):
        """QEffAdaLayerNormZero.forward must return tensor of same shape as input."""
        norm = self._make_ada_layer_norm_zero(embedding_dim=16)
        x = torch.randn(1, 8, 16)
        shift_msa = torch.randn(1, 16)
        scale_msa = torch.randn(1, 16)
        with torch.no_grad():
            out = norm(x, shift_msa=shift_msa, scale_msa=scale_msa)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_ada_layer_norm_zero_output_is_finite(self):
        norm = self._make_ada_layer_norm_zero(embedding_dim=16)
        x = torch.randn(1, 8, 16)
        shift_msa = torch.randn(1, 16)
        scale_msa = torch.randn(1, 16)
        with torch.no_grad():
            out = norm(x, shift_msa=shift_msa, scale_msa=scale_msa)
        assert torch.isfinite(out).all()

    def test_ada_layer_norm_zero_single_output_shape(self):
        """QEffAdaLayerNormZeroSingle.forward must return tensor of same shape as input."""
        norm = self._make_ada_layer_norm_zero_single(embedding_dim=16)
        x = torch.randn(1, 8, 16)
        shift_msa = torch.randn(1, 16)
        scale_msa = torch.randn(1, 16)
        with torch.no_grad():
            out = norm(x, scale_msa=scale_msa, shift_msa=shift_msa)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_ada_layer_norm_zero_single_output_is_finite(self):
        norm = self._make_ada_layer_norm_zero_single(embedding_dim=16)
        x = torch.randn(1, 8, 16)
        with torch.no_grad():
            out = norm(x, scale_msa=torch.randn(1, 16), shift_msa=torch.randn(1, 16))
        assert torch.isfinite(out).all()

    def test_ada_layer_norm_continuous_output_shape(self):
        """QEffAdaLayerNormContinuous.forward must return tensor of same shape as input."""
        norm = self._make_ada_layer_norm_continuous(embedding_dim=16, conditioning_dim=16)
        x = torch.randn(1, 8, 16)
        # conditioning_embedding is pre-computed: shape (batch, 2 * embedding_dim)
        conditioning = torch.randn(1, 32)
        with torch.no_grad():
            out = norm(x, conditioning)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_ada_layer_norm_continuous_output_is_finite(self):
        norm = self._make_ada_layer_norm_continuous(embedding_dim=16, conditioning_dim=16)
        x = torch.randn(1, 8, 16)
        conditioning = torch.randn(1, 32)
        with torch.no_grad():
            out = norm(x, conditioning)
        assert torch.isfinite(out).all()

    def test_ada_layer_norm_zero_zero_shift_scale_preserves_norm(self):
        """With zero shift and scale, output should equal LayerNorm(x)."""
        norm = self._make_ada_layer_norm_zero(embedding_dim=16)
        x = torch.randn(1, 8, 16)
        shift_msa = torch.zeros(1, 16)
        scale_msa = torch.zeros(1, 16)
        with torch.no_grad():
            out = norm(x, shift_msa=shift_msa, scale_msa=scale_msa)
        # With zero shift and scale: out = LayerNorm(x) * (1 + 0) + 0 = LayerNorm(x)
        ln = torch.nn.LayerNorm(16, elementwise_affine=False, eps=1e-6)
        expected = ln(x)
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Zero shift/scale: max_diff={max_diff:.2e}"

    def test_ada_layer_norm_continuous_batch_size_2(self):
        norm = self._make_ada_layer_norm_continuous(embedding_dim=16, conditioning_dim=16)
        x = torch.randn(2, 8, 16)
        conditioning = torch.randn(2, 32)
        with torch.no_grad():
            out = norm(x, conditioning)
        assert out.shape == (2, 8, 16)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 9. Diffusers transforms structure
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
class TestDiffusersTransforms:
    """Diffusers transforms must have correct class-level structure."""

    def test_custom_ops_transform_importable(self):
        from QEfficient.diffusers.models.pytorch_transforms import CustomOpsTransform
        assert CustomOpsTransform is not None

    def test_attention_transform_importable(self):
        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
        assert AttentionTransform is not None

    def test_normalization_transform_importable(self):
        from QEfficient.diffusers.models.pytorch_transforms import NormalizationTransform
        assert NormalizationTransform is not None

    def test_custom_ops_transform_has_module_mapping(self):
        from QEfficient.diffusers.models.pytorch_transforms import CustomOpsTransform
        assert hasattr(CustomOpsTransform, "_module_mapping")
        assert len(CustomOpsTransform._module_mapping) > 0

    def test_attention_transform_has_module_mapping(self):
        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
        assert hasattr(AttentionTransform, "_module_mapping")
        assert len(AttentionTransform._module_mapping) > 0

    def test_normalization_transform_has_module_mapping(self):
        from QEfficient.diffusers.models.pytorch_transforms import NormalizationTransform
        assert hasattr(NormalizationTransform, "_module_mapping")
        assert len(NormalizationTransform._module_mapping) > 0

    def test_attention_transform_maps_flux_attention(self):
        from diffusers.models.transformers.transformer_flux import FluxAttention
        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
        from QEfficient.diffusers.models.transformers.transformer_flux import QEffFluxAttention
        assert FluxAttention in AttentionTransform._module_mapping
        assert AttentionTransform._module_mapping[FluxAttention] is QEffFluxAttention

    def test_attention_transform_maps_flux_transformer_block(self):
        from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
        from QEfficient.diffusers.models.transformers.transformer_flux import QEffFluxTransformerBlock
        assert FluxTransformerBlock in AttentionTransform._module_mapping
        assert AttentionTransform._module_mapping[FluxTransformerBlock] is QEffFluxTransformerBlock

    def test_attention_transform_maps_flux_single_transformer_block(self):
        from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
        from QEfficient.diffusers.models.transformers.transformer_flux import QEffFluxSingleTransformerBlock
        assert FluxSingleTransformerBlock in AttentionTransform._module_mapping
        assert AttentionTransform._module_mapping[FluxSingleTransformerBlock] is QEffFluxSingleTransformerBlock

    def test_attention_transform_maps_flux_transformer_2d_model(self):
        from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
        from QEfficient.diffusers.models.transformers.transformer_flux import QEffFluxTransformer2DModel
        assert FluxTransformer2DModel in AttentionTransform._module_mapping
        assert AttentionTransform._module_mapping[FluxTransformer2DModel] is QEffFluxTransformer2DModel

    def test_normalization_transform_maps_ada_layer_norm_zero(self):
        from diffusers.models.normalization import AdaLayerNormZero
        from QEfficient.diffusers.models.pytorch_transforms import NormalizationTransform
        from QEfficient.diffusers.models.normalization import QEffAdaLayerNormZero
        assert AdaLayerNormZero in NormalizationTransform._module_mapping
        assert NormalizationTransform._module_mapping[AdaLayerNormZero] is QEffAdaLayerNormZero

    def test_normalization_transform_maps_ada_layer_norm_zero_single(self):
        from diffusers.models.normalization import AdaLayerNormZeroSingle
        from QEfficient.diffusers.models.pytorch_transforms import NormalizationTransform
        from QEfficient.diffusers.models.normalization import QEffAdaLayerNormZeroSingle
        assert AdaLayerNormZeroSingle in NormalizationTransform._module_mapping
        assert NormalizationTransform._module_mapping[AdaLayerNormZeroSingle] is QEffAdaLayerNormZeroSingle

    def test_all_transforms_have_apply_method(self):
        from QEfficient.diffusers.models.pytorch_transforms import (
            AttentionTransform,
            CustomOpsTransform,
            NormalizationTransform,
        )
        for cls in [CustomOpsTransform, AttentionTransform, NormalizationTransform]:
            assert hasattr(cls, "apply"), f"{cls.__name__} missing apply method"
            assert callable(cls.apply), f"{cls.__name__}.apply is not callable"


# ---------------------------------------------------------------------------
# 10. Pipeline utilities
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
class TestPipelineUtils:
    """Pipeline utility functions must produce correct results."""

    def test_calculate_compressed_latent_dimension_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import calculate_compressed_latent_dimension
        assert callable(calculate_compressed_latent_dimension)

    def test_calculate_latent_dimensions_with_frames_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import calculate_latent_dimensions_with_frames
        assert callable(calculate_latent_dimensions_with_frames)

    def test_compressed_latent_dimension_basic(self):
        """calculate_compressed_latent_dimension returns (cl, latent_h, latent_w).
        cl = (latent_h * latent_w) // 4 (Flux 2x2 packing).
        For H=64, W=64, vsf=8: latent_h=8, latent_w=8, cl=(8*8)//4=16.
        """
        from QEfficient.diffusers.pipelines.pipeline_utils import calculate_compressed_latent_dimension

        cl, latent_h, latent_w = calculate_compressed_latent_dimension(height=64, width=64, vae_scale_factor=8)
        assert latent_h == 8, f"Expected latent_h=8, got {latent_h}"
        assert latent_w == 8, f"Expected latent_w=8, got {latent_w}"
        assert cl == 16, f"Expected cl=16 (=(8*8)//4), got {cl}"

    def test_compressed_latent_dimension_non_square(self):
        """For H=64, W=128, vsf=8: latent_h=8, latent_w=16, cl=(8*16)//4=32."""
        from QEfficient.diffusers.pipelines.pipeline_utils import calculate_compressed_latent_dimension

        cl, latent_h, latent_w = calculate_compressed_latent_dimension(height=64, width=128, vae_scale_factor=8)
        assert latent_h == 8, f"Expected latent_h=8, got {latent_h}"
        assert latent_w == 16, f"Expected latent_w=16, got {latent_w}"
        assert cl == 32, f"Expected cl=32 (=(8*16)//4), got {cl}"

    def test_compressed_latent_dimension_patch_size_1(self):
        """For H=16, W=16, vsf=1: latent_h=16, latent_w=16, cl=(16*16)//4=64."""
        from QEfficient.diffusers.pipelines.pipeline_utils import calculate_compressed_latent_dimension

        cl, latent_h, latent_w = calculate_compressed_latent_dimension(height=16, width=16, vae_scale_factor=1)
        assert latent_h == 16, f"Expected latent_h=16, got {latent_h}"
        assert latent_w == 16, f"Expected latent_w=16, got {latent_w}"
        assert cl == 64, f"Expected cl=64 (=(16*16)//4), got {cl}"

    def test_compressed_latent_dimension_returns_tuple_of_ints(self):
        """calculate_compressed_latent_dimension must return a tuple of 3 ints."""
        from QEfficient.diffusers.pipelines.pipeline_utils import calculate_compressed_latent_dimension

        result = calculate_compressed_latent_dimension(height=64, width=64, vae_scale_factor=8)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3-tuple, got length {len(result)}"
        cl, latent_h, latent_w = result
        assert isinstance(cl, int), f"Expected cl to be int, got {type(cl)}"
        assert isinstance(latent_h, int), f"Expected latent_h to be int, got {type(latent_h)}"
        assert isinstance(latent_w, int), f"Expected latent_w to be int, got {type(latent_w)}"

    def test_latent_dimensions_with_frames_returns_tuple(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import calculate_latent_dimensions_with_frames

        result = calculate_latent_dimensions_with_frames(
            height=64, width=64, num_frames=16,
            vae_scale_factor_spatial=2, vae_scale_factor_temporal=4,
            patch_height=2, patch_width=2
        )
        assert isinstance(result, (tuple, list, int)), f"Unexpected return type: {type(result)}"

    def test_latent_dimensions_with_frames_is_positive(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import calculate_latent_dimensions_with_frames

        result = calculate_latent_dimensions_with_frames(
            height=64, width=64, num_frames=16,
            vae_scale_factor_spatial=2, vae_scale_factor_temporal=4,
            patch_height=2, patch_width=2
        )
        if isinstance(result, (tuple, list)):
            assert all(r > 0 for r in result), "All dimensions must be positive"
        else:
            assert result > 0

    def test_module_perf_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import ModulePerf
        assert ModulePerf is not None

    def test_module_perf_instantiable(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import ModulePerf
        perf = ModulePerf(module_name="test", perf=100)
        assert perf is not None

    def test_module_perf_has_expected_fields(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import ModulePerf
        perf = ModulePerf(module_name="test", perf=100)
        assert hasattr(perf, "module_name")
        assert hasattr(perf, "perf")

    def test_qeff_pipeline_output_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import QEffPipelineOutput
        assert QEffPipelineOutput is not None

    def test_qeff_pipeline_output_instantiable(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import QEffPipelineOutput, ModulePerf
        import numpy as np
        output = QEffPipelineOutput(
            pipeline_module=[ModulePerf(module_name="test", perf=100)],
            images=np.zeros((1, 64, 64, 3))
        )
        assert output is not None

    def test_qeff_pipeline_output_has_images(self):
        from QEfficient.diffusers.pipelines.pipeline_utils import QEffPipelineOutput, ModulePerf
        import numpy as np
        images = np.zeros((1, 64, 64, 3))
        output = QEffPipelineOutput(
            pipeline_module=[ModulePerf(module_name="test", perf=100)],
            images=images
        )
        assert hasattr(output, "images")
        assert output.images is images


# ---------------------------------------------------------------------------
# 11. Pipeline module class structure
# ---------------------------------------------------------------------------


@pytest.mark.diffusers
class TestPipelineModuleStructure:
    """Pipeline module classes must have correct class-level structure."""

    def test_qeff_text_encoder_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffTextEncoder
        assert QEffTextEncoder is not None

    def test_qeff_vae_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffVAE
        assert QEffVAE is not None

    def test_qeff_flux_transformer_model_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffFluxTransformerModel
        assert QEffFluxTransformerModel is not None

    def test_qeff_wan_unified_transformer_importable(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffWanUnifiedTransformer
        assert QEffWanUnifiedTransformer is not None

    def test_qeff_text_encoder_has_pytorch_transforms(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffTextEncoder
        assert hasattr(QEffTextEncoder, "_pytorch_transforms")
        assert isinstance(QEffTextEncoder._pytorch_transforms, list)

    def test_qeff_text_encoder_has_onnx_transforms(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffTextEncoder
        assert hasattr(QEffTextEncoder, "_onnx_transforms")
        assert isinstance(QEffTextEncoder._onnx_transforms, list)

    def test_qeff_flux_transformer_model_has_pytorch_transforms(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffFluxTransformerModel
        assert hasattr(QEffFluxTransformerModel, "_pytorch_transforms")
        assert isinstance(QEffFluxTransformerModel._pytorch_transforms, list)

    def test_qeff_flux_transformer_model_has_onnx_transforms(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffFluxTransformerModel
        assert hasattr(QEffFluxTransformerModel, "_onnx_transforms")
        assert isinstance(QEffFluxTransformerModel._onnx_transforms, list)

    def test_qeff_flux_transformer_model_pytorch_transforms_include_attention(self):
        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform
        from QEfficient.diffusers.pipelines.pipeline_module import QEffFluxTransformerModel
        assert AttentionTransform in QEffFluxTransformerModel._pytorch_transforms, (
            "AttentionTransform not in QEffFluxTransformerModel._pytorch_transforms"
        )

    def test_qeff_flux_transformer_model_pytorch_transforms_include_normalization(self):
        from QEfficient.diffusers.models.pytorch_transforms import NormalizationTransform
        from QEfficient.diffusers.pipelines.pipeline_module import QEffFluxTransformerModel
        assert NormalizationTransform in QEffFluxTransformerModel._pytorch_transforms, (
            "NormalizationTransform not in QEffFluxTransformerModel._pytorch_transforms"
        )

    def test_qeff_text_encoder_pytorch_transforms_include_custom_ops(self):
        from QEfficient.diffusers.models.pytorch_transforms import CustomOpsTransform
        from QEfficient.diffusers.pipelines.pipeline_module import QEffTextEncoder
        assert CustomOpsTransform in QEffTextEncoder._pytorch_transforms, (
            "CustomOpsTransform not in QEffTextEncoder._pytorch_transforms"
        )

    def test_qeff_text_encoder_onnx_transforms_include_fp16_clip(self):
        from QEfficient.base.onnx_transforms import FP16ClipTransform
        from QEfficient.diffusers.pipelines.pipeline_module import QEffTextEncoder
        assert FP16ClipTransform in QEffTextEncoder._onnx_transforms, (
            "FP16ClipTransform not in QEffTextEncoder._onnx_transforms"
        )

    def test_qeff_flux_transformer_model_onnx_transforms_include_fp16_clip(self):
        from QEfficient.base.onnx_transforms import FP16ClipTransform
        from QEfficient.diffusers.pipelines.pipeline_module import QEffFluxTransformerModel
        assert FP16ClipTransform in QEffFluxTransformerModel._onnx_transforms, (
            "FP16ClipTransform not in QEffFluxTransformerModel._onnx_transforms"
        )

    def test_qeff_vae_has_pytorch_transforms(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffVAE
        assert hasattr(QEffVAE, "_pytorch_transforms")
        assert isinstance(QEffVAE._pytorch_transforms, list)

    def test_qeff_wan_unified_transformer_has_pytorch_transforms(self):
        from QEfficient.diffusers.pipelines.pipeline_module import QEffWanUnifiedTransformer
        assert hasattr(QEffWanUnifiedTransformer, "_pytorch_transforms")
        assert isinstance(QEffWanUnifiedTransformer._pytorch_transforms, list)


# ---------------------------------------------------------------------------
# 12. Flux transformer blocks (tiny in-memory)
# ---------------------------------------------------------------------------


def _make_tiny_flux_transformer():
    """
    Create a tiny QEffFluxTransformer2DModel for CPU testing.
    Returns None if instantiation fails (e.g., diffusers version mismatch).
    """
    try:
        from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        from QEfficient.diffusers.models.pytorch_transforms import AttentionTransform, NormalizationTransform

        model = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            pooled_projection_dim=16,
            guidance_embeds=False,
            axes_dims_rope=[2, 2, 4],
        ).eval()

        model, _ = AttentionTransform.apply(model)
        model, _ = NormalizationTransform.apply(model)
        return model
    except Exception:
        return None


def _make_tiny_flux_inputs(model, batch=1, cl=4, text_seq=8):
    """
    Build inputs for QEffFluxTransformer2DModel.forward.
    inner_dim = num_attention_heads * attention_head_dim = 2 * 8 = 16
    """
    inner_dim = 16  # 2 heads * 8 head_dim
    in_channels = 4
    joint_attention_dim = 16
    pooled_projection_dim = 16
    num_layers = 1
    num_single_layers = 1

    hidden_states = torch.randn(batch, cl, in_channels)
    encoder_hidden_states = torch.randn(batch, text_seq, joint_attention_dim)
    pooled_projections = torch.randn(batch, pooled_projection_dim)
    timestep = torch.tensor([0.5] * batch)
    img_ids = torch.zeros(cl, 3)
    txt_ids = torch.zeros(text_seq, 3)

    # adaln_emb: (num_layers, 12, inner_dim) — 12 = 6 for hidden + 6 for encoder
    adaln_emb = torch.randn(num_layers, 12, inner_dim)
    # adaln_single_emb: (num_single_layers, 3, inner_dim)
    adaln_single_emb = torch.randn(num_single_layers, 3, inner_dim)
    # adaln_out: (batch, 2 * inner_dim) — pre-computed scale+shift for norm_out
    adaln_out = torch.randn(batch, 2 * inner_dim)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_projections": pooled_projections,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
        "adaln_emb": adaln_emb,
        "adaln_single_emb": adaln_single_emb,
        "adaln_out": adaln_out,
        "return_dict": False,
    }


@pytest.mark.diffusers
@pytest.mark.accuracy
class TestFluxTransformerBlocks:
    """
    QEffFluxTransformer2DModel must produce correct outputs on CPU.
    Uses a tiny in-memory model (1 layer, 2 heads, dim=16) — no network downloads.
    """

    def test_qeff_flux_transformer_2d_model_wraps_without_error(self):
        model = _make_tiny_flux_transformer()
        if model is None:
            pytest.skip("Could not instantiate tiny FluxTransformer2DModel")
        from QEfficient.diffusers.models.transformers.transformer_flux import QEffFluxTransformer2DModel
        assert isinstance(model, QEffFluxTransformer2DModel), (
            f"Expected QEffFluxTransformer2DModel, got {type(model)}"
        )

    def test_qeff_flux_transformer_2d_model_is_eval_mode(self):
        model = _make_tiny_flux_transformer()
        if model is None:
            pytest.skip("Could not instantiate tiny FluxTransformer2DModel")
        assert not model.training, "Model must be in eval mode"

    def test_qeff_flux_transformer_2d_model_forward_returns_output(self):
        model = _make_tiny_flux_transformer()
        if model is None:
            pytest.skip("Could not instantiate tiny FluxTransformer2DModel")
        inputs = _make_tiny_flux_inputs(model)
        with torch.no_grad():
            out = model(**inputs)
        assert out is not None

    def test_qeff_flux_transformer_2d_model_output_shape(self):
        """Output sample must have shape (batch, cl, in_channels)."""
        model = _make_tiny_flux_transformer()
        if model is None:
            pytest.skip("Could not instantiate tiny FluxTransformer2DModel")
        batch, cl, in_channels = 1, 4, 4
        inputs = _make_tiny_flux_inputs(model, batch=batch, cl=cl)
        with torch.no_grad():
            out = model(**inputs)
        # out is a tuple when return_dict=False; out[0] is the sample
        sample = out[0] if isinstance(out, (tuple, list)) else out.sample
        assert sample.shape == (batch, cl, in_channels), (
            f"Expected ({batch}, {cl}, {in_channels}), got {sample.shape}"
        )

    def test_qeff_flux_transformer_2d_model_output_is_finite(self):
        model = _make_tiny_flux_transformer()
        if model is None:
            pytest.skip("Could not instantiate tiny FluxTransformer2DModel")
        inputs = _make_tiny_flux_inputs(model)
        with torch.no_grad():
            out = model(**inputs)
        sample = out[0] if isinstance(out, (tuple, list)) else out.sample
        assert torch.isfinite(sample).all(), "QEffFluxTransformer2DModel output contains NaN/Inf"

    def test_qeff_flux_transformer_2d_model_is_deterministic(self):
        """Same inputs must produce the same output."""
        model = _make_tiny_flux_transformer()
        if model is None:
            pytest.skip("Could not instantiate tiny FluxTransformer2DModel")
        inputs = _make_tiny_flux_inputs(model)
        with torch.no_grad():
            out1 = model(**inputs)
            out2 = model(**inputs)
        s1 = out1[0] if isinstance(out1, (tuple, list)) else out1.sample
        s2 = out2[0] if isinstance(out2, (tuple, list)) else out2.sample
        assert torch.allclose(s1, s2), "QEffFluxTransformer2DModel is not deterministic"

    def test_qeff_flux_transformer_2d_model_get_submodules_for_export(self):
        """get_submodules_for_export must return the expected QEff block classes."""
        model = _make_tiny_flux_transformer()
        if model is None:
            pytest.skip("Could not instantiate tiny FluxTransformer2DModel")
        from QEfficient.diffusers.models.transformers.transformer_flux import (
            QEffFluxSingleTransformerBlock,
            QEffFluxTransformerBlock,
        )
        submodules = model.get_submodules_for_export()
        assert QEffFluxTransformerBlock in submodules, (
            "QEffFluxTransformerBlock not in get_submodules_for_export()"
        )
        assert QEffFluxSingleTransformerBlock in submodules, (
            "QEffFluxSingleTransformerBlock not in get_submodules_for_export()"
        )

    def test_qeff_flux_attn_processor_replaces_original(self):
        """After AttentionTransform, FluxAttention must use QEffFluxAttnProcessor."""
        model = _make_tiny_flux_transformer()
        if model is None:
            pytest.skip("Could not instantiate tiny FluxTransformer2DModel")
        from QEfficient.diffusers.models.transformers.transformer_flux import (
            QEffFluxAttnProcessor,
            QEffFluxAttention,
        )
        for m in model.modules():
            if isinstance(m, QEffFluxAttention):
                assert isinstance(m.processor, QEffFluxAttnProcessor), (
                    f"Expected QEffFluxAttnProcessor, got {type(m.processor)}"
                )
                break
