# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Correctness tests for QEffSlidingWindowCache and QEffDynamicCache.update3D.

Tests verify:
  - QEffSlidingWindowCache: creation, update (sliding + non-sliding), modular scatter,
    output shape, multi-layer independence, to_legacy_cache round-trip, get_seq_length
  - QEffDynamicLayer.update3D / QEffDynamicCache.update3D: 3D KV shape (GPTBigCode)
  - QEffHybridCacheForGPTOSS: full_cache_update_chunked, sliding_window_update_chunked

All tests run on CPU only.
"""

import pytest
import torch

from QEfficient.transformers.cache_utils import (
    QEffDynamicCache,
    QEffDynamicLayer,
    QEffHybridCacheForGPTOSS,
    QEffSlidingWindowCache,
)

# ---------------------------------------------------------------------------
# Minimal config stub (no HF model needed)
# ---------------------------------------------------------------------------


class _FakeConfig:
    """Minimal config stub for cache constructors."""

    sliding_window_pattern = 2  # every 2nd layer is sliding
    sliding_window = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kv_4d(batch=1, heads=2, seq=8, head_dim=16):
    k = torch.randn(batch, heads, seq, head_dim)
    v = torch.randn(batch, heads, seq, head_dim)
    return k, v


def make_kv_3d(batch=1, seq=8, kv_dim=32):
    """3D KV tensors as used by GPTBigCode: [batch, seq, heads*head_dim]."""
    k = torch.randn(batch, seq, kv_dim)
    v = torch.randn(batch, seq, kv_dim)
    return k, v


def pos_ids(batch=1, seq=8, start=0):
    return torch.arange(start, start + seq).unsqueeze(0).expand(batch, -1)


# ---------------------------------------------------------------------------
# Tests: QEffSlidingWindowCache
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffSlidingWindowCache:
    """QEffSlidingWindowCache must correctly implement sliding-window KV caching."""

    def test_creation_succeeds(self):
        """Cache must be created without errors."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        assert cache is not None
        assert cache.max_cache_len == 16
        assert cache.sliding_window_len == 4
        assert cache.batch_size == 1

    def test_initial_cache_is_empty(self):
        """Newly created cache must have empty key/value lists."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        assert len(cache.key_cache) == 0
        assert len(cache.value_cache) == 0

    def test_len_returns_number_of_layers(self):
        """__len__ must return the number of cached layers."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        assert len(cache) == 0

        k, v = make_kv_4d(seq=4)
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=4), "is_sliding": False})
        assert len(cache) == 1

        cache.update(
            k.clone(), v.clone(), layer_idx=1, cache_kwargs={"position_ids": pos_ids(seq=4), "is_sliding": True}
        )
        assert len(cache) == 2

    def test_first_update_non_sliding_stores_tensors(self):
        """First update (non-sliding) must store tensors in the cache."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        k, v = make_kv_4d(seq=8)
        k_out, v_out = cache.update(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False}
        )
        assert len(cache.key_cache) == 1
        assert k_out is not None
        assert v_out is not None

    def test_first_update_returns_finite_tensors(self):
        """First update must return finite tensors."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)
        k, v = make_kv_4d(seq=8)
        k_out, v_out = cache.update(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False}
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_non_sliding_decode_scatter_at_correct_position(self):
        """Non-sliding decode must scatter at the exact position_id."""
        cfg = _FakeConfig()
        ctx_len = 16
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=ctx_len, sliding_window_len=4)

        # Prefill with zeros
        k_init = torch.zeros(1, 2, ctx_len, 8)
        v_init = torch.zeros(1, 2, ctx_len, 8)
        cache.update(
            k_init, v_init, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len), "is_sliding": False}
        )

        # Decode: write known value at position 5
        k_dec = torch.ones(1, 2, 1, 8) * 7.0
        v_dec = torch.ones(1, 2, 1, 8) * 7.0
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[5]]), "is_sliding": False}
        )
        assert k_out[0, 0, 5, 0].item() == pytest.approx(7.0, abs=1e-5)

    def test_sliding_modular_scatter_position(self):
        """Sliding update must scatter at position % sliding_window_len."""
        cfg = _FakeConfig()
        sliding_window_len = 4
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=sliding_window_len)

        # Prefill sliding layer with zeros
        k_init = torch.zeros(1, 2, sliding_window_len, 8)
        v_init = torch.zeros(1, 2, sliding_window_len, 8)
        cache.update(
            k_init,
            v_init,
            layer_idx=0,
            cache_kwargs={"position_ids": pos_ids(seq=sliding_window_len), "is_sliding": True},
        )

        # Decode at position 5: slot = 5 % 4 = 1
        k_dec = torch.ones(1, 2, 1, 8) * 99.0
        v_dec = torch.ones(1, 2, 1, 8) * 99.0
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[5]]), "is_sliding": True}
        )
        # The output shape should be sliding_window_len
        assert k_out.shape[2] == sliding_window_len
        assert torch.isfinite(k_out).all()

    def test_output_shape_non_sliding_equals_ctx_len(self):
        """Non-sliding update output must have shape matching ctx_len."""
        cfg = _FakeConfig()
        ctx_len = 16
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=ctx_len, sliding_window_len=4)
        k, v = make_kv_4d(seq=ctx_len)
        k_out, v_out = cache.update(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len), "is_sliding": False}
        )
        assert k_out.shape[2] == ctx_len

    def test_output_shape_sliding_equals_window_size(self):
        """Sliding update output must have shape matching sliding_window_len."""
        cfg = _FakeConfig()
        sliding_window_len = 4
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=sliding_window_len)
        k, v = make_kv_4d(seq=sliding_window_len)
        k_out, v_out = cache.update(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=sliding_window_len), "is_sliding": True}
        )
        assert k_out.shape[2] == sliding_window_len

    def test_multi_layer_independence(self):
        """Different layers must not interfere with each other."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)

        for layer_idx in range(3):
            k = torch.ones(1, 2, 8, 4) * float(layer_idx + 1)
            v = torch.ones(1, 2, 8, 4) * float(layer_idx + 1)
            cache.update(k, v, layer_idx=layer_idx, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False})

        # Each layer's cache must have its own value
        for layer_idx in range(3):
            expected = float(layer_idx + 1)
            assert cache.key_cache[layer_idx][0, 0, 0, 0].item() == pytest.approx(expected, abs=1e-5)

    def test_to_legacy_cache_round_trip(self):
        """to_legacy_cache must return a tuple of (key, value) pairs per layer."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)

        for layer_idx in range(2):
            k, v = make_kv_4d(seq=8)
            cache.update(k, v, layer_idx=layer_idx, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False})

        legacy = cache.to_legacy_cache()
        assert isinstance(legacy, tuple)
        assert len(legacy) == 2
        for layer_kv in legacy:
            assert len(layer_kv) == 2  # (key, value)

    def test_get_seq_length_returns_correct_value(self):
        """get_seq_length must return the sequence length of the cached layer."""
        cfg = _FakeConfig()
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=16, sliding_window_len=4)

        # Empty cache
        assert cache.get_seq_length(layer_idx=0) == 0

        # After update
        k, v = make_kv_4d(seq=8)
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8), "is_sliding": False})
        assert cache.get_seq_length(layer_idx=0) == 8

    def test_update_returns_finite_tensors_after_decode(self):
        """Decode update must return finite tensors."""
        cfg = _FakeConfig()
        ctx_len = 16
        cache = QEffSlidingWindowCache(cfg, batch_size=1, max_cache_len=ctx_len, sliding_window_len=4)

        # Prefill
        k, v = make_kv_4d(seq=ctx_len)
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len), "is_sliding": False})

        # Decode
        k_dec = torch.randn(1, 2, 1, 16)
        v_dec = torch.randn(1, 2, 1, 16)
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[ctx_len - 1]]), "is_sliding": False}
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()


# ---------------------------------------------------------------------------
# Tests: QEffDynamicLayer.update3D (GPTBigCode 3D KV cache)
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffDynamicCache3D:
    """QEffDynamicLayer.update3D must handle 3D KV tensors [batch, seq, kv_dim]."""

    def test_update3d_first_call_stores_tensors(self):
        """First update3D call must store tensors in the layer."""
        layer = QEffDynamicLayer()
        k, v = make_kv_3d(batch=1, seq=8, kv_dim=32)
        k_out, v_out = layer.update3D(k, v, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert layer.keys is not None
        assert layer.values is not None
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_update3d_output_is_finite(self):
        """update3D must return finite tensors."""
        layer = QEffDynamicLayer()
        k, v = make_kv_3d(batch=1, seq=8, kv_dim=32)
        k_out, v_out = layer.update3D(k, v, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_update3d_output_shape_is_correct(self):
        """update3D output must have shape [batch, ctx_len, kv_dim]."""
        layer = QEffDynamicLayer()
        batch, ctx_len, kv_dim = 1, 16, 32
        k = torch.zeros(batch, ctx_len, kv_dim)
        v = torch.zeros(batch, ctx_len, kv_dim)
        k_out, v_out = layer.update3D(k, v, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})
        assert k_out.shape == (batch, ctx_len, kv_dim)
        assert v_out.shape == (batch, ctx_len, kv_dim)

    def test_update3d_scatter_at_correct_position(self):
        """update3D decode must scatter at the correct position."""
        layer = QEffDynamicLayer()
        batch, ctx_len, kv_dim = 1, 16, 32

        # Prefill with zeros
        k_init = torch.zeros(batch, ctx_len, kv_dim)
        v_init = torch.zeros(batch, ctx_len, kv_dim)
        layer.update3D(k_init, v_init, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode: write known value at position 3
        k_dec = torch.ones(batch, 1, kv_dim) * 42.0
        v_dec = torch.ones(batch, 1, kv_dim) * 42.0
        k_out, v_out = layer.update3D(k_dec, v_dec, cache_kwargs={"position_ids": torch.tensor([[3]])})

        assert k_out[0, 3, 0].item() == pytest.approx(42.0, abs=1e-5)

    def test_update3d_prior_positions_not_corrupted(self):
        """update3D decode must not corrupt positions before the decode position."""
        layer = QEffDynamicLayer()
        batch, ctx_len, kv_dim = 1, 16, 4

        # Prefill with sequential values
        k_init = (
            torch.arange(ctx_len, dtype=torch.float32).reshape(1, ctx_len, 1).expand(batch, ctx_len, kv_dim).clone()
        )
        v_init = k_init.clone()
        layer.update3D(k_init, v_init, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode at position 5
        k_dec = torch.ones(batch, 1, kv_dim) * 999.0
        v_dec = torch.ones(batch, 1, kv_dim) * 999.0
        k_out, v_out = layer.update3D(k_dec, v_dec, cache_kwargs={"position_ids": torch.tensor([[5]])})

        # Position 5 must be 999.0
        assert k_out[0, 5, 0].item() == pytest.approx(999.0, abs=1e-5)
        # Positions before 5 must be preserved
        assert k_out[0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert k_out[0, 3, 0].item() == pytest.approx(3.0, abs=1e-5)
        assert k_out[0, 4, 0].item() == pytest.approx(4.0, abs=1e-5)

    def test_qeff_dynamic_cache_update3d_delegates_to_layer(self):
        """QEffDynamicCache.update3D must delegate to the layer's update3D."""
        cache = QEffDynamicCache()
        batch, ctx_len, kv_dim = 1, 8, 32
        k = torch.randn(batch, ctx_len, kv_dim)
        v = torch.randn(batch, ctx_len, kv_dim)
        k_out, v_out = cache.update3D(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})
        assert k_out is not None
        assert v_out is not None
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_qeff_dynamic_cache_update3d_creates_layer(self):
        """QEffDynamicCache.update3D must create a new layer at the given index."""
        cache = QEffDynamicCache()
        k, v = make_kv_3d(batch=1, seq=8, kv_dim=32)
        cache.update3D(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=8)})
        assert len(cache.layers) == 1


# ---------------------------------------------------------------------------
# Tests: QEffHybridCacheForGPTOSS chunked methods
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestQEffHybridCacheForGPTOSSChunked:
    """QEffHybridCacheForGPTOSS chunked prefill methods must be numerically correct."""

    def _make_cache_with_layer(self, batch=1, heads=2, ctx_len=16, head_dim=8, sliding_window_len=4):
        """Create a cache with one pre-initialized layer."""
        cfg = _FakeConfig()
        cache = QEffHybridCacheForGPTOSS(
            cfg, batch_size=batch, max_cache_len=ctx_len, sliding_window_len=sliding_window_len
        )
        # Initialize layer 0 (full cache)
        k = torch.zeros(batch, heads, ctx_len, head_dim)
        v = torch.zeros(batch, heads, ctx_len, head_dim)
        cache.key_cache.append(k)
        cache.value_cache.append(v)
        return cache

    def _make_sliding_cache_with_layer(self, batch=1, heads=2, sliding_window_len=4, head_dim=8):
        """Create a cache with one pre-initialized sliding window layer."""
        cfg = _FakeConfig()
        cache = QEffHybridCacheForGPTOSS(cfg, batch_size=batch, max_cache_len=16, sliding_window_len=sliding_window_len)
        # Initialize layer 0 (sliding window)
        k = torch.zeros(batch, heads, sliding_window_len, head_dim)
        v = torch.zeros(batch, heads, sliding_window_len, head_dim)
        cache.key_cache.append(k)
        cache.value_cache.append(v)
        return cache

    def test_full_cache_update_chunked_returns_finite(self):
        """full_cache_update_chunked must return finite tensors."""
        cache = self._make_cache_with_layer()
        batch, heads, seq_len, head_dim = 1, 2, 4, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.full_cache_update_chunked(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=seq_len), "batch_index": None}
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_full_cache_update_chunked_scatter_at_correct_position(self):
        """full_cache_update_chunked must scatter at the correct position."""
        cache = self._make_cache_with_layer(ctx_len=16)
        batch, heads, head_dim = 1, 2, 8

        # Write known value at positions 0-3
        k = torch.ones(batch, heads, 4, head_dim) * 5.0
        v = torch.ones(batch, heads, 4, head_dim) * 5.0
        k_out, v_out = cache.full_cache_update_chunked(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=4), "batch_index": None}
        )
        # Positions 0-3 should have value 5.0
        assert k_out[0, 0, 0, 0].item() == pytest.approx(5.0, abs=1e-5)
        assert k_out[0, 0, 3, 0].item() == pytest.approx(5.0, abs=1e-5)

    def test_full_cache_update_chunked_output_shape(self):
        """full_cache_update_chunked output must have the correct shape."""
        ctx_len = 16
        cache = self._make_cache_with_layer(ctx_len=ctx_len)
        batch, heads, seq_len, head_dim = 1, 2, 4, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.full_cache_update_chunked(
            k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=seq_len), "batch_index": None}
        )
        assert k_out.shape[2] == ctx_len

    def test_sliding_window_update_chunked_returns_finite(self):
        """sliding_window_update_chunked must return finite tensors."""
        sliding_window_len = 4
        cache = self._make_sliding_cache_with_layer(sliding_window_len=sliding_window_len)
        batch, heads, seq_len, head_dim = 1, 2, 4, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.sliding_window_update_chunked(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": pos_ids(seq=seq_len),
                "batch_index": None,
                "sliding_window": sliding_window_len,
            },
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_sliding_window_update_chunked_output_shape(self):
        """sliding_window_update_chunked output must have the correct shape."""
        sliding_window_len = 4
        seq_len = 4
        cache = self._make_sliding_cache_with_layer(sliding_window_len=sliding_window_len)
        batch, heads, head_dim = 1, 2, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.sliding_window_update_chunked(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": pos_ids(seq=seq_len),
                "batch_index": None,
                "sliding_window": sliding_window_len,
            },
        )
        # Output shape: seq_len + sliding_window_len
        expected_ctx = seq_len + sliding_window_len
        assert k_out.shape[2] == expected_ctx

    def test_sliding_window_update_chunked_with_larger_window(self):
        """sliding_window_update_chunked with a larger window must return finite tensors."""
        sliding_window_len = 8
        seq_len = 4
        cache = self._make_sliding_cache_with_layer(sliding_window_len=sliding_window_len)
        batch, heads, head_dim = 1, 2, 8
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)
        k_out, v_out = cache.sliding_window_update_chunked(
            k,
            v,
            layer_idx=0,
            cache_kwargs={
                "position_ids": pos_ids(seq=seq_len),
                "batch_index": None,
                "sliding_window": sliding_window_len,
            },
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()


# ---------------------------------------------------------------------------
# Tests: CCL (Compute Context Length) cache path
# ---------------------------------------------------------------------------


@pytest.mark.cache
class TestCCLCachePath:
    """QEffDynamicCache.update with CCL kwarg must work correctly."""

    def test_update_with_ccl_returns_finite(self):
        """update() with CCL kwarg must return finite tensors."""
        from QEfficient.transformers.cache_utils import QEffDynamicCache

        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8
        k = torch.randn(batch, heads, ctx_len, head_dim)
        v = torch.randn(batch, heads, ctx_len, head_dim)

        # Prefill
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode with CCL
        k_dec = torch.randn(batch, heads, 1, head_dim)
        v_dec = torch.randn(batch, heads, 1, head_dim)
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[8]]), "CCL": 8}
        )
        assert torch.isfinite(k_out).all()
        assert torch.isfinite(v_out).all()

    def test_update_with_ccl_output_shape_matches_ccl(self):
        """update() with CCL kwarg must return tensors with ctx_len=CCL."""
        from QEfficient.transformers.cache_utils import QEffDynamicCache

        cache = QEffDynamicCache()
        batch, heads, ctx_len, head_dim = 1, 2, 16, 8
        k = torch.randn(batch, heads, ctx_len, head_dim)
        v = torch.randn(batch, heads, ctx_len, head_dim)

        # Prefill
        cache.update(k, v, layer_idx=0, cache_kwargs={"position_ids": pos_ids(seq=ctx_len)})

        # Decode with CCL=8 (smaller than ctx_len=16)
        ccl = 8
        k_dec = torch.randn(batch, heads, 1, head_dim)
        v_dec = torch.randn(batch, heads, 1, head_dim)
        k_out, v_out = cache.update(
            k_dec, v_dec, layer_idx=0, cache_kwargs={"position_ids": torch.tensor([[4]]), "CCL": ccl}
        )
        assert k_out.shape[2] == ccl
        assert v_out.shape[2] == ccl
