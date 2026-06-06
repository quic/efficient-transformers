# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Parity test: QEffPagedDynamicLayer (block pool) vs QEffDynamicLayer (contiguous).

Feeds the SAME key/value/position stream to both cache layers and asserts the
gathered attention KV (the [batch, heads, ctx, head_dim] returned by ``update``)
matches over valid positions, across prefill + decode. This validates the paged
cache layer end-to-end against the proven contiguous CB path — using the REAL
``QEfficient.transformers.cache_utils`` module (loaded with the heavy package
``__init__`` stubbed, real ``transformers`` + real custom ops).
"""

import importlib.util
import pathlib
import sys
import types

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]


def _load_real_cache_utils():
    if "QEfficient.transformers.cache_utils" in sys.modules:
        return sys.modules["QEfficient.transformers.cache_utils"]

    # --- stub the package tree to avoid the heavy QEfficient/__init__ ---
    def _pkg(name, path):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m
        return m

    if "QEfficient" not in sys.modules:
        _pkg("QEfficient", REPO / "QEfficient")
        _pkg("QEfficient.utils", REPO / "QEfficient" / "utils")
        const = types.ModuleType("QEfficient.utils.constants")
        const.ONNX_EXPORT_OPSET = 17
        sys.modules["QEfficient.utils.constants"] = const
        customop = _pkg("QEfficient.customop", REPO / "QEfficient" / "customop")
        # Load the real op modules and expose their funcs on the customop stub.
        for fname in ("ctx_scatter_gather", "ctx_scatter_gather_cb", "ctx_paged_scatter_gather"):
            spec = importlib.util.spec_from_file_location(
                f"QEfficient.customop.{fname}", str(REPO / "QEfficient" / "customop" / f"{fname}.py")
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            for attr in dir(mod):
                if attr.startswith("Ctx"):
                    setattr(customop, attr, getattr(mod, attr))
        _pkg("QEfficient.transformers", REPO / "QEfficient" / "transformers")

    spec = importlib.util.spec_from_file_location(
        "QEfficient.transformers.cache_utils", str(REPO / "QEfficient" / "transformers" / "cache_utils.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_cu = _load_real_cache_utils()
QEffDynamicLayer = _cu.QEffDynamicLayer
QEffPagedDynamicLayer = _cu.QEffPagedDynamicLayer


def _make_block_table(bsz, max_blocks, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for b in range(bsz):
        base = b * max_blocks
        rows.append(torch.randperm(max_blocks, generator=g) + base)
    return torch.stack(rows, 0).to(torch.int32)


def test_paged_layer_matches_contiguous_layer():
    torch.manual_seed(2024)
    bsz, heads, dim = 2, 4, 8
    page_size, max_blocks = 4, 8
    ctx_len = page_size * max_blocks  # 32
    num_blocks = bsz * max_blocks

    block_table = _make_block_table(bsz, max_blocks, seed=11)
    batch_index = torch.arange(bsz).view(bsz, 1).to(torch.int32)

    # Pre-allocated caches (as in the AIC graph: passed in as zeros).
    cont = QEffDynamicLayer.from_tensors(
        torch.zeros(bsz, heads, ctx_len, dim), torch.zeros(bsz, heads, ctx_len, dim)
    )
    paged = QEffPagedDynamicLayer.from_tensors(
        torch.zeros(num_blocks, heads, page_size, dim), torch.zeros(num_blocks, heads, page_size, dim)
    )

    def step(key, value, position_ids, valid_len):
        ck, cv = cont.update(
            key, value, {"batch_index": batch_index, "position_ids": position_ids}
        )
        pk, pv = paged.update(
            key,
            value,
            {"batch_index": batch_index, "position_ids": position_ids, "block_table": block_table},
        )
        assert ck.shape == pk.shape, f"shape mismatch {ck.shape} vs {pk.shape}"
        assert torch.allclose(pk[:, :, :valid_len, :], ck[:, :, :valid_len, :]), "K mismatch"
        assert torch.allclose(pv[:, :, :valid_len, :], cv[:, :, :valid_len, :]), "V mismatch"

    # ---- prefill ----
    prompt_len = 12
    pos = torch.arange(prompt_len).view(1, -1).expand(bsz, prompt_len).to(torch.int32)
    step(torch.randn(bsz, heads, prompt_len, dim), torch.randn(bsz, heads, prompt_len, dim), pos, prompt_len)

    # ---- decode (crosses page boundaries) ----
    cur = prompt_len
    for _ in range(10):
        pos1 = torch.full((bsz, 1), cur, dtype=torch.int32)
        step(torch.randn(bsz, heads, 1, dim), torch.randn(bsz, heads, 1, dim), pos1, cur + 1)
        cur += 1


def test_paged_layer_ccl_partial_read():
    """CCL (comp-ctx-length) restricts the gather window; paged must match contiguous."""
    torch.manual_seed(7)
    bsz, heads, dim = 2, 2, 4
    page_size, max_blocks = 4, 8
    ctx_len = page_size * max_blocks
    num_blocks = bsz * max_blocks
    block_table = _make_block_table(bsz, max_blocks, seed=5)
    batch_index = torch.arange(bsz).view(bsz, 1).to(torch.int32)

    cont = QEffDynamicLayer.from_tensors(
        torch.zeros(bsz, heads, ctx_len, dim), torch.zeros(bsz, heads, ctx_len, dim)
    )
    paged = QEffPagedDynamicLayer.from_tensors(
        torch.zeros(num_blocks, heads, page_size, dim), torch.zeros(num_blocks, heads, page_size, dim)
    )
    prompt_len = 20
    ccl = 16
    pos = torch.arange(prompt_len).view(1, -1).expand(bsz, prompt_len).to(torch.int32)
    k = torch.randn(bsz, heads, prompt_len, dim)
    v = torch.randn(bsz, heads, prompt_len, dim)
    ck, cv = cont.update(k, v, {"batch_index": batch_index, "position_ids": pos, "CCL": ccl})
    pk, pv = paged.update(
        k, v, {"batch_index": batch_index, "position_ids": pos, "block_table": block_table, "CCL": ccl}
    )
    assert ck.shape[2] == ccl and pk.shape[2] == ccl, "CCL read length wrong"
    assert torch.allclose(pk, ck), "CCL K mismatch"
    assert torch.allclose(pv, cv), "CCL V mismatch"


def test_paged_padding_routes_to_null_block_no_corruption():
    """Different-length sequences with padding (position_ids < 0).

    Padded writes must go to the reserved null block (never read) and must NOT
    corrupt any real block. Valid-region reads must match an independent oracle.
    """
    torch.manual_seed(99)
    bsz, heads, dim = 2, 2, 4
    page_size, max_blocks = 4, 6
    ctx_len = page_size * max_blocks
    num_blocks = bsz * max_blocks + 1  # reserve last as null block
    null_block = num_blocks - 1

    block_table = _make_block_table(bsz, max_blocks, seed=21)  # references blocks [0, bsz*max_blocks)
    batch_index = torch.arange(bsz).view(bsz, 1).to(torch.int32)

    paged = QEffPagedDynamicLayer.from_tensors(
        torch.zeros(num_blocks, heads, page_size, dim), torch.zeros(num_blocks, heads, page_size, dim)
    )

    # Sequence lengths differ: seq0=10, seq1=5. Prefill width = 10, seq1 padded with pos=-1.
    lens = [10, 5]
    width = max(lens)
    pos = torch.full((bsz, width), -1, dtype=torch.int32)
    for b, L in enumerate(lens):
        pos[b, :L] = torch.arange(L)
    key = torch.randn(bsz, heads, width, dim)
    val = torch.randn(bsz, heads, width, dim)

    # Independent oracle: contiguous cache, write ONLY valid tokens.
    oracle_k = torch.zeros(bsz, heads, ctx_len, dim)
    oracle_v = torch.zeros(bsz, heads, ctx_len, dim)
    for b, L in enumerate(lens):
        oracle_k[b, :, :L, :] = key[b, :, :L, :]
        oracle_v[b, :, :L, :] = val[b, :, :L, :]

    pk, pv = paged.update(
        key, val, {"batch_index": batch_index, "position_ids": pos, "block_table": block_table}
    )

    for b, L in enumerate(lens):
        assert torch.allclose(pk[b, :, :L, :], oracle_k[b, :, :L, :]), f"seq{b} K corrupted by padding"
        assert torch.allclose(pv[b, :, :L, :], oracle_v[b, :, :L, :]), f"seq{b} V corrupted by padding"

    # The null block must have received the padded writes (proves routing happened):
    # seq1 had width-L1 = 5 padded tokens.
    assert paged.keys[null_block].abs().sum() > 0, "padded writes did not reach the null block"


if __name__ == "__main__":
    test_paged_layer_matches_contiguous_layer()
    test_paged_layer_ccl_partial_read()
    test_paged_padding_routes_to_null_block_no_corruption()
    print("PAGED CACHE LAYER PARITY: ALL PASS")
