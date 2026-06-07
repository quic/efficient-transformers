# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Parity test for paged (block-table) KV scatter/gather vs contiguous KV.

The property under test: storing KV in a *shuffled* block pool and reading it
back through a ``block_table`` must produce the **exact same** per-position KV
as the contiguous ``[batch, heads, ctx_len, head_dim]`` layout. This validates
the index math in ``CtxScatterPagedFunc`` / ``CtxGatherPagedFunc`` (the core of
the paged-attention port) across prefill + multiple decode steps.

The real shipped op file is loaded directly (the heavy ``QEfficient`` package
``__init__`` is stubbed) so this runs on any CPU box. The on-device op + ONNX
export checks run on the QAIC/Linux host as the Step-1/2 gates.
"""

import importlib.util
import pathlib
import sys
import types

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
OPS_PATH = REPO / "QEfficient" / "customop" / "ctx_paged_scatter_gather.py"


def _load_paged_ops():
    """Load the real ctx_paged_scatter_gather module without QEfficient/__init__."""
    if "QEfficient" not in sys.modules:
        pkg = types.ModuleType("QEfficient")
        pkg.__path__ = [str(REPO / "QEfficient")]
        sys.modules["QEfficient"] = pkg
        utils = types.ModuleType("QEfficient.utils")
        utils.__path__ = [str(REPO / "QEfficient" / "utils")]
        sys.modules["QEfficient.utils"] = utils
        constants = types.ModuleType("QEfficient.utils.constants")
        constants.ONNX_EXPORT_OPSET = 17
        sys.modules["QEfficient.utils.constants"] = constants
        customop = types.ModuleType("QEfficient.customop")
        customop.__path__ = [str(REPO / "QEfficient" / "customop")]
        sys.modules["QEfficient.customop"] = customop
    spec = importlib.util.spec_from_file_location(
        "QEfficient.customop.ctx_paged_scatter_gather", str(OPS_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    # onnxscript's @script introspects sys.modules[func.__module__]; register first.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_ops = _load_paged_ops()
CtxScatterPagedFunc = _ops.CtxScatterPagedFunc
CtxGatherPagedFunc = _ops.CtxGatherPagedFunc


def _make_block_table(bsz, max_blocks, seed=0):
    """Per-sequence disjoint, shuffled logical->physical block map.

    Sequence b owns physical blocks [b*max_blocks, (b+1)*max_blocks), shuffled,
    so indirection is non-trivial and sequences never collide.
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    for b in range(bsz):
        base = b * max_blocks
        perm = torch.randperm(max_blocks, generator=g) + base
        rows.append(perm)
    return torch.stack(rows, 0).to(torch.int32)  # [bsz, max_blocks]


def _paged_write(pool, block_table, position_ids, key, page_size):
    logical_block = (position_ids // page_size).clamp(min=0)
    offset = position_ids % page_size
    phys_block = torch.gather(block_table, 1, logical_block.to(torch.int64)).to(torch.int64)
    return CtxScatterPagedFunc.forward(pool, phys_block, offset.to(torch.int64), key)


def _paged_read(pool, block_table, ctx_len, page_size):
    bsz = block_table.shape[0]
    max_blocks = block_table.shape[1]
    ctx_idx = torch.arange(ctx_len).view(1, -1).expand(bsz, ctx_len)  # [bsz, ctx]
    logical_block = ctx_idx // page_size
    # Valid range is enforced by the caller (ctx_len <= max_blocks*page_size); assert
    # rather than silently clamp so a malformed block_table fails visibly (stricter oracle).
    assert int(logical_block.max()) < max_blocks, "ctx_len exceeds block_table capacity"
    offset = (ctx_idx % page_size).to(torch.int64)
    phys_block = torch.gather(block_table, 1, logical_block.to(torch.int64)).to(torch.int64)
    num_blocks = pool.shape[0]
    assert int(phys_block.min()) >= 0 and int(phys_block.max()) < num_blocks, "block id out of pool range"
    return CtxGatherPagedFunc.forward(pool, phys_block, offset)


def test_paged_matches_contiguous_prefill_and_decode():
    torch.manual_seed(1234)
    bsz, heads, dim = 2, 4, 8
    page_size, max_blocks = 4, 6
    ctx_len = page_size * max_blocks  # 24
    num_blocks = bsz * max_blocks

    block_table = _make_block_table(bsz, max_blocks, seed=7)

    # Ground-truth contiguous cache and shuffled paged pool.
    cont = torch.zeros(bsz, heads, ctx_len, dim)
    pool = torch.zeros(num_blocks, heads, page_size, dim)

    # ---- Prefill: write positions [0, prompt_len) ----
    prompt_len = 10
    pos = torch.arange(prompt_len).view(1, -1).expand(bsz, prompt_len).to(torch.int32)
    kp = torch.randn(bsz, heads, prompt_len, dim)
    for b in range(bsz):
        cont[b, :, :prompt_len, :] = kp[b]
    pool = _paged_write(pool, block_table, pos, kp, page_size)

    paged_read = _paged_read(pool, block_table, ctx_len, page_size)
    from _metrics import report_precision

    report_precision("ops_parity.prefill", paged_read[:, :, :prompt_len, :], cont[:, :, :prompt_len, :])
    assert torch.allclose(paged_read[:, :, :prompt_len, :], cont[:, :, :prompt_len, :]), "prefill mismatch"

    # Indirection must be REAL: reading the same pool with a DIFFERENT block_table
    # must NOT reproduce the contiguous content (proves gather depends on block_table,
    # not on the logical position alone).
    wrong_bt = _make_block_table(bsz, max_blocks, seed=999)
    paged_wrong = _paged_read(pool, wrong_bt, ctx_len, page_size)
    assert not torch.allclose(paged_wrong[:, :, :prompt_len, :], cont[:, :, :prompt_len, :]), (
        "gather did not actually use block_table (indirection is a no-op)"
    )

    # CCL / partial-context read: reading only the first `ccl` logical positions must
    # match the contiguous prefix (validates the comp-ctx-length read path).
    ccl = 6
    paged_ccl = _paged_read(pool, block_table, ccl, page_size)
    assert torch.allclose(paged_ccl, cont[:, :, :ccl, :]), "CCL partial read mismatch"

    # ---- Decode: append one token at a time, compare full read each step ----
    cur = prompt_len
    for step in range(8):
        pos1 = torch.full((bsz, 1), cur, dtype=torch.int32)
        k1 = torch.randn(bsz, heads, 1, dim)
        for b in range(bsz):
            cont[b, :, cur, :] = k1[b, :, 0, :]
        pool = _paged_write(pool, block_table, pos1, k1, page_size)
        cur += 1

        paged_read = _paged_read(pool, block_table, ctx_len, page_size)
        assert torch.allclose(paged_read[:, :, :cur, :], cont[:, :, :cur, :]), f"decode step {step} mismatch"


def test_sequences_do_not_collide():
    """Writes for one sequence must never corrupt another's KV (block disjointness)."""
    torch.manual_seed(0)
    bsz, heads, dim = 3, 2, 4
    page_size, max_blocks = 2, 4
    ctx_len = page_size * max_blocks
    num_blocks = bsz * max_blocks
    block_table = _make_block_table(bsz, max_blocks, seed=3)

    pool = torch.zeros(num_blocks, heads, page_size, dim)
    cont = torch.zeros(bsz, heads, ctx_len, dim)
    pos = torch.arange(ctx_len).view(1, -1).expand(bsz, ctx_len).to(torch.int32)
    k = torch.randn(bsz, heads, ctx_len, dim)
    for b in range(bsz):
        cont[b] = k[b]
    pool = _paged_write(pool, block_table, pos, k, page_size)
    paged_read = _paged_read(pool, block_table, ctx_len, page_size)
    assert torch.allclose(paged_read, cont), "per-sequence isolation violated"


if __name__ == "__main__":
    from _metrics import measure

    with measure("test_paged_kv_parity"):
        test_paged_matches_contiguous_prefill_and_decode()
        test_sequences_do_not_collide()
    print("PAGED KV PARITY: ALL PASS")
