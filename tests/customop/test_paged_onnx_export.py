# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""ONNX export smoke test for the paged scatter/gather ops.

Validates the SYMBOLIC (onnxscript) path — the thing the eager parity tests
cannot cover — by exporting a tiny module that scatters into and gathers from a
block pool, then checking the produced ONNX graph: it must export without error,
expose the block/offset index tensors as graph inputs, and lower to the expected
ScatterND / GatherND (under the com.qualcomm.cloud opset) that the AIC compiler
consumes.
"""

import tempfile

import onnx
import pytest
import torch

pytest.importorskip("QEfficient")

from QEfficient.customop import CtxGatherPagedFunc, CtxScatterPagedFunc  # noqa: E402


class _PagedRoundTrip(torch.nn.Module):
    def forward(self, pool, block_idx, offset_idx, updates, g_block, g_offset):
        pool = CtxScatterPagedFunc.apply(pool, block_idx, offset_idx, updates)
        return CtxGatherPagedFunc.apply(pool, g_block, g_offset)


def _collect_op_types(model):
    ops = set()

    def walk(graph):
        for n in graph.node:
            ops.add((n.domain, n.op_type))
            for attr in n.attribute:
                if attr.g.ByteSize():
                    walk(attr.g)

    walk(model.graph)
    for f in model.functions:
        for n in f.node:
            ops.add((n.domain, n.op_type))
    return ops


def test_paged_ops_onnx_export():
    nb, heads, page, dim = 9, 2, 4, 8
    bsz, seq, ctx = 2, 5, 16
    pool = torch.zeros(nb, heads, page, dim)
    block_idx = torch.zeros(bsz, seq, dtype=torch.int64)
    offset_idx = torch.zeros(bsz, seq, dtype=torch.int64)
    updates = torch.randn(bsz, heads, seq, dim)
    g_block = torch.zeros(bsz, ctx, dtype=torch.int64)
    g_offset = torch.zeros(bsz, ctx, dtype=torch.int64)

    input_names = ["pool", "block_idx", "offset_idx", "updates", "g_block", "g_offset"]
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    torch.onnx.export(
        _PagedRoundTrip(),
        (pool, block_idx, offset_idx, updates, g_block, g_offset),
        path,
        input_names=input_names,
        output_names=["out"],
        opset_version=17,
        dynamo=False,
    )

    model = onnx.load(path)
    onnx.checker.check_model(model)

    graph_inputs = {i.name for i in model.graph.input}
    for name in input_names:
        assert name in graph_inputs, f"missing graph input {name}"

    ops = _collect_op_types(model)
    op_names = {op for _, op in ops}
    # Either the custom function nodes are present, or they inlined to ScatterND/GatherND.
    assert ("ScatterND" in op_names) or any("Scatter" in o for o in op_names), f"no scatter op in {ops}"
    assert ("GatherND" in op_names) or any("Gather" in o for o in op_names), f"no gather op in {ops}"


def _export(module, inputs, input_names):
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    torch.onnx.export(
        module, inputs, path, input_names=input_names, output_names=["out"], opset_version=17, dynamo=False
    )
    return path


def test_paged_ops_onnx_numeric_matches_eager():
    """Run the EXPORTED ONNX under onnxruntime and compare to eager — validates the
    onnxscript symbolic lowering (what the AIC compiler consumes), not just that it
    exports. Uses random non-trivial block/offset indices so a wrong index
    construction in the onnxscript would surface as a numeric mismatch.
    """
    import numpy as np
    import onnxruntime as ort
    from onnx import inliner

    torch.manual_seed(5)
    nb, heads, page, dim = 11, 2, 4, 8
    bsz, seq, ctx = 2, 5, 16
    # Disjoint per-sequence blocks so scatter/gather indices are non-trivial & valid.
    g = torch.Generator().manual_seed(1)
    max_blocks = 5
    bt = torch.stack([torch.randperm(max_blocks, generator=g) + b * max_blocks for b in range(bsz)], 0)
    wpos = torch.arange(seq).view(1, -1).expand(bsz, seq)
    block_idx = torch.gather(bt, 1, wpos // page).to(torch.int64)
    offset_idx = (wpos % page).to(torch.int64)
    gpos = torch.arange(ctx).view(1, -1).expand(bsz, ctx)
    g_block = torch.gather(bt, 1, (gpos // page).clamp(max=max_blocks - 1)).to(torch.int64)
    g_offset = (gpos % page).to(torch.int64)
    pool = torch.randn(nb, heads, page, dim)
    updates = torch.randn(bsz, heads, seq, dim)

    inputs = (pool.clone(), block_idx, offset_idx, updates, g_block, g_offset)
    input_names = ["pool", "block_idx", "offset_idx", "updates", "g_block", "g_offset"]

    with torch.no_grad():
        eager = _PagedRoundTrip()(*[x.clone() if torch.is_tensor(x) else x for x in inputs]).numpy()

    path = _export(_PagedRoundTrip(), inputs, input_names)
    model = onnx.load(path)
    # com.qualcomm.cloud custom functions must inline to standard ops so ORT can run them.
    model = inliner.inline_local_functions(model)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {
        "pool": pool.numpy(),
        "block_idx": block_idx.numpy(),
        "offset_idx": offset_idx.numpy(),
        "updates": updates.numpy(),
        "g_block": g_block.numpy(),
        "g_offset": g_offset.numpy(),
    }
    ort_out = sess.run(None, feeds)[0]

    max_diff = float(np.abs(eager - ort_out).max())
    print(f"[metrics] onnx_numeric: precision max_abs_diff={max_diff:.3e} (eager vs onnxruntime)")
    assert np.allclose(eager, ort_out, atol=1e-5, rtol=1e-5), f"ONNX symbolic path differs from eager: {max_diff}"


if __name__ == "__main__":
    from _metrics import measure

    with measure("test_paged_onnx_export"):
        test_paged_ops_onnx_export()
        test_paged_ops_onnx_numeric_matches_eager()
    print("PAGED ONNX EXPORT: PASS")
