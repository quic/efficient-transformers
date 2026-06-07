# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Paged (block-table) KV-cache scatter/gather custom ops for Cloud AI 100.

These mirror the continuous-batching ops in ``ctx_scatter_gather_cb.py`` but
address a *block-pool* KV tensor of shape ``[num_blocks, num_heads, page_size,
head_dim]`` instead of a per-sequence-contiguous tensor of shape
``[full_batch_size, num_heads, ctx_len, head_dim]``.

The physical location of a logical token is decomposed into two per-token index
tensors (computed by the caller from a ``block_table``)::

    physical_block = block_table[seq, logical_pos // page_size]
    intra_offset   = logical_pos % page_size

Both ``block_idx`` and ``offset_idx`` have shape ``[batch, seq]`` (write) or
``[batch, ctx]`` (read), exactly like ``position_ids`` / ``ctx_indices`` in the
contiguous ops. The ONNX lowering is the same ``ScatterND`` / ``GatherND`` used
by the existing ops — only the index construction differs (two varying axes,
block + offset, instead of one ctx axis), so the AIC compiler support is
identical.
"""

import onnxscript
import torch

from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatterPaged(
    data: onnxscript.FLOAT,
    block_idx: onnxscript.INT64,
    offset_idx: onnxscript.INT64,
    updates: onnxscript.FLOAT,
) -> onnxscript.FLOAT:
    # data: [num_blocks, num_heads, page_size, head_dim]
    # block_idx, offset_idx: [batch, seq]   updates: [batch, num_heads, seq, head_dim]
    batch_size = ops.Gather(ops.Shape(block_idx), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(block_idx), [1])

    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)

    # Range requires scalar (0-d) args per the ONNX spec.
    zero_s = ops.Constant(value_int=0)
    one_s = ops.Constant(value_int=1)
    num_heads_s = ops.Gather(ops.Shape(data), 1)

    blk_idx = ops.Expand(ops.Unsqueeze(block_idx, [1, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero_s, num_heads_s, one_s), [0, 2, 3]), exp_shape)
    off_idx = ops.Expand(ops.Unsqueeze(offset_idx, [1, 3]), exp_shape)
    indices = ops.Concat(blk_idx, head_idx, off_idx, axis=3)

    return ops.ScatterND(data, indices, updates)


class CtxScatterPagedFunc(torch.autograd.Function):
    """Scatter new KV (``updates``) into a block-pool at block/offset indices."""

    @staticmethod
    def forward(
        data: torch.Tensor,
        block_idx: torch.Tensor,
        offset_idx: torch.Tensor,
        updates: torch.Tensor,
    ):
        # data: [num_blocks, num_heads, page_size, head_dim]
        # block_idx/offset_idx: [batch, seq]; updates: [batch, num_heads, seq, head_dim]
        blk = block_idx.unsqueeze(1)  # [batch, 1, seq]
        head = torch.arange(data.shape[1]).view(1, -1, 1)  # [1, num_heads, 1]
        off = offset_idx.unsqueeze(1)  # [batch, 1, seq]
        data[blk, head, off] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        data: torch.Value,
        block_idx: torch.Value,
        offset_idx: torch.Value,
        updates: torch.Value,
    ) -> torch.Value:
        return g.onnxscript_op(CtxScatterPaged, data, block_idx, offset_idx, updates).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGatherPaged(
    data: onnxscript.FLOAT,
    block_idx: onnxscript.INT64,
    offset_idx: onnxscript.INT64,
) -> onnxscript.FLOAT:
    # data: [num_blocks, num_heads, page_size, head_dim]
    # block_idx, offset_idx: [batch, ctx]   -> out: [batch, num_heads, ctx, head_dim]
    batch_size = ops.Gather(ops.Shape(block_idx), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    ctx_len = ops.Gather(ops.Shape(block_idx), [1])

    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, ctx_len, one, axis=0)

    # Range requires scalar (0-d) args per the ONNX spec.
    zero_s = ops.Constant(value_int=0)
    one_s = ops.Constant(value_int=1)
    num_heads_s = ops.Gather(ops.Shape(data), 1)

    blk_idx = ops.Expand(ops.Unsqueeze(block_idx, [1, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero_s, num_heads_s, one_s), [0, 2, 3]), exp_shape)
    off_idx = ops.Expand(ops.Unsqueeze(offset_idx, [1, 3]), exp_shape)
    indices = ops.Concat(blk_idx, head_idx, off_idx, axis=3)

    return ops.GatherND(data, indices)


class CtxGatherPagedFunc(torch.autograd.Function):
    """Gather KV from a block-pool at block/offset indices for attention."""

    @staticmethod
    def forward(data: torch.Tensor, block_idx: torch.Tensor, offset_idx: torch.Tensor):
        # data: [num_blocks, num_heads, page_size, head_dim]
        # block_idx/offset_idx: [batch, ctx]; out: [batch, num_heads, ctx, head_dim]
        blk = block_idx.unsqueeze(1)  # [batch, 1, ctx]
        head = torch.arange(data.shape[1]).view(1, -1, 1)  # [1, num_heads, 1]
        off = offset_idx.unsqueeze(1)  # [batch, 1, ctx]
        return data[blk, head, off]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(
        g: torch.Graph,
        data: torch.Value,
        block_idx: torch.Value,
        offset_idx: torch.Value,
    ) -> torch.Value:
        return g.onnxscript_op(CtxGatherPaged, data, block_idx, offset_idx).setTypeAs(data)
