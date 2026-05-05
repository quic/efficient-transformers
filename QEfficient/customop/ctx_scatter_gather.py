# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import onnxscript
import torch

from QEfficient.utils import constants

ops = getattr(onnxscript, "opset" + str(constants.ONNX_EXPORT_OPSET))


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatter(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    num_heads = ops.Gather(ops.Shape(data), [1])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, num_heads, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2, 3]), exp_shape)
    head_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, num_heads, one), [0, 2, 3]), exp_shape)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [1, 3]), exp_shape)
    indices = ops.Concat(batch_idx, head_idx, ctx_idx, axis=3)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFunc(torch.autograd.Function):
    """
    Function to scatter the current key values into KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        batch_idx = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_idx = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_idx = position_ids.unsqueeze(1)
        data[batch_idx, head_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter, data, position_ids, updates).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatter3D(data: onnxscript.FLOAT, position_ids: onnxscript.INT32, updates: onnxscript.FLOAT) -> onnxscript.FLOAT:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2]), exp_shape)

     # keep index tensor types aligned for backend that require exact dtype match
    batch_idx = ops.Cast(batch_idx, to=onnxscript.INT32.dtype)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFunc3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        data = data.clone()
        batch_idx = torch.arange(data.shape[0]).view(-1, 1)
        ctx_idx = torch.where(position_ids == torch.iinfo(torch.int32).max, data.shape[1] - 1, position_ids)
        data[batch_idx, ctx_idx] = updates
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter3D, data, position_ids, updates).setTypeAs(data)


class CtxScatterFunc3DGeneralized(torch.autograd.Function):
    """Scatter variant that preserves ``data`` at invalid (INT32_MAX) positions.

    Unlike :class:`CtxScatterFunc3D`, which writes updates for invalid rows to
    ``data.shape[1]-1`` (potentially clobbering valid content), this version
    masks out invalid rows before scattering so ``data`` is left untouched where
    ``position_ids == INT32_MAX``.
    """

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        data = data.clone()
        valid = position_ids != torch.iinfo(torch.int32).max
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1).expand_as(position_ids)
        data[batch_idx[valid], position_ids[valid].long()] = updates[valid]
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter3D, data, position_ids, updates).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxScatter3DInt(
    data: onnxscript.INT32, position_ids: onnxscript.INT32, updates: onnxscript.INT32
) -> onnxscript.INT32:
    # Find dims
    batch_size = ops.Gather(ops.Shape(data), [0])
    seq_len = ops.Gather(ops.Shape(position_ids), [1])

    # Expanded shape to create indices
    zero = ops.Constant(value_ints=[0])
    one = ops.Constant(value_ints=[1])
    exp_shape = ops.Concat(batch_size, seq_len, one, axis=0)

    # Create indices
    batch_idx = ops.Expand(ops.Unsqueeze(ops.Range(zero, batch_size, one), [1, 2]), exp_shape)
    batch_idx = ops.Cast(batch_idx, to=onnxscript.INT32.dtype)
    ctx_idx = ops.Expand(ops.Unsqueeze(position_ids, [2]), exp_shape)
    indices = ops.Concat(batch_idx, ctx_idx, axis=2)

    return ops.ScatterND(data, indices, updates)


class CtxScatterFunc3DInt(torch.autograd.Function):
    """Int32-typed scatter used to build a packed->original index table."""

    @staticmethod
    def forward(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor):
        data = data.clone()
        valid = position_ids != torch.iinfo(torch.int32).max
        batch_idx = torch.arange(data.shape[0], device=data.device).view(-1, 1).expand_as(position_ids)
        data[batch_idx[valid], position_ids[valid].long()] = updates[valid]
        return data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, position_ids: torch.Value, updates: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxScatter3DInt, data, position_ids, updates).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGather3D(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    ctx_indices = ops.Expand(ctx_indices, ops.Slice(ops.Shape(data), starts=[0], ends=[2], axes=[0]))
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=1)


class CtxGatherFunc3D(torch.autograd.Function):
    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGather3D, data, ctx_indices).setTypeAs(data)


class CtxGatherFunc3DGeneralized(torch.autograd.Function):
    """Gather variant that tolerates INT32_MAX indices (invalid rows read from 0).

    Semantically equivalent to :class:`CtxGatherFunc3D` on the PyTorch side but
    exposed as a separate autograd op so callers using the packed/cumsum scatter
    pipeline can be easily recognized and so the ONNX symbolic omits
    ``setTypeAs`` (needed when the caller already has a matching dtype on
    ``data`` and wants the op signature to flow through without dtype pinning).
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGather3D, data, ctx_indices)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGather(
    data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32, comp_ctx_len: onnxscript.INT32
) -> onnxscript.FLOAT:
    # Create a shape tensor based on comp_ctx_len
    shape_tensor = ops.Concat(ops.Shape(data)[:2], ops.Reshape(comp_ctx_len, [1]), axis=0)

    # Directly use the shape tensor without validation
    ctx_indices = ops.Expand(ctx_indices, shape_tensor)
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=2)


class CtxGatherFunc(torch.autograd.Function):
    """
    Function to gather only the valid key values from KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor, comp_ctx_len: int):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value, comp_ctx_len: int) -> torch.Value:
        return g.onnxscript_op(CtxGather, data, ctx_indices, comp_ctx_len).setTypeAs(data)


@onnxscript.script(onnxscript.values.Opset("com.qualcomm.cloud", 1))
def CtxGatherBlockedKV(data: onnxscript.FLOAT, ctx_indices: onnxscript.INT32) -> onnxscript.FLOAT:
    ctx_indices = ops.Unsqueeze(ctx_indices, [-1])
    return ops.GatherND(data, ctx_indices, batch_dims=2)


class CtxGatherFuncBlockedKV(torch.autograd.Function):
    """
    Function to gather only the valid key values from KV-cache.
    """

    @staticmethod
    def forward(data: torch.Tensor, ctx_indices: torch.Tensor):
        batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
        head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
        ctx_indices = torch.where(ctx_indices == torch.iinfo(torch.int32).max, 0, ctx_indices)
        return data[batch_indices, head_indices, ctx_indices]

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, data: torch.Value, ctx_indices: torch.Value) -> torch.Value:
        return g.onnxscript_op(CtxGatherBlockedKV, data, ctx_indices).setTypeAs(data)
