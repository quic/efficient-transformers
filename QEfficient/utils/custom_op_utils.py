import torch


def select_interface(eager_impl, custom_op_impl):
    use_custom_op = torch._dynamo.is_compiling()
    return custom_op_impl if use_custom_op else eager_impl


@torch.library.custom_op("qefficient::rms_norm", mutates_args=())
def rms_norm_op(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Custom RMS Norm operation for QEfficient"""
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
    return weight * hidden_states


@rms_norm_op.register_fake
def _(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Fake implementation for torch.export - just returns tensor with same shape/dtype"""
    return torch.empty_like(hidden_states)


@torch.library.custom_op("qefficient::ctx_scatter", mutates_args=())
def ctx_scatter_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Custom context scatter operation"""
    result = data.clone()
    batch_idx = torch.arange(result.shape[0]).view(-1, 1, 1)
    head_idx = torch.arange(result.shape[1]).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    result[batch_idx, head_idx, ctx_idx] = updates
    return result


@ctx_scatter_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export - just returns data tensor with same shape/dtype"""
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_scatter_3d", mutates_args=())
def ctx_scatter_3d_op(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Custom 3D context scatter operation"""
    # Clone the data to avoid aliasing issues with torch.library.custom_op
    result = data.clone()
    batch_idx = torch.arange(result.shape[0]).view(-1, 1)
    ctx_idx = position_ids
    result[batch_idx, ctx_idx] = updates
    return result


@ctx_scatter_3d_op.register_fake
def _(data: torch.Tensor, position_ids: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export - just returns data tensor with same shape/dtype"""
    return torch.empty_like(data)


@torch.library.custom_op("qefficient::ctx_gather_3d", mutates_args=())
def ctx_gather_3d_op(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Custom 3D context gather operation"""
    batch_indices = torch.arange(data.shape[0]).view(-1, 1)
    return data[batch_indices, ctx_indices]


@ctx_gather_3d_op.register_fake
def _(data: torch.Tensor, ctx_indices: torch.Tensor) -> torch.Tensor:
    """Fake implementation for torch.export"""
    # Return tensor with shape [batch_size, seq_len]
    batch_size = data.shape[0]
    seq_len = ctx_indices.shape[1]
    return torch.empty(batch_size, seq_len, dtype=data.dtype, device=data.device)


@torch.library.custom_op("qefficient::ctx_gather", mutates_args=())
def ctx_gather_op(data: torch.Tensor, ctx_indices: torch.Tensor, comp_ctx_len: int) -> torch.Tensor:
    batch_indices = torch.arange(data.shape[0]).view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1]).view(1, -1, 1)
    return data[batch_indices, head_indices, ctx_indices]


@ctx_gather_op.register_fake
def _(
    data: torch.Tensor,
    ctx_indices: torch.Tensor,
    comp_ctx_len: int,
) -> torch.Tensor:
    return torch.empty_like(data)


# SCATTER CB (4D with heads, context, etc.)
@torch.library.custom_op("qefficient::ctx_scatter_cb", mutates_args=())
def ctx_scatter_cb_op(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """
    Custom 4D context scatter op with batch_index (CB version).
    Semantics: same as CtxScatterFuncCB.forward, but returns a new tensor.
    """
    # Clone to avoid aliasing issues with custom_op
    result = data.clone()
    batch_idx = batch_index.view(-1, 1, 1)
    head_idx = torch.arange(result.shape[1], device=result.device).view(1, -1, 1)
    ctx_idx = position_ids.unsqueeze(1)
    result[batch_idx, head_idx, ctx_idx] = updates
    return result


@ctx_scatter_cb_op.register_fake
def _(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """
    Fake implementation for torch.export: correct shape/dtype/device, values don't matter.
    Output shape matches data.
    """
    return torch.empty_like(data)


# SCATTER CB 3D
@torch.library.custom_op("qefficient::ctx_scatter_cb_3d", mutates_args=())
def ctx_scatter_cb_3d_op(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """
    Custom 3D context scatter op with batch_index (CB3D version).
    Semantics: same as CtxScatterFuncCB3D.forward but returns new tensor.
    """
    result = data.clone()
    batch_idx = batch_index.view(-1, 1)
    ctx_idx = position_ids
    result[batch_idx, ctx_idx] = updates
    return result


@ctx_scatter_cb_3d_op.register_fake
def _(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    position_ids: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """
    Fake implementation for torch.export: same shape/dtype/device as data.
    """
    return torch.empty_like(data)


# GATHER CB
@torch.library.custom_op("qefficient::ctx_gather_cb", mutates_args=())
def ctx_gather_cb_op(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    ctx_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Custom 4D context gather op with batch_index (CB version).
    Semantics: similar to CtxGatherFuncCB.forward.
    """
    batch_indices = batch_index.view(-1, 1, 1)
    head_indices = torch.arange(data.shape[1], device=data.device).view(1, -1, 1)
    return data[batch_indices, head_indices, ctx_indices]


@ctx_gather_cb_op.register_fake
def _(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    ctx_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Fake implementation for torch.export.

    We derive the output shape from input shapes:
    - batch_size: from batch_index
    - num_heads: from data
    - seq_len: from ctx_indices (dimension 1, typically)
    - hidden dims: from data (starting from dim 3)
    """
    batch_size = batch_index.shape[0]
    num_heads = data.shape[1]
    seq_len = ctx_indices.shape[1]

    # Remaining feature dimensions (e.g., head_dim or more)
    feature_shape = data.shape[3:]  # could be () if 3D

    out_shape = (batch_size, num_heads, seq_len, *feature_shape)
    return torch.empty(out_shape, dtype=data.dtype, device=data.device)


# GATHER CB 3D
@torch.library.custom_op("qefficient::ctx_gather_cb_3d", mutates_args=())
def ctx_gather_cb_3d_op(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    ctx_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Custom 3D context gather op with batch_index (CB3D version).
    Semantics: similar to CtxGatherFuncCB3D.forward.
    """
    batch_indices = batch_index.view(-1, 1)
    return data[batch_indices, ctx_indices]


@ctx_gather_cb_3d_op.register_fake
def _(
    data: torch.Tensor,
    batch_index: torch.Tensor,
    ctx_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Fake implementation for torch.export.

    Output shape:
    - batch_size: from batch_index
    - seq_len: from ctx_indices (dim 1)
    - any trailing dims from data
    """
    batch_size = batch_index.shape[0]
    seq_len = ctx_indices.shape[1]
    feature_shape = data.shape[2:]  # if data is [B, C], this is ()

    out_shape = (batch_size, seq_len, *feature_shape)
    return torch.empty(out_shape, dtype=data.dtype, device=data.device)
