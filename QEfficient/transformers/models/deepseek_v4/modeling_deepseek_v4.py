# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Optional, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


def qeff_apply_deepseek_v4_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """ONNX-friendly DeepSeek-V4 interleaved RoPE.

    Upstream Transformers expands the half-sized cos/sin tables with
    ``repeat_interleave`` and then rotates with a helper that stacks/slices the
    already-expanded tensor. The legacy torch ONNX tracer can infer an extra
    broadcast rank for that sequence in DeepSeek-V4 and then fail while lowering
    the final ``cat([nope, rotated])``. Keep the same math but operate on paired
    even/odd channels directly, so cos/sin stay half-width and broadcast with a
    stable rank during export.
    """
    cos = cos.reshape(cos.shape[0], cos.shape[1], -1).unsqueeze(unsqueeze_dim)
    sin = sin.reshape(sin.shape[0], sin.shape[1], -1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1] * 2
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rope_even = rope[..., 0::2].float()
    rope_odd = rope[..., 1::2].float()
    rotated_even = rope_even * cos - rope_odd * sin
    rotated_odd = rope_odd * cos + rope_even * sin
    rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(x.dtype)
    return torch.cat((nope, rotated), dim=-1)


def patch_deepseek_v4_rotary_for_qeff_export() -> None:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf_deepseek_v4

    if getattr(hf_deepseek_v4.apply_rotary_pos_emb, "_qeff_patched", False):
        return
    qeff_apply_deepseek_v4_rotary_pos_emb._qeff_patched = True
    hf_deepseek_v4.apply_rotary_pos_emb = qeff_apply_deepseek_v4_rotary_pos_emb


def qeff_deepseek_v4_cache_update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
        self.values = self.keys

    self.cumulative_length += key_states.shape[-2]
    full = torch.cat([self.keys, key_states], dim=-2)
    if self.keys.dim() == key_states.dim() and self.keys.shape[-2] >= key_states.shape[-2]:
        self.keys = torch.cat([self.keys[:, :, key_states.shape[-2] :, :], key_states], dim=-2)
    else:
        start = max(0, full.shape[-2] - self.sliding_window + 1)
        self.keys = full[:, :, start:, :]
    self.values = self.keys
    return full, full


def patch_deepseek_v4_cache_for_qeff_export() -> None:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf_deepseek_v4

    if getattr(hf_deepseek_v4.DeepseekV4HCACache.update, "_qeff_patched", False):
        return
    qeff_deepseek_v4_cache_update._qeff_patched = True
    hf_deepseek_v4.DeepseekV4HCACache.update = qeff_deepseek_v4_cache_update
    hf_deepseek_v4.DeepseekV4CSACache.update = qeff_deepseek_v4_cache_update


DEEPSEEK_V4_MASK_VALUE = -10000.0


DEEPSEEK_V4_HCA_STATE_NAMES = (
    "compressor_buffer_kv",
    "compressor_buffer_gate",
    "compressor_compressed_kv",
)
DEEPSEEK_V4_CSA_STATE_NAMES = (
    "compressor_buffer_kv",
    "compressor_buffer_gate",
    "compressor_compressed_kv",
    "compressor_overlap_kv",
    "compressor_overlap_gate",
    "indexer_buffer_kv",
    "indexer_buffer_gate",
    "indexer_compressed_kv",
    "indexer_overlap_kv",
    "indexer_overlap_gate",
)


def deepseek_v4_compression_state_names_for_layer(config, layer_idx: int) -> tuple[str, ...]:
    layer_type = config.layer_types[layer_idx]
    if layer_type == "compressed_sparse_attention":
        return DEEPSEEK_V4_CSA_STATE_NAMES
    if layer_type == "heavily_compressed_attention":
        return DEEPSEEK_V4_HCA_STATE_NAMES
    return ()


def get_deepseek_v4_compression_state_names(config) -> list[list[str]]:
    return [list(deepseek_v4_compression_state_names_for_layer(config, i)) for i in range(config.num_hidden_layers)]


def get_deepseek_v4_compression_state_shapes(config, batch_size: int, ctx_len: int) -> list[list[tuple[int, ...]]]:
    head_dim = config.head_dim
    index_head_dim = config.index_head_dim
    shapes = []
    for layer_type in config.layer_types:
        if layer_type == "compressed_sparse_attention":
            rate = config.compress_rates["compressed_sparse_attention"]
            comp_ctx = max(1, ctx_len // rate + 1)
            shapes.append(
                [
                    (batch_size, rate, 2 * head_dim),
                    (batch_size, rate, 2 * head_dim),
                    (batch_size, comp_ctx, head_dim),
                    (batch_size, rate, head_dim),
                    (batch_size, rate, head_dim),
                    (batch_size, rate, 2 * index_head_dim),
                    (batch_size, rate, 2 * index_head_dim),
                    (batch_size, comp_ctx, index_head_dim),
                    (batch_size, rate, index_head_dim),
                    (batch_size, rate, index_head_dim),
                ]
            )
        elif layer_type == "heavily_compressed_attention":
            rate = config.compress_rates["heavily_compressed_attention"]
            comp_ctx = max(1, ctx_len // rate + 1)
            shapes.append(
                [
                    (batch_size, rate, head_dim),
                    (batch_size, rate, head_dim),
                    (batch_size, comp_ctx, head_dim),
                ]
            )
        else:
            shapes.append([])
    return shapes


def get_deepseek_v4_compression_state_initializers(
    config,
    batch_size: int,
    ctx_len: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> list[list[torch.Tensor]]:
    state_names = get_deepseek_v4_compression_state_names(config)
    state_shapes = get_deepseek_v4_compression_state_shapes(config, batch_size, ctx_len)
    states = []
    for layer_names, layer_shapes in zip(state_names, state_shapes):
        layer_states = []
        for state_name, state_shape in zip(layer_names, layer_shapes):
            state = torch.zeros(state_shape, dtype=dtype, device=device)
            if state_name in {"compressor_overlap_gate", "indexer_overlap_gate"}:
                state = state.fill_(DEEPSEEK_V4_MASK_VALUE)
            layer_states.append(state)
        states.append(layer_states)
    return states


def _qeff_last_position(position_ids: torch.Tensor) -> torch.Tensor:
    return position_ids[:, -1]


def _qeff_slot(position_ids: torch.Tensor, rate: int) -> torch.Tensor:
    position = _qeff_last_position(position_ids)
    return position - torch.div(position, rate, rounding_mode="floor") * rate


def _qeff_slot_mask(position_ids: torch.Tensor, rate: int, dtype: torch.dtype) -> torch.Tensor:
    slot = _qeff_slot(position_ids, rate).view(-1, 1)
    entries = torch.arange(rate, device=position_ids.device).view(1, -1)
    return torch.eq(entries, slot).to(dtype)


def _qeff_ready_mask(position_ids: torch.Tensor, rate: int, dtype: torch.dtype) -> torch.Tensor:
    ready = torch.eq(_qeff_slot(position_ids, rate), rate - 1)
    return ready.to(dtype)


def _qeff_update_window_buffer(buffer: torch.Tensor, states: torch.Tensor, position_ids: torch.Tensor, rate: int):
    mask = _qeff_slot_mask(position_ids, rate, buffer.dtype).unsqueeze(-1)
    return buffer * (1.0 - mask) + states[:, -1:, :] * mask


def _qeff_scatter_compressed_state(
    compressed_state: torch.Tensor,
    compressed: torch.Tensor,
    position_ids: torch.Tensor,
    rate: int,
) -> torch.Tensor:
    comp_len = int(compressed_state.shape[1])
    write_index = torch.div(_qeff_last_position(position_ids), rate, rounding_mode="floor")
    write_index = torch.clamp(write_index, 0, comp_len - 1).view(-1, 1)
    entries = torch.arange(comp_len, device=position_ids.device).view(1, -1)
    write_mask = torch.eq(entries, write_index).to(compressed_state.dtype)
    ready = _qeff_ready_mask(position_ids, rate, compressed_state.dtype).view(-1, 1)
    write_mask = (write_mask * ready).unsqueeze(-1)
    return compressed_state * (1.0 - write_mask) + compressed[:, -1:, :] * write_mask


def _qeff_future_block_bias(reference: torch.Tensor, position_ids: torch.Tensor, rate: int) -> torch.Tensor:
    batch = reference.shape[0]
    seq_len = position_ids.shape[1]
    compressed_len = int(reference.shape[2])
    entry_indices = torch.arange(compressed_len, device=reference.device)
    causal_threshold = (position_ids + 1) // rate
    block_bias = reference.new_zeros((batch, 1, seq_len, compressed_len))
    return block_bias.masked_fill(
        entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
        DEEPSEEK_V4_MASK_VALUE,
    )


def _qeff_csa_compress_from_buffers(module, buffer_kv, buffer_gate, overlap_kv, overlap_gate, position_ids):
    rate = module.compress_rate
    new_kv = torch.cat((overlap_kv, buffer_kv[..., module.head_dim :]), dim=1).unsqueeze(1)
    new_gate = torch.cat((overlap_gate, buffer_gate[..., module.head_dim :]), dim=1).unsqueeze(1)
    compressed = module.kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2))
    positions = (position_ids[:, -1:] + 1 - rate).clamp(min=0)
    cos, sin = module.rotary_emb(compressed, position_ids=positions, layer_type=module.rope_layer_type)
    return qeff_apply_deepseek_v4_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)


def qeff_deepseek_v4_hca_compressor_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
    cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
    if cache_layer is None or not getattr(cache_layer, "qeff_has_compression_states", False):
        return self._qeff_original_forward(hidden_states, q_residual, position_ids, past_key_values, layer_idx)

    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    cache_layer.buffer_kv["compressor"] = _qeff_update_window_buffer(
        cache_layer.buffer_kv["compressor"], kv, position_ids, self.compress_rate
    )
    cache_layer.buffer_gate["compressor"] = _qeff_update_window_buffer(
        cache_layer.buffer_gate["compressor"], gate, position_ids, self.compress_rate
    )
    compressed_kv = cache_layer.compressed_kv["compressor"].unsqueeze(1)
    if int(cache_layer.compressed_kv["compressor"].shape[1]) == 1:
        return compressed_kv, _qeff_future_block_bias(compressed_kv, position_ids, self.compress_rate)

    gate_with_bias = cache_layer.buffer_gate["compressor"] + self.position_bias
    compressed = self.kv_norm(
        (cache_layer.buffer_kv["compressor"] * gate_with_bias.softmax(dim=1, dtype=torch.float32).to(kv.dtype)).sum(
            dim=1, keepdim=True
        )
    )
    positions = (position_ids[:, -1:] + 1 - self.compress_rate).clamp(min=0)
    cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
    compressed = qeff_apply_deepseek_v4_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
    cache_layer.compressed_kv["compressor"] = _qeff_scatter_compressed_state(
        cache_layer.compressed_kv["compressor"], compressed, position_ids, self.compress_rate
    )
    compressed_kv = cache_layer.compressed_kv["compressor"].unsqueeze(1)
    return compressed_kv, _qeff_future_block_bias(compressed_kv, position_ids, self.compress_rate)


def _qeff_update_csa_indexer_states(module, hidden_states, position_ids, past_key_values, layer_idx):
    cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
    kv = module.kv_proj(hidden_states)
    gate = module.gate_proj(hidden_states)
    cache_layer.buffer_kv["indexer"] = _qeff_update_window_buffer(
        cache_layer.buffer_kv["indexer"], kv, position_ids, module.compress_rate
    )
    cache_layer.buffer_gate["indexer"] = _qeff_update_window_buffer(
        cache_layer.buffer_gate["indexer"], gate, position_ids, module.compress_rate
    )
    buffer_gate = cache_layer.buffer_gate["indexer"] + module.position_bias
    compressed = _qeff_csa_compress_from_buffers(
        module,
        cache_layer.buffer_kv["indexer"],
        buffer_gate,
        cache_layer.overlap_kv["indexer"],
        cache_layer.overlap_gate["indexer"],
        position_ids,
    )
    cache_layer.compressed_kv["indexer"] = _qeff_scatter_compressed_state(
        cache_layer.compressed_kv["indexer"], compressed, position_ids, module.compress_rate
    )
    ready = torch.eq(_qeff_slot(position_ids, module.compress_rate), module.compress_rate - 1).view(-1, 1, 1)
    current_overlap_kv = cache_layer.buffer_kv["indexer"][..., : module.head_dim]
    current_overlap_gate = buffer_gate[..., : module.head_dim]
    cache_layer.overlap_kv["indexer"] = torch.where(ready, current_overlap_kv, cache_layer.overlap_kv["indexer"])
    cache_layer.overlap_gate["indexer"] = torch.where(ready, current_overlap_gate, cache_layer.overlap_gate["indexer"])
    return cache_layer


def qeff_deepseek_v4_indexer_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
    cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
    if cache_layer is None or not getattr(cache_layer, "qeff_has_compression_states", False):
        return self._qeff_original_forward(hidden_states, q_residual, position_ids, past_key_values, layer_idx)

    cache_layer = _qeff_update_csa_indexer_states(self, hidden_states, position_ids, past_key_values, layer_idx)
    compressed_kv = cache_layer.compressed_kv["indexer"]
    compressed_len = int(compressed_kv.shape[1])
    causal_threshold = (position_ids + 1) // self.compress_rate
    if compressed_len <= self.index_topk:
        batch, seq_len = hidden_states.shape[:2]
        entry_indices = torch.arange(compressed_len, device=compressed_kv.device).view(1, 1, -1)
        all_indices = entry_indices.expand(batch, seq_len, -1)
        invalid = all_indices >= causal_threshold.unsqueeze(-1)
        return torch.where(invalid, torch.full_like(all_indices, -1), all_indices)

    cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
    batch, seq_len, _ = hidden_states.shape
    q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
    q = qeff_apply_deepseek_v4_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)
    index_scores = self.scorer(q, compressed_kv, hidden_states)
    entry_indices = torch.arange(compressed_len, device=index_scores.device)
    future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
    index_scores = index_scores.masked_fill(future_mask, DEEPSEEK_V4_MASK_VALUE)
    top_k = min(self.index_topk, compressed_len)
    top_k_indices = index_scores.topk(top_k, dim=-1).indices
    invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)


def _qeff_block_bias_from_topk(reference: torch.Tensor, top_k_indices: torch.Tensor) -> torch.Tensor:
    compressed_len = int(reference.shape[2])
    entry_indices = torch.arange(compressed_len, device=reference.device).view(1, 1, 1, 1, -1)
    selected = top_k_indices.unsqueeze(1).unsqueeze(-1) == entry_indices
    valid = top_k_indices.unsqueeze(1).unsqueeze(-1) >= 0
    keep = (selected & valid).to(reference.dtype).sum(dim=-2) > 0
    return torch.where(
        keep,
        torch.zeros((), dtype=reference.dtype, device=reference.device),
        torch.full((), DEEPSEEK_V4_MASK_VALUE, dtype=reference.dtype, device=reference.device),
    )


def qeff_deepseek_v4_csa_compressor_forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
    cache_layer = past_key_values.layers[layer_idx] if past_key_values is not None else None
    if cache_layer is None or not getattr(cache_layer, "qeff_has_compression_states", False):
        return self._qeff_original_forward(hidden_states, q_residual, position_ids, past_key_values, layer_idx)

    kv = self.kv_proj(hidden_states)
    gate = self.gate_proj(hidden_states)
    cache_layer.buffer_kv["compressor"] = _qeff_update_window_buffer(
        cache_layer.buffer_kv["compressor"], kv, position_ids, self.compress_rate
    )
    cache_layer.buffer_gate["compressor"] = _qeff_update_window_buffer(
        cache_layer.buffer_gate["compressor"], gate, position_ids, self.compress_rate
    )
    buffer_gate = cache_layer.buffer_gate["compressor"] + self.position_bias
    compressed = _qeff_csa_compress_from_buffers(
        self,
        cache_layer.buffer_kv["compressor"],
        buffer_gate,
        cache_layer.overlap_kv["compressor"],
        cache_layer.overlap_gate["compressor"],
        position_ids,
    )
    cache_layer.compressed_kv["compressor"] = _qeff_scatter_compressed_state(
        cache_layer.compressed_kv["compressor"], compressed, position_ids, self.compress_rate
    )
    ready = torch.eq(_qeff_slot(position_ids, self.compress_rate), self.compress_rate - 1).view(-1, 1, 1)
    current_overlap_kv = cache_layer.buffer_kv["compressor"][..., : self.head_dim]
    current_overlap_gate = buffer_gate[..., : self.head_dim]
    cache_layer.overlap_kv["compressor"] = torch.where(ready, current_overlap_kv, cache_layer.overlap_kv["compressor"])
    cache_layer.overlap_gate["compressor"] = torch.where(
        ready, current_overlap_gate, cache_layer.overlap_gate["compressor"]
    )

    compressed_kv = cache_layer.compressed_kv["compressor"].unsqueeze(1)
    if int(compressed_kv.shape[2]) <= self.indexer.index_topk:
        _qeff_update_csa_indexer_states(self.indexer, hidden_states, position_ids, past_key_values, layer_idx)
        return compressed_kv, _qeff_future_block_bias(compressed_kv, position_ids, self.compress_rate)
    top_k_indices = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
    return compressed_kv, _qeff_block_bias_from_topk(compressed_kv, top_k_indices)


def patch_deepseek_v4_compressors_for_qeff_export() -> None:
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf_deepseek_v4

    patches = (
        (hf_deepseek_v4.DeepseekV4HCACompressor, qeff_deepseek_v4_hca_compressor_forward),
        (hf_deepseek_v4.DeepseekV4CSACompressor, qeff_deepseek_v4_csa_compressor_forward),
        (hf_deepseek_v4.DeepseekV4Indexer, qeff_deepseek_v4_indexer_forward),
    )
    for cls, forward in patches:
        if getattr(cls.forward, "_qeff_patched", False):
            continue
        cls._qeff_original_forward = cls.forward
        forward._qeff_patched = True
        cls.forward = forward


def _assign_deepseek_v4_compression_states(layer, layer_type: str, states: list[torch.Tensor]) -> None:
    if not states:
        return
    layer.qeff_has_compression_states = True
    if layer_type == "heavily_compressed_attention":
        layer.buffer_kv["compressor"] = states[0]
        layer.buffer_gate["compressor"] = states[1]
        layer.compressed_kv["compressor"] = states[2]
    elif layer_type == "compressed_sparse_attention":
        layer.buffer_kv["compressor"] = states[0]
        layer.buffer_gate["compressor"] = states[1]
        layer.compressed_kv["compressor"] = states[2]
        layer.overlap_kv["compressor"] = states[3]
        layer.overlap_gate["compressor"] = states[4]
        layer.buffer_kv["indexer"] = states[5]
        layer.buffer_gate["indexer"] = states[6]
        layer.compressed_kv["indexer"] = states[7]
        layer.overlap_kv["indexer"] = states[8]
        layer.overlap_gate["indexer"] = states[9]


def extract_deepseek_v4_compression_states(cache: Cache) -> list[list[torch.Tensor]]:
    states = []
    for layer in cache.layers:
        if not getattr(layer, "qeff_has_compression_states", False):
            states.append([])
        elif getattr(layer, "layer_type", None) == "compressed_sparse_attention":
            states.append(
                [
                    layer.buffer_kv["compressor"],
                    layer.buffer_gate["compressor"],
                    layer.compressed_kv["compressor"],
                    layer.overlap_kv["compressor"],
                    layer.overlap_gate["compressor"],
                    layer.buffer_kv["indexer"],
                    layer.buffer_gate["indexer"],
                    layer.compressed_kv["indexer"],
                    layer.overlap_kv["indexer"],
                    layer.overlap_gate["indexer"],
                ]
            )
        else:
            states.append(
                [
                    layer.buffer_kv["compressor"],
                    layer.buffer_gate["compressor"],
                    layer.compressed_kv["compressor"],
                ]
            )
    return states


def rename_deepseek_v4_compression_onnx_inputs(onnx_model, config):
    expected_names = []
    for layer_idx, layer_names in enumerate(get_deepseek_v4_compression_state_names(config)):
        expected_names.extend(f"deepseek_v4_{state_name}.{layer_idx}" for state_name in layer_names)
    if not expected_names or len(onnx_model.graph.input) < len(expected_names):
        return onnx_model

    start = len(onnx_model.graph.input) - len(expected_names)
    old_names = [onnx_model.graph.input[idx].name for idx in range(start, len(onnx_model.graph.input))]
    tmp_names = [f"{old_name}__qeff_tmp_{idx}" for idx, old_name in enumerate(old_names)]

    def replace_name(old_name, new_name):
        for graph_input in onnx_model.graph.input:
            if graph_input.name == old_name:
                graph_input.name = new_name
        for graph_output in onnx_model.graph.output:
            if graph_output.name == old_name:
                graph_output.name = new_name
        for value_info in onnx_model.graph.value_info:
            if value_info.name == old_name:
                value_info.name = new_name
        for node in onnx_model.graph.node:
            for input_idx, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[input_idx] = new_name
            for output_idx, output_name in enumerate(node.output):
                if output_name == old_name:
                    node.output[output_idx] = new_name

    for old_name, tmp_name in zip(old_names, tmp_names):
        replace_name(old_name, tmp_name)
    for tmp_name, expected_name in zip(tmp_names, expected_names):
        replace_name(tmp_name, expected_name)

    for graph_output in list(onnx_model.graph.output):
        if graph_output.name.endswith("_InternalRetainedState"):
            replace_name(
                graph_output.name,
                graph_output.name[: -len("_InternalRetainedState")] + "_RetainedState",
            )
    return onnx_model


def build_deepseek_v4_cache(config, legacy_cache=None, compression_states=None) -> Cache:
    """Build DeepSeek-V4 CSA/HCA cache layers from native Transformers classes.

    DeepSeek-V4 attention layers need cache layers with compressor/indexer state
    methods such as ``store_compression_weights``. Generic ``DynamicCache`` still
    instantiates plain ``DynamicLayer`` for V4 layer types in Transformers 5.10.1,
    so QEff builds the native V4 cache layers explicitly. During ONNX export,
    QEff passes legacy tensor tuples; ``legacy_cache`` seeds the sliding KV rows
    into those native cache layers before forward runs.
    """
    patch_deepseek_v4_rotary_for_qeff_export()
    patch_deepseek_v4_cache_for_qeff_export()
    patch_deepseek_v4_compressors_for_qeff_export()

    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4CSACache, DeepseekV4HCACache

    legacy_cache = tuple(legacy_cache or ())
    compression_states = compression_states or [[] for _ in range(config.num_hidden_layers)]
    layers = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "compressed_sparse_attention":
            layer = DeepseekV4CSACache(config)
        else:
            layer = DeepseekV4HCACache(config)

        if layer_idx < len(legacy_cache) and len(legacy_cache[layer_idx]) >= 2:
            key_states, value_states = legacy_cache[layer_idx][0], legacy_cache[layer_idx][1]
            layer.keys = key_states
            layer.values = key_states if value_states is None else value_states
            layer.dtype = key_states.dtype
            layer.device = key_states.device
            layer.is_initialized = True
            layer.cumulative_length = int(key_states.shape[-2])
        if layer_idx < len(compression_states):
            _assign_deepseek_v4_compression_states(layer, layer_type, list(compression_states[layer_idx]))
        layers.append(layer)
    return Cache(layers=layers)


class QEffDeepseekV4RMSNorm(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from QEfficient.customop.rms_norm import CustomRMSNormFunc

        return CustomRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)


class QEffDeepseekV4Experts(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        expanded_hidden_states = hidden_states.unsqueeze(0).expand(self.num_experts, tokens, self.hidden_dim)
        gate_up = torch.bmm(expanded_hidden_states, self.gate_up_proj.transpose(1, 2))
        expert_hidden_states = self._apply_gate(gate_up)
        expert_outputs = torch.bmm(expert_hidden_states, self.down_proj.transpose(1, 2))

        expert_outputs = expert_outputs.transpose(0, 1)
        gather_index = top_k_index.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        selected_outputs = torch.gather(expert_outputs, 1, gather_index)
        return (selected_outputs * top_k_weights.unsqueeze(-1)).sum(dim=1).to(hidden_states.dtype)


class QEffDeepseekV4ForCausalLM(nn.Module):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return set()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        deepseek_v4_compression_states: Optional[list[list[torch.Tensor]]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        patch_deepseek_v4_rotary_for_qeff_export()
        patch_deepseek_v4_cache_for_qeff_export()
        patch_deepseek_v4_compressors_for_qeff_export()

        if past_key_values is None:
            past_key_values = build_deepseek_v4_cache(self.config, compression_states=deepseek_v4_compression_states)
        elif not isinstance(past_key_values, Cache):
            past_key_values = build_deepseek_v4_cache(
                self.config, past_key_values, compression_states=deepseek_v4_compression_states
            )
        elif deepseek_v4_compression_states is not None:
            for layer_idx, states in enumerate(deepseek_v4_compression_states):
                if layer_idx < len(past_key_values.layers):
                    _assign_deepseek_v4_compression_states(
                        past_key_values.layers[layer_idx], self.config.layer_types[layer_idx], list(states)
                    )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if position_ids is not None:
            hidden_states = hidden_states[:, -1:, :]
        elif isinstance(logits_to_keep, int) and logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif not isinstance(logits_to_keep, int):
            hidden_states = hidden_states[:, logits_to_keep, :]

        logits = self.lm_head(hidden_states)
        state_anchor = None
        if isinstance(past_key_values, Cache):
            for layer in past_key_values.layers:
                for state in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                    if isinstance(state, torch.Tensor) and state.numel() > 0:
                        state_value = state.reshape(-1)[0].to(logits.dtype)
                        state_anchor = state_value if state_anchor is None else state_anchor + state_value
        if deepseek_v4_compression_states is not None:
            for layer_states in deepseek_v4_compression_states:
                for state in layer_states:
                    state_value = state.reshape(-1)[0].to(logits.dtype)
                    state_anchor = state_value if state_anchor is None else state_anchor + state_value
        if state_anchor is not None:
            logits = logits + (state_anchor - state_anchor)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        return build_deepseek_v4_cache(config)
