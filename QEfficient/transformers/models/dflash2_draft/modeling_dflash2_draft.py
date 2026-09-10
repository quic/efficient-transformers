# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# PyTorch DFlash-2 draft model (Qwen3-based diffusion LLM draft for SpD).

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import onnx
import torch
import torch.utils.checkpoint
import yaml
from torch import nn
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RMSNorm,
    repeat_kv,
    rotate_half,
)

from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

CONV_MODULE_NAMES = ("attention_conv", "mlp_conv")


def _draft_config(config) -> dict:
    """The ``dflash_config`` sub-dict, mirroring upstream ``_draft_config``."""
    return getattr(config, "dflash_config", None) or {}


def _draft_value(config, name: str, default=None):
    """A key in ``dflash_config`` wins, then a top-level config attribute, then ``default``.

    Mirrors upstream ``_draft_value`` exactly so a stock DFlash-2 checkpoint config works
    unchanged.
    """
    return _draft_config(config).get(name, getattr(config, name, default))


def _required_draft_value(config, name: str) -> int:
    value = _draft_value(config, name)
    if value is None:
        raise ValueError(
            f"DFlash-2 requires {name!r}, but it is present neither in config.dflash_config nor as a "
            f"top-level config attribute. Got dflash_config={_draft_config(config)!r}."
        )
    return value


def _create_mask(
    position_ids: torch.Tensor,
    target_length: int,
    causal: bool = False,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Build the noise-block attention mask for one layer type.

    Args:
        position_ids: ``[bsz, num_queries]`` absolute positions of the noise block.
        target_length: physical length of the K/V that ``eager_attention_forward`` will see for
            this layer type -- i.e. the per-layer cache capacity, NOT a logical sequence length.
        causal: whether this layer is causal within the noise block. Comes from
            ``DFlash2Attention.is_causal``, i.e. ``config.is_causal`` when set, else
            ``layer_type == "sliding_attention"``.
        sliding_window: ``config.sliding_window`` on sliding layers, ``None`` on full layers.

    Returns:
        ``[bsz, 1, num_queries, target_length]`` bool tensor; ``True`` means MASK OUT.
    """
    bsz, num_queries = position_ids.shape

    slots = torch.arange(target_length).view(1, 1, target_length)
    row_max = position_ids.max(dim=-1, keepdim=True).values.unsqueeze(-1)

    invalid = None
    if sliding_window is None:
        kv_positions = slots  # [1, 1, target_length]
    else:
        wraps = torch.div(row_max - slots, target_length, rounding_mode="floor")
        kv_positions = slots + target_length * wraps  # [bsz, 1, target_length]
        invalid = kv_positions < 0

    cutoff = position_ids.unsqueeze(-1) if causal else row_max
    attention_mask = kv_positions > cutoff

    if invalid is not None:
        attention_mask = attention_mask | invalid

    if sliding_window is not None and sliding_window < target_length:
        attention_mask = attention_mask | (kv_positions <= position_ids.unsqueeze(-1) - sliding_window)

    return attention_mask.expand(bsz, num_queries, target_length).unsqueeze(1)


def qeff_apply_rope_two_streams(
    q_noise: torch.Tensor,
    k_ctx: torch.Tensor,
    k_noise: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_ctx: int,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Apply rope to the two-stream (context + noise) attention inputs.

    cos_ctx = cos[:, :n_ctx].unsqueeze(unsqueeze_dim)
    sin_ctx = sin[:, :n_ctx].unsqueeze(unsqueeze_dim)
    cos_noise = cos[:, n_ctx:].unsqueeze(unsqueeze_dim)
    sin_noise = sin[:, n_ctx:].unsqueeze(unsqueeze_dim)

    k_ctx_embed = k_ctx * cos_ctx + rotate_half(k_ctx) * sin_ctx
    q_noise_embed = q_noise * cos_noise + rotate_half(q_noise) * sin_noise
    k_noise_embed = k_noise * cos_noise + rotate_half(k_noise) * sin_noise

    return (
        q_noise_embed.to(q_noise.dtype),
        k_ctx_embed.to(k_ctx.dtype),
        k_noise_embed.to(k_noise.dtype),
    )


def _causal_shift(hidden: torch.Tensor, offset: int) -> torch.Tensor:
    """Shift ``[bsz, length, hidden]`` right along ``length`` by ``offset``, zero-filling the front.

    ``offset`` is a Python int, so this lowers to a static Slice + Concat against a constant and
    stays shape-agnostic in ``length`` (safe across specializations). Upstream writes this as
    ``F.pad(blocks[:, :-offset], (0, 0, 0, 0, offset, 0))`` on a 4-D view, where the 6-tuple pads
    the third-from-last axis; the explicit concat is the same thing with less to get wrong.

    Taps at or beyond ``length`` reach entirely into the zero left-context, so they contribute
    nothing. Upstream raises a shape error there (its ``F.pad`` grows the tensor instead of
    clamping); returning zeros is the arithmetically correct answer. It only matters for
    ``conv_kernel_size > block_size``, which no real config uses.
    """
    if offset == 0:
        return hidden
    if offset >= hidden.shape[1]:
        return torch.zeros_like(hidden)
    pad = hidden.new_zeros((hidden.shape[0], offset, hidden.shape[2]))
    return torch.cat([pad, hidden[:, :-offset]], dim=1)


class GroupedDynamicCausalConv(nn.Module):
    """Depthwise causal conv with static per-channel + dynamic per-group tap weights.

    out[b,t,g,c] = sum_o ( base[o, g*gs+c] + dynamic[b,t,o,g] ) * hidden[b, t-o, g*gs+c]
    """

    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        if hidden_size % group_size != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by conv_group_size {group_size}")
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * self.groups, bias=False)

    def _convolve(self, hidden: torch.Tensor, dynamic: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[B, L, H]``, ``dynamic`` ``[B, L, K, groups]``, ``base`` ``[K, H]``."""
        bsz, length, hidden_size = hidden.shape
        k, groups, group_size = self.kernel_size, self.groups, self.group_size
        shifted = torch.stack([_causal_shift(hidden, offset) for offset in range(k)], dim=1)
        shifted = shifted.view(bsz, k, length, groups, group_size)

        weight = base.view(1, k, 1, groups, group_size).to(hidden.dtype) + dynamic.permute(0, 2, 1, 3).unsqueeze(-1)
        return (weight * shifted).sum(dim=1).view(bsz, length, hidden_size)

    def prepare(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convolve the sublayer input; also return the dynamic kernel that ``finish`` will use."""
        dynamic = self.kernel_projection(hidden).view(*hidden.shape[:-1], 2, self.kernel_size, self.groups)
        return (
            self._convolve(hidden, dynamic[..., 0, :, :], self.base_kernel[0]),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        """Convolve the sublayer output with the kernel stashed by ``prepare``."""
        return self._convolve(hidden, dynamic, self.base_kernel[1])


class CandidateSelector(nn.Module):
    # Bigram rescore over per-position top-k candidates.
    def __init__(self, config):
        super().__init__()
        rank = int(_required_draft_value(config, "selector_rank"))
        self.rank = rank
        self.top_k = int(_required_draft_value(config, "selector_top_k"))
        self.predecessor_codebook = nn.Embedding(config.vocab_size, rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, rank)
        self.hidden_projection = nn.Linear(config.hidden_size, rank, bias=False)

    @torch.no_grad()
    def select(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Greedy bigram-rescored path. CPU reference; mirrors upstream ``select`` at temperature 0.

        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        projected = self.hidden_projection(hidden)
        predecessor = anchor_ids
        path = []
        for position in range(block_size if block_size is not None else hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * projected[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            index = torch.argmax(scores, dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        return torch.stack(path, dim=1), candidates


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups).to(query.dtype)
    value_states = repeat_kv(value, module.num_key_value_groups).to(query.dtype)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DFlash2Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_types = getattr(config, "layer_types", None)
        layer_type = layer_types[layer_idx] if layer_types else "full_attention"
        is_causal = getattr(config, "is_causal", None)
        self.is_causal = layer_type == "sliding_attention" if is_causal is None else bool(is_causal)
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids_target: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        kwargs.pop("output_attentions", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("labels", None)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k_ctx = self.k_norm(k_ctx.view(bsz, ctx_len, -1, self.head_dim)).transpose(1, 2)
        k_noise = self.k_norm(k_noise.view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        v_ctx = v_ctx.view(bsz, ctx_len, -1, self.head_dim).transpose(1, 2)
        v_noise = v_noise.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, k_ctx, k_noise = qeff_apply_rope_two_streams(query_states, k_ctx, k_noise, cos, sin, ctx_len)

        key_states, value_states = k_noise, v_noise
        if past_key_value is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids_target}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            # Seed the verified context first; write_only does not advance seen_tokens.
            past_key_value.write_only(k_ctx, v_ctx, self.layer_idx, cache_kwargs)

            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(k_noise, v_noise, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class DFlash2DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace Qwen3's single-stream attention with the two-stream variant.
        self.self_attn = DFlash2Attention(config=config, layer_idx=layer_idx)

        kernel_size = int(_required_draft_value(config, "conv_kernel_size"))
        group_size = int(_required_draft_value(config, "conv_group_size"))
        self.attention_conv = GroupedDynamicCausalConv(config.hidden_size, kernel_size, group_size)
        self.mlp_conv = GroupedDynamicCausalConv(config.hidden_size, kernel_size, group_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor = None,
        position_ids_target: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_kernel = None
        if self.attention_conv is not None:
            hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_ids_target=position_ids_target,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if attention_kernel is not None:
            hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_kernel = None
        if self.mlp_conv is not None:
            hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if mlp_kernel is not None:
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        hidden_states = residual + hidden_states

        return hidden_states


class DFlash2Model(Qwen3Model):
    def __init__(self, config: Qwen3Config):
        num_hidden_layers = config.num_hidden_layers
        config.num_hidden_layers = 0
        try:
            super().__init__(config)
        finally:
            config.num_hidden_layers = num_hidden_layers

        self.layers = nn.ModuleList([DFlash2DecoderLayer(config, layer_idx) for layer_idx in range(num_hidden_layers)])
        self.candidate_selector = CandidateSelector(config)
        self.block_size = int(_draft_value(config, "block_size", 16))
        self.mask_token_id = _draft_value(config, "mask_token_id")
        self.__qeff_init__()
        self.post_init()

    def __qeff_init__(self):
        full_layer = next(
            (index + 1 for index, layer_type in enumerate(self.config.layer_types) if layer_type == "full_attention"),
            self.config.num_hidden_layers,
        )
        self.config._sliding_window_pattern = full_layer

    def forward(
        self,
        target_hidden: torch.Tensor = None,
        noise_embeds: torch.FloatTensor = None,
        position_ids_target: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        from QEfficient.transformers.cache_utils import QEffSlidingWindowCache

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if noise_embeds is None:
            raise ValueError(
                "noise_embeds is required. Embed the mask/noise token ids at the "
                "DFlash2ForCausalLM level (which also applies input_embedding_scale) and pass the "
                "result here."
            )

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, (Cache, QEffSlidingWindowCache)):
            if past_key_values is None:
                raise ValueError(
                    "past_key_values must be a QEffSlidingWindowCache or a legacy list of per-layer "
                    "(key, value) tensors. The draft cannot allocate one itself: the sliding-window "
                    "and full-attention cache capacities come from the compiled specialization."
                )
            return_legacy_cache = True
            past_key_values = QEffSlidingWindowCache.from_legacy_cache(self.config, past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + noise_embeds.shape[1])
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        sliding_len = getattr(past_key_values, "sliding_window_len", None)
        full_len = getattr(past_key_values, "max_cache_len", None)
        if isinstance(attention_mask, torch.Tensor):
            sliding_len = full_len = attention_mask.shape[-1]

        layer_types = getattr(self.config, "layer_types", None) or ["sliding_attention"] * len(self.layers)
        mask_mapping: Dict[str, Optional[torch.Tensor]] = {}
        for layer_type, length in (("full_attention", full_len), ("sliding_attention", sliding_len)):
            if layer_type not in layer_types:
                continue

            reference_layer = self.layers[layer_types.index(layer_type)]
            mask_mapping[layer_type] = _create_mask(
                position_ids=position_ids,
                target_length=length,
                causal=reference_layer.self_attn.is_causal,
                sliding_window=reference_layer.self_attn.sliding_window,
            )

        hidden_states = noise_embeds

        position_ids_combined = torch.cat([position_ids_target, position_ids], dim=-1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids_combined)

        all_hidden_states = () if output_hidden_states else None

        for index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                target_hidden=target_hidden,
                position_ids_target=position_ids_target,
                attention_mask=mask_mapping[layer_types[index]],
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class DFlash2ForCausalLM(Qwen3ForCausalLM):
    emit_selector_hidden: bool = False

    def __init__(self, config: Qwen3Config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = DFlash2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.input_embedding_scale = float(_draft_value(config, "input_embedding_scale", 1.0))
        self.post_init()

    def generate_npi_file(self, onnx_path: Union[str, Path], model_name: Optional[str] = None) -> str:
        del model_name
        onnx_path = Path(onnx_path)
        model = onnx.load(str(onnx_path), load_external_data=False)
        fp32_ops = {"CustomRMSNorm", "Sigmoid", "Softmax", "Tanh"}
        nodes = [*model.graph.node, *(node for function in model.functions for node in function.node)]
        fp32_names = [
            output_name
            for node in nodes
            if node.op_type in fp32_ops or any(name in node.name for name in CONV_MODULE_NAMES)
            for output_name in node.output
            if output_name
        ]
        npi_path = onnx_path.with_name(f"{onnx_path.stem}_dflash2_draft_npi.yaml")
        with open(npi_path, "w") as fp:
            yaml.safe_dump({"FP32NodeInstanceNames": list(dict.fromkeys(fp32_names))}, fp, sort_keys=False)
        return str(npi_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = kwargs.pop("config", None)
        hub_kwargs = {key: kwargs.pop(key) for key in ("cache_dir", "revision", "token") if key in kwargs}
        trust_remote_code = kwargs.pop("trust_remote_code", False)
        dtype = kwargs.pop("dtype", None) or kwargs.pop("torch_dtype", None)

        for ignored in ("attn_implementation", "attn_implementation_autoset", "low_cpu_mem_usage", "use_safetensors"):
            kwargs.pop(ignored, None)
        if kwargs:
            warnings.warn(f"DFlash2ForCausalLM.from_pretrained ignoring unsupported kwargs: {sorted(kwargs)}")

        if config is None:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **hub_kwargs
            )

        model = cls(config, *args)
        state_dict = _read_checkpoint_tensors(pretrained_model_name_or_path, **hub_kwargs)
        missing, unexpected = model.load_state_dict(remap_dflash2_state_dict(state_dict), strict=False)

        if unexpected:
            warnings.warn(f"DFlash2ForCausalLM: unexpected checkpoint keys ignored: {sorted(unexpected)}")
        unexplained = [
            key for key in missing if not key.endswith(("embed_tokens.weight", "lm_head.weight", "lm_head.bias"))
        ]
        if unexplained:
            warnings.warn(f"DFlash2ForCausalLM: randomly initialized parameters: {sorted(unexplained)}")

        if dtype is not None:
            model.config.torch_dtype = dtype
            model = model.to(dtype)
        return model

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states).float()
        logits = logits * float(_draft_value(self.config, "output_multiplier", 1.0))
        softcap = _draft_value(self.config, "final_logit_softcapping")
        if softcap is not None and float(softcap) > 0:
            softcap = float(softcap)
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def forward(
        self,
        target_hidden: torch.Tensor = None,
        noise_embeds: torch.FloatTensor = None,
        position_ids_target: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) == (noise_embeds is None):
            raise ValueError("Specify exactly one of input_ids or noise_embeds.")
        if input_ids is not None:
            noise_embeds = self.model.embed_tokens(input_ids)
            if self.input_embedding_scale != 1.0:
                noise_embeds = noise_embeds * self.input_embedding_scale

        outputs = self.model(
            target_hidden=target_hidden,
            noise_embeds=noise_embeds,
            position_ids_target=position_ids_target,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        logits = self.compute_logits(outputs.last_hidden_state)

        hidden_states = outputs.hidden_states
        if getattr(self, "emit_selector_hidden", False):
            selector = getattr(self.model, "candidate_selector", None)
            if selector is None:
                raise RuntimeError(
                    "emit_selector_hidden was requested but candidate_selector is absent from the "
                    "draft. DFlash2DLMTransform drops it at selector_top_k <= 1, where the rescore "
                    "provably degenerates to logits.argmax(-1) and the host needs no extra output."
                )
            hidden_states = selector.hidden_projection(outputs.last_hidden_state)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )


def _read_checkpoint_tensors(checkpoint_path, **hub_kwargs) -> Dict[str, torch.Tensor]:
    """Read every tensor from a checkpoint given as a local directory or an HF repo id."""
    path = Path(checkpoint_path)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                str(checkpoint_path), allow_patterns=["*.safetensors", "pytorch_model*.bin"], **hub_kwargs
            )
        )

    state_dict: Dict[str, torch.Tensor] = {}
    shards = sorted(path.glob("*.safetensors"))
    if shards:
        from safetensors import safe_open

        for shard in shards:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                state_dict.update({key: handle.get_tensor(key) for key in handle.keys()})
    else:
        for shard in sorted(path.glob("pytorch_model*.bin")):
            state_dict.update(torch.load(str(shard), map_location="cpu", weights_only=True))
    if not state_dict:
        raise FileNotFoundError(f"No .safetensors or pytorch_model*.bin weights found in {path}")
    return state_dict


def remap_dflash2_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    codebooks = ("candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook")
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key in ("fc.weight", "hidden_norm.weight"):
            continue
        if key in codebooks:
            key = f"{key}.weight"
        if key.startswith(("layers.", "norm.", "embed_tokens.", "candidate_selector.", "rotary_emb.")):
            key = f"model.{key}"
        remapped[key] = value
    return remapped
