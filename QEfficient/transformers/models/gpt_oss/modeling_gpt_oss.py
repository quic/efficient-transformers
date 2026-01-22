# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import math
import os
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssConfig,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssForCausalLM,
    GptOssMLP,
    GptOssModel,
    GptOssRotaryEmbedding,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from QEfficient.transformers.cache_utils import QEffHybridCacheForGPTOSS
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE
from QEfficient.utils.logging_utils import logger
import math


class QEffGptOssExperts(GptOssExperts):
    def __qeff_init__(self):
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.expert_dim))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.expert_dim))
        self.gate_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.expert_dim))
        self.up_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.expert_dim))


class QEffPrefillOnlyChunkedGptOssMLP(GptOssMLP):
    def forward(self, hidden: torch.Tensor):
        B, S, H = hidden.shape
        T = B * S
        hidden = hidden.view(T, H)

        # Router computation
        router_logits = F.linear(hidden, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, top_i = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        masked_logits = torch.zeros_like(router_logits)
        masked_logits.scatter_(1, top_i, top_w)

        # Routing weights for each expert [T, E]
        routing_weights = masked_logits

        # ────────────────── allocate the output tensor ─────
        expert_out = hidden.new_zeros((T, H))  # accumulation buffer

        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)  # [T, 1]

            W_g, W_u = self.experts.gate_proj[e], self.experts.up_proj[e]  # [H, I], [H, I]
            b_g, b_u = self.experts.gate_proj_bias[e], self.experts.up_proj_bias[e]  # [I], [I]
            W_d = self.experts.down_proj[e]  # [I, H]
            b_d = self.experts.down_proj_bias[e]  # [H]

            # Gate and Up projections
            gate = (hidden @ W_g) + b_g  # [T, I]
            up = (hidden @ W_u) + b_u  # [T, I]

            # Apply GptOss activation with clamping
            gate = gate.clamp(min=torch.finfo(torch.float16).min, max=self.experts.limit)
            up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

            # GLU activation
            glu = gate * torch.sigmoid(gate * self.experts.alpha)
            intermediate = (up + 1) * glu  # [T, I]

            # Down projection
            down_out = (intermediate @ W_d) + b_d  # [T, H]

            # Apply routing weights and accumulate
            expert_out += down_out * routing_weight

        # original shape [B, S, H]
        return expert_out.view(B, S, H), router_logits


class QEffPrefillOnlyGptOssMLP(GptOssMLP):
    def forward(self, hidden: torch.Tensor):
        if os.environ.get("NUM_FFN_BLOCKS", None) is not None:
            return self.blocked_ffn_forward(hidden)
        B, S, H = hidden.shape
        T = B * S
        hidden = hidden.view(T, H)

        # Router computation
        router_logits = F.linear(hidden, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, top_i = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        masked_logits = torch.zeros_like(router_logits)
        masked_logits.scatter_(1, top_i, top_w)

        # Routing weights for each expert [T, E]
        routing_weights = masked_logits

        # ────────────────── allocate the output tensor ─────
        expert_out = hidden.new_zeros((T, H))  # accumulation buffer

        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)  # [T, 1]

            W_g, W_u = self.experts.gate_proj[e], self.experts.up_proj[e]  # [H, I], [H, I]
            b_g, b_u = self.experts.gate_proj_bias[e], self.experts.up_proj_bias[e]  # [I], [I]
            W_d = self.experts.down_proj[e]  # [I, H]
            b_d = self.experts.down_proj_bias[e]  # [H]

            # Gate and Up projections
            gate = (hidden @ W_g) + b_g  # [T, I]
            up = (hidden @ W_u) + b_u  # [T, I]

            # Apply GptOss activation with clamping
            gate = gate.clamp(min=torch.finfo(torch.float16).min, max=self.experts.limit)
            up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

            # GLU activation
            glu = gate * torch.sigmoid(gate * self.experts.alpha)
            intermediate = (up + 1) * glu  # [T, I]

            # Down projection
            down_out = (intermediate @ W_d) + b_d  # [T, H]

            # Apply routing weights and accumulate
            expert_out += down_out * routing_weight

        # original shape [B, S, H]
        return expert_out.view(B, S, H), router_logits

    def blocked_ffn_forward(self, hidden: torch.Tensor):
        B, S, H = hidden.shape
        T = B * S
        hidden = hidden.view(T, H)

        # Router computation
        router_logits = F.linear(hidden, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, top_i = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        masked_logits = torch.zeros_like(router_logits)
        masked_logits.scatter_(1, top_i, top_w)

        # Routing weights for each expert [T, E]
        routing_weights = masked_logits

        # ────────────────── allocate the output tensor ─────
        expert_out = hidden.new_zeros((T, H))  # accumulation buffer
        target_blocks = int(os.environ.get("NUM_FFN_BLOCKS", 1))
        block_positions = []
        for j in range(target_blocks):
            block_positions.append(j * (T // target_blocks))
        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)  # [T, 1]

            W_g, W_u = self.experts.gate_proj[e], self.experts.up_proj[e]  # [H, I], [H, I]
            b_g, b_u = self.experts.gate_proj_bias[e], self.experts.up_proj_bias[e]  # [I], [I]
            W_d = self.experts.down_proj[e]  # [I, H]
            b_d = self.experts.down_proj_bias[e]  # [H]

            block_count = 0
            outs = []
            for block_idx in range(target_blocks):
                block_count += 1
                qi = block_positions[block_idx]

                # Calculate block size (last block should be handled with remainder)
                if block_idx == target_blocks - 1:
                    real_q_len = T - qi
                else:
                    real_q_len = block_positions[block_idx + 1] - qi

                tgb = hidden[qi : qi + real_q_len, :]
                # Gate and Up projections
                # Gate and Up projections
                gate = (tgb @ W_g) + b_g  # [T, I]
                up = (tgb @ W_u) + b_u  # [T, I]

                # Apply GptOss activation with clamping
                gate = gate.clamp(min=torch.finfo(torch.float16).min, max=self.experts.limit)
                up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

                # GLU activation
                glu = gate * torch.sigmoid(gate * self.experts.alpha)
                intermediate = (up + 1) * glu  # [T, I]

                # Down projection
                down_out_block = (intermediate @ W_d) + b_d  # [T, H]

                outs.append(down_out_block)

            down_out = torch.cat(outs, dim=0)

            # Apply routing weights and accumulate
            expert_out += down_out * routing_weight

        # original shape [B, S, H]
        return expert_out.view(B, S, H), router_logits

    def blocked_ffn_forward_block_weights(self, hidden: torch.Tensor):
        B, S, H = hidden.shape
        T = B * S
        hidden = hidden.view(T, H)

        # Router computation
        router_logits = F.linear(hidden, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, top_i = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        masked_logits = torch.zeros_like(router_logits)
        masked_logits.scatter_(1, top_i, top_w)

        # Routing weights for each expert [T, E]
        routing_weights = masked_logits

        # ────────────────── allocate the output tensor ─────
        expert_out = hidden.new_zeros((T, H))  # accumulation buffer
        target_blocks = int(os.environ.get("NUM_BLOCKS", 1))
        block_positions = []
        for j in range(target_blocks):
            block_positions.append(j * (T // target_blocks))
        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            routing_weight = routing_weights[:, e].unsqueeze(-1)  # [T, 1]

            W_g, W_u = self.experts.gate_proj[e], self.experts.up_proj[e]  # [H, I], [H, I]
            b_g, b_u = self.experts.gate_proj_bias[e], self.experts.up_proj_bias[e]  # [I], [I]
            W_d = self.experts.down_proj[e]  # [I, H]
            b_d = self.experts.down_proj_bias[e]  # [H]

            block_count = 0
            outs = []
            for block_idx in range(target_blocks):
                block_count += 1
                qi = block_positions[block_idx]

                # Calculate block size (last block should be handled with remainder)
                if block_idx == target_blocks - 1:
                    real_q_len = T - qi
                else:
                    real_q_len = block_positions[block_idx + 1] - qi

                tgb = hidden[qi : qi + real_q_len, :]
                # Gate and Up projections

                wg_col_shape = W_g.shape[1]
                wg_num_blocks = math.ceil(wg_col_shape / 128)
                last_block_size = wg_col_shape % 128 if wg_col_shape % 128 != 0 else 128

                intermediates = []
                for i in range(wg_num_blocks):
                    if i == wg_num_blocks - 1:
                        cur_gate = (tgb @ W_g[:, -last_block_size:]) + b_g[-last_block_size:]
                        cur_up = (tgb @ W_u[:, -last_block_size:]) + b_u[-last_block_size:]
                    else:
                        cur_gate = (tgb @ W_g[:, i * 128 : (i + 1) * 128]) + b_g[i * 128 : (i + 1) * 128]
                        cur_up = (tgb @ W_u[:, i * 128 : (i + 1) * 128]) + b_u[i * 128 : (i + 1) * 128]

                    cur_gate = cur_gate.clamp(min=torch.finfo(torch.float16).min, max=self.experts.limit)
                    cur_up = cur_up.clamp(min=-self.experts.limit, max=self.experts.limit)
                    cur_glu = cur_gate * torch.sigmoid(cur_gate * self.experts.alpha)
                    cur_intermediate = (cur_up + 1) * cur_glu
                    intermediates.append(cur_intermediate)

                intermediate = torch.cat(intermediates, dim=-1)

                downs = []
                for i in range(wg_num_blocks):
                    if i == wg_num_blocks - 1:
                        downs.append((intermediate @ W_d[:, -last_block_size:]) + b_d[-last_block_size:])
                    else:
                        downs.append((intermediate @ W_d[:, i * 128 : (i + 1) * 128]) + b_d[i * 128 : (i + 1) * 128])

                down_out_block = torch.cat(downs, dim=1)
                outs.append(down_out_block)

            down_out = torch.cat(outs, dim=0)

            # Apply routing weights and accumulate
            masked_down = torch.where(routing_weight > 0, down_out * routing_weight, torch.zeros_like(expert_out))
            expert_out += masked_down

        # original shape [B, S, H]
        return expert_out.view(B, S, H), router_logits


class QEffGptOssMLP(GptOssMLP):
    # ------------------- Gather based, weights as activation approach ---------------
    def forward_weights_as_activation(self, hidden_states):
        bs, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(bs * seq_len, self.experts.hidden_size)

        # Router computation
        router_logits = F.linear(hidden_states, self.router.weight, self.router.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.router.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)

        # GATHER - collect weights for selected experts
        gate_up_proj = self.experts.gate_up_proj[router_indices.flatten()]
        gate_up_proj_bias = self.experts.gate_up_proj_bias[router_indices.flatten()]
        down_proj = self.experts.down_proj[router_indices.flatten()]
        down_proj_bias = self.experts.down_proj_bias[router_indices.flatten()]

        # Apply Chosen Experts (without routing weights first)
        # expert_in = hidden_states.repeat_interleave(self.router.top_k, dim=0)
        # expert_in = expert_in.view(-1, 1, self.experts.hidden_size)
        # Reshape for bmm: (bs*seq_len*top_k, 1, hidden_size)
        expert_in = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.router.top_k, -1)
            .contiguous()
            .view(-1, 1, self.experts.hidden_size)
        )

        gate_up = torch.bmm(expert_in, gate_up_proj) + gate_up_proj_bias.unsqueeze(1)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]

        # Apply activation with clamping
        gate = gate.clamp(min=None, max=self.experts.limit)
        up = up.clamp(min=-self.experts.limit, max=self.experts.limit)
        glu = gate * torch.sigmoid(gate * self.experts.alpha)
        gated_output = (up + 1) * glu

        experts_out = torch.bmm(gated_output, down_proj) + down_proj_bias.unsqueeze(1)
        experts_out = experts_out.view(bs * seq_len, self.router.top_k, self.experts.hidden_size)

        # Apply routing weights AFTER expert computation (This is before on Llama4)
        experts_out = experts_out * router_top_value.unsqueeze(-1)
        experts_out = experts_out.sum(dim=1)

        return experts_out, router_logits

    # ------------------- Gather based, weights as activation approach, With Seperate Gate, up Projections ---------------
    def forward(self, hidden_states):
        bs, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(bs * seq_len, self.experts.hidden_size)

        # Router computation
        router_logits = F.linear(hidden_states, self.router.weight, self.router.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.router.top_k, dim=-1)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)

        # GATHER - collect weights for selected experts (separate gate and up projections)
        gate_proj = self.experts.gate_proj[router_indices.flatten()]
        gate_proj_bias = self.experts.gate_proj_bias[router_indices.flatten()]
        up_proj = self.experts.up_proj[router_indices.flatten()]
        up_proj_bias = self.experts.up_proj_bias[router_indices.flatten()]
        down_proj = self.experts.down_proj[router_indices.flatten()]
        down_proj_bias = self.experts.down_proj_bias[router_indices.flatten()]

        # Reshape for bmm: (bs*seq_len*top_k, 1, hidden_size)
        expert_in = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.router.top_k, -1)
            .contiguous()
            .view(-1, 1, self.experts.hidden_size)
        )

        # Apply gate and up projections separately using bmm
        gate = torch.bmm(expert_in, gate_proj) + gate_proj_bias.unsqueeze(1)
        up = torch.bmm(expert_in, up_proj) + up_proj_bias.unsqueeze(1)

        # Apply activation with clamping
        gate = gate.clamp(min=torch.finfo(torch.float16).min, max=self.experts.limit)
        up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

        # GLU activation
        glu = gate * torch.sigmoid(gate * self.experts.alpha)
        gated_output = (up + 1) * glu

        # Down projection
        experts_out = torch.bmm(gated_output, down_proj) + down_proj_bias.unsqueeze(1)
        experts_out = experts_out.view(bs * seq_len, self.router.top_k, self.experts.hidden_size)

        # Apply routing weights AFTER expert computation
        experts_out = experts_out * router_top_value.unsqueeze(-1)
        experts_out = experts_out.sum(dim=1)

        return experts_out, router_logits

    def optimized_moe_forward(self, hidden_states: torch.Tensor):
        B, S, H = hidden_states.shape
        T = B * S
        hidden_states = hidden_states.view(T, H)

        # Router computation
        router_logits = F.linear(hidden_states, self.router.weight, self.router.bias)

        # Top-k selection
        top_w, selected_experts = torch.topk(router_logits, self.router.top_k, dim=-1)  # both [T, K]
        top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)

        # Creating experts mask and routing weights masked
        awesome_experts_mask_1 = (
            torch.nn.functional.one_hot(selected_experts[:, 0], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_2 = (
            torch.nn.functional.one_hot(selected_experts[:, 1], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_3 = (
            torch.nn.functional.one_hot(selected_experts[:, 2], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )
        awesome_experts_mask_4 = (
            torch.nn.functional.one_hot(selected_experts[:, 3], num_classes=self.experts.num_experts)
            .bool()
            .T.unsqueeze(-1)
        )

        gateupout1 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout2 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout3 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs
        gateupout4 = torch.zeros(hidden_states.shape[0], self.experts.intermediate_size)  # T, hs

        # ───────────────────────── Expert computation loop ─────────────────────────────
        for e in range(self.experts.num_experts):
            W_g, W_u = self.experts.gate_proj[e], self.experts.up_proj[e]  # [H, I], [H, I]
            b_g, b_u = self.experts.gate_proj_bias[e], self.experts.up_proj_bias[e]  # [I], [I]

            # Gate and Up projections
            gate = (hidden_states @ W_g) + b_g  # [T, I]
            up = (hidden_states @ W_u) + b_u  # [T, I]

            # Apply GptOss activation with clamping
            gate = gate.clamp(min=None, max=self.experts.limit)
            up = up.clamp(min=-self.experts.limit, max=self.experts.limit)

            # GLU activation
            glu = gate * torch.sigmoid(gate * self.experts.alpha)
            intermediate = (up + 1) * glu  # [T, I]

            gateupout1 += torch.where(awesome_experts_mask_1[e], intermediate, torch.zeros_like(gateupout1))
            gateupout2 += torch.where(awesome_experts_mask_2[e], intermediate, torch.zeros_like(gateupout2))
            gateupout3 += torch.where(awesome_experts_mask_3[e], intermediate, torch.zeros_like(gateupout3))
            gateupout4 += torch.where(awesome_experts_mask_4[e], intermediate, torch.zeros_like(gateupout4))

        concat_down = torch.zeros((self.router.top_k, T, H))
        concat_mask = torch.cat(
            (
                awesome_experts_mask_1.unsqueeze(0),
                awesome_experts_mask_2.unsqueeze(0),
                awesome_experts_mask_3.unsqueeze(0),
                awesome_experts_mask_4.unsqueeze(0),
            ),
            dim=0,
        )

        concat_gateout = torch.cat(
            (gateupout1.unsqueeze(0), gateupout2.unsqueeze(0), gateupout3.unsqueeze(0), gateupout4.unsqueeze(0)), dim=0
        )

        for e in range(self.experts.num_experts):
            W_d = self.experts.down_proj[e]  # [I, H]
            b_d = self.experts.down_proj_bias[e]  # [H]

            # Down projection
            down_out = (concat_gateout @ W_d) + b_d  # [T, H]

            concat_down += torch.where(concat_mask[:, e, :], down_out, torch.zeros_like(concat_down))

        downout1, downout2, downout3, downout4 = concat_down[0], concat_down[1], concat_down[2], concat_down[3]
        hidden_states = (
            downout1 * top_w[:, 0].unsqueeze(-1)
            + downout2 * top_w[:, 1].unsqueeze(-1)
            + downout3 * top_w[:, 2].unsqueeze(-1)
            + downout4 * top_w[:, 3].unsqueeze(-1)
        ).reshape(B, S, H)

        # original shape [B, S, H]
        return hidden_states, router_logits


#  Can be replaced with llama/modeling_llama.py::QEffLlamaRotaryEmbedding but keeping it following transformers ideology
class QEffGptOssRotaryEmbedding(GptOssRotaryEmbedding):
    """
    Copied from LlamaForCausalLM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config: GptOssConfig, device=None):
        super().__init__(config=config)
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
            self.sin_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def eager_attention_forward_blocked(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    BS, NH, CL, DH = query.shape
    target_blocks = int(os.environ.get("NUM_Q_BLOCKS", 1))
    block_positions = []
    for j in range(target_blocks):
        block_positions.append(j * (CL // target_blocks))
    block_count = 0

    outs = []
    for block_idx in range(target_blocks):
        block_count += 1
        qi = block_positions[block_idx]

        # Calculate block size (last block should be handled with remainder)
        if block_idx == target_blocks - 1:
            real_q_len = CL - qi
        else:
            real_q_len = block_positions[block_idx + 1] - qi

        q_block = query[:, :, qi : qi + real_q_len, :]
        scores = torch.matmul(q_block, key_states.transpose(2, 3)) * scaling
        attn_mask_block = attention_mask[:, :, qi : qi + real_q_len, :]
        curr_attn_weights = torch.where(
            attn_mask_block, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), scores
        )
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(
            curr_attn_weights.shape[0], -1, curr_attn_weights.shape[-2], -1
        )
        combined_logits = torch.cat([curr_attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        curr_attn_weights = nn.functional.softmax(combined_logits, dim=-1, dtype=torch.float32)
        curr_attn_weights = curr_attn_weights[..., :-1]
        out_block = torch.matmul(curr_attn_weights, value_states)
        outs.append(out_block)
    output = torch.cat(outs, dim=2)

    output = output.view(BS, NH, CL, DH).transpose(1, 2).contiguous()
    return output, output

def eager_attention_forward_blockedKV(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    num_kv_blocks: Optional[int] = None,
    skip_threshold: Optional[bool] = None,
    cache_kwargs: Optional[Dict[str, Any]] = None,
    layer_idx: int = None,
    past_key_value: Optional[Cache] = None,
    dropout: float = 0.0,
    **kwargs,
):
    # Initialize result tensor
    output = torch.zeros_like(query)
    
    # Initialize Running Maximum
    batch_size, num_heads, seq_len, head_dim = query.shape
    current_max = torch.full((batch_size, num_heads, seq_len), float(MIN_MASKED_ATTENTION_VALUE), dtype=query.dtype)
    
    current_denominator = torch.zeros(batch_size, num_heads, seq_len, dtype=query.dtype)
    
    past_seen_tokens = cache_kwargs.get("past_seen_tokens")
    position_ids = cache_kwargs.get("position_ids")
    block_size = -(-past_seen_tokens // num_kv_blocks)
    masked_tensor = torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32)
    
    # sinks = module.sinks.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, seq_len, -1)
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(
            batch_size, -1, seq_len, -1
        )

    if skip_threshold:
        threshold = math.log(skip_threshold)

    for j in range(num_kv_blocks):
        start_index = j * block_size
        end_index = (j + 1) * block_size
        
        K_block, V_block = past_key_value.read_only_blockedKV(start_index, end_index, layer_idx, cache_kwargs)
        K_block_states = repeat_kv(K_block, module.num_key_value_groups)
        V_block_states = repeat_kv(V_block, module.num_key_value_groups)
        
        past_seen_tokens_start = start_index
        past_seen_tokens_end = torch.where(
            torch.tensor(past_seen_tokens, dtype=torch.int) < torch.tensor(end_index, dtype=torch.int),
            past_seen_tokens,
            end_index,
        )
        
        causal_mask_block = _create_causal_mask(
            position_ids=position_ids, target_length=past_seen_tokens_end, start_index=past_seen_tokens_start
        )
        
        attn_weights_block = torch.matmul(query, K_block_states.transpose(2, 3)) * scaling
        
        if attention_mask is not None:
            attn_weights_block = torch.where(causal_mask_block, masked_tensor, attn_weights_block)
        
        block_max = attn_weights_block.max(dim=-1, keepdim=True).values
        
        prev_max = current_max
        current_max = torch.maximum(prev_max, block_max.squeeze(-1))
        delta_max = prev_max - current_max

        # skip condition
        if skip_threshold:
            local_delta = current_max.unsqueeze(2) - block_max > threshold
        
        attn_weights_block_stable = attn_weights_block - current_max.unsqueeze(-1)
        kv_exp = torch.exp(attn_weights_block_stable)
        
        prev_denominator = current_denominator
        current_denominator = prev_denominator * torch.exp(delta_max) + kv_exp.sum(dim=-1)
        
        block_contribution = torch.matmul(kv_exp, V_block_states)
        prev_output = output
        output_curr_step = prev_output * torch.exp(delta_max).unsqueeze(-1) + block_contribution

        if skip_threshold:
            output = torch.where(local_delta, output, output_curr_step) # if current_max - local_max > threshold, use previous output effectively skipping this block
        else:
            output = output_curr_step

    # Apply attention sinks    
    prev_max = current_max
    sink_max = sinks.max(dim=-1, keepdim=True).values.squeeze(-1)
    current_max = torch.maximum(prev_max, sink_max)
    delta_max = prev_max - current_max
    
    sinks_stable = sinks - current_max.unsqueeze(-1)
    sink_exp = torch.exp(sinks_stable)
    
    prev_denominator = current_denominator
    current_denominator = prev_denominator * torch.exp(delta_max) + sink_exp.sum(dim=-1)
    
    output = output * torch.exp(delta_max).unsqueeze(-1)
    
    output = output / current_denominator.unsqueeze(-1)
    
    attn_output = output.transpose(1, 2).contiguous()
    attn_weights = None

    return attn_output, attn_weights


def opt_eager_attention_forward_blocked(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    BS, NH, CL, DH = query.shape
    target_blocks = int(os.environ.get("NUM_Q_BLOCKS", 1))
    block_positions = []
    for j in range(target_blocks):
        block_positions.append(j * (CL // target_blocks))
    block_count = 0
    outs = []
    for block_idx in range(target_blocks):
        block_count += 1
        qi = block_positions[block_idx]
        # Calculate block size (last block should be handled with remainder)

        if block_idx == target_blocks - 1:
            real_q_len = CL - qi
        else:
            real_q_len = block_positions[block_idx + 1] - qi

        if block_idx == 0:
            kv_start_idx = 0
        else:
            kv_start_idx = qi - 128

        q_block = query[:, :, qi : qi + real_q_len, :]
        if kwargs.get("sliding_window"):
            k_block = key_states[:, :, kv_start_idx : qi + real_q_len, :]
            v_block = value_states[:, :, kv_start_idx : qi + real_q_len, :]
            attn_mask_block = attention_mask[:, :, qi : qi + real_q_len, kv_start_idx : qi + real_q_len]
        else:
            k_block = key_states
            v_block = value_states
            attn_mask_block = attention_mask[:, :, qi : qi + real_q_len, :]

        scores = torch.matmul(q_block, k_block.transpose(2, 3)) * scaling
        curr_attn_weights = torch.where(
            attn_mask_block, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), scores
        )
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(
            curr_attn_weights.shape[0], -1, curr_attn_weights.shape[-2], -1
        )
        combined_logits = torch.cat([curr_attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        curr_attn_weights = nn.functional.softmax(combined_logits, dim=-1, dtype=torch.float32)
        curr_attn_weights = curr_attn_weights[..., :-1]
        out_block = torch.matmul(curr_attn_weights, v_block)
        outs.append(out_block)
    output = torch.cat(outs, dim=2)

    output = output.view(BS, NH, CL, DH).transpose(1, 2).contiguous()
    return output, output


class QEffPrefillOnlyChunkedGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __qeff_init__(self):
        self.rotary_emb = QEffGptOssRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sliding_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        hidden_shape = (*input_shape, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if not (max_seq_len_cached := getattr(self.config, "max_seq_len_cached")):
            max_seq_len_cached = 32 * 1024
        cos, sin = self.rotary_emb(value_states, seq_len=max_seq_len_cached)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "batch_index": batch_index,
                "position_ids": position_ids,
                "config": self.config,
                "is_sliding": self.sliding_window is not None,
                "sliding_window": self.sliding_window,
            }
            if self.sliding_window is not None:
                key_states, value_states = past_key_value.sliding_window_update_chunked(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states, value_states = past_key_value.full_cache_update_chunked(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

        if self.sliding_window is not None:
            attention_mask = sliding_mask
            # positive_pos_ids = torch.where(position_ids<0, 0, position_ids)
            ctx_len = position_ids.shape[1] + self.sliding_window
            ctx_indices = torch.arange(ctx_len)
            first_pos_idx = position_ids[0][0]
            add_idx = torch.where(first_pos_idx >= self.sliding_window, first_pos_idx - self.sliding_window, 0)
            # start_idx = torch.where(first_pos_idx>=self.sliding_window, first_pos_idx-self.sliding_window, 0)
            # end_idx = torch.where(first_pos_idx >= self.sliding_window, first_pos_idx+position_ids.shape[1], position_ids.shape[1]+self.sliding_window)
            ctx_indices += add_idx
            attention_mask = attention_mask[:, :, :, ctx_indices]
        else:
            attention_mask = attention_mask

        attention_interface: Callable = eager_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class QEffPrefillOnlyGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __qeff_init__(self):
        self.rotary_emb = QEffGptOssRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sliding_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        hidden_shape = (*input_shape, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if not (max_seq_len_cached := getattr(self.config, "max_seq_len_cached")):
            max_seq_len_cached = 32 * 1024
        cos, sin = self.rotary_emb(value_states, seq_len=max_seq_len_cached)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "batch_index": batch_index,
                "position_ids": position_ids,
                "config": self.config,
                "is_sliding": self.sliding_window is not None,
                "sliding_window": past_key_value.sliding_window_len,
            }
            if self.sliding_window is not None:
                sliding_window_len = past_key_value.sliding_window_len
                short_read_idx = torch.arange(past_key_value.key_cache[self.layer_idx].shape[2])
                read_idx = short_read_idx + torch.where(
                    position_ids.max() > sliding_window_len - 1, position_ids.max() - sliding_window_len + 1, 0
                )
                # This is a trick to export with seq_len<sliding_window_len
                read_idx = torch.where(read_idx > position_ids.max(), 0, read_idx)
                k_cache = key_states[:, :, read_idx, :]
                v_cache = value_states[:, :, read_idx, :]
            else:
                k_cache, v_cache = key_states, value_states
            _, _ = past_key_value.write_only(k_cache, v_cache, self.layer_idx, cache_kwargs)

        if self.sliding_window is not None:
            attention_mask = sliding_mask
        else:
            attention_mask = attention_mask

        if os.environ.get("ENABLE_OPT_SWA", "0") == "1":
            attention_interface: Callable = opt_eager_attention_forward_blocked
        else:
            attention_interface: Callable = eager_attention_forward_blocked
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class QEffGptOssAttention(GptOssAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __qeff_init__(self):
        self.rotary_emb = QEffGptOssRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sliding_mask=None,
        num_kv_blocks: Optional[torch.Tensor] = None,
        skip_threshold: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        if not (max_seq_len_cached := getattr(self.config, "max_seq_len_cached")):
            max_seq_len_cached = 32 * 1024
        cos, sin = self.rotary_emb(value_states, seq_len=max_seq_len_cached)
        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if num_kv_blocks is not None:
                past_seen_tokens = past_key_value.get_seq_length() if past_key_value is not None else 0
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "batch_index": batch_index,
                    "position_ids": position_ids,
                    "past_seen_tokens": past_seen_tokens,
                    "config": self.config,
                    "is_sliding": self.sliding_window is not None,
                    "sliding_window": past_key_value.sliding_window_len,
                }
                past_key_value.write_only(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "batch_index": batch_index,
                    "position_ids": position_ids,
                    "config": self.config,
                    "is_sliding": self.sliding_window is not None,
                    "sliding_window": past_key_value.sliding_window_len,
                }
                if comp_ctx_lengths is not None:
                    attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                    cache_kwargs["CCL"] = attention_mask.shape[-1]
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                

        if self.sliding_window is not None:
            attention_mask = sliding_mask
        else:
            attention_mask = attention_mask

        if num_kv_blocks is not None: 
            attention_interface: Callable = eager_attention_forward_blockedKV
            if query_states.shape[2] != 1: # don't skip for prefill
                skip_threshold = None
        else:
            attention_interface: Callable = eager_attention_forward

        kwargs["cache_kwargs"] = cache_kwargs

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            num_kv_blocks=num_kv_blocks,
            skip_threshold=skip_threshold,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,  # diff with Llama
            past_key_value=past_key_value,
            layer_idx=self.layer_idx,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class QEffGptOssDecoderLayer(GptOssDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        sliding_mask=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            sliding_mask=sliding_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)  # diff with llama: router scores
        hidden_states = hidden_states.reshape(residual.shape)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QEffPrefillOnlyGptOssModel(GptOssModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffHybridCacheForGPTOSS.from_legacy_cache(self.config, past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values.max_cache_len)
        sliding_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=past_key_values.max_cache_len,
            sliding_window=self.config.sliding_window,
        )
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                sliding_mask=sliding_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffGptOssModel(GptOssModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffHybridCacheForGPTOSS.from_legacy_cache(self.config, past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_key_values.max_cache_len)
        sliding_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=past_key_values.sliding_window_len,
            sliding_window=past_key_values.sliding_window_len,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position,
                sliding_mask=sliding_mask,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class QEffGptOssForCausalLM(GptOssForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GptOssForCausalLM

        >>> model = GptOssForCausalLM.from_pretrained("mistralai/GptOss-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/GptOss-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return MoeCausalLMOutputWithPast(
            loss=None,
            aux_loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def get_pkv_dynamic_axes(self, retain_full_kv: Optional[bool] = False, continuous_batching: Optional[bool] = False):
        pkv_dynamic_axes = []
        for layer_type in self.config.layer_types:
            if layer_type == "sliding_attention" and not retain_full_kv:
                pkv_dynamic_axes.append(
                    {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
                )
            else:
                pkv_dynamic_axes.append({0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"})
        return pkv_dynamic_axes

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        **kwargs,
    ):
        batch_size = batch_size if batch_size else 1
        if kwargs.get("prefill_only") and not kwargs.get("enable_chunking") and ctx_len != prefill_seq_len:
            ctx_len = prefill_seq_len
            logger.warning(
                f"overriding ctx_len={prefill_seq_len}, currently we don't support ctx_len different than prefill_seq_len for prefill_only model"
            )

        specializations = [
            {
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": 128,
            },
            {
                "batch_size": batch_size,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "sliding_window": 128,
            },
        ]
        return specializations
