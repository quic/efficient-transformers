import math
import os
from typing import Dict, List, Optional, Tuple, Type, Union

import os
import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffDynamicCompressedKVRopeCache
from QEfficient.customop.matmulnbits import QMOE, QuantLinearTorchFunction
from QEfficient.customop.quantization_ops import DequantizeLinearFunc, UnpackUInt8ToUInt4
from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffDynamicCompressedKVRopeCache

from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

    # def forward(self, x, position_ids):
    #     seq_len = torch.max(position_ids) + 1
    #     if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
    #         self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

    #     # Use position_ids to slice the precomputed caches
    #     cos = self.cos_cached[position_ids]
    #     sin = self.sin_cached[position_ids]
    #     return cos.to(x.dtype), sin.to(x.dtype)


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def orig_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QEffDeepseekV3Attention(nn.Module):
    """Adapted DeepseekV3Attention with QEff logic, adding batch_index and proper position_ids handling."""

    def __qeff_init__(
        self,
    ):
        q_up, q_rope = self.q_b_proj.weight.T.view(
            -1, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim
        ).split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_up = q_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0)

        self.q_up = torch.nn.Parameter(q_up.detach().clone())
        q_rope = q_rope.reshape(-1, self.num_heads * self.qk_rope_head_dim).unsqueeze(0)

        self.q_rope = torch.nn.Parameter(q_rope.detach().clone())
        k_up, v_up = self.kv_b_proj.weight.T.view(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_up = k_up.reshape(-1, self.num_heads * self.qk_nope_head_dim).unsqueeze(0)
        v_up = v_up.reshape(-1, self.num_heads * self.v_head_dim).unsqueeze(0)

        self.k_up = torch.nn.Parameter(k_up.detach().clone())
        self.v_up = torch.nn.Parameter(v_up.detach().clone())
        per_head_q_up = self.q_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1)
        per_head_k_up = (
            self.k_up.squeeze(0).view(-1, self.num_heads, self.qk_nope_head_dim).transpose(0, 1).transpose(1, 2)
        )
        self.per_head_q_up = torch.nn.Parameter(per_head_q_up.detach().clone())
        self.per_head_k_up = torch.nn.Parameter(per_head_k_up.detach().clone())

        fusedqk_list = []
        for i in range(self.num_heads):
            fusedqk_list.append(torch.matmul(per_head_q_up[i,:,:], per_head_k_up[i,:,:]))
        fusedqk = torch.cat(fusedqk_list, dim=0)
        fusedqk = fusedqk.reshape(self.num_heads, -1, self.kv_lora_rank)

        self.fusedqk = torch.nn.Parameter(fusedqk.detach().clone())

    def fused_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        mla_absorption: Optional[Dict[str, bool]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv.view(bsz, q_len, -1, self.kv_lora_rank+self.qk_rope_head_dim).transpose(1, 2)
        compressed_kv, k_pe = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        q_a_proj_out = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pe = torch.bmm(q_a_proj_out, self.q_rope)
        q_pe = q_pe.view(bsz, q_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)
        q_nope = torch.bmm(q_a_proj_out, self.q_up)
        q_nope = q_nope.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim).transpose(1, 2)

        compressed_kv = self.kv_a_layernorm(compressed_kv)
        cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
        if compressed_kvs is not None:
            compressed_kv = compressed_kvs.update_ckv(compressed_kv, self.layer_idx, cache_kwargs)

        kva = compressed_kv

        if mla_absorption is not None:
            enable_absorption = mla_absorption.get("enable", False)
            absorb_online = mla_absorption.get("online", False)
        else:
            enable_absorption = False

        n_head_ckv = compressed_kv.shape[1]
        p = self.num_heads//n_head_ckv

        blocking_config = getattr(self, "attn_blocking_config", None)
        num_kv_blocks = 1
        if blocking_config is not None:
            num_kv_blocks = blocking_config.num_kv_blocks
        print("num_kv_blocks : ", num_kv_blocks)
        ctx_len = compressed_kv.shape[-2]
        block_size = -(-ctx_len // num_kv_blocks)

        value_out = []
        for i in range(n_head_ckv):
          value_states_ph = torch.matmul(kva[:,i,:,:], self.v_up[:, :, i*p*self.v_head_dim: (i+1)*p*self.v_head_dim])
          value_states_ph = value_states_ph.view(bsz, -1, p, self.qk_nope_head_dim).transpose(1, 2)
          value_out.append(value_states_ph)
        value_states = torch.cat(value_out, dim=1)

        cos, sin = self.rotary_emb(value_states_ph, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        if compressed_kvs is not None:
            k_pe = compressed_kvs.update_k_pe(k_pe, self.layer_idx, cache_kwargs)

        attn_output_list = []
        for k in range(n_head_ckv):
            attn_weights_list = []
            for j in range(num_kv_blocks):
                kv_start_index = j*block_size
                kv_end_index = min(ctx_len, (j+1)*block_size)
                if enable_absorption:
                    if absorb_online:
                        if j==0:
                            print("online absorption")
                        out = torch.matmul(self.per_head_q_up[k*p:(k+1)*p,:,:], self.per_head_k_up[k*p:(k+1)*p,:,:])
                        out2 = torch.matmul(q_a_proj_out.unsqueeze(1), out)
                    else:
                        if j==0:
                            print("using fused qk")
                        out2 = torch.matmul(q_a_proj_out.unsqueeze(1), self.fusedqk[k*p:(k+1)*p,:,:])

                    out3 = torch.cat((out2, q_pe[:,k*p:(k+1)*p,:,:]), -1)
                    kva_kpe = torch.cat((kva[:,k,kv_start_index:kv_end_index,:],k_pe[:,k,kv_start_index:kv_end_index,:]), -1).unsqueeze(1)
                    attn_weights = torch.matmul(out3, kva_kpe.transpose(2,3)) * self.softmax_scale
                else:
                    if j==0:
                        print("no absorption")
                    k_nope = torch.matmul(kva[:,k,:,:], self.k_up[:, :, k*p*self.qk_nope_head_dim: (k+1)*p*self.qk_nope_head_dim])
                    k_nope = k_nope.view(bsz, -1, p, self.qk_nope_head_dim).transpose(1, 2)
                    key_states = torch.cat((k_nope[:,:,kv_start_index:kv_end_index,:], k_pe[:,k,kv_start_index:kv_end_index,:].unsqueeze(1).repeat(1,p,1,1)), -1)
                    query_states = torch.cat((q_nope[:,k*p:(k+1)*p,:,:], q_pe[:,k*p:(k+1)*p,:,:]), -1)
                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

                attn_weights_list.append(attn_weights)

            attn_weights = torch.cat(attn_weights_list, dim=-1)

            if attention_mask is not None:  # no matter the length, we just slice it
                attn_weights = torch.where(attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights)

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_pe.dtype)
            attn_output = torch.matmul(attn_weights, value_states[:,k*p:(k+1)*p,:,:])
            attn_output_list.append(attn_output)

        attn_output = torch.cat(attn_output_list, dim=1)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, compressed_kvs, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = self.rotary_emb(value_states, seq_len=32 * 1024)
        q_pe, k_pe = orig_apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = torch.cat((q_nope, q_pe), -1)
        k_pe_new = k_pe.expand(-1, self.num_heads, -1, -1)
        key_states = torch.cat((k_nope, k_pe_new), -1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = torch.where(attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, value_states


class QEffDeepseekMoEGate(nn.Module):

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        ### select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(f"insupportable TopK function for MoE gating: {self.topk_method}")

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor  # must multiply the scaling factor

        router_probs = tmp_scores
        router_weights = scores
        return topk_idx, topk_weight, router_probs, router_weights



class QEffDeepseekV3MoE(nn.Module):
    def __qeff_init__(
        self,
    ):

        # Get common parameters from first expert
        first_expert = self.experts[0]
        self.bits = first_expert.gate_proj.bits
        self.group_size = first_expert.gate_proj.group_size
        self.act_fn = first_expert.act_fn
        assert first_expert.gate_proj.act_order == first_expert.up_proj.act_order == first_expert.down_proj.act_order, "act_order mismatch"
        self.act_order = first_expert.gate_proj.act_order
        
        # Store dimensions for dequantization
        self.in_features_gate, self.out_features_gate = first_expert.gate_proj.in_features, first_expert.gate_proj.out_features
        self.in_features_up, self.out_features_up = first_expert.up_proj.in_features, first_expert.up_proj.out_features
        self.in_features_down, self.out_features_down = first_expert.down_proj.in_features, first_expert.down_proj.out_features

        # Stack all parameters along a new dimension (expert dimension)
        self.all_gate_qweight = torch.nn.Parameter(torch.stack([exp.gate_proj.qweight for exp in self.experts], dim=0).reshape(-1, self.out_features_gate, self.in_features_gate//2), requires_grad=False)
        self.all_gate_scales = torch.nn.Parameter(torch.stack([exp.gate_proj.scales for exp in self.experts], dim=0).reshape(-1, self.out_features_gate, self.in_features_gate//self.group_size), requires_grad=False)
        # TODO: Since we know qzeros is always 8 -> Just embed this once into the operator as parameter -> explore this later
        self.all_gate_qzeros = torch.nn.Parameter(torch.stack([exp.gate_proj.qzeros for exp in self.experts], dim=0).reshape(-1, self.out_features_gate, self.in_features_gate//(self.group_size*2)), requires_grad=False)
        self.all_gate_gidx = torch.nn.Parameter(torch.stack([exp.gate_proj.g_idx for exp in self.experts], dim=0), requires_grad=False)
        
        self.all_up_qweight = torch.nn.Parameter(torch.stack([exp.up_proj.qweight for exp in self.experts], dim=0).reshape(-1, self.out_features_up, self.in_features_up//2), requires_grad=False)
        self.all_up_scales = torch.nn.Parameter(torch.stack([exp.up_proj.scales for exp in self.experts], dim=0).reshape(-1, self.out_features_up, self.in_features_up//self.group_size), requires_grad=False)
        self.all_up_qzeros = torch.nn.Parameter(torch.stack([exp.up_proj.qzeros for exp in self.experts], dim=0).reshape(-1, self.out_features_up, self.in_features_up//(self.group_size*2)), requires_grad=False)
        self.all_up_gidx = torch.nn.Parameter(torch.stack([exp.up_proj.g_idx for exp in self.experts], dim=0), requires_grad=False)
        
        self.all_down_qweight = torch.nn.Parameter(torch.stack([exp.down_proj.qweight for exp in self.experts], dim=0).reshape(-1, self.out_features_down, self.in_features_down//2), requires_grad=False)
        self.all_down_scales = torch.nn.Parameter(torch.stack([exp.down_proj.scales for exp in self.experts], dim=0).reshape(-1, self.out_features_down, self.in_features_down//self.group_size), requires_grad=False)
        self.all_down_qzeros = torch.nn.Parameter(torch.stack([exp.down_proj.qzeros for exp in self.experts], dim=0).reshape(-1, self.out_features_down, self.in_features_down//(self.group_size*2)), requires_grad=False)
        self.all_down_gidx = torch.nn.Parameter(torch.stack([exp.down_proj.g_idx for exp in self.experts], dim=0), requires_grad=False)

        # self.fc1_experts_weights = torch.nn.Parameter(all_gate_qweight.view(self.config.n_routed_experts, self.out_features_gate, -1), requires_grad=False)
        # self.fc1_scales = torch.nn.Parameter(all_gate_scales.view(self.config.n_routed_experts, self.out_features_gate, -1), requires_grad=False)
        
        # self.fc2_experts_weights = torch.nn.Parameter(all_up_qweight.view(self.config.n_routed_experts, self.out_features_up, -1), requires_grad=False)
        # self.fc2_scales = torch.nn.Parameter(all_up_scales.view(self.config.n_routed_experts, self.out_features_up, -1), requires_grad=False)
        
        # self.fc3_experts_weights = torch.nn.Parameter(all_down_qweight.view(self.config.n_routed_experts, self.out_features_down, -1), requires_grad=False)
        # self.fc3_scales = torch.nn.Parameter(all_down_scales.view(self.config.n_routed_experts, self.out_features_down, -1), requires_grad=False)
        
        
        
    def moe_old(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)

        for i in range(self.gate.top_k):
            expert_idx = topk_indices[:, i]
            curr_weight = topk_weights[:, i]
            gate_qweight = self.all_gate_qweight[expert_idx].reshape(seq_len*self.out_features_gate, self.in_features_gate//self.group_size, (self.group_size*self.bits)//8)
            gate_scales = self.all_gate_scales[expert_idx].reshape(seq_len*self.out_features_gate * (self.in_features_gate//self.group_size))
            gate_qzeros = self.all_gate_qzeros[expert_idx].reshape(seq_len*self.out_features_gate, self.in_features_gate//self.group_size)
            gate_gidx = self.all_gate_gidx[expert_idx].reshape(seq_len*self.in_features_gate)

            up_qweight = self.all_up_qweight[expert_idx].reshape(seq_len*self.out_features_up, self.in_features_up//self.group_size, (self.group_size*self.bits)//8)
            up_scales = self.all_up_scales[expert_idx].reshape(seq_len*self.out_features_up * (self.in_features_up//self.group_size))
            up_qzeros = self.all_up_qzeros[expert_idx].reshape(seq_len*self.out_features_up, self.in_features_up//self.group_size)
            up_gidx = self.all_up_gidx[expert_idx].reshape(seq_len*self.in_features_up)

            down_qweight = self.all_down_qweight[expert_idx].reshape(seq_len*self.out_features_down, self.in_features_down//self.group_size, (self.group_size*self.bits)//8)
            down_scales = self.all_down_scales[expert_idx].reshape(seq_len*self.out_features_down * (self.in_features_down//self.group_size))
            down_qzeros = self.all_down_qzeros[expert_idx].reshape(seq_len*self.out_features_down, self.in_features_down//self.group_size)
            down_gidx = self.all_down_gidx[expert_idx].reshape(seq_len*self.in_features_down)

            gate_out = QuantLinearTorchFunction.apply(
                hidden_states,
                gate_qweight,
                gate_scales,
                gate_qzeros,
                gate_gidx if self.act_order else None,
                self.bits,
                self.group_size,
                self.in_features_gate,
                self.out_features_gate * seq_len
            )
            
            up_out = QuantLinearTorchFunction.apply(
                hidden_states,
                up_qweight,
                up_scales,
                up_qzeros,
                up_gidx if self.act_order else None,
                self.bits,
                self.group_size,
                self.in_features_up,
                self.out_features_up * seq_len
            )

            hidden = self.act_fn(gate_out) * up_out
            down_out = QuantLinearTorchFunction.apply(
                hidden,
                down_qweight,
                down_scales,
                down_qzeros,
                down_gidx if self.act_order else None,
                self.bits,
                self.group_size,
                self.in_features_down,
                self.out_features_down
            )
            down_out = down_out.reshape(seq_len, self.out_features_down)
            final_hidden_states += down_out * curr_weight.unsqueeze(1)
        
        return final_hidden_states

    def moe_weights_as_activations(self, hidden_states, router_probs, router_weights):

        return QMOE.apply(hidden_states, router_weights, self.fc1_experts_weights, self.fc1_scales, self.fc2_experts_weights, self.fc2_scales, self.fc3_experts_weights, self.fc3_scales, router_probs,
                          self.config.hidden_act, self.group_size, self.bits, self.num_experts_per_tok)

    @torch.no_grad()
    def original_moe(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out



    def moe_waa_unpack(self, hidden_states, topk_indices, topk_weights):
        # GATHER - collect weights for selected experts
        gate_proj_qweight = self.all_gate_qweight[topk_indices.flatten()]
        gate_proj_scales = self.all_gate_scales[topk_indices.flatten()]
        gate_proj_qzeros = self.all_gate_qzeros[topk_indices.flatten()]
        
        up_proj_qweight = self.all_up_qweight[topk_indices.flatten()]
        up_proj_scales = self.all_up_scales[topk_indices.flatten()]
        up_proj_qzeros = self.all_up_qzeros[topk_indices.flatten()]
        
        down_proj_qweight = self.all_down_qweight[topk_indices.flatten()]
        down_proj_scales = self.all_down_scales[topk_indices.flatten()]
        down_proj_qzeros = self.all_down_qzeros[topk_indices.flatten()]

        gate_proj_unpacked = UnpackUInt8ToUInt4.apply(gate_proj_qweight)
        gate_zeros_unpacked = UnpackUInt8ToUInt4.apply(gate_proj_qzeros)
        gate_proj_dq = DequantizeLinearFunc.apply(gate_proj_unpacked, gate_proj_scales, gate_zeros_unpacked, self.group_size)
        
        up_proj_unpacked = UnpackUInt8ToUInt4.apply(up_proj_qweight)
        up_zeros_unpacked = UnpackUInt8ToUInt4.apply(up_proj_qzeros)
        up_proj_dq = DequantizeLinearFunc.apply(up_proj_unpacked, up_proj_scales, up_zeros_unpacked, self.group_size)
        
        down_proj_unpacked = UnpackUInt8ToUInt4.apply(down_proj_qweight)
        down_zeros_unpacked = UnpackUInt8ToUInt4.apply(down_proj_qzeros)
        down_proj_dq = DequantizeLinearFunc.apply(down_proj_unpacked, down_proj_scales, down_zeros_unpacked, self.group_size)
        
        # Reshape for bmm: (bs*seq_len*top_k, 1, hidden_size)
        expert_in = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.gate.top_k, -1)
            .contiguous()
            .view(-1, 1, self.in_features_gate)
        )


        gate_out = torch.bmm(expert_in, gate_proj_dq.transpose(1, 2))
        up_out = torch.bmm(expert_in, up_proj_dq.transpose(1, 2))
        hidden = self.act_fn(gate_out) * up_out
        down_out = torch.bmm(hidden, down_proj_dq.transpose(1, 2))
        
        down_out = down_out.view(-1, self.gate.top_k, self.out_features_down)
        
        down_out = down_out * topk_weights.unsqueeze(-1)
        
        return torch.einsum("abc-> ac", down_out)

    def forward(self, hidden_states):
        print("Using new MoE forward with weights as activations")
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights, router_probs, router_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        # hidden_states = self.moe_weights_as_activations(hidden_states, router_probs, router_weights).view(*orig_shape)
        hidden_states = self.moe_waa_unpack(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffPrefillOnlyDeepseekV3MoE(nn.Module):
    '''def __qeff_init__(
        self,
    ):
        for exp in self.experts:
          gate_proj = torch.nn.Linear(self.config.hidden_size, self.config.moe_intermediate_size, bias=False)
          up_proj = torch.nn.Linear(self.config.hidden_size, self.config.moe_intermediate_size, bias=False)
          down_proj = torch.nn.Linear(self.config.moe_intermediate_size, self.config.hidden_size, bias=False)

          gate_proj.weight = torch.nn.Parameter(exp.gate_proj.compressor.decompress_module(exp.gate_proj))
          up_proj.weight = torch.nn.Parameter(exp.up_proj.compressor.decompress_module(exp.up_proj))
          down_proj.weight = torch.nn.Parameter(exp.down_proj.compressor.decompress_module(exp.down_proj))

          setattr(exp,"gate_proj", gate_proj)
          setattr(exp,"up_proj", up_proj)
          setattr(exp,"down_proj", down_proj)
    '''
    def moe(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, expert_mask: torch.Tensor, num_experts: int):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        for expert_idx in range(num_experts):
            expert = self.experts[expert_idx]
            gate_out = expert.gate_proj(hidden_states)
            up_out = expert.up_proj(hidden_states)
            hidden = expert.act_fn(gate_out) * up_out
            expert_output = expert.down_proj(hidden)
            current_hidden_states = expert_output * expert_mask[:, expert_idx].unsqueeze(-1)
            final_hidden_states += current_hidden_states

        print("\n\ninside prefill only moe\n")
        return final_hidden_states.type(hidden_states.dtype)

    def orig_moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        r"""
        CALL FOR CONTRIBUTION! I don't have time to optimise this right now, but expert weights need to be fused
        to not have to do a loop here (deepseek has 256 experts soooo yeah).
        """
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)
        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        # in original deepseek, the output of the experts are gathered once we leave this module
        # thus the moe module is itelsf an IsolatedParallel module
        # and all expert are "local" meaning we shard but we don't gather
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        """
        Forward pass of MoE block.
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights, _, _ = self.gate(hidden_states)
        # orig_out = self.orig_moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        mask = torch.zeros(hidden_states.shape[0], self.config.n_routed_experts)
        mask.scatter_(1, topk_indices, topk_weights)
        if os.environ.get("NUM_FFN_BLOCKS", None) is not None and os.environ.get("FFN_W_BLOCK_SIZE", None) is not None:
            hidden_states = self.moe_blocked_weights_forward(
                hidden_states, topk_weights, mask, self.config.n_routed_experts
            ).view(*orig_shape)
        elif os.environ.get("NUM_FFN_BLOCKS", None) is not None:
            hidden_states = self.moe_blocked_forward(
                hidden_states, topk_weights, mask, self.config.n_routed_experts
            ).view(*orig_shape)
        else:
            hidden_states = self.moe(hidden_states, topk_weights, mask, self.config.n_routed_experts).view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffDeepseekV3DecoderLayer(nn.Module):
    """Adapted DeepseekV3DecoderLayer with batch_index and proper position_ids handling."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        compressed_kvs: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        enable_mla: Optional[bool] = False,
        mla_absorption: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if enable_mla:
            hidden_states, self_attn_weights, present_compressed_kvs, vs = self.self_attn.fused_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                compressed_kvs=compressed_kvs,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                mla_absorption=mla_absorption,
                **kwargs,
            )

        else:
            hidden_states, self_attn_weights, present_key_value, vs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            if enable_mla:
                outputs += (present_compressed_kvs,)
            else:
                outputs += (present_key_value,)

        return outputs


class QEffDeepseekV3Model(nn.Module):
    """Adapted DeepseekV3Model with batch_index and QEff rotary embedding."""

    def __qeff_init__(self):
        scaling_factor = self.config.rope_scaling["factor"]
        kwargs = {
            key: self.config.rope_scaling[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in self.config.rope_scaling
        }
        self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
            self.config.qk_rope_head_dim,
            max_position_embeddings=32 * 1024,
            scaling_factor=scaling_factor,
            base=self.config.rope_theta,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_kvs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Cache) and past_key_values is not None:
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
        enable_mla = getattr(self, "enable_mla", False)

        if enable_mla:
            compressed_kvs = QEffDynamicCompressedKVRopeCache.from_legacy_cache(compressed_kvs)
            target_len = compressed_kvs.layers[0].ckv.shape[-2]
        else:
            target_len = past_key_values[0][0].shape[2]

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=target_len)
        hidden_states = inputs_embeds
        position_embeddings = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                compressed_kvs=compressed_kvs,
                past_key_value=past_key_values,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                enable_mla=getattr(self, "enable_mla", False),
                mla_absorption=getattr(self, "mla_absorption_config", None),
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        next_cache = next_cache.to_legacy_cache()
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QEffDeepseekV3ForCausalLM(nn.Module):
    """Adapted DeepseekV3ForCausalLM with batch_index and QEff optimizations."""
    
    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {self.model.layers[0].__class__}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        compressed_kvs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        before_keys = self.state_dict().keys()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compressed_kvs=compressed_kvs,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
