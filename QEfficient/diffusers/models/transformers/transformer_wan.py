# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
import os
from typing import Any, Callable, Dict, List, Tuple, Optional, Union
from venv import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from diffusers.models.modeling_outputs import  Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import WanAttention, _get_qkv_projections, _get_added_kv_projections, dispatch_attention_fn, WanTransformer3DModel, WanAttnProcessor
from diffusers.utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

class QEFFWanAttnProcessor(WanAttnProcessor):
    "To Enable H/Q/KV/HQKV blocking, update rope embedding calculation"
    def forward_head_qkv_blocked(
            self,
            q: torch.FloatTensor,
            k: torch.FloatTensor,
            v: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            *args,
            **kwargs,
        ) -> torch.FloatTensor:
            BS, NH, CL, DH = q.shape  # Input: (BS, NH, CL, DH) = (1, 38, 4429, 64)
            scale_factor = 1.0 / math.sqrt(DH)
            head_block_size = int(os.environ.get('head_block_size', NH))
            num_head_blocks = math.ceil(NH / head_block_size)
            target_blocks_kv = int(os.environ.get('num_kv_blocks', CL))  # KV blocks
            target_blocks_q = int(os.environ.get('num_q_blocks', CL))  # Q blocks
            kv_block_positions = [(i * CL) // target_blocks_kv for i in range(target_blocks_kv)]
            q_block_positions = [(i * CL) // target_blocks_q for i in range(target_blocks_q)]
            # print(f" head_block_size : {head_block_size}, num_head_blocks: {num_head_blocks}, target_blocks_kv: {target_blocks_kv}, target_blocks_q ; {target_blocks_q}")
            # To Handle small CL directly
            BS, NH, K_CL, DH = k.shape
            if K_CL <= 512:
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
                if attention_mask is not None:
                    scores = torch.where(attention_mask, scores,
                                    torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
                probs = torch.softmax(scores, dim=-1)
                out = torch.matmul(probs, v)
                return out

            head_outputs = []

            # Head blocking
            for head_block_idx in range(num_head_blocks):
                h_start = head_block_idx * head_block_size
                h_end = min(h_start + head_block_size, NH)
                num_h = h_end - h_start

                q_g = q[:, h_start:h_end, :, :]
                k_g = k[:, h_start:h_end, :, :]
                v_g = v[:, h_start:h_end, :, :]
                q_output_list = []

                # Q blocking
                for q_block_idx in range(target_blocks_q):
                    qi = q_block_positions[q_block_idx]
                    # Calculate Q block size
                    if q_block_idx == target_blocks_q - 1:
                        real_q_len = CL - qi
                    else:
                        real_q_len = q_block_positions[q_block_idx + 1] - qi

                    q_block = q_g[:, :, qi:qi + real_q_len, :]
                    running_exp_sum = torch.zeros((BS, num_h, real_q_len), device=q.device, dtype=q.dtype)
                    running_max = torch.full((BS, num_h, real_q_len), float('-inf'), device=q.device, dtype=q.dtype)
                    output_blocks = torch.zeros((BS, num_h, real_q_len, DH), device=q.device, dtype=q.dtype)

                    # Process K,V in blocks for this Q block
                    for kv_block_idx in range(target_blocks_kv):
                        ki = kv_block_positions[kv_block_idx]
                        if kv_block_idx == target_blocks_kv - 1:
                            real_kv_len = CL - ki
                        else:
                            real_kv_len = kv_block_positions[kv_block_idx + 1] - ki

                        k_block = k_g[:, :, ki:ki + real_kv_len, :]
                        v_block = v_g[:, :, ki:ki + real_kv_len, :]

                        qkblock = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale_factor

                        prev_max = running_max.clone()
                        if qkblock.shape[-1] == 0:
                            running_max = prev_max
                        else:
                            running_max = torch.maximum(prev_max, torch.max(qkblock, dim=-1)[0])

                        # running_max = torch.maximum(prev_max, torch.max(qkblock, dim=-1)[0])

                        delta_max = prev_max - running_max
                        curr_exp = torch.exp(qkblock - running_max.unsqueeze(-1))

                        prev_exp_sum = running_exp_sum.clone()
                        # running_exp_sum = prev_exp_sum * torch.exp(delta_max) + curr_exp.sum(dim=-1)
                        curr_exp_sum = torch.einsum('bhqk->bhq', curr_exp)
                        running_exp_sum = prev_exp_sum * torch.exp(delta_max) + curr_exp_sum

                        # Compute softmax for this block
                        inv_running_exp_sum = 1.0 / running_exp_sum
                        softmax_qkblock = curr_exp * inv_running_exp_sum.unsqueeze(-1)

                        prev_out = output_blocks.clone()
                        rescale_factor = (prev_exp_sum * inv_running_exp_sum) * torch.exp(delta_max)
                        output_blocks = (rescale_factor.unsqueeze(-1) * prev_out +
                                    torch.matmul(softmax_qkblock, v_block))

                    q_output_list.append(output_blocks)

                head_output = torch.cat(q_output_list, dim=2)
                head_outputs.append(head_output)

            out = torch.cat(head_outputs, dim=1)  # (BS, NH, CL, DH)
            return out

    def forward_head_blocked(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        BS, NH, CL, DH = q.shape  # Input: (BS, NH, CL, DH) = (1, 38, 4429, 64)
        scale_factor = 1.0 / math.sqrt(DH)
        head_block_size = int(os.environ.get('head_block_size', NH))
        num_head_blocks = math.ceil(NH / head_block_size)
        # To Handle small CL directly
        BS, NH, K_CL, DH = k.shape
        if K_CL <= 512:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            if attention_mask is not None:
                scores = torch.where(attention_mask, scores,
                                torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
            return out

        outputs = []

        for head_block_idx in range(num_head_blocks):
            h_start = head_block_idx * head_block_size
            h_end = min(h_start + head_block_size, NH)
            num_h = h_end - h_start

            # Extract head blocks
            q_g = q[:, h_start:h_end, :, :]
            k_g = k[:, h_start:h_end, :, :]
            v_g = v[:, h_start:h_end, :, :]

            qkblock = torch.matmul(q_g, k_g.transpose(-2, -1)) * scale_factor  # (BS, num_h, CL, CL)
            # import pdb; pdb.set_trace()

            # Softmax computation (same as blocked Q version)
            probs = torch.softmax(qkblock, dim=-1)
            # self.softmax_count += 1

            # Compute output
            output_blocks = torch.matmul(probs, v_g)

            outputs.append(output_blocks)

        # Concatenate all head blocks along head dimension
        out = torch.cat(outputs, dim=1)  # (BS, NH, CL, DH)

        return out

    def forward_head_kv_blocked(
            self,
            q: torch.FloatTensor,
            k: torch.FloatTensor,
            v: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            *args,
            **kwargs,
        ) -> torch.FloatTensor:
            BS, NH, CL, DH = q.shape  # Input: (BS, NH, CL, DH) = (1, 38, 4429, 64)
            scale_factor = 1.0 / math.sqrt(DH)
            head_block_size = int(os.environ.get('head_block_size', NH))
            num_head_blocks = math.ceil(NH / head_block_size)
            target_blocks = int(os.environ.get('num_kv_blocks', CL))
            block_positions = [(i * CL) // target_blocks for i in range(target_blocks)]

            # Handle small CL directly
            BS, NH, K_CL, DH = k.shape
            if K_CL <= 512:
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
                if attention_mask is not None:
                    scores = torch.where(attention_mask, scores,
                                    torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
                probs = torch.softmax(scores, dim=-1)
                out = torch.matmul(probs, v)
                return out

            head_outputs = []

            # Head blocking
            for head_block_idx in range(num_head_blocks):
                h_start = head_block_idx * head_block_size
                h_end = min(h_start + head_block_size, NH)
                num_h = h_end - h_start

                q_g = q[:, h_start:h_end, :, :]
                k_g = k[:, h_start:h_end, :, :]
                v_g = v[:, h_start:h_end, :, :]

                running_exp_sum = torch.zeros((BS, num_h, CL), device=q.device, dtype=q.dtype)
                running_max = torch.full((BS, num_h, CL), float('-inf'), device=q.device, dtype=q.dtype)
                output_blocks = torch.zeros_like(q_g)

                # Process K,V in blocks
                for kv_block_idx in range(target_blocks):
                    ki = block_positions[kv_block_idx]
                    if kv_block_idx == target_blocks - 1:
                        real_kv_len = CL - ki
                    else:
                        real_kv_len = block_positions[kv_block_idx + 1] - ki
                    k_block = k_g[:, :, ki:ki + real_kv_len, :]
                    v_block = v_g[:, :, ki:ki + real_kv_len, :]
                    qkblock = torch.matmul(q_g, k_block.transpose(-2, -1)) * scale_factor

                    prev_max = running_max.clone()
                    running_max = torch.maximum(prev_max, torch.max(qkblock, dim=-1)[0])

                    delta_max = prev_max - running_max
                    curr_exp = torch.exp(qkblock - running_max.unsqueeze(-1))
                    prev_exp_sum = running_exp_sum.clone()
                    # running_exp_sum = prev_exp_sum * torch.exp(delta_max) + curr_exp.sum(dim=-1)
                    curr_exp_sum = torch.einsum('bhqk->bhq', curr_exp)
                    running_exp_sum = prev_exp_sum * torch.exp(delta_max) + curr_exp_sum

                    # Compute softmax for this block
                    inv_running_exp_sum = 1.0 / running_exp_sum
                    softmax_qkblock = curr_exp * inv_running_exp_sum.unsqueeze(-1)

                    prev_out = output_blocks.clone()
                    rescale_factor = (prev_exp_sum * inv_running_exp_sum) * torch.exp(delta_max)
                    output_blocks = (rescale_factor.unsqueeze(-1) * prev_out +
                                torch.matmul(softmax_qkblock, v_block))
                head_outputs.append(output_blocks)

            out = torch.cat(head_outputs, dim=1)  # (BS, NH, CL, DH)
            return out

    def forward_head_q_blocked(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        BS, NH, CL, DH = q.shape  # Input: (BS, NH, CL, DH) = (1, 38, 4429, 64)
        scale_factor = 1.0 / math.sqrt(DH)
        head_block_size = int(os.environ.get('head_block_size', NH))
        num_head_blocks = math.ceil(NH / head_block_size)
        target_blocks_q = int(os.environ.get('num_q_blocks', CL))  # Q blocks
        q_block_positions = [(i * CL) // target_blocks_q for i in range(target_blocks_q)]
        if CL <= 512:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            if attention_mask is not None:
                scores = torch.where(attention_mask, scores,
                                torch.tensor(-1e4, dtype=scores.dtype, device=scores.device))
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
            return out

        head_outputs = []

        # Head blocking
        for head_block_idx in range(num_head_blocks):
            h_start = head_block_idx * head_block_size
            h_end = min(h_start + head_block_size, NH)

            q_g = q[:, h_start:h_end, :, :]
            k_g = k[:, h_start:h_end, :, :]
            v_g = v[:, h_start:h_end, :, :]

            q_output_list = []

            for q_block_idx in range(target_blocks_q):
                qi = q_block_positions[q_block_idx]
                if q_block_idx == target_blocks_q - 1:
                    real_q_len = CL - qi
                else:
                    real_q_len = q_block_positions[q_block_idx + 1] - qi

                q_block = q_g[:, :, qi:qi + real_q_len, :]

                # Matmul with 4D tensors
                scores = torch.matmul(q_block, k_g.transpose(-2, -1)) * scale_factor
                probs = torch.softmax(scores, dim=-1)
                out_block = torch.matmul(probs, v_g)

                q_output_list.append(out_block)

            head_output = torch.cat(q_output_list, dim=2)
            head_outputs.append(head_output)

        out = torch.cat(head_outputs, dim=1)  # (BS, NH, CL, DH)

        return out

    def _get_blocking_mode(self):
        """Get blocking mode from environment variable"""
        mode = os.environ.get('ATTENTION_BLOCKING_MODE', 'default').lower()
        valid_modes = ['kv', 'qkv', 'q', 'default']
        if mode not in valid_modes:
            raise ValueError(f"Invalid ATTENTION_BLOCKING_MODE: {mode}. Must be one of {valid_modes}")
        return mode

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2].type_as(hidden_states)
                sin = freqs_sin[..., 1::2].type_as(hidden_states)
                real = x1 * cos - x2 * sin
                img = x1 * sin + x2 * cos
                x_rot = torch.stack([real,img],dim=-1)
                return x_rot.flatten(-2).type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        blocking_mode = self._get_blocking_mode()
        # print('>>>>>>>>>>> blocking_mode', blocking_mode)
        if blocking_mode == "kv":
            hidden_states = self.forward_head_kv_blocked(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))
        elif blocking_mode == "q":
            hidden_states = self.forward_head_q_blocked(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))
        elif blocking_mode == "qkv":
            hidden_states = self.forward_head_qkv_blocked(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))
        else: # default
            hidden_states = self.forward_head_blocked(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2))

        hidden_states = hidden_states.transpose(1, 2)
        # hidden_states = dispatch_attention_fn(
        #     query,
        #     key,
        #     value,
        #     attn_mask=attention_mask,
        #     dropout_p=0.0,
        #     is_causal=False,
        #     backend=self._attention_backend,
        # )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class QEFFWanAttention(WanAttention):
    def __qeff_init__(self):
        processor = QEFFWanAttnProcessor()
        self.processor = processor


class QEFFWanTransformer3DModel(WanTransformer3DModel):

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
        r"Set the currently active adapters for use in the diffusion network"
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # Expand weights into a list, one entry per adapter
        # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        # Set None values to default of 1.0
        # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
        weights = [w if w is not None else 1.0 for w in weights]

        # e.g. [{...}, 7] -> [{expanded dict...}, 7]
        scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[self.config._class_name] # updated to use WanTransformer3DModel
        weights = scale_expansion_fn(self, weights)
        set_weights_and_activate_adapters(self, adapter_names, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        # timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        temb: torch.Tensor,
        timestep_proj: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # if attention_kwargs is not None:
        #     attention_kwargs = attention_kwargs.copy()
        #     lora_scale = attention_kwargs.pop("scale", 1.0)
        # else:
        #     lora_scale = 1.0

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(self, lora_scale)
        # else:
        #     if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        #         logger.warning(
        #             "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
        #         )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # rotary_emb = self.rope(hidden_states)
        rotary_emb = torch.split(rotary_emb, 1, dim=0)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        # if timestep.ndim == 2:
        #     ts_seq_len = timestep.shape[1]
        #     timestep = timestep.flatten()  # batch_size * seq_len
        # else:
        #     ts_seq_len = None

        # temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        #     timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        # )
        # if ts_seq_len is not None:
        #     # batch_size, seq_len, 6, inner_dim
        #     timestep_proj = timestep_proj.unflatten(2, (6, -1))
        # else:
        #     # batch_size, 6, inner_dim
        #     timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # hidden_states = hidden_states.reshape(
        #     batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        # )
        # hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        ## Compiler Fix ##
        output = hidden_states

        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
