# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
    _get_qkv_projections,
)

from QEfficient.diffusers.models.modeling_utils import compute_blocked_attention, get_attention_blocking_config
from QEfficient.utils.logging_utils import logger


def qeff_apply_rotary_emb(
    x: torch.Tensor, freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos, sin = freqs_cis  # [S, D]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    cos, sin = cos.to(x.device), sin.to(x.device)
    B, S, H, D = x.shape
    x_real, x_imag = x.reshape(B, -1, H, D // 2, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


class QEffFluxAttnProcessor(FluxAttnProcessor):
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "QEffFluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = qeff_apply_rotary_emb(query, image_rotary_emb)
            key = qeff_apply_rotary_emb(key, image_rotary_emb)

        # Get blocking configuration
        blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config()
        # Apply blocking using pipeline_utils
        hidden_states = compute_blocked_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            blocking_mode=blocking_mode,
            head_block_size=head_block_size,
            num_kv_blocks=num_kv_blocks,
            num_q_blocks=num_q_blocks,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class QEffFluxAttention(FluxAttention):
    def __qeff_init__(self):
        processor = QEffFluxAttnProcessor()
        self.processor = processor


class QEffFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        shift_msa, scale_msa, gate = torch.split(temb, 1)
        residual = hidden_states
        norm_hidden_states = self.norm(hidden_states, scale_msa, shift_msa)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        # if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(torch.finfo(torch.float32).min, torch.finfo(torch.float32).max)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


class QEffFluxTransformerBlock(FluxTransformerBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temb1 = tuple(torch.split(temb[:6], 1))
        temb2 = tuple(torch.split(temb[6:], 1))
        norm_hidden_states = self.norm1(hidden_states, shift_msa=temb1[0], scale_msa=temb1[1])
        gate_msa, shift_mlp, scale_mlp, gate_mlp = temb1[-4:]

        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, shift_msa=temb2[0], scale_msa=temb2[1])

        c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = temb2[-4:]

        joint_attention_kwargs = joint_attention_kwargs or {}

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        # if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QEffFluxTransformer2DModel(FluxTransformer2DModel):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {QEffFluxTransformerBlock, QEffFluxSingleTransformerBlock}

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        adaln_emb: torch.Tensor = None,
        adaln_single_emb: torch.Tensor = None,
        adaln_out: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        
        
        ## inputs for the cache
        prev_first_block_residuals: torch.tensor= None,
        prev_remain_block_residuals: torch.tensor = None,
        prev_remain_encoder_residuals: torch.tensor = None,
        cache_threshold: torch.tensor = None,
        # cache_warmup: torch.tensor =None,  # for now lets skip this
        current_step: torch.tensor = None,
        
        # end of inputs
    
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})
            
        # Here concept of first cache will be there
        # Initialize cache outputs to None (returned as-is when cache is disabled)
        cfbr, hrbr, ehrbr = None, None, None

        if cache_threshold is not None:
            hidden_states, encoder_hidden_states, cfbr, hrbr, ehrbr = self.forward_with_cache(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cache_threshold=cache_threshold,
                image_rotary_emb=image_rotary_emb,
                prev_first_block_residuals=prev_first_block_residuals,
                prev_remain_encoder_residuals=prev_remain_encoder_residuals,
                prev_remain_block_residuals=prev_remain_block_residuals,
                adaln_emb=adaln_emb,
                adaln_single_emb=adaln_single_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
   
        else:
            
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=adaln_emb[index_block],
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            for index_block, block in enumerate(self.single_transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=adaln_single_emb[index_block],
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = self.norm_out(hidden_states, adaln_out)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,cfbr, hrbr, ehrbr)

        return Transformer2DModelOutput(sample=output), cfbr, hrbr, ehrbr
    
    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.tensor,
        adaln_emb: torch.tensor,
        adaln_single_emb: torch.tensor,
        image_rotary_emb: torch.tensor,
        
        prev_first_block_residuals: torch.tensor= None,
        prev_remain_block_residuals: torch.tensor = None,
        prev_remain_encoder_residuals: torch.tensor = None,
        cache_threshold: torch.tensor = None,
        
        joint_attention_kwargs:Optional[Dict[str, Any]] = None,
    ):
        original_hidden_states=hidden_states
        # original_encoder_hidden_state=encoder_hidden_states
        
        encoder_hidden_states, hidden_states = self.transformer_blocks[0](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=adaln_emb[0],
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        
        current_first_cache_residuals= hidden_states - original_hidden_states
        
        similarity=self._check_similarity(current_first_cache_residuals, prev_first_block_residuals, cache_threshold)
        
        
        encoder_hidden_state_residual,hidden_state_residual =self._compute_remaining_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            adaln_emb=adaln_emb,
            adaln_single_emb=adaln_single_emb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs
        ) 
        
        # similarity < cache_threshold → cache HIT → reuse prev (cached) residuals
        # similarity >= cache_threshold → cache MISS → use freshly computed residuals
        final_hidden_state_residal = torch.where(
            (similarity < cache_threshold),
            prev_remain_block_residuals,    # cache HIT: reuse cached residual
            hidden_state_residual,          # cache MISS: use fresh residual
        )
        final_encoder_hidden_state_residual = torch.where(
            (similarity < cache_threshold),
            prev_remain_encoder_residuals,   # cache HIT: reuse cached residual
            encoder_hidden_state_residual,  # cache MISS: use fresh residual
        )
        
        final_hidden_state_output= hidden_states+final_hidden_state_residal
        final_encoder_hidden_state_output= encoder_hidden_states+final_encoder_hidden_state_residual
        
        return final_hidden_state_output, final_encoder_hidden_state_output, current_first_cache_residuals,final_hidden_state_residal, final_encoder_hidden_state_residual
    
    def _check_similarity(
        self,
        first_block_residual: torch.Tensor,
        prev_first_block_residual: torch.Tensor,
        cache_threshold: torch.tensor,
    ) -> torch.Tensor:
        """
        Compute cache decision (returns boolean tensor).

        Cache is used when:
        1. Not in warmup period (current_step >= cache_warmup_steps)
        2. Previous residual exists (not first step)
        3. Similarity is below threshold
        """
        # Compute similarity (L1 distance normalized by magnitude)
        # This must be computed BEFORE any conditional logic
        diff = (first_block_residual - prev_first_block_residual).abs().mean()
        norm = first_block_residual.abs().mean()
        
        similarity = diff / (norm + 1e-8)
        

        # is_similar = similarity < cache_threshold  # scalar bool tensor


        # use_cache = torch.where(
        #     current_step < cache_warmup_steps,
        #     torch.zeros_like(is_similar),  # During warmup: always False (same dtype as is_similar)
        #     is_similar,                    # If not warmup: use is_similar
        # )
        
        return   similarity
    
    def _compute_remaining_block(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.tensor,
        adaln_emb: torch.tensor,
        adaln_single_emb: torch.tensor,
        image_rotary_emb: torch.tensor,
        joint_attention_kwargs:Optional[Dict[str, Any]] = None,
    ):
        original_hidden_state=hidden_states
        original_encoder_hidden_state=encoder_hidden_states
        
        for index_block, block in enumerate(self.transformer_blocks[1:], start=1):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=adaln_emb[index_block],
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        for index_block, block in enumerate(self.single_transformer_blocks):
            
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=adaln_single_emb[index_block],
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        
        hidden_state_residual= hidden_states - original_hidden_state
        encoder_hidden_states_residual=encoder_hidden_states-original_encoder_hidden_state
        
        return encoder_hidden_states_residual, hidden_state_residual
