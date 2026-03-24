# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
QEfficient WAN Transformer Implementation

This module provides optimized implementations of WAN transformers
with various attention blocking strategies for memory efficiency and performance optimization.
The implementation includes multiple blocking modes: head-only, KV-blocking, Q-blocking,
and combined QKV-blocking.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    WanTransformer3DModel,
    WanTransformerBlock,
    _get_qkv_projections,
)
from QEfficient.customop.ctx_scatter_gather import CtxGatherFunc3D,CtxScatterFunc3D
from diffusers.utils import set_weights_and_activate_adapters

from QEfficient.diffusers.models.modeling_utils import (
    compute_blocked_attention,
    get_attention_blocking_config,
)


class QEffWanAttnProcessor(WanAttnProcessor):
    """
    QEfficient WAN Attention Processor with Memory-Efficient Blocking Strategies.

    This processor implements multiple attention blocking modes to reduce memory usage
    and enable processing of longer sequences. It supports:
    - Head blocking: Process attention heads in chunks
    - KV blocking: Process key-value pairs in blocks
    - Q blocking: Process query tokens in blocks
    - QKV blocking: Combined query, key, and value blocking

    Environment Variables:
        ATTENTION_BLOCKING_MODE: Controls blocking strategy ('kv', 'q', 'qkv', 'default')
        head_block_size: Number of attention heads to process per block
        num_kv_blocks: Number of blocks for key-value processing
        num_q_blocks: Number of blocks for query processing
    """

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Main attention processing pipeline with support for multiple blocking strategies.

        This method orchestrates the complete attention computation including:
        1. QKV projection and normalization
        2. Rotary position embedding application
        3. Attention computation with selected blocking strategy
        4. Output projection

        Args:
            attn (WanAttention): The attention module instance
            hidden_states (torch.Tensor): Input hidden states
            encoder_hidden_states (Optional[torch.Tensor]): Cross-attention encoder states
            attention_mask (Optional[torch.Tensor]): Attention mask
            rotary_emb (Optional[Tuple[torch.Tensor, torch.Tensor]]): Rotary embeddings (cos, sin)

        Returns:
            torch.Tensor: Processed hidden states after attention
        """
        # Project inputs to query, key, value
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # Apply layer normalization to queries and keys
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Reshape for multi-head attention: (batch, seq, dim) -> (batch, seq, heads, head_dim)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Apply rotary position embeddings if provided
        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                """Apply rotary position embeddings to the input tensor."""
                # Split into real and imaginary parts for complex rotation
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2].type_as(hidden_states)
                sin = freqs_sin[..., 1::2].type_as(hidden_states)

                # Apply rotation: (x1 + ix2) * (cos + isin) = (x1*cos - x2*sin) + i(x1*sin + x2*cos)
                real = x1 * cos - x2 * sin
                img = x1 * sin + x2 * cos
                x_rot = torch.stack([real, img], dim=-1)
                return x_rot.flatten(-2).type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # Get blocking configuration
        blocking_mode, head_block_size, num_kv_blocks, num_q_blocks = get_attention_blocking_config()
        # Apply blocking using pipeline_utils
        hidden_states = compute_blocked_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            head_block_size,
            num_kv_blocks,
            num_q_blocks,
            blocking_mode=blocking_mode,
            attention_mask=attention_mask,
        )

        # Reshape back to original format
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        # Apply output projection layers
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class QEffWanAttention(WanAttention):
    """
    QEfficient WAN Attention module with optimized processor.

    This class extends the base WanAttention with QEfficient optimizations,
    automatically setting up the QEffWanAttnProcessor for memory-efficient
    attention computation.
    """

    def __qeff_init__(self):
        """Initialize the QEfficient attention processor."""
        processor = QEffWanAttnProcessor()
        self.processor = processor


class QEffWanTransformer3DModel(WanTransformer3DModel):
    """
    QEfficient 3D WAN Transformer Model with adapter support and optional first block cache.

    This model extends the base WanTransformer3DModel with QEfficient optimizations,
    including optional first block cache for faster inference.
    """

    def __qeff_init__(self):
        """
        Initialize QEfficient-specific attributes.

        Args:
            enable_first_cache: Whether to enable first block cache optimization
        """

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
        """
        Set the currently active adapters for use in the diffusion network.

        This method manages PEFT adapters, allowing for efficient fine-tuning
        and model customization without modifying the base model parameters.

        Args:
            adapter_names (Union[List[str], str]): Names of adapters to activate
            weights (Optional[Union[float, Dict, List[float], List[Dict], List[None]]]):
                Weights for each adapter. Can be:
                - Single float: Applied to all adapters
                - List of floats: One weight per adapter
                - Dict: Detailed weight configuration
                - None: Uses default weight of 1.0

        Raises:
            ValueError: If adapter names and weights lists have different lengths

        Note:
            - Adapters enable parameter-efficient fine-tuning
            - Multiple adapters can be active simultaneously with different weights
            - Weights control the influence of each adapter on the model output
        """
        # Normalize adapter names to list format
        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # Expand weights into a list, one entry per adapter
        # Examples for 2 adapters: [{...}, 7] -> [7,7] ; None -> [None, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        # Set None values to default of 1.0
        # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
        weights = [w if w is not None else 1.0 for w in weights]

        # Expand weights using model-specific scaling function
        # e.g. [{...}, 7] -> [{expanded dict...}, 7]
        scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING[
            self.config._class_name
        ]  # updated to use WanTransformer3DModel
        weights = scale_expansion_fn(self, weights)
        set_weights_and_activate_adapters(self, adapter_names, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        temb: torch.Tensor,
        timestep_proj: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        # Cache inputs (only used when enable_first_cache=True)
        prev_remaining_blocks_residual: Optional[torch.Tensor] = None,
        use_cache: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the 3D WAN Transformer with optional first block cache support.

        When enable_first_cache=True and cache inputs are provided:
            - Executes first block always
            - Conditionally executes remaining blocks based on similarity
            - Returns cache outputs for next iteration

        Otherwise:
            - Standard forward pass

        Args:
            hidden_states (torch.Tensor): Input tensor to transform
            encoder_hidden_states (torch.Tensor): Cross-attention encoder states
            rotary_emb (torch.Tensor): Rotary position embeddings
            temb (torch.Tensor): Time embedding for diffusion process
            timestep_proj (torch.Tensor): Projected timestep embeddings
            encoder_hidden_states_image (Optional[torch.Tensor]): Image encoder states for I2V
            prev_first_block_residual (Optional[torch.Tensor]): Cached first block residual from previous step
            prev_remaining_blocks_residual (Optional[torch.Tensor]): Cached remaining blocks residual from previous step
            current_step (Optional[torch.Tensor]): Current denoising step number (for cache warmup logic)
            return_dict (bool): Whether to return a dictionary or tuple
            attention_kwargs (Optional[Dict[str, Any]]): Additional attention arguments

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]:
                Transformed hidden states, either as tensor or in a dictionary.
                When cache is enabled, includes first_block_residual and remaining_blocks_residual.
        """
        # Check if cache should be used
        cache_enabled = getattr(self, 'enable_first_cache', False)
    

        # Prepare rotary embeddings by splitting along batch dimension
        rotary_emb = torch.split(rotary_emb, 1, dim=0)

        # # Apply patch embedding and reshape for transformer processing
        # hidden_states = self.patch_embedding(hidden_states)
        # hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # # Concatenate image and text encoder states if image conditioning is present
        # if encoder_hidden_states_image is not None:
        #     encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Execute transformer blocks (with or without cache)
        if cache_enabled:
            hidden_states, remaining_residual = self._forward_blocks_with_cache(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep_proj=timestep_proj,
                rotary_emb=rotary_emb,
                prev_remaining_blocks_residual=prev_remaining_blocks_residual,
                use_cache=use_cache,
            )
        else:
            # Standard forward pass
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            first_residual = None
            remaining_residual = None

        # Output normalization, projection & unpatchify
        if temb.ndim == 3:
            # Handle 3D time embeddings: batch_size, seq_len, inner_dim (WAN 2.2 T2V)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # Handle 2D time embeddings: batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Ensure tensors are on the same device as hidden_states
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # Apply adaptive layer normalization with time conditioning
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)

        # Final output projection
        hidden_states = self.proj_out(hidden_states)

        # Store output for return (compiler optimization)
        output = hidden_states

        # Return in requested format
        # Note: When cache is enabled, we always return tuple format
        # because Transformer2DModelOutput doesn't support custom fields
        if cache_enabled:
            return (output, remaining_residual)
        
        if not return_dict:
            return (output,)
        
        return Transformer2DModelOutput(sample=output)

    def _forward_blocks_with_cache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: torch.Tensor,
        prev_remaining_blocks_residual: torch.Tensor,
        use_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Core cache logic - AOT compilable.

        Executes the first transformer block always, then conditionally executes
        remaining blocks based on similarity of first block output to previous step.

        Single torch.where pattern (matches other modeling files):
          - True  branch (cache hit):  prev_remaining_blocks_residual  — cheap, always available
          - False branch (cache miss): new_remaining_blocks_residual   — expensive, compiler skips when use_cache=True
          - final_output = hidden_states + final_remaining_residual
            (equivalent to: torch.where(use_cache, hs+prev_res, remaining_output))
        """
        # Step 1: Always execute first block
        # original_hidden_states = hidden_states
        # hidden_states = self.blocks[0](hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        # first_block_residual = hidden_states - original_hidden_states

        # # condition
        # similarty = self._check_similarity(
        #     first_block_residual, prev_first_block_residual
        # )
        
        
        # if use_cache false
        original_hidden_states = hidden_states
        for block in self.blocks[1:]:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
        new_remaining_blocks_residual= hidden_states - original_hidden_states
        
        # If use_cache true
        # Final_remaining_residuals
        final_remaining_residual =torch.where(
            use_cache.bool(),
            prev_remaining_blocks_residual,
            new_remaining_blocks_residual)  # Placeholder, will be replaced by new_remaining_blocks_re
        final_output = original_hidden_states + final_remaining_residual
        
        return final_output, final_remaining_residual

    def _check_similarity(
        self,
        first_block_residual: torch.Tensor,
        prev_first_block_residual: torch.Tensor,
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

    def _compute_remaining_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute transformer blocks 1 to N and return the residual for caching.

        Returns only the residual (output - input) so the caller can use a single
        torch.where to select between prev_residual and new_residual, then derive
        the final output as: hidden_states + selected_residual.
        """
        original_hidden_states = hidden_states

        # Execute remaining blocks (blocks[1:])
        for block in self.blocks[1:]:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )

        # Return only the residual; output = original_hidden_states + residual
        return hidden_states - original_hidden_states


class QEffWanUnifiedWrapper(nn.Module):
    """
    A wrapper class that combines WAN high and low noise transformers into a single unified transformer.

    This wrapper dynamically selects between high and low noise transformers based on the timestep shape
    in the ONNX graph during inference. This approach enables efficient deployment of both transformer
    variants in a single model.

    When first block cache is enabled, this wrapper maintains separate cache states for high and low
    noise transformers, as they are different models with different block structures.

    Attributes:
        transformer_high(nn.Module): The high noise transformer component
        transformer_low(nn.Module): The low noise transformer component
        config: Configuration shared between both transformers (from high noise transformer)
    """

    def __init__(self, transformer_high, transformer_low):
        super().__init__()
        self.transformer_high = transformer_high
        self.transformer_low = transformer_low
        # Both high and low noise transformers share the same configuration
        self.config = transformer_high.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        Notes:
            This method should return the *class object* (not an instance).
            Downstream code can use this to find/build subfunctions for repeated blocks.
        """
        return {WanTransformerBlock}

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        rotary_emb,
        temb,
        timestep_proj,
        tsp,
        # Separate cache inputs for high and low noise transformers
        # prev_first_block_residual_high: Optional[torch.Tensor] = None,
        prev_remaining_blocks_residual_high: Optional[torch.Tensor] = None,
        # prev_first_block_residual_low: Optional[torch.Tensor] = None,
        prev_remaining_blocks_residual_low: Optional[torch.Tensor] = None,
        # current_step: Optional[torch.Tensor] = None,
        # cache_threshold: Optional[torch.Tensor] = None,
        # warmup_steps: Optional[torch.Tensor] = None,
        use_cache: Optional[torch.Tensor] = None,
        attention_kwargs=None,
        return_dict=False,
    ):
        """
        Forward pass with separate cache management for high and low noise transformers.

        Args:
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states for cross-attention
            rotary_emb: Rotary position embeddings
            temb: Time embeddings
            timestep_proj: Projected timestep embeddings
            tsp: Transformer stage pointer (determines high vs low noise)
            prev_first_block_residual_high: Cache for high noise transformer's first block
            prev_remaining_blocks_residual_high: Cache for high noise transformer's remaining blocks
            prev_first_block_residual_low: Cache for low noise transformer's first block
            prev_remaining_blocks_residual_low: Cache for low noise transformer's remaining blocks
            current_step: Current denoising step number
            attention_kwargs: Additional attention arguments
            return_dict: Whether to return dictionary or tuple

        Returns:
            If cache enabled: (noise_pred, first_residual_high, remaining_residual_high, 
                              first_residual_low, remaining_residual_low)
            Otherwise: noise_pred
        """
        # Condition based on timestep shape
        is_high_noise = tsp.shape[0] == torch.tensor(1)

        # Check if cache is enabled (both transformers should have same setting)
        cache_enabled = getattr(self.transformer_high, 'enable_first_cache', False)
        high_hs = hidden_states.detach()
        ehs = encoder_hidden_states.detach() 
        rhs = rotary_emb.detach()
        ths = temb.detach()
        projhs = timestep_proj.detach()

        # Execute high noise transformer with its cache
        if cache_enabled:
            # When cache is enabled, transformer returns tuple: (output, first_residual, remaining_residual)
            high_output = self.transformer_high(
                hidden_states=high_hs,
                encoder_hidden_states=ehs,
                rotary_emb=rhs,
                temb=ths,
                timestep_proj=projhs,
                prev_remaining_blocks_residual=prev_remaining_blocks_residual_high,
                use_cache=use_cache,
                attention_kwargs=attention_kwargs,
                return_dict=False,  # Must be False when cache is enabled
            )
            noise_pred_high, remaining_residual_high = high_output
        else:
            noise_pred_high = self.transformer_high(
                hidden_states=high_hs,
                encoder_hidden_states=ehs,
                rotary_emb=rhs,
                temb=ths,
                timestep_proj=projhs,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )[0]
            remaining_residual_high = None

        # Execute low noise transformer with its cache
        if cache_enabled:
            # When cache is enabled, transformer returns tuple: (output, first_residual, remaining_residual)
            low_output = self.transformer_low(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_emb=rotary_emb,
                temb=temb,
                timestep_proj=timestep_proj,
                prev_remaining_blocks_residual=prev_remaining_blocks_residual_low,
                use_cache=use_cache,
                attention_kwargs=attention_kwargs,
                return_dict=False,  # Must be False when cache is enabled
            )
            noise_pred_low, remaining_residual_low = low_output
        else:
            noise_pred_low = self.transformer_low(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_emb=rotary_emb,
                temb=temb,
                timestep_proj=timestep_proj,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )[0]
            remaining_residual_low = None

        # Select output based on timestep condition
        noise_pred = torch.where(is_high_noise, noise_pred_high, noise_pred_low)
        new_remaining_residual_high = torch.where(is_high_noise, remaining_residual_high, prev_remaining_blocks_residual_high ) 
        new_remaining_residual_low = torch.where(is_high_noise, prev_remaining_blocks_residual_low, remaining_residual_low) 

        # Return with cache outputs if enabled
        if cache_enabled:
            return (
                noise_pred,
                new_remaining_residual_high,
                new_remaining_residual_low,
            )
        return noise_pred
