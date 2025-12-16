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

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    WanTransformer3DModel,
    _get_qkv_projections,
)
from diffusers.utils import set_weights_and_activate_adapters


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

    def forward_head_qkv_blocked(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Forward pass with combined Query, Key, Value blocking and head blocking.

        This method implements the most memory-efficient attention computation by blocking
        along all three dimensions: heads, queries, and key-values.
        Args:
            q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
            k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
            v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
            attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

        Returns:
            torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)

        Note:
            - BS: Batch Size, NH: Number of Heads, CL: Context Length, DH: Head Dimension
        """
        BS, NH, CL, DH = q.shape  # Input: (BS, NH, CL, DH) = (1, 38, 4429, 64)
        scale_factor = 1.0 / math.sqrt(DH)
        # Get blocking configuration from environment variables
        head_block_size = int(os.environ.get("head_block_size", NH))
        num_head_blocks = math.ceil(NH / head_block_size)
        target_blocks_kv = int(os.environ.get("num_kv_blocks", CL))  # KV blocks
        target_blocks_q = int(os.environ.get("num_q_blocks", CL))  # Q blocks
        # Calculate block positions for even distribution
        kv_block_positions = [(i * CL) // target_blocks_kv for i in range(target_blocks_kv)]
        q_block_positions = [(i * CL) // target_blocks_q for i in range(target_blocks_q)]

        BS, NH, K_CL, DH = k.shape
        # Optimization: Use standard attention for small sequences
        if K_CL <= 512:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            if attention_mask is not None:
                scores = torch.where(
                    attention_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device)
                )
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
            return out

        head_outputs = []

        # Process attention heads in blocks to reduce memory usage
        for head_block_idx in range(num_head_blocks):
            h_start = head_block_idx * head_block_size
            h_end = min(h_start + head_block_size, NH)
            num_h = h_end - h_start

            # Extract current head block
            q_g = q[:, h_start:h_end, :, :]
            k_g = k[:, h_start:h_end, :, :]
            v_g = v[:, h_start:h_end, :, :]
            q_output_list = []

            # Process queries in blocks within each head block
            for q_block_idx in range(target_blocks_q):
                qi = q_block_positions[q_block_idx]

                # Calculate actual Q block size (handle remainder for last block)
                if q_block_idx == target_blocks_q - 1:
                    real_q_len = CL - qi
                else:
                    real_q_len = q_block_positions[q_block_idx + 1] - qi

                q_block = q_g[:, :, qi : qi + real_q_len, :]

                # Initialize online softmax statistics for this Q block
                running_exp_sum = torch.zeros((BS, num_h, real_q_len), device=q.device, dtype=q.dtype)
                running_max = torch.full((BS, num_h, real_q_len), float("-inf"), device=q.device, dtype=q.dtype)
                output_blocks = torch.zeros((BS, num_h, real_q_len, DH), device=q.device, dtype=q.dtype)

                # Process K,V in blocks for this Q block (online softmax)
                for kv_block_idx in range(target_blocks_kv):
                    ki = kv_block_positions[kv_block_idx]

                    # Calculate actual KV block size
                    if kv_block_idx == target_blocks_kv - 1:
                        real_kv_len = CL - ki
                    else:
                        real_kv_len = kv_block_positions[kv_block_idx + 1] - ki

                    k_block = k_g[:, :, ki : ki + real_kv_len, :]
                    v_block = v_g[:, :, ki : ki + real_kv_len, :]

                    # Compute attention scores for current Q-K block
                    qkblock = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale_factor

                    # Online softmax: Update running maximum
                    prev_max = running_max.clone()
                    if qkblock.shape[-1] == 0:
                        running_max = prev_max
                    else:
                        running_max = torch.maximum(prev_max, torch.max(qkblock, dim=-1)[0])

                    # Calculate adjustment factor for numerical stability
                    delta_max = prev_max - running_max
                    curr_exp = torch.exp(qkblock - running_max.unsqueeze(-1))

                    # Online softmax: Update running sum of exponentials
                    prev_exp_sum = running_exp_sum.clone()
                    curr_exp_sum = torch.einsum("bhqk->bhq", curr_exp)
                    running_exp_sum = prev_exp_sum * torch.exp(delta_max) + curr_exp_sum

                    # Compute normalized attention weights for this block
                    inv_running_exp_sum = 1.0 / running_exp_sum
                    softmax_qkblock = curr_exp * inv_running_exp_sum.unsqueeze(-1)

                    # Online softmax: Update output with rescaling of previous blocks
                    prev_out = output_blocks.clone()
                    rescale_factor = (prev_exp_sum * inv_running_exp_sum) * torch.exp(delta_max)
                    output_blocks = rescale_factor.unsqueeze(-1) * prev_out + torch.matmul(softmax_qkblock, v_block)

                q_output_list.append(output_blocks)

            # Concatenate all Q blocks for this head block
            head_output = torch.cat(q_output_list, dim=2)
            head_outputs.append(head_output)

        # Concatenate all head blocks
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
        """
        Forward pass with head-only blocking (default mode).

        This method processes attention heads in blocks while computing full attention
        matrices for each head block. It's less memory-efficient than other blocking
        modes but simpler and faster for moderate sequence lengths.

        Args:
            q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
            k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
            v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
            attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

        Returns:
            torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)
        """
        BS, NH, CL, DH = q.shape  # Input: (BS, NH, CL, DH) = (1, 38, 4429, 64)
        scale_factor = 1.0 / math.sqrt(DH)

        # Get head blocking configuration
        head_block_size = int(os.environ.get("head_block_size", NH))
        num_head_blocks = math.ceil(NH / head_block_size)

        # Optimization: Handle small sequences with standard attention
        BS, NH, K_CL, DH = k.shape
        if K_CL <= 512:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            if attention_mask is not None:
                scores = torch.where(
                    attention_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device)
                )
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
            return out

        outputs = []

        # Process each head block independently
        for head_block_idx in range(num_head_blocks):
            h_start = head_block_idx * head_block_size
            h_end = min(h_start + head_block_size, NH)

            # Extract head blocks
            q_g = q[:, h_start:h_end, :, :]
            k_g = k[:, h_start:h_end, :, :]
            v_g = v[:, h_start:h_end, :, :]

            # Compute full attention matrix for this head block
            qkblock = torch.matmul(q_g, k_g.transpose(-2, -1)) * scale_factor  # (BS, num_h, CL, CL)

            # Standard softmax computation
            probs = torch.softmax(qkblock, dim=-1)

            # Compute attention output
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
        """
        Forward pass with Key-Value blocking and head blocking.

        This method processes key-value pairs in blocks while keeping queries intact.
        It uses online softmax to maintain numerical stability and reduce memory usage
        compared to computing full attention matrices.

        Args:
            q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
            k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
            v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
            attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

        Returns:
            torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)
        """
        BS, NH, CL, DH = q.shape  # Input: (BS, NH, CL, DH) = (1, 38, 4429, 64)
        scale_factor = 1.0 / math.sqrt(DH)

        # Get blocking configuration
        head_block_size = int(os.environ.get("head_block_size", NH))
        num_head_blocks = math.ceil(NH / head_block_size)
        target_blocks = int(os.environ.get("num_kv_blocks", CL))
        block_positions = [(i * CL) // target_blocks for i in range(target_blocks)]

        # Handle small sequences with standard attention
        BS, NH, K_CL, DH = k.shape
        if K_CL <= 512:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            if attention_mask is not None:
                scores = torch.where(
                    attention_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device)
                )
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
            return out

        head_outputs = []

        # Process each head block
        for head_block_idx in range(num_head_blocks):
            h_start = head_block_idx * head_block_size
            h_end = min(h_start + head_block_size, NH)
            num_h = h_end - h_start

            q_g = q[:, h_start:h_end, :, :]
            k_g = k[:, h_start:h_end, :, :]
            v_g = v[:, h_start:h_end, :, :]

            # Initialize online softmax statistics
            running_exp_sum = torch.zeros((BS, num_h, CL), device=q.device, dtype=q.dtype)
            running_max = torch.full((BS, num_h, CL), float("-inf"), device=q.device, dtype=q.dtype)
            output_blocks = torch.zeros_like(q_g)

            # Process K,V in blocks using online softmax
            for kv_block_idx in range(target_blocks):
                ki = block_positions[kv_block_idx]

                # Calculate KV block size
                if kv_block_idx == target_blocks - 1:
                    real_kv_len = CL - ki
                else:
                    real_kv_len = block_positions[kv_block_idx + 1] - ki

                k_block = k_g[:, :, ki : ki + real_kv_len, :]
                v_block = v_g[:, :, ki : ki + real_kv_len, :]

                # Compute attention scores for current KV block
                qkblock = torch.matmul(q_g, k_block.transpose(-2, -1)) * scale_factor

                # Online softmax: Update running maximum
                prev_max = running_max.clone()
                running_max = torch.maximum(prev_max, torch.max(qkblock, dim=-1)[0])

                # Calculate numerical stability adjustments
                delta_max = prev_max - running_max
                curr_exp = torch.exp(qkblock - running_max.unsqueeze(-1))

                # Update running sum of exponentials
                prev_exp_sum = running_exp_sum.clone()
                curr_exp_sum = torch.einsum("bhqk->bhq", curr_exp)
                running_exp_sum = prev_exp_sum * torch.exp(delta_max) + curr_exp_sum

                # Compute normalized attention weights
                inv_running_exp_sum = 1.0 / running_exp_sum
                softmax_qkblock = curr_exp * inv_running_exp_sum.unsqueeze(-1)

                # Update output with rescaling
                prev_out = output_blocks.clone()
                rescale_factor = (prev_exp_sum * inv_running_exp_sum) * torch.exp(delta_max)
                output_blocks = rescale_factor.unsqueeze(-1) * prev_out + torch.matmul(softmax_qkblock, v_block)

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
        """
        Forward pass with Query blocking and head blocking.

        This method processes query tokens in blocks while keeping key-value pairs intact.
        It's useful when the sequence length is large but memory constraints are primarily
        due to the query dimension.

        Args:
            q (torch.FloatTensor): Query tensor of shape (BS, NH, CL, DH)
            k (torch.FloatTensor): Key tensor of shape (BS, NH, CL, DH)
            v (torch.FloatTensor): Value tensor of shape (BS, NH, CL, DH)
            attention_mask (Optional[torch.FloatTensor]): Attention mask tensor

        Returns:
            torch.FloatTensor: Attention output of shape (BS, NH, CL, DH)

        Note:
            This method computes full attention for each query block, making it
            less memory-efficient than KV or QKV blocking for very long sequences.
        """
        BS, NH, CL, DH = q.shape  # Input: (BS, NH, CL, DH) = (1, 38, 4429, 64)
        scale_factor = 1.0 / math.sqrt(DH)

        # Get blocking configuration
        head_block_size = int(os.environ.get("head_block_size", NH))
        num_head_blocks = math.ceil(NH / head_block_size)
        target_blocks_q = int(os.environ.get("num_q_blocks", CL))  # Q blocks
        q_block_positions = [(i * CL) // target_blocks_q for i in range(target_blocks_q)]

        # Handle small sequences with standard attention
        if CL <= 512:
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            if attention_mask is not None:
                scores = torch.where(
                    attention_mask, scores, torch.tensor(-1e4, dtype=scores.dtype, device=scores.device)
                )
            probs = torch.softmax(scores, dim=-1)
            out = torch.matmul(probs, v)
            return out

        head_outputs = []

        # Process each head block
        for head_block_idx in range(num_head_blocks):
            h_start = head_block_idx * head_block_size
            h_end = min(h_start + head_block_size, NH)

            q_g = q[:, h_start:h_end, :, :]
            k_g = k[:, h_start:h_end, :, :]
            v_g = v[:, h_start:h_end, :, :]

            q_output_list = []

            # Process queries in blocks
            for q_block_idx in range(target_blocks_q):
                qi = q_block_positions[q_block_idx]

                # Calculate Q block size
                if q_block_idx == target_blocks_q - 1:
                    real_q_len = CL - qi
                else:
                    real_q_len = q_block_positions[q_block_idx + 1] - qi

                q_block = q_g[:, :, qi : qi + real_q_len, :]

                # Compute attention for this query block against all keys
                scores = torch.matmul(q_block, k_g.transpose(-2, -1)) * scale_factor
                probs = torch.softmax(scores, dim=-1)
                out_block = torch.matmul(probs, v_g)

                q_output_list.append(out_block)

            # Concatenate query blocks
            head_output = torch.cat(q_output_list, dim=2)
            head_outputs.append(head_output)

        out = torch.cat(head_outputs, dim=1)  # (BS, NH, CL, DH)
        return out

    def _get_blocking_mode(self):
        """
        Get the attention blocking mode from environment variable.

        Returns:
            str: The blocking mode to use ('kv', 'qkv', 'q', or 'default')

        Raises:
            ValueError: If an invalid blocking mode is specified

        Environment Variables:
            ATTENTION_BLOCKING_MODE: Controls the blocking strategy
                - 'kv': Use key-value blocking
                - 'q': Use query blocking
                - 'qkv': Use combined query, key-value blocking (most memory efficient)
                - 'default': Use head-only blocking (fastest for moderate sequences)
        """
        mode = os.environ.get("ATTENTION_BLOCKING_MODE", "default").lower()
        valid_modes = ["kv", "qkv", "q", "default"]
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

        # Select and apply the appropriate blocking strategy
        blocking_mode = self._get_blocking_mode()
        if blocking_mode == "kv":
            # KV blocking: Most memory efficient for long sequences
            hidden_states = self.forward_head_kv_blocked(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            )
        elif blocking_mode == "q":
            # Q blocking: Efficient when query dimension is the bottleneck
            hidden_states = self.forward_head_q_blocked(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            )
        elif blocking_mode == "qkv":
            # QKV blocking: Maximum memory efficiency for very long sequences
            hidden_states = self.forward_head_qkv_blocked(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            )
        else:  # default
            # Head-only blocking: Fastest for moderate sequences
            hidden_states = self.forward_head_blocked(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))

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
    QEfficient 3D WAN Transformer Model with adapter support.

    This model extends the base WanTransformer3DModel with QEfficient optimizations.
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
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the 3D WAN Transformer.

        This method implements the complete forward pass including:
        1. Patch embedding of input
        2. Rotary embedding preparation
        3. Cross-attention with encoder states
        4. Transformer block processing (with optional gradient checkpointing)
        5. Output normalization and projection

        Args:
            hidden_states (torch.Tensor): Input tensor to transform
            encoder_hidden_states (torch.Tensor): Cross-attention encoder states
            rotary_emb (torch.Tensor): Rotary position embeddings
            temb (torch.Tensor): Time embedding for diffusion process
            timestep_proj (torch.Tensor): Projected timestep embeddings
            encoder_hidden_states_image (Optional[torch.Tensor]): Image encoder states for I2V
            return_dict (bool): Whether to return a dictionary or tuple
            attention_kwargs (Optional[Dict[str, Any]]): Additional attention arguments

        Returns:
            Union[torch.Tensor, Dict[str, torch.Tensor]]:
                Transformed hidden states, either as tensor or in a dictionary
        """
        # Prepare rotary embeddings by splitting along batch dimension
        rotary_emb = torch.split(rotary_emb, 1, dim=0)

        # Apply patch embedding and reshape for transformer processing
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Concatenate image and text encoder states if image conditioning is present
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # Process through transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # Use gradient checkpointing to save memory during training
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            # Standard forward pass
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

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
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
