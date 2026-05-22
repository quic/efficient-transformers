# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
FLUX first-block-cache monkey patch utilities.

These helpers patch a Flux transformer wrapper instance at runtime:
1. patch underlying transformer `forward` to expose retained-state cache IO,
2. patch wrapper `get_onnx_params` to export retained-state tensors, and
3. patch wrapper `compile` to enable retained-state compilation with custom IO.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput


def _check_similarity(first_hidden_states_residuals: torch.Tensor, prev_first_hidden_states_residuals: torch.Tensor):
    """
    Compute normalized L1 difference between current and previous first-block residuals.
    """
    scale = 1.0 / 1024.0
    current = first_hidden_states_residuals * scale
    previous = prev_first_hidden_states_residuals * scale
    diff = (current - previous).abs().mean()
    norm = current.abs().mean()
    return diff / (norm + 1e-8)


def _flux_forward_with_first_block_cache(
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
    prev_first_hidden_states_residuals: Optional[torch.Tensor] = None,
    prev_hidden_states_residuals: Optional[torch.Tensor] = None,
    cache_threshold: Optional[float] = None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    Patched FLUX transformer forward with first-block-cache retained-state support.
    """
    if cache_threshold is None:
        return self._qeff_original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            adaln_emb=adaln_emb,
            adaln_single_emb=adaln_single_emb,
            adaln_out=adaln_out,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=return_dict,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    # First dual block output is the cache-key source.
    original_hidden_states = hidden_states
    encoder_hidden_states, hidden_states = self.transformer_blocks[0](
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=adaln_emb[0],
        image_rotary_emb=image_rotary_emb,
        joint_attention_kwargs=joint_attention_kwargs,
    )
    current_first_hidden_states_residuals = hidden_states - original_hidden_states

    ds_factor = self._qeff_first_block_cache_downsample_factor
    downsampled_first_hidden_states_residuals = current_first_hidden_states_residuals[..., ::ds_factor].contiguous()
    difference = _check_similarity(downsampled_first_hidden_states_residuals, prev_first_hidden_states_residuals)

    # Compute residual from the remaining blocks.
    remaining_hidden_states = hidden_states
    remaining_encoder_hidden_states = encoder_hidden_states

    for index_block, block in enumerate(self.transformer_blocks[1:], start=1):
        remaining_encoder_hidden_states, remaining_hidden_states = block(
            hidden_states=remaining_hidden_states,
            encoder_hidden_states=remaining_encoder_hidden_states,
            temb=adaln_emb[index_block],
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            if controlnet_blocks_repeat:
                remaining_hidden_states = (
                    remaining_hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                remaining_hidden_states = (
                    remaining_hidden_states + controlnet_block_samples[index_block // interval_control]
                )

    for index_block, block in enumerate(self.single_transformer_blocks):
        remaining_encoder_hidden_states, remaining_hidden_states = block(
            hidden_states=remaining_hidden_states,
            encoder_hidden_states=remaining_encoder_hidden_states,
            temb=adaln_single_emb[index_block],
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            remaining_hidden_states = (
                remaining_hidden_states + controlnet_single_block_samples[index_block // interval_control]
            )

    current_hidden_states_residuals = remaining_hidden_states - hidden_states

    cache_hit = difference < cache_threshold
    final_hidden_states_residuals = torch.where(
        cache_hit,
        prev_hidden_states_residuals,
        current_hidden_states_residuals,
    )

    hidden_states = hidden_states + final_hidden_states_residuals
    hidden_states = self.norm_out(hidden_states, adaln_out)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output, downsampled_first_hidden_states_residuals, final_hidden_states_residuals)

    return (
        Transformer2DModelOutput(sample=output),
        downsampled_first_hidden_states_residuals,
        final_hidden_states_residuals,
    )


def _get_onnx_params_with_first_block_cache(self):
    """
    Wrapper-level ONNX params for retained-state first-block-cache export.
    """
    example_inputs, dynamic_axes, output_names = self._qeff_original_get_onnx_params()

    hidden_states = example_inputs["hidden_states"]
    batch_size = hidden_states.shape[0]
    cl = hidden_states.shape[1]

    hidden_size = getattr(self.model, "inner_dim", None)
    if hidden_size is None:
        num_heads = self.model.config.get("num_attention_heads")
        head_dim = self.model.config.get("attention_head_dim")
        hidden_size = num_heads * head_dim

    ds_factor = self.model._qeff_first_block_cache_downsample_factor
    example_inputs["prev_hidden_states_residuals"] = torch.randn(batch_size, cl, hidden_size, dtype=torch.float32)
    example_inputs["prev_first_hidden_states_residuals"] = torch.randn(
        batch_size, cl, hidden_size // ds_factor, dtype=torch.float32
    )
    example_inputs["cache_threshold"] = torch.tensor(0.0, dtype=torch.float32)

    output_names.extend(
        [
            "prev_first_hidden_states_residuals_RetainedState",
            "prev_hidden_states_residuals_RetainedState",
        ]
    )

    dynamic_axes["prev_hidden_states_residuals"] = {0: "batch_size", 1: "cl"}
    dynamic_axes["prev_first_hidden_states_residuals"] = {0: "batch_size", 1: "cl"}

    return example_inputs, dynamic_axes, output_names


def _compile_with_first_block_cache(self, specializations, **compiler_options):
    """
    Wrapper-level compile with retained-state custom IO enabled.
    """
    custom_io = {
        "prev_first_hidden_states_residuals": "float16",
        "prev_hidden_states_residuals": "float16",
        "prev_first_hidden_states_residuals_RetainedState": "float16",
        "prev_hidden_states_residuals_RetainedState": "float16",
    }
    return self._compile(
        specializations=specializations,
        retained_state=True,
        custom_io=custom_io,
        **compiler_options,
    )


def enable_flux_first_block_cache(transformer_wrapper: Any, downsample_factor: int = 4) -> Any:
    """
    Enable first-block-cache on a FLUX transformer wrapper instance.

    Args:
        transformer_wrapper: QEff Flux transformer wrapper (`QEffFluxTransformerModel`)
        downsample_factor: hidden-dim downsampling factor for first-block cache key.
    """
    if getattr(transformer_wrapper, "_qeff_first_block_cache_enabled", False):
        return transformer_wrapper

    if downsample_factor <= 0:
        raise ValueError(f"downsample_factor must be > 0, got {downsample_factor}")

    hidden_size = getattr(transformer_wrapper.model, "inner_dim", None)
    if hidden_size is None:
        num_heads = transformer_wrapper.model.config.get("num_attention_heads")
        head_dim = transformer_wrapper.model.config.get("attention_head_dim")
        hidden_size = num_heads * head_dim
    if hidden_size % downsample_factor != 0:
        raise ValueError(
            f"downsample_factor must divide hidden_size exactly, got hidden_size={hidden_size}, "
            f"downsample_factor={downsample_factor}"
        )

    transformer_model = transformer_wrapper.model
    if not getattr(transformer_model, "_qeff_first_block_cache_patched", False):
        transformer_model._qeff_original_forward = transformer_model.forward
        transformer_model._qeff_first_block_cache_downsample_factor = downsample_factor
        transformer_model.forward = MethodType(_flux_forward_with_first_block_cache, transformer_model)
        transformer_model._qeff_first_block_cache_patched = True

    transformer_wrapper._qeff_original_get_onnx_params = transformer_wrapper.get_onnx_params
    transformer_wrapper.get_onnx_params = MethodType(_get_onnx_params_with_first_block_cache, transformer_wrapper)
    transformer_wrapper.compile = MethodType(_compile_with_first_block_cache, transformer_wrapper)
    transformer_wrapper._qeff_first_block_cache_enabled = True

    transformer_wrapper.hash_params["first_block_cache"] = True
    transformer_wrapper.hash_params["first_block_cache_downsample_factor"] = downsample_factor

    return transformer_wrapper
