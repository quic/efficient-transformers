# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------
"""
WAN first-block-cache monkey patch utilities.

These helpers patch a non-unified WAN transformer wrapper instance at runtime:
1. patch underlying transformer `forward` to expose retained-state cache IO,
2. patch wrapper `get_onnx_params` to export retained-state tensors, and
3. patch wrapper `compile` to enable retained-state compilation with custom IO.
"""

from __future__ import annotations

import time
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from QEfficient.utils import constants


def _check_similarity(first_block_residuals: torch.Tensor, prev_first_block_residuals: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized L1 difference between current and previous first-block residuals.
    """
    scale = 1.0 / 1024.0
    current = first_block_residuals * scale
    previous = prev_first_block_residuals * scale
    diff = (current - previous).abs().mean()
    norm = current.abs().mean()
    return diff / (norm + 1e-8)


def _wan_forward_with_first_block_cache(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    temb: torch.Tensor,
    timestep_proj: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    prev_first_block_residuals: Optional[torch.Tensor] = None,
    prev_remain_block_residuals: Optional[torch.Tensor] = None,
    cache_threshold: Optional[float] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Patched WAN transformer forward with first-block-cache retained-state support.
    """
    if cache_threshold is None:
        return self._qeff_original_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            rotary_emb=rotary_emb,
            temb=temb,
            timestep_proj=timestep_proj,
            encoder_hidden_states_image=encoder_hidden_states_image,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

    rotary_emb = torch.split(rotary_emb, 1, dim=0)
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    first_block_out = self.blocks[0](hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
    current_first_block_residuals = first_block_out - hidden_states

    ds_factor = self._qeff_first_block_cache_downsample_factor
    downsampled_first_block_residuals = current_first_block_residuals[..., ::ds_factor].contiguous()

    difference = _check_similarity(downsampled_first_block_residuals, prev_first_block_residuals)

    remaining_hidden_state = first_block_out
    for block in self.blocks[1:]:
        remaining_hidden_state = block(remaining_hidden_state, encoder_hidden_states, timestep_proj, rotary_emb)
    current_remain_block_residuals = remaining_hidden_state - first_block_out

    cache_hit = difference < cache_threshold
    final_remaining_block_residual = torch.where(
        cache_hit,
        prev_remain_block_residuals,
        current_remain_block_residuals,
    )

    hidden_states = final_remaining_block_residual + first_block_out

    if temb.ndim == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output, downsampled_first_block_residuals, final_remaining_block_residual)

    return (
        Transformer2DModelOutput(sample=output),
        downsampled_first_block_residuals,
        final_remaining_block_residual,
    )


def _get_onnx_params_with_first_block_cache(self):
    """
    Wrapper-level ONNX params for retained-state first-block-cache export.
    """
    example_inputs, dynamic_axes, output_names = self._qeff_original_get_onnx_params()

    batch_size = constants.WAN_ONNX_EXPORT_BATCH_SIZE
    cl = constants.WAN_ONNX_EXPORT_CL_45P
    cfg = self.model.config
    hidden_size = getattr(cfg, "hidden_size", None)
    if hidden_size is None and hasattr(cfg, "get"):
        hidden_size = cfg.get("hidden_size")
    if hidden_size is None:
        num_heads = getattr(cfg, "num_attention_heads", None)
        if num_heads is None and hasattr(cfg, "get"):
            num_heads = cfg.get("num_attention_heads")
        head_dim = getattr(cfg, "attention_head_dim", None)
        if head_dim is None and hasattr(cfg, "get"):
            head_dim = cfg.get("attention_head_dim")
        hidden_size = num_heads * head_dim

    ds_factor = self.model._qeff_first_block_cache_downsample_factor

    example_inputs["prev_remain_block_residuals"] = torch.randn(batch_size, cl, hidden_size, dtype=torch.float32)
    example_inputs["prev_first_block_residuals"] = torch.randn(
        batch_size, cl, hidden_size // ds_factor, dtype=torch.float32
    )
    example_inputs["cache_threshold"] = torch.tensor(0.0, dtype=torch.float32)

    output_names.extend(
        [
            "prev_first_block_residuals_RetainedState",
            "prev_remain_block_residuals_RetainedState",
        ]
    )

    dynamic_axes["prev_remain_block_residuals"] = {0: "batch_size", 1: "cl"}
    dynamic_axes["prev_first_block_residuals"] = {0: "batch_size", 1: "cl"}

    return example_inputs, dynamic_axes, output_names


def _compile_with_first_block_cache(self, specializations, **compiler_options):
    """
    Wrapper-level compile with retained-state custom IO enabled.
    """
    custom_io = {
        "prev_first_block_residuals": "float16",
        "prev_remain_block_residuals": "float16",
        "prev_first_block_residuals_RetainedState": "float16",
        "prev_remain_block_residuals_RetainedState": "float16",
    }
    return self._compile(
        specializations=specializations,
        retained_state=True,
        custom_io=custom_io,
        **compiler_options,
    )


def enable_wan_first_block_cache(transformer_wrapper: Any, downsample_factor: int = 4) -> Any:
    """
    Enable first-block-cache on a non-unified WAN transformer wrapper instance.

    Args:
        transformer_wrapper: QEff WAN non-unified transformer wrapper (`QEffWanTransformer`)
        downsample_factor: hidden-dim downsampling factor for first-block cache key.
    """
    if getattr(transformer_wrapper, "_qeff_first_block_cache_enabled", False):
        return transformer_wrapper

    if downsample_factor <= 0:
        raise ValueError(f"downsample_factor must be > 0, got {downsample_factor}")

    transformer_model = transformer_wrapper.model
    if not getattr(transformer_model, "_qeff_first_block_cache_patched", False):
        transformer_model._qeff_original_forward = transformer_model.forward
        transformer_model._qeff_first_block_cache_downsample_factor = downsample_factor
        transformer_model.forward = MethodType(_wan_forward_with_first_block_cache, transformer_model)
        transformer_model._qeff_first_block_cache_patched = True

    transformer_wrapper._qeff_original_get_onnx_params = transformer_wrapper.get_onnx_params
    transformer_wrapper.get_onnx_params = MethodType(_get_onnx_params_with_first_block_cache, transformer_wrapper)
    transformer_wrapper.compile = MethodType(_compile_with_first_block_cache, transformer_wrapper)
    transformer_wrapper._qeff_first_block_cache_enabled = True

    transformer_wrapper.hash_params["first_block_cache"] = True
    transformer_wrapper.hash_params["first_block_cache_downsample_factor"] = downsample_factor

    return transformer_wrapper


def run_wan_non_unified_first_block_cache_denoise(
    pipeline: Any,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    batch_size: int,
    guidance_scale: float,
    guidance_scale_2: float,
    boundary_timestep: Optional[float],
    transformer_dtype: torch.dtype,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    mask: torch.Tensor,
    num_inference_steps: int,
    num_warmup_steps: int,
    callback_on_step_end: Optional[Callable],
    callback_on_step_end_tensor_inputs: List[str],
    cache_threshold_high: Optional[float] = None,
    cache_threshold_low: Optional[float] = None,
):
    """
    Cache-aware non-unified WAN denoise loop.

    This is intentionally module-scoped so cache runtime behavior lives next to the
    first-block-cache adapter patching logic and stays out of the core pipeline class.
    """
    transformer_perf = []
    low_stage_counter = 0

    with pipeline.model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline._interrupt:
                continue

            pipeline._current_timestep = t

            if boundary_timestep is None or t >= boundary_timestep:
                current_transformer_module = pipeline.transformer_high
                current_guidance_scale = guidance_scale
                stage_cache_threshold = 0.0 if cache_threshold_high is None else cache_threshold_high
            else:
                current_transformer_module = pipeline.transformer_low
                current_guidance_scale = guidance_scale_2
                low_stage_counter += 1
                low_threshold = 0.0 if cache_threshold_low is None else cache_threshold_low
                # Keep first few low-noise steps uncached to stabilize rollout.
                stage_cache_threshold = 0.0 if low_stage_counter < 3 else low_threshold

            current_model = current_transformer_module.model

            latent_model_input = latents.to(transformer_dtype)
            if pipeline.model.config.expand_timesteps:
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                timestep = t.expand(latents.shape[0])

            _, _, latent_frames, latent_height, latent_width = latents.shape
            p_t, p_h, p_w = current_model.config.patch_size
            post_patch_num_frames = latent_frames // p_t
            post_patch_height = latent_height // p_h
            post_patch_width = latent_width // p_w

            rotary_emb = current_model.rope(latent_model_input)
            rotary_emb = torch.cat(rotary_emb, dim=0)
            timestep = timestep.flatten()

            temb, timestep_proj, encoder_hidden_states, _ = current_model.condition_embedder(
                timestep, prompt_embeds, encoder_hidden_states_image=None, timestep_seq_len=None
            )

            if pipeline.do_classifier_free_guidance:
                _, _, encoder_hidden_states_neg, _ = current_model.condition_embedder(
                    timestep,
                    negative_prompt_embeds,
                    encoder_hidden_states_image=None,
                    timestep_seq_len=None,
                )

            timestep_proj = timestep_proj.unflatten(1, (6, -1))
            inputs_aic = {
                "hidden_states": latents.detach().numpy(),
                "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
                "rotary_emb": rotary_emb.detach().numpy(),
                "temb": temb.detach().numpy(),
                "timestep_proj": timestep_proj.detach().numpy(),
                "cache_threshold": np.array(stage_cache_threshold, dtype=np.float32),
            }

            if pipeline.do_classifier_free_guidance:
                inputs_aic2 = {
                    "hidden_states": latents.detach().numpy(),
                    "encoder_hidden_states": encoder_hidden_states_neg.detach().numpy(),
                    "rotary_emb": rotary_emb.detach().numpy(),
                    "temb": temb.detach().numpy(),
                    "timestep_proj": timestep_proj.detach().numpy(),
                    "cache_threshold": np.array(stage_cache_threshold, dtype=np.float32),
                }

            with current_model.cache_context("cond"):
                start_transformer_step_time = time.perf_counter()
                outputs = current_transformer_module.qpc_session.run(inputs_aic)
                end_transformer_step_time = time.perf_counter()
                transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                noise_pred = pipeline._reshape_noise_prediction(
                    outputs,
                    batch_size,
                    post_patch_num_frames,
                    post_patch_height,
                    post_patch_width,
                    p_t,
                    p_h,
                    p_w,
                )

            if pipeline.do_classifier_free_guidance:
                with current_model.cache_context("uncond"):
                    start_transformer_step_time = time.perf_counter()
                    outputs = current_transformer_module.qpc_session.run(inputs_aic2)
                    end_transformer_step_time = time.perf_counter()
                    transformer_perf.append(end_transformer_step_time - start_transformer_step_time)
                    noise_uncond = pipeline._reshape_noise_prediction(
                        outputs,
                        batch_size,
                        post_patch_num_frames,
                        post_patch_height,
                        post_patch_width,
                        p_t,
                        p_h,
                        p_w,
                    )
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(pipeline, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

    return latents, transformer_perf
