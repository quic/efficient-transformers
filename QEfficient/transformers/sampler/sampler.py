# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput

from QEfficient.customop import CtxScatterFuncCB3D
from QEfficient.utils.constants import Constants


@dataclass
class SamplerOutput(ModelOutput):
    """
    Dataclass for the output of the On Device Sampler.
    """

    probs: torch.FloatTensor = None
    next_tokens: torch.IntTensor = None
    vision_embeds: Optional[torch.FloatTensor] = None  # For VLMs
    image_idx: Optional[torch.IntTensor] = None  # for VLMs
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_repetition_penalty_buffer: Optional[torch.Tensor] = None
    past_presence_penalty_buffer: Optional[torch.Tensor] = None


def prefill_path(
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor,
    batch_index: torch.LongTensor,
    batch_index_reshaped: torch.LongTensor,
    past_repetition_penalty_buffer: torch.Tensor,
    past_presence_penalty_buffer: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize or update RetainedState buffers for prefill stage based on `input_ids`.
    """

    # Initialize retain states for first input chunk
    mul_value = torch.ones(past_repetition_penalty_buffer.shape[0], 1, dtype=torch.bool)
    zero_tensor = torch.zeros(batch_index.shape, dtype=torch.long)
    positions_mask = (position_ids[:, :1] != zero_tensor).view(-1, 1)
    mul_value = CtxScatterFuncCB3D.apply(mul_value, batch_index, zero_tensor, positions_mask)
    past_repetition_penalty_buffer *= mul_value

    # Mask out-of-bounds or invalid position_ids or input_ids
    input_ids = torch.where(position_ids == -1, torch.iinfo(torch.int32).max, input_ids)

    # Update retain states for chunked input
    past_repetition_penalty_buffer = CtxScatterFuncCB3D.apply(
        past_repetition_penalty_buffer,
        batch_index,
        input_ids,
        torch.ones(input_ids.shape, dtype=torch.bool),
    )

    mul_value = torch.zeros(past_presence_penalty_buffer.shape[0], 1, dtype=torch.bool)
    past_presence_penalty_buffer *= mul_value
    return past_repetition_penalty_buffer, past_presence_penalty_buffer


def decode_path(
    last_accepted_output_tokens: torch.LongTensor,
    position_ids: torch.LongTensor,
    batch_index: torch.LongTensor,
    batch_index_reshaped: torch.LongTensor,
    past_repetition_penalty_buffer: torch.Tensor,
    past_presence_penalty_buffer: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Update RetainedState buffers for decode stage based on `last_accepted_output_tokens`.
    """

    # Mask out-of-bounds or invalid position_ids or last_accepted_output_tokens
    last_accepted_output_tokens = torch.where(
        position_ids == -1, torch.iinfo(torch.int32).max, last_accepted_output_tokens
    )

    # Update retained states
    scatter_values = torch.ones(last_accepted_output_tokens.shape, dtype=torch.bool)
    past_repetition_penalty_buffer = CtxScatterFuncCB3D.apply(
        past_repetition_penalty_buffer,
        batch_index,
        last_accepted_output_tokens,
        scatter_values,
    )
    past_presence_penalty_buffer = CtxScatterFuncCB3D.apply(
        past_presence_penalty_buffer,
        batch_index,
        last_accepted_output_tokens,
        scatter_values,
    )
    # TODO: For frequency retain state, first gather and then scatter
    return past_repetition_penalty_buffer, past_presence_penalty_buffer


def sampler_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    comp_ctx_lengths: Optional[torch.LongTensor] = None,
    batch_index: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: Optional[int] = None,
    vision_embeds: Optional[torch.FloatTensor] = None,
    image_idx: Optional[torch.IntTensor] = None,
    last_accepted_output_tokens: Optional[torch.Tensor] = None,  # (batch_size, spec_length or less)
    past_repetition_penalty_buffer: Optional[torch.Tensor] = None,
    repetition_penalties: Optional[torch.Tensor] = None,
    past_presence_penalty_buffer: Optional[torch.Tensor] = None,
    presence_penalties: Optional[torch.Tensor] = None,
    temperatures: Optional[torch.Tensor] = None,
    top_ks: Optional[torch.Tensor] = None,
    top_ps: Optional[torch.Tensor] = None,
    min_ps: Optional[torch.Tensor] = None,
    random_numbers: Optional[torch.Tensor] = None,
) -> Union[Tuple, SamplerOutput]:
    r"""
    Perform the sampling of next tokens on the QAIC device (instead of the host)
    and return the next tokens and/or probability distributions.

    The vision_embeds and image_idx parameters are optional
    and are used only for VLMs when supported by the original forward function.

    Args:
        last_accepted_output_tokens (`torch.Tensor`, *optional*):
            Output tokens accepted by the Speculative Decoding Draft Language Model.

        past_repetition_penalty_buffer (`torch.Tensor`, *optional*):
            RetainedState buffer used as a mask to apply repetition penalty to the input
            prompt and the output generated so far.

        repetition_penalties (`torch.Tensor`, *optional*):
            Sampling parameter that penalizes new tokens based on whether they appear in the
            prompt and the generated text so far. Values > 1 encourage the model to use
            new tokens, while values < 1 encourage the model to repeat tokens.

        past_presence_penalty_buffer (`torch.Tensor`, *optional*):
            RetainedState buffer used as a mask to apply presence penalty to the output
            generated so far.

        presence_penalties (`torch.Tensor`, *optional*):
            Sampling parameter that penalizes new tokens based on whether they appear in the
            generated text so far. Values > 0 encourage the model to use new tokens, while
            values < 0 encourage the model to repeat tokens.

        temperatures (`torch.Tensor`, *optional*):
            Sampling parameter that controls the randomness of the sampling. Lower values
            make the model more deterministic, while higher values make the model more
            random. Zero means greedy sampling.

        top_ks (`torch.Tensor`, *optional*):
            Sampling parameter that controls the number of top tokens to consider.

        top_ps (`torch.Tensor`, *optional*):
            Sampling parameter that controls the cumulative probability of the top tokens to
            consider. Must be in (0, 1]. Set to 1.0 to consider all tokens.

        min_ps (`torch.Tensor`, *optional*):
            Sampling parameter that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token. Must be in
            [0, 1]. Set to 0.0 to disable this.

        random_numbers (`torch.Tensor`, *optional*):
            Sampling parameter that represents the random seeds to use for random sampling.
            Must be in [-1, 1].
    """
    if vision_embeds is not None:
        forward_kwargs = dict(
            input_ids=input_ids,
            vision_embeds=vision_embeds,
            position_ids=position_ids,
            image_idx=image_idx,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
        )
        if batch_index is not None:
            forward_kwargs["batch_index"] = batch_index

        logits, vision_embeds, image_idx, past_key_values = self.old_forward(**forward_kwargs)
        outputs = dict(logits=logits, vision_embeds=vision_embeds, image_idx=image_idx, past_key_values=past_key_values)
        if position_ids.dim() == 3:  # For models using m-rope
            position_ids = position_ids[0]
    else:
        outputs = self.old_forward(
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
            return_dict=return_dict,
            cache_position=cache_position,
        )

    logits = outputs.get("logits", None)
    assert logits is not None, f"{self.model.__class__.__name__} does not return logits."
    logits = logits.float()  # (batch_size, num_logits_to_keep aka spec_length, vocab_size)

    # Perform Sampling
    batch_size, spec_length, vocab_size = logits.shape
    logits = logits.reshape(-1, vocab_size)  # Reshape tensor to 2D

    if batch_index is None:  # Regular model execution
        batch_index = torch.arange(batch_size).view(-1, 1)

    batch_index_reshaped = batch_index.view(-1)
    # Prefill
    past_repetition_penalty_buffer_prefill, past_presence_penalty_buffer_prefill = prefill_path(
        input_ids=input_ids,
        position_ids=position_ids,
        batch_index=batch_index,
        batch_index_reshaped=batch_index_reshaped,
        past_repetition_penalty_buffer=past_repetition_penalty_buffer.clone(),
        past_presence_penalty_buffer=past_presence_penalty_buffer.clone(),
    )
    # Decode
    past_repetition_penalty_buffer_decode, past_presence_penalty_buffer_decode = decode_path(
        last_accepted_output_tokens=last_accepted_output_tokens,
        position_ids=position_ids,
        batch_index=batch_index,
        batch_index_reshaped=batch_index_reshaped,
        past_repetition_penalty_buffer=past_repetition_penalty_buffer.clone(),
        past_presence_penalty_buffer=past_presence_penalty_buffer.clone(),
    )
    # Select the correct repetition and presence penalty buffers
    is_prefill = torch.ones(past_repetition_penalty_buffer.shape, dtype=torch.bool) * (input_ids.shape[1] > spec_length)
    past_repetition_penalty_buffer = torch.where(
        is_prefill, past_repetition_penalty_buffer_prefill, past_repetition_penalty_buffer_decode
    )
    past_presence_penalty_buffer = torch.where(
        is_prefill, past_presence_penalty_buffer_prefill, past_presence_penalty_buffer_decode
    )

    # Repetition Penalty
    if (repetition_penalties != 1.0).any():
        past_repetition_penalty_buffer_selected = past_repetition_penalty_buffer[batch_index_reshaped].repeat(
            spec_length, 1
        )  # (batch_size * spec_length, vocab_size)
        repetition_penalties_mask = torch.where(past_repetition_penalty_buffer_selected, repetition_penalties, 1.0)
        logits *= repetition_penalties_mask ** (-torch.sign(logits))

    # Presence Penalty
    if (presence_penalties != 0.0).any():
        presence_penalties = presence_penalties.repeat(
            spec_length, 1
        )  # (batch_size, 1) -> (batch_size * spec_length, 1)
        past_presence_penalty_buffer_selected = past_presence_penalty_buffer[batch_index_reshaped].repeat(
            spec_length, 1
        )  # (batch_size * spec_length, vocab_size)
        logits -= presence_penalties * past_presence_penalty_buffer_selected

    # Greedy Sampling
    greedy_samples = torch.argmax(logits, dim=1, keepdim=True)  # (batch_size * spec_length, 1)
    if (temperatures == 0).all() and not self.qaic_config.get("return_pdfs", False):
        return SamplerOutput(
            probs=None,
            next_tokens=greedy_samples.reshape(-1, spec_length, 1),  # Return sampled next tokens instead of logits
            vision_embeds=outputs.get("vision_embeds", None),
            image_idx=outputs.get("image_idx", None),
            past_key_values=outputs.get("past_key_values", None),
            past_repetition_penalty_buffer=past_repetition_penalty_buffer,
            past_presence_penalty_buffer=past_presence_penalty_buffer,
        )

    # TODO: Frequency Penalty

    # Temperature Scaling
    temperatures = temperatures.repeat(spec_length, 1)  # (batch_size, 1) -> (batch_size * spec_length, 1)
    logits = torch.where(temperatures != 0, logits / temperatures, logits)

    # Top K
    # TODO (Optimization): if (top_ks != -1 or top_ks != max_top_k_ids).any() is False: skip but will need topk_values_asc and topk_indices_asc
    max_top_k_ids = self.qaic_config.get("max_top_k_ids", Constants.MAX_TOP_K_IDS)
    topk_values, topk_indices = torch.topk(logits, k=max_top_k_ids, dim=1)  # (batch_size * spec_length, vocab_size)
    topk_values_asc = topk_values.flip(dims=[1])
    topk_indices_asc = topk_indices.flip(dims=[1])
    top_ks[top_ks > max_top_k_ids] = max_top_k_ids  # Clip k to max value
    # True values in this mask indicate the positions of the non-top K values
    topk_mask = torch.arange(topk_values_asc.shape[1]).unsqueeze(0) < (
        topk_values_asc.size(1) - top_ks.to(torch.long)
    ).repeat(spec_length, 1)  # (batch_size * spec_length, max_top_k_ids)
    topk_values_asc[topk_mask] = torch.finfo(torch.float16).min

    # Top P
    # TODO (Optimization): if (top_ps != 1.).any() is False: skip but will need top_probs for Min P
    top_probs = torch.softmax(topk_values_asc, dim=1)  # (batch_size * spec_length, max_top_k_ids)
    topk_probs_sum = torch.cumsum(top_probs, dim=1)
    top_p_mask = topk_probs_sum <= 1 - top_ps.repeat(spec_length, 1)  # (batch_size * spec_length, max_top_k_ids)
    top_p_mask[:, max_top_k_ids - 1] = False
    topk_values_asc[top_p_mask] = torch.finfo(torch.float16).min

    # Min P
    if (min_ps != 0.0).any():
        scaled_min_p = torch.mul(
            min_ps.repeat(spec_length, 1),
            top_probs[:, max_top_k_ids - 1 :],
        )  # (batch_size * spec_length, 1)
        min_p_mask = top_probs < scaled_min_p  # (batch_size * spec_length, max_top_k_ids)
        topk_values_asc[min_p_mask] = torch.finfo(torch.float16).min

    probs = None
    if self.qaic_config.get("return_pdfs", False):
        # Update the logits
        logits.fill_(torch.finfo(torch.float16).min)
        logits = logits.scatter(1, topk_indices_asc, topk_values_asc)  # (batch_size * spec_length, vocab_size)
        # Softmax
        probs = torch.softmax(logits, dim=1).reshape(
            -1, spec_length, vocab_size
        )  # (batch_size, spec_length, vocab_size)

    # Random Sampling
    gumbel_noise = -torch.log(-torch.log(random_numbers.repeat(spec_length, 1)))  # Gumbel-Max Trick
    y = topk_values_asc + gumbel_noise  # (batch_size * spec_length, max_top_k_ids)
    random_samples_indices = torch.argmax(y, dim=1, keepdim=True)
    random_samples = torch.gather(topk_indices_asc, 1, random_samples_indices)  # (batch_size * spec_length, 1)

    # Sample the next tokens
    next_tokens = torch.where(temperatures == 0, greedy_samples, random_samples).reshape(
        -1, spec_length, 1
    )  # (batch_size, spec_length, 1)

    return SamplerOutput(
        probs=probs,
        next_tokens=next_tokens,  # Return sampled next tokens instead of logits
        vision_embeds=outputs.get("vision_embeds", None),
        image_idx=outputs.get("image_idx", None),
        past_key_values=outputs.get("past_key_values", None),
        past_repetition_penalty_buffer=past_repetition_penalty_buffer,
        past_presence_penalty_buffer=past_presence_penalty_buffer,
    )
