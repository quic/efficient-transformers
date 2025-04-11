from dataclasses import dataclass
import torch
import torch.nn.functional as F

# from QEfficient.customop import CtxScatterFunc
from QEfficient.utils.constants import Constants
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from typing import List, Optional, Tuple, Union


@dataclass
class QEffCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor | torch.IntTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    repetition_penalty_retain_state: Optional[torch.Tensor] = None
    presence_penalty_retain_state: Optional[torch.Tensor] = None


def filter_hidden_states(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    num_logits_to_keep: Optional[int] = None,
) -> torch.Tensor:
    """
    Filter hidden states based on whether this is a TLM SpD model

    ``Mandatory`` Args:
        :hidden_states (torch.Tensor): Hidden states tensor.
        :position_ids (torch.Tensor): Position ids tensor.
    ``Optional`` Args:
        :num_logits_to_keep (int, optional): Number of speculative tokens, specified only for TLM SpD model

    Returns:
        :torch.Tensor: Filtered hidden states.
    """
    batch_size = position_ids.size(0)
    batch_indices = torch.arange(batch_size)
    # Cast to INT32 to avoid issue while running in ONNXRT
    logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)

    if num_logits_to_keep is None:
        # return the last logit
        return hidden_states[batch_indices.view(-1, 1), logit_index]

    # gather approach
    num_logits_to_keep = num_logits_to_keep.shape[0]
    lower_idx = torch.where(
        logit_index < num_logits_to_keep,
        0,
        logit_index + 1 - num_logits_to_keep,
    ).view(
        -1, 1
    )  # shape: [bsz, 1]
    spec_idx = torch.arange(num_logits_to_keep).view(1, -1)  # shape: [1, k]
    indices = torch.add(lower_idx, spec_idx).unsqueeze(2)  # shape: [bsz, k, 1]
    indices = indices.repeat(
        1, 1, hidden_states.size(-1)
    )  # shape: [bsz, ,k, d_model]
    hidden_states = torch.gather(
        hidden_states, dim=1, index=indices
    )  # shape: [bsz, k, d_model]
    return hidden_states


def sampler_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    batch_index: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: Optional[int] = None,
    last_accepted_output_tokens: Optional[torch.Tensor] = None,  # (batch_size, spec_length or less)
    repetition_penalty_retain_state: Optional[torch.Tensor] = None,
    repetition_penalties: Optional[torch.Tensor] = None,
    presence_penalty_retain_state: Optional[torch.Tensor] = None,
    presence_penalties: Optional[torch.Tensor] = None,
    temperatures: Optional[torch.Tensor] = None,
    top_ks: Optional[torch.Tensor] = None,
    top_ps: Optional[torch.Tensor] = None,
    min_ps: Optional[torch.Tensor] = None,
    random_numbers: Optional[torch.Tensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        last_accepted_output_tokens (`torch.Tensor`, *optional*):
            Output tokens accepted by the Speculative Decoding Draft Language Model.

        repetition_penalty_retain_state (`torch.Tensor`, *optional*):
            RetainedState buffer used as a mask to apply repetition penalty to the input
            prompt and the output generated so far.

        repetition_penalties (`torch.Tensor`, *optional*):
            Sampling parameter that penalizes new tokens based on whether they appear in the
            prompt and the generated text so far. Values > 1 encourage the model to use
            new tokens, while values < 1 encourage the model to repeat tokens.

        presence_penalty_retain_state (`torch.Tensor`, *optional*):
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

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        batch_index=batch_index,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = filter_hidden_states(
        outputs[0], position_ids, num_logits_to_keep
    )
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(
            self.vocab_size // self.config.pretraining_tp, dim=0
        )
        logits = [
            F.linear(hidden_states, lm_head_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()  # (batch_size, num_logits_to_keep aka spec_length, vocab_size)

    # Perform Sampling
    batch_size, spec_length, vocab_size = logits.shape

    # Select relevant rows
    batch_index_reshaped = batch_index.view(-1)
    repetition_penalty_retain_state_selected = torch.index_select(repetition_penalty_retain_state, 0, batch_index_reshaped)
    presence_penalty_retain_state_selected = torch.index_select(presence_penalty_retain_state, 0, batch_index_reshaped)

    logits = logits.reshape(-1, vocab_size)  # Reshape tensor to 2D

    if input_ids.shape[1] > spec_length:  # Prefill phase, initialize retained states
        # TODO: Replace scatter_ with CtxScatterFunc; Replace -1 with int_max while exporting on onnx
        # repetition_penalty_retain_state_selected = CtxScatterFunc.apply(repetition_penalty_retain_state_selected.unsqueeze(1), input_ids, 1).squeeze(1)
        repetition_penalty_retain_state_selected.scatter_(1, input_ids, 1)
        presence_penalty_retain_state_selected.scatter_(1, input_ids, 0)
    else:  # Decode phase, update retained states
        repetition_penalty_retain_state_selected.scatter_(1, last_accepted_output_tokens, 1)
        presence_penalty_retain_state_selected.scatter_(1, last_accepted_output_tokens, 1)
        # TODO: For frequency retain state, first gather and then scatter

    # Update relevant rows in original tensors
    repetition_penalty_retain_state[batch_index_reshaped] = repetition_penalty_retain_state_selected
    presence_penalty_retain_state[batch_index_reshaped] = presence_penalty_retain_state_selected

    # Repetition Penalty
    if (repetition_penalties != 1.).any():
        repetition_penalties = repetition_penalties.repeat(spec_length, vocab_size)  # (batch_size, 1) -> (batch_size * spec_length, vocab_size)
        repetition_penalty_retain_state_selected = repetition_penalty_retain_state_selected.repeat(spec_length, 1)  # (batch_size, vocab_size) -> (batch_size * spec_length, vocab_size)
        repetition_penalties[~repetition_penalty_retain_state_selected.bool()] = 1.0
        logits = torch.where(logits > 0, logits / repetition_penalties, logits * repetition_penalties)

    # Presence Penalty
    if (presence_penalties != 0.).any():
        presence_penalties = presence_penalties.repeat(spec_length, 1)  # (batch_size, 1) -> (batch_size * spec_length, 1)
        presence_penalty_retain_state_selected = presence_penalty_retain_state_selected.repeat(spec_length, 1)  # (batch_size, vocab_size) -> (batch_size * spec_length, vocab_size)
        logits -= presence_penalties * presence_penalty_retain_state_selected

    # TODO: Frequency Penalty

    # Temperature Scaling
    if (temperatures != 0).any():
        temperatures = temperatures.repeat(spec_length, 1)  # (batch_size, 1) -> (batch_size * spec_length, 1)
        logits = torch.where(temperatures != 0, logits / temperatures, logits)

    # Top K
    # TODO (Optimization): if (top_ks != -1 or top_ks != Constants.MAX_TOP_K_IDS).any(): skip
    topk_values, topk_indices = torch.topk(logits, k=Constants.MAX_TOP_K_IDS, dim=1)  # (batch_size * spec_length, vocab_size)
    topk_values_asc = topk_values.flip(dims=[1])
    topk_indices_asc = topk_indices.flip(dims=[1])
    top_ks[top_ks > Constants.MAX_TOP_K_IDS] = Constants.MAX_TOP_K_IDS  # Clip k to max value
    # True values in this mask indicate the positions of the non-top K values
    topk_mask = torch.arange(topk_values_asc.shape[1]).unsqueeze(0) < (topk_values_asc.size(1) - top_ks.to(torch.long)).repeat(spec_length, 1)  # (batch_size * spec_length, Constants.MAX_TOP_K_IDS)
    topk_values_asc[topk_mask] = torch.finfo(torch.float16).min

    # Top P
    # TODO (Optimization): if (top_ps != 1.).any(): skip but will need top_probs for Min P
    top_probs = torch.softmax(topk_values_asc, dim=1)  # (batch_size * spec_length, Constants.MAX_TOP_K_IDS)
    topk_probs_sum = torch.cumsum(top_probs, dim=1)
    top_p_mask = topk_probs_sum <= 1 - top_ps.repeat(spec_length, 1)  # (batch_size * spec_length, Constants.MAX_TOP_K_IDS)
    top_p_mask[:, Constants.MAX_TOP_K_IDS - 1] = False
    topk_values_asc[top_p_mask] = torch.finfo(torch.float16).min

    # Min P
    # TODO (Optimization): if (min_ps != 0.).any(): skip
    scaled_min_p = torch.mul(min_ps.repeat(spec_length, 1), top_probs[:, -1:])  # (batch_size * spec_length, 1)
    min_p_mask = top_probs < scaled_min_p  # (batch_size * spec_length, Constants.MAX_TOP_K_IDS)
    topk_values_asc[min_p_mask] = torch.finfo(torch.float16).min

    logits.fill_(torch.finfo(torch.float16).min)
    logits = logits.scatter(1, topk_indices_asc, topk_values_asc)  # (batch_size * spec_length, vocab_size)

    # Softmax
    # TODO (Optimization): if (temperatures == 0).all(): skip and perform only greedy sampling
    probs = torch.softmax(logits, dim=1)  # (batch_size * spec_length, vocab_size)

    # Sample the next tokens
    # TODO (Optimization): if self.return_pds: skip
    greedy_samples = torch.argmax(probs, dim=-1, keepdim=True)  # Greedy Sampling
    gumbel_noise = -torch.log(-torch.log(random_numbers.repeat(spec_length, 1)))  # Gumbel-Max Trick
    y = probs + gumbel_noise
    random_samples = torch.argmax(y, dim=-1, keepdim=True)  # Random Sampling
    next_tokens = torch.where(temperatures == 0, greedy_samples, random_samples)  # (batch_size * spec_length, 1)

    # Reshape tensor back to 3D
    probs = probs.reshape(-1, spec_length, vocab_size)
    next_tokens = next_tokens.reshape(-1, spec_length, 1)

    return QEffCausalLMOutputWithPast(
        loss=None,
        logits=probs if self.return_pdfs else next_tokens,  # Return probabilities or sampled next tokens instead of logits
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        repetition_penalty_retain_state=repetition_penalty_retain_state,
        presence_penalty_retain_state=presence_penalty_retain_state,
    )
