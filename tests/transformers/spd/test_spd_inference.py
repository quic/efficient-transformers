# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional

import numpy as np
import pytest
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.device_utils import get_available_device_id

configs = [
    pytest.param(
        ["My name is", "Hello", "Hi", "My name is"], # prompt
        2, # num_speculative_tokens
        32, # prefill_seq_len
        128, # ctx_len
        1, # prefill_bsz
        "JackFram/llama-68m", # draft_model_name
        "JackFram/llama-68m", # target_model_name
        4, # full_batch_size
        id="CB llama",
    ),
]



def run_prefill_on_draft_and_target(
    tlm_session: QAICInferenceSession, 
    dlm_session: QAICInferenceSession, 
    prompt: dict, 
    prefill_seq_len: int, 
    ctx_len: int, 
    prefill_batch_size: int, 
    decode_batch_size: int, 
    slot_idx: int
):
    tlm_decode_start_input = dict()
    dlm_decode_start_input = dict()
    inputs = prompt
    input_len = prompt.input_ids.shape[1]
    num_chunks = -(input_len // -prefill_seq_len)  # ceil divide without float
    input_len = num_chunks * prefill_seq_len  # Convert input_len to a multiple of prefill_seq_len
    assert input_len <= ctx_len, "input_len should be less than ctx_len"
    # pad the prompt tokens to match the input_len
    inputs = prompt
    # TODO need to store the attention mask and position ids for each batch element so that we can access them
    # at decode time
    inputs["attention_mask"] = np.concatenate(
        [inputs["attention_mask"].astype(bool) for j in range(decode_batch_size)], 0
    )
    inputs["position_ids"] = (np.cumsum(inputs["attention_mask"][0:1], 1) - 1) * inputs["attention_mask"][0:1]

    # FIXME "not" does not work for below line in place of the "== False" check, but code formatter recommends it
    inputs["position_ids"][inputs["attention_mask"][0:1] == False] = -1
    cache_index = np.array([[0]], np.int64)
    batch_index = np.array([[slot_idx]], np.int64)
    inputs["batch_index"] = batch_index

    # Run chunked prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prefill_seq_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prefill_seq_len]

        chunk_inputs.pop("attention_mask")
        tlm_outputs = tlm_session.run(chunk_inputs)
        dlm_outputs = dlm_session.run(chunk_inputs)
        cache_index += prefill_seq_len

    tlm_logits = tlm_outputs["logits"]
    dlm_logits = dlm_outputs["logits"]
    assert (tlm_logits == dlm_logits).sum().all()

    if len(tlm_logits.shape) == 2:
        tlm_logits = np.expand_dims(tlm_logits, 1)
    if len(dlm_logits.shape) == 2:
        dlm_logits = np.expand_dims(dlm_logits, 1)

    tlm_decode_start_pos_id = inputs["attention_mask"][0:1].sum(1, keepdims=True)
    tlm_decode_start_input_id = tlm_logits.argmax(2)
    dlm_decode_start_input_id = dlm_logits.argmax(2)
    dlm_decode_start_pos_id = inputs["attention_mask"][0:1].sum(1, keepdims=True)

    inputs.pop("attention_mask")

    tlm_decode_start_input = {
        "logits": tlm_logits,
        "input_ids": tlm_decode_start_input_id,
        "position_ids": tlm_decode_start_pos_id,
        "batch_index": batch_index,
        "input_len": tlm_decode_start_pos_id[0, 0],
    }
    dlm_decode_start_input = {
        "logits": dlm_logits,
        "input_ids": dlm_decode_start_input_id,
        "position_ids": dlm_decode_start_pos_id,
        "batch_index": batch_index,
        "input_len": tlm_decode_start_pos_id[0, 0],
    }

    return tlm_decode_start_input, dlm_decode_start_input


def get_padded_input_len(input_len: int, prefill_seq_len: int, ctx_len: int):
    """return padded input length (must be factor of `prefill_seq_len`)

    Args:
        input_len (int): prompt length
        prefill_seq_len (int): prefill sequence length 
        ctx_len (int): context length

    Returns:
        input_len_padded (int): padded input length
    """
    num_chunks = -(input_len // -prefill_seq_len)  # ceil divide without float
    input_len_padded = num_chunks * prefill_seq_len  # Convert input_len to a multiple of prefill_seq_len
    assert input_len_padded <= ctx_len, "input_len rounded to nearest prefill_seq_len multiple should be less than ctx_len"
    return input_len_padded


def populate_inputs(source, dest, index=None):
    for k, v in dest.items():
        if k == "batch_index":
            continue
        if index is None:
            # during decode
            dest[k] = source[k]
        else:
            # during prefill with bs=1
            dest[k][index] = source[k]

def split_dlm_bonus_token_inputs(dlm_decode_inputs):
    bonus_token_inputs = dict()
    bonus_token_inputs["input_ids"] = dlm_decode_inputs["input_ids"][:,0:1]
    bonus_token_inputs["position_ids"] = dlm_decode_inputs["input_ids"][:,0:1]
    dlm_decode_inputs["input_ids"] = dlm_decode_inputs["input_ids"][:,1:]
    dlm_decode_inputs["position_ids"] = dlm_decode_inputs["position_ids"][:,1:]
    return bonus_token_inputs, dlm_decode_inputs

@pytest.mark.parametrize(
    "prompt, num_speculative_tokens, prefill_seq_len, ctx_len, prefill_bsz, draft_model_name, target_model_name, full_batch_size",
    configs,
)
def test_spec_decode_inference(
    prompt: List[str],
    num_speculative_tokens: int,
    prefill_seq_len: int,
    ctx_len: int,
    prefill_bsz: int,
    draft_model_name: str,
    target_model_name: str,
    full_batch_size: Optional[int],
):
    # get device group
    device_group: List[int] = get_available_device_id()
    if not device_group:
        pytest.skip("No available devices to run model on Cloud AI 100")
    # assumes dlm and tlm are compiled to the same prompt-chunk-size, context length and full_batch_size/batch-size
    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    # export_and_compile tlm and dlm
    continuous_batching = full_batch_size is not None
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, continuous_batching=continuous_batching, is_tlm=True)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, continuous_batching=continuous_batching)

    num_devices = len(device_group)
    target_model_qpc_path: str = target_model.compile(num_cores=11,
                                                      num_devices=num_devices,
                                                      prefill_seq_len=prefill_seq_len,
                                                      ctx_len=ctx_len,
                                                      aic_enable_depth_first=True, 
                                                      full_batch_size=full_batch_size, 
                                                      num_speculative_tokens=num_speculative_tokens)
    draft_model_qpc_path: str = draft_model.compile(num_cores=5,
                                                    prefill_seq_len=prefill_seq_len,
                                                    ctx_len=ctx_len,
                                                    aic_enable_depth_first=True,
                                                    full_batch_size=full_batch_size)
    # init qaic session
    target_model_session = QAICInferenceSession(target_model_qpc_path, device_ids=device_group)
    draft_model_session = QAICInferenceSession(draft_model_qpc_path, device_ids=device_group)

    # skip inputs/outputs buffers
    target_model_session.skip_buffers(set([x for x in target_model_session.input_names if x.startswith("past_")]))
    target_model_session.skip_buffers(
        set([x for x in target_model_session.output_names if x.endswith("_RetainedState")])
    )
    draft_model_session.skip_buffers(set([x for x in draft_model_session.input_names if x.startswith("past_")]))
    draft_model_session.skip_buffers(set([x for x in draft_model_session.output_names if x.endswith("_RetainedState")]))

    is_cb = full_batch_size is not None
    if not is_cb:
        prompts = prompt * prefill_bsz
        decode_batch_size = prefill_bsz
    else:
        prompts = prompt
        decode_batch_size = full_batch_size
    # tokenize the prompts
    prompts_tokenized: List[dict] = []
    for p in prompts:
        input_len: int = tokenizer(p, return_tensors="np", padding=True).input_ids.shape[1]
        input_len_padded: int = get_padded_input_len(input_len, prefill_seq_len, ctx_len)
        p_tok: dict = tokenizer(p, return_tensors="np", padding="max_length", max_length=input_len_padded)
        prompts_tokenized.append(p_tok)
    # create caches to hold generated ids and input prompt lengths
    generated_ids = [[] for i in range(decode_batch_size)]
    input_lengths = [0] * decode_batch_size
    # run prefill on both draft and target models
    dlm_decode_inputs = dict()
    dlm_decode_inputs["position_ids"] = np.zeros((decode_batch_size, 1), np.int64)
    dlm_decode_inputs["input_ids"] = np.full((decode_batch_size, 1), tokenizer.pad_token_id)
    dlm_decode_inputs["batch_index"] = np.reshape(
        np.array(np.arange(decode_batch_size), np.int64), (decode_batch_size, 1)
    )
    # mock input key "logits" to store the first batch of output logits
    dlm_decode_inputs["logits"] = np.full((decode_batch_size, 1, vocab_size), 0)
    tlm_precode_inputs = dict(dlm_decode_inputs)
    is_prefill = True
    generation_done = False
    max_gen_len = [ctx_len] * decode_batch_size
    num_logits_to_keep = num_speculative_tokens+1
    all_accept = np.full((decode_batch_size, num_speculative_tokens), False, dtype=bool)
    tlm_prefill_logits_ph = np.zeros((prefill_bsz, 1, vocab_size), dtype=np.float32)
    dlm_prefill_logits_ph = np.zeros((prefill_bsz, 1, vocab_size), dtype=np.float32)
    decode_logits_ph = np.zeros((decode_batch_size, 1, vocab_size), dtype=np.float32)
    precode_logits_ph = np.zeros((decode_batch_size, num_logits_to_keep, vocab_size), dtype=np.float32)

    target_model_session.set_buffers({"logits": tlm_prefill_logits_ph})
    draft_model_session.set_buffers({"logits": dlm_prefill_logits_ph})
    for bi in range(decode_batch_size):
        # assumes that prefill queue will always be popped from the front
        tlm_prefill_output, dlm_prefill_output = run_prefill_on_draft_and_target(
            tlm_session=target_model_session,
            dlm_session=draft_model_session,
            prompt=prompts_tokenized[bi],
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            prefill_batch_size=prefill_bsz,
            decode_batch_size=decode_batch_size,
            slot_idx=bi,
        )
        # this way, we will directly get the updated full batch input dict to run decode
        populate_inputs(dlm_prefill_output, dlm_decode_inputs, bi)
        populate_inputs(tlm_prefill_output, tlm_precode_inputs, bi)
        # assumes that prefill queue will always be popped from the front
        input_lengths[bi] = tlm_prefill_output["input_len"]
        max_gen_len[bi] -= input_lengths[bi]

    target_model_session.set_buffers({"logits": precode_logits_ph})
    draft_model_session.set_buffers({"logits": decode_logits_ph})
    num_tokens_selected_per_validation = []
    dlm_run_bonus_token = False
    while not generation_done:
        # compute the processed context length before each iteration to prepare the position id inputs
        processed_context = [len(generated_ids[j]) + input_lengths[j] for j in range(decode_batch_size)]
        # generate proposals from draft model
        if is_prefill:
            draft_logits = [dlm_decode_inputs.pop("logits")]
            target_logits = [tlm_precode_inputs.pop("logits")]
        else:
            if np.any(all_accept):
                input_ids = []
                position_ids = []
                dlm_run_bonus_token = True
                for bi in range(decode_batch_size):
                    if all_accept[bi]:
                        # both last DLM token and bonus TLM token to be passed as input to DLM
                        input_ids.append([generated_ids[bi][-2], generated_ids[bi][-1]])
                        position_ids.append([processed_context[bi] - 2, processed_context[bi] - 1])
                    else:
                        # only the correct token from TLM from previous iteration and the pad_token as a dummy
                        input_ids.append([generated_ids[bi][-1], tokenizer.pad_token_id])
                        position_ids.append([processed_context[bi] - 1, -1])
                dlm_decode_inputs["input_ids"] = np.array(input_ids)
                dlm_decode_inputs["position_ids"] = np.array(position_ids)
            else:
                dlm_decode_inputs["input_ids"] = np.array([gid[-1] for gid in generated_ids], dtype=np.int64).reshape(
                    (decode_batch_size, 1)
                )
                dlm_decode_inputs["position_ids"] = np.array(
                    [(pc - 1) for pc in processed_context], dtype=np.int64
                ).reshape((decode_batch_size, 1))
            # prepare the inputs for the dlm speculation
            # TODO in case of even one of the batch having all_accept, we have to use the seqlen=2 specialization
            # hence need to have dummy -1 position id for other sequences.
            # dlm_decode_inputs["position_ids"] = len(generated_ids per batch)
            # dlm_decode_inputs["input_ids"] = (last gen dlm token) + last true token from TLM
        for k_ in range(num_speculative_tokens):
            if dlm_run_bonus_token:
                #running decode one extra time in the first speculative iteration
                # workaround to avoid the incorrect precode with 3-specialized multi-batch DLM
                bonus_token_inputs, dlm_decode_inputs = split_dlm_bonus_token_inputs(dlm_decode_inputs)
                dlm_outputs = draft_model_session.run(bonus_token_inputs)
                dlm_run_bonus_token = False
            dlm_outputs = draft_model_session.run(dlm_decode_inputs)
            draft_logits.append(dlm_outputs["logits"])
            dlm_decode_inputs["input_ids"] = dlm_outputs["logits"].argmax(-1)
            dlm_decode_inputs["position_ids"] = dlm_decode_inputs["position_ids"][:, -1:] + 1

        draft_logits = np.array(draft_logits).squeeze(2).transpose((1, 0, 2))
        # greedy sampling from draft model
        draft_tokens = draft_logits.argmax(-1)

        # construct precode inputs
        tlm_precode_inputs["input_ids"] = draft_tokens
        if not is_prefill:
            last_genid = np.array([gid[-1] for gid in generated_ids], dtype=np.int64).reshape(decode_batch_size, 1)
            tlm_precode_inputs["input_ids"] = np.concatenate((last_genid, tlm_precode_inputs["input_ids"]), axis=1)
            # in case of general precode, first token in input sequence is = last generated TLM token (kv cache backfill)
            tlm_precode_inputs["position_ids"] = np.array(
                [np.arange(start=pc - 1, stop=pc + num_speculative_tokens) for pc in processed_context], dtype=np.int64
            )
        else:
            # in case of just first precode, we are feeding in all new positions
            tlm_precode_inputs["position_ids"] = np.array(
                [np.arange(start=pc, stop=pc + num_speculative_tokens + 1) for pc in processed_context], dtype=np.int64
            )

        # run precode on TLM to score the proposed tokens
        tlm_outputs = target_model_session.run(tlm_precode_inputs)
        target_precode_logits = tlm_outputs["logits"]
        if is_prefill:
            target_logits = np.concatenate((target_logits[0], target_precode_logits), axis=1)
            # stack the prefill output logit and precode logits into a single tensor
        else:
            target_logits = target_precode_logits
        # greedy sampling from target model
        target_tokens = target_logits.argmax(-1)
        # exact matching between draft and target tokens
        matching = draft_tokens == target_tokens[:, :-1]
        num_tokens_selected = np.argmin(matching, axis=1)
        all_accept = matching[np.arange(decode_batch_size), num_tokens_selected]
        num_tokens_selected = np.where(all_accept, matching.shape[1], num_tokens_selected)
        num_tokens_selected_per_validation.append(num_tokens_selected)

        # append selected tokens to the generated_ids
        for bi in range(decode_batch_size):
            if len(generated_ids[bi]) >= max_gen_len[bi]:
                continue
            num_tokens_to_append = min(num_tokens_selected[bi], max_gen_len[bi] - len(generated_ids[bi]))
            generated_ids[bi] += list(draft_tokens[bi, :num_tokens_to_append])
        # append bonus/corrected token where applicable
        for bi in range(decode_batch_size):
            if len(generated_ids[bi]) >= max_gen_len[bi]:
                continue
            if all_accept[bi]:
                # bonus token
                generated_ids[bi].append(target_tokens[bi, -1])
            else:
                # correct token
                generated_ids[bi].append(target_tokens[bi, num_tokens_selected[bi]])
        generation_done = True
        for bi in range(decode_batch_size):
            if len(generated_ids[bi]) < max_gen_len[bi]:
                generation_done = False
        is_prefill = False
        draft_logits = []
        target_logits = []
    num_tokens_selected_per_validation = np.concatenate(num_tokens_selected_per_validation).reshape(len(num_tokens_selected_per_validation), decode_batch_size)
    mean_num_accepted_tokens_per_batch = num_tokens_selected_per_validation.mean(axis=0)
    print("mean number of accepted tokens per batch = ", mean_num_accepted_tokens_per_batch)
    print("max generation len = ", max_gen_len)
    print("actual generation len = ", [len(gid) for gid in generated_ids])
    print(tokenizer.batch_decode(generated_ids))
