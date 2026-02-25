# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from time import perf_counter
from typing import List, Optional

import numpy as np
import pytest
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import Constants
from QEfficient.utils.device_utils import get_available_device_id

configs = [
    pytest.param(
        Constants.INPUT_STR,  # prompts
        4,  # num_speculative_tokens
        32,  # prefill_seq_len
        128,  # ctx_len
        1,  # prefill_bsz
        "JackFram/llama-160m",  # draft_model_name
        "JackFram/llama-160m",  # target_model_name
        1,  # full_batch_size
        id="CB llama",
    ),
    pytest.param(
        Constants.INPUT_STR,  # prompts
        4,  # num_speculative_tokens
        32,  # prefill_seq_len
        128,  # ctx_len
        1,  # prefill_bsz
        "Qwen/Qwen2-0.5B",  # draft_model_name
        "Qwen/Qwen2-0.5B",  # target_model_name
        1,  # full_batch_size
        id="CB qwen",
    ),
]


def run_prefill_on_draft_and_target(
    tlm_session: QAICInferenceSession,
    dlm_session: QAICInferenceSession,
    inputs: dict,
    prefill_seq_len: int,
    slot_idx: int,
):
    input_len = inputs.input_ids.shape[1]
    num_chunks = input_len // prefill_seq_len
    cache_index = np.array([[0]], np.int64)
    batch_index = np.array([[slot_idx]], np.int64)
    inputs["batch_index"] = batch_index

    # Run chunked prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prefill_seq_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][
            :, cache_index[0, 0] : cache_index[0, 0] + prefill_seq_len
        ]

        tlm_outputs = tlm_session.run(chunk_inputs)
        _ = dlm_session.run(chunk_inputs)
        cache_index += prefill_seq_len

    tlm_logits = tlm_outputs["logits"]
    return tlm_logits


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
    assert input_len_padded <= ctx_len, (
        "input_len rounded to nearest prefill_seq_len multiple should be less than ctx_len"
    )
    return input_len_padded


def split_dlm_bonus_token_inputs(dlm_decode_inputs):
    bonus_token_inputs = dict()
    bonus, regular = np.hsplit(dlm_decode_inputs["input_ids"], 2)
    bonus_token_inputs["input_ids"] = bonus
    dlm_decode_inputs["input_ids"] = regular
    bonus, regular = np.hsplit(dlm_decode_inputs["position_ids"], 2)
    bonus_token_inputs["position_ids"] = bonus
    dlm_decode_inputs["position_ids"] = regular
    bonus_token_inputs["batch_index"] = dlm_decode_inputs["batch_index"]
    return bonus_token_inputs, dlm_decode_inputs


@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize(
    "prompts, num_speculative_tokens, prefill_seq_len, ctx_len, prefill_bsz, draft_model_name, target_model_name, full_batch_size",
    configs,
)
def test_spec_decode_inference(
    prompts: List[str],
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
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    if target_model_name == "Qwen/Qwen2-0.5B":
        vocab_size = 151936

    # export_and_compile tlm and dlm
    continuous_batching = full_batch_size is not None
    qaic_config = dict(speculative_model_type="target")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name, continuous_batching=continuous_batching, qaic_config=qaic_config
    )
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, continuous_batching=continuous_batching)

    target_model_qpc_path: str = target_model.compile(
        num_cores=6,
        num_devices=1,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        aic_enable_depth_first=True,
        full_batch_size=full_batch_size,
        num_speculative_tokens=num_speculative_tokens,
    )
    draft_model_qpc_path: str = draft_model.compile(
        num_cores=2,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        aic_enable_depth_first=True,
        full_batch_size=full_batch_size,
    )
    # init qaic session
    target_model_session = QAICInferenceSession(target_model_qpc_path)
    draft_model_session = QAICInferenceSession(draft_model_qpc_path)

    # skip inputs/outputs buffers
    target_model_session.skip_buffers(set([x for x in target_model_session.input_names if x.startswith("past_")]))
    target_model_session.skip_buffers(
        set([x for x in target_model_session.output_names if x.endswith("_RetainedState")])
    )
    draft_model_session.skip_buffers(set([x for x in draft_model_session.input_names if x.startswith("past_")]))
    draft_model_session.skip_buffers(set([x for x in draft_model_session.output_names if x.endswith("_RetainedState")]))

    is_cb = full_batch_size is not None
    decode_batch_size = full_batch_size if is_cb else prefill_bsz
    if len(prompts) < decode_batch_size:
        prompts_exp = prompts * decode_batch_size
        prompts = prompts_exp[:decode_batch_size]
    # tokenize the prompts
    prompts_tokenized: List[dict] = []
    for p in prompts:
        input_len: int = tokenizer(p, return_tensors="np", padding=True).input_ids.shape[1]
        input_len_padded: int = get_padded_input_len(input_len, prefill_seq_len, ctx_len)
        p_tok: dict = tokenizer(p, return_tensors="np", padding="max_length", max_length=input_len_padded)
        position_ids = np.where(p_tok.pop("attention_mask"), np.arange(input_len_padded), -1)
        p_tok["position_ids"] = position_ids
        p_tok["num_logits_to_keep"] = np.zeros((1, 1), dtype=np.int64)
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
    num_logits_to_keep = num_speculative_tokens + 1
    tlm_precode_inputs = dict(
        input_ids=np.zeros((decode_batch_size, num_speculative_tokens + 1), dtype=np.int64),
        position_ids=np.zeros((decode_batch_size, num_speculative_tokens + 1), dtype=np.int64),
        batch_index=np.arange(decode_batch_size, dtype=np.int64).reshape(-1, 1),
        num_logits_to_keep=np.zeros((num_logits_to_keep, 1), dtype=np.int64),
    )
    max_gen_len = [ctx_len] * decode_batch_size
    # setup buffers
    tlm_prefill_logits_ph = np.zeros((prefill_bsz, 1, vocab_size), dtype=np.float32)
    dlm_prefill_logits_ph = np.zeros((prefill_bsz, 1, vocab_size), dtype=np.float32)
    decode_logits_ph = np.zeros((decode_batch_size, 1, vocab_size), dtype=np.float32)
    precode_logits_ph = np.zeros((decode_batch_size, num_logits_to_keep, vocab_size), dtype=np.float32)

    target_model_session.set_buffers({"logits": tlm_prefill_logits_ph})
    draft_model_session.set_buffers({"logits": dlm_prefill_logits_ph})
    e2e_start = perf_counter()
    ttfts = []
    for bi in range(decode_batch_size):
        # assumes that prefill queue will always be popped from the front
        start = perf_counter()
        tlm_logits = run_prefill_on_draft_and_target(
            tlm_session=target_model_session,
            dlm_session=draft_model_session,
            inputs=prompts_tokenized[bi],
            prefill_seq_len=prefill_seq_len,
            slot_idx=bi,
        )
        ttft = perf_counter() - start
        ttfts.append(ttft)
        input_ids = tlm_logits.argmax(2).astype(np.int64)
        generated_ids[bi].append(input_ids.item())
        dlm_decode_inputs["input_ids"][bi, 0] = input_ids
        tlm_precode_inputs["input_ids"][bi, 0] = input_ids.item()
        input_len = prompts_tokenized[bi]["position_ids"].max(1).item() + 1
        dlm_decode_inputs["position_ids"][bi, 0] = input_len
        tlm_precode_inputs["position_ids"][bi] = np.arange(
            input_len, input_len + num_speculative_tokens + 1, dtype=np.int64
        )
        # assumes that prefill queue will always be popped from the front
        input_lengths[bi] = input_len
        max_gen_len[bi] -= input_lengths[bi]
    batch_ttft = perf_counter() - e2e_start

    # set decode logits buffers
    target_model_session.set_buffers({"logits": precode_logits_ph})
    draft_model_session.set_buffers({"logits": decode_logits_ph})
    # start decode phase
    valid_batch_indices = np.full(decode_batch_size, True, dtype=bool)
    all_accept = False
    it = 0
    decode_start = perf_counter()
    mean_num_accepted_tokens = 0
    all_accept = np.full(decode_batch_size, False, dtype=bool)
    while True:
        it += 1
        # generate proposals from draft model
        for k_ in range(num_speculative_tokens):
            if all_accept.any():
                # running decode one extra time in the first speculative iteration
                # workaround to avoid the incorrect precode with 3-specialized multi-batch DLM
                bonus_token_inputs, dlm_decode_inputs = split_dlm_bonus_token_inputs(dlm_decode_inputs)
                _ = draft_model_session.run(bonus_token_inputs)
                all_accept[:] = False
            dlm_outputs = draft_model_session.run(dlm_decode_inputs)
            input_ids = dlm_outputs["logits"].argmax(2)
            tlm_precode_inputs["input_ids"][:, k_ + 1] = input_ids.flatten()
            dlm_decode_inputs["input_ids"] = input_ids
            dlm_decode_inputs["position_ids"][valid_batch_indices] += 1
        # run precode on TLM to score the proposed tokens
        tlm_outputs = target_model_session.run(tlm_precode_inputs)
        target_logits = tlm_outputs["logits"]
        # greedy sampling from target model
        target_tokens = target_logits.argmax(-1)
        # exact matching between draft and target tokens
        draft_tokens = tlm_precode_inputs["input_ids"][:, 1:]
        matching = draft_tokens == target_tokens[:, :-1]  # shape: [decode_batch_size, num_speculative_tokens]
        num_tokens_selected = matching.cumprod(axis=1).sum(axis=1) + 1  # shape: [decode_batch_size]
        all_accept[valid_batch_indices] = num_tokens_selected[valid_batch_indices] == num_speculative_tokens + 1
        mean_num_accepted_tokens += num_tokens_selected[valid_batch_indices].mean().item()
        # append selected tokens to the generated_ids
        for bi, valid in enumerate(valid_batch_indices):
            if not valid:
                continue
            accepted_tokens = num_tokens_selected[bi]
            num_tokens_to_append = min(accepted_tokens, max_gen_len[bi] - len(generated_ids[bi]))
            generated_ids[bi].extend(target_tokens[bi, :num_tokens_to_append].tolist())
            if len(generated_ids[bi]) + num_logits_to_keep >= max_gen_len[bi]:
                valid_batch_indices[bi] = False
        # check if all generations are done
        if not valid_batch_indices.any():
            break
        # prepare decode inputs for next decode iteration
        num_valid_batch_indices = valid_batch_indices.sum().item()
        common_input_ids = target_tokens[valid_batch_indices, num_tokens_selected[valid_batch_indices] - 1].reshape(
            num_valid_batch_indices, 1
        )
        common_position_ids = (
            tlm_precode_inputs["position_ids"][
                valid_batch_indices, num_tokens_selected[valid_batch_indices] - 1
            ].reshape(num_valid_batch_indices, 1)
            + 1
        )
        if all_accept.any():
            # all_accept input_ids
            input_ids = np.zeros((decode_batch_size, 2), dtype=np.int64)
            last_spec_token_id = target_tokens[valid_batch_indices, -2].reshape(-1, 1)
            input_ids[valid_batch_indices] = np.concatenate([last_spec_token_id, common_input_ids], axis=1)
            dlm_decode_inputs["input_ids"] = input_ids
            # all_accept position_ids
            position_ids = np.full((decode_batch_size, 2), -1, dtype=np.int64)
            last_spec_position_id = tlm_precode_inputs["position_ids"][valid_batch_indices, -1].reshape(-1, 1)
            position_ids[valid_batch_indices] = np.concatenate([last_spec_position_id, common_position_ids], axis=1)
            dlm_decode_inputs["position_ids"] = position_ids
        else:
            dlm_decode_inputs["input_ids"][valid_batch_indices] = common_input_ids
            dlm_decode_inputs["position_ids"][valid_batch_indices] = common_position_ids
        tlm_precode_inputs["input_ids"][valid_batch_indices, 0] = common_input_ids.flatten()
        tlm_precode_inputs["position_ids"][valid_batch_indices] += num_tokens_selected[valid_batch_indices].reshape(
            num_valid_batch_indices, 1
        )
    end = perf_counter()
    decode_end = end - decode_start
    e2e_end = end - e2e_start
    mean_ttft = sum(ttfts) / len(ttfts)
    generated_tokens_per_prompt = [len(gid) + 1 for gid in generated_ids]
    decode_throughput = sum(generated_tokens_per_prompt) / decode_end
    e2e_throughput = (sum(generated_tokens_per_prompt) + decode_batch_size) / e2e_end
    batch_decode = tokenizer.batch_decode(generated_ids)
    mean_num_accepted_tokens /= it
    print(f"Avg TLM+DLM TTFT = {mean_ttft}")
    print(f"Total TLM+DLM Batch TTFT = {batch_ttft}")
    print(f"Decode Throughput = {decode_throughput}")
    print(f"E2E Throughput = {e2e_throughput}")
    print("Avg number of accepted tokens = ", mean_num_accepted_tokens)
    print("Max generation len = ", max_gen_len)
    print("Total Generated Tokens per Prompt: = ", generated_tokens_per_prompt)
    for prompt, generation in zip(prompts, batch_decode):
        print(f"{prompt=} {generation=}")
    # validation check
    assert mean_num_accepted_tokens == float(num_speculative_tokens + 1), (
        f"mean number of accepted tokens is {mean_num_accepted_tokens} but should be {num_speculative_tokens + 1}"
    )
    del target_model_session
    del draft_model_session
    generated_ids = np.asarray(generated_ids[0]).flatten()
    gen_len = generated_ids.shape[0]
    exec_info = draft_model.generate(tokenizer, Constants.INPUT_STR)
    cloud_ai_100_tokens = exec_info.generated_ids[0][
        :gen_len
    ]  # Because we always run for single input and single batch size
    all_matching = np.array_equal(cloud_ai_100_tokens, generated_ids)
    assert all_matching, "Tokens don't match for SpD output and vanilla DLM output."
    assert os.path.isfile(os.path.join(os.path.dirname(target_model_qpc_path), "qconfig.json"))
    assert os.path.isfile(os.path.join(os.path.dirname(draft_model_qpc_path), "qconfig.json"))
