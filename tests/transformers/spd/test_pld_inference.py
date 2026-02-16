# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Union

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
        "JackFram/llama-68m",  # target_model_name
        1,  # full_batch_size
        3,  # max_ngram_size
        id="CB llama",
    ),
]


@dataclass
class PerfMetrics:
    """
    Holds all performance metrics

    Args:
        :mean_ttft (float): Average TLM+DLM TTFT.
        :batch_ttft (float): Total TLM+DLM Batch TTFT.
        :decode_throughput (float): Decode throughput.
        :e2e_throughput (float): E2E throughput.
        :mean_num_accepted_tokens (float): Average number of accepted tokens.
        :max_gen_len (int): Max generation length.
        :generated_tokens_per_prompt (List[int]): Total generated tokens per prompt.
    """

    mean_ttft: float
    batch_ttft: float
    decode_throughput: float
    e2e_throughput: float
    mean_num_accepted_tokens: float
    max_gen_len: int
    generated_tokens_per_prompt: List[int]


@dataclass
class CloudAI100ExecInfo:
    """
    Holds all the information about Cloud AI 100 execution

    Args:
        :prompts (List[str]): Prompts to perfrom inferencing on.
        :batch_size (int): Batch size of the QPC compilation.
        :generated_texts (Union[List[List[str]], List[str]]): Generated text(s).
        :generated_ids (Union[List[np.ndarray], np.ndarray]): Generated IDs.
        :perf_metrics (PerfMetrics): Performance metrics.
        :num_speculative_tokens (int): Number of speculative tokens.
        :prefill_seq_len (int): Prefill sequence length.
        :ctx_len (int): Context length.
        :prefill_bsz (int): Prefill batch size.
        :draft_model_name (str): Draft model name.
        :target_model_name (str): Target model name.
        :full_batch_size (Optional[int]): Full batch size.
    """

    prompts: List[str]
    batch_size: int
    generated_texts: Union[List[str], List[List[str]]]
    generated_ids: Union[List[np.ndarray], np.ndarray]
    perf_metrics: PerfMetrics
    num_speculative_tokens: int
    prefill_seq_len: int
    ctx_len: int
    prefill_bsz: int
    draft_model_name: str
    target_model_name: str
    full_batch_size: Optional[int]

    def __repr__(self):
        return (
            f"Avg TLM+DLM TTFT = {round(self.perf_metrics.mean_ttft, 2)}\n"
            f"Total TLM+DLM Batch TTFT = {round(self.perf_metrics.batch_ttft, 2)}\n"
            f"Decode Throughput = {round(self.perf_metrics.decode_throughput, 2)}\n"
            f"E2E Throughput = {round(self.perf_metrics.e2e_throughput, 2)}\n"
            f"Avg number of accepted tokens = {round(self.perf_metrics.mean_num_accepted_tokens, 2)}\n"
            f"Max generation len = {self.perf_metrics.max_gen_len}\n"
            f"Total Generated Tokens per Prompt: = {self.perf_metrics.generated_tokens_per_prompt}"
        )


def run_prefill_on_draft_and_target(
    tlm_session: QAICInferenceSession,
    dlm_session: Optional[QAICInferenceSession],
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
        if dlm_session is not None:
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


def find_candidate_pred_tokens(
    input_ids: np.ndarray, fill_tok: int, max_ngram_size: int = 3, num_pred_tokens: int = 10
) -> np.ndarray:
    """find candidate predicted tokens
    code is a numpy-adaptation of the function `find_candidate_pred_tokens` in
    https://github.com/apoorvumang/prompt-lookup-decoding?tab=readme-ov-file

    Args:
        input_ids (np.ndarray): _description_, shape: [1, seq_len]
        fill_tok (int): _description_
        max_ngram_size (int, optional): _description_. Defaults to 3.
        num_pred_tokens (int, optional): _description_. Defaults to 10.

    Returns:
        np.ndarray: speculated tokenss, shape: [1, num_pred_tokens] if match is found
    """
    decode_batch_size, input_length = input_ids.shape
    assert decode_batch_size == 1

    # Ensure max_ngram_size and num_pred_tokens are valid
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size or num_pred_tokens")

    has_empty_tokens = False
    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:]

        # Create sliding windows of size ngram_size
        windows = np.lib.stride_tricks.sliding_window_view(input_ids[0], window_shape=ngram_size)

        # Find where the windows match the ngram
        matches = np.all(windows == ngram, axis=1)

        # Get the indices of matches
        match_indices = np.where(matches)[0]

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens

            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx], has_empty_tokens

    # If no match is found, return invalid array
    has_empty_tokens = True
    return np.full(num_pred_tokens, fill_tok, dtype=np.int64), has_empty_tokens


@pytest.mark.on_qaic
@pytest.mark.feature
@pytest.mark.parametrize(
    "prompts, num_speculative_tokens, prefill_seq_len, ctx_len, prefill_bsz, target_model_name, full_batch_size, max_ngram_size",
    configs,
)
def test_pld_spec_decode_inference(
    prompts: List[str],
    num_speculative_tokens: int,
    prefill_seq_len: int,
    ctx_len: int,
    prefill_bsz: int,
    target_model_name: str,
    full_batch_size: Optional[int],
    max_ngram_size: int,
) -> CloudAI100ExecInfo:
    """
    Perform draft speculative decode inference on the given prompts.

    Args:
        prompts (List[str]): List of prompts to perform inference on.
        num_speculative_tokens (int): Number of speculative tokens.
        prefill_seq_len (int): Prefill sequence length.
        ctx_len (int): Context length.
        prefill_bsz (int): Prefill batch size.
        target_model_name (str): Name of the target model.
        full_batch_size (Optional[int]): Full batch size.
        device_group (List[int]): List of device IDs.
        max_ngram_size (int): Max ngram size

    Returns:
        CloudAI100ExecInfo: Execution information, including performance metrics and generated text.
    """
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

    # export_and_compile tlm and dlm
    continuous_batching = full_batch_size is not None
    qaic_config = dict(speculative_model_type="target")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name, continuous_batching=continuous_batching, qaic_config=qaic_config
    )

    target_model_qpc_path: str = target_model.compile(
        num_cores=8,
        num_devices=1,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        aic_enable_depth_first=True,
        full_batch_size=full_batch_size,
        num_speculative_tokens=num_speculative_tokens,
    )
    # init qaic session
    target_model_session = QAICInferenceSession(target_model_qpc_path)
    draft_model_session = None

    # skip inputs/outputs buffers
    target_model_session.skip_buffers(set([x for x in target_model_session.input_names if x.startswith("past_")]))
    target_model_session.skip_buffers(
        set([x for x in target_model_session.output_names if x.endswith("_RetainedState")])
    )

    is_cb = full_batch_size is not None
    decode_batch_size = full_batch_size if is_cb else prefill_bsz
    if len(prompts) < decode_batch_size:
        prompts_exp = prompts * decode_batch_size
        prompts = prompts_exp[:decode_batch_size]
    # tokenize the prompts
    prefill_nltk = np.zeros((1, 1), dtype=np.int64)
    prompts_tokenized: List[dict] = []
    for p in prompts:
        input_len: int = tokenizer(p, return_tensors="np", padding=True).input_ids.shape[1]
        input_len_padded: int = get_padded_input_len(input_len, prefill_seq_len, ctx_len)
        p_tok: dict = tokenizer(p, return_tensors="np", padding="max_length", max_length=input_len_padded)
        position_ids = np.where(p_tok.pop("attention_mask"), np.arange(input_len_padded), -1)
        p_tok["position_ids"] = position_ids
        p_tok["num_logits_to_keep"] = prefill_nltk
        prompts_tokenized.append(p_tok)
    # create caches to hold generated ids and input prompt lengths
    generated_ids = [[] for i in range(decode_batch_size)]
    input_lengths = [0] * decode_batch_size
    # run prefill on both draft and target models
    # mock input key "logits" to store the first batch of output logits
    tlm_precode_inputs = dict(
        input_ids=np.zeros((decode_batch_size, num_speculative_tokens + 1), dtype=np.int64),
        position_ids=np.zeros((decode_batch_size, num_speculative_tokens + 1), dtype=np.int64),
        batch_index=np.arange(decode_batch_size, dtype=np.int64).reshape(-1, 1),
        num_logits_to_keep=np.zeros((num_speculative_tokens + 1, 1), dtype=np.int64),
    )
    num_logits_to_keep = num_speculative_tokens + 1
    max_gen_len = [ctx_len] * decode_batch_size
    # setup buffers
    tlm_prefill_logits_ph = np.zeros((prefill_bsz, 1, vocab_size), dtype=np.float32)
    precode_logits_ph = np.zeros((decode_batch_size, num_logits_to_keep, vocab_size), dtype=np.float32)

    target_model_session.set_buffers({"logits": tlm_prefill_logits_ph})
    e2e_start = perf_counter()
    ttfts = []
    all_ids = np.zeros((decode_batch_size, ctx_len), dtype=np.int64)
    prompt_plus_gen_idx = np.zeros(decode_batch_size, dtype=np.int64)
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
        tlm_precode_inputs["input_ids"][bi, 0] = input_ids.item()
        input_len = prompts_tokenized[bi]["position_ids"].max(1).item() + 1
        tlm_precode_inputs["position_ids"][bi] = np.arange(
            input_len, input_len + num_speculative_tokens + 1, dtype=np.int64
        )
        # assumes that prefill queue will always be popped from the front
        input_lengths[bi] = input_len
        max_gen_len[bi] -= input_lengths[bi]
        all_ids[bi, : input_len + 1] = prompts_tokenized[bi]["input_ids"][0, :input_len].tolist() + [input_ids.item()]
        prompt_plus_gen_idx[bi] = input_len + 1
    batch_ttft = perf_counter() - e2e_start

    # set decode logits buffers
    target_model_session.set_buffers({"logits": precode_logits_ph})
    # start decode phase
    valid_batch_indices = np.full(decode_batch_size, True, dtype=bool)
    all_accept = False
    it = 0
    decode_start = perf_counter()
    mean_num_accepted_tokens = 0
    all_accept = np.full(decode_batch_size, False, dtype=bool)
    tlm_position_ids = np.arange(num_speculative_tokens + 1).reshape(1, -1).repeat(decode_batch_size, axis=0)
    empty_indices = np.zeros(decode_batch_size, dtype=bool)
    while True:
        it += 1
        for bi, valid in enumerate(valid_batch_indices):
            if not valid:
                continue
            # generate n-grapm proposals
            (
                spec_tokens,  # shape: [num_speculative_tokens]
                has_empty_tokens,
            ) = find_candidate_pred_tokens(
                all_ids[bi : bi + 1, : prompt_plus_gen_idx[bi]],
                fill_tok=-1,
                max_ngram_size=max_ngram_size,
                num_pred_tokens=num_speculative_tokens,
            )
            empty_indices[bi] = has_empty_tokens
            # prepare target model inputs
            if has_empty_tokens:
                # avoid read/write of KV$ for meaningless tokens
                tlm_precode_inputs["position_ids"][bi, 1:] = -1
            else:
                tlm_precode_inputs["input_ids"][bi, 1:] = spec_tokens
        # run precode on TLM to score the proposed tokens
        tlm_outputs = target_model_session.run(tlm_precode_inputs)
        target_logits = tlm_outputs["logits"]
        # greedy sampling from target model
        target_tokens = target_logits.argmax(-1)
        # exact matching between draft and target tokens
        num_tokens_selected = np.ones(decode_batch_size, dtype=np.int64)
        tlm_precode_position_ids = np.full((decode_batch_size, num_speculative_tokens + 1), -1, dtype=np.int64)
        non_empty_valid_indices = ~empty_indices & valid_batch_indices
        matching = (
            tlm_precode_inputs["input_ids"][non_empty_valid_indices, 1:] == target_tokens[non_empty_valid_indices, :-1]
        )  # shape: [non_empty_valid_indices, num_speculative_tokens]
        num_tokens_selected[non_empty_valid_indices] = matching.cumprod(axis=1).sum(axis=1) + 1
        if empty_indices.sum() > 0:
            tlm_precode_position_ids[empty_indices] = tlm_position_ids[empty_indices] + (
                tlm_precode_inputs["position_ids"][empty_indices, 0] + 1
            ).reshape(-1, 1)
        if non_empty_valid_indices.sum() > 0:
            tlm_precode_position_ids[non_empty_valid_indices] = tlm_precode_inputs["position_ids"][
                non_empty_valid_indices
            ] + num_tokens_selected[non_empty_valid_indices].reshape(-1, 1)
        # record accepted tokens
        all_accept[valid_batch_indices] = num_tokens_selected[valid_batch_indices] == num_speculative_tokens + 1
        mean_num_accepted_tokens += num_tokens_selected[valid_batch_indices].mean().item()
        # append selected tokens to the generated_ids
        for bi, valid in enumerate(valid_batch_indices):
            if not valid:
                continue
            accepted_tokens = num_tokens_selected[bi]
            num_tokens_to_append = min(accepted_tokens, max_gen_len[bi] - len(generated_ids[bi]))
            gen_ids = target_tokens[bi, :num_tokens_to_append]
            all_ids[bi, prompt_plus_gen_idx[bi] : prompt_plus_gen_idx[bi] + num_tokens_to_append] = gen_ids
            prompt_plus_gen_idx[bi] += num_tokens_to_append
            generated_ids[bi].extend(gen_ids.tolist())
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
        tlm_precode_inputs["input_ids"][valid_batch_indices, 0] = common_input_ids.flatten()
        tlm_precode_position_ids[~valid_batch_indices] = -1
        tlm_precode_inputs["position_ids"] = tlm_precode_position_ids
    end = perf_counter()
    # calculate performance metrics
    decode_end = end - decode_start
    e2e_end = end - e2e_start
    mean_ttft = sum(ttfts) / len(ttfts)
    generated_tokens_per_prompt = [len(gid) + 1 for gid in generated_ids]
    decode_throughput = sum(generated_tokens_per_prompt) / decode_end
    e2e_throughput = (sum(generated_tokens_per_prompt) + decode_batch_size) / e2e_end
    batch_decode = tokenizer.batch_decode(generated_ids)
    mean_num_accepted_tokens /= it
    perf_metrics = PerfMetrics(
        mean_ttft,
        batch_ttft,
        decode_throughput,
        e2e_throughput,
        mean_num_accepted_tokens,
        max_gen_len,
        generated_tokens_per_prompt,
    )
    draft_model_name = "PLD"
    exec_info = CloudAI100ExecInfo(
        prompts,
        decode_batch_size,
        batch_decode,
        generated_ids,
        perf_metrics,
        num_speculative_tokens,
        prefill_seq_len,
        ctx_len,
        prefill_bsz,
        draft_model_name,
        target_model_name,
        full_batch_size,
    )
    del target_model_session
    del draft_model_session
    generated_ids = np.asarray(generated_ids[0]).flatten()
    gen_len = generated_ids.shape[0]
    exec_info = target_model.generate(tokenizer, Constants.INPUT_STR)
    cloud_ai_100_tokens = exec_info.generated_ids[0][
        :gen_len
    ]  # Because we always run for single input and single batch size
    all_matching = np.array_equal(cloud_ai_100_tokens, generated_ids)
    assert all_matching, "Tokens don't match for SpD output and vanilla DLM output."
