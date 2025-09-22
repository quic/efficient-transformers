# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Union

import numpy as np
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import Constants


@dataclass
class SpDPerfMetrics:
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
    e2e_time: float
    decode_time: float
    decode_target_time: float
    decode_iterations: int


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
    perf_metrics: SpDPerfMetrics
    num_speculative_tokens: int
    prefill_seq_len: int
    ctx_len: int
    prefill_bsz: int
    model_name: str
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


def run_prefill(
    session: QAICInferenceSession,
    inputs: dict,
    prefill_seq_len: int,
    slot_idx: int,
) -> np.ndarray:
    input_len = inputs.input_ids.shape[1]
    num_chunks = input_len // prefill_seq_len
    cache_index = np.array([[0]], np.int64)
    batch_index = np.array([[slot_idx]], np.int64)
    inputs["batch_index"] = batch_index

    # Run chunked prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prefill_seq_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, cache_index[0, 0] : cache_index[0, 0] + prefill_seq_len]

        outputs = session.run(chunk_inputs)
        cache_index += prefill_seq_len

    logits = outputs["logits"]
    return logits


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


def multiprojs_spec_decode_inference(
    prompts: List[str],
    num_speculative_tokens: int,
    prefill_seq_len: int,
    ctx_len: int,
    prefill_bsz: int,
    pretrained_model_name_or_path: str,
    full_batch_size: Optional[int],
    session: QAICInferenceSession,
    ignore_eos_token: bool = False,
) -> CloudAI100ExecInfo:
    """
    Perform draft speculative decode inference on the given prompts.

    Args:
        prompts (List[str]): List of prompts to perform inference on.
        num_speculative_tokens (int): Number of speculative tokens.
        prefill_seq_len (int): Prefill sequence length.
        ctx_len (int): Context length.
        prefill_bsz (int): Prefill batch size.
        pretrained_model_name_or_path (str): Name of multiprojection model
        full_batch_size (Optional[int]): Full batch size.
        device_group (List[int]): List of device IDs.

    Returns:
        CloudAI100ExecInfo: Execution information, including performance metrics and generated text.
    """
    # assumes dlm and tlm are compiled to the same prompt-chunk-size, context length and full_batch_size/batch-size
    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    # skip inputs/outputs buffers
    session.skip_buffers(set([x for x in session.input_names if x.startswith("past_")]))
    session.skip_buffers(set([x for x in session.output_names if x.endswith("_RetainedState")]))

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
    # mock input key "logits" to store the first batch of output logits
    num_logits_to_keep = num_speculative_tokens + 1  # number of logits to keep
    precode_inputs = dict(
        input_ids=np.zeros((decode_batch_size, num_logits_to_keep), dtype=np.int64),
        position_ids=np.zeros((decode_batch_size, num_logits_to_keep), dtype=np.int64),
        batch_index=np.arange(decode_batch_size, dtype=np.int64).reshape(-1, 1),
        num_logits_to_keep=np.zeros((num_logits_to_keep, 1), dtype=np.int64),
    )
    max_gen_len = [ctx_len] * decode_batch_size
    # setup buffers
    prefill_logits_ph = np.zeros((prefill_bsz, 1, num_logits_to_keep, vocab_size), dtype=np.float32)
    session.set_buffers({"logits": prefill_logits_ph})
    e2e_start = perf_counter()
    ttfts = []
    for bi in range(decode_batch_size):
        # assumes that prefill queue will always be popped from the front
        start = perf_counter()
        logits = run_prefill(  # shape: [1, 1, num_logits_to_keep, vocab_size]
            session=session,
            inputs=prompts_tokenized[bi],
            prefill_seq_len=prefill_seq_len,
            slot_idx=bi,
        )
        ttft = perf_counter() - start
        ttfts.append(ttft)
        input_ids = logits.argmax(-1).astype(np.int64)  # shape: [1, 1, num_logits_to_keep]
        generated_ids[bi].append(input_ids[0, 0, 0].item())
        precode_inputs["input_ids"][bi] = input_ids.flatten()
        input_len = prompts_tokenized[bi]["position_ids"].max(1).item() + 1
        precode_inputs["position_ids"][bi] = np.arange(input_len, input_len + num_logits_to_keep, dtype=np.int64)
        # assumes that prefill queue will always be popped from the front
        input_lengths[bi] = input_len
        max_gen_len[bi] -= input_lengths[bi]
    batch_ttft = perf_counter() - e2e_start

    # set decode logits buffers
    precode_logits_ph = np.zeros((decode_batch_size, num_logits_to_keep, num_logits_to_keep, vocab_size), dtype=np.float32)
    session.set_buffers({"logits": precode_logits_ph})
    # start decode phase
    valid_batch_indices = np.full(decode_batch_size, True, dtype=bool)
    seq_batch_indices = np.arange(decode_batch_size, dtype=np.int64)
    it = 0
    mean_num_accepted_tokens = 0
    decode_target_time = 0.0
    decode_start = perf_counter()
    while True:
        it += 1
        # run precode
        target_start = perf_counter()
        tlm_outputs = session.run(precode_inputs)
        target_logits = tlm_outputs["logits"]  # shape: [decode_batch_size, num_logits_to_keep, num_logits_to_keep, vocab_size]
        # greedy sampling from target model
        target_tokens = target_logits[:, :, 0].argmax(-1)  # shape: [decode_batch_size, num_logits_to_keep]
        target_end = perf_counter() - target_start
        decode_target_time += target_end
        # exact matching between draft and target tokens
        draft_tokens = precode_inputs["input_ids"][:, 1:]  # shape: [decode_batch_size, num_speculative_tokens]
        matching = draft_tokens == target_tokens[:, :-1]  # shape: [decode_batch_size, num_speculative_tokens]
        num_tokens_selected = matching.cumprod(axis=1).sum(axis=1) + 1  # shape: [decode_batch_size]
        mean_num_accepted_tokens += num_tokens_selected[valid_batch_indices].mean().item()
        # append selected tokens to the generated_ids
        for bi, valid in enumerate(valid_batch_indices):
            if not valid:
                continue
            accepted_tokens = num_tokens_selected[bi]
            num_tokens_to_append = min(accepted_tokens, max_gen_len[bi] - len(generated_ids[bi]))
            accepted_tokens_arr = target_tokens[bi, :num_tokens_to_append]
            generated_ids[bi].extend(accepted_tokens_arr.tolist())
            if len(generated_ids[bi]) >= max_gen_len[bi] or ((not ignore_eos_token) and (accepted_tokens_arr == tokenizer.eos_token_id).any()):
                valid_batch_indices[bi] = False
        # check if all generations are done
        if not valid_batch_indices.any():
            break
        # prepare decode inputs for next decode iteration
        next_input_ids = target_logits[seq_batch_indices, num_tokens_selected - 1].argmax(-1).astype(np.int64)  # shape: [decode_batch_size, num_logits_to_keep]
        next_position_ids = precode_inputs["position_ids"] + num_tokens_selected[:, np.newaxis]
        next_position_ids[~valid_batch_indices] = -1
        precode_inputs["input_ids"] = next_input_ids
        precode_inputs["position_ids"] = next_position_ids
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
    perf_metrics = SpDPerfMetrics(
        mean_ttft,
        batch_ttft,
        decode_throughput,
        e2e_throughput,
        mean_num_accepted_tokens,
        max_gen_len,
        generated_tokens_per_prompt,
        e2e_end,
        decode_end,
        decode_target_time,
        it,
    )
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
        pretrained_model_name_or_path,
        full_batch_size,
    )
    return exec_info


def optional_int(x):
    if x is None:
        return None
    return int(x)


def comma_separated_ints(x: str):
    return [int(qid) for qid in x.split(",")]


def arg_parse():
    parser = argparse.ArgumentParser(description="Draft-based SpD Inference")
    parser.add_argument("--prompts", action="append", default=None, help="Input prompt(s)")
    parser.add_argument("--prefill-seq-len", type=int, default=32, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=128, help="Context length")
    parser.add_argument("--prefill-bsz", type=int, default=1, help="Prefill batch size")
    parser.add_argument(
        "--pretrained-model-name-or-path",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Target model name",
    )
    parser.add_argument("--full-batch-size", type=optional_int, default=None, help="Full batch size")
    parser.add_argument("--device-group", type=comma_separated_ints, default="0", help="device QIDs")
    parser.add_argument("--ignore-eos-token", action="store_true")
    args = parser.parse_args()
    return args


def get_session(
    pretrained_model_name_or_path,
    device_group,
    prefill_seq_len,
    ctx_len,
    full_batch_size=None,
):
    is_cb = full_batch_size is not None
    qaic_config = dict(speculative_model_type="turbo")
    qeff_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        continuous_batching=is_cb,
        qaic_config=qaic_config,
    )
    num_devices = len(device_group)
    model_qpc_path: str = qeff_model.compile(
        num_cores=16,
        num_devices=num_devices,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        aic_enable_depth_first=True,
        full_batch_size=full_batch_size,
    )
    print(f"{model_qpc_path=}")
    # init qaic session
    session = QAICInferenceSession(model_qpc_path, device_ids=device_group)
    num_speculative_tokens = qeff_model.model.config.speculative_config["num_speculative_tokens"]
    return session, num_speculative_tokens


def main():
    args = arg_parse()
    if args.prompts is None:
        args.prompts = Constants.INPUT_STR

    session, num_speculative_tokens = get_session(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        device_group=args.device_group,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        full_batch_size=args.full_batch_size,
    )
    args.session = session
    exec_info = multiprojs_spec_decode_inference(
        args.prompts,
        num_speculative_tokens,
        args.prefill_seq_len,
        args.ctx_len,
        args.prefill_bsz,
        args.pretrained_model_name_or_path,
        args.full_batch_size,
        args.session,
        args.ignore_eos_token,
    )
    print(exec_info)
    prompts = exec_info.prompts
    generated_texts = exec_info.generated_texts
    for prompt, generation in zip(prompts, generated_texts):
        print(f"{prompt=} {generation=}")


if __name__ == "__main__":
    main()
