# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from time import perf_counter
from typing import Dict, List, Optional, Union

import numpy as np
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import logger

io_files = []


def write_io_files(
    inputs: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    write_io_dir: str,
    write_io_subdir: str,
    write_io_name: str,
    include_dims: bool = False,
    reset: bool = False,
):
    global io_files
    if reset:
        io_files = []
    io = []
    os.makedirs(f"{write_io_dir}/{write_io_subdir}", exist_ok=True)
    for iname, iarray in inputs.items():
        iarray.tofile(f"{write_io_dir}/{write_io_subdir}/{iname}.raw")
        ispec = {
            "path": f"{write_io_subdir}/{iname}.raw",
            "io-direction": "in",
            "elem-size": iarray.itemsize,
            "map-to": iname,
        }
        if include_dims:
            ispec["dims"] = iarray.shape
        io.append(ispec)
    for oname, oarray in outputs.items():
        oarray.tofile(f"{write_io_dir}/{write_io_subdir}/{oname}.raw")
        ospec = {
            "path": f"{write_io_subdir}/{oname}.raw",
            "io-direction": "out",
            "elem-size": oarray.itemsize,
            "map-to": oname,
        }
        if include_dims or oname.endswith("_RetainedState"):
            ospec["dims"] = oarray.shape
        io.append(ospec)
    io_files.append(io)
    with open(f"{write_io_dir}/{write_io_name}.json", "w") as fp:
        json.dump({"IO-files": io_files}, fp, indent=True)


def latency_stats_bertstyle(
    model_name: str,
    qpc: str,
    seq_len: int,
    prompt: str,
    device_id: List[int] = [0],
):
    session = QAICInferenceSession(qpc, device_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="np", max_length=seq_len, padding="max_length")
    next_token_id = inputs["input_ids"][0, -1]
    cur_len = inputs["attention_mask"].sum().item()
    print(prompt, end=" ", flush=True)
    init_len = cur_len
    start = perf_counter()
    while next_token_id != tokenizer.eos_token_id and cur_len <= seq_len:
        outputs = session.run(inputs)
        logits = outputs["logits"]
        next_token_id = logits[0, -1].argmax().item()
        inputs["input_ids"] = np.concatenate(
            [
                inputs["input_ids"][:, 1:],
                np.ones((1, 1), dtype=np.int64) * next_token_id,
            ],
            1,
        )
        inputs["attention_mask"] = np.concatenate([inputs["attention_mask"][:, 1:], np.ones((1, 1), dtype=np.int64)], 1)
        print(tokenizer.decode(next_token_id), end=" ", flush=True)
        cur_len += 1
    end = perf_counter()
    print()
    print(round((cur_len - init_len) / (end - start), 2), "tok/s")


def get_compilation_batch_size(qpc_path: str):
    qpc_base_path = os.path.dirname(os.path.normpath(qpc_path))
    specialization_file_path = os.path.join(qpc_base_path, "specializations.json")
    logger.info(f"specialization_file_path : {specialization_file_path}")
    with open(specialization_file_path, "r") as file:
        data = json.load(file)
    compilation_batch_size = int(data["specializations"][0]["batch_size"])
    return compilation_batch_size


def check_batch_size_and_num_prompts(prompt, prompts_txt_file_path, batch_size) -> List[str]:
    assert (
        prompt is not None or prompts_txt_file_path is not None
    ), "Please pass atleast one argument either using --prompt or --prompts_txt_file_path"
    if prompts_txt_file_path is not None:
        if prompt is not None:
            logger.warning("Found inputs passed using txt file as well as CLI, taking inputs from given txt file")
        prompt = read_prompts_txt_file(prompts_txt_file_path)
    if isinstance(prompt, str):
        prompt = eval(prompt)
    num_prompts = len(prompt)
    if batch_size > 1:
        assert (
            batch_size == num_prompts
        ), f"Mismatch between number of prompts {num_prompts} and batch size {batch_size}; please pass correct input argument"
    return prompt


def read_prompts_txt_file(prompts_txt_file_path: str):
    prompt = []
    with open(prompts_txt_file_path, "r") as file:
        for line in file:
            prompt.append(line.strip())
    return prompt

def create_decode_inputs(decode_batch_size, tokenizer, batch_index, ctx_len):
    """
    This function creates the decode inputs.

    Returns:
        dict: The decode inputs.
    """
    decode_inputs = {}
    # Create position IDs filled with zeros
    decode_inputs["position_ids"] = np.zeros((decode_batch_size, 1), np.int64)
    # Create input IDs filled with the pad token ID
    decode_inputs["input_ids"] = np.full((decode_batch_size, 1), tokenizer.pad_token_id)
    decode_inputs["batch_index"] = batch_index
    # Create attention mask filled with zeros
    decode_inputs["attention_mask"] = np.zeros((decode_batch_size, ctx_len), dtype=bool)
    return decode_inputs

def run_prefill(prompt, tokenizer, session, num_chunks, padded_len, prefill_seq_len, write_io_dir):
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    # Need to use -1 as position_ids for invalid tokens

    # Run prefill
    for i in range(num_chunks):
        chunk_inputs = inputs.copy()
        chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        outputs = session.run(chunk_inputs)
        if write_io_dir:
                write_io_files(inputs, outputs, write_io_dir, "prefill", "aic_batch_io", True, False)

    return outputs

def cloud_ai_100_exec_kv_helper(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc: str,
    prompt: List[str],
    full_batch_size: int = 1,
    input_len: Optional[int] = None,
    generation_len: Optional[int] = None,
    device_id: List[int] = [0],
    enable_debug_logs: bool = False,
    stream: bool = True,
    write_io_dir: Optional[str] = None,
):
    if tokenizer.padding_side != "right":
        logger.warning("Please use padding_side='right' while initializing the tokenizer")
        tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load QPC
    session = QAICInferenceSession(qpc, device_id, enable_debug_logs=enable_debug_logs)

    # Skip inputs/outputs
    session.skip_buffers([x for x in session.input_names + session.output_names if x.startswith("past_")])

    # Read prompt and ctx len from session
    batch_size, _, ctx_len, _ = session.bindings[session.binding_index_map["past_key.0"]].dims
    prefill_seq_len = max(
        [x[session.binding_index_map["input_ids"]][1][1] for x in session.allowed_shapes]
        + [session.bindings[session.binding_index_map["input_ids"]].dims[1]])
    
    # Initiate a prefill queue. 
    prefill_queue = []

    # FIXME need to support batch size and/or decode batch size.
    # if len(prompt) < batch_size:
    #     prompt = prompt * -(batch_size // -len(prompt))  # Repeat prompt to required size
    # prompt = prompt[:batch_size]  # Truncate prompts to required size

    # Truncate prompts to required size
    if len(prompt) < full_batch_size:
        print(f"Repeating prompt {full_batch_size} times")
        prompt = prompt * -(full_batch_size // -len(prompt))  # Repeat prompt to required size
    prompt = prompt[:full_batch_size]
   
    # add all prompts to the prefill queue
    prefill_queue = list(prompt)
    logger.info(f"Request queue initially:  {prefill_queue}")  
    
    # initialize batch index
    batch_index = np.reshape(np.array(np.arange(batch_size), np.int64), (batch_size, 1))

    # Create decoder input dict 
    decode_inputs = create_decode_inputs(full_batch_size, tokenizer, batch_index, ctx_len)

    # initialize empty list to store generated tokens for each prompt
    generated_ids = [[] for _ in range(full_batch_size)]
    # store the length of each prompt requested
    input_lengths = [0 for _ in range(full_batch_size)]
    # store the number of prompts processed out of the prompt_queue
    num_prompts_processed = 0
    # initialize dynamic container which will hold all the global request ids (position in prompt request queue)
    # of the prompts currently being processed
    current_batch_req_ids = []
    
    if generation_len is None:
        generation_len = ctx_len #- input_len.max()
    assert generation_len > 0, "generation length should be greater than zero"
    
    generated_ids = np.full((batch_size, generation_len + 1), tokenizer.pad_token_id)
    if stream:
        streamer = transformers.TextStreamer(tokenizer)
        streamer.on_finalized_text(prompt[0] + " ")

    # Prepare inputs for prefill
    start = perf_counter()
    # inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    # inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
    # # Need to use -1 as position_ids for invalid tokens

    # # Run prefill
    # for i in range(num_chunks):
    #     chunk_inputs = inputs.copy()
    #     chunk_inputs["input_ids"] = inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
    #     chunk_inputs["position_ids"] = inputs["position_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
    #     outputs = session.run(chunk_inputs)
    
    # run prefill and accumulate results for all inputs in the queue. 
    for decode_batch_id in range(full_batch_size):
        prompt = prefill_queue[decode_batch_id] # Get the prompt from the prefill queue
        inputs = tokenizer(prompt, return_tensors="np", padding=True) # Tokenize the input. 
        input_len = inputs["attention_mask"].sum(1, keepdims=True) # Generate input length from the attention mask
        padded_len = inputs["input_ids"].shape[1] # calculate the padded length
        num_chunks = -(padded_len // -prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

        assert padded_len <= ctx_len, "input_len should be less than ctx_len"\
        
        # TODO need to store the attention mask and position ids for each batch element so that we can access them
        # at decode time
        # inputs["position_ids"] = (np.cumsum(inputs["attention_mask"][0:1], 1) - 1) * inputs["attention_mask"][0:1]
        # inputs["attention_mask"] = np.concatenate(
        #     [inputs["attention_mask"].astype(bool),np.zeros((full_batch_size, ctx_len - input_len), dtype=bool),], 1,)
        
        # cache_index = np.array([[0]], np.int64)
        batch_index = np.array([[decode_batch_id]], np.int64)
        # inputs["cache_index"] = cache_index
        inputs["batch_index"] = batch_index
        # run prefill for num_chunks
        outputs =  run_prefill(prompt, tokenizer, session, num_chunks, padded_len, prefill_seq_len, write_io_dir)
        
        logits = outputs["logits"]
        print("##############logits.shape", logits.shape)
    
        # Get output token
        inputs["input_ids"] = outputs["logits"].argmax(2)
        inputs["position_ids"] = input_len
        generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
        finished_sequences = inputs["input_ids"] == tokenizer.eos_token_id
        if stream:
            streamer.put(inputs["input_ids"][0])
        
        # FIXME assumes that prefill queue will always be popped from the front
        current_batch_req_ids.append(decode_batch_id)
        input_lengths[current_batch_req_ids[decode_batch_id]] = inputs["input_len"]
        num_prompts_processed += 1
        # update generated id list for this request, right after running prefill
        generated_ids[current_batch_req_ids[decode_batch_id]].append(inputs["input_ids"][0, 0])
        # pop the front of the prefill queue
        # assumes that prefill queue will always be popped from the front
        prefill_queue = prefill_queue[1:]

    # Decode loop
    loop_start = perf_counter()
    for num_token in range(1, generation_len):
        outputs = session.run(inputs)

        if write_io_dir:
            write_io_files(inputs, outputs, write_io_dir, "decode", "aic_batch_io", True, False)
            write_io_dir = None

        # Prepare inputs for next iteration
        inputs["input_ids"] = outputs["logits"].argmax(2)
        inputs["position_ids"] += 1
        generated_ids[:, num_token] = inputs["input_ids"].squeeze(1)
        finished_sequences |= inputs["input_ids"] == tokenizer.eos_token_id
        if stream:
            streamer.put(inputs["input_ids"][0])

        if finished_sequences.all():
            break

    end = perf_counter()
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print("generated_texts", generated_texts)

    for i in range(1 if stream else 0, batch_size):
        print()
        print(i, prompt[i], generated_texts[i])

    prefill_perf = 1 / (loop_start - start)
    decode_perf = (num_token - 1) / (end - loop_start)
    total_perf = num_token / (end - start)
    total_time = end - start
    print()

    latency_stats = (generated_texts, prefill_perf, decode_perf, total_perf, total_time)
    return latency_stats


def print_latency_stats_kv(
    prompt, generated_texts, batch_size, prefill_time, decode_perf, total_perf, total_time, automation: bool = False
):
    if automation:
        print()
        print("input=", prompt)
        print("output=", generated_texts)
        print("Prefill time a.k.a TTFT is=", round(prefill_time * batch_size, 2))
        print("Decode token/sec is=", round(decode_perf * batch_size, 2))
        print("Total token/sec is=", round(total_perf * batch_size, 2))
        print("Total (E2E) inference time is=", round(total_time, 2))
        return
    print()

    print("===================== Performance Stats =====================")
    if batch_size > 1:
        print("Prefill time a.k.a TTFT (batch) is :", round(prefill_time * batch_size, 2), "s")
        print("Decode (batch):", round(decode_perf * batch_size, 2), "tok/s")
        print("E2E (batch):", round(total_perf * batch_size, 2), "tok/s")
        print("Total (E2E) inference time (batch) is=", round(total_time, 2), "s")
    else:
        print("Prefill time a.k.a TTFT is=", round(prefill_time, 2), "s")
        print("Decode:", round(decode_perf, 2), "tok/s")
        print("E2E:", round(total_perf, 2), "tok/s")
        print("Total (E2E) inference time is=", round(total_time, 2), "s")
    print("=============================================================")


def cloud_ai_100_exec_kv(
    batch_size,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc_path: str,
    prompt: Optional[List[str]] = None,
    device_id: List[int] = [0],
    input_len: Optional[int] = None,
    generation_len: Optional[int] = None,
    enable_debug_logs: bool = False,
    stream: bool = True,
    write_io_dir: Optional[str] = None,
    automation=False,
    full_batch_size: int = 1,
):
    if batch_size == 1:
        prefill_time = []
        decode_perf = []
        total_perf = []
        total_time = []
        generated_texts = []
        for i in range(len(prompt)):
            latency_stats = cloud_ai_100_exec_kv_helper(
                tokenizer=tokenizer,
                prompt=[prompt[i]],
                qpc=qpc_path,
                device_id=device_id,
                input_len=input_len,
                generation_len=generation_len,
                enable_debug_logs=enable_debug_logs,
                stream=stream,
                write_io_dir=write_io_dir,
                full_batch_size=full_batch_size,
            )
            generated_texts.append(latency_stats[0])
            prefill_time.append(latency_stats[1])
            decode_perf.append(latency_stats[2])
            total_perf.append(latency_stats[3])
            total_time.append(latency_stats[4])

        prefill_time = np.average(prefill_time)
        decode_perf = np.average(decode_perf)
        total_perf = np.average(total_perf)
        total_time = np.average(total_time)

    else:
        latency_stats = cloud_ai_100_exec_kv_helper(
            tokenizer=tokenizer,
            prompt=prompt,
            qpc=qpc_path,
            device_id=device_id,
            input_len=input_len,
            generation_len=generation_len,
            enable_debug_logs=enable_debug_logs,
            stream=stream,
            write_io_dir=write_io_dir,
        )
        generated_texts, prefill_time, decode_perf, total_perf, total_time = latency_stats

    print_latency_stats_kv(
        prompt,
        generated_texts,
        batch_size,
        prefill_time,
        decode_perf,
        total_perf,
        total_time,
        automation=automation,
    )
