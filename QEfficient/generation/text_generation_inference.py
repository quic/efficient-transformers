# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import padding_check_and_fix
from QEfficient.utils.logging_utils import logger


@dataclass
class CloudAI100ExecInfo:
    """
    Holds all the information about Cloud AI 100 execution

    Args:
        :batch_size (int): Batch size of the QPC compilation.
        :generated_texts (Union[List[List[str]], List[str]]): Generated text(s).
        :generated_ids (Union[List[np.ndarray], np.ndarray]): Generated IDs.
        :prefill_time (float): Time for prefilling.
        :decode_perf (float): Decoding performance.
        :total_perf (float): Total performance.
        :total_time (float): Total time.
    """

    batch_size: int
    generated_texts: Union[List[str], List[List[str]]]
    generated_ids: Union[List[np.ndarray], np.ndarray]
    prefill_time: float
    decode_perf: float
    total_perf: float
    total_time: float

    def __repr__(self):
        return f"Prefill time a.k.a TTFT is= {round(self.prefill_time, 2)}\
        \nDecode token/sec is= {round(self.decode_perf * self.batch_size, 2)}\
        \nTotal token/sec is= {round(self.total_perf * self.batch_size, 2)}\
        \nTotal (E2E) inference time is= {round(self.total_time, 2)}"


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
    qpc_path: str,
    seq_len: int,
    prompt: str,
    device_id: Optional[List[int]] = None,
):
    """
    Function to execute Bertstyle ONNX model on Cloud AI 100.

    Args:
        :model_name (str): Hugging Face Model Card name, Example: gpt2.
        :qpc_path (str): Path to save generated binary file after compilation.
        :seq_len (int): Sequence length.
        :prompt (str): Sample prompt for the model text generation.
        :device_id (List[int]): Device Ids to be used for compilation. If devices > 1, it enables multiple card setup.
    """
    session = QAICInferenceSession(qpc_path, device_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
    padding_check_and_fix(tokenizer)  # Check and fix tokenizer viability
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


def get_compilation_dims(qpc_path: str) -> Tuple[int, int]:
    qpc_base_path = os.path.dirname(os.path.normpath(qpc_path))
    specialization_file_path = os.path.join(qpc_base_path, "specializations.json")
    logger.info(f"specialization_file_path : {specialization_file_path}")

    if os.path.exists(specialization_file_path):
        with open(specialization_file_path, "r") as file:
            data = json.load(file)
    else:
        raise FileNotFoundError(f"expected specializations.json file at path, {qpc_base_path}")

    compilation_batch_size = int(data["specializations"][0]["batch_size"])
    compilation_ctx_len = int(data["specializations"][0]["ctx_len"])
    return compilation_batch_size, compilation_ctx_len


def get_input_prompts(prompt: str, prompts_txt_file_path: str) -> List[str]:
    assert (
        prompt is not None or prompts_txt_file_path is not None
    ), "Please pass at least one argument either using --prompt or --prompts_txt_file_path"
    if prompts_txt_file_path is not None:
        if prompt is not None:
            logger.warning("Found inputs passed using txt file as well as CLI, taking inputs from given txt file")
        prompt = read_prompts_txt_file(prompts_txt_file_path)
    if isinstance(prompt, str):
        prompt = [prompt]
    return prompt


def fix_prompts(prompt: List[str], batch_size: int):
    if len(prompt) < batch_size:
        logger.warning("Number of prompts are less than batch size, repeating to required batch size")
        prompt = prompt * -(batch_size // -len(prompt))  # Repeat prompt to required size
        prompt = prompt[:batch_size]  # Truncate prompts to required size
    else:
        if (len(prompt) % batch_size) > 0:
            logger.warning(
                "Number of prompts are not multiple of batch size, dropping last incomplete batch from given input prompts"
            )
            prompt = prompt[: batch_size * (len(prompt) // batch_size)]
    return prompt


def read_prompts_txt_file(prompts_txt_file_path: str):
    prompt = []
    with open(prompts_txt_file_path, "r") as file:
        for line in file:
            prompt.append(line.strip())
    return prompt


def cloud_ai_100_exec_kv_helper(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc_path: str,
    prompt: List[str],
    ctx_len: int,
    generation_len: Optional[int] = None,
    device_id: Optional[List[int]] = None,
    enable_debug_logs: bool = False,
    stream: bool = True,
    write_io_dir: Optional[str] = None,
):
    """
    Helper function to execute QEfficient transformed ONNX model on ``Cloud AI 100`` using compiled QPC file.

    ``Mandatory`` Args:
        :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Model tokenizer.
        :qpc_path (str): Path to the saved generated binary file after compilation.
        :prompt (str): Sample prompt for the model text generation.
        :ctx_len (int): Input length of the prompt to determine the number of chunks to execute on ``Cloud AI 100``.
    ``Optional`` Args:
        :generation_len (int): Maximum context length for the model during compilation. ``Defaults to None``.
        :device_id (List[int]): Device IDs to be used for compilation. If ``len(device_id) > 1``, it enables multiple card setup. ``Defaults to [0]``.
        :enable_debug_logs (bool): If True, it enables debugging logs. ``Defaults to False``.
        :stream (bool): If True, enable streamer, which returns tokens one by one as the model generates them.``Defaults to True``.
        :Write_io_dir (str): Path to write the input and output files.``Defaults to None``.
    """

    if tokenizer.padding_side != "right":
        logger.warning("Please use padding_side='right' while initializing the tokenizer")
        tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load QPC
    session = QAICInferenceSession(qpc_path, device_id, enable_debug_logs=enable_debug_logs)

    # Skip inputs/outputs
    session.skip_buffers([x for x in session.input_names + session.output_names if x.startswith("past_")])

    # Read batch_size and prefill_seq_len from session
    if session.allowed_shapes:
        batch_size = max([x[session.binding_index_map["input_ids"]][1][0] for x in session.allowed_shapes])
        prefill_seq_len = max([x[session.binding_index_map["input_ids"]][1][1] for x in session.allowed_shapes])
    else:
        batch_size, prefill_seq_len = session.bindings[session.binding_index_map["input_ids"]].dims

    inputs = tokenizer(prompt, return_tensors="np", padding=True)

    position_ids_update = inputs["attention_mask"].sum(1, keepdims=True)
    padded_len = inputs["input_ids"].shape[1]
    num_chunks = -(padded_len // -prefill_seq_len)  # ceil divide without float
    padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len
    max_gen_len = ctx_len - position_ids_update.max()
    if generation_len is None:
        if ctx_len is None:
            raise ValueError("At least one of ctx_len or generation_len is needed")
        generation_len = max_gen_len
    elif generation_len > max_gen_len:
        logger.warning(
            "Passed generation_len is greater than allowed length. "
            "Make sure this model supports sliding window, such as Mistral"
        )
    assert generation_len > 0, "generation length should be greater than zero"
    generated_ids = np.full((batch_size, generation_len), tokenizer.pad_token_id)
    if stream:
        streamer = transformers.TextStreamer(tokenizer)
        streamer.on_finalized_text("\nPrompt : " + prompt[0] + "\nCompletion :")

    # Prepare inputs for prefill/first iteration
    start = perf_counter()
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

    # Get first token
    inputs["input_ids"] = outputs["logits"].argmax(2)
    inputs["position_ids"] = position_ids_update
    generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
    finished_sequences = inputs["input_ids"] == tokenizer.eos_token_id
    if stream:
        streamer.put(inputs["input_ids"][0])

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

    if stream:
        for i in range(1, batch_size):
            print("\n\nPrompt : ", prompt[i])
            print("Completion :", generated_texts[i])

    prefill_time = loop_start - start
    decode_perf = (num_token - 1) / (end - loop_start)
    total_perf = num_token / (end - start)
    total_time = end - start

    return CloudAI100ExecInfo(
        batch_size=batch_size,
        generated_texts=generated_texts,
        generated_ids=generated_ids,
        prefill_time=prefill_time,
        decode_perf=decode_perf,
        total_perf=total_perf,
        total_time=total_time,
    )


def print_latency_stats_kv(prompt, execinfo, automation: bool = False):
    if automation:
        print("input=", prompt)
        print("output=", execinfo.generated_texts)
        print(execinfo)
        return
    print("========================= Performance Stats =========================")
    if execinfo.batch_size > 1:
        print("Batch Performance : \n")
    print(execinfo)
    print("=====================================================================")


def cloud_ai_100_exec_kv(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    qpc_path: str,
    prompt: Optional[str] = None,
    prompts_txt_file_path: Optional[str] = None,
    device_id: Optional[List[int]] = None,
    generation_len: Optional[int] = None,
    enable_debug_logs: bool = False,
    stream: bool = True,
    write_io_dir: Optional[str] = None,
    automation=False,
):
    """
    This method generates output until ``eos`` or ``generation_len`` by executing the compiled ``qpc`` on ``Cloud AI 100`` Hardware cards.
    This is a sequential execution based on the ``batch_size`` of the compiled model and the number of prompts passed.
    If the number of prompts cannot be divided by the ``batch_size``, the last unfulfilled batch will be dropped.

    ``Mandatory`` Args:
        :tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): Model tokenizer.
        :qpc_path (str): Path to the saved generated binary file after compilation.

    ``Optional`` Args:
        :prompt (str): Sample prompt for the model text generation. ``Defaults to None``.
        :prompts_txt_file_path (str): Path of the prompt text file. ``Defaults to None``.
        :generation_len (int): Maximum context length for the model during compilation. ``Defaults to None``.
        :device_id (List[int]): Device IDs to be used for execution. If ``len(device_id) > 1``, it enables multiple card setup. If ``None``, auto-device-picker will be used. ``Defaults to None``.
        :enable_debug_logs (bool): If True, it enables debugging logs. ``Defaults to False``.
        :stream (bool): If True, enable streamer, which returns tokens one by one as the model generates them. ``Defaults to True``.
        :Write_io_dir (str): Path to write the input and output files. ``Defaults to None``.
        :automation (bool): If true, it prints input, output, and performance stats. ``Defaults to False``.

    Returns:
        :CloudAI100ExecInfo: Object holding execution output and performance details.

    .. code-block:: python

        import transformers
        import QEfficient
        base_path, onnx_model_path = QEfficient.export(model_name="gpt2")
        qpc_path = QEfficient.compile(onnx_path=onnx_model_path, qpc_path=os.path.join(base_path, "qpc"), num_cores=14, device_group=[0])
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        execinfo = QEfficient.cloud_ai_100_exec_kv(tokenizer=tokenizer, qpc_path=qpc_path, prompt="Hi there!!", device_id=[0])

    """
    batch_size, ctx_len = get_compilation_dims(qpc_path)
    prompt: List[str] = get_input_prompts(prompt, prompts_txt_file_path)
    prompt = fix_prompts(prompt, batch_size)

    prefill_time = []
    decode_perf = []
    total_perf = []
    total_time = []
    generated_texts = []
    generated_ids = []

    for i in range(0, len(prompt), batch_size):
        print("Inference iteration =", i // batch_size)
        execinfo = cloud_ai_100_exec_kv_helper(
            tokenizer=tokenizer,
            prompt=prompt[i : i + batch_size],
            qpc_path=qpc_path,
            device_id=device_id,
            ctx_len=ctx_len,
            generation_len=generation_len,
            enable_debug_logs=enable_debug_logs,
            stream=stream,
            write_io_dir=write_io_dir,
        )
        generated_ids.append(execinfo.generated_ids)
        generated_texts.append(execinfo.generated_texts)
        prefill_time.append(execinfo.prefill_time)
        decode_perf.append(execinfo.decode_perf)
        total_perf.append(execinfo.total_perf)
        total_time.append(execinfo.total_time)

    prefill_time = np.average(prefill_time)
    decode_perf = np.average(decode_perf)
    total_perf = np.average(total_perf)
    total_time = np.average(total_time)

    execinfo = CloudAI100ExecInfo(
        batch_size=batch_size,
        generated_texts=generated_texts,
        generated_ids=generated_ids,
        prefill_time=prefill_time,
        decode_perf=decode_perf,
        total_perf=total_perf,
        total_time=total_time,
    )
    print_latency_stats_kv(
        prompt,
        execinfo=execinfo,
        automation=automation,
    )
    return execinfo
