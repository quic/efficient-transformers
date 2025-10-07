# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from transformers import AutoImageProcessor, PreTrainedTokenizer, PreTrainedTokenizerFast

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import padding_check_and_fix
from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger
from QEfficient.utils.sampler_utils import validate_sampler_inputs


@dataclass
class PerfMetrics:
    """
    Holds all performance metrics

    Args:
        :prefill_time (float): Time for prefilling.
        :decode_perf (float): Decoding performance.
        :total_perf (float): Total performance.
        :total_time (float): Total time.
    """

    prefill_time: float
    decode_perf: float
    total_perf: float
    total_time: float


@dataclass
class CloudAI100ExecInfo:
    """
    Holds all the information about Cloud AI 100 execution

    Args:
        :batch_size (int): Batch size of the QPC compilation.
        :generated_texts (Union[List[List[str]], List[str]]): Generated text(s).
        :generated_ids (Union[List[np.ndarray], np.ndarray]): Generated IDs.
        :perf_metrics (PerfMetrics): Performance metrics.
    """

    batch_size: int
    generated_texts: Union[List[str], List[List[str]]]
    generated_ids: Union[List[np.ndarray], np.ndarray]
    perf_metrics: PerfMetrics

    def __repr__(self):
        return f"Average Prefill time a.k.a TTFT is= {round(self.perf_metrics.prefill_time, 2)} sec\
        \nDecode is= {round(self.perf_metrics.decode_perf * self.batch_size, 2)} tokens/sec\
        \nTotal is= {round(self.perf_metrics.total_perf * self.batch_size, 2)} tokens/sec\
        \nTotal (E2E) inference time is= {round(self.perf_metrics.total_time, 2)} sec"


@dataclass
class CloudAI100ExecInfoNew:
    batch_size: int
    generated_ids: Union[List[np.ndarray], np.ndarray]
    perf_metrics: PerfMetrics

    def __repr__(self):
        return f"Average Prefill time a.k.a TTFT is= {round(self.perf_metrics.prefill_time, 2)} sec\
        \nDecode is= {round(self.perf_metrics.decode_perf * self.batch_size, 2)} token/sec\
        \nTotal is= {round(self.perf_metrics.total_perf * self.batch_size, 2)} token/sec\
        \nTotal (E2E) inference time is= {round(self.perf_metrics.total_time, 2)} sec"


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
    for iname, i_array in inputs.items():
        i_array.tofile(f"{write_io_dir}/{write_io_subdir}/{iname}.raw")
        i_spec = {
            "path": f"{write_io_subdir}/{iname}.raw",
            "io-direction": "in",
            "elem-size": i_array.itemsize,
            "map-to": iname,
        }
        if include_dims:
            i_spec["dims"] = i_array.shape
        io.append(i_spec)
    for o_name, o_array in outputs.items():
        o_array.tofile(f"{write_io_dir}/{write_io_subdir}/{o_name}.raw")
        o_spec = {
            "path": f"{write_io_subdir}/{o_name}.raw",
            "io-direction": "out",
            "elem-size": o_array.itemsize,
            "map-to": o_name,
        }
        if include_dims or o_name.endswith("_RetainedState"):
            o_spec["dims"] = o_array.shape
        io.append(o_spec)
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


def get_compilation_dims(qpc_path: str) -> Tuple[int, int, Optional[int]]:
    """
    Function to fetch compilation dimensions from specializations.json.
    Uses qpc path to compute path to specializations.json.

    Args:
        qpc_path (str): Path to directory comprising generated binary file after compilation.

    Returns:
    :tuple: compilation batch size, compilation context length, compilation full batch size
    """
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
    if compilation_fbs := data["specializations"][0].get("full_batch_size", None):
        compilation_fbs = int(compilation_fbs)
    return compilation_batch_size, compilation_ctx_len, compilation_fbs


def get_input_prompts(prompt: str, prompts_txt_file_path: str) -> List[str]:
    if prompt is None and prompts_txt_file_path is None:
        raise ValueError("Please pass at least one argument either using --prompt or --prompts_txt_file_path")
    if prompts_txt_file_path is not None:
        if prompt is not None:
            logger.warning("Found inputs passed using txt file as well as CLI, taking inputs from given txt file")
        prompt = read_prompts_txt_file(prompts_txt_file_path)
    if isinstance(prompt, str):
        prompt = [prompt]
    return prompt


def fix_prompts(prompt: List[str], batch_size: int, full_batch_size: int = None):
    """
    Adjusts the list of prompts to match the required batch size.

    ``Mandatory`` Args:
        prompt (List[str]): List of input prompts.
        batch_size (int): The batch size to process at a time.

    ``Optional`` Args:
        full_batch_size (Optional[int]): The full batch size if different from batch_size.

    Returns:
        List[str]: Adjusted list of prompts.
    """
    exec_batch_size = full_batch_size if full_batch_size is not None else batch_size

    if len(prompt) < exec_batch_size:
        logger.warning("Number of prompts are less than batch size/full batch size, repeating to required batch size")
        prompt = (prompt * (exec_batch_size // len(prompt) + 1))[:exec_batch_size]
    elif full_batch_size is None and len(prompt) % batch_size != 0:
        logger.warning(
            "Number of prompts are not multiple of batch size, dropping last incomplete batch from given input prompts"
        )
        prompt = prompt[: batch_size * (len(prompt) // batch_size)]

    return prompt


def fix_prompt_to_lora_id_mapping(prompt_to_lora_id_mapping: List[int], batch_size: int, full_batch_size: int = None):
    """
    Adjusts the list of prompt_to_lora_id_mapping to match the required batch size.

    ``Mandatory`` Args:
        prompt_to_lora_id_mapping (Optional[List[int]]): Mapping to associate prompts with their respective LoRA adapter.
        batch_size (int): The batch size to process at a time.

    ``Optional`` Args:
        full_batch_size (Optional[int]): The full batch size if different from batch_size.

    Returns:
        List[int]: Adjusted list of prompt_to_lora_id_mapping.
    """
    exec_batch_size = full_batch_size if full_batch_size is not None else batch_size

    if len(prompt_to_lora_id_mapping) < exec_batch_size:
        logger.warning(
            "Prompt_to_lora_id_mapping are less than batch size/full batch size, repeating to required batch size"
        )
        prompt_to_lora_id_mapping = (
            prompt_to_lora_id_mapping * (exec_batch_size // len(prompt_to_lora_id_mapping) + 1)
        )[:exec_batch_size]
    elif full_batch_size is None and len(prompt_to_lora_id_mapping) % batch_size != 0:
        logger.warning(
            "prompt_to_lora_id_mapping are not multiple of batch size, dropping last incomplete batch from given input prompts"
        )
        prompt_to_lora_id_mapping = prompt_to_lora_id_mapping[
            : batch_size * (len(prompt_to_lora_id_mapping) // batch_size)
        ]

    return prompt_to_lora_id_mapping


def read_prompts_txt_file(prompts_txt_file_path: str):
    prompt = []
    with open(prompts_txt_file_path, "r") as file:
        for line in file:
            prompt.append(line.strip())
    return prompt


def print_latency_stats_kv(prompt, exec_info, automation: bool = False):
    if automation:
        print("input=", prompt)
        print("output=", exec_info.generated_texts)
        print(exec_info)
        return
    print("\n========================= Performance Stats =========================")
    if exec_info.batch_size > 1:
        print("Batch Performance : \n")
    print(exec_info)
    print("=====================================================================")


def calculate_latency(total_decoded_tokens, loop_start, start, end, decode_pause_time=0):
    """
    Method will calculate the latency metrics using the time loops and based on the total decoded token count.

    Args:
        :total_decoded_tokens (int): Number of tokens generated in decode stage.
        :loop_start (float): Start time of decode loop.
        :start (float): Start time.
        :end (float): End time.
        :decode_pause_time (float): Total decode pause time in continuous batching decode stage.

    Returns:
    :tuple: prefill time, decode performance, total performance, total time
    """
    prefill_time = loop_start - start + decode_pause_time
    decode_perf = (total_decoded_tokens) / (end - loop_start - decode_pause_time)
    total_perf = (total_decoded_tokens) / (end - start)
    total_time = end - start
    return prefill_time, decode_perf, total_perf, total_time


def cloud_ai_100_exec_kv(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    lang_qpc_path: str,
    processor: Optional[AutoImageProcessor] = None,
    vision_qpc_path: Optional[str] = None,
    images: Optional[str] = None,
    prompt: Optional[str] = None,
    prompts_txt_file_path: Optional[str] = None,
    device_id: Optional[List[int]] = None,
    generation_len: Optional[int] = None,
    enable_debug_logs: bool = False,
    stream: bool = True,
    write_io_dir: Optional[str] = None,
    automation=False,
    iteration: int = 1,
    prompt_to_lora_id_mapping: Optional[List[int]] = None,
    is_tlm: bool = False,
    include_sampler: bool = False,
    return_pdfs: bool = False,
    sampling_params: Optional[Dict[str, Any]] = None,
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
        :iteration (int): Number of iterations to run the inference. ``Defaults to 1``.
        :prompt_to_lora_id_mapping (List[int]): Mapping to associate prompts with their respective LoRA adapter.
        :include_sampler (bool, default=False): Enable/Disable sampling of next tokens.
        :return_pdfs (bool, default=False): Return probability distributions along with sampled
        next tokens. For Speculative Decoding Target Language Model,
        `return_pdfs`=True always. Otherwise, `return_pdfs`=True for Speculative
        Decoding Draft Language Model and `return_pdfs`=False for regular model.
        sampling_params (Dict[str, Any], default=None): A dictionary of sampling parameters supported by the QAIC backend.
        The dictionary should contain the following keys:
        `repetition_penalties`, `presence_penalties`, `temperatures`, `top_ks`, `top_ps`,
        `min_ps`, and `random_numbers`. Each value should be a numpy array of shape (batch_size, 1).

    Returns:
        :CloudAI100ExecInfo: Object holding execution output and performance details.

    .. code-block:: python

        import transformers
        import QEfficient
        base_path, onnx_model_path = QEfficient.export(model_name="gpt2")
        qpc_path = QEfficient.compile(onnx_path=onnx_model_path, qpc_path=os.path.join(base_path, "qpc"), num_cores=14, device_group=[0])
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        exec_info = QEfficient.cloud_ai_100_exec_kv(tokenizer=tokenizer, qpc_path=qpc_path, prompt="Hi there!!", device_id=[0])

    """
    batch_size, ctx_len, full_batch_size = get_compilation_dims(lang_qpc_path)
    prompt: List[str] = get_input_prompts(prompt, prompts_txt_file_path)
    prompt = fix_prompts(prompt, batch_size, full_batch_size)
    if prompt_to_lora_id_mapping is not None:
        prompt_to_lora_id_mapping = fix_prompt_to_lora_id_mapping(
            prompt_to_lora_id_mapping, batch_size, full_batch_size
        )
    generate_text = TextGeneration(
        tokenizer=tokenizer,
        processor=processor,
        lang_qpc_path=lang_qpc_path,
        vision_qpc_path=vision_qpc_path,
        device_id=device_id,
        ctx_len=ctx_len,
        enable_debug_logs=enable_debug_logs,
        write_io_dir=write_io_dir,
        full_batch_size=full_batch_size,
        is_tlm=is_tlm,
        include_sampler=include_sampler,
        return_pdfs=return_pdfs,
        sampling_params=sampling_params,
    )
    if full_batch_size is None:
        exec_info = [
            generate_text.generate(
                prompt=prompt[i : i + batch_size],
                generation_len=generation_len,
                stream=stream,
                prompt_to_lora_id_mapping=prompt_to_lora_id_mapping,
            )
            for i in range(0, len(prompt), batch_size)
        ]
        prefill_time = np.average([info.perf_metrics.prefill_time for info in exec_info])
        decode_perf = np.average([info.perf_metrics.decode_perf for info in exec_info])
        total_perf = np.average([info.perf_metrics.total_perf for info in exec_info])
        total_time = np.average([info.perf_metrics.total_time for info in exec_info])
        generated_texts = [info.generated_texts for info in exec_info]
        generated_ids = [info.generated_ids for info in exec_info]

        exec_info = CloudAI100ExecInfo(
            batch_size=batch_size,
            generated_texts=generated_texts,
            generated_ids=generated_ids,
            perf_metrics=PerfMetrics(prefill_time, decode_perf, total_perf, total_time),
        )
    else:
        exec_info = generate_text.generate(
            prompt=prompt,
            images=images,
            generation_len=generation_len,
            prompt_to_lora_id_mapping=prompt_to_lora_id_mapping,
        )

            exec_info = CloudAI100ExecInfo(
                batch_size=batch_size,
                generated_texts=generated_texts,
                generated_ids=generated_ids,
                perf_metrics=PerfMetrics(prefill_time, decode_perf, total_perf, total_time),
            )
        else:
            exec_info = generate_text.generate(
                prompt=prompt, generation_len=generation_len, prompt_to_lora_id_mapping=prompt_to_lora_id_mapping
            )

        print_latency_stats_kv(prompt, exec_info=exec_info, automation=automation)

    # TODO: Need to handle the case where exec_info if given for n iterations
    return exec_info


class QEffTextGenerationBase:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        lang_qpc_path: str,
        processor: Optional[AutoImageProcessor] = None,
        vision_qpc_path: Optional[str] = None,
        full_batch_size: Optional[int] = None,
        ctx_len: Optional[int] = None,
        device_id: Optional[List[int]] = None,
        enable_debug_logs: bool = False,
        write_io_dir: Optional[str] = None,
        is_tlm: Optional[int] = None,
        include_sampler: bool = False,
        return_pdfs: bool = False,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ctx_len = ctx_len
        self._write_io_dir = write_io_dir
        self.is_tlm = is_tlm
        self.return_pdfs = return_pdfs
        self.sampling_params = sampling_params

        # Load QPC
        self._lang_session = None
        self._vision_session = None
        if not lang_qpc_path:
            raise TypeError("Please run compile API for language model first!")
        self._lang_session = QAICInferenceSession(lang_qpc_path, device_id, activate=False)
        if vision_qpc_path:
            self._vision_session = QAICInferenceSession(vision_qpc_path, device_id, activate=False)

        # Validate sampler inputs for On-Device Sampling
        self.include_sampler = validate_sampler_inputs(
            session_inputs=set(self._lang_session.input_names), include_sampler=include_sampler
        )

        # Fetch the variables from the QPC
        self._vocab_size = self._fetch_vocab_size()  # Fetch Vocab size
        self.batch_size, self._prefill_seq_len = self._fetch_batch_size_prefill_seq_len()
        self._decode_seq_len = self._fetch_decode_seq_len()
        self.full_batch_size = (
            full_batch_size if full_batch_size else self._fetch_full_batch_size()
        )  # Check and fetch full batch size if CB is enabled

        # Initialize the storage variables.
        self.batch_index = None

        # Variables to be re-initialized for every run
        # These parameters will be initialized in initialize_lora_id_mapping method
        self._prompt_to_lora_id_mapping_prefill = None
        self._prompt_to_lora_id_mapping_decode = None
        # These parameters will be initialized to np arrays in initialize_decode_inputs method
        self.generated_ids = None
        self.decode_input_ids = None
        self.decode_pos_ids = None
        self.generation_len = None

        self.tokenizer = tokenizer
        self.processor = processor
        self._set_tokenizer_params()  # set tokenizer params
        # Skip inputs/outputs
        if self._vision_session:
            self._vision_session.skip_buffers(
                [
                    x
                    for x in self._vision_session.input_names + self._vision_session.output_names
                    if x.startswith("past_") or x.endswith("_RetainedState")
                ]
            )
        self._lang_session.skip_buffers(
            [
                x
                for x in self._lang_session.input_names + self._lang_session.output_names
                if x.startswith("past_") or x.endswith("_RetainedState")
            ]
        )

    def _set_tokenizer_params(self):
        """
        Sets the tokenizer parameters for the model.
        """
        if self.tokenizer.padding_side != "right":
            logger.warning("Please use padding_side='right' while initializing the tokenizer")
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _fetch_full_batch_size(
        self,
    ):
        """
        Fetches the full batch size from the session's bindings or allowed shapes.

        Returns:
        full_batch_size: The full batch size fetched from the session's bindings or allowed shapes. If "batch_index" is not
        in the session's binding index map, full_batch_size will be None.

        """
        full_batch_size = None
        if "batch_index" in self._lang_session.binding_index_map:
            if self._lang_session.allowed_shapes:
                full_batch_size, _ = [
                    x[self._lang_session.binding_index_map["batch_index"]][1][0]
                    for x in self._lang_session.allowed_shapes
                ]
            else:
                full_batch_size, _ = self._lang_session.bindings[
                    self._lang_session.binding_index_map["batch_index"]
                ].dims
        return full_batch_size

    def _fetch_batch_size_prefill_seq_len(
        self,
    ):
        """
        Fetches the batch size and prefill sequence length from the session's bindings or allowed shapes.

        Returns:
            batch_size: The batch size fetched from the session's bindings or allowed shapes.
            prefill_seq_len: The prefill sequence length fetched from the session's bindings or allowed shapes.
        """
        if self._lang_session.allowed_shapes:
            batch_size = max(
                [x[self._lang_session.binding_index_map["input_ids"]][1][0] for x in self._lang_session.allowed_shapes]
            )
            prefill_seq_len = max(
                [x[self._lang_session.binding_index_map["input_ids"]][1][1] for x in self._lang_session.allowed_shapes]
            )
        else:
            batch_size, prefill_seq_len = self._lang_session.bindings[
                self._lang_session.binding_index_map["input_ids"]
            ].dims
        return batch_size, prefill_seq_len

    def _fetch_decode_seq_len(
        self,
    ):
        """
        Fetches the decode sequence length from the session's bindings or allowed shapes.

        Returns:
            decode_seq_len: The decode sequence length fetched from the session's bindings or allowed shapes.
        """
        decode_seq_len = None
        if self._lang_session.allowed_shapes:
            decode_seq_len = min(
                [x[self._lang_session.binding_index_map["input_ids"]][1][1] for x in self._lang_session.allowed_shapes]
            )
        return decode_seq_len

    def _fetch_vocab_size(
        self,
    ):
        """
        Fetches the vocabulary size from the session's allowed shapes.
        Returns:
            vocab_size: The vocabulary size fetched from the session's allowed shapes.
        """
        key = (
            "probs"
            if self.include_sampler and self.return_pdfs
            else "next_tokens"
            if self.include_sampler
            else "logits"
        )
        if self._lang_session.allowed_shapes:
            return [x[self._lang_session.binding_index_map[key]] for x in self._lang_session.allowed_shapes][0][1][2]

        return self._lang_session.bindings[self._lang_session.binding_index_map[key]].dims[2]

    def _fetch_generation_len(self, generation_len, max_gen_len):
        """
        Fetches the generation length for the model.
        Args:
            generation_len: The generation length provided. If None, the method uses max_gen_len.
            max_gen_len: The maximum allowed generation length.

        Returns:
            generation_len: The final generation length, which is either the provided generation_len (if it is not None and not greater than max_gen_len) or max_gen_len.
        """

        if generation_len is None:
            if self._ctx_len is None:
                raise ValueError("At least one of ctx_len or generation_len is needed")
            generation_len = max_gen_len
        elif generation_len > max_gen_len:
            logger.warning(
                "Passed generation_len is greater than allowed length. "
                "Make sure this model supports sliding window, such as Mistral"
            )
        if generation_len <= 0:
            raise ValueError("generation length should be greater than zero")
        return generation_len

    def prepare_decode_inputs(self):
        """
        This function creates the decode inputs.

        Returns:
            dict: The decode inputs.
        """
        batch_size = self.full_batch_size if self.full_batch_size is not None else self.batch_size
        decode_inputs = {}
        if self.is_tlm:
            position_ids = np.full((batch_size, self._decode_seq_len), -1, dtype=np.int64)
            position_ids[:, -1] = self.decode_pos_ids.flatten()
            input_ids = np.zeros((batch_size, self._decode_seq_len), dtype=np.int64)
            input_ids[:, -1] = self.decode_input_ids.flatten()
            decode_inputs["input_ids"] = input_ids
            decode_inputs["position_ids"] = position_ids
            decode_inputs["num_logits_to_keep"] = np.zeros((self._decode_seq_len, 1))
        else:
            decode_inputs["input_ids"] = self.decode_input_ids
            decode_inputs["position_ids"] = self.decode_pos_ids
        if self.batch_index is not None:
            decode_inputs["batch_index"] = self.batch_index
        if self.include_sampler:
            decode_inputs["last_accepted_output_tokens"] = decode_inputs["input_ids"]
            for op in Constants.SAMPLER_OPS:
                if self.batch_index is not None:
                    decode_inputs[op] = self.sampling_params[op][self.batch_index.flatten()]
                else:
                    decode_inputs[op] = self.sampling_params[op]

        if self._prompt_to_lora_id_mapping_decode:
            if self.full_batch_size:
                first_batch_lora_ids = [self._prompt_to_lora_id_mapping_decode[i] for i in range(self.full_batch_size)]
                decode_inputs["lora_ids"] = np.array(first_batch_lora_ids, dtype=np.int64).reshape(
                    self.full_batch_size, 1
                )
            else:
                batch_lora_ids = [self._prompt_to_lora_id_mapping_decode.popleft() for i in range(self.batch_size)]
                decode_inputs["lora_ids"] = np.array(batch_lora_ids, dtype=np.int64).reshape(self.batch_size, 1)

        return decode_inputs

    def _fetch_next_token_id(self, outputs):
        """
        Fetches the next token ID from the model's output.

        Args:
            outputs (dict): A dictionary containing the model's output.

        Returns:
            numpy.ndarray: An array of the next token IDs for each sequence in the batch.
        """
        if self.include_sampler:
            if self.return_pdfs:
                return outputs["probs"].argmax(2)
            else:
                return outputs["next_tokens"].reshape(outputs["next_tokens"].shape[0], outputs["next_tokens"].shape[1])
        else:
            logits = outputs["logits"]
            if len(logits.shape) == 2:
                logits = np.expand_dims(logits, 1)
            return logits.argmax(2)

    def initialize_decode_inputs(self, num_prompts, execution_batch_size, max_gen_length):
        """
        Initialize np arrays for storing the prefill output for all the decode batch size.
        """
        self.generated_ids = np.full((num_prompts, max_gen_length), self.tokenizer.pad_token_id)
        self.decode_input_ids = np.zeros((execution_batch_size, 1), np.int64)
        self.decode_pos_ids = np.zeros((execution_batch_size, 1), np.int64)
        self.generation_len = np.zeros((execution_batch_size, 1), np.int64)

    def initialize_lora_id_mapping(self, prompt_to_lora_id_mapping):
        """
        Initializes the LoRA ID mapping for prefill and decode phases.

        Args:
            prompt_to_lora_id_mapping (list): An iterable containing the mapping of prompts to LoRA IDs.

        Sets:
            self._prompt_to_lora_id_mapping_prefill (deque): A deque containing the prompt to LoRA ID mapping for the prefill phase.
            self._prompt_to_lora_id_mapping_decode (iterable or deque): The prompt to LoRA ID mapping for the decode phase. If full_batch_size is set, it uses the original iterable; otherwise, it converts it to a deque.
        """
        self._prompt_to_lora_id_mapping_prefill = deque(prompt_to_lora_id_mapping)
        if self.full_batch_size:
            self._prompt_to_lora_id_mapping_decode = prompt_to_lora_id_mapping
        else:
            self._prompt_to_lora_id_mapping_decode = deque(prompt_to_lora_id_mapping)

    def update_decode_input(self, outputs, position_ids, generation_len, decode_batch_id=None):
        """
        Updates the decode input with the generated values.
        Args:
            outputs (dict): The outputs of the model.
            position_ids (array): The position IDs.
            generation_len (int): The generation length.
            decode_batch_id (int, optional): The decode batch ID. If None, all values are updated. Defaults to None.

        Returns:
            next_token_id (array): The next token ID.
        """
        next_token_id = self._fetch_next_token_id(outputs)

        # Store the generated values.
        self.decode_input_ids[decode_batch_id or slice(None)] = next_token_id
        self.decode_pos_ids[decode_batch_id or slice(None)] = position_ids
        self.generated_ids[decode_batch_id or slice(None), 0] = next_token_id.squeeze(1)
        self.generation_len[decode_batch_id or slice(None)] = generation_len
        return next_token_id

    def run_prefill_for_all_inputs(self, image_queue, prompt_queue, generation_len):
        """
        Runs prefill for all inputs in the prompt queue and updates the decode input.

        Method iterates over the full batch size and for each decode batch ID, it pops the next prompt from the queue.  It then runs prefill for the next prompt and updates the decode input with the outputs.

        Args:
            prompt_queue (deque): The queue of prompts.
            generation_len (int): The generation length.

        """
        next_prompt = None
        next_image = None
        for decode_batch_id in range(self.full_batch_size):
            if prompt_queue:
                next_prompt = prompt_queue.popleft()
            if image_queue:
                next_image = image_queue.popleft()

            # run prefill for num_chunks
            outputs, position_ids, generation_len = self.run_prefill(
                next_prompt,
                next_image,
                generation_len,
                decode_batch_id=np.array(decode_batch_id, dtype=np.int64).reshape(1, 1),
            )

            _ = self.update_decode_input(outputs, position_ids, generation_len, decode_batch_id)

    def _set_output_buffers(self, batch_size: int = 1, sequence_length: int = 1):
        """
        Sets the sizes of the output buffers.

        Args:
            batch_size (int): The batch size.
        """
        if self.include_sampler:
            if self.return_pdfs:
                probs_out_placeholder = np.zeros((batch_size, sequence_length, self._vocab_size), dtype=np.float32)
                self._lang_session.set_buffers({"probs": probs_out_placeholder})
            next_tokens_out_placeholder = np.zeros((batch_size, sequence_length, 1), dtype=np.int64)
            self._lang_session.set_buffers({"next_tokens": next_tokens_out_placeholder})
        else:
            logits_out_placeholder = np.zeros((batch_size, sequence_length, self._vocab_size), dtype=np.float32)
            self._lang_session.set_buffers({"logits": logits_out_placeholder})

            if self._vision_session:
                vision_embeds_out_placeholder = np.zeros((2448, 5120), dtype=np.float16)
                self._vision_session.set_buffers({"vision_embeds": vision_embeds_out_placeholder})

    def prepare_vision_language_inputs(self, prompt, image_url):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_url},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        return inputs

    def run_prefill(
        self,
        prompt: str,
        image: Optional[str] = None,
        generation_len: Optional[int] = None,
        prefill_logit_bs=1,
        decode_batch_id=None,
    ):
        """
        Runs prefill for a given prompt and generation length.

        This method tokenize the prompt and calculates the padded length and number of chunks. Calculates the
        maximum generation length and fetches the generation length. If a batch index for prefill is provided, it sets the batch index in the inputs. The method then runs prefill for each chunk and updates the inputs and outputs.

        Args:
            prompt (str): The prompt for which to run prefill.
            generation_len (int): The generation length.
            prefill_logit_bs (int, optional): The prefill logit batch size. Defaults to 1.

        Returns:
            outputs (dict): The outputs of the prefill.
            position_ids (array): The position IDs.
            generation_len (int): The generation length.
        """

        # Run prefill
        if image:
            inputs = self.prepare_vision_language_inputs(prompt, image)
        else:
            inputs = self.tokenizer(prompt, return_tensors="np", padding=True)

        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self._prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * self._prefill_seq_len  # Convert to a multiple of prompt_len

        # Initialize variables specific to request
        # Calculate the max generation length.
        max_gen_len = self._ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        # Set the prefill output buffers
        self._set_output_buffers(batch_size=prefill_logit_bs, sequence_length=1)

        vision_inputs = {}
        vision_outputs = {}
        if image:
            pad_token_id = 1
            input_ids_length = inputs["input_ids"].shape[1]
            num_chunks = -(input_ids_length // -self._prefill_seq_len)  # ceil divide without float
            padded_len = num_chunks * self._prefill_seq_len  # Convert to a multiple of prompt_len

            inputs["input_ids"] = torch.nn.functional.pad(
                inputs["input_ids"],
                (0, padded_len - input_ids_length),
                "constant",
                pad_token_id,
            )
            inputs["attention_mask"] = torch.nn.functional.pad(
                inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
            )
            if "cross_attention_mask" in inputs:
                inputs["cross_attention_mask"] = torch.nn.functional.pad(
                    inputs["cross_attention_mask"], (0, 0, 0, 0, 0, padded_len - input_ids_length)
                )

            for k, v in inputs.items():
                inputs[k] = np.array(v)

            vision_inputs = {
                k: v for k, v in inputs.items() if k in {"pixel_values", "aspect_ratio_ids", "aspect_ratio_mask"}
            }
            if vision_inputs:
                vision_inputs["pixel_values"] = vision_inputs["pixel_values"].astype("float16")

            # Run vision prefill
            if vision_inputs:
                self._vision_session.activate()
                vision_outputs = self._vision_session.run(vision_inputs)
                self._vision_session.deactivate()
        else:
            inputs = self.tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
            inputs.pop("token_type_ids", None)

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
        lang_inputs["position_ids"] = np.where(
            lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
        )  # Need to use -1 as position_ids for invalid tokens

        # not_mllama = hasattr(self.model.config, "model_type") and self.model.config.model_type != "mllama"
        # if not_mllama:
        if image:
            lang_inputs["image_idx"] = np.array([[0]])

        self._lang_session.activate()
        self._lang_session.set_buffers(vision_outputs)

        if decode_batch_id is not None:
            lang_inputs["batch_index"] = decode_batch_id
        if self.is_tlm:
            lang_inputs["num_logits_to_keep"] = np.zeros((1, 1))
        if self.include_sampler:
            lang_inputs["last_accepted_output_tokens"] = lang_inputs["input_ids"]
            for op in Constants.SAMPLER_OPS:
                if decode_batch_id is not None:
                    lang_inputs[op] = self.sampling_params[op][decode_batch_id.flatten()]
                else:
                    lang_inputs[op] = self.sampling_params[op]

        if self._prompt_to_lora_id_mapping_prefill:
            if self.full_batch_size:
                lang_inputs["lora_ids"] = np.array(
                    self._prompt_to_lora_id_mapping_prefill.popleft(), dtype=np.int64
                ).reshape(1, 1)
            else:
                batch_lora_ids = [self._prompt_to_lora_id_mapping_prefill.popleft() for i in range(self.batch_size)]
                lang_inputs["lora_ids"] = np.array(batch_lora_ids, dtype=np.int64).reshape(self.batch_size, 1)

        # Run language prefill

        for i in range(num_chunks):
            chunk_inputs = lang_inputs.copy()
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][
                :, i * self._prefill_seq_len : (i + 1) * self._prefill_seq_len
            ]
            if self.include_sampler:
                chunk_inputs["last_accepted_output_tokens"] = chunk_inputs["input_ids"]
            outputs = self._lang_session.run(chunk_inputs)
            if image:
                chunk_inputs["image_idx"] = outputs["image_idx_output"]
            if self._write_io_dir is not None:
                write_io_files(inputs, outputs, self._write_io_dir, "prefill", "aic_batch_io", True, False)

        # Skip inputs/outputs again
        self._lang_session.skip_buffers(
            [
                x
                for x in self._lang_session.input_names + self._lang_session.output_names
                if x.startswith("past_") or x.endswith("_RetainedState")
            ]
        )
        self._lang_session.deactivate()

        return (
            outputs,
            position_ids,
            generation_len,
        )

    def run_continuous_batching_decode(self, prompt_queue, generation_len):
        """
        Runs continuous batching decode for the given prompt queue and generation length.

        Method sets up the initial conditions for decoding and preparing the decode inputs. Then enters a loop that continues as long as there are prompts in the queue or any decoding is ongoing. In each iteration of the loop, it runs the session with the current decode inputs, prepares the inputs for the next iteration and updates the decode inputs. If a prompt has been fully decoded, it runs prefill for the next prompt in the queue if available.

        Args:
            prompt_queue (deque): The queue of prompts to be decoded.
            generation_len (int): The generation length.

        """

        # Set output placeholders for decode
        self._set_output_buffers(
            batch_size=self.full_batch_size,
            sequence_length=self._decode_seq_len,
        )

        # Generate flag for tracking progress for each batch ID
        current_decode_ongoing = np.full((self.full_batch_size, 1), True)

        # Generate an array for maintaining the tokens generated in each batch ID
        generated_id_current_index = np.ones((self.full_batch_size, 1), np.int64)

        # Generate a batch ID map for mapping the batch ID if input > full_batch_size.
        # This ID map will be used for storing all generated tokens
        batch_id_map = {i: i for i in range(self.full_batch_size)}
        decode_pause_time = 0
        # Prepare decode inputs inputs.
        decode_inputs = self.prepare_decode_inputs()

        while prompt_queue or current_decode_ongoing.any():
            self._lang_session.activate()
            outputs = self._lang_session.run(decode_inputs)

            # Prepare inputs for next iteration
            next_token_id = self._fetch_next_token_id(outputs)

            for decode_batch_id in range(self.full_batch_size):
                if (
                    next_token_id[decode_batch_id, -1] == self.tokenizer.eos_token_id
                    or generated_id_current_index[decode_batch_id] >= self.generation_len[decode_batch_id]
                ):
                    if prompt_queue:
                        start = perf_counter()
                        # run prefill for next prompt input.
                        outputs, position_ids, generation_len = self.run_prefill(
                            prompt=prompt_queue.popleft(),
                            generation_len=generation_len,
                            decode_batch_id=np.array(decode_batch_id, dtype=np.int64).reshape(1, 1),
                        )

                        new_token_id = self.update_decode_input(outputs, position_ids, generation_len, decode_batch_id)

                        batch_id_map[decode_batch_id] = max(batch_id_map.values()) + 1
                        self.generated_ids[batch_id_map[decode_batch_id], 0] = new_token_id.squeeze(1)
                        generated_id_current_index[decode_batch_id] = 1

                        self._set_output_buffers(
                            batch_size=self.full_batch_size,
                            sequence_length=self._decode_seq_len,
                        )
                        decode_pause_time += perf_counter() - start

                        if self._prompt_to_lora_id_mapping_decode:
                            decode_inputs["lora_ids"][decode_batch_id] = self._prompt_to_lora_id_mapping_decode[
                                batch_id_map[decode_batch_id]
                            ]

                    else:
                        current_decode_ongoing[decode_batch_id] = False
                else:
                    # If the generated sequence is valid and within generation len prepare for next decode
                    decode_inputs["input_ids"][decode_batch_id, -1] = next_token_id[decode_batch_id, -1]
                    decode_inputs["position_ids"][decode_batch_id, -1] += 1
                    self.generated_ids[batch_id_map[decode_batch_id], generated_id_current_index[decode_batch_id]] = (
                        next_token_id[decode_batch_id, -1]
                    )
                    if self.include_sampler:
                        decode_inputs["last_accepted_output_tokens"] = decode_inputs["input_ids"]

                    generated_id_current_index[decode_batch_id] += 1

        self._lang_session.deactivate()

        return decode_pause_time

    def run_decode(
        self, decode_inputs, generation_len, automation, streamer: Optional[transformers.TextStreamer] = None
    ):
        """
        Default method for running decode. Executes the decoding process for a given set of inputs and a specified generation length.

        Enters a loop that continues until all sequences are finished or the maximum generation length is reached. In each iteration, it runs the session with the decode inputs, prepares the inputs for the next iteration and checks if all sequences are finished.

        Args:
            decode_inputs (dict): The initial inputs for decoding. This should be a dictionary containing 'input_ids' and 'position_ids'.
            generation_len (int): Max allowed length for generating tokens. The decoding process will be terminated  when generation length is reached.
            streamer (transformers.TextStreamer): TextStreamer object to print decoded tokens to console.
        Returns:
            num_token (int): The number of tokens processed in the decoding process.
        """
        if self.is_tlm:
            logits_out_placeholder = np.zeros(
                (self.batch_size, self._decode_seq_len, self._vocab_size), dtype=np.float32
            )
            self._lang_session.set_buffers({"logits": logits_out_placeholder})
        finished_sequences = decode_inputs["input_ids"] == self.tokenizer.eos_token_id
        num_token = 0
        self._lang_session.activate()
        for num_token in range(1, generation_len):
            if streamer:
                streamer.put(decode_inputs["input_ids"][0])
            outputs = self._lang_session.run(decode_inputs)

            if self._write_io_dir is not None:
                write_io_files(decode_inputs, outputs, self._write_io_dir, "decode", "aic_batch_io", True, False)
                self._write_io_dir = None

            # Prepare inputs for next iteration
            decode_inputs["input_ids"] = self._fetch_next_token_id(outputs)
            decode_inputs["position_ids"][:, -1] += 1
            self.generated_ids[:, num_token] = decode_inputs["input_ids"][:, -1]
            finished_sequences |= decode_inputs["input_ids"] == self.tokenizer.eos_token_id
            if self.include_sampler:
                decode_inputs["last_accepted_output_tokens"] = decode_inputs["input_ids"]

            if finished_sequences.all() and not automation:
                break
        self._lang_session.deactivate()
        return num_token

    def generate_decode_stream(self, decode_inputs, generation_len, automation):
        """
        Generator method for yielding decode tokens. Executes the decoding process for a given set of inputs and a specified generation length.

        Enters a loop that continues until all sequences are finished or the maximum generation length is reached. In each iteration, it runs the session with the decode inputs, prepares the inputs for the next iteration and checks if all sequences are finished.

        Args:
            decode_inputs (dict): The initial inputs for decoding. This should be a dictionary containing 'input_ids' and 'position_ids'.
            generation_len (int): Max allowed length for generating tokens. The decoding process will be terminated  when generation length is reached.

        Yields:
            token_id (int): The token generated in the decoding process.
        """
        finished_sequences = decode_inputs["input_ids"] == self.tokenizer.eos_token_id
        self._lang_session.activate()
        for num_token in range(1, generation_len):
            yield decode_inputs["input_ids"]
            outputs = self._lang_session.run(decode_inputs)

            if self._write_io_dir is not None:
                write_io_files(decode_inputs, outputs, self._write_io_dir, "decode", "aic_batch_io", True, False)
                self._write_io_dir = None

            # Prepare inputs for next iteration
            decode_inputs["input_ids"] = outputs["logits"].argmax(2)
            decode_inputs["position_ids"] += 1
            self.generated_ids[:, num_token] = decode_inputs["input_ids"].squeeze(1)
            finished_sequences |= decode_inputs["input_ids"] == self.tokenizer.eos_token_id

            if finished_sequences.all() and not automation:
                break
        self._lang_session.deactivate()
        yield decode_inputs["input_ids"]  # yield the last token


class TextGeneration:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        lang_qpc_path: str,
        processor: Optional[AutoImageProcessor] = None,
        vision_qpc_path: Optional[str] = None,
        full_batch_size: Optional[int] = None,
        ctx_len: Optional[int] = None,
        device_id: Optional[List[int]] = None,
        enable_debug_logs: bool = False,
        write_io_dir: Optional[str] = None,
        is_tlm: bool = False,
        include_sampler: bool = False,
        return_pdfs: bool = False,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._qaic_model = QEffTextGenerationBase(
            tokenizer=tokenizer,
            lang_qpc_path=lang_qpc_path,
            processor=processor,
            vision_qpc_path=vision_qpc_path,
            full_batch_size=full_batch_size,
            ctx_len=ctx_len,
            device_id=device_id,
            enable_debug_logs=enable_debug_logs,
            write_io_dir=write_io_dir,
            is_tlm=is_tlm,
            include_sampler=include_sampler,
            return_pdfs=return_pdfs,
            sampling_params=sampling_params,
        )
        self._full_batch_size = self._qaic_model.full_batch_size
        self._tokenizer = self._qaic_model.tokenizer
        self._processor = self._qaic_model.processor
        self._ctx_len = ctx_len
        self._perf_metrics = None
        self._prompt_queue = None
        self._image_queue = None
        self._text_streamer = None

    @property
    def perf_metrics(self):
        return self._perf_metrics

    def _setup_model_execution_inputs(
        self,
        prompt: List[str],
        images: Optional[List[str]] = None,
        generation_len: Optional[int] = None,
        prompt_to_lora_id_mapping: Optional[List[int]] = None,
    ):
        """
        This method should be called to set/reset inputs
        Args:
            :prompt (List[str]): prompts for the model text generation
            :generation_len (Optional[int], optional): Number of tokens to be generated.
            :prompt_to_lora_id_mapping (Optional[List[int]], optional): Mapping to associate prompts with their respective LoRA adapter.
        """
        execution_batch_size = (
            self._full_batch_size if self._full_batch_size is not None else self._qaic_model.batch_size
        )
        max_gen_length = self._ctx_len if not generation_len else max(self._ctx_len, generation_len)

        # Create a prompt queue.
        self._prompt_queue = deque(prompt)
        if images:
            self._image_queue = deque(images)
        # Initialize np arrays for storing the prefill output for all the decode batch size.
        num_prompts = len(self._prompt_queue)

        if prompt_to_lora_id_mapping:
            self._qaic_model.initialize_lora_id_mapping(prompt_to_lora_id_mapping)

        self._qaic_model.initialize_decode_inputs(num_prompts, execution_batch_size, max_gen_length)

    def _regular_model_execution(
        self,
        prompt: List[str],
        generation_len: Optional[int] = None,
        stream: Optional[bool] = True,
        automation: Optional[bool] = False,
        prompt_to_lora_id_mapping: Optional[List[int]] = None,
    ):
        """
        Executes the model in regular mode.
        This method runs the prefill, prepares the decode inputs, and then runs the decode. The generated texts are decoded and optionally streamed. Latency metrics are calculated and returned.
        Args:
            :prompt (List[str]): The list of prompts for the model.
            :generation_len (Optional[int], optional): The generation length.
            :stream (Optional[bool], optional): Boolean flag to enable stream output to console.
            :prompt_to_lora_id_mapping (Optional[List[int]], optional): Mapping to associate prompts with their respective LoRA adapter.

        Returns:
        :tuple: A tuple containing performance metrics and generated texts.

        """
        self._setup_model_execution_inputs(
            prompt=prompt, generation_len=generation_len, prompt_to_lora_id_mapping=prompt_to_lora_id_mapping
        )
        if stream and self._text_streamer is None:
            self._text_streamer = transformers.TextStreamer(self._tokenizer)
        start = perf_counter()
        outputs, position_ids, generation_len = self._qaic_model.run_prefill(
            prompt=prompt, generation_len=generation_len, prefill_logit_bs=self._qaic_model.batch_size
        )
        self._qaic_model.update_decode_input(outputs, position_ids, generation_len)

        decode_inputs = self._qaic_model.prepare_decode_inputs()

        loop_start = perf_counter()  # Start decode loop timer
        num_token = self._qaic_model.run_decode(decode_inputs, generation_len, automation, self._text_streamer)
        end = perf_counter()
        generated_texts = self._tokenizer.batch_decode(self._qaic_model.generated_ids, skip_special_tokens=True)

        total_decode_tokens = num_token
        prefill_time, decode_perf, total_perf, total_time = calculate_latency(
            total_decode_tokens, loop_start, start, end
        )
        self._perf_metrics = PerfMetrics(prefill_time, decode_perf, total_perf, total_time)
        return self._perf_metrics, generated_texts

    def _continuous_batching_execution(
        self,
        prompt: List[str],
        images: Optional[List[str]] = None,
        generation_len: Optional[int] = None,
        prompt_to_lora_id_mapping: Optional[List[int]] = None,
    ):
        """
        Executes the model using continuous batching.
        This method handles the execution of the model when continuous batching is enabled. It runs the prefill step for all inputs, performs continuous batching decode, and then decodes the generated texts. The texts are optionally streamed. Latency metrics are calculated and returned.

        Args:
            :prompt (List[str]): The list of prompts for the model.
            :generation_len (Optional[int], optional): The generation length.
            :prompt_to_lora_id_mapping (Optional[List[int]], optional): Mapping to associate prompts with their respective LoRA adapter.

        Returns:
        :tuple: A tuple containing performance metrics and generated texts.
        """
        self._setup_model_execution_inputs(prompt, images, generation_len, prompt_to_lora_id_mapping)
        self._qaic_model.batch_index = np.arange(self._full_batch_size).reshape(-1, 1)
        start = perf_counter()
        self._qaic_model.run_prefill_for_all_inputs(self._image_queue, self._prompt_queue, generation_len)

        loop_start = perf_counter()  # Start decode loop timer
        decode_pause_time = self._qaic_model.run_continuous_batching_decode(self._prompt_queue, generation_len)
        end = perf_counter()

        generated_texts = self._tokenizer.batch_decode(self._qaic_model.generated_ids, skip_special_tokens=True)

        total_decode_tokens = sum(
            np.sum(self._qaic_model.generated_ids[i] != self._tokenizer.pad_token_id) - 1 for i in range(len(prompt))
        )
        prefill_time, decode_perf, total_perf, total_time = calculate_latency(
            total_decode_tokens, loop_start, start, end, decode_pause_time
        )
        prefill_time /= len(prompt)  # Average prefill time for continuous batching
        self._perf_metrics = PerfMetrics(prefill_time, decode_perf, total_perf, total_time)
        return self._perf_metrics, generated_texts

    def generate_stream_tokens(
        self,
        prompt: List[str],
        generation_len: Optional[int] = None,
        automation: Optional[bool] = False,
        prompt_to_lora_id_mapping: Optional[List[int]] = None,
    ):
        """
        Executes the model for a given list of prompts and a specified generation length.
        This method runs the prefill, prepares the decode inputs, and then runs the decode. The tokens are decoded and streamed as they are generated. Latency metrics are calculated and can be retrieved
        after all tokens are streamed.

        Args:
            :prompt (List[str]): The list of prompts for the model.
            :generation_len (Optional[int], optional): The generation length.
            :prompt_to_lora_id_mapping (Optional[List[int]], optional): Mapping to associate prompts with their respective LoRA adapter.

        Yields:
        :list: A list containing decoded tokens corresponding to each index of batch size.

        """
        if self._full_batch_size is not None:
            raise NotImplementedError("Streaming tokens is currently unavailable for continuous batch execution.")
        self._setup_model_execution_inputs(prompt, generation_len, prompt_to_lora_id_mapping)
        start = perf_counter()
        outputs, position_ids, generation_len = self._qaic_model.run_prefill(
            prompt=prompt, generation_len=generation_len, prefill_logit_bs=self._qaic_model.batch_size
        )
        self._qaic_model.update_decode_input(outputs, position_ids, generation_len)

        decode_inputs = self._qaic_model.prepare_decode_inputs()

        loop_start = perf_counter()  # Start decode loop timer
        num_token = 0
        for token_id in self._qaic_model.generate_decode_stream(decode_inputs, generation_len, automation):
            decoded_tokens = []
            for idx in range(self._qaic_model.batch_size):
                decoded_tokens.append(self._tokenizer.decode(token_id[idx], skip_special_tokens=True))
            yield decoded_tokens
            num_token += 1
        end = perf_counter()

        total_decode_tokens = num_token
        prefill_time, decode_perf, total_perf, total_time = calculate_latency(
            total_decode_tokens, loop_start, start, end
        )
        self._perf_metrics = PerfMetrics(prefill_time, decode_perf, total_perf, total_time)

    def generate(
        self,
        prompt: List[str],
        images: Optional[List[str]] = None,
        generation_len: Optional[int] = None,
        stream: bool = True,
        automation: Optional[bool] = False,
        prompt_to_lora_id_mapping: Optional[List[int]] = None,
    ):
        """
        Executes the model for a given list of prompts and a specified generation length.

        Args:
            prompt (List[str]): The list of prompts for the model.
            generation_len (Optional[int], optional): The generation length.
            stream (Optional[bool], optional): Boolean flag to enable stream output to console.
            prompt_to_lora_id_mapping (Optional[List[int]], optional): Mapping to associate prompts with their respective LoRA adapter.
        Returns:
            latency_stats (tuple): A tuple containing the generated texts, performance metrics.
        """

        if self._full_batch_size is not None:
            logger.warning("Streamer is currently unavailable for continuous batch execution.")
            perf_metrics, generated_texts = self._continuous_batching_execution(
                prompt, images, generation_len, prompt_to_lora_id_mapping
            )
        else:
            if stream:
                print("\nPrompt : " + prompt[0] + "\nCompletion :", flush=True, end="")
            perf_metrics, generated_texts = self._regular_model_execution(
                prompt, generation_len, stream, automation, prompt_to_lora_id_mapping
            )

        if stream:
            stream_start = 0 if self._full_batch_size else 1
            stream_end = len(prompt) if self._full_batch_size else self._qaic_model.batch_size
            for i in range(stream_start, stream_end):
                print("\n" + "-" * 20)
                print("\nPrompt : ", prompt[i])
                print("Completion : ", generated_texts[i])

        latency_stats = CloudAI100ExecInfo(
            batch_size=1 if self._full_batch_size else self._qaic_model.batch_size,
            generated_texts=generated_texts,
            generated_ids=self._qaic_model.generated_ids,
            perf_metrics=perf_metrics,
        )
        return latency_stats
