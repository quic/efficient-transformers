# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from collections import deque
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
        return f"Average Prefill time a.k.a TTFT is= {round(self.prefill_time, 2)}\
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
    full_batch_size: Optional[int] = None,
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
    prompt = fix_prompts(prompt, batch_size, full_batch_size)
    generate_text = TextGeneration(
        tokenizer=tokenizer,
        prompt=prompt,
        qpc_path=qpc_path,
        device_id=device_id,
        ctx_len=ctx_len,
        generation_len=generation_len,
        enable_debug_logs=enable_debug_logs,
        stream=stream,
        write_io_dir=write_io_dir,
        full_batch_size=full_batch_size,
    )
    if full_batch_size is None:
        exec_info = [
            generate_text.cloud_ai_100_exec_kv_helper(prompt[i : i + batch_size], generation_len)
            for i in range(0, len(prompt), batch_size)
        ]
        prefill_time = np.average([info.prefill_time for info in exec_info])
        decode_perf = np.average([info.decode_perf for info in exec_info])
        total_perf = np.average([info.total_perf for info in exec_info])
        total_time = np.average([info.total_time for info in exec_info])
        generated_texts = [info.generated_texts for info in exec_info]
        generated_ids = [info.generated_ids for info in exec_info]

        exec_info = CloudAI100ExecInfo(
            batch_size=batch_size,
            generated_texts=generated_texts,
            generated_ids=generated_ids,
            prefill_time=prefill_time,
            decode_perf=decode_perf,
            total_perf=total_perf,
            total_time=total_time,
        )
    else:
        exec_info = generate_text.cloud_ai_100_exec_kv_helper(prompt=prompt, generation_len=generation_len)

    print_latency_stats_kv(prompt, exec_info=exec_info, automation=automation)
    return exec_info


class TextGeneration:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        qpc_path: str,
        prompt: List[str],
        full_batch_size: Optional[int] = None,
        ctx_len: Optional[int] = None,
        generation_len: Optional[int] = None,
        device_id: Optional[List[int]] = None,
        enable_debug_logs: bool = False,
        stream: bool = True,
        write_io_dir: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.qpc_path = qpc_path
        self.device_id = device_id
        self.ctx_len = ctx_len
        self.generation_len = generation_len
        self.enable_debug_logs = enable_debug_logs
        self.stream = stream

        self.write_io_dir = write_io_dir

        # Load QPC
        self.session = QAICInferenceSession(qpc_path, device_id, enable_debug_logs=enable_debug_logs)
        self.streamer = transformers.TextStreamer(self.tokenizer)

        # Fetch the variables from the QPC
        self.vocab_size = self._fetch_vocab_size()  # Fetch Vocab size
        self.batch_size, self.prefill_seq_len = self._fetch_batch_size_prefill_seq_len()
        self.full_batch_size = (
            full_batch_size if full_batch_size else self._fetch_full_batch_size()
        )  # Check and fetch full batch size if CB is enabled

        self.set_tokenizer_params()  # set tokenizer params

        # Initialize the storage variables.
        self.batch_index = None
        self.batch_index_prefill = None

    def set_tokenizer_params(self):
        """
        Sets the tokenizer parameters for the model.
        """
        if self.tokenizer.padding_side != "right":
            logger.warning("Please use padding_side='right' while initializing the tokenizer")
            self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def calculate_latency(self, total_decoded_tokens, loop_start, start, end, decode_pause_time=0):
        """
        Method will calculate the latency metrics using the time loops and based on the total decoded token count.

        Returns:
        total_num_decoded_tokens, prefill_perf, decode_perf, total_perf
        """
        prefill_time = loop_start - start + decode_pause_time
        decode_perf = (total_decoded_tokens) / (end - loop_start - decode_pause_time)
        total_perf = (total_decoded_tokens) / (end - start)
        total_time = end - start

        return prefill_time, decode_perf, total_perf, total_time

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
        if "batch_index" in self.session.binding_index_map:
            if self.session.allowed_shapes:
                full_batch_size, _ = [
                    x[self.session.binding_index_map["batch_index"]][1][0] for x in self.session.allowed_shapes
                ]
            else:
                full_batch_size, _ = self.session.bindings[self.session.binding_index_map["batch_index"]].dims
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
        if self.session.allowed_shapes:
            batch_size = max(
                [x[self.session.binding_index_map["input_ids"]][1][0] for x in self.session.allowed_shapes]
            )
            prefill_seq_len = max(
                [x[self.session.binding_index_map["input_ids"]][1][1] for x in self.session.allowed_shapes]
            )
        else:
            batch_size, prefill_seq_len = self.session.bindings[self.session.binding_index_map["input_ids"]].dims
        return batch_size, prefill_seq_len

    def _fetch_vocab_size(
        self,
    ):
        """
        Fetches the vocabulary size from the session's allowed shapes.
        Returns:
            vocab_size: The vocabulary size fetched from the session's allowed shapes.
        """
        return [x[self.session.binding_index_map["logits"]] for x in self.session.allowed_shapes][0][1][2]

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
            if self.ctx_len is None:
                raise ValueError("At least one of ctx_len or generation_len is needed")
            generation_len = max_gen_len
        elif generation_len > max_gen_len:
            logger.warning(
                "Passed generation_len is greater than allowed length. "
                "Make sure this model supports sliding window, such as Mistral"
            )
        assert generation_len > 0, "generation length should be greater than zero"
        return generation_len

    def prepare_decode_inputs(self):
        """
        This function creates the decode inputs.

        Returns:
            dict: The decode inputs.
        """
        decode_inputs = {}
        decode_inputs["input_ids"] = self.decode_input_ids
        decode_inputs["position_ids"] = self.decode_pos_ids
        if self.batch_index is not None:
            decode_inputs["batch_index"] = self.batch_index

        return decode_inputs

    def _update_decode_input(self, outputs, position_ids, generation_len, decode_batch_id=None):
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
        logits = outputs["logits"]
        if len(logits.shape) == 2:
            logits = np.expand_dims(logits, 1)

        # Get output token
        next_token_id = logits.argmax(2)

        # Store the generated values.
        self.decode_input_ids[decode_batch_id or slice(None)] = next_token_id
        self.decode_pos_ids[decode_batch_id or slice(None)] = position_ids
        self.generated_ids[decode_batch_id or slice(None), 0] = next_token_id.squeeze(1)
        self.generation_len[decode_batch_id or slice(None)] = generation_len
        return next_token_id

    def run_prefill_for_all_inputs(self, prompt_queue, generation_len):
        """
        Runs prefill for all inputs in the prompt queue and updates the decode input.

        Method iterates over the full batch size and for each decode batch ID, it pops the next prompt from the queue.  It then runs prefill for the next prompt and updates the decode input with the outputs.

        Args:
            prompt_queue (deque): The queue of prompts.
            generation_len (int): The generation length.

        """
        for decode_batch_id in range(self.full_batch_size):
            next_prompt = prompt_queue.popleft()

            # run prefill for num_chunks
            outputs, position_ids, generation_len = self.run_prefill(
                next_prompt, generation_len, decode_batch_id=np.array(decode_batch_id, dtype=np.int64).reshape(1, 1)
            )

            _ = self._update_decode_input(outputs, position_ids, generation_len, decode_batch_id)

    def run_prefill(self, prompt, generation_len, prefill_logit_bs=1, decode_batch_id=None):
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
        inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
        position_ids = inputs["attention_mask"].sum(1, keepdims=True)
        padded_len = inputs["input_ids"].shape[1]
        num_chunks = -(padded_len // -self.prefill_seq_len)  # ceil divide without float
        padded_len = num_chunks * self.prefill_seq_len  # Convert to a multiple of prompt_len

        # Calculate the max generation length.
        max_gen_len = self.ctx_len - position_ids.max()
        generation_len = self._fetch_generation_len(generation_len, max_gen_len)

        # Set the prefill logic buffer
        logits_out_placeholder = np.zeros((prefill_logit_bs, 1, self.vocab_size), dtype=np.float32)
        self.session.set_buffers({"logits": logits_out_placeholder})

        inputs = self.tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)

        if decode_batch_id is not None:
            inputs["batch_index"] = decode_batch_id

        for i in range(num_chunks):
            chunk_inputs = inputs.copy()
            chunk_inputs["input_ids"] = inputs["input_ids"][
                :, i * self.prefill_seq_len : (i + 1) * self.prefill_seq_len
            ]
            chunk_inputs["position_ids"] = inputs["position_ids"][
                :, i * self.prefill_seq_len : (i + 1) * self.prefill_seq_len
            ]
            outputs = self.session.run(chunk_inputs)
            if self.write_io_dir is not None:
                write_io_files(inputs, outputs, self.write_io_dir, "prefill", "aic_batch_io", True, False)
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

        # Set logits placeholder for decode
        logits_out_placeholder = np.zeros((self.full_batch_size, 1, self.vocab_size), dtype=np.float32)
        self.session.set_buffers({"logits": logits_out_placeholder})
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
            outputs = self.session.run(decode_inputs)

            # Prepare inputs for next iteration
            logits = outputs["logits"]
            if len(logits.shape) == 2:
                logits = np.expand_dims(logits, 1)
            next_token_id = logits.argmax(2)

            for decode_batch_id in range(self.full_batch_size):
                if (
                    next_token_id[decode_batch_id] == self.tokenizer.eos_token_id
                    or generated_id_current_index[decode_batch_id] >= self.generation_len[decode_batch_id]
                ):
                    if prompt_queue:
                        start = perf_counter()
                        # run prefill for next prompt input.
                        outputs, position_ids, generation_len = self.run_prefill(
                            prompt_queue.popleft(),
                            generation_len,
                            decode_batch_id=np.array(decode_batch_id, dtype=np.int64).reshape(1, 1),
                        )

                        new_token_id = self._update_decode_input(outputs, position_ids, generation_len, decode_batch_id)

                        batch_id_map[decode_batch_id] = max(batch_id_map.values()) + 1
                        self.generated_ids[batch_id_map[decode_batch_id], 0] = new_token_id.squeeze(1)
                        generated_id_current_index[decode_batch_id] = 1

                        self.session.set_buffers({"logits": logits_out_placeholder})
                        decode_pause_time += perf_counter() - start
                    else:
                        current_decode_ongoing[decode_batch_id] = False
                else:
                    # If the generated sequence is valid and within generation len prepare for next decode
                    decode_inputs["input_ids"][decode_batch_id] = next_token_id[decode_batch_id]
                    decode_inputs["position_ids"][decode_batch_id] += 1
                    self.generated_ids[batch_id_map[decode_batch_id], generated_id_current_index[decode_batch_id]] = (
                        next_token_id[decode_batch_id]
                    )

                    generated_id_current_index[decode_batch_id] += 1
        return decode_pause_time

    def run_decode(self, decode_inputs, generation_len):
        """
        Default method for running decode. Executes the decoding process for a given set of inputs and a specified generation length.

        Enters a loop that continues until all sequences are finished or the maximum generation length is reached. In each iteration, it runs the session with the decode inputs, prepares the inputs for the next iteration and checks if all sequences are finished.

        Args:
            decode_inputs (dict): The initial inputs for decoding. This should be a dictionary containing 'input_ids' and 'position_ids'.
            generation_len (int): Max allowed length for generating tokens. The decoding process will be terminated  when generation length is reached.

        Returns:
            num_token (int): The number of tokens processed in the decoding process.
        """
        finished_sequences = decode_inputs["input_ids"] == self.tokenizer.eos_token_id
        num_token = 0
        for num_token in range(1, generation_len):
            if self.stream:
                self.streamer.put(decode_inputs["input_ids"][0])
            outputs = self.session.run(decode_inputs)

            if self.write_io_dir is not None:
                write_io_files(decode_inputs, outputs, self.write_io_dir, "decode", "aic_batch_io", True, False)
                self.write_io_dir = None

            # Prepare inputs for next iteration
            decode_inputs["input_ids"] = outputs["logits"].argmax(2)
            decode_inputs["position_ids"] += 1
            self.generated_ids[:, num_token] = decode_inputs["input_ids"].squeeze(1)
            finished_sequences |= decode_inputs["input_ids"] == self.tokenizer.eos_token_id

            if finished_sequences.all():
                break
        return num_token

    def regular_model_execution(self, prompt, generation_len):
        """
        Executes the model in regular mode.
        This method runs the prefill, prepares the decode inputs, and then runs the decode. The generated texts are decoded and optionally streamed. Latency metrics are calculated and returned.
        Args:
            :prompt (str): Sample prompt for the model text generation.
            :generation_len (int): Number of tokens to be generated.

        Returns:
        :tuple: A tuple containing prefill time, decode performance, total performance, total time and generated_texts.

        """
        start = perf_counter()
        outputs, position_ids, generation_len = self.run_prefill(
            prompt, generation_len, prefill_logit_bs=self.batch_size
        )
        self._update_decode_input(outputs, position_ids, generation_len)

        decode_inputs = self.prepare_decode_inputs()

        loop_start = perf_counter()  # Start decode loop timer
        num_token = self.run_decode(decode_inputs, generation_len)
        end = perf_counter()

        generated_texts = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)

        total_decode_tokens = num_token
        prefill_time, decode_perf, total_perf, total_time = self.calculate_latency(
            total_decode_tokens, loop_start, start, end
        )
        return prefill_time, decode_perf, total_perf, total_time, generated_texts

    def continuous_batching_execution(self, prompt, prompt_queue, generation_len):
        """
        Executes the model using continuous batching.
        This method handles the execution of the model when continuous batching is enabled. It runs the prefill step for all inputs, performs continuous batching decode, and then decodes the generated texts. The texts are optionally streamed. Latency metrics are calculated and returned.
        Args:
            :prompt (list): prompts for the model text generation
            :prompt_queue (list): Queue of prompts for the model text generation.
            :generation_len (int): Number of tokens to be generated.

        Returns:
        :tuple: A tuple containing prefill time, decode performance, total performance, total time and generated_texts.
        """
        self.batch_index = np.arange(self.full_batch_size).reshape(-1, 1)
        start = perf_counter()
        self.run_prefill_for_all_inputs(prompt_queue, generation_len)

        loop_start = perf_counter()  # Start decode loop timer
        decode_pause_time = self.run_continuous_batching_decode(prompt_queue, generation_len)
        end = perf_counter()

        generated_texts = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)

        total_decode_tokens = sum(
            np.sum(self.generated_ids[i] != self.tokenizer.pad_token_id) - 1 for i in range(len(prompt))
        )
        prefill_time, decode_perf, total_perf, total_time = self.calculate_latency(
            total_decode_tokens, loop_start, start, end, decode_pause_time
        )
        prefill_time /= len(prompt)  # Average prefill time for continuous batching
        return prefill_time, decode_perf, total_perf, total_time, generated_texts

    def cloud_ai_100_exec_kv_helper(self, prompt: List[str], generation_len: Optional[int] = None):
        """
        Executes the model for a given list of prompts and a specified generation length.

        Args:
            prompt (List[str]): The list of prompts for the model.
            generation_len (Optional[int], optional): The generation length.

        Returns:
            latency_stats (tuple): A tuple containing the generated texts, prefill time, decode performance, total
            performance, and total time.
        """

        # Skip inputs/outputs
        self.session.skip_buffers(
            [x for x in self.session.input_names + self.session.output_names if x.startswith("past_")]
        )

        execution_batch_size = self.full_batch_size if self.full_batch_size is not None else self.batch_size

        # Create a prompt queue.
        prompt_queue = deque(prompt)

        max_gen_length = self.ctx_len if not generation_len else max(self.ctx_len, generation_len)

        # initialize np arrays for storing the prefill output for all the decode batch size.
        self.generated_ids = np.full((len(prompt_queue), max_gen_length), self.tokenizer.pad_token_id)
        self.decode_input_ids = np.zeros((execution_batch_size, 1), np.int64)
        self.decode_pos_ids = np.zeros((execution_batch_size, 1), np.int64)
        self.generation_len = np.zeros((execution_batch_size, 1), np.int64)

        if self.full_batch_size is not None:
            logger.warning("Streamer is currently unavailable for continuous batch execution.")
            prefill_time, decode_perf, total_perf, total_time, generated_texts = self.continuous_batching_execution(
                prompt, prompt_queue, generation_len
            )
        else:
            if self.stream:
                self.streamer.on_finalized_text("\nPrompt : " + prompt[0] + "\nCompletion :")
            prefill_time, decode_perf, total_perf, total_time, generated_texts = self.regular_model_execution(
                prompt, generation_len
            )

        if self.stream:
            stream_start = 0 if self.full_batch_size else 1
            stream_end = len(prompt) if self.full_batch_size else self.batch_size
            for i in range(stream_start, stream_end):
                print("\n" + "-" * 20)
                print("\nPrompt : ", prompt[i])
                print("Completion : ", generated_texts[i])

        latency_stats = CloudAI100ExecInfo(
            batch_size=1 if self.full_batch_size else self.batch_size,
            generated_texts=generated_texts,
            generated_ids=self.generated_ids,
            prefill_time=prefill_time,
            decode_perf=decode_perf,
            total_perf=total_perf,
            total_time=total_time,
        )
        return latency_stats
