# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from time import perf_counter
from typing import Dict, List, Optional
import sys
from threading import Thread
from typing import *
import torch
import numpy as np
import torch.nn as nn

import transformers

# from aic_infer import QAICInferenceSession

from QEfficient.generation.aic_infer import QAICInferenceSession

io_files = []


import io

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
)


from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
)


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def write_io_files(
    inputs: Dict[str, np.ndarray],
    outputs: Dict[str, np.ndarray],
    write_io_dir: str,
    write_io_subdir: str,
):
    io = []
    os.makedirs(f"{write_io_dir}/{write_io_subdir}", exist_ok=True)

    for iname, iarray in inputs.items():
        iarray.tofile(f"{write_io_dir}/{write_io_subdir}/{iname}.raw")
        io.append(
            {
                "path": f"{write_io_subdir}/{iname}.raw",
                "io-direction": "in",
                "dims": iarray.shape,
                "elem-size": iarray.itemsize,
                "map-to": iname,
            }
        )

    for oname, oarray in outputs.items():
        oarray.tofile(f"{write_io_dir}/{write_io_subdir}/{oname}.raw")
        io.append(
            {
                "path": f"{write_io_subdir}/{oname}.raw",
                "io-direction": "out",
                "dims": oarray.shape,
                "elem-size": oarray.itemsize,
                "map-to": oname,
            }
        )

    io_files.append(io)
    with open(f"{write_io_dir}/aic_batch_io.json", "w") as fp:
        json.dump({"IO-files": io_files}, fp, indent=True)


class LLMGenerator:
    def __init__(
        self,
        qpc_path,
        model_name,
        device_id: Optional[List[int]] = [0],
        prompt_len: Optional[int] = 32,
        ctx_len: Optional[int] = 128,
        streamer: Optional["BaseStreamer"] = None,
        logits_processor: Optional = None,
        logits_warper: Optional = None,
    ):
        self.session = None
        self.tokenizer = None
        self.is_first_prompt = False
        self.model_name = ""
        self.qpc_path = ""
        self.device_id = [0]
        self.curr_cache_index = 0
        self.ctx_len = ctx_len
        self.retained_state = True
        self.write_io_dir = False
        self.prompt_len = prompt_len
        self.generated_ids = []
        self.inputs = None
        self.stop_indicator = True

        self.qpc_path = (
            qpc_path if os.path.exists(qpc_path) else OSError(f"{qpc_path} not found !")
        )
        self.device_id = device_id

        self.model_name = model_name

        try:
            self.session = QAICInferenceSession(
                self.qpc_path, self.device_id, enable_debug_logs=False
            )
            if self.retained_state:
                self.session.skip_buffers(
                    [x for x in self.session.input_names if x.startswith("past_")]
                )
                self.session.skip_buffers(
                    [
                        x
                        for x in self.session.output_names
                        if x.endswith("_RetainedState")
                    ]
                )

                # self.session.skip_buffers(
                #     set([x for x in self.session.input_names if x.startswith("past_")])
                # )
                # self.session.skip_buffers(
                #     set(
                #         [
                #             x
                #             for x in self.session.output_names
                #             if x.endswith("_RetainedState")
                #         ]
                #     )
                # )

        except Exception as err:
            raise RuntimeError("Unable to load qpc on device , {err}")

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            self.tokenizer = tokenizer
        except Exception as err:
            raise RuntimeError("Unable to load tokenizer, {err}")

        if streamer:
            self.streamer = streamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None
            )

        # instatiate deault logit processor and wrapper here
        # TODO : change default values with temperature and top_p
        # self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        # self.logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        # instantiate logits processors
        self.logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(
                    15, eos_token_id=2
                ),  # model.generation_config.eos_token_id
            ]
        )

        # instantiate logits processors
        self.logits_warper = LogitsProcessorList(
            [
                TopKLogitsWarper(50),
                TemperatureLogitsWarper(0.7),
            ]
        )

        self.stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=ctx_len)]
        )

    def _generate_next_token(self, outputs, sample=False):
        logits = outputs["logits"]

        if sample:
            # pre-process distribution
            # input_ids = torch.Tensor(self.generated_ids)
            input_ids = torch.Tensor(self.inputs["input_ids"])
            next_token_logits = torch.from_numpy(logits)
            # next_token_scores = self.logits_processor(input_ids, next_token_logits)
            next_token_scores = self.logits_warper(input_ids, next_token_logits)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_token_id = next_tokens.numpy().reshape(1, 1)
        else:
            # greedy search
            if len(logits.shape) == 2:
                logits = np.expand_dims(logits, 1)
            next_token_id = logits.argmax(2)
            # print("next_token_id: ", next_token_id.shape)

        # print("next_token_id", next_token_id)

        return next_token_id

    def _stopping_criteria(self, next_token_id, max_new_tokens=None):
        # if self.curr_cache_index > self.ctx_len:
        if self.curr_cache_index >= self.ctx_len - 1:
            print("self.curr_cache_index reach limit")
            return True

        if max_new_tokens:
            if len(self.generated_ids) > max_new_tokens:
                print(
                    "len(self.generated_ids) > max_new_tokens",
                    len(self.generated_ids) > max_new_tokens,
                )
                return True
        # assert type(self.tokenizer.eot_id) == List
        # assert type(self.tokenizer.eos_token_id) == List

        # breakpoint()

        if next_token_id == self.tokenizer.eos_token_id:
            print(
                next_token_id == self.tokenizer.eos_token_id,
                "next_token_id == self.tokenizer.eos_token_id",
            )
            return True

        return False

    def prepare_inputs_for_inference(self, prompt):
        # prepare inputs for prefill part
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            max_length=self.prompt_len,
        )
        batch_size, prompt_len = inputs["input_ids"].shape

        ctx_len = self.ctx_len

        # assert ctx_len > prompt_len, "Model cannot support prompt_len > ctx_len"

        inputs["position_ids"] = (np.cumsum(inputs["attention_mask"], 1) - 1) * inputs[
            "attention_mask"
        ]
        inputs["attention_mask"] = np.concatenate(
            [
                inputs["attention_mask"].astype(bool),
                np.zeros((batch_size, ctx_len - prompt_len), dtype=bool),
            ],
            1,
        )
        cache_index = np.array([0])
        inputs["cache_index"] = cache_index

        return inputs, prompt_len

    def update_inputs_for_inference(self, inputs, next_token_id):
        # breakpoint()
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["cache_index"] += prompt_len

        inputs["input_ids"] = next_token_id

        batch_size, prompt_len = inputs["input_ids"].shape

        if "attention_mask" in inputs.keys():
            inputs["position_ids"] = inputs.pop("attention_mask").sum(1, keepdims=True)
        else:
            inputs["position_ids"] += 1

        batch_size, prompt_len = inputs["input_ids"].shape
        return inputs, prompt_len

    def generate(self, prompt: str, sample: bool = False, max_new_tokens: int = None):
        session = self.session
        # if self.write_io_dir:
        #     write_io_files(inputs, outputs, write_io_dir, "prefill")

        multi_turn_input_ids = []

        if self.curr_cache_index == 0:
            self.inputs, prompt_len = self.prepare_inputs_for_inference(prompt)
            outputs = session.run(self.inputs)
            self.curr_cache_index += prompt_len
            session.skip_buffers(["attention_mask"])

        else:
            multi_turn_input_ids = self.tokenizer(
                prompt,
                return_tensors="np",
            ).input_ids
            self.generated_ids = []

        while self.stop_indicator:
            if len(multi_turn_input_ids) == 0:
                next_token_id = self._generate_next_token(outputs, sample)
                # next_token_id will be from prompt till prompt
                self.generated_ids.append(next_token_id)

                if self.streamer:
                    self.streamer.put(next_token_id[0])

                if self._stopping_criteria(next_token_id, max_new_tokens):
                    print("Stopping criteria hit")
                    break
            elif (
                len(multi_turn_input_ids.shape) == 2
                and multi_turn_input_ids.shape[1] > 0
            ):
                next_token_id, multi_turn_input_ids = (
                    multi_turn_input_ids[:, 0],
                    multi_turn_input_ids[:, 1:],
                )
                next_token_id = np.expand_dims(next_token_id, 1)
            elif (
                len(multi_turn_input_ids.shape) == 2
                and multi_turn_input_ids.shape[1] == 0
            ):
                multi_turn_input_ids = []

            self.inputs, next_prompt_len = self.update_inputs_for_inference(
                self.inputs, next_token_id
            )
            # print(self.curr_cache_index) # for debug
            outputs = session.run(self.inputs)
            # next_prompt_len from next iteration onwards is 1
            self.curr_cache_index += next_prompt_len

        if self.streamer:
            return self.streamer.end()
        else:
            return ""

    def stream(self, prompt: str, sample: bool = False, max_new_tokens: int = None):
        generate_args = {
            "prompt": prompt,
            "sample": sample,
            "max_new_tokens": max_new_tokens,
        }

        t = Thread(target=self.generate, kwargs=generate_args)
        t.start()

        outputs = []
        for text in self.streamer:
            outputs.append(text)
            yield "".join(outputs)

        print("".join(outputs))
        # return "".join(outputs)


def test_llm(
    model_name: str,
    prompt_len: int,
    ctx_len: int,
    qpc: str,
    prompt: str,
    session: QAICInferenceSession = None,
    stream: bool = True,
    sample: bool = False,
    device_id: List[int] = [0],
    write_io_dir: Optional[str] = None,
):
    # python LLMGenerator.py --model-name codellama/CodeLlama-13b-Instruct-hf --prompt-len 128 --ctx-len 1024 --qpc /home/hupreti/demo/CodeLlama-13b-Instruct-hf-kv-128pl-1024cl-16c-mxfp6 --prompt "Complete the paragraph with 2048 tokens: My name is Himanshu, and" 2>&1 | tee output.log

    # print(prompt)

    # working with TextStreamer
    # model_aic = LLMGenerator(qpc, model_name, device_id, prompt_len, ctx_len,
    #                 streamer = TextStreamer)

    model_aic = LLMGenerator(
        qpc, model_name, device_id, prompt_len, ctx_len, streamer=TextStreamer
    )

    generate_kwargs = {"prompt": prompt, "sample": sample, "max_new_tokens": ctx_len}

    t = Thread(target=model_aic.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in model_aic.streamer:
        # print(text, end=" ")
        outputs.append(text)

        # yield "".join(outputs)
    print("".join(outputs))

    t.join()

    # Uncomment below to test mulit-turn
    # generate_kwargs = {
    #     "prompt" : "Indian Cricket Team. But 2014",
    #     "sample" : False,
    #     "max_new_tokens" : 128
    # }

    # t = Thread(target=model_aic.generate, kwargs=generate_kwargs)
    # t.start()

    # t.join()

    # print(generate_kwargs["prompt"])
    # outputs = []
    # for text in model_aic.streamer:
    #     # print(text)
    #     outputs.append(text)

    #     # yield "".join(outputs)
    # print("".join(outputs))

    return


if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument("--model-name", required=True, help="Model name to run")
    argp.add_argument("--prompt-len", type=int, default=128, help="Prompt length")
    argp.add_argument("--ctx-len", type=int, default=512, help="Context length")
    argp.add_argument("--qpc", required=True, help="Compiled binary QPC")
    argp.add_argument(
        "--prompt",
        default="My name is Sarah and I am",
        help="Input prompt to generate for",
    )
    argp.add_argument(
        "--no-stream",
        action="store_false",
        dest="stream",
        help="Don't stream output text",
    )
    argp.add_argument(
        "--device_id",
        default=[0],
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        help="QAIC device ids (comma-separated)",
    )
    argp.add_argument("--write-io-dir", help="Directory to write inputs/outputs into")
    argp.add_argument(
        "--sample", action="store_true", dest="sample", help="Use sampling"
    )

    args = argp.parse_args()
    # main(**vars(args))
    test_llm(**vars(args))
