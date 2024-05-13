# -----------------------------------------------------------------------------
#
# Copyright (c)  2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn

from typing import List, Optional, Union
from threading import Thread

from transformers import (
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
    AutoTokenizer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

from QEfficient.generation.cloud_infer import QAICInferenceSession


class LLMGenerator:
    def __init__(
        self,
        qpc_path :str,
        model_name : str,
        device_id: Optional[List[int]] = [0],
        prompt_len: Optional[int] = 32,
        ctx_len: Optional[int] = 128,
        streamer: Optional[Union[TextStreamer, TextIteratorStreamer]] = None,
        retained_state :bool = True
    ):
        self.session = None
        self.tokenizer = None
        self.is_first_prompt = False
        self.model_name = model_name
        self.device_id = device_id
        self.curr_cache_index = 0
        self.ctx_len = ctx_len
        self.prompt_len = prompt_len
        self.generated_ids = []
        self.inputs = None
        self.stop_indicator = True
        self.retained_state = retained_state
        
        self.qpc_path = (
            qpc_path if os.path.exists(qpc_path) else OSError(f"{qpc_path} not found !")
        )

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

        except Exception as err:
            raise RuntimeError(f"Unable to load qpc on device , {err}")

        try:
            hf_token = None
            if os.getenv("HF_TOKEN") is not None:
                hf_token = os.getenv('HF_TOKEN')
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", hf_token=hf_token
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            self.tokenizer = tokenizer
        except Exception as err:
            raise RuntimeError(f"Unable to load tokenizer, {err}")

        if streamer:
            self.streamer = streamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None
            )
        else:
            self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)

        # instantiate default logit processor and wrapper here
        # TODO : change default values with temperature and top_p
        # instantiate logits processors
        self.logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(
                    15, eos_token_id=self.tokenizer.eos_token_id
                ),
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
            input_ids = torch.Tensor(self.inputs["input_ids"])
            next_token_logits = torch.from_numpy(logits)

            # Qeff is maintaining [1,1,VOCAB_SIZE]
            if len(next_token_logits.shape) == 3:
                next_token_logits = next_token_logits.squeeze(0)
            next_token_scores = self.logits_warper(input_ids, next_token_logits)

            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_token_id = next_tokens.numpy().reshape(1, 1)
        else:
            # greedy search
            if len(logits.shape) == 2:
                logits = np.expand_dims(logits, 1)
            next_token_id = logits.argmax(2)

        return next_token_id

    def _stopping_criteria(self, next_token_id, max_new_tokens=None):
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

        if next_token_id == self.tokenizer.eos_token_id:
            print(
                next_token_id == self.tokenizer.eos_token_id,
                "next_token_id == self.tokenizer.eos_token_id",
            )
            return True
        
        # llama3
        if next_token_id == self.tokenizer.convert_tokens_to_ids("<|eot_id|>"):
            print(
                next_token_id == self.tokenizer.eos_token_id,
                "next_token_id == <|eot_id|>",
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
        _, prompt_len = inputs["input_ids"].shape
        inputs["cache_index"] += prompt_len
        inputs["input_ids"] = next_token_id
        if "attention_mask" in inputs.keys():
            inputs["position_ids"] = inputs.pop("attention_mask").sum(1, keepdims=True)
        else:
            inputs["position_ids"] += 1
        _, prompt_len = inputs["input_ids"].shape
        return inputs, prompt_len

    def generate(self, prompt: str, sample: bool = False, max_new_tokens: int = None):
        session = self.session

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
            outputs = session.run(self.inputs)
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
        
    def apply_chat_template(self, chat):
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)