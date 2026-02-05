# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

dir_path = os.path.dirname(os.path.realpath(__file__))
# subfunc_npi_file_path = os.path.join(dir_path, "subfunction_120b_npi.yaml")
# non_subfunc_npi_file_path = os.path.join(dir_path, "non_subfunction_120b_npi.yaml")

model_id = "openai/gpt-oss-20b"  # weights are not required to convert to fp32

prompt = """
Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant lands, magical creatures, and extraordinary adventures.

As Alex flipped through the pages, he discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a compass, and set off into the unknown.

The path to the treasure was not an easy one. Alex had to navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's location.
"""
# Run prefill
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
PREFILL_SEQ_LEN = 128
CTX_LEN = 4096

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
    model_id,
    qaic_config={
        "ccl_enabled": True,
    },
)

comp_ctx_lengths_decode = [1024, 2048, 4096]

decode_qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation in the next step
    retain_full_kv=True,
    prefill_only=False,
    comp_ctx_lengths_decode=comp_ctx_lengths_decode,
    # # split_retained_state_io=True,   # This should be used for disagg serving via VLLM
    # node_precision_info=non_subfunc_npi_file_path,
)


qeff_model1 = QEFFAutoModelForCausalLM.from_pretrained(model_id)

# Following command errors out by default, the user is supposed to run the printed command and provide the generated qpc path as prefill_qpc_path commenting out lines 55-68
# prefill_qpc_path = "provide path here"
prefill_qpc_path = qeff_model1.compile(
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    prefill_only=True,
    enable_chunking=True,
    use_onnx_subfunctions=True,
    # # split_retained_state_io=True,  # This should be used for disagg serving via VLLM
    # node_precision_info=subfunc_npi_file_path,
)


inputs = tokenizer(prompt, return_tensors="np", padding=True)
position_ids = inputs["attention_mask"].sum(1, keepdims=True)
generation_len = 100  # CTX_LEN - position_ids.max()
padded_len = inputs["input_ids"].shape[1]
num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len
inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
inputs.pop("token_type_ids", None)
inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
inputs.pop("past_key_values", None)
inputs = {k: v.detach().numpy() for k, v in inputs.items()}


decode_session = QAICInferenceSession(decode_qpc_path)
prefill_session = QAICInferenceSession(prefill_qpc_path)

all_outputs = []

for i in range(num_chunks):
    chunk_inputs = inputs.copy()
    chunk_inputs["input_ids"] = inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = inputs["position_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]

    ins = time.time()
    qpc_out = prefill_session.run(chunk_inputs)

    print(f"time for this run={time.time() - ins}")
    for i in range(config.num_hidden_layers):
        inputs[f"past_key.{i}"] = qpc_out[f"past_key.{i}_RetainedState"]
        inputs[f"past_value.{i}"] = qpc_out[f"past_value.{i}_RetainedState"]

all_outputs.append(np.argmax(qpc_out["logits"]))


def initialize_ccl(decode_inputs, comp_ctx_lengths_decode):
    list_of_comp_ctx_lengths_decode = [np.zeros(length, dtype=np.int8) for length in comp_ctx_lengths_decode]
    max_ccl_id = len(comp_ctx_lengths_decode) - 1
    max_position_id = np.max(decode_inputs["position_ids"])
    ccl_id_initial = 0
    ccl_id = ccl_id_initial
    for i in range(ccl_id_initial, len(comp_ctx_lengths_decode)):
        if max_position_id < comp_ctx_lengths_decode[i]:
            ccl_id = i
            break

    return ccl_id, max_ccl_id, list_of_comp_ctx_lengths_decode


decode_inputs = {
    "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
    "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
}
for i in range(config.num_hidden_layers):
    decode_inputs[f"past_key.{i}"] = qpc_out[f"past_key.{i}_RetainedState"]
    decode_inputs[f"past_value.{i}"] = qpc_out[f"past_value.{i}_RetainedState"]

if comp_ctx_lengths_decode is not None:
    ccl_id, max_ccl_id, list_of_comp_ctx_lengths_decode = initialize_ccl(decode_inputs, comp_ctx_lengths_decode)
    decode_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

st = time.time()
decode_out = decode_session.run(decode_inputs)
print(f"time for first run of decode with KV as input = {time.time() - st} sec\n")
all_outputs.append(np.argmax(decode_out["logits"]))
pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
    "position_ids": pos_id,
}

for i in range(config.num_hidden_layers):
    loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
    loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]

if comp_ctx_lengths_decode is not None:
    ccl_id, max_ccl_id, list_of_comp_ctx_lengths_decode = initialize_ccl(loop_decode_inputs, comp_ctx_lengths_decode)
    loop_decode_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

st = time.time()
for i in range(generation_len - 2):
    if comp_ctx_lengths_decode is not None:
        # Update ccl_id and comp_ctx_lengths_decode based on the maximum position id
        if pos_id >= comp_ctx_lengths_decode[ccl_id] - 1:
            ccl_id = min(ccl_id + 1, max_ccl_id)
            loop_decode_inputs["comp_ctx_lengths"] = list_of_comp_ctx_lengths_decode[ccl_id]

    decode_out = decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"]))
    pos_id += 1
    for i in range(config.num_hidden_layers):
        loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
        loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]

    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
    )
ft = time.time()

print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
print(f"input\n{prompt}\noutput\n{tokenizer.decode(all_outputs)}")
