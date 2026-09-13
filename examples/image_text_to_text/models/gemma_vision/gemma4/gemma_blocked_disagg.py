# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from time import perf_counter

import numpy as np
import torch
import transformers
from gemma4_utils import (
    CHAT_TEMPLATE,
    build_messages,
    remove_fp16clip_transform_if_disabled,
)
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "google/gemma-4-26B-A4B-it"
PREFILL_SEQ_LEN = 1024
CTX_LEN = 4096
GENERATION_LEN = 200
BATCH_SIZE = 1
NUM_CORES = 16
NUM_DEVICES_PREFILL = 1
NUM_DEVICES_DECODE = 2

config = AutoConfig.from_pretrained(MODEL_ID)

# For faster execution user can run with lesser layers, For Testing Purpose Only
config.text_config.num_hidden_layers = 6
config.vision_config.num_hidden_layers = 2

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    MODEL_ID, attn_implementation="eager", kv_offload=True, config=config, dtype="float32", trust_remote_code=True
)


tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

ENABLE_FP16_CLIP = True
remove_fp16clip_transform_if_disabled(qeff_model, ENABLE_FP16_CLIP)

# Blocking options
ENABLE_BLOCKING = True
ENABLE_HEADPAR = True
PREFILL_MODE = "qkv"  # None, "online", or "qkv"
PREFILL_NUM_KV_BLOCKS = 16
PREFILL_QL_CHUNK = 128
PREFILL_BLOCK_CHUNKS = -(-PREFILL_SEQ_LEN // PREFILL_QL_CHUNK)
PREFILL_N_REP_CHUNK = 1
DECODE_NUM_KV_BLOCKS = 8
EXPERT_PARALLEL_CHUNK_SIZE = 256

SYSTEM_PROMPT = "You are a helpful assistant."
TEXT_PROMPT = "Tell me about Taj Mahal?"
IMAGE_PROMPT = "Can you Describe this image in detail?"
IMAGE_URL = "https://wallup.net/wp-content/uploads/2017/03/28/351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"


def _decode_qaic_config() -> dict:
    if not ENABLE_BLOCKING:
        return {}
    if ENABLE_HEADPAR:
        blocking_mode = "kv_headpar"
    else:
        blocking_mode = "kv"
    return {
        "blocking_mode": blocking_mode,
        "num_kv_blocks": DECODE_NUM_KV_BLOCKS,
        "ctx_len": CTX_LEN,
        "skip_kv": True,
    }


def _prefill_qaic_config() -> dict:
    cfg = {
        "moe_config": {
            "flavour": "expert_parallel",
            "expert_parallel_chunk_size": EXPERT_PARALLEL_CHUNK_SIZE,
        }
    }
    if PREFILL_MODE is None:
        return cfg
    cfg.update(
        {
            "blocking_mode": f"prefill_{PREFILL_MODE}",
            "num_kv_blocks": PREFILL_NUM_KV_BLOCKS,
            "num_q_blocks": PREFILL_BLOCK_CHUNKS,
            "n_rep_chunk": PREFILL_N_REP_CHUNK,
        }
    )
    return cfg


prefill_qaic_config = _prefill_qaic_config()
decode_qaic_config = _decode_qaic_config()
print("prefill", prefill_qaic_config)
print("decode", decode_qaic_config)


skip_vision = False
if not skip_vision:
    vision_compile_kwargs = dict(
        batch_size=BATCH_SIZE,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        num_devices=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=False,
        skip_vision=skip_vision,
        split_model_io=True,
        skip_lang=True,
    )
    vision_qpc_path = qeff_model.compile(**vision_compile_kwargs)

prefill_compile_kwargs = dict(
    batch_size=BATCH_SIZE,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=NUM_CORES,
    num_devices=NUM_DEVICES_PREFILL,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    retain_full_kv=True,
    split_model_io=True,
    node_precision_info=False,
    prefill_only=True,
    enable_chunking=True,
    use_onnx_subfunctions=False,
    # Uncomment to compile for PP2
    # mdp_num_partitions=2, # also update NUM_DEVICES_PREFILL=2
    # mdp_strategy="intersection"
    skip_vision=True,
    qaic_config=prefill_qaic_config,
    user_tiled=True,
)

print("prefill_compile_kwargs :", prefill_compile_kwargs)
prefill_qpc_path = qeff_model.compile(**prefill_compile_kwargs)

decode_compile_kwargs = dict(
    batch_size=BATCH_SIZE,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=NUM_CORES,
    num_devices=NUM_DEVICES_DECODE,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    split_model_io=True,
    node_precision_info=False,
    prefill_only=False,
    retain_full_kv=True,
    use_onnx_subfunctions=False,
    skip_vision=True,
    qaic_config=decode_qaic_config,
    user_tiled=True,
)
print("decode_compile_kwargs:", decode_compile_kwargs)
decode_qpc_path = qeff_model.compile(**decode_compile_kwargs)


def _resolve_lang_qpc_path(qpc_obj, preferred_keys):
    if isinstance(qpc_obj, dict):
        for key in preferred_keys:
            if key in qpc_obj:
                return qpc_obj[key]
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_obj.keys())}")
    if isinstance(qpc_obj, (list, tuple)):
        # Backward-compat: some codepaths return (vision_qpc, lang_qpc)
        return qpc_obj[1]
    return qpc_obj


def _resolve_vision_qpc_path(qpc_obj, preferred_keys=("vision_qpc_path",)):
    if isinstance(qpc_obj, dict):
        for key in preferred_keys:
            if key in qpc_obj:
                return qpc_obj[key]
        raise KeyError(f"Could not find any of {preferred_keys} in compile output keys: {list(qpc_obj.keys())}")
    if isinstance(qpc_obj, (list, tuple)):
        # Backward-compat: some codepaths return (vision_qpc, lang_qpc)
        return qpc_obj[0]
    return qpc_obj


lang_prefill_qpc = _resolve_lang_qpc_path(prefill_qpc_path, ("lang_prefill_qpc_path", "lang_qpc_path"))
lang_decode_qpc = _resolve_lang_qpc_path(decode_qpc_path, ("lang_decode_qpc_path", "lang_qpc_path"))

lang_prefill_session = QAICInferenceSession(lang_prefill_qpc)
lang_decode_session = QAICInferenceSession(lang_decode_qpc)
chat_template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE
if skip_vision:
    messages = build_messages(SYSTEM_PROMPT, TEXT_PROMPT, use_image=False)
    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
else:
    messages = build_messages(SYSTEM_PROMPT, IMAGE_PROMPT, use_image=True)
    messages[-1]["content"][0]["url"] = IMAGE_URL
    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    vision_qpc = _resolve_vision_qpc_path(vision_qpc_path)
    vision_session = QAICInferenceSession(vision_qpc)
pad_token_id = 1
input_len = inputs["attention_mask"].sum(1, keepdims=True)
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len
generation_len = GENERATION_LEN
print(f"generation_len : {generation_len}")
generated_ids = np.full((BATCH_SIZE, generation_len + 1), pad_token_id)

inputs["input_ids"] = torch.nn.functional.pad(
    inputs["input_ids"],
    (0, padded_len - input_ids_length),
    "constant",
    pad_token_id,
)
inputs["attention_mask"] = torch.nn.functional.pad(
    inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
)

for k, v in inputs.items():
    inputs[k] = np.array(v)

vision_inputs = {
    k: v
    for k, v in inputs.items()
    if k
    in {
        "pixel_values",
        "image_position_ids",
        "image_masks",
        "image_input_idx",
        "valid_idx",
        "aspect_ratio_ids",
        "aspect_ratio_mask",
    }
}
vision_inputs_fp16 = {"pixel_values", "image_masks"}
vision_inputs.update({k: vision_inputs[k].astype("float16") for k in vision_inputs_fp16 if k in vision_inputs})

vision_start = perf_counter()
vision_outputs = {}
if vision_inputs:
    vision_outputs = vision_session.run(vision_inputs)
vision_end = perf_counter()

lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
if "position_ids" in inputs:
    lang_inputs["position_ids"] = inputs["position_ids"]
    lang_inputs.pop("attention_mask")
else:
    lang_inputs["position_ids"] = np.where(
        lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
    )  # Need to use -1 as position_ids for invalid tokens

lang_inputs["image_idx"] = np.array([[0]])
if not skip_vision:
    lang_inputs["vision_embeds"] = vision_outputs["vision_embeds"]

# RUN prefill
lang_start = perf_counter()
lang_prefill_session.set_buffers(vision_outputs)
all_outputs = []
chunk_inputs = lang_inputs.copy()

for i in range(num_chunks):
    chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    if "mm_token_type_ids" in lang_inputs:
        chunk_inputs["mm_token_type_ids"] = lang_inputs["mm_token_type_ids"][
            ..., i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN
        ]

    outputs = lang_prefill_session.run(chunk_inputs)
    for layer_idx in range(config.text_config.num_hidden_layers):
        chunk_inputs[f"past_key.{layer_idx}"] = outputs[f"past_key.{layer_idx}_RetainedState"]
        chunk_inputs[f"past_value.{layer_idx}"] = outputs[f"past_value.{layer_idx}_RetainedState"]
    if "image_idx_output" in outputs:
        chunk_inputs["image_idx"] = outputs["image_idx_output"]
prefill_time = perf_counter() - lang_start + vision_end - vision_start
print(f"Prefill time : {prefill_time:.2f} secs")
all_outputs.append(np.argmax(outputs["logits"]))
decode_inputs = {
    "input_ids": np.argmax(outputs["logits"]).reshape(1, 1),
    "position_ids": np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1,
}
for layer_idx in range(config.text_config.num_hidden_layers):
    decode_inputs[f"past_key.{layer_idx}"] = outputs[f"past_key.{layer_idx}_RetainedState"]
    decode_inputs[f"past_value.{layer_idx}"] = outputs[f"past_value.{layer_idx}_RetainedState"]
if "image_idx_output" in outputs:
    decode_inputs["image_idx"] = outputs["image_idx_output"]
if "vision_embeds_RetainedState" in outputs:
    decode_inputs["vision_embeds"] = outputs["vision_embeds_RetainedState"]

st = perf_counter()
decode_out = lang_decode_session.run(decode_inputs)
print(f"time for first run of decode with KV as input = {perf_counter() - st} sec\n")

all_outputs.append(np.argmax(decode_out["logits"]))
pos_id = np.max(decode_inputs["position_ids"], axis=-1, keepdims=True) + 1
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
    "position_ids": pos_id,
}

for layer_idx in range(config.text_config.num_hidden_layers):
    loop_decode_inputs[f"past_key.{layer_idx}"] = decode_out[f"past_key.{layer_idx}_RetainedState"]
    loop_decode_inputs[f"past_value.{layer_idx}"] = decode_out[f"past_value.{layer_idx}_RetainedState"]
if "image_idx_output" in decode_out:
    loop_decode_inputs["image_idx"] = decode_out["image_idx_output"]
if "vision_embeds_RetainedState" in decode_out:
    loop_decode_inputs["vision_embeds"] = decode_out["vision_embeds_RetainedState"]

st = perf_counter()
for i in range(generation_len - 2):
    decode_out = lang_decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"]))
    pos_id += 1
    for j in range(config.text_config.num_hidden_layers):
        loop_decode_inputs[f"past_key.{j}"] = decode_out[f"past_key.{j}_RetainedState"]
        loop_decode_inputs[f"past_value.{j}"] = decode_out[f"past_value.{j}_RetainedState"]
    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
    )
ft = perf_counter()
print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
print(f"\noutput\n{tokenizer.decode(all_outputs)}")
