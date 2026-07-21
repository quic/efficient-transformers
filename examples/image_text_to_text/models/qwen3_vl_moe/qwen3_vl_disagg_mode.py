# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from time import perf_counter

import numpy as np
import requests
import torch
import transformers
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

model_id = "Qwen/Qwen3-VL-235B-A22B-Instruct"
# model_id = "tiny-random/qwen3-vl-moe"
config = AutoConfig.from_pretrained(model_id)
config.dtype = "float16"
config.torch_dtype = torch.float16

# For faster execution user can run with lesser layers, For Testing Purpose Only
config.vision_config.depth = 9
config.text_config.num_hidden_layers = 2
config.vision_config.deepstack_visual_indexes = [8]

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=True, config=config, dtype=torch.float16, layerwise=False
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

PREFILL_SEQ_LEN = 1024
CTX_LEN = 2048 * 2
BS = 256

NUM_KV_BLOCKS = 4
NUM_Q_BLOCKS = 2
HEAD_BLOCK_SIZE = 8
PREFILL_BLOCK_CHUNKS = None
PREFILL_MODE = None  # None, "online" or "qkv" depending on whether we want online prefill or headparallel prefill


###############
# Decode modes:
# - standard attention - pass enable_blocking and blocking_mode
# - head parallel blocking - pass  enable_blocking, blocking_mode: “kv” and kv_block_headpar_split: 0
# - batch fold head parallel - pass enable_blocking, blocking_mode: “kv” and batch_fold: True


def _decode_qaic_config() -> dict:
    return {
        "blocking_mode": "kv",
        "num_kv_blocks": NUM_KV_BLOCKS,
        # "kv_blocking_headpar_split": 0,  # 0 → resolved to num_cores at compile time
        "batch_fold": True,
        "ctx_len": CTX_LEN,
    }


def _qaic_config() -> dict:
    cfg = _decode_qaic_config()
    if PREFILL_MODE is None:
        return cfg
    cfg["prefill_block_chunks"] = PREFILL_BLOCK_CHUNKS
    cfg["prefill_blocking_mode"] = PREFILL_MODE
    cfg["prefill_n_rep_chunk"] = PREFILL_N_REP_CHUNK
    return cfg


skip_vision = True
if not skip_vision:
    vision_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        height=354,
        width=536,
        num_cores=16,
        num_devices=1,
        mos=1,
        mxfp6_matmul=True,
        aic_enable_depth_first=True,
        skip_vision=skip_vision,
        split_model_io=True,
        skip_lang=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )
decode_qaic_config = _qaic_config()
print("decode", decode_qaic_config)
decode_start_time = perf_counter()
decode_qpc_path = qeff_model.compile(
    batch_size=BS,
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    height=354,
    width=536,
    num_cores=16,
    num_devices=8,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    split_model_io=True,  # This should be used for disagg serving via VLLM
    mos=1,
    user_tiled=True,
    prefill_only=False,
    expert_parallel=True,  # This forces the model to use expert parallelism for the MoE layers
    tree_reduce=True,  # This enables tree reduction for the MoE layers, which can improve performance when using multiple devices
    cores_per_expert=1,  # number_of_parallelized_experts_per_device = total_experts * cores_per_expert / total_cores , total_cores = num_devices * num_cores, number_of_pipline_stages = total_experts / number_of_parallelized_experts_per_device
    skip_vision=True,
    use_onnx_subfunctions=False,
    layerwise=False,
    offload_pt_weights=False,
    qaic_config=decode_qaic_config,
)
print(f"Decode export + compile time is {(perf_counter() - decode_start_time):.3f}s")

# exit(0)

################
# Prefill modes:
# - follow decode attention - pass nothing extra
# - head parallel offline prefill - pass prefill_blocking_mode: “qkv”, prefill_block_chunks: 2
# - online prefill - pass prefill_blocking_mode: “online”, prefill_block_chunks: 2
PREFILL_MODE = "online"
PREFILL_QL_CHUNK = 128
PREFILL_BLOCK_CHUNKS = -(-PREFILL_SEQ_LEN // PREFILL_QL_CHUNK)
PREFILL_N_REP_CHUNK = 4
MOE_PREFILL_PACKED_CHUNK_SIZE = 256
prefill_qaic_config = _qaic_config()
print("prefill", prefill_qaic_config)

prefill_start_time = perf_counter()
prefill_qpc_path = qeff_model.compile(
    batch_size=1,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    moe_prefill_packed_chunk_size=MOE_PREFILL_PACKED_CHUNK_SIZE,
    height=354,
    width=536,
    num_cores=16,
    num_devices=1,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    retain_full_kv=True,
    split_model_io=True,  # This should be used for disagg serving via VLLM
    mos=1,
    aic_enable_depth_first=True,
    prefill_only=True,
    enable_chunking=True,
    skip_vision=True,
    use_onnx_subfunctions=False,
    layerwise=False,
    offload_pt_weights=True,
    qaic_config=prefill_qaic_config,
)
print(f"Prefill export + compile time is {(perf_counter() - prefill_start_time):.3f}s")

print(f"Prefill qpc path {prefill_qpc_path}")
print(f"Decode qpc path {decode_qpc_path}")

lang_prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"))
lang_decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"))

if skip_vision:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tell me about yourself."},
            ],
        },
    ]
else:
    ### IMAGE + TEXT ###
    image_url = "https://picsum.photos/id/237/536/354"
    image = Image.open(requests.get(image_url, stream=True).raw)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe all the colors seen in the image."},
                # {"type": "text", "text": "Can you describe the image in detail?"},
            ],
        },
    ]
    vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))


messages = [messages] * BS

texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=PREFILL_SEQ_LEN, batch_size=BS)

pad_token_id = 1
input_len = inputs["attention_mask"].sum(1, keepdims=True)
input_ids_length = inputs["input_ids"].shape[1]
num_chunks = -(input_ids_length // -PREFILL_SEQ_LEN)  # ceil divide without float
padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len
generation_len = 30  # CTX_LEN - input_len.max()
print(f"generation_len : {generation_len}")
generated_ids = np.full((BS, generation_len + 1), pad_token_id)


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
    if k in {"pixel_values", "image_masks", "image_input_idx", "valid_idx", "aspect_ratio_ids", "aspect_ratio_mask"}
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
    lang_inputs["deepstack_features"] = vision_outputs["deepstack_features"]

# RUN prefill (batch_size=1; inputs are sliced to first batch item since all are identical)
lang_start = perf_counter()
lang_prefill_session.set_buffers(vision_outputs)
all_outputs = []
chunk_inputs = lang_inputs.copy()
for i in range(num_chunks):
    chunk_inputs["input_ids"] = lang_inputs["input_ids"][0:1, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    chunk_inputs["position_ids"] = lang_inputs["position_ids"][:, 0:1, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
    outputs = lang_prefill_session.run(chunk_inputs)
    for i in range(config.text_config.num_hidden_layers):
        chunk_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
        chunk_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]
    chunk_inputs["image_idx"] = outputs["image_idx_output"]
prefill_time = perf_counter() - lang_start + vision_end - vision_start
print(f"Prefill time : {prefill_time:.2f} secs")

# Next token from batch=1 prefill; position for all BS decode requests
next_token_id = np.argmax(outputs["logits"])  # scalar
all_outputs.append(next_token_id)
next_pos = np.max(lang_inputs["position_ids"], axis=-1, keepdims=True) + 1

# Tile KV from prefill [1, num_kv_heads, ctx_len, head_dim]
# → batch-fold decode layout [1, BS*num_kv_heads, ctx_len, head_dim]
decode_inputs = {
    "input_ids": np.full((BS, 1), next_token_id, dtype=lang_inputs["input_ids"].dtype),
    "position_ids": next_pos,
}

for layer_idx in range(config.text_config.num_hidden_layers):
    # RetainedState from prefill has shape [1, num_kv_heads, ctx_len, head_dim].
    # Replicate across BS decode requests, then fold into [1, BS*num_kv_heads, ctx_len, head_dim].
    _, h, c, d = outputs[f"past_key.{layer_idx}_RetainedState"].shape
    decode_inputs[f"past_key.{layer_idx}"] = np.tile(
        outputs[f"past_key.{layer_idx}_RetainedState"], (1, BS, 1, 1)
    ).reshape(1, BS * h, c, d)
    decode_inputs[f"past_value.{layer_idx}"] = np.tile(
        outputs[f"past_value.{layer_idx}_RetainedState"], (1, BS, 1, 1)
    ).reshape(1, BS * h, c, d)

st = perf_counter()
decode_out = lang_decode_session.run(decode_inputs)
print(f"time for first run of decode with KV as input = {perf_counter() - st} sec\n")

# exit(0)

all_outputs.append(np.argmax(decode_out["logits"][0]))  # track batch 0
pos_id = decode_inputs["position_ids"] + 1  # [BS, 1]
loop_decode_inputs = {
    "input_ids": np.argmax(decode_out["logits"], axis=-1),  # [BS, 1]
    "position_ids": pos_id,
}

# for i in range(config.text_config.num_hidden_layers):
#     loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
#     loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]


st = perf_counter()
for i in range(generation_len - 2):
    decode_out = lang_decode_session.run(loop_decode_inputs)
    all_outputs.append(np.argmax(decode_out["logits"][0]))
    pos_id += 1
    # for j in range(config.text_config.num_hidden_layers):
    #     loop_decode_inputs[f"past_key.{j}"] = decode_out[f"past_key.{j}_RetainedState"]
    #     loop_decode_inputs[f"past_value.{j}"] = decode_out[f"past_value.{j}_RetainedState"]
    loop_decode_inputs.update(
        {
            "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
            "position_ids": pos_id,
        }
    )
ft = perf_counter()
print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
print(f"\noutput\n{tokenizer.decode(all_outputs)}")
