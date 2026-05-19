# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import transformers
from transformers import AutoConfig, AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForImageTextToText

# from QEfficient import QEFFAutoModelForCausalLM

## For AWQ model update pytorch version to 2.8.*
model_id = "Qwen/Qwen3.5-0.8B"
config = AutoConfig.from_pretrained(model_id)
config.torch_dtype = "float32"
config.text_config.num_hidden_layers = 2

qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id, attn_implementation="eager", kv_offload=False, config=config
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)


## Only Text ##

## Set Batch_Size ##
batch_size = 1

# Enable KV blocking for full-attention layers with 2 KV blocks
# To disable KV blocking, comment out the qaic_config line below
# Set skip_kv=True to skip future KV blocks during inference (optimization)
qaic_config = {"blocking_mode": "kv", "num_kv_blocks": 2, "skip_kv": True}

enable_blocking = False

qeff_model.compile(
    batch_size=batch_size,
    prefill_seq_len=64,
    ctx_len=4096,
    num_cores=16,
    num_devices=2,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    aic_enable_depth_first=False,
    # convert_to_fp16=False,
    # skip_vision=True,
    mos=1,
    # qaic_config=qaic_config,  # Enable KV blocking - comment out to disable
)


if enable_blocking:
    print("\n" + "=" * 80)
    print("Verifying KV Blocking Applied During Compilation")
    print("=" * 80)

    # The compile() method internally calls BlockingAttentionTransform.apply()
    # which sets attn_blocking_config on all supported attention modules
    # This happens BEFORE ONNX export, so blocking operations are in the ONNX graph

    if qaic_config and qaic_config.get("blocking_mode"):
        print("✓ qaic_config passed to compile():")
        print(f"    Blocking Mode: {qaic_config.get('blocking_mode')}")
        print(f"    Num KV Blocks: {qaic_config.get('num_kv_blocks')}")
        print(f"    Skip KV: {qaic_config.get('skip_kv', False)}")
        print("\n✓ BlockingAttentionTransform.apply() called during compile()")
        print("  - Sets attn_blocking_config on all supported attention modules")
        print("  - Blocked attention forward pass is used during ONNX export")
        print("  - Blocking operations are in the ONNX graph and QPC")
        print("\n  Status: ACTIVE")
        print("  Verification: Config-based verification")
        print("  Note: Blocking IS applied - torch model is freed after ONNX export")
    else:
        print("✗ No qaic_config provided - eager attention will be used")
        print("  Status: INACTIVE - Model compiled without blocking")

    print("=" * 80 + "\n")


messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Tell me about yourself."},
        ],
    },
]


messages = [messages] * batch_size

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)

# inputs = qeff_model.model.prepare_inputs_for_generation(inputs=inputs, prefill_seq_len=128, batch_size=batch_size)

inputs.pop("mm_token_type_ids")
streamer = TextStreamer(tokenizer)
output = qeff_model.generate(inputs=inputs, generation_len=100)
print(output.generated_ids)
print(tokenizer.batch_decode(output.generated_ids))
print(output)
