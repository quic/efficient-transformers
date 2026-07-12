# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""GLM-OCR VLM inference on Cloud AI 100.

GLM-OCR compiles as a single QPC (kv_offload=False). Its M-RoPE position_ids are
4D (text row + T/H/W rows), which the generic single-QPC generate() path does not
support — it always derives 2D position_ids from attention_mask. So this example
builds position_ids via prepare_inputs_for_generation() and drives inference with
QAICInferenceSession directly, chunking the prefill and looping token-by-token
for decode.
"""

import argparse
import time

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.modeling_auto import get_compilation_dims

MODEL_NAME = "zai-org/GLM-OCR"
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
PROMPT = "Describe this image in detail."


def make_kv_cache(num_layers: int, num_kv_heads: int, head_dim: int, ctx_len: int) -> dict:
    zeros = np.zeros((1, num_kv_heads, ctx_len, head_dim), dtype=np.float32)
    return {name: zeros.copy() for i in range(num_layers) for name in (f"past_key.{i}", f"past_value.{i}")}


def run_model(
    model_name=MODEL_NAME,
    image_url=IMAGE_URL,
    prompt=PROMPT,
    prefill_seq_len=320,
    ctx_len=1024,
    gen_len=20,
    num_cores=16,
    num_devices=4,
):
    ## STEP 1: Load the Processor and Model

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = QEFFAutoModelForImageTextToText.from_pretrained(model_name, kv_offload=False, dtype=torch.float32)

    ## STEP 2: Export & Compile the Model

    model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
    )

    ## STEP 3: Load and Process the Inputs for Inference

    # GLM-OCR's vision encoder patches a fixed grid; resize so the compiled
    # prefill_seq_len covers the resulting image token count.
    image_size = model.model.config.vision_config.image_size
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB").resize((image_size, image_size))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    inputs = model.model.prepare_inputs_for_generation(inputs, prefill_seq_len=prefill_seq_len, batch_size=1)

    ## STEP 4: Run Inference on the Compiled Model

    seq_len = inputs["input_ids"].shape[1]
    padded_len = inputs["position_ids"].shape[-1]
    n_chunks = padded_len // prefill_seq_len
    pad_id = processor.tokenizer.pad_token_id or 0

    input_ids = F.pad(inputs["input_ids"], (0, padded_len - seq_len), value=pad_id).numpy()
    position_ids = inputs["position_ids"].numpy()
    pixel_values = inputs["pixel_values"].float().numpy()
    image_idx = np.zeros((1, 1), dtype=np.int64)

    text_cfg = model.model.config.text_config
    kv_cache = make_kv_cache(text_cfg.num_hidden_layers, text_cfg.num_key_value_heads, text_cfg.head_dim, ctx_len)

    _, compiled_ctx_len, _ = get_compilation_dims(model.qpc_path)
    session = QAICInferenceSession(str(model.qpc_path), list(range(num_devices)), activate=False)
    session.skip_buffers(
        [x for x in session.input_names + session.output_names if x.startswith("past_") or x.endswith("_RetainedState")]
    )
    session.activate()
    input_names = set(session.input_names)

    prefill_start = time.perf_counter()
    outputs = None
    for i in range(n_chunks):
        start, end = i * prefill_seq_len, (i + 1) * prefill_seq_len
        feed = {
            "input_ids": input_ids[:, start:end],
            "position_ids": position_ids[:, :, start:end],
            "pixel_values": pixel_values,
            "image_idx": image_idx,
            **kv_cache,
        }
        outputs = session.run({k: v for k, v in feed.items() if k in input_names})
        image_idx = outputs["image_idx_output"].astype(np.int64)
        if "pixel_values_RetainedState" in session.output_names:
            session.skip_buffers(["pixel_values"])
            pixel_values = None
    prefill_time = time.perf_counter() - prefill_start

    logits = outputs["logits"]
    generated = [int(logits[0, 0].argmax())]
    cur_pos = int(position_ids[0, 0, :seq_len].max()) + 1

    eos_ids = text_cfg.eos_token_id
    eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])

    decode_start = time.perf_counter()
    for _ in range(gen_len - 1):
        if generated[-1] in eos_ids:
            break
        feed = {
            "input_ids": np.array([[generated[-1]]], dtype=np.int64),
            "position_ids": np.full((4, 1, 1), cur_pos, dtype=np.int64),
            "image_idx": image_idx,
            **kv_cache,
        }
        if pixel_values is not None:
            feed["pixel_values"] = pixel_values
        outputs = session.run({k: v for k, v in feed.items() if k in input_names})
        image_idx = outputs["image_idx_output"].astype(np.int64)
        generated.append(int(outputs["logits"][0, 0].argmax()))
        cur_pos += 1
    decode_time = time.perf_counter() - decode_start
    session.deactivate()

    output_text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    decode_tps = (len(generated) - 1) / decode_time if decode_time > 0 else 0.0

    print(f"Output: {output_text!r}")
    print(f"Prefill time: {prefill_time:.2f}s, decode: {decode_tps:.1f} tok/s")
    return output_text


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR VLM inference")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="HuggingFace model ID")
    parser.add_argument("--image-url", type=str, default=IMAGE_URL, help="URL of the image to process")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Text query about the image")
    parser.add_argument("--prefill-seq-len", type=int, default=320, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=1024, help="Context length")
    parser.add_argument("--gen-len", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument("--num-devices", type=int, default=4, help="Number of devices")
    args = parser.parse_args()

    run_model(
        model_name=args.model_name,
        image_url=args.image_url,
        prompt=args.prompt,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        gen_len=args.gen_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
    )


if __name__ == "__main__":
    main()
