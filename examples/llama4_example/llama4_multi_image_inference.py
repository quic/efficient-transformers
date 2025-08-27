from io import BytesIO
from typing import List

from time import perf_counter
import transformers
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    AutoConfig, 
    AutoProcessor, 
    AutoModelForImageTextToText,
    TextStreamer,
)

import os
import json
import torch
import decord
from decord import VideoReader, cpu
from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.text_generation_inference import (
    get_compilation_dims,
)
from QEfficient.generation.text_generation_inference import (
    CloudAI100ExecInfoNew,
    PerfMetrics,
    calculate_latency,
    get_compilation_dims,
)

from PIL import Image
import requests


def get_index(fps, max_frame, first_idx=0,num_frames=8):
    start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_frames
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_frames)
    ])
    return frame_indices

def load_video(video_path:str, output_dir:str, input_size=336, num_frames=13):
    vr = VideoReader(video_path, ctx=cpu(0))
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    # transform = build_transform(input_size=input_size)
    frame_indices = get_index(fps, max_frame, first_idx=0, num_frames=num_frames)
    for i, frame_index in enumerate(frame_indices):
        frame = vr[frame_index].asnumpy()  
        if frame.ndim == 4:
            frame = frame[0] 
        image = Image.fromarray(frame)  
        # image = transform(image)
        image = image.resize((672,672))  
        path = f"{output_dir}/{i}.jpg"
        image.save(path)

def main(
    model: str,
    hf_token: str,
    qpc_vision: str,
    qpc_text: str,
    prompt: str,
    output_dir: str,
    video_path: str,
    device_id_vision: List[int] = [0,1],
    device_id_text: List[int] = [0,1],
    generation_len: Optional[int] = None,
):
    config = AutoConfig.from_pretrained(model, token=hf_token)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, token=hf_token, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model, token=hf_token)
    streamer = TextStreamer(tokenizer)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    load_video(video_path, output_dir, input_size=672, num_frames=8)
    messages=[]
    os.makedirs(output_dir, exist_ok=True)

    # Build the content list
    content = []
    for filename in sorted(os.listdir(output_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            image_path = os.path.join(output_dir, filename)
            content.append({"type": "image", "url": image_path})

    # Add the analysis prompt
    content.append({
        "type": "text",
        "text": (
            "You are a video analysis expert. Given a continuous set of frames from the video, your task is to generate a concise and informative summary. Focus on identifying key events, important dialogues, visual highlights, and emotional tone. Structure the summary to reflect the overall narrative or progression of the video. If the video contains multiple scenes or segments, break the summary into logical parts. Ensure the summary is clear, coherent, and suitable for someone who hasn’t watched the video."
        )
    })

    # Final messages structure
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    )

    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    if not qpc_text:
        raise TypeError("Please run compile API for language model first!")

    lang_session = QAICInferenceSession(qpc_text, device_id_text)

    if qpc_vision:
        vision_session = QAICInferenceSession(qpc_vision, device_id_vision)
    batch_size, ctx_len, fbs = get_compilation_dims(qpc_text)

    pad_token_id = 1

    # Skip inputs/outputs
    lang_session.skip_buffers(
        [
            x
            for x in lang_session.input_names + lang_session.output_names
            if x.startswith("past_") or x.endswith("_RetainedState")
        ]
    )

    # Read prompt and ctx len from session
    batch_size = max(
        [x[lang_session.binding_index_map["input_ids"]][1][0] for x in lang_session.allowed_shapes]
        + [lang_session.bindings[lang_session.binding_index_map["input_ids"]].dims[0]]
    )

    prefill_seq_len = max(
        [x[lang_session.binding_index_map["input_ids"]][1][1] for x in lang_session.allowed_shapes]
        + [lang_session.bindings[lang_session.binding_index_map["input_ids"]].dims[1]]
    )

    input_len = inputs["attention_mask"].sum(1, keepdims=True)
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
    padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

    if generation_len is None:
        generation_len = ctx_len - input_len.max()
    assert generation_len > 0, "generation length should be greater than zero"
    generated_ids = np.full((batch_size, generation_len + 1), pad_token_id)

    inputs["input_ids"] = torch.nn.functional.pad(
        inputs["input_ids"],
        (0, padded_len - input_ids_length),
        "constant",
        1,
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

    llama4 = hasattr(config, "model_type") and config.model_type == "llama4"
    if llama4:
        qpc_base_path = os.path.dirname(os.path.normpath(qpc_text))
        specialization_file_path = os.path.join(qpc_base_path, "specializations.json")
        if os.path.exists(specialization_file_path):
            with open(specialization_file_path, "r") as file:
                data = json.load(file)
        else:
            raise FileNotFoundError(f"expected specializations.json file at path, {qpc_base_path}")
        num_patches = int(data["specializations"][0]["max_num_tiles"])
        if vision_inputs['pixel_values'].shape[0] != num_patches:
            single_patch = np.expand_dims(vision_inputs['pixel_values'][0], axis=0)
            while vision_inputs['pixel_values'].shape[0] < num_patches:
                vision_inputs['pixel_values'] = np.concatenate((vision_inputs['pixel_values'], single_patch), axis=0)


    if vision_inputs:
        vision_inputs["pixel_values"] = vision_inputs["pixel_values"].astype("float16")
    vision_start = perf_counter()

    vision_outputs = {}
    if vision_inputs:
        vision_outputs = vision_session.run(vision_inputs)
    vision_end = perf_counter()

    lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
    lang_inputs["position_ids"] = np.where(
        lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
    )  # Need to use -1 as position_ids for invalid tokens

    if qpc_vision:
        vision_session.deactivate()
    lang_session.activate()

    lang_session.set_buffers(vision_outputs)

    # Prepare inputs for prefill
    chunk_inputs = lang_inputs.copy()
    prefill_start = perf_counter()

    # Prepare inputs for prefill
    chunk_inputs = lang_inputs.copy()
    prefill_start = perf_counter()

    # Run prefill
    chunk_inputs = lang_inputs.copy()
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][
            :, i * prefill_seq_len : (i + 1) * prefill_seq_len
        ]
        outputs = lang_session.run(chunk_inputs)
        chunk_inputs["image_idx"] = outputs["image_idx_output"]

    prefill_time = perf_counter() - prefill_start + vision_end - vision_start
    # Skip inputs/outputs again
    lang_session.skip_buffers(
        [
            x
            for x in lang_session.input_names + lang_session.output_names
            if x.startswith("past_") or x.endswith("_RetainedState")
        ]
    )

    # Get first token
    lang_inputs["input_ids"] = outputs["logits"].argmax(2)
    lang_inputs["position_ids"] = input_len.numpy()
    if "cross_attention_mask" in lang_inputs:
        bs, _, num_images, img_tiles = lang_inputs["cross_attention_mask"].shape
        lang_inputs["cross_attention_mask"] = torch.ones((bs, 1, num_images, img_tiles), dtype=torch.int64).numpy()
    generated_ids[:, 0] = lang_inputs["input_ids"].squeeze(1)

    if streamer:
        streamer.put(lang_inputs["input_ids"][0])

    # Decode loop
    decode_start = perf_counter()
    for num_token in range(1, generation_len):
        outputs = lang_session.run(lang_inputs)

        # Prepare inputs for next iteration
        lang_inputs["input_ids"] = outputs["logits"].argmax(2)
        lang_inputs["position_ids"] += 1
        generated_ids[:, num_token] = lang_inputs["input_ids"].squeeze(1)
        if streamer:
            streamer.put(lang_inputs["input_ids"][0])

    decode_end = perf_counter()
    if streamer:
        streamer.end()

    decode_perf = (num_token - 1) / (decode_end - decode_start)
    total_time = decode_end - decode_start + prefill_time
    total_perf = num_token / total_time

    print(CloudAI100ExecInfoNew(
        batch_size=batch_size,
        generated_ids=generated_ids,
        perf_metrics=PerfMetrics(
            prefill_time=prefill_time, decode_perf=decode_perf, total_perf=total_perf, total_time=total_time
        ),
    )
    )

if __name__ == "__main__":
    import argparse

    argp = argparse.ArgumentParser()
    argp.add_argument("--model", required=True, help="Model name to run")
    argp.add_argument("--hf-token", required=True, help="Hugging Face Token")
    argp.add_argument("--qpc-vision", required=True, help="Compiled binary QPC of image and text input model")
    argp.add_argument("--qpc-text", required=True, help="Compiled binary QPC of text only model")
    argp.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        default="Please describe the image in detail.",
        help="Input prompt(s) to generate for (pipe-separated)",
    )
    # argp.add_argument("--image", required=True, help="Image to be passed as input")
    argp.add_argument(
        "--device-id-vision",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        help="QAIC device ids (comma-separated)",
        default=[0,1,2,3]
    )
    argp.add_argument(
        "--device-id-text",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        help="QAIC device ids (comma-separated)",
        default=[0,1,2,3]
    )
    argp.add_argument(
        "--generation-len",
        type=int,
        help="Number of tokens to generate. \
    Note: For models without rolling buffer, (generation length + input length) should \
    be less than model context length",
    )

    
    argp.add_argument("--video-path", required=True, help="Path to the input video file")
    argp.add_argument("--output-dir", required=True, help="Directory to save the output")

    args = argp.parse_args()
    main(**vars(args))
