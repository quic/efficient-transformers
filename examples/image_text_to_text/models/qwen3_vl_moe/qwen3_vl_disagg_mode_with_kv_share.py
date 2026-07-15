# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Disaggregated prefill/decode for Qwen3-VL-MoE — DMA-based KV handoff."""

import argparse
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

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_PROMPT = "Tell me about yourself."
DEFAULT_IMAGE_PROMPT = "Describe all the colors seen in the image."
DEFAULT_IMAGE_URL = "https://picsum.photos/id/237/536/354"
DEFAULT_PREFILL_SEQ_LEN = 128
DEFAULT_CTX_LEN = 2048
DEFAULT_GENERATION_LEN = 100
STAGES = 4
PREFILL_NUM_DEVICES = 8
DECODE_NUM_DEVICES = 4
BS = 1

# Vision-tower inputs are routed to the vision QPC; everything else feeds the lang QPCs.
VISION_INPUT_KEYS = {
    "pixel_values",
    "image_masks",
    "image_input_idx",
    "valid_idx",
    "aspect_ratio_ids",
    "aspect_ratio_mask",
}
VISION_FP16_KEYS = {"pixel_values", "image_masks"}
VISION_OUTPUT_KEYS = ("vision_embeds", "deepstack_features")


def _build_config(model_id: str):
    """Load the model config, pinned to float16 (matches the baseline)."""
    config = AutoConfig.from_pretrained(model_id)
    config.dtype = "float16"
    config.torch_dtype = torch.float16

    # For faster execution user can run with fewer layers. For testing purposes only.
    # config.vision_config.depth = 9
    # config.text_config.num_hidden_layers = 6
    # config.vision_config.deepstack_visual_indexes = [8]
    return config


def run(
    model_id: str = DEFAULT_MODEL_ID,
    prompt: str = DEFAULT_PROMPT,
    prefill_seq_len: int = DEFAULT_PREFILL_SEQ_LEN,
    ctx_len: int = DEFAULT_CTX_LEN,
    generation_len: int = DEFAULT_GENERATION_LEN,
    skip_vision: bool = False,
    image_url: str = DEFAULT_IMAGE_URL,
    stages: int = STAGES,
    prefill_num_devices: int = PREFILL_NUM_DEVICES,
    decode_num_devices: int = DECODE_NUM_DEVICES,
):
    """Run chunked-prefill + decode with the DMA-based KV handoff.

    ``skip_vision=False`` (default) fetches ``image_url`` and runs an image+text prompt
    through the vision QPC; ``skip_vision=True`` runs a text-only prompt. Returns a dict
    with the prefill ``logits``, the ``first_token`` (argmax over those logits) and the
    full decoded ``tokens`` list.
    """
    config = _build_config(model_id)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id, attn_implementation="eager", kv_offload=True, config=config, dtype=torch.float16, layerwise=False
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    vision_session = None
    if not skip_vision:
        vision_qpc_path = qeff_model.compile(
            batch_size=BS,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            height=354,
            width=536,
            num_cores=16,
            num_devices=2,
            mos=1,
            mxfp6_matmul=True,
            aic_enable_depth_first=True,
            skip_vision=False,
            split_model_io=True,
            skip_lang=True,
            use_onnx_subfunctions=True,
            layerwise=False,
            offload_pt_weights=False,
        )
        vision_session = QAICInferenceSession(vision_qpc_path.get("vision_qpc_path"))

    decode_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=1,
        ctx_len=ctx_len,
        height=354,
        width=536,
        num_cores=16,
        num_devices=decode_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,  # required for DMA slice writes into full KV
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        prefill_only=False,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
        offload_pt_weights=False,  # keep weights for the prefill export/compile below
    )

    prefill_qpc_path = qeff_model.compile(
        batch_size=BS,
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        height=354,
        width=536,
        num_cores=16,
        num_devices=prefill_num_devices,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        retain_full_kv=True,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=False,
        mdp_num_partitions=stages,
        prefill_only=True,
        enable_chunking=True,
        skip_vision=True,
        use_onnx_subfunctions=True,
        layerwise=False,
    )

    prefill_session = QAICInferenceSession(prefill_qpc_path.get("lang_prefill_qpc_path"), kv_dma_share=True)
    decode_session = QAICInferenceSession(decode_qpc_path.get("lang_decode_qpc_path"), kv_dma_share=True)

    assert "image_idx" in prefill_session.binding_index_map, "image_idx not a compiled prefill input binding"
    decode_has_image_idx = "image_idx" in decode_session.binding_index_map

    if skip_vision:
        content = [{"type": "text", "text": prompt}]
    else:
        image = Image.open(requests.get(image_url, stream=True).raw)
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
    messages = [[{"role": "user", "content": content}]] * BS

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = qeff_model.model.prepare_inputs_for_generation(
        inputs=inputs, prefill_seq_len=prefill_seq_len, batch_size=BS
    )

    pad_token_id = 1
    input_ids_length = inputs["input_ids"].shape[1]
    num_chunks = -(input_ids_length // -prefill_seq_len)  # ceil divide without float
    padded_len = num_chunks * prefill_seq_len  # Convert to a multiple of prompt_len

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

    # ---- Vision (producer): run the vision QPC once; its outputs feed every lang step ----
    vision_inputs = {k: v for k, v in inputs.items() if k in VISION_INPUT_KEYS}
    vision_inputs.update({k: vision_inputs[k].astype("float16") for k in VISION_FP16_KEYS if k in vision_inputs})
    vision_outputs = {}
    if not skip_vision and vision_inputs:
        vision_outputs = vision_session.run(vision_inputs)

    lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
    if "position_ids" in inputs:
        lang_inputs["position_ids"] = inputs["position_ids"]
        lang_inputs.pop("attention_mask", None)
    else:
        lang_inputs["position_ids"] = np.where(
            lang_inputs.pop("attention_mask"), np.arange(padded_len), -1
        )  # Need to use -1 as position_ids for invalid tokens

    lang_inputs["image_idx"] = np.array([[0]])
    if not skip_vision:
        # vision_embeds and deepstack_features are constant across every prefill chunk and
        # decode step, so register them once as persistent inputs
        vision_persist = {k: vision_outputs[k] for k in VISION_OUTPUT_KEYS if k in vision_outputs}
        prefill_session.set_persistent_inputs(vision_persist)
        decode_session.set_persistent_inputs(
            {k: v for k, v in vision_persist.items() if k in decode_session.binding_index_map}
        )

    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]

    prefill_start = perf_counter()
    chunk_inputs = dict(lang_inputs)
    exec_idx = None
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, i * prefill_seq_len : (i + 1) * prefill_seq_len]
        chunk_inputs["position_ids"] = lang_inputs["position_ids"][..., i * prefill_seq_len : (i + 1) * prefill_seq_len]
        last_chunk = i == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        chunk_inputs["image_idx"] = prefill_session.get_outputs(index=exec_idx)["image_idx_output"]
    print(f"Prefill time : {perf_counter() - prefill_start:.2f} secs")

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    prefill_logits = prefill_out["logits"]
    first_token = int(np.argmax(prefill_logits))
    all_outputs = [first_token]

    # ---- Decode (consumer): re-point DMA descriptor at kv_caches EVERY step ----
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    num_pos_sections = lang_inputs["position_ids"].shape[0]
    phys_pos = int(lang_inputs["position_ids"][0].max()) + 1
    mrope_pos = int(lang_inputs["position_ids"][1:].max()) + 1

    def _decode_position_ids(next_phys: int, next_mrope: int) -> np.ndarray:
        pos = np.empty((num_pos_sections, BS, 1), dtype=np.int64)
        pos[0] = next_phys
        pos[1:] = next_mrope
        return pos

    decode_inputs = {
        "input_ids": np.array(first_token, dtype=np.int64).reshape(1, 1),
        "position_ids": _decode_position_ids(phys_pos, mrope_pos),
    }
    if decode_has_image_idx:
        decode_inputs["image_idx"] = prefill_out["image_idx_output"]
    st = perf_counter()
    for _ in range(generation_len - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        out = decode_session.get_outputs(index=exec_idx)
        tok = int(np.argmax(out["logits"]))
        all_outputs.append(tok)
        phys_pos += 1
        mrope_pos += 1
        decode_inputs = {
            "input_ids": np.array(tok, dtype=np.int64).reshape(1, 1),
            "position_ids": _decode_position_ids(phys_pos, mrope_pos),
        }
        if decode_has_image_idx:
            decode_inputs["image_idx"] = out["image_idx_output"]
    ft = perf_counter()

    print(f"decode tok/sec={(generation_len - 1) / (ft - st)}")
    print(f"input\n{prompt}\noutput\n{tokenizer.decode(all_outputs)}")

    return {"logits": prefill_logits, "first_token": first_token, "tokens": all_outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="text prompt")
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument("--stages", type=int, default=STAGES, help="prefill pipeline depth (mdp_num_partitions)")
    parser.add_argument(
        "--prefill-num-devices", type=int, default=PREFILL_NUM_DEVICES, help="num devices for the prefill QPC"
    )
    parser.add_argument(
        "--decode-num-devices", type=int, default=DECODE_NUM_DEVICES, help="num devices for the decode QPC"
    )
    vision = parser.add_mutually_exclusive_group()
    vision.add_argument(
        "--skip-vision",
        dest="skip_vision",
        action="store_true",
        help="text-only lang path",
    )
    vision.add_argument(
        "--with-vision",
        dest="skip_vision",
        action="store_false",
        help="image+text: compile and run the vision QPC on --image-url (default)",
    )
    parser.set_defaults(skip_vision=False)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL, help="image URL when --with-vision")
    args = parser.parse_args()

    # With vision the default text prompt is a poor fit; use the image prompt unless
    # the user overrode --prompt.
    prompt = args.prompt
    if not args.skip_vision and prompt == DEFAULT_PROMPT:
        prompt = DEFAULT_IMAGE_PROMPT

    run(
        model_id=args.model_id,
        prompt=prompt,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        generation_len=args.generation_len,
        skip_vision=args.skip_vision,
        image_url=args.image_url,
        stages=args.stages,
        prefill_num_devices=args.prefill_num_devices,
        decode_num_devices=args.decode_num_devices,
    )
