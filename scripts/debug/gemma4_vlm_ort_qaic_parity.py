# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Standalone Gemma4 VLM parity probe.

Checks, in order:
1. HF float32 reference generation
2. HF float32 vision tower vs QEff modified-PyTorch vision wrapper exactness
3. Exported dual-ONNX ORT greedy decode
4. Optional QAic dual-QPC greedy decode

This is intentionally focused on tiny-random/gemma-4-dense.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="tiny-random/gemma-4-dense")
    parser.add_argument(
        "--image",
        default="https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/GoldenGate.png",
    )
    parser.add_argument("--prompt", default="What is shown in this image?")
    parser.add_argument("--generation-len", type=int, default=4)
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=96)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--device-group", nargs="+", type=int, default=None)
    parser.add_argument("--run-qaic", action="store_true")
    parser.add_argument("--enable-npi", action="store_true")
    parser.add_argument("--enable-vision-npi", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/gemma4_vlm_ort_qaic_parity"))
    return parser.parse_args()


def build_inputs(processor, image_url: str, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    rendered_prompt = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
    return rendered_prompt, inputs


def decode_new_tokens(tokenizer, token_ids):
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def resolve_effective_ctx_len(requested_ctx_len: int, prompt_len: int, generation_len: int) -> int:
    return max(requested_ctx_len, prompt_len + generation_len)


def resolve_effective_prefill_seq_len(requested_prefill_seq_len: int, prompt_len: int) -> int:
    return max(requested_prefill_seq_len, prompt_len)


def pad_prefill_inputs(inputs, prefill_seq_len: int, pad_token_id: int):
    input_ids = inputs["input_ids"]
    batch_size, input_len = input_ids.shape
    num_chunks = -(input_len // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len
    pad_len = padded_len - input_len

    if pad_len == 0:
        return inputs, num_chunks, padded_len

    padded = dict(inputs)
    padded["input_ids"] = torch.nn.functional.pad(input_ids, (0, pad_len), "constant", pad_token_id)
    padded["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, pad_len), "constant", 0)
    padded["mm_token_type_ids"] = torch.nn.functional.pad(inputs["mm_token_type_ids"], (0, pad_len), "constant", 0)
    return padded, num_chunks, padded_len


def build_initial_decoder_inputs(model, inputs, vision_embeds, ctx_len: int):
    lang_cfg = model.model.model.language_model.config
    seq_len = inputs["input_ids"].shape[1]
    past_key_values = model.model.get_dummy_pkv_cache(lang_cfg, batch_size=1, seq_len=ctx_len)
    if "attention_mask" in inputs:
        position_ids = torch.where(
            inputs["attention_mask"].bool(),
            inputs["attention_mask"].cumsum(1) - 1,
            torch.full_like(inputs["attention_mask"], -1),
        )
    else:
        position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len)

    decoder_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "vision_embeds": vision_embeds.astype(np.float32),
        "position_ids": position_ids.cpu().numpy(),
        "image_idx": np.zeros((1, 1), dtype=np.int64),
        "mm_token_type_ids": inputs["mm_token_type_ids"].cpu().numpy(),
    }
    for i, (key, value) in enumerate(past_key_values):
        decoder_inputs[f"past_key.{i}"] = key.cpu().numpy()
        decoder_inputs[f"past_value.{i}"] = value.cpu().numpy()
    return decoder_inputs


def run_ort_session(session: ort.InferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = {x.name for x in session.get_inputs()}
    feed = {k: v for k, v in inputs.items() if k in input_names}
    output_names = [x.name for x in session.get_outputs()]
    outputs = session.run(output_names, feed)
    return dict(zip(output_names, outputs))


def update_decoder_inputs(inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray], num_layers: int):
    next_inputs = {
        "input_ids": outputs["logits"].argmax(-1).astype(np.int64),
        "position_ids": np.max(inputs["position_ids"], axis=1, keepdims=True) + 1,
        "image_idx": outputs["image_idx_output"],
    }
    if "vision_embeds_RetainedState" in outputs:
        next_inputs["vision_embeds"] = outputs["vision_embeds_RetainedState"]
    else:
        next_inputs["vision_embeds"] = inputs["vision_embeds"]

    if "mm_token_type_ids" in inputs:
        next_inputs["mm_token_type_ids"] = np.zeros_like(
            next_inputs["input_ids"], dtype=inputs["mm_token_type_ids"].dtype
        )

    for i in range(num_layers):
        next_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
        next_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]
    return next_inputs


def run_dual_ort(model, onnx_paths, inputs, generation_len: int, ctx_len: int, prefill_seq_len: int, pad_token_id: int):
    vision_session = ort.InferenceSession(str(onnx_paths[0]))
    decoder_session = ort.InferenceSession(str(onnx_paths[1]))

    vision_outputs = run_ort_session(
        vision_session,
        {
            "pixel_values": inputs["pixel_values"].cpu().numpy(),
            "image_position_ids": inputs["image_position_ids"].cpu().numpy(),
        },
    )
    vision_embeds = vision_outputs["vision_embeds"]

    padded_inputs, num_chunks, _ = pad_prefill_inputs(inputs, prefill_seq_len, pad_token_id)
    decoder_state = build_initial_decoder_inputs(model, padded_inputs, vision_embeds, ctx_len=ctx_len)
    generated = []
    num_layers = model.model.model.language_model.config.num_hidden_layers

    chunk_image_idx = np.array([[0]], dtype=np.int64)
    outputs = None
    for chunk_idx in range(num_chunks):
        chunk_inputs = {
            "input_ids": decoder_state["input_ids"][:, chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len],
            "position_ids": decoder_state["position_ids"][
                :, chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
            ],
            "vision_embeds": decoder_state["vision_embeds"],
            "image_idx": chunk_image_idx,
            "mm_token_type_ids": decoder_state["mm_token_type_ids"][
                :, chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
            ],
        }
        for i in range(num_layers):
            chunk_inputs[f"past_key.{i}"] = decoder_state[f"past_key.{i}"]
            chunk_inputs[f"past_value.{i}"] = decoder_state[f"past_value.{i}"]

        outputs = run_ort_session(decoder_session, chunk_inputs)
        chunk_image_idx = outputs["image_idx_output"]
        if "vision_embeds_RetainedState" in outputs:
            decoder_state["vision_embeds"] = outputs["vision_embeds_RetainedState"]
        for i in range(num_layers):
            decoder_state[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
            decoder_state[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]

    if outputs is None:
        raise RuntimeError("No prefill chunks were executed.")

    decoder_inputs = {
        "input_ids": outputs["logits"].argmax(-1).astype(np.int64),
        "position_ids": np.max(decoder_state["position_ids"], axis=1, keepdims=True) + 1,
        "vision_embeds": decoder_state["vision_embeds"],
        "image_idx": chunk_image_idx,
        "mm_token_type_ids": np.zeros_like(
            outputs["logits"].argmax(-1), dtype=decoder_state["mm_token_type_ids"].dtype
        ),
    }
    for i in range(num_layers):
        decoder_inputs[f"past_key.{i}"] = decoder_state[f"past_key.{i}"]
        decoder_inputs[f"past_value.{i}"] = decoder_state[f"past_value.{i}"]

    generated.append(decoder_inputs["input_ids"].reshape(-1, 1))
    for _ in range(1, generation_len):
        outputs = run_ort_session(decoder_session, decoder_inputs)
        next_token = outputs["logits"].argmax(-1).astype(np.int64)
        generated.append(next_token.reshape(-1, 1))
        decoder_inputs = update_decoder_inputs(decoder_inputs, outputs, num_layers)

    return np.concatenate(generated, axis=1)


def run_hf_reference(model_name: str, inputs, generation_len: int):
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    )
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=generation_len, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    return model, outputs[:, prompt_len:].cpu().numpy()


def run_qeff_vision_exactness(model, hf_model, inputs):
    encoder = model.model.get_qeff_vision_encoder()
    with torch.inference_mode():
        hf_feats = hf_model.model.get_image_features(
            inputs["pixel_values"],
            inputs["image_position_ids"],
            return_dict=True,
        ).pooler_output.float()
        qeff_feats = encoder(inputs["pixel_values"], inputs["image_position_ids"])
        if qeff_feats.dim() == 3:
            qeff_feats = qeff_feats[0]
        qeff_feats = qeff_feats.float()
    diff = (hf_feats - qeff_feats).abs()
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_1e-6": bool(torch.allclose(hf_feats, qeff_feats, atol=1e-6, rtol=1e-6)),
        "allclose_1e-5": bool(torch.allclose(hf_feats, qeff_feats, atol=1e-5, rtol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(hf_feats, qeff_feats, atol=1e-4, rtol=1e-4)),
        "allclose_1e-3": bool(torch.allclose(hf_feats, qeff_feats, atol=1e-3, rtol=1e-3)),
    }


def main():
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    export_dir = args.work_dir / "export"
    compile_dir = args.work_dir / "compile"
    export_dir.mkdir(parents=True, exist_ok=True)
    compile_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    rendered_prompt, inputs = build_inputs(processor, args.image, args.prompt)
    prompt_len = inputs["input_ids"].shape[1]
    effective_ctx_len = resolve_effective_ctx_len(args.ctx_len, prompt_len, args.generation_len)
    effective_prefill_seq_len = resolve_effective_prefill_seq_len(args.prefill_seq_len, prompt_len)

    hf_model, hf_ids = run_hf_reference(args.model_name, inputs, args.generation_len)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        kv_offload=True,
        dtype="float32",
    )

    vision_exactness = run_qeff_vision_exactness(qeff_model, hf_model, inputs)

    onnx_paths = qeff_model.export(
        export_dir=str(export_dir),
        use_onnx_subfunctions=True,
        vision_use_onnx_subfunctions=False,
        lang_use_onnx_subfunctions=True,
    )

    ort_ids = run_dual_ort(
        qeff_model,
        onnx_paths,
        inputs,
        args.generation_len,
        ctx_len=effective_ctx_len,
        prefill_seq_len=effective_prefill_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )

    result = {
        "model_name": args.model_name,
        "rendered_prompt": rendered_prompt,
        "prompt_len": prompt_len,
        "requested_ctx_len": args.ctx_len,
        "effective_ctx_len": effective_ctx_len,
        "requested_prefill_seq_len": args.prefill_seq_len,
        "effective_prefill_seq_len": effective_prefill_seq_len,
        "onnx_paths": [str(x) for x in onnx_paths],
        "vision_exactness": vision_exactness,
        "hf_ids": hf_ids.tolist(),
        "ort_ids": ort_ids.tolist(),
        "hf_text": decode_new_tokens(tokenizer, hf_ids),
        "ort_text": decode_new_tokens(tokenizer, ort_ids),
        "ort_match": bool(np.array_equal(hf_ids, ort_ids)),
    }

    if args.run_qaic:
        compile_kwargs = dict(
            vision_onnx_path=str(onnx_paths[0]),
            lang_onnx_path=str(onnx_paths[1]),
            compile_dir=str(compile_dir),
            prefill_seq_len=effective_prefill_seq_len,
            ctx_len=effective_ctx_len,
            num_cores=args.num_cores,
            num_devices=1 if args.device_group is None else len(args.device_group),
            use_onnx_subfunctions=True,
            vision_use_onnx_subfunctions=False,
            lang_use_onnx_subfunctions=True,
            node_precision_info=args.enable_npi,
            vision_node_precision_info=args.enable_vision_npi,
        )
        qpc_paths = qeff_model.compile(**compile_kwargs)
        exec_info = qeff_model.generate(inputs=inputs, generation_len=args.generation_len, device_ids=args.device_group)
        qaic_ids = np.asarray(exec_info.generated_ids)
        if qaic_ids.ndim == 1:
            qaic_ids = qaic_ids.reshape(1, -1)
        elif qaic_ids.ndim > 2:
            qaic_ids = qaic_ids.reshape(qaic_ids.shape[0], -1)
        qaic_ids = qaic_ids[:, : args.generation_len]
        result.update(
            {
                "qpc_paths": [str(x) for x in qpc_paths] if isinstance(qpc_paths, (list, tuple)) else str(qpc_paths),
                "qaic_ids": qaic_ids.tolist(),
                "qaic_text": decode_new_tokens(tokenizer, qaic_ids),
                "qaic_match": bool(np.array_equal(hf_ids, qaic_ids)),
            }
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
