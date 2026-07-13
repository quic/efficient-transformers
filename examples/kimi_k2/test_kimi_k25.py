# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import os
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import requests
import torch
from export_kimi_k25_vision import (
    LOADED_EXPERT_IDS,
    NUM_EXPERTS_PER_TOKEN,
    _ensure_torch_fx_import_compatibility,
    _load_layer_subset_model,
    _patch_deepseek_init_weights_compat,
    _patch_kimi_tie_weights_compat,
    _prepare_config,
)
from huggingface_hub import snapshot_download
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_NAME = "moonshotai/Kimi-K2.5"
IMAGE_URL = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png"
TEXT_PROMPT = "Describe this image."
NUM_VISION_LAYERS = 2
NUM_TEXT_LAYERS = 2
NEW_GENERATION_TOKENS = 10
CTX_LEN = 1024


def _has_qaic_runtime_access() -> bool:
    try:
        _ = QAICInferenceSession
    except Exception:
        return False
    try:
        import qaicrt

        _ctx = qaicrt.Context()
        return True
    except Exception:
        return False


def _set_deterministic(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _resolve_model_path() -> Path:
    return Path(snapshot_download(repo_id=MODEL_NAME, cache_dir=os.environ.get("HF_HUB_CACHE")))


def _prepare_inputs(processor):
    image = Image.open(BytesIO(requests.get(IMAGE_URL, timeout=30).content)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]
    return processor(
        messages=messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )


def _decode_tokens(tokenizer, token_ids: torch.Tensor) -> str:
    decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return decoded[0] if decoded else ""


def _clone_inputs(inputs):
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in inputs.items()}


def _greedy_generate_hf(model, inputs, max_new_tokens: int):
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    grid_thws = inputs["grid_thws"]

    eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None and hasattr(model.config, "text_config"):
        eos_token_id = getattr(model.config.text_config, "eos_token_id", None)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )

        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break

    return generated_ids[:, -max_new_tokens:]


def _match_indexed_name(name: str, base: str):
    pattern = rf"^{re.escape(base)}\.(\d+)$"
    match = re.match(pattern, name)
    return int(match.group(1)) if match else None


def _build_cache_inputs_from_dummy(transformed_model, input_names, batch_size: int):
    pkv = transformed_model.language_model.get_dummy_pkv_cache(
        transformed_model.config.text_config,
        batch_size,
        CTX_LEN,
    )
    cache_inputs = {}
    for name in input_names:
        key_idx = _match_indexed_name(name, "past_key")
        value_idx = _match_indexed_name(name, "past_value")
        compressed_idx = _match_indexed_name(name, "compressed_kv")
        k_pe_idx = _match_indexed_name(name, "k_pe")

        if key_idx is not None:
            cache_inputs[name] = pkv[key_idx][0].detach().cpu().numpy().astype(np.float32)
        elif value_idx is not None:
            cache_inputs[name] = pkv[value_idx][1].detach().cpu().numpy().astype(np.float32)
        elif compressed_idx is not None:
            cache_inputs[name] = pkv[compressed_idx][0].detach().cpu().numpy().astype(np.float32)
        elif k_pe_idx is not None:
            cache_inputs[name] = pkv[k_pe_idx][1].detach().cpu().numpy().astype(np.float32)
    return cache_inputs


def _greedy_generate_qeff_wrapper(transformed_model, inputs, max_new_tokens: int):
    qeff_encoder = transformed_model.get_qeff_vision_encoder().eval()
    decoder_wrapper = transformed_model.get_qeff_language_decoder().eval()

    grid_thws = inputs["grid_thws"].to(torch.long)
    h_shape = torch.ones(int(grid_thws[0, 1].item()), dtype=torch.int64)
    w_shape = torch.ones(int(grid_thws[0, 2].item()), dtype=torch.int64)
    image_embeds = qeff_encoder(inputs["pixel_values"].to(torch.float32), h_shape, w_shape).detach()

    generated_ids = inputs["input_ids"].to(torch.long)
    attention_mask = inputs["attention_mask"].to(torch.long)

    for _ in range(max_new_tokens):
        position_ids = torch.where(attention_mask.bool(), attention_mask.cumsum(-1) - 1, -1)
        past_key_values = transformed_model.language_model.get_dummy_pkv_cache(
            transformed_model.config.text_config,
            generated_ids.shape[0],
            CTX_LEN,
        )
        logits = decoder_wrapper(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_embeds=image_embeds,
            image_idx=torch.zeros((generated_ids.shape[0], 1), dtype=torch.int64),
            past_key_values=past_key_values,
            use_cache=True,
        )[0]
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
            dim=1,
        )

    return generated_ids[:, -max_new_tokens:]


def _greedy_generate_onnx(transformed_model, onnx_paths, inputs, max_new_tokens, session_options):
    vision_onnx_path, lang_onnx_path = onnx_paths
    vision_session = ort.InferenceSession(str(vision_onnx_path))
    lang_session = ort.InferenceSession(str(lang_onnx_path), session_options)

    pixel_values = inputs["pixel_values"].detach().cpu().numpy().astype(np.float32)
    grid_thws = inputs["grid_thws"].detach().cpu().numpy().astype(np.int64)
    h = int(grid_thws[0, 1])
    w = int(grid_thws[0, 2])

    vision_outputs = vision_session.run(
        None,
        {
            "pixel_values": pixel_values,
            "h_shape": np.ones((h,), dtype=np.int64),
            "w_shape": np.ones((w,), dtype=np.int64),
        },
    )
    vision_output_names = [out.name for out in vision_session.get_outputs()]
    image_embeds = {name: value for name, value in zip(vision_output_names, vision_outputs)}.get(
        "image_embeds", vision_outputs[0]
    )

    lang_input_names = [meta.name for meta in lang_session.get_inputs()]
    lang_output_names = [meta.name for meta in lang_session.get_outputs()]

    prompt_input_ids = inputs["input_ids"].detach().cpu().numpy().astype(np.int64)
    prompt_attention_mask = inputs["attention_mask"].detach().cpu().numpy().astype(np.int64)
    prompt_len = prompt_input_ids.shape[1]

    prefill_seq_len = 32
    num_chunks = -(prompt_len // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len

    pad_token_id = 1
    padded_input_ids = np.pad(prompt_input_ids, ((0, 0), (0, padded_len - prompt_len)), constant_values=pad_token_id)
    padded_attention = np.pad(prompt_attention_mask, ((0, 0), (0, padded_len - prompt_len)), constant_values=0)
    padded_position_ids = np.where(padded_attention > 0, np.arange(padded_len), -1).astype(np.int64)

    cache_inputs = _build_cache_inputs_from_dummy(
        transformed_model=transformed_model,
        input_names=lang_input_names,
        batch_size=prompt_input_ids.shape[0],
    )
    image_idx = np.zeros((prompt_input_ids.shape[0], 1), dtype=np.int64)

    output_map = None
    for chunk_idx in range(num_chunks):
        start = chunk_idx * prefill_seq_len
        end = (chunk_idx + 1) * prefill_seq_len
        ort_inputs = {
            "input_ids": padded_input_ids[:, start:end],
            "position_ids": padded_position_ids[:, start:end],
            "image_embeds": image_embeds.astype(np.float32),
            "image_idx": image_idx,
            **cache_inputs,
        }
        ort_outputs = lang_session.run(None, ort_inputs)
        output_map = {name: value for name, value in zip(lang_output_names, ort_outputs)}
        if "image_idx_output" in output_map:
            image_idx = output_map["image_idx_output"].astype(np.int64)
        for key in list(cache_inputs.keys()):
            retained_name = f"{key}_RetainedState"
            if retained_name in output_map:
                cache_inputs[key] = output_map[retained_name]

    if output_map is None:
        raise RuntimeError("ONNX prefill did not execute.")

    next_token = np.argmax(output_map["logits"][:, -1, :], axis=-1, keepdims=True).astype(np.int64)
    generated_new_tokens = [next_token]

    decode_input_ids = next_token
    decode_position_ids = np.max(padded_position_ids, axis=1, keepdims=True).astype(np.int64) + 1

    for _ in range(1, max_new_tokens):
        ort_inputs = {
            "input_ids": decode_input_ids,
            "position_ids": decode_position_ids,
            "image_embeds": image_embeds.astype(np.float32),
            "image_idx": image_idx,
            **cache_inputs,
        }
        ort_outputs = lang_session.run(None, ort_inputs)
        output_map = {name: value for name, value in zip(lang_output_names, ort_outputs)}

        decode_input_ids = np.argmax(output_map["logits"][:, -1, :], axis=-1, keepdims=True).astype(np.int64)
        generated_new_tokens.append(decode_input_ids)
        decode_position_ids = decode_position_ids + 1

        if "image_idx_output" in output_map:
            image_idx = output_map["image_idx_output"].astype(np.int64)
        for key in list(cache_inputs.keys()):
            retained_name = f"{key}_RetainedState"
            if retained_name in output_map:
                cache_inputs[key] = output_map[retained_name]

    return torch.from_numpy(np.concatenate(generated_new_tokens, axis=1))


def check_kimi_k25_pytorch_vs_ai100():
    _set_deterministic(1234)
    _ensure_torch_fx_import_compatibility()
    model_path = _resolve_model_path()
    config = _prepare_config(model_path)
    kimi_cls = get_class_from_dynamic_module("modeling_kimi_k25.KimiK25ForConditionalGeneration", str(model_path))
    _patch_kimi_tie_weights_compat(kimi_cls)
    _patch_deepseek_init_weights_compat(kimi_cls)

    model, tokenizer, processor = _load_layer_subset_model(
        model_path=model_path,
        kimi_cls=kimi_cls,
        config=config,
        num_vision_layers=NUM_VISION_LAYERS,
        num_text_layers=NUM_TEXT_LAYERS,
        loaded_expert_ids=LOADED_EXPERT_IDS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOKEN,
        dtype=torch.float32,
    )

    model.vision_tower.patch_embed.pos_emb.interpolation_mode = "bilinear"
    model = model.eval().to("cpu")

    inputs = _prepare_inputs(processor)
    inputs = {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in inputs.items()}

    hf_tokens = _greedy_generate_hf(copy.deepcopy(model), _clone_inputs(inputs), max_new_tokens=NEW_GENERATION_TOKENS)
    print("HF:", _decode_tokens(tokenizer, hf_tokens), "\n", hf_tokens)

    # qaic_config = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}

    qeff_model = QEFFAutoModelForImageTextToText(model)  # , qaic_config=qaic_config)

    qeff_tokens = _greedy_generate_qeff_wrapper(
        transformed_model=qeff_model.model,
        inputs=_clone_inputs(inputs),
        max_new_tokens=NEW_GENERATION_TOKENS,
    )
    print("QEFF:", _decode_tokens(tokenizer, qeff_tokens), "\n", qeff_tokens)

    onnx_paths = qeff_model.export()

    # Replace invalid index value for INT32 max to 0 using add_initializer
    m = onnx.load(onnx_paths[1], load_external_data=False)
    # NOTE: OrtValue objects should be kept around until the session is run, hence this dict is required
    added_initializers = {}
    for node in m.graph.node:
        if node.op_type == "Constant":
            np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(onnx_paths[1]))
            if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                added_initializers[node.output[0]] = ort.OrtValue.ortvalue_from_numpy(np.array(0, np_tensor.dtype))

    session_options = ort.SessionOptions()
    for name, value in added_initializers.items():
        session_options.add_initializer(name, value)

    try:
        onnx_tokens = _greedy_generate_onnx(
            transformed_model=qeff_model.model,
            onnx_paths=onnx_paths,
            inputs=_clone_inputs(inputs),
            max_new_tokens=NEW_GENERATION_TOKENS,
            session_options=session_options,
        )
        print("ONNX:", _decode_tokens(tokenizer, onnx_tokens) if onnx_tokens is not None else "<onnx failed>")
    except Exception as exc:
        print(f"ONNX generation failed: {exc}")
        onnx_tokens = None

    qeff_model.compile(
        # qaic_config=qaic_config,
        num_devices=1,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        mxfp6=False,
        num_patches=int(inputs["pixel_values"].shape[0]),
        h=int(inputs["grid_thws"][0, 1].item()),
        w=int(inputs["grid_thws"][0, 2].item()),
        num_image_tokens=600,
        num_cores=16,
    )

    qaic_tokens = None
    if _has_qaic_runtime_access():
        inputs["pixel_values"] = inputs["pixel_values"].to(qeff_model.model.config.torch_dtype)
        qaic_tokens = qeff_model.generate(
            inputs=_clone_inputs(inputs), generation_len=NEW_GENERATION_TOKENS
        ).generated_ids[:, :-1]
        print("QAIC:", _decode_tokens(tokenizer, qaic_tokens) if qaic_tokens is not None else "<qaic skipped>")
    else:
        print("QAIC generation skipped: no QAIC runtime access (user/group permissions).")

    print(
        "Loaded subset:",
        f"vision_layers={model.config.vision_config.vt_num_hidden_layers}",
        f"text_layers={model.config.text_config.num_hidden_layers}",
    )

    print("Prompt:", repr(TEXT_PROMPT))
    print("HF:", _decode_tokens(tokenizer, hf_tokens))
    print("QEFF:", _decode_tokens(tokenizer, qeff_tokens))
    print("ONNX:", _decode_tokens(tokenizer, onnx_tokens) if onnx_tokens is not None else "<onnx failed>")
    print("QAIC:", _decode_tokens(tokenizer, qaic_tokens) if qaic_tokens is not None else "<qaic skipped>")

    assert torch.equal(hf_tokens, qeff_tokens), "HF and QEFF(Pytorch runtime) tokens do not match"
    if onnx_tokens is not None:
        assert torch.equal(hf_tokens, onnx_tokens), "HF and ONNXRuntime tokens do not match"
    if qaic_tokens is not None:
        assert torch.equal(hf_tokens, torch.as_tensor(qaic_tokens)), "HF and QAIC tokens do not match"


def test_kimi_k25_image_text_to_text_pytorch_vs_ai100():
    check_kimi_k25_pytorch_vs_ai100()
