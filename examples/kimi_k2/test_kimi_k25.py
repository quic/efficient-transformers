# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
from io import BytesIO

import requests
import torch
from PIL import Image

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.load_kimi_utils import (
    KIMI_K25_MODEL_NAME,
    LOADED_EXPERT_IDS,
    NUM_EXPERTS_PER_TOKEN,
    load_kimi_k25_class,
)
from QEfficient.utils.load_kimi_utils import (
    ensure_torch_fx_import_compatibility as _ensure_torch_fx_import_compatibility,
)
from QEfficient.utils.load_kimi_utils import (
    load_layer_subset_model as _load_layer_subset_model,
)
from QEfficient.utils.load_kimi_utils import (
    prepare_config as _prepare_config,
)
from QEfficient.utils.load_kimi_utils import (
    resolve_model_path as _resolve_model_path,
)
from QEfficient.utils.load_kimi_utils import (
    set_deterministic as _set_deterministic,
)

MODEL_NAME = KIMI_K25_MODEL_NAME
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


def _greedy_generate_qeff_wrapper(transformed_model, inputs, max_new_tokens: int):
    qeff_encoder = transformed_model.get_qeff_vision_encoder().eval()
    decoder_wrapper = transformed_model.get_qeff_language_decoder().eval()

    grid_thws = inputs["grid_thws"].to(torch.long)
    h_shape = torch.ones(int(grid_thws[0, 1].item()), dtype=torch.int64)
    w_shape = torch.ones(int(grid_thws[0, 2].item()), dtype=torch.int64)
    vision_embeds = qeff_encoder(inputs["pixel_values"].to(torch.float32), h_shape, w_shape).detach()

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
            vision_embeds=vision_embeds,
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


def check_kimi_k25_pytorch_vs_ai100():
    _set_deterministic(1234)
    _ensure_torch_fx_import_compatibility()
    model_path = _resolve_model_path()
    config = _prepare_config(model_path)
    kimi_cls = load_kimi_k25_class(model_path)

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

    merge_kernel_size = getattr(model.config.vision_config, "merge_kernel_size", (2, 2))
    if isinstance(merge_kernel_size, int):
        kernel_height = kernel_width = merge_kernel_size
        merge_kernel_size = (merge_kernel_size, merge_kernel_size)
    else:
        kernel_height, kernel_width = merge_kernel_size

    num_patches = int(inputs["pixel_values"].shape[0])
    h = int(inputs["grid_thws"][0, 1].item())
    w = int(inputs["grid_thws"][0, 2].item())
    num_image_tokens = int(inputs["pixel_values"].shape[0] // (kernel_height * kernel_width))

    qeff_model.compile(
        # qaic_config=qaic_config,
        num_devices=1,
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        mxfp6=False,
        num_patches=num_patches,
        h=h,
        w=w,
        num_image_tokens=num_image_tokens,
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
    print("QAIC:", _decode_tokens(tokenizer, qaic_tokens) if qaic_tokens is not None else "<qaic skipped>")

    assert torch.equal(hf_tokens, qeff_tokens), "HF and QEFF(Pytorch runtime) tokens do not match"
    if qaic_tokens is not None:
        assert torch.equal(hf_tokens, torch.as_tensor(qaic_tokens)), "HF and QAIC tokens do not match"


def test_kimi_k25_image_text_to_text_pytorch_vs_ai100():
    check_kimi_k25_pytorch_vs_ai100()
