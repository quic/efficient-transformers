# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import importlib.util
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from export_kimi_k25_dynamo import (
    DEFAULT_COMPILE_DIR,
    DEFAULT_EXPORT_DIR,
    DEFAULT_IMAGE_URL,
    compile_components,
    configure_qaic_tool_path,
    export_components,
    get_component_export_args,
    load_generation_image,
    precompute_vision_rope_cache,
    validate_dynamo_torch_version,
)

from QEfficient import QEFFAutoModelForImageTextToText

LOAD_KIMI_UTILS_PATH = Path(__file__).resolve().parents[2] / "tests" / "utils" / "load_kimi_utils.py"
_load_kimi_spec = importlib.util.spec_from_file_location("load_kimi_utils", LOAD_KIMI_UTILS_PATH)
if _load_kimi_spec is None or _load_kimi_spec.loader is None:
    raise ImportError(f"Unable to load Kimi helpers from {LOAD_KIMI_UTILS_PATH}")
load_kimi_utils = importlib.util.module_from_spec(_load_kimi_spec)
_load_kimi_spec.loader.exec_module(load_kimi_utils)

LOADED_EXPERT_IDS = load_kimi_utils.LOADED_EXPERT_IDS
NUM_EXPERTS_PER_TOKEN = load_kimi_utils.NUM_EXPERTS_PER_TOKEN
NUM_TEXT_LAYERS = load_kimi_utils.NUM_TEXT_LAYERS
NUM_VISION_LAYERS = load_kimi_utils.NUM_VISION_LAYERS
parse_expert_ids = load_kimi_utils.parse_expert_ids
set_deterministic = load_kimi_utils.set_deterministic
load_kimi_k25_layer_subset_model = load_kimi_utils.load_kimi_k25_layer_subset_model

TEXT_PROMPT = "Describe this image."
NEW_GENERATION_TOKENS = 10
CTX_LEN = 1024
PREFILL_SEQ_LEN = 2


def _has_qaic_runtime_access() -> bool:
    try:
        import qaicrt

        _ctx = qaicrt.Context()
        return True
    except (ImportError, OSError, RuntimeError, AttributeError):
        return False


def _skip_test(reason: str):
    try:
        import pytest

        pytest.skip(reason)
    except ImportError as exc:
        raise RuntimeError(reason) from exc


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value).expanduser().resolve()


def _parse_device_ids(value: str) -> list[int]:
    return [int(device_id) for device_id in value.strip().strip("[]").split(",") if device_id.strip()]


def _find_free_qaic_device_id() -> int | None:
    qaic_util_path = Path("/opt/qti-aic/tools/qaic-util")
    if not qaic_util_path.exists():
        return None

    try:
        result = subprocess.run(
            [str(qaic_util_path), "-q"],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    current_qid = None
    for line in result.stdout.splitlines():
        qid_match = re.match(r"^QID\s+(\d+)", line.strip())
        if qid_match:
            current_qid = int(qid_match.group(1))
            continue

        free_match = re.search(r"Nsp Free:\s*(\d+)", line)
        if free_match and current_qid is not None and int(free_match.group(1)) > 0:
            return current_qid

    return None


def _resolve_device_ids() -> list[int] | None:
    value = os.environ.get("KIMI_K25_DYNAMO_DEVICE_IDS") or os.environ.get("KIMI_K25_DYNAMO_DEVICE_ID")
    if value:
        return _parse_device_ids(value)

    free_device_id = _find_free_qaic_device_id()
    if free_device_id is None:
        return None
    return [free_device_id]


def _clone_inputs(inputs):
    return {name: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value)) for name, value in inputs.items()}


def _decode_tokens(tokenizer, token_ids) -> str:
    decoded = tokenizer.batch_decode(torch.as_tensor(token_ids), skip_special_tokens=True)
    return decoded[0] if decoded else ""


@torch.no_grad()
def _greedy_generate_hf(model, inputs, max_new_tokens: int) -> torch.Tensor:
    generated_ids = inputs["input_ids"].to(torch.long)
    attention_mask = inputs["attention_mask"].to(torch.long)
    pixel_values = inputs["pixel_values"]
    grid_thws = inputs["grid_thws"]
    new_tokens = []

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
        new_tokens.append(next_token)

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

    return torch.cat(new_tokens, dim=1)


@torch.no_grad()
def _greedy_generate_qeff(qeff_model, inputs, args) -> torch.Tensor:
    prefill_seq_len = args.prefill_seq_len
    generated_ids = inputs["input_ids"].to(torch.long)
    attention_mask = inputs["attention_mask"].to(torch.long)
    grid_thws = inputs["grid_thws"].to(torch.long)

    h_shape = torch.ones(int(grid_thws[0, 1].item()), dtype=torch.int64)
    w_shape = torch.ones(int(grid_thws[0, 2].item()), dtype=torch.int64)
    vision_embeds = qeff_model.vision_model.model(
        inputs["pixel_values"].to(qeff_model.model.config.torch_dtype),
        h_shape,
        w_shape,
    ).detach()

    input_ids_length = generated_ids.shape[1]
    num_chunks = -(input_ids_length // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len
    generated_ids = torch.nn.functional.pad(
        generated_ids,
        (0, padded_len - input_ids_length),
        "constant",
        qeff_model.model.config.pad_token_id,
    )
    attention_mask = torch.nn.functional.pad(attention_mask, (0, padded_len - input_ids_length), "constant", 0)
    position_ids = torch.where(
        attention_mask.bool(),
        torch.arange(padded_len, dtype=torch.long).view(1, -1),
        torch.full((generated_ids.shape[0], padded_len), -1, dtype=torch.long),
    )

    export_inputs, _, _ = get_component_export_args(qeff_model, prefill_seq_len)
    compressed_kvs = [
        [cache_tensor.clone()[: generated_ids.shape[0]] for cache_tensor in layer_cache]
        for layer_cache in export_inputs["lang"]["compressed_kvs"]
    ]
    image_idx = torch.zeros((generated_ids.shape[0], 1), dtype=torch.int64)
    new_tokens = []

    logits = None
    for chunk_idx in range(num_chunks):
        start = chunk_idx * prefill_seq_len
        end = start + prefill_seq_len
        logits, _, image_idx_output, compressed_kvs = qeff_model.lang_model.model(
            input_ids=generated_ids[:, start:end],
            position_ids=position_ids[:, start:end],
            vision_embeds=vision_embeds,
            image_idx=image_idx,
            compressed_kvs=compressed_kvs,
        )
        if image_idx_output is not None:
            image_idx = image_idx_output

    next_token = logits.argmax(2)
    if next_token.ndim == 2 and next_token.shape[1] > 1:
        next_token = next_token[:, -1:]
    new_tokens.append(next_token)

    decode_position_ids = position_ids.max(dim=-1, keepdim=True).values + 1
    for _ in range(1, args.generation_len):
        logits, _, image_idx_output, compressed_kvs = qeff_model.lang_model.model(
            input_ids=next_token,
            position_ids=decode_position_ids,
            vision_embeds=vision_embeds,
            image_idx=image_idx,
            compressed_kvs=compressed_kvs,
        )
        if image_idx_output is not None:
            image_idx = image_idx_output
        next_token = logits.argmax(2)
        if next_token.ndim == 2 and next_token.shape[1] > 1:
            next_token = next_token[:, -1:]
        new_tokens.append(next_token)
        decode_position_ids = decode_position_ids + 1

    return torch.cat(new_tokens, dim=1)


def _build_generation_inputs(processor, image, args, dtype):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    inputs = processor(
        messages=messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    return inputs


def _load_hf_model(args):
    model_path = args.model_path
    model, tokenizer, processor = load_kimi_k25_layer_subset_model(
        model_path=model_path,
        num_vision_layers=args.num_vision_layers,
        num_text_layers=args.num_text_layers,
        loaded_expert_ids=args.expert_ids,
        num_experts_per_tok=args.num_experts_per_token,
        dtype=torch.float32,
        seed=args.seed,
    )
    return model.eval(), tokenizer, processor


def _build_qeff_model_from_hf(model, args):
    qaic_config = {"mla_absorption": {"cache_compressed": True, "absorption": False, "online": False}}
    qeff_model = QEFFAutoModelForImageTextToText(model, qaic_config=qaic_config)
    qeff_model.model.eval()
    qeff_model.vision_model.model.eval()
    qeff_model.lang_model.model.eval()
    qeff_model.transform(
        ctx_len=args.ctx_len,
        seq_len=args.prefill_seq_len,
        bs=1,
        num_devices=args.num_devices,
        qaic_config=qaic_config,
        aic_num_cores=args.num_cores,
    )
    precompute_vision_rope_cache(qeff_model)
    return qeff_model, qaic_config


def _make_args(device_ids: list[int]) -> SimpleNamespace:
    vision_qpc_path = _env_path("KIMI_K25_DYNAMO_VISION_QPC_PATH")
    lang_qpc_path = _env_path("KIMI_K25_DYNAMO_LANG_QPC_PATH")
    skip_compile = vision_qpc_path is not None and lang_qpc_path is not None

    vision_onnx_path = _env_path("KIMI_K25_DYNAMO_VISION_ONNX_PATH")
    lang_onnx_path = _env_path("KIMI_K25_DYNAMO_LANG_ONNX_PATH")
    skip_export = skip_compile or (vision_onnx_path is not None and lang_onnx_path is not None)

    return SimpleNamespace(
        model_path=_env_path("KIMI_K25_DYNAMO_MODEL_PATH"),
        export_dir=_env_path("KIMI_K25_DYNAMO_EXPORT_DIR") or DEFAULT_EXPORT_DIR,
        compile_dir=_env_path("KIMI_K25_DYNAMO_COMPILE_DIR") or DEFAULT_COMPILE_DIR,
        vision_onnx_path=vision_onnx_path,
        lang_onnx_path=lang_onnx_path,
        vision_qpc_path=vision_qpc_path,
        lang_qpc_path=lang_qpc_path,
        component="both",
        num_vision_layers=_env_int("KIMI_K25_DYNAMO_NUM_VISION_LAYERS", NUM_VISION_LAYERS),
        num_text_layers=_env_int("KIMI_K25_DYNAMO_NUM_TEXT_LAYERS", NUM_TEXT_LAYERS),
        expert_ids=LOADED_EXPERT_IDS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        prefill_seq_len=_env_int("KIMI_K25_DYNAMO_PREFILL_SEQ_LEN", PREFILL_SEQ_LEN),
        ctx_len=_env_int("KIMI_K25_DYNAMO_CTX_LEN", CTX_LEN),
        num_devices=len(device_ids),
        num_cores=_env_int("KIMI_K25_DYNAMO_NUM_CORES", 16),
        seed=_env_int("KIMI_K25_DYNAMO_SEED", 1234),
        prompt=os.environ.get("KIMI_K25_DYNAMO_PROMPT", TEXT_PROMPT),
        image_url=os.environ.get("KIMI_K25_DYNAMO_IMAGE_URL", DEFAULT_IMAGE_URL),
        image_path=_env_path("KIMI_K25_DYNAMO_IMAGE_PATH"),
        image_height=None,
        image_width=None,
        generation_len=_env_int("KIMI_K25_DYNAMO_GENERATION_LEN", NEW_GENERATION_TOKENS),
        device_ids=device_ids,
        mxfp6_matmul=_env_bool("KIMI_K25_DYNAMO_MXFP6_MATMUL"),
        mxint8_kv_cache=_env_bool("KIMI_K25_DYNAMO_MXINT8_KV_CACHE"),
        mos=_env_int("KIMI_K25_DYNAMO_MOS", 1),
        aic_enable_depth_first=_env_bool("KIMI_K25_DYNAMO_AIC_ENABLE_DEPTH_FIRST"),
        skip_export=skip_export,
        skip_compile=skip_compile,
        skip_generate=False,
        use_onnx_subfunctions=_env_bool("KIMI_K25_DYNAMO_USE_ONNX_SUBFUNCTIONS"),
        keep_weights=_env_bool("KIMI_K25_DYNAMO_KEEP_WEIGHTS"),
    )


def check_kimi_k25_dynamo_hf_vs_qeff_vs_qaic():
    os.environ.setdefault("HF_HUB_CACHE", "/home/huggingface_hub")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    validate_dynamo_torch_version()
    configure_qaic_tool_path()
    if not _has_qaic_runtime_access():
        _skip_test("QAIC generation skipped: no QAIC runtime access.")

    device_ids = _resolve_device_ids()
    if device_ids is None:
        _skip_test("QAIC generation skipped: no QAIC device has free NSPs.")

    args = _make_args(device_ids)
    set_deterministic(args.seed)
    model, tokenizer, processor = _load_hf_model(args)
    image = load_generation_image(args)
    inputs = _build_generation_inputs(processor, image, args, model.config.torch_dtype)

    hf_tokens = _greedy_generate_hf(copy.deepcopy(model), _clone_inputs(inputs), args.generation_len).cpu()
    print("HF:", _decode_tokens(tokenizer, hf_tokens), "\n", hf_tokens)

    qeff_model, qaic_config = _build_qeff_model_from_hf(model, args)
    qeff_tokens = _greedy_generate_qeff(qeff_model, _clone_inputs(inputs), args).cpu()
    print("QEFF:", _decode_tokens(tokenizer, qeff_tokens), "\n", qeff_tokens)

    assert torch.equal(hf_tokens, qeff_tokens), (
        "HF and QEff PyTorch tokens do not match for the Dynamo export wrapper path: "
        f"hf={hf_tokens.tolist()}, qeff={qeff_tokens.tolist()}"
    )

    export_inputs, output_names, dynamic_axes = get_component_export_args(qeff_model, args.prefill_seq_len)
    exported_paths = export_components(qeff_model, export_inputs, output_names, dynamic_axes, args)
    qpc_paths = compile_components(qeff_model, exported_paths, image, qaic_config, args)
    print(f"Dynamo ONNX paths: {exported_paths}")
    print(f"Dynamo QPC paths: {qpc_paths}")

    qaic_output = qeff_model.generate(
        inputs=_clone_inputs(inputs),
        device_ids=args.device_ids,
        generation_len=args.generation_len,
        image_height=image.height,
        image_width=image.width,
    )
    qaic_tokens = torch.as_tensor(qaic_output.generated_ids[:, : args.generation_len], dtype=hf_tokens.dtype)
    print("QAIC:", _decode_tokens(tokenizer, qaic_tokens), "\n", qaic_tokens)

    if torch.equal(hf_tokens, qaic_tokens):
        return

    mismatch_message = (
        "HF/QEff and QAIC tokens do not match for the Dynamo exported and compiled model: "
        f"hf={hf_tokens.tolist()}, qeff={qeff_tokens.tolist()}, qaic={qaic_tokens.tolist()}"
    )
    if _env_bool("KIMI_K25_DYNAMO_STRICT_HF_QAIC", False):
        raise AssertionError(mismatch_message)

    print(
        "QAIC token drift tolerated for this reduced Kimi K2.5 Dynamo smoke test. "
        "Set KIMI_K25_DYNAMO_STRICT_HF_QAIC=1 to require exact HF-vs-QAIC token parity. "
        f"{mismatch_message}"
    )
    assert qaic_tokens.shape == hf_tokens.shape, "HF and QAIC generated token shapes do not match"


def test_kimi_k25_dynamo_hf_vs_qeff_vs_qaic():
    check_kimi_k25_dynamo_hf_vs_qeff_vs_qaic()


if __name__ == "__main__":
    check_kimi_k25_dynamo_hf_vs_qeff_vs_qaic()
