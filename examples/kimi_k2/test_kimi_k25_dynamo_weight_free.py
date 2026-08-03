# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import onnx
import torch
from export_kimi_k25_dynamo import (
    DEFAULT_COMPILE_DIR,
    DEFAULT_EXPORT_DIR,
    DEFAULT_IMAGE_URL,
    build_generation_inputs,
    build_qeff_model,
    compile_components,
    configure_qaic_tool_path,
    export_components,
    get_component_export_args,
    load_generation_image,
    validate_dynamo_torch_version,
)
from onnx import numpy_helper
from safetensors.torch import save_file

from QEfficient.generation.cloud_infer import QAICInferenceSession

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


class _QEffHolder:
    def __init__(self, model):
        self.model = model


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


def _env_path_any(*names: str) -> Path | None:
    for name in names:
        path = _env_path(name)
        if path is not None:
            return path
    return None


def _env_int_any(default: int, *names: str) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def _env_bool_any(default: bool, *names: str) -> bool:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _parse_device_ids(value: str) -> list[int]:
    return [int(device_id) for device_id in value.strip().strip("[]").split(",") if device_id.strip()]


def _find_free_qaic_device_id() -> int | None:
    candidate_paths = [Path("/opt/qti-aic/tools/qaic-util"), Path("/opt/qti-aic/exec/qaic-util")]
    qaic_util_path = next((path for path in candidate_paths if path.exists()), None)
    if qaic_util_path is None:
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


def _device_can_activate(qpc_path: Path, device_id: int) -> bool:
    session = None
    try:
        session = QAICInferenceSession(str(qpc_path), [device_id])
        session.deactivate()
        return True
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError):
        return False
    finally:
        del session


def _find_activatable_device_id(vision_qpc_path: Path, lang_qpc_path: Path, max_device_id: int) -> int | None:
    for device_id in range(max_device_id + 1):
        if _device_can_activate(vision_qpc_path, device_id) and _device_can_activate(lang_qpc_path, device_id):
            return device_id
    return None


def _resolve_device_ids(args: SimpleNamespace | None = None) -> list[int] | None:
    value = (
        os.environ.get("KIMI_K25_DYNAMO_WEIGHT_FREE_DEVICE_IDS")
        or os.environ.get("KIMI_K25_DYNAMO_WEIGHT_FREE_DEVICE_ID")
        or os.environ.get("KIMI_K25_DYNAMO_DEVICE_IDS")
        or os.environ.get("KIMI_K25_DYNAMO_DEVICE_ID")
    )
    if value:
        return _parse_device_ids(value)

    if args is not None and args.vision_qpc_path is not None and args.lang_qpc_path is not None:
        device_id = _find_activatable_device_id(
            args.vision_qpc_path,
            args.lang_qpc_path,
            _env_int("KIMI_K25_DYNAMO_WEIGHT_FREE_MAX_DEVICE_ID", 63),
        )
        if device_id is not None:
            return [device_id]

    free_device_id = _find_free_qaic_device_id()
    if free_device_id is None:
        return None
    return [free_device_id]


def _has_explicit_device_ids() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "KIMI_K25_DYNAMO_WEIGHT_FREE_DEVICE_IDS",
            "KIMI_K25_DYNAMO_WEIGHT_FREE_DEVICE_ID",
            "KIMI_K25_DYNAMO_DEVICE_IDS",
            "KIMI_K25_DYNAMO_DEVICE_ID",
        )
    )


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

    return torch.cat(new_tokens, dim=1)


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
        subset_model_path=args.export_dir / "kimi_k25_weightfree_subset" if args.use_weight_free_export else None,
    )
    return model.eval(), tokenizer, processor


def _make_args(device_ids: list[int] | None = None) -> SimpleNamespace:
    device_ids = device_ids or [0]
    vision_qpc_path = _env_path_any("KIMI_K25_DYNAMO_WEIGHT_FREE_VISION_QPC_PATH", "KIMI_K25_DYNAMO_VISION_QPC_PATH")
    lang_qpc_path = _env_path_any("KIMI_K25_DYNAMO_WEIGHT_FREE_LANG_QPC_PATH", "KIMI_K25_DYNAMO_LANG_QPC_PATH")
    skip_compile = vision_qpc_path is not None and lang_qpc_path is not None

    vision_onnx_path = _env_path_any("KIMI_K25_DYNAMO_WEIGHT_FREE_VISION_ONNX_PATH", "KIMI_K25_DYNAMO_VISION_ONNX_PATH")
    lang_onnx_path = _env_path_any("KIMI_K25_DYNAMO_WEIGHT_FREE_LANG_ONNX_PATH", "KIMI_K25_DYNAMO_LANG_ONNX_PATH")
    skip_export = skip_compile or (vision_onnx_path is not None and lang_onnx_path is not None)

    return SimpleNamespace(
        model_path=_env_path_any("KIMI_K25_DYNAMO_WEIGHT_FREE_MODEL_PATH", "KIMI_K25_DYNAMO_MODEL_PATH"),
        export_dir=_env_path_any("KIMI_K25_DYNAMO_WEIGHT_FREE_EXPORT_DIR", "KIMI_K25_DYNAMO_EXPORT_DIR")
        or DEFAULT_EXPORT_DIR.with_name(f"{DEFAULT_EXPORT_DIR.name}_weight_free"),
        compile_dir=_env_path_any("KIMI_K25_DYNAMO_WEIGHT_FREE_COMPILE_DIR", "KIMI_K25_DYNAMO_COMPILE_DIR")
        or DEFAULT_COMPILE_DIR.with_name(f"{DEFAULT_COMPILE_DIR.name}_weight_free"),
        vision_onnx_path=vision_onnx_path,
        lang_onnx_path=lang_onnx_path,
        vision_qpc_path=vision_qpc_path,
        lang_qpc_path=lang_qpc_path,
        component="both",
        num_vision_layers=_env_int_any(
            NUM_VISION_LAYERS, "KIMI_K25_DYNAMO_WEIGHT_FREE_NUM_VISION_LAYERS", "KIMI_K25_DYNAMO_NUM_VISION_LAYERS"
        ),
        num_text_layers=_env_int_any(
            NUM_TEXT_LAYERS, "KIMI_K25_DYNAMO_WEIGHT_FREE_NUM_TEXT_LAYERS", "KIMI_K25_DYNAMO_NUM_TEXT_LAYERS"
        ),
        expert_ids=LOADED_EXPERT_IDS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        prefill_seq_len=_env_int_any(
            PREFILL_SEQ_LEN, "KIMI_K25_DYNAMO_WEIGHT_FREE_PREFILL_SEQ_LEN", "KIMI_K25_DYNAMO_PREFILL_SEQ_LEN"
        ),
        ctx_len=_env_int_any(CTX_LEN, "KIMI_K25_DYNAMO_WEIGHT_FREE_CTX_LEN", "KIMI_K25_DYNAMO_CTX_LEN"),
        num_devices=len(device_ids),
        num_cores=_env_int_any(16, "KIMI_K25_DYNAMO_WEIGHT_FREE_NUM_CORES", "KIMI_K25_DYNAMO_NUM_CORES"),
        seed=_env_int_any(1234, "KIMI_K25_DYNAMO_WEIGHT_FREE_SEED", "KIMI_K25_DYNAMO_SEED"),
        prompt=os.environ.get(
            "KIMI_K25_DYNAMO_WEIGHT_FREE_PROMPT", os.environ.get("KIMI_K25_DYNAMO_PROMPT", TEXT_PROMPT)
        ),
        image_url=os.environ.get(
            "KIMI_K25_DYNAMO_WEIGHT_FREE_IMAGE_URL", os.environ.get("KIMI_K25_DYNAMO_IMAGE_URL", DEFAULT_IMAGE_URL)
        ),
        image_path=_env_path_any("KIMI_K25_DYNAMO_WEIGHT_FREE_IMAGE_PATH", "KIMI_K25_DYNAMO_IMAGE_PATH"),
        image_height=_env_int_any(None, "KIMI_K25_DYNAMO_WEIGHT_FREE_IMAGE_HEIGHT", "KIMI_K25_DYNAMO_IMAGE_HEIGHT"),
        image_width=_env_int_any(None, "KIMI_K25_DYNAMO_WEIGHT_FREE_IMAGE_WIDTH", "KIMI_K25_DYNAMO_IMAGE_WIDTH"),
        generation_len=_env_int_any(
            NEW_GENERATION_TOKENS, "KIMI_K25_DYNAMO_WEIGHT_FREE_GENERATION_LEN", "KIMI_K25_DYNAMO_GENERATION_LEN"
        ),
        device_ids=device_ids,
        mxfp6_matmul=_env_bool_any(False, "KIMI_K25_DYNAMO_WEIGHT_FREE_MXFP6_MATMUL", "KIMI_K25_DYNAMO_MXFP6_MATMUL"),
        mxint8_kv_cache=_env_bool_any(
            False, "KIMI_K25_DYNAMO_WEIGHT_FREE_MXINT8_KV_CACHE", "KIMI_K25_DYNAMO_MXINT8_KV_CACHE"
        ),
        mos=_env_int_any(1, "KIMI_K25_DYNAMO_WEIGHT_FREE_MOS", "KIMI_K25_DYNAMO_MOS"),
        aic_enable_depth_first=_env_bool_any(
            False, "KIMI_K25_DYNAMO_WEIGHT_FREE_AIC_ENABLE_DEPTH_FIRST", "KIMI_K25_DYNAMO_AIC_ENABLE_DEPTH_FIRST"
        ),
        skip_export=skip_export,
        skip_compile=skip_compile,
        skip_generate=False,
        use_onnx_subfunctions=_env_bool_any(
            False, "KIMI_K25_DYNAMO_WEIGHT_FREE_USE_ONNX_SUBFUNCTIONS", "KIMI_K25_DYNAMO_USE_ONNX_SUBFUNCTIONS"
        ),
        keep_weights=True,
        use_weight_free_export=True,
    )


def _update_onnx_weight_spec_metadata(onnx_path: Path, spec: dict):
    model = onnx.load(str(onnx_path), load_external_data=False)
    value = json.dumps(spec, separators=(",", ":"), sort_keys=True)
    for prop in model.metadata_props:
        if prop.key == "com.qti.aisw.extdata":
            prop.value = value
            break
    else:
        prop = model.metadata_props.add()
        prop.key = "com.qti.aisw.extdata"
        prop.value = value
    tmp_path = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(model, str(tmp_path))
    tmp_path.replace(onnx_path)


def _materialize_weight_free_extdata(component_model, component_dir: Path, onnx_name: str):
    spec_path = component_dir / "weight_spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Missing weight-free spec: {spec_path}")

    spec = json.loads(spec_path.read_text())
    state_dict = component_model.state_dict()
    tensors = {}
    missing = []
    for entry in spec["inputs"]:
        name = entry["name"]
        tensor = state_dict.get(name)
        if tensor is None:
            missing.append(name)
            continue
        tensors[name] = tensor.detach().cpu().contiguous()
    if missing:
        raise RuntimeError(
            f"Weight-free extdata tensors are missing from {component_model.__class__.__name__}: {missing}"
        )

    checkpoint_dir = component_dir / "loaded_weight_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.safetensors"
    save_file(tensors, str(checkpoint_path))

    spec["files"] = [{"format": "safetensors", "path": "loaded_weight_checkpoint/model.safetensors"}]
    for entry in spec["inputs"]:
        entry["location"]["file"] = 0
        entry["location"]["key"] = entry["name"]

    spec_path.write_text(json.dumps(spec, indent=2))
    _update_onnx_weight_spec_metadata(component_dir / onnx_name, spec)


def _embed_packed_uint8_weights(component_dir: Path, onnx_name: str):
    spec_path = component_dir / "weight_spec.json"
    checkpoint_path = component_dir / "loaded_weight_checkpoint" / "model.safetensors"
    spec = json.loads(spec_path.read_text())
    packed_names = {
        entry["name"]
        for entry in spec["inputs"]
        if entry["name"].endswith("_qweight") or entry["name"].endswith("_qzeros")
    }
    if not packed_names:
        return

    from safetensors import safe_open

    model = onnx.load(str(component_dir / onnx_name), load_external_data=False)
    existing_initializers = {initializer.name for initializer in model.graph.initializer}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        for name in sorted(packed_names):
            if name not in existing_initializers:
                model.graph.initializer.append(numpy_helper.from_array(handle.get_tensor(name).numpy(), name=name))

    keep_inputs = [value for value in model.graph.input if value.name not in packed_names]
    del model.graph.input[:]
    model.graph.input.extend(keep_inputs)

    spec["inputs"] = [entry for entry in spec["inputs"] if entry["name"] not in packed_names]
    value = json.dumps(spec, separators=(",", ":"), sort_keys=True)
    for prop in model.metadata_props:
        if prop.key == "com.qti.aisw.extdata":
            prop.value = value
            break
    else:
        prop = model.metadata_props.add()
        prop.key = "com.qti.aisw.extdata"
        prop.value = value

    onnx_path = component_dir / onnx_name
    tmp_path = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(model, str(tmp_path))
    tmp_path.replace(onnx_path)
    spec_path.write_text(json.dumps(spec, indent=2))


def _prepare_weight_free_artifacts(qeff_model, exported_paths: dict[str, Path]):
    vision_onnx_path = Path(exported_paths["vision"])
    lang_onnx_path = Path(exported_paths["lang"])
    _materialize_weight_free_extdata(qeff_model.vision_model.model, vision_onnx_path.parent, vision_onnx_path.name)
    _materialize_weight_free_extdata(qeff_model.lang_model.model, lang_onnx_path.parent, lang_onnx_path.name)
    _embed_packed_uint8_weights(lang_onnx_path.parent, lang_onnx_path.name)


def check_kimi_k25_dynamo_weight_free_hf_vs_qaic():
    os.environ.setdefault("HF_HUB_CACHE", "/home/huggingface_hub")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    validate_dynamo_torch_version()
    configure_qaic_tool_path()
    if not _has_qaic_runtime_access():
        _skip_test("QAIC generation skipped: no QAIC runtime access.")

    args = _make_args()
    if args.skip_compile:
        device_ids = _resolve_device_ids(args)
    else:
        device_ids = _resolve_device_ids()
    if device_ids is None and not args.skip_compile:
        device_ids = [0]
    if device_ids is None:
        _skip_test("QAIC generation skipped: no QAIC device has free NSPs.")
    args = _make_args(device_ids)

    set_deterministic(args.seed)
    hf_model, tokenizer, processor = _load_hf_model(args)
    image = load_generation_image(args)
    hf_inputs = build_generation_inputs(processor, _QEffHolder(hf_model), image, args)
    hf_tokens = _greedy_generate_hf(hf_model, _clone_inputs(hf_inputs), args.generation_len).cpu()
    print("HF:", _decode_tokens(tokenizer, hf_tokens), "\n", hf_tokens)

    qeff_model, tokenizer, processor, qaic_config = build_qeff_model(args)
    qeff_inputs = build_generation_inputs(processor, qeff_model, image, args)
    export_inputs, output_names, dynamic_axes = get_component_export_args(qeff_model, args.prefill_seq_len)

    export_start = perf_counter()
    exported_paths = export_components(qeff_model, export_inputs, output_names, dynamic_axes, args)
    export_end = perf_counter()
    print(f"Weight-free Dynamo ONNX export time: {export_end - export_start:.2f} sec (skip_export={args.skip_export})")

    if not args.skip_compile:
        prep_start = perf_counter()
        _prepare_weight_free_artifacts(qeff_model, exported_paths)
        prep_end = perf_counter()
        print(f"Weight-free external-data prep time: {prep_end - prep_start:.2f} sec")

    compile_start = perf_counter()
    qpc_paths = compile_components(qeff_model, exported_paths, image, qaic_config, args)
    compile_end = perf_counter()
    print(
        f"Weight-free Dynamo QAIC compile time: {compile_end - compile_start:.2f} sec (skip_compile={args.skip_compile})"
    )
    print(f"Weight-free Dynamo ONNX paths: {exported_paths}")
    print(f"Weight-free Dynamo QPC paths: {qpc_paths}")

    if not _has_explicit_device_ids():
        activatable_device_id = _find_activatable_device_id(
            Path(qpc_paths["vision_qpc_path"]),
            Path(qpc_paths["lang_qpc_path"]),
            _env_int("KIMI_K25_DYNAMO_WEIGHT_FREE_MAX_DEVICE_ID", 63),
        )
        if activatable_device_id is not None:
            args.device_ids = [activatable_device_id]
            print(f"Using activatable QAIC device id: {activatable_device_id}")

    qaic_output = qeff_model.generate(
        inputs=_clone_inputs(qeff_inputs),
        device_ids=args.device_ids,
        generation_len=args.generation_len,
    )
    qaic_tokens = torch.as_tensor(qaic_output.generated_ids[:, : args.generation_len], dtype=hf_tokens.dtype)
    print("QAIC:", _decode_tokens(tokenizer, qaic_tokens), "\n", qaic_tokens)

    assert torch.equal(hf_tokens, qaic_tokens), (
        "HF and QAIC tokens do not match for the weight-free Dynamo exported and compiled model: "
        f"hf={hf_tokens.tolist()}, qaic={qaic_tokens.tolist()}"
    )


def test_kimi_k25_dynamo_weight_free_hf_vs_qaic():
    check_kimi_k25_dynamo_weight_free_hf_vs_qaic()


if __name__ == "__main__":
    check_kimi_k25_dynamo_weight_free_hf_vs_qaic()
