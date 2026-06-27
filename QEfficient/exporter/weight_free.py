#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import numpy as np
import onnx_ir as ir
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from QEfficient.transformers.embeddings.embedding_utils import PooledModel
from QEfficient.transformers.models.pytorch_transforms import PoolingTransform
from QEfficient.exporter.weight_spec import (
    ExternalDataFile,
    TiedWeightAlias,
    WeightSpec,
    WeightSpecInput,
    WeightSpecLocation,
    load_weight_spec,
    resolve_weight_spec_path,
    save_weight_spec,
)
from QEfficient.utils.export_utils import (
    _cleanup_onnx_subfunctions,
    _setup_onnx_subfunctions,
    get_decoder_layer_classes_for_export,
)
from QEfficient.utils.logging_utils import logger
from QEfficient.utils.torch_patches import (
    temporarily_disable_nested_compile_regions,
    temporarily_enable_nested_compile_regions,
)


def _to_meta(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value, device="meta")
    if isinstance(value, tuple):
        return tuple(_to_meta(item) for item in value)
    if isinstance(value, list):
        return [_to_meta(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_meta(item) for key, item in value.items()}
    return value


@lru_cache(maxsize=None)
def _resolve_checkpoint_dir(model_id_or_path: str) -> Path:
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate

    snapshot_dir = snapshot_download(
        repo_id=model_id_or_path,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.txt", "*.pdf", "*.msgpack", "*.h5", "*.pth"],
        resume_download=True,
    )
    return Path(snapshot_dir)


def _resolve_checkpoint_files(model_id_or_path: str) -> List[str]:
    checkpoint_dir = _resolve_checkpoint_dir(model_id_or_path)
    checkpoint_files = sorted(str(path) for path in checkpoint_dir.glob("*.safetensors"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No safetensors checkpoint files found for {model_id_or_path}")
    return checkpoint_files


def _module_name_map(model: nn.Module) -> Dict[int, str]:
    return {id(module): name for name, module in model.named_modules()}


def _collect_tied_weights(model: nn.Module) -> List[TiedWeightAlias]:
    if not getattr(model.config, "tie_word_embeddings", False):
        return []

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        return []

    module_names = _module_name_map(model)
    canonical_name = module_names.get(id(input_embeddings))
    alias_name = module_names.get(id(output_embeddings))
    if not canonical_name or not alias_name or canonical_name == alias_name:
        return []

    return [TiedWeightAlias(alias=f"{alias_name}.weight", canonical=f"{canonical_name}.weight")]


def _build_meta_qeff_model(qeff_model):
    model_ref = qeff_model.hash_params.get("pretrained_model_name_or_path")
    if not model_ref:
        raise ValueError(
            "Weight-free export requires checkpoint metadata. "
            "Pass `pretrained_model_name_or_path=...` when constructing the QEff model manually."
        )

    quant_config = getattr(qeff_model.model.config, "quantization_config", None)

    config = copy.deepcopy(qeff_model.model.config)
    with init_empty_weights():
        meta_model = qeff_model._hf_auto_class.from_config(config, attn_implementation="eager")

    if quant_config is None:
        target_dtype = getattr(config, "dtype", torch.float32) or torch.float32
        if target_dtype == torch.bfloat16:
            target_dtype = torch.float16
        meta_model = meta_model.to(dtype=target_dtype)

    meta_qeff_model = qeff_model.__class__(
        meta_model,
        continuous_batching=getattr(qeff_model, "continuous_batching", False),
        qaic_config=copy.deepcopy(getattr(qeff_model.model, "qaic_config", None)),
        max_seq_len_cached=getattr(qeff_model.model.config, "max_seq_len_cached", None),
        pretrained_model_name_or_path=model_ref,
    )
    meta_qeff_model.hash_params.update(copy.deepcopy(qeff_model.hash_params))

    
    
    if isinstance(qeff_model.model, PooledModel):
        meta_qeff_model.model, _ = PoolingTransform.apply(
            meta_qeff_model.model, qeff_model.model.pooling_fn
        )

    if quant_config is not None:
        # For quantized models the meta model must use the same quantized layer types as the
        # checkpoint so that ONNX initializer names match the checkpoint's storage keys.
        # We apply the quantizer's architecture preprocessing (layer-type replacement only,
        # no weight loading) AFTER __init__ so that Mxfp4GptOssExpertDequantizeTransform —
        # which is part of _pytorch_transforms and targets QEffMxfp4GptOssExperts — has
        # already run as a no-op and will not undo the replacement below.
        from QEfficient.transformers.quantizers.auto import QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING, QEFF_AUTO_QUANTIZER_MAPPING

        # quantization_config may be a plain dict (AutoConfig.from_pretrained) or a proper
        # config object (QEFFAutoModelForCausalLM.from_pretrained).  Normalise to an object.
        if isinstance(quant_config, dict):
            quant_type = quant_config.get("quant_method") or quant_config.get("quant_type")
            config_cls = QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.get(quant_type)
            if config_cls is None:
                raise NotImplementedError(
                    f"Weight-free export is not implemented for quantization type '{quant_type}'. "
                    "Supported: mxfp4"
                )
            init_kwargs = {k: v for k, v in quant_config.items() if k != "quant_method"}
            quant_config = config_cls(**init_kwargs)
        else:
            quant_method = getattr(quant_config, "quant_method", None) or getattr(quant_config, "quant_type", None)
            quant_type = quant_method.value if hasattr(quant_method, "value") else quant_method

        quantizer_cls = QEFF_AUTO_QUANTIZER_MAPPING.get(quant_type) if quant_type else None
        if quantizer_cls is None:
            raise NotImplementedError(
                f"Weight-free export is not implemented for quantization type '{quant_type}'. "
                "Supported: mxfp4"
            )
        quantizer = quantizer_cls(quant_config)
        # Run inside init_empty_weights so newly created quantized layer buffers stay on
        # the meta device and are treated as weight-spec entries, not embedded constants.
        with init_empty_weights():
            quantizer._process_model_before_weight_loading(meta_qeff_model.model)

    meta_qeff_model.model.eval()
    return meta_qeff_model


def _checkpoint_root(model_id_or_path: str, checkpoint_files: Sequence[str]) -> Optional[Path]:
    if not checkpoint_files:
        return None

    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate.parent

    first_checkpoint = Path(checkpoint_files[0])
    for parent in first_checkpoint.parents:
        if parent.name.startswith("models--"):
            return parent.parent
    return first_checkpoint.parent


def _load_checkpoint_index(checkpoint_files: List[str]) -> Dict[str, str]:
    tensor_to_file = {}
    for checkpoint_file in checkpoint_files:
        handle = safe_open(checkpoint_file, framework="pt")
        for key in handle.keys():
            tensor_to_file[key] = checkpoint_file
    return tensor_to_file


def _build_location(
    checkpoint_files: Sequence[str],
    checkpoint_file: Optional[str],
    tensor_key: str,
) -> Optional[WeightSpecLocation]:
    if checkpoint_file is None:
        return None

    return WeightSpecLocation(file=list(checkpoint_files).index(checkpoint_file), key=tensor_key)


def _find_checkpoint_key(
    onnx_name: str,
    checkpoint_index: Dict[str, str],
    backbone: nn.Module,
) -> Optional[str]:
    """
    Resolve an ONNX initializer name to its key in the safetensors checkpoint.

    Three lookups are attempted in order:

    1. Direct match — ONNX name == checkpoint key.
       Covers decoder-only LLMs (Llama, GPT-OSS) and any model exported
       without extra wrappers.

    2. Strip our PooledModel prefix ("base_model.") and try the bare key.
       Covers embedding models whose checkpoint was saved from the base class
       directly (e.g. BAAI/bge-base saved from BertModel → bare keys).

    3. Strip "base_model." then prepend backbone.base_model_prefix.
       Covers embedding models whose checkpoint was saved from a task-specific
       class (e.g. BAAI/bge-reranker saved from XLMRobertaForSequenceClassification
       → "roberta." prefix).  HuggingFace defines base_model_prefix on every
       model class as exactly the attribute name the task class uses to store
       the backbone, so this is generalized — no per-model-family list needed.
    """
    # 1. Direct match
    if onnx_name in checkpoint_index:
        return onnx_name

    # Strip PooledModel's "base_model." wrapper prefix
    stripped = onnx_name[len("base_model."):] if onnx_name.startswith("base_model.") else onnx_name

    # 2. Bare key — checkpoint saved from base class
    if stripped in checkpoint_index:
        return stripped

    # 3. Task-class prefix — checkpoint saved from ForSequenceClassification etc.
    prefix = getattr(backbone, "base_model_prefix", "")
    if prefix:
        prefixed = f"{prefix}.{stripped}"
        if prefixed in checkpoint_index:
            return prefixed

    return None


def _promote_initializers_and_build_spec(onnx_program, model_ref: str, model_name: str, qeff_model) -> WeightSpec:
    from QEfficient.transformers.embeddings.embedding_utils import PooledModel
    model_ir = onnx_program.model
    model_names = {name for name, _ in qeff_model.model.named_parameters()}
    model_names.update({name for name, _ in qeff_model.model.named_buffers()})
    tied_weights = _collect_tied_weights(qeff_model.model)
    tied_weight_map = {entry.alias: entry.canonical for entry in tied_weights}
    checkpoint_files = _resolve_checkpoint_files(model_ref)
    checkpoint_root = _checkpoint_root(model_ref, checkpoint_files)
    checkpoint_index = _load_checkpoint_index(checkpoint_files)
    relative_checkpoint_files = [
        ExternalDataFile(
            path=str(Path(checkpoint_file).relative_to(checkpoint_root))
            if checkpoint_root is not None
            else Path(checkpoint_file).name,
            format="safetensors",
        )
        for checkpoint_file in checkpoint_files
    ]
    backbone = qeff_model.model.base_model if isinstance(qeff_model.model, PooledModel) else qeff_model.model
    promoted_inputs: List[WeightSpecInput] = []

    for name, init_value in list(model_ir.graph.initializers.items()):
        if name not in model_names:
            continue

        onnx_name = tied_weight_map.get(name, name)
        checkpoint_key = _find_checkpoint_key(onnx_name, checkpoint_index, backbone)

        if checkpoint_key is None:
            # Computed buffer (e.g. sin_cached, cos_cached) — leave as ONNX initializer.
            # The compiler embeds it in the model; it is not loaded from a checkpoint file.
            continue

        location = _build_location(checkpoint_files, checkpoint_index[checkpoint_key], checkpoint_key)

        model_ir.graph.inputs.append(
            ir.Value(
                name=name,
                shape=init_value.shape,
                type=ir.TensorType(init_value.dtype),
            )
        )
        del model_ir.graph.initializers[name]
        promoted_inputs.append(WeightSpecInput(name=name, location=location))

    return WeightSpec(
        model_name=model_name,
        model_id=model_ref,
        files=relative_checkpoint_files,
        inputs=promoted_inputs,
    )


def export_weight_free_onnx(
    qeff_model,
    tmp_onnx_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    dynamic_shapes: Dict[str, Any],
    export_kwargs: Dict[str, Any],
    onnx_transform_kwargs: Dict[str, Any],
):
    meta_qeff_model = _build_meta_qeff_model(qeff_model)
    cleanup_required = False

    if getattr(qeff_model, "_use_onnx_subfunctions", False):
        _, subfunc_kwargs = _setup_onnx_subfunctions(
            meta_qeff_model,
            (),
            {
                "use_dynamo": True,
                "onnx_transform_kwargs": copy.deepcopy(onnx_transform_kwargs),
                "output_names": list(output_names),
            },
        )
        onnx_transform_kwargs = subfunc_kwargs.get("onnx_transform_kwargs", onnx_transform_kwargs)
        cleanup_required = True

    decoder_layer_classes = get_decoder_layer_classes_for_export(meta_qeff_model.model)
    if getattr(meta_qeff_model, "_use_onnx_subfunctions", False) and decoder_layer_classes:
        export_context = temporarily_enable_nested_compile_regions(meta_qeff_model.model, decoder_layer_classes)
    else:
        export_context = temporarily_disable_nested_compile_regions(meta_qeff_model.model, decoder_layer_classes)

    meta_example_inputs = _to_meta(example_inputs)
    model_ref = meta_qeff_model.hash_params["pretrained_model_name_or_path"]
    meta_qeff_model.model.requires_grad_(False)
    with export_context:
        onnx_program = torch.onnx.export(
            meta_qeff_model.model,
            args=(),
            f=None,
            kwargs=meta_example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            dynamic_shapes=dynamic_shapes,
            **export_kwargs,
        )
        if onnx_program is None:
            raise RuntimeError("torch.onnx.export returned None for weight-free dynamo export")

        spec = _promote_initializers_and_build_spec(
            onnx_program=onnx_program,
            model_ref=model_ref,
            model_name=qeff_model.model_name,
            qeff_model=meta_qeff_model,
        )
        onnx_program.save(str(tmp_onnx_path))
        save_weight_spec(resolve_weight_spec_path(tmp_onnx_path), spec)

    def cleanup():
        if cleanup_required:
            _cleanup_onnx_subfunctions(meta_qeff_model)

    return meta_qeff_model, onnx_transform_kwargs, cleanup


def _load_checkpoint_tensor(checkpoint_file: str, key: str) -> np.ndarray:
    handle = safe_open(checkpoint_file, framework="pt")
    tensor = handle.get_tensor(key).detach().cpu()
    # numpy does not support bfloat16; cast to float32 for ORT compatibility
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy()


def _default_weights_roots(weight_spec_path: Path, spec) -> List[Path]:
    roots = []
    ext_root = os.environ.get("AIC_EXTERNAL_DATA_ROOT")
    if ext_root:
        roots.append(Path(ext_root).expanduser())
    roots.append(weight_spec_path.parent)
    candidate = Path(spec.model_id).expanduser()
    if candidate.exists():
        roots.append(candidate.parent)
    else:
        checkpoint_dir = _resolve_checkpoint_dir(spec.model_id)
        checkpoint_root = _checkpoint_root(spec.model_id, [str(path) for path in checkpoint_dir.glob("*.safetensors")])
        if checkpoint_root is not None:
            roots.append(checkpoint_root)

    deduped_roots: List[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_roots.append(resolved)
    return deduped_roots


def _resolve_location_file(
    location: WeightSpecLocation,
    files: Sequence[ExternalDataFile],
    candidate_roots: Sequence[Path],
) -> Path:
    if isinstance(location.file, int):
        location_path = Path(files[location.file].path)
    else:
        location_path = Path(location.file)
    if location_path.is_absolute():
        return location_path

    for root in candidate_roots:
        candidate = root / location_path
        if candidate.exists():
            return candidate

    return candidate_roots[0] / location_path if candidate_roots else location_path


def load_weight_free_ort_inputs(
    weight_spec_path: Path,
    runtime_inputs: Dict[str, np.ndarray],
    weights_root: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    weight_spec_path = Path(weight_spec_path)
    spec = load_weight_spec(weight_spec_path)
    candidate_roots = []
    if weights_root is not None:
        candidate_roots.append(Path(weights_root).expanduser().resolve())
    candidate_roots.extend(_default_weights_roots(weight_spec_path, spec))

    ort_inputs = dict(runtime_inputs)
    for spec_input in spec.inputs:
        if spec_input.name in ort_inputs:
            continue
        checkpoint_file = _resolve_location_file(spec_input.location, spec.files, candidate_roots)
        ort_inputs[spec_input.name] = _load_checkpoint_tensor(str(checkpoint_file), spec_input.location.key)

    return ort_inputs


def log_weight_free_export(onnx_path: Path) -> None:
    logger.info(f"Weight-free ONNX exported to {onnx_path} with spec {resolve_weight_spec_path(onnx_path)}")
