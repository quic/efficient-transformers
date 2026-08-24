# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import contextlib
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import init_empty_weights

from QEfficient.exporter.weight_free.checkpoint_key_resolver import promote_initializers_and_build_spec
from QEfficient.exporter.weight_free.weight_spec import load_weight_spec, resolve_weight_spec_path, save_weight_spec
from QEfficient.transformers.embeddings.embedding_utils import PooledModel
from QEfficient.transformers.models.pytorch_transforms import PoolingTransform
from QEfficient.utils import load_json
from QEfficient.utils.checkpoint_utils import resolve_checkpoint_dir
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

_last_prep_peak_rss_mb: Optional[float] = None
_last_prep_duration_seconds: float = 0.0
_checkpoint_prep_ran: bool = False


def _to_meta(value: Any) -> Any:
    """Recursively move tensor-like example inputs to equivalent meta tensors."""
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value, device="meta")
    if isinstance(value, tuple):
        return tuple(_to_meta(item) for item in value)
    if isinstance(value, list):
        return [_to_meta(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_meta(item) for key, item in value.items()}
    return value


def _move_materialized_buffers_to_meta(module):
    for child in module.modules():
        for name, buffer in list(child._buffers.items()):
            if torch.is_tensor(buffer) and not buffer.is_meta:
                child._buffers[name] = torch.empty_like(buffer, device="meta")


@contextlib.contextmanager
def _cache_torch_export_signature_maps():
    """Avoid quadratic PyTorch graph-signature map rebuilding on huge weight-free graphs."""
    import torch._export.utils as export_utils
    import torch.export.exported_program as exported_program
    from torch.export.graph_signature import InputKind, TensorArgument

    original_utils_fn = export_utils._populate_param_buffer_metadata_to_new_gm
    original_exported_program_fn = getattr(
        exported_program, "_populate_param_buffer_metadata_to_new_gm", original_utils_fn
    )

    def patched_populate(params_buffers_to_node_meta, gm, new_sig):
        for metadata in params_buffers_to_node_meta.values():
            metadata.pop("nn_module_stack", None)
            metadata.pop("stack_trace", None)

        inputs_to_parameters = {
            spec.arg.name: spec.target
            for spec in new_sig.input_specs
            if spec.kind == InputKind.PARAMETER
            and isinstance(spec.arg, TensorArgument)
            and isinstance(spec.target, str)
        }
        inputs_to_buffers = {
            spec.arg.name: spec.target
            for spec in new_sig.input_specs
            if spec.kind == InputKind.BUFFER and isinstance(spec.arg, TensorArgument) and isinstance(spec.target, str)
        }

        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            param_name = inputs_to_parameters.get(node.target)
            if param_name in params_buffers_to_node_meta:
                node.meta.update(params_buffers_to_node_meta[param_name])
            buffer_name = inputs_to_buffers.get(node.target)
            if buffer_name in params_buffers_to_node_meta:
                node.meta.update(params_buffers_to_node_meta[buffer_name])

    export_utils._populate_param_buffer_metadata_to_new_gm = patched_populate
    if hasattr(exported_program, "_populate_param_buffer_metadata_to_new_gm"):
        exported_program._populate_param_buffer_metadata_to_new_gm = patched_populate
    try:
        yield
    finally:
        export_utils._populate_param_buffer_metadata_to_new_gm = original_utils_fn
        if hasattr(exported_program, "_populate_param_buffer_metadata_to_new_gm"):
            exported_program._populate_param_buffer_metadata_to_new_gm = original_exported_program_fn


def _build_meta_qeff_model(qeff_model):
    """Build a QEfficient wrapper backed by an equivalent meta-device HF model."""
    model_ref = qeff_model.hash_params.get("pretrained_model_name_or_path")
    if not model_ref:
        raise ValueError(
            "Weight-free export requires checkpoint metadata. "
            "Pass `pretrained_model_name_or_path=...` when constructing the QEff model manually."
        )

    config = copy.deepcopy(qeff_model.model.config)
    quant_config = getattr(config, "quantization_config", None)
    quantizer_applied_during_build = False

    if getattr(config, "model_type", None) == "kimi_k25":
        from QEfficient.transformers.models.modeling_auto import _build_image_text_weight_free_config_model

        meta_model = _build_image_text_weight_free_config_model(
            model_ref, {"config": config, "trust_remote_code": True}
        )
        quantizer_applied_during_build = quant_config is not None
    else:
        with init_empty_weights():
            meta_model = qeff_model._hf_auto_class.from_config(config, attn_implementation="eager")

    if quant_config is None:
        target_dtype = getattr(config, "dtype", None) or torch.float32
        if target_dtype == torch.bfloat16:
            target_dtype = torch.float16
        meta_model = meta_model.to(dtype=target_dtype)

    meta_qeff_model = qeff_model.__class__(
        meta_model,
        continuous_batching=getattr(qeff_model, "continuous_batching", False),
        qaic_config=copy.deepcopy(getattr(qeff_model.model, "qaic_config", None)),
        max_seq_len_cached=getattr(qeff_model.model.config, "max_seq_len_cached", None),
        pretrained_model_name_or_path=model_ref,
        enable_proxy=getattr(qeff_model, "_enable_proxy", False),
    )
    meta_qeff_model.hash_params.update(copy.deepcopy(qeff_model.hash_params))
    _move_materialized_buffers_to_meta(meta_qeff_model.model)

    if isinstance(qeff_model.model, PooledModel):
        meta_qeff_model.model, _ = PoolingTransform.apply(meta_qeff_model.model, qeff_model.model.pooling_fn)

    if quant_config is not None and not quantizer_applied_during_build:
        # For quantized models the meta model must use the same quantized layer types as the
        # checkpoint so that ONNX initializer names match the checkpoint's storage keys.
        # We apply the quantizer's architecture preprocessing (layer-type replacement only,
        # no weight loading) AFTER __init__ so that Mxfp4GptOssExpertDequantizeTransform —
        # which is part of _pytorch_transforms and targets QEffMxfp4GptOssExperts — has
        # already run as a no-op and will not undo the replacement below.
        from QEfficient.transformers.quantizers.auto import (
            QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING,
            QEFF_AUTO_QUANTIZER_MAPPING,
        )

        # quantization_config may be a plain dict (AutoConfig.from_pretrained) or a proper
        # config object (QEFFAutoModelForCausalLM.from_pretrained).  Normalise to an object.
        if isinstance(quant_config, dict):
            quant_type = quant_config.get("quant_method") or quant_config.get("quant_type")
            config_cls = QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.get(quant_type)
            if config_cls is None:
                raise NotImplementedError(
                    f"Weight-free export is not implemented for quantization type '{quant_type}'. Supported: mxfp4"
                )
            init_kwargs = {k: v for k, v in quant_config.items() if k != "quant_method"}
            quant_config = config_cls(**init_kwargs)
        else:
            quant_method = getattr(quant_config, "quant_method", None) or getattr(quant_config, "quant_type", None)
            quant_type = quant_method.value if hasattr(quant_method, "value") else quant_method

        quantizer_cls = QEFF_AUTO_QUANTIZER_MAPPING.get(quant_type) if quant_type else None
        if quantizer_cls is None:
            raise NotImplementedError(
                f"Weight-free export is not implemented for quantization type '{quant_type}'. Supported: mxfp4"
            )
        quantizer = quantizer_cls(quant_config)
        # Run inside init_empty_weights so newly created quantized layer buffers stay on
        # the meta device and are treated as weight-spec entries, not embedded constants.
        with init_empty_weights():
            quantizer._process_model_before_weight_loading(meta_qeff_model.model)

    meta_qeff_model.model.eval()
    return meta_qeff_model


def _prune_unused_fake_initializers(onnx_program) -> None:
    """Remove FakeTensor initializers not referenced by any graph node.

    During weight-free dynamo export, meta-device parameters that are not
    actually consumed in the forward graph can end up as orphan initializers.
    Serialising them fails (no data to write), so we drop them here before save.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    initializers = onnx_program.model.graph.initializers
    used_names = {name for node in onnx_program.model.graph for name in node.inputs}
    used_names.update(output.name for output in onnx_program.model.graph.outputs)
    for name in list(initializers):
        const_value = getattr(initializers[name], "const_value", None)
        raw_value = getattr(const_value, "raw", None)
        if isinstance(raw_value, FakeTensor) and name not in used_names:
            del initializers[name]


def _upsert_metadata_prop(model, key: str, value: str) -> None:
    """Insert or update a metadata_props entry on an ONNX model.

    Used to embed weight_spec.json into the ONNX so the QAIC compiler
    can locate external weight files without a separate sidecar lookup.
    """
    import onnx

    for entry in model.metadata_props:
        if entry.key == key:
            entry.value = value
            return
    model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))


def _checkpoint_key_variants(name: str) -> set[str]:
    variants = {name}
    stripped = name[len("model.") :] if name.startswith("model.") else name
    variants.add(stripped)
    if stripped.startswith("language_model."):
        variants.add(stripped[len("language_model.") :])
    elif stripped.startswith("lm_head."):
        variants.add(f"language_model.{stripped}")
    return variants


def _collect_allowed_checkpoint_keys(qeff_model, onnx_initializer_names: Optional[set[str]] = None) -> set[str]:
    if onnx_initializer_names is None:
        names = {name for name, _ in qeff_model.model.named_parameters()}
        names.update(name for name, _ in qeff_model.model.named_buffers())
    else:
        names = set(onnx_initializer_names)
    allowed = set()
    for name in names:
        allowed.update(_checkpoint_key_variants(name))
    return allowed


def _prepare_checkpoint_for_weight_free_export(
    qeff_model,
    model_ref: str,
    target_dtype: torch.dtype,
    onnx_initializer_names: Optional[set[str]] = None,
) -> str:
    """Prepare a checkpoint directory for weight-free ONNX export.

    Parameters
    ----------
    qeff_model
        QEfficient model wrapper that provides checkpoint transform classes.
    model_ref : str
        Local model path or Hugging Face Hub model id.
    target_dtype : torch.dtype
        Floating-point dtype expected by the exported ONNX graph.

    Returns
    -------
    str
        Path to the prepared checkpoint directory.
    """
    from QEfficient.base.checkpoint_transforms import CheckpointTransformPipeline

    source_dir = resolve_checkpoint_dir(model_ref)
    dtype_suffix = str(target_dtype).replace("torch.", "")
    allowed_checkpoint_keys = _collect_allowed_checkpoint_keys(qeff_model, onnx_initializer_names)
    allowed_hash = hashlib.sha256("\n".join(sorted(allowed_checkpoint_keys)).encode("utf-8")).hexdigest()[:12]
    config = getattr(qeff_model.model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    vision_config = getattr(config, "vision_config", None)
    prep_suffix = f"qeff-prepared-{dtype_suffix}"
    if getattr(config, "model_type", None) == "kimi_k25":
        text_layers = getattr(text_config, "num_hidden_layers", None)
        vision_layers = getattr(vision_config, "vt_num_hidden_layers", None)
        experts = getattr(text_config, "n_routed_experts", None)
        prep_suffix = f"{prep_suffix}-t{text_layers}-v{vision_layers}-e{experts}"
    prep_suffix = f"{prep_suffix}-{qeff_model.model_name}-{allowed_hash}"
    prepared_out = source_dir.parent / (source_dir.name + f"-{prep_suffix}")
    prep_pipeline = CheckpointTransformPipeline(transforms=qeff_model._checkpoint_transforms)
    return str(
        prep_pipeline.apply(
            src=source_dir,
            out=prepared_out,
            target_dtype=target_dtype,
            model_config=config,
            allowed_checkpoint_keys=allowed_checkpoint_keys,
        )
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
    """Export a QEfficient model to ONNX with checkpoint weights externalized.

    Parameters
    ----------
    qeff_model
        Loaded QEfficient model wrapper used as the source of config and export options.
    tmp_onnx_path : Path
        Temporary ONNX path where the dynamo export is saved.
    example_inputs : Dict[str, torch.Tensor]
        Example inputs used to trace the model.
    input_names : List[str]
        ONNX graph input names.
    output_names : List[str]
        ONNX graph output names.
    dynamic_shapes : Dict[str, Any]
        Dynamo export dynamic shape specification.
    export_kwargs : Dict[str, Any]
        Additional keyword arguments forwarded to ``torch.onnx.export``.
    onnx_transform_kwargs : Dict[str, Any]
        ONNX transform options carried through the export flow.

    Returns
    -------
    tuple
        Meta QEfficient model, updated ONNX transform kwargs, and cleanup callback.
    """
    global _checkpoint_prep_ran, _last_prep_duration_seconds, _last_prep_peak_rss_mb

    meta_qeff_model = _build_meta_qeff_model(qeff_model)
    cleanup_required = False

    if getattr(qeff_model, "_use_onnx_subfunctions", False):
        _, subfunc_kwargs, _ = _setup_onnx_subfunctions(
            meta_qeff_model,
            (),
            {
                "onnx_transform_kwargs": copy.deepcopy(onnx_transform_kwargs),
                "output_names": list(output_names),
            },
            dynamo=True,
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
    with export_context, _cache_torch_export_signature_maps():
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

        target_dtype = getattr(qeff_model.model.config, "dtype", None) or torch.float32
        if target_dtype == torch.bfloat16:
            target_dtype = torch.float16

        prep_start = time.perf_counter()
        onnx_initializer_names = set(onnx_program.model.graph.initializers)
        prepared_model_ref = _prepare_checkpoint_for_weight_free_export(
            meta_qeff_model, model_ref, target_dtype, onnx_initializer_names
        )

        _last_prep_duration_seconds = time.perf_counter() - prep_start
        _last_prep_peak_rss_mb = None
        _checkpoint_prep_ran = True
        logger.info(
            "Weight-free checkpoint preparation completed in %.2fs: %s",
            _last_prep_duration_seconds,
            prepared_model_ref,
        )

        spec = promote_initializers_and_build_spec(
            onnx_program=onnx_program,
            model_ref=prepared_model_ref,
            model_name=qeff_model.model_name,
            qeff_model=meta_qeff_model,
        )
        _prune_unused_fake_initializers(onnx_program)
        onnx_program.save(str(tmp_onnx_path))
        save_weight_spec(resolve_weight_spec_path(tmp_onnx_path), spec)

    def cleanup():
        """Release ONNX subfunction state created for this export, if any."""
        if cleanup_required:
            _cleanup_onnx_subfunctions(meta_qeff_model)

    return meta_qeff_model, onnx_transform_kwargs, cleanup


def embed_weight_spec_as_metadata(model, weight_spec_path) -> None:
    """Embed weight_spec.json into the ONNX model as com.qti.aisw.extdata metadata.

    The QAIC compiler reads this key to locate and load external checkpoint weights
    at compile time. Separating this from the private _upsert_metadata_prop keeps
    the base exporter free of knowledge about the QAIC-specific metadata key.
    """
    weight_spec_json = json.dumps(load_json(Path(weight_spec_path)), separators=(",", ":"), sort_keys=True)
    _upsert_metadata_prop(model, "com.qti.aisw.extdata", weight_spec_json)


def link_prepared_checkpoint_dir(onnx_path: Path, weight_spec_path: Path) -> None:
    """Place the checkpoint directory referenced by the weight spec next to the ONNX export.

    Weight specs store checkpoint files relative to their checkpoint root.
    Keeping this link beside the ONNX lets the compiler resolve those paths when
    it consumes the embedded weight spec.
    """
    spec = load_weight_spec(Path(weight_spec_path))
    prepared_out = Path(spec.model_id)
    symlink = Path(onnx_path).parent / prepared_out.name
    if prepared_out.exists() and not symlink.exists():
        try:
            symlink.symlink_to(prepared_out)
        except OSError as exc:
            logger.warning(
                "Could not create symlink %s -> %s: %s. "
                "Checkpoint files may not resolve at compile time if paths are relative.",
                symlink,
                prepared_out,
                exc,
            )
