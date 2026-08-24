# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import init_empty_weights

from QEfficient.exporter.weight_free.checkpoint_key_resolver import promote_initializers_and_build_spec
from QEfficient.exporter.weight_free.weight_spec import load_weight_spec, resolve_weight_spec_path, save_weight_spec
from QEfficient.utils import load_json
from QEfficient.utils.checkpoint_utils import resolve_checkpoint_dir
from QEfficient.utils.logging_utils import logger

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


def _resolve_weight_free_target_dtype(model_dtype: Optional[torch.dtype]) -> torch.dtype:
    """Resolve the export/checkpoint dtype for weight-free export.

    Delegates to _resolve_torch_dtype (modeling_auto.py) so weight-free export
    downgrades bfloat16 the same way the real-weight from_pretrained path
    does: to float32 on ai100 (bfloat16 unsupported), left as bfloat16 on
    ai200 (supported). Deferred import avoids a circular import at module
    load time (modeling_auto.py imports from this package's sibling module).
    """
    from QEfficient.transformers.models.modeling_auto import _resolve_torch_dtype

    kwargs = {"torch_dtype": model_dtype or torch.float32}
    _resolve_torch_dtype(kwargs)
    return kwargs["torch_dtype"]


def _build_meta_qeff_model(qeff_model):
    """Finish preparing a meta-device QEfficient wrapper for weight-free tracing, in place."""
    model_ref = qeff_model.hash_params.get("pretrained_model_name_or_path")
    if not model_ref:
        raise ValueError(
            "Weight-free export requires checkpoint metadata. "
            "Pass `pretrained_model_name_or_path=...` when constructing the QEff model manually."
        )

    quant_config = getattr(qeff_model.model.config, "quantization_config", None)

    if quant_config is not None:
        # For quantized models the meta model must use the same quantized layer types as the
        # checkpoint so that ONNX initializer names match the checkpoint's storage keys.
        # qeff_model.model was built via from_config (no real quantized checkpoint load), so
        # Mxfp4GptOssExpertDequantizeTransform already ran as a no-op during __init__ and will
        # not undo the replacement applied here.
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
            quantizer._process_model_before_weight_loading(qeff_model.model)
    else:
        target_dtype = _resolve_weight_free_target_dtype(getattr(qeff_model.model.config, "dtype", None))
        qeff_model.model = qeff_model.model.to(dtype=target_dtype)

    qeff_model.model.eval()
    return qeff_model


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


def _prepare_checkpoint_for_weight_free_export(
    qeff_model,
    model_ref: str,
    target_dtype: torch.dtype,
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
    from QEfficient.utils.cache import QEFF_CHECKPOINT_HOME

    source_dir = resolve_checkpoint_dir(model_ref)
    dtype_suffix = str(target_dtype).replace("torch.", "")
    prepared_name = f"{source_dir.name}-wf-{dtype_suffix}"
    if QEFF_CHECKPOINT_HOME:
        prepared_out = QEFF_CHECKPOINT_HOME.expanduser() / prepared_name
    else:
        prepared_out = source_dir.parent / prepared_name
    prep_pipeline = CheckpointTransformPipeline(transforms=qeff_model._checkpoint_transforms)
    return str(
        prep_pipeline.apply(
            src=source_dir,
            out=prepared_out,
            target_dtype=target_dtype,
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

    # export_wrapper (the @export_wrapper decorator on _export) already ran
    # _setup_onnx_subfunctions on this same object before calling into this
    # function, and already has temporarily_enable_nested_compile_regions active
    # around the whole call chain — meta_qeff_model is qeff_model is self, mutated
    # in place, not a clone. Re-running subfunction setup here would only be
    # necessary if _build_meta_qeff_model's quantizer rewrite (which runs after
    # export_wrapper's setup) changed the repeated decoder-block class itself;
    # quantizer rewrites replace layer types inside a decoder block (e.g. MoE
    # experts), not the decoder block class, so the classes export_wrapper already
    # resolved remain correct.

    meta_example_inputs = _to_meta(example_inputs)
    model_ref = meta_qeff_model.hash_params["pretrained_model_name_or_path"]

    meta_qeff_model.model.requires_grad_(False)
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

    target_dtype = _resolve_weight_free_target_dtype(getattr(qeff_model.model.config, "dtype", None))

    prep_start = time.perf_counter()
    prepared_model_ref = _prepare_checkpoint_for_weight_free_export(meta_qeff_model, model_ref, target_dtype)

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
