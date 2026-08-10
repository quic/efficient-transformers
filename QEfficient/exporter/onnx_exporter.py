# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Standalone ONNX export backend functions.

These functions contain the actual export logic and are called directly from
QEFFBaseModel._export(). Keeping them outside the base class makes the logic
testable in isolation and avoids polluting QEFFBaseModel with backend details.

_export_layerwise is intentionally kept in QEFFBaseModel because it contains
VLM-specific branching (is_vision checks) that requires a model-level hook to
refactor cleanly. It will be moved here once _build_weight_free_model() and
similar override hooks are in place.
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch

from QEfficient.base.onnx_transforms import PruneFakeInitializersTransform
from QEfficient.utils import constants
from QEfficient.utils.torch_patches import layerwise_safe_onnx_export_patches


@dataclass
class ExportResult:
    onnx_path: Path
    onnx_transform_kwargs: Dict[str, Any] = field(default_factory=dict)
    excluded_onnx_transforms: Tuple[Type[Any], ...] = ()
    weight_spec_path: Optional[Path] = None
    post_transform_hooks: Tuple[Callable[[Any], None], ...] = ()
    post_export_hooks: Tuple[Callable[[], None], ...] = ()


def export_via_legacy(
    qeff_model: Any,
    onnx_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict,
    export_kwargs: Dict,
) -> ExportResult:
    """Export via TorchScript symbolic tracing (dynamo=False)."""
    from QEfficient.base.modeling_qeff import QEFFBaseModel

    with layerwise_safe_onnx_export_patches(enabled=bool(QEFFBaseModel._layerwise_active)):
        torch.onnx.export(
            qeff_model.model,
            (),
            str(onnx_path),
            kwargs=example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
            opset_version=constants.ONNX_LEGACY_EXPORT_OPSET,
            **export_kwargs,
        )
    return ExportResult(onnx_path=onnx_path)


def export_via_dynamo(
    qeff_model: Any,
    onnx_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    dynamic_shapes: Optional[Dict],
    export_kwargs: Dict,
) -> ExportResult:
    """Export via torch.export (dynamo=True) with custom op translation."""
    from QEfficient.utils.export_utils import build_dynamo_export_kwargs, reorder_inputs_by_signature

    example_inputs, dynamic_shapes = reorder_inputs_by_signature(qeff_model.model, example_inputs, dynamic_shapes)
    export_kwargs = build_dynamo_export_kwargs(export_kwargs)

    prev_invoke_fallback = os.environ.get("TORCH_INVOKE_ALLOW_CREATE_FALLBACK")
    os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = "1"
    try:
        onnx_program = torch.onnx.export(
            qeff_model.model,
            args=(),
            f=None,
            kwargs=example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            dynamic_shapes=dynamic_shapes,
            **export_kwargs,
        )
        if onnx_program is None:
            raise RuntimeError("torch.onnx.export returned None for dynamo export")
        PruneFakeInitializersTransform.apply(onnx_program)
        onnx_program.save(str(onnx_path))
    finally:
        if prev_invoke_fallback is None:
            os.environ.pop("TORCH_INVOKE_ALLOW_CREATE_FALLBACK", None)
        else:
            os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = prev_invoke_fallback
    return ExportResult(onnx_path=onnx_path)


def export_via_weightfree(
    qeff_model: Any,
    onnx_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    dynamic_shapes: Optional[Dict],
    export_kwargs: Dict,
    onnx_transform_kwargs: Optional[Dict] = None,
) -> ExportResult:
    """Export via weight-free dynamo path: meta-device model, no embedded weights.

    dynamic_shapes must be pre-computed by the caller (export_wrapper already does
    this for dynamo=True).  Accepting it directly keeps the signature parallel with
    export_via_dynamo and avoids a redundant convert_dynamic_axes_to_dynamic_shapes
    call.
    """
    from QEfficient.base.onnx_transforms import SplitTensorsTransform
    from QEfficient.exporter.weight_free.export import export_weight_free_onnx
    from QEfficient.utils.export_utils import build_dynamo_export_kwargs, reorder_inputs_by_signature

    example_inputs, dynamic_shapes = reorder_inputs_by_signature(qeff_model.model, example_inputs, dynamic_shapes)
    wf_export_kwargs = build_dynamo_export_kwargs(export_kwargs)
    tmp_onnx_dir = onnx_path.parent / "onnx_tmp"
    tmp_onnx_path = tmp_onnx_dir / onnx_path.name
    tmp_onnx_dir.mkdir(parents=True, exist_ok=True)

    prev_invoke_fallback = os.environ.get("TORCH_INVOKE_ALLOW_CREATE_FALLBACK")
    os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = "1"
    cleanup = None
    try:
        _, updated_onnx_transform_kwargs, cleanup = export_weight_free_onnx(
            qeff_model=qeff_model,
            tmp_onnx_path=tmp_onnx_path,
            example_inputs=example_inputs,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            export_kwargs=wf_export_kwargs,
            onnx_transform_kwargs=onnx_transform_kwargs or {},
        )

        shutil.move(str(tmp_onnx_path), str(onnx_path))
        tmp_weight_spec = tmp_onnx_dir / "weight_spec.json"
        final_weight_spec = onnx_path.with_name("weight_spec.json")
        weight_spec_path = None
        if tmp_weight_spec.exists():
            shutil.move(str(tmp_weight_spec), str(final_weight_spec))
            weight_spec_path = final_weight_spec

        post_transform_hooks = ()
        post_export_hooks = ()
        if weight_spec_path is not None:

            def _embed_weight_spec(model, spec_path=weight_spec_path):
                from QEfficient.exporter.weight_free.export import embed_weight_spec_as_metadata

                embed_weight_spec_as_metadata(model, spec_path)

            def _link_prepared_checkpoint(onnx_file=onnx_path, spec_path=weight_spec_path):
                from QEfficient.exporter.weight_free.export import link_prepared_checkpoint_dir

                link_prepared_checkpoint_dir(onnx_file, spec_path)

            post_transform_hooks = (_embed_weight_spec,)
            post_export_hooks = (_link_prepared_checkpoint,)

        return ExportResult(
            onnx_path=onnx_path,
            onnx_transform_kwargs=updated_onnx_transform_kwargs or {},
            excluded_onnx_transforms=(SplitTensorsTransform,),
            weight_spec_path=weight_spec_path,
            post_transform_hooks=post_transform_hooks,
            post_export_hooks=post_export_hooks,
        )
    finally:
        if cleanup is not None:
            cleanup()
        shutil.rmtree(tmp_onnx_dir, ignore_errors=True)
        if prev_invoke_fallback is None:
            os.environ.pop("TORCH_INVOKE_ALLOW_CREATE_FALLBACK", None)
        else:
            os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = prev_invoke_fallback
