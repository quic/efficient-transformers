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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from QEfficient.base.onnx_transforms import PruneFakeInitializersTransform
from QEfficient.utils import constants
from QEfficient.utils.torch_patches import layerwise_safe_onnx_export_patches


def export_via_legacy(
    qeff_model: Any,
    onnx_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict,
    export_kwargs: Dict,
) -> None:
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


def export_via_dynamo(
    qeff_model: Any,
    onnx_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    dynamic_shapes: Optional[Dict],
    export_kwargs: Dict,
) -> None:
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


def export_via_weightfree(
    qeff_model: Any,
    tmp_onnx_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    input_names: List[str],
    output_names: List[str],
    dynamic_shapes: Optional[Dict],
    export_kwargs: Dict,
    onnx_transform_kwargs: Optional[Dict] = None,
) -> Tuple[Optional[Dict], Any]:
    """Export via weight-free dynamo path: meta-device model, no embedded weights.

    dynamic_shapes must be pre-computed by the caller (export_wrapper already does
    this for dynamo=True).  Accepting it directly keeps the signature parallel with
    export_via_dynamo and avoids a redundant convert_dynamic_axes_to_dynamic_shapes
    call.
    """
    from QEfficient.exporter.weight_free.core import export_weight_free_onnx
    from QEfficient.utils.export_utils import build_dynamo_export_kwargs, reorder_inputs_by_signature

    example_inputs, dynamic_shapes = reorder_inputs_by_signature(qeff_model.model, example_inputs, dynamic_shapes)
    wf_export_kwargs = build_dynamo_export_kwargs(export_kwargs)

    prev_invoke_fallback = os.environ.get("TORCH_INVOKE_ALLOW_CREATE_FALLBACK")
    os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = "1"
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
    finally:
        if prev_invoke_fallback is None:
            os.environ.pop("TORCH_INVOKE_ALLOW_CREATE_FALLBACK", None)
        else:
            os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = prev_invoke_fallback

    return updated_onnx_transform_kwargs, cleanup
