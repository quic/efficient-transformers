# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Monkey patches for torch.onnx.utils to fix ONNX export issues."""

from contextlib import contextmanager

import torch
import torch.onnx.utils as onnx_utils
from torch import _C

# Store original references before patching
_original_setup_trace_module_map = onnx_utils._setup_trace_module_map
_original_get_module_attributes = getattr(onnx_utils, "_get_module_attributes", None)
_safe_export_patch_depth = 0
_safe_export_original_passes = {}
_SAFE_EXPORT_REQUIRED_PASSES = {
    "_jit_pass_dce",
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects",
    "_jit_pass_constant_propagation",
    "_jit_pass_cse",
    # Keep ONNX constant fold enabled to reduce topology drift between
    # layerwise prefill exports and regular (non-layerwise) prefill exports.
    "_jit_pass_onnx_constant_fold",
}


def _noop(*args, **kwargs):
    return None


def _return_false(*args, **kwargs):
    return False


def _return_graph(graph, *args, **kwargs):
    return graph


def _return_params(_graph, params_dict, *args, **kwargs):
    return params_dict


_SAFE_EXPORT_PASS_REPLACEMENTS = {
    "_jit_pass_constant_propagation": _noop,
    "_jit_pass_dce": _noop,
    "_jit_pass_cse": _return_false,
    "_jit_pass_canonicalize_graph_fuser_ops": _noop,
    "_jit_pass_peephole": _noop,
    "_jit_pass_fuse_addmm": _noop,
    "_jit_pass_onnx_eval_peephole": _return_params,
    "_jit_pass_onnx_constant_fold": _return_params,
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects": _noop,
    "_jit_pass_canonicalize": _return_graph,
    "_jit_pass_onnx_graph_shape_type_inference": _noop,
    "_jit_pass_onnx_deduplicate_initializers": _return_params,
}


def _setup_trace_module_map_patched(
    model,
    export_modules_as_functions,
):
    """Patched version of _setup_trace_module_map that fixes onnx_attrs type mismatch."""

    def __register_attribute_hook():
        attr_name = "_onnx_attrs"

        def _track_module_attributes_forward_pre_hook(module, input):
            setattr(module, attr_name, _get_module_attributes(module))

        def _track_module_attributes_forward_hook(module, input, output):
            tracing_state = _C._get_tracing_state()
            if not tracing_state:
                return
            graph = tracing_state.graph()
            onnx_attrs = {}
            if hasattr(module, attr_name):
                onnx_attrs = getattr(module, attr_name)
                delattr(module, attr_name)
            try:
                onnx_attrs = {}  # HACK: to reduce export time # TODO: study behaviour across models
                _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)
            except Exception:
                # Silently skip: scope-attribute tracking is best-effort and not required for export.
                pass

        for m in model.modules():
            m.register_forward_hook(_track_module_attributes_forward_hook)
            m.register_forward_pre_hook(_track_module_attributes_forward_pre_hook)

    def _unqualified_variable_name(qualified_name: str) -> str:
        name_atoms = qualified_name.split(".")
        for i, atom in reversed(list(enumerate(name_atoms))):
            if not atom.isnumeric():
                return ".".join(name_atoms[i:])
        return qualified_name

    trace_module_map = {
        _m: torch._C._jit_onnx_create_full_scope_name(torch.typename(type(_m)), _unqualified_variable_name(_n))
        for _n, _m in model.named_modules()
    }
    torch.jit._trace._trace_module_map = trace_module_map

    if isinstance(export_modules_as_functions, bool) and export_modules_as_functions:
        module_typenames = {torch.typename(type(module)) for module in trace_module_map}
    elif isinstance(export_modules_as_functions, set) and export_modules_as_functions:

        def _find_typename(v):
            if isinstance(v, type):
                return torch.typename(v)
            else:
                raise RuntimeError(
                    "Only type of the `nn.Module` should be passed in the set for argument `export_modules_as_functions`. "
                    f"Got `{type(v).__name__}`."
                )

        module_typenames = {_find_typename(v) for v in export_modules_as_functions}
    else:
        module_typenames = set()

    if module_typenames:
        __register_attribute_hook()

    return module_typenames


def _get_module_attributes(module):
    """Helper function to get module attributes safely."""
    import typing

    import torch.nn

    annotations = typing.get_type_hints(type(module))
    base_m_annotations = typing.get_type_hints(torch.nn.Module)
    [annotations.pop(k, None) for k in base_m_annotations]

    attrs = {}
    for k in annotations:
        try:
            attrs[k] = getattr(module, k)
        except AttributeError:
            _C._jit_onnx_log(f"Skipping module attribute '{k}'")
            continue
    return attrs


def _layerwise_safe_export_passes_enabled():
    try:
        from QEfficient.base.modeling_qeff import QEFFBaseModel
    except Exception:
        return False
    return bool(getattr(QEFFBaseModel, "_layerwise_active", False))


def _enable_safe_export_pass_patches(keep_passes=None):
    global _safe_export_patch_depth

    keep_passes = _SAFE_EXPORT_REQUIRED_PASSES | set(keep_passes or ())
    if _safe_export_patch_depth == 0:
        _safe_export_original_passes.clear()
        for name, replacement in _SAFE_EXPORT_PASS_REPLACEMENTS.items():
            if name in keep_passes:
                continue
            if hasattr(_C, name):
                _safe_export_original_passes[name] = getattr(_C, name)
                setattr(_C, name, replacement)
    _safe_export_patch_depth += 1


def _disable_safe_export_pass_patches():
    global _safe_export_patch_depth

    if _safe_export_patch_depth == 0:
        return

    _safe_export_patch_depth -= 1
    if _safe_export_patch_depth == 0:
        for name, original in _safe_export_original_passes.items():
            setattr(_C, name, original)
        _safe_export_original_passes.clear()


@contextmanager
def layerwise_safe_onnx_export_patches(enabled: bool = True, keep_passes=None):
    """Temporarily disable expensive ONNX exporter passes for layerwise prefill.

    This is a no-op unless the caller explicitly enables it and the process is
    inside the layerwise export context. Regular/non-layerwise export therefore
    keeps the original PyTorch ONNX exporter behavior. DCE stays enabled by
    default because some exported graphs need it to remove aten/prim nodes before
    PyTorch serializes ONNX. ``keep_passes`` can retain additional passes.
    """
    if not enabled or not _layerwise_safe_export_passes_enabled():
        yield
        return

    _enable_safe_export_pass_patches(keep_passes=keep_passes)
    try:
        yield
    finally:
        _disable_safe_export_pass_patches()


def apply_torch_patches():
    """Apply monkey patches for ONNX export."""
    onnx_utils._setup_trace_module_map = _setup_trace_module_map_patched
    if hasattr(onnx_utils, "_get_module_attributes"):
        onnx_utils._get_module_attributes = _get_module_attributes


def undo_torch_patches():
    """Undo monkey patches and restore original functions."""
    onnx_utils._setup_trace_module_map = _original_setup_trace_module_map
    if _original_get_module_attributes:
        onnx_utils._get_module_attributes = _original_get_module_attributes
