# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Runtime monkey patches for ONNX export compatibility.

Patches kept here:
  - TorchScript ONNX exporter (_setup_trace_module_map, _get_module_attributes,
    _jit_pass_onnx_track_scope_attributes): fix attribute-type mismatches in the
    legacy trace-based exporter (dynamo=False path).
  - Layerwise safe export pass patches: disable expensive ONNX exporter passes
    for layerwise prefill export (TorchScript path).
  - temporarily_enable_nested_compile_regions / temporarily_disable_nested_compile_regions:
    context managers for dynamo export path subgraph boundary management.
  - preserve_subfunction_source_lines: preserve FX source metadata while Dynamo
    retraces GraphModule subfunctions.
  - invoke_subgraph export reuse_hash_fn patches: backport PyTorch changes that
    allow reuse_hash_fn to partition repeated subgraphs during torch.export.

Patches removed (upstreamed to PyTorch):
  - FunctionalTensorMode.__torch_dispatch__ tracker-entry KeyError
  - _verify_exported_program_signature repeated_subgraph buffer handling
  - ExportedProgram.named_buffers constants fallback
  - invoke_subgraph_placeholder kwargs forwarding
  - materialize_as_graph FunctionalTensor mode handling
  - InvokeSubgraphHOP.gen_schema GraphModule reuse
  - _translate_fx_graph / _convert_fx_arg_to_onnx_arg nested tensor constants
"""

import importlib
import inspect
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

import torch
import torch.onnx.utils as onnx_utils
from torch import _C

try:
    from torch.onnx._internal.torchscript_exporter import utils as ts_utils

    _ts_utils_available = True
except ModuleNotFoundError:
    ts_utils = None
    _ts_utils_available = False

# Store original references before patching
_original_setup_trace_module_map = onnx_utils._setup_trace_module_map
_original_get_module_attributes = getattr(onnx_utils, "_get_module_attributes", None)
_original_model_to_graph = onnx_utils._model_to_graph
_original_track_scope_attrs = getattr(_C, "_jit_pass_onnx_track_scope_attributes", None)
_original_ts_setup_trace_module_map = ts_utils._setup_trace_module_map if _ts_utils_available else None
_original_ts_get_module_attributes = getattr(ts_utils, "_get_module_attributes", None) if _ts_utils_available else None

_PATCHES_ACTIVE = False
_MISSING_INSTANCE_ATTR = object()
_safe_export_patch_depth = 0
_safe_export_original_passes = {}
_INVOKE_SUBGRAPH_EXPORT_PATCH_LOCK = threading.RLock()
_invoke_subgraph_export_patch_depth = 0
_invoke_subgraph_export_patch_state = {}
_SAFE_EXPORT_REQUIRED_PASSES = {
    "_jit_pass_dce",
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects",
    "_jit_pass_constant_propagation",
    "_jit_pass_cse",
    # Keep ONNX constant fold enabled to reduce topology drift between
    # layerwise prefill exports and regular (non-layerwise) prefill exports.
    "_jit_pass_onnx_constant_fold",
}

_GRAPH_CALL_COMPATIBLE_TENSOR_METADATA_EXCLUDED_FIELDS = frozenset({"storage_bytes"})


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


def _model_to_graph_patched(model, *args, **kwargs):
    """Preserve model parameter names through TorchScript ONNX export."""
    graph, params_dict, torch_out = _original_model_to_graph(model, *args, **kwargs)

    parameter_names = {}
    for name, parameter in model.named_parameters():
        if parameter.numel():
            parameter_names.setdefault(parameter.data_ptr(), name)

    graph_inputs = {value.debugName(): value for value in graph.inputs()}
    renamed_params = {}
    for old_name, parameter in params_dict.items():
        new_name = parameter_names.get(parameter.data_ptr())
        graph_input = graph_inputs.get(old_name)
        if (
            old_name.startswith("onnx::")
            and new_name
            and new_name not in params_dict
            and new_name not in renamed_params
            and graph_input is not None
        ):
            graph_input.setDebugName(new_name)
        else:
            new_name = old_name
        renamed_params[new_name] = parameter

    return graph, renamed_params, torch_out


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

    # added _is_safe_value guard to prevent IValue-incompatible
    # types from being passed into the ONNX scope-attribute tracker.
    def _is_safe_value(value):
        if isinstance(value, (int, float, bool, str, torch.Tensor)) or value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(_is_safe_value(item) for item in value)
        return False

    annotations = typing.get_type_hints(type(module))
    base_m_annotations = typing.get_type_hints(torch.nn.Module)
    [annotations.pop(k, None) for k in base_m_annotations]

    attrs = {}
    for k in annotations:
        try:
            value = getattr(module, k)
            # Only include IValue-compatible attribute types
            if _is_safe_value(value):
                attrs[k] = value
        except AttributeError:
            _C._jit_onnx_log(f"Skipping module attribute '{k}'")
            continue
    return attrs


def _track_scope_attributes_patched(graph, attrs):
    """Ensure scope attributes passed to ONNX are IValue-compatible."""
    safe_attrs = {}
    for key, value in attrs.items():
        if isinstance(value, (int, float, bool, str, torch.Tensor)) or value is None:
            safe_attrs[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (int, float, bool, str, torch.Tensor)) or item is None for item in value
        ):
            safe_attrs[key] = value
    return _original_track_scope_attrs(graph, safe_attrs)


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
    if not enabled:
        yield
        return

    _enable_safe_export_pass_patches(keep_passes=keep_passes)
    try:
        yield
    finally:
        _disable_safe_export_pass_patches()


def _normalize_symbolic_graph_metadata_value(value: Any, symbol_map: dict[str, str]) -> Any:
    import re

    import sympy

    node = getattr(value, "node", None)
    if node is None:
        return (type(value), repr(value))

    expr = getattr(node, "_expr", None)
    if expr is None:
        return (type(value), repr(value))

    expr_str = str(sympy.simplify(expr))

    def replace_symbol(match: re.Match[str]) -> str:
        symbol = match.group(0)
        if symbol not in symbol_map:
            symbol_map[symbol] = f"_s{len(symbol_map)}"
        return symbol_map[symbol]

    expr_str = re.sub(r"\bs\d+\b", replace_symbol, expr_str)
    pytype = getattr(value, "ty", getattr(node, "pytype", type(value)))
    return (type(value), pytype, getattr(node, "constant", None), expr_str)


def _normalize_graph_metadata_value(value: Any, symbol_map: dict[str, str]) -> Any:
    try:
        from torch._subclasses._fake_tensor_utils import _PySymInputStub
    except Exception:
        _PySymInputStub = ()

    if isinstance(value, _PySymInputStub):
        return (_PySymInputStub, _normalize_symbolic_graph_metadata_value(value.value, symbol_map))
    return value


def _flatten_graph_metadata_value_for_call_compatibility(value: Any, fake_mode) -> list[Any]:
    from torch._subclasses._fake_tensor_utils import _CacheKeyState

    result: list[Any] = []
    state = _CacheKeyState(fake_mode.shape_env)
    sym_int_cls = getattr(torch, "SymInt", None)
    if isinstance(value, (tuple, list, torch.Size)):
        id_hashed_objects: list[Any] = []
        fake_mode._prep_args_for_hash(result, value, state, id_hashed_objects)
        id_hashed_objects.clear()
    elif sym_int_cls is not None and isinstance(value, sym_int_cls):
        state.convert_sym_int(result, value)
    else:
        result.append(value)
    return result


def _normalize_tensor_metadata_for_graph_call_compatibility(
    metadata: Any, fake_mode
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    import dataclasses

    symbol_map: dict[str, str] = {}
    result = []
    for field in dataclasses.fields(metadata):
        if field.name in _GRAPH_CALL_COMPATIBLE_TENSOR_METADATA_EXCLUDED_FIELDS:
            result.append((field.name, ("<ignored>",)))
            continue
        flattened = _flatten_graph_metadata_value_for_call_compatibility(getattr(metadata, field.name), fake_mode)
        result.append((field.name, tuple(_normalize_graph_metadata_value(value, symbol_map) for value in flattened)))
    return tuple(result)


def _tensor_metadata_graph_call_mismatch(a_metadata: Any, b_metadata: Any, fake_mode) -> str | None:
    a_normalized = _normalize_tensor_metadata_for_graph_call_compatibility(a_metadata, fake_mode)
    b_normalized = _normalize_tensor_metadata_for_graph_call_compatibility(b_metadata, fake_mode)
    if a_normalized == b_normalized:
        return None

    for idx, (a_item, b_item) in enumerate(zip(a_normalized, b_normalized)):
        if a_item != b_item:
            return f"idx={idx}: {a_item!r} ({type(a_item).__name__}) != {b_item!r} ({type(b_item).__name__})"
    return f"metadata length mismatch: {len(a_normalized)} != {len(b_normalized)}"


def _qeff_are_same_graph_modules(
    fn_name: str, a_mod: torch.fx.GraphModule, b_mod: torch.fx.GraphModule, fake_mode
) -> bool:
    import torch.utils._pytree as pytree
    from torch._subclasses.fake_tensor import extract_tensor_metadata

    hop_vars = importlib.import_module("torch._dynamo.variables.higher_order_ops")
    hc_log = getattr(hop_vars, "hc_log", None)

    def log_mismatch(reason: str, a_node=None, b_node=None) -> None:
        if hc_log is not None:
            hc_log.debug("%s: Graph comparison failed: %s; a_node=%s; b_node=%s", fn_name, reason, a_node, b_node)

    node_map = {}

    def check_all_args(a_nodes, b_nodes) -> str | None:
        a_nodes = list(a_nodes)
        b_nodes = list(b_nodes)
        if len(a_nodes) != len(b_nodes):
            return f"arg length mismatch: {len(a_nodes)} != {len(b_nodes)}"
        for idx, (arg_a, arg_b) in enumerate(zip(a_nodes, b_nodes)):
            if isinstance(arg_a, torch.fx.Node):
                mapped = node_map.get(arg_a)
                if mapped != arg_b:
                    return f"arg {idx} node mismatch: {arg_a} maps to {mapped}, got {arg_b}"
            elif isinstance(arg_a, slice):
                if not isinstance(arg_b, slice):
                    return f"arg {idx} slice mismatch: {arg_a} != {arg_b}"
                nested = check_all_args((arg_a.start, arg_a.stop, arg_a.step), (arg_b.start, arg_b.stop, arg_b.step))
                if nested is not None:
                    return f"arg {idx} slice mismatch: {nested}"
            elif arg_a != arg_b:
                return f"arg {idx} value mismatch: {arg_a!r} != {arg_b!r}"
        return None

    a_graph_nodes = list(a_mod.graph.nodes)
    b_graph_nodes = list(b_mod.graph.nodes)
    if len(a_graph_nodes) != len(b_graph_nodes):
        log_mismatch(f"node count mismatch: {len(a_graph_nodes)} != {len(b_graph_nodes)}")
        return False

    sym_int_cls = getattr(torch, "SymInt", None)
    for a_node, b_node in zip(a_graph_nodes, b_graph_nodes):
        if a_node.op != b_node.op:
            log_mismatch(f"op mismatch: {a_node.op} != {b_node.op}", a_node, b_node)
            return False

        if a_node.op == "placeholder":
            a_value = a_node.meta["example_value"]
            b_value = b_node.meta["example_value"]

            if isinstance(a_value, torch.Tensor):
                if not isinstance(b_value, torch.Tensor):
                    log_mismatch(f"placeholder type mismatch: tensor != {type(b_value).__name__}", a_node, b_node)
                    return False
                a_metadata = extract_tensor_metadata(a_value)
                b_metadata = extract_tensor_metadata(b_value)
                mismatch_detail = _tensor_metadata_graph_call_mismatch(a_metadata, b_metadata, fake_mode)
                if mismatch_detail is not None:
                    log_mismatch(f"placeholder tensor metadata mismatch: {mismatch_detail}", a_node, b_node)
                    return False
            elif sym_int_cls is not None and isinstance(a_value, sym_int_cls):
                if not isinstance(b_value, sym_int_cls):
                    log_mismatch(f"placeholder type mismatch: SymInt != {type(b_value).__name__}", a_node, b_node)
                    return False
                if _normalize_symbolic_graph_metadata_value(a_value, {}) != _normalize_symbolic_graph_metadata_value(
                    b_value, {}
                ):
                    log_mismatch(f"placeholder SymInt mismatch: {a_value!r} != {b_value!r}", a_node, b_node)
                    return False
        elif a_node.op == "call_function":
            if a_node.target is not b_node.target:
                log_mismatch(f"call_function target mismatch: {a_node.target} != {b_node.target}", a_node, b_node)
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if reason := check_all_args(a_flat, b_flat):
                log_mismatch(reason, a_node, b_node)
                return False
        elif a_node.op == "call_method":
            if a_node.target != b_node.target:
                log_mismatch(f"call_method target mismatch: {a_node.target} != {b_node.target}", a_node, b_node)
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if reason := check_all_args(a_flat, b_flat):
                log_mismatch(reason, a_node, b_node)
                return False
        elif a_node.op == "output":
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if reason := check_all_args(a_flat, b_flat):
                log_mismatch(reason, a_node, b_node)
                return False
        elif a_node.op == "get_attr":
            a_attr = getattr(a_mod, a_node.target)
            b_attr = getattr(b_mod, b_node.target)
            if isinstance(a_attr, torch.fx.GraphModule):
                if not isinstance(b_attr, torch.fx.GraphModule):
                    log_mismatch(f"get_attr type mismatch: GraphModule != {type(b_attr).__name__}", a_node, b_node)
                    return False
                if not _qeff_are_same_graph_modules(fn_name, a_attr, b_attr, fake_mode):
                    log_mismatch("nested get_attr graph mismatch", a_node, b_node)
                    return False
            else:
                raise NotImplementedError(f"get_attr with {type(a_attr)}")
        else:
            raise NotImplementedError(f"Graph equivalence check saw a {a_node.op}")

        node_map[a_node] = b_node

    return True


def _qeff_cache_table(cache, attr_name):
    table = getattr(cache, attr_name, None)
    if table is None:
        table = defaultdict(list)
        setattr(cache, attr_name, table)
    return table


def _qeff_proxy_cache_table(cache):
    table = getattr(cache, "_qeff_proxy_dispatch_cache_by_reuse_group", None)
    if table is None:
        table = {}
        setattr(cache, "_qeff_proxy_dispatch_cache_by_reuse_group", table)
    return table


def _qeff_add_dynamo_installed_submodule(self, fn_code, identifier, reuse_group_key=None):
    original = _invoke_subgraph_export_patch_state["cache_add_dynamo_installed_submodule"]
    if reuse_group_key is None:
        return original(self, fn_code, identifier)
    cache_key = (fn_code, reuse_group_key)
    _qeff_cache_table(self, "_qeff_dynamo_installed_submodules_by_reuse_group")[cache_key].append(identifier)
    return None


def _qeff_get_dynamo_installed_submodules(self, fn_code, reuse_group_key=None):
    original = _invoke_subgraph_export_patch_state["cache_get_dynamo_installed_submodules"]
    if reuse_group_key is None:
        return original(self, fn_code)
    cache_key = (fn_code, reuse_group_key)
    return _qeff_cache_table(self, "_qeff_dynamo_installed_submodules_by_reuse_group").get(cache_key, [])


def _qeff_add_proxy_dispatch_entry(self, identifier, key, reuse_group_key=None):
    original = _invoke_subgraph_export_patch_state["cache_add_proxy_dispatch_entry"]
    if reuse_group_key is None:
        return original(self, identifier, key)
    cache_key = (identifier, reuse_group_key)
    _qeff_proxy_cache_table(self)[cache_key] = key
    return None


def _qeff_get_proxy_dispatch_entry(self, identifier, reuse_group_key=None):
    original = _invoke_subgraph_export_patch_state["cache_get_proxy_dispatch_entry"]
    if reuse_group_key is None:
        return original(self, identifier)
    cache_key = (identifier, reuse_group_key)
    return _qeff_proxy_cache_table(self).get(cache_key, None)


def _qeff_create_wrapped_node(self, *args, reuse_group_key=None, **kwargs):
    original_create = _invoke_subgraph_export_patch_state["wrap_create_wrapped_node"]
    if reuse_group_key is None:
        return original_create(self, *args, **kwargs)

    instance_previous_install = self.__dict__.get("install_subgraph_in_output_graph", _MISSING_INSTANCE_ATTR)
    previous_install = getattr(self, "install_subgraph_in_output_graph")

    def install_with_extra_kwargs(*install_args, **install_kwargs):
        install_kwargs["reuse_group_key"] = reuse_group_key
        return previous_install(*install_args, **install_kwargs)

    setattr(self, "install_subgraph_in_output_graph", install_with_extra_kwargs)
    try:
        return original_create(self, *args, **kwargs)
    finally:
        if instance_previous_install is _MISSING_INSTANCE_ATTR:
            delattr(self, "install_subgraph_in_output_graph")
        else:
            setattr(self, "install_subgraph_in_output_graph", instance_previous_install)


def _qeff_install_subgraph_in_output_graph(
    self, tx, fn_vt, fn_args_vt, kwargs, body_gmod, attr_name, reuse_group_key=None
):
    invoke_vars = importlib.import_module("torch._dynamo.variables.invoke_subgraph")
    hop_vars = importlib.import_module("torch._dynamo.variables.higher_order_ops")

    UserFunctionVariable = invoke_vars.UserFunctionVariable
    UnspecializedNNModuleVariable = invoke_vars.UnspecializedNNModuleVariable
    graph_break_hints = invoke_vars.graph_break_hints
    unimplemented = invoke_vars.unimplemented
    hc_log = invoke_vars.hc_log

    if not isinstance(fn_vt, (UnspecializedNNModuleVariable, UserFunctionVariable)):
        unimplemented(
            gb_type="Encountered non user function variable during invoke_subgraph HOP tracing",
            context=str(fn_vt),
            explanation="invoke_subgraph does not support non user function variable",
            hints=[*graph_break_hints.SUPPORTABLE],
        )

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )

    if isinstance(fn_vt, UserFunctionVariable):
        fn_code = fn_vt.get_function().__code__
        fn_name = fn_vt.get_function().__name__
    else:
        if not isinstance(fn_vt, UnspecializedNNModuleVariable):
            raise AssertionError(f"expected UnspecializedNNModuleVariable, got {type(fn_vt).__name__}")
        fn_code = fn_vt.value.forward.__func__.__code__
        fn_name = fn_vt.value.forward.__name__
    if reuse_group_key is not None:
        body_gmod.meta["invoke_subgraph_reuse_group_key"] = reuse_group_key

    previously_installed_submodules = []
    if invoke_subgraph_cache:
        previously_installed_submodules = invoke_subgraph_cache.get_dynamo_installed_submodules(
            fn_code, reuse_group_key
        )
        if reuse_group_key is not None:
            hc_log.debug("subgraph_reuse: install lookup for '%s' using reuse group key %s", fn_name, reuse_group_key)
        current_mod = body_gmod
        for submodule_name in reversed(previously_installed_submodules):
            if submodule_name not in tx.output.nn_modules:
                raise AssertionError(f"submodule '{submodule_name}' not found in nn_modules")
            previous_mod = tx.output.nn_modules[submodule_name]
            if not tx.fake_mode:
                raise AssertionError("tx.fake_mode must be set for subgraph comparison")
            if hop_vars.are_same_graph_modules(fn_name, previous_mod, current_mod, tx.fake_mode):
                return submodule_name

    body_name = hop_vars.WrapHigherOrderVariable.install_subgraph_in_output_graph(
        self, tx, fn_vt, fn_args_vt, kwargs, body_gmod, attr_name
    )
    hc_log.debug(
        "%s: Installing subgraph with identifier '%s', bringing total count for '%s' function to %s",
        fn_name,
        body_name,
        fn_name,
        len(previously_installed_submodules) + 1,
    )
    if invoke_subgraph_cache:
        invoke_subgraph_cache.add_dynamo_installed_submodule(fn_code, body_name, reuse_group_key)

    return body_name


def _qeff_invoke_subgraph_call_function(self, tx, args, kwargs):
    invoke_vars = importlib.import_module("torch._dynamo.variables.invoke_subgraph")
    from torch._dynamo.utils import dynamo_timed
    from torch._dynamo.variables.higher_order_ops import _call_function_with_auto_output_flattening

    required_helpers = (
        "build_input_fingerprint",
        "build_reuse_condition",
        "find_reuse_entry_by_key",
        "find_reuse_match",
        "has_reuse_entries",
        "is_reuse_eligible",
        "save_reuse_entry",
        "stamp_out_subgraph",
        "trace_reuse_hash_fn",
    )
    if any(not hasattr(invoke_vars, name) for name in required_helpers):
        original = _invoke_subgraph_export_patch_state["invoke_call_function"]
        return original(self, tx, args, kwargs)

    fn_var = args[0]
    fn_args_vt = args[1:]

    config = None
    max_reuse_entries = 8
    reuse_hash_fn = None
    if hasattr(fn_var, "get_function"):
        try:
            fn = fn_var.get_function()
            config = getattr(fn, "__marked_compile_region_config__", None)
            max_reuse_entries = getattr(fn, "__marked_compile_region_max_reuse_entries__", 8)
            reuse_hash_fn = getattr(fn, "__marked_compile_region_reuse_hash_fn__", None)
        except Exception:
            invoke_vars.log.warning(
                "Failed to extract nested_compile_region() config from InvokeSubgraphHigherOrderVariable. ",
                exc_info=True,
            )
            raise

    invoke_vars.hc_log.debug("reuse_hash_fn for %s: %s", fn_var, reuse_hash_fn)
    is_exporting = getattr(torch.compiler, "is_exporting", lambda: False)
    is_export = tx.output.export or is_exporting()
    reuse = not is_export
    export_reuse_group_key = None

    if is_export and reuse_hash_fn is not None:
        with dynamo_timed("invoke_subgraph_export_reuse_hash_fn"):
            export_reuse_group_key = invoke_vars.trace_reuse_hash_fn(tx, reuse_hash_fn, fn_args_vt, kwargs)
        invoke_vars.hc_log.debug("subgraph_reuse: export reuse_hash_fn key %d for '%s'", export_reuse_group_key, fn_var)

    if reuse and reuse_hash_fn is not None:
        with dynamo_timed("invoke_subgraph_reuse_hash_fn"):
            hash_key = invoke_vars.trace_reuse_hash_fn(tx, reuse_hash_fn, fn_args_vt, kwargs)

        cached = invoke_vars.find_reuse_entry_by_key(tx, fn_var, hash_key)
        if cached is not None:
            invoke_vars.hc_log.debug(
                "subgraph_reuse: hash key %d hit for '%s', reusing subgraph '%s'",
                hash_key,
                fn_var,
                cached.body_name,
            )
            fingerprint = invoke_vars.build_input_fingerprint(tx, fn_args_vt, kwargs)
            with dynamo_timed("invoke_subgraph_reuse_stamp_out"):
                return invoke_vars.stamp_out_subgraph(tx, fingerprint, cached)
    elif reuse and invoke_vars.has_reuse_entries(tx, fn_var):
        with dynamo_timed("invoke_subgraph_reuse_lookup"):
            fingerprint = invoke_vars.build_input_fingerprint(tx, fn_args_vt, kwargs)
            match = invoke_vars.find_reuse_match(tx, fn_var, fingerprint)
        if match is not None:
            invoke_vars.hc_log.debug(
                "subgraph_reuse: cache hit for '%s', reusing subgraph '%s'", fn_var, match.body_name
            )
            with dynamo_timed("invoke_subgraph_reuse_stamp_out"):
                return invoke_vars.stamp_out_subgraph(tx, fingerprint, match)

    if self._HOP_NAME is None:
        raise AssertionError("_HOP_NAME must not be None")
    subgraph_name = "subgraph"
    if export_reuse_group_key is not None:
        subgraph_name = f"{subgraph_name}_{export_reuse_group_key}"
    with dynamo_timed("invoke_subgraph_trace"):
        (
            p_args,
            p_kwargs,
            example_value,
            body_r,
            body_gmod,
            body_name,
            body_graph_output_vts,
            tracing_info,
        ) = self.create_wrapped_node(
            tx,
            fn_var,
            fn_args_vt,
            kwargs,
            self._HOP_NAME,
            subgraph_name=subgraph_name,
            reuse_group_key=export_reuse_group_key,
        )

    if len(p_kwargs) > 0:
        invoke_vars.unimplemented(
            gb_type="invoke_subgraph: kwargs unexpected",
            context=f"args: {args}, kwargs: {kwargs}",
            explanation="kwargs should have been flattened into lifted args.",
            hints=[*invoke_vars.graph_break_hints.DYNAMO_BUG],
        )

    NestedCompileRegionOptions = getattr(
        importlib.import_module("torch._higher_order_ops.invoke_subgraph"),
        "NestedCompileRegionOptions",
        None,
    )
    if NestedCompileRegionOptions is not None and isinstance(config, NestedCompileRegionOptions):
        body_gmod.meta["nested_region_config"] = config

    p_args = (p_args[0], body_name, *p_args[1:])

    if reuse:
        fingerprint = invoke_vars.build_input_fingerprint(tx, fn_args_vt, kwargs)
        if reuse_hash_fn is not None:
            traced_sources = tracing_info.traced_sources
            if not invoke_vars.is_reuse_eligible(
                tx, body_r, fingerprint, tracing_info, traced_sources, has_reuse_hash_fn=True
            ):
                raise RuntimeError(
                    "reuse_hash_fn was provided but the subgraph is not eligible for reuse. "
                    "Check the logs with TORCH_LOGS='+hierarchical_compile' for details."
                )
            invoke_vars.save_reuse_entry(
                tx,
                fn_var,
                fingerprint,
                body_name,
                body_gmod,
                config,
                p_args,
                body_r,
                example_value,
                max_reuse_entries,
                hash_key=hash_key,
            )
        else:
            traced_sources = tracing_info.traced_sources
            if invoke_vars.is_reuse_eligible(tx, body_r, fingerprint, tracing_info, traced_sources):
                condition = invoke_vars.build_reuse_condition(tx, fingerprint, traced_sources)
                if condition is not None:
                    invoke_vars.save_reuse_entry(
                        tx,
                        fn_var,
                        fingerprint,
                        body_name,
                        body_gmod,
                        config,
                        p_args,
                        body_r,
                        example_value,
                        max_reuse_entries,
                        condition=condition,
                    )

    return _call_function_with_auto_output_flattening(
        tx,
        torch._higher_order_ops.invoke_subgraph,
        tuple(p_args),
        p_kwargs,
        example_value,
        body_r,
        body_graph_output_vts,
        config=config,
    )


def _enable_invoke_subgraph_export_reuse_hash_patches():
    global _invoke_subgraph_export_patch_depth

    with _INVOKE_SUBGRAPH_EXPORT_PATCH_LOCK:
        if _invoke_subgraph_export_patch_depth == 0:
            hop_vars = importlib.import_module("torch._dynamo.variables.higher_order_ops")
            invoke_vars = importlib.import_module("torch._dynamo.variables.invoke_subgraph")
            guards = importlib.import_module("torch._guards")

            _invoke_subgraph_export_patch_state.clear()
            _invoke_subgraph_export_patch_state.update(
                {
                    "are_same_graph_modules": hop_vars.are_same_graph_modules,
                    "wrap_create_wrapped_node": hop_vars.WrapHigherOrderVariable.create_wrapped_node,
                    "invoke_install_subgraph": (
                        invoke_vars.InvokeSubgraphHigherOrderVariable.install_subgraph_in_output_graph
                    ),
                    "invoke_call_function": invoke_vars.InvokeSubgraphHigherOrderVariable._call_function,
                    "cache_add_dynamo_installed_submodule": guards.InvokeSubgraphCache.add_dynamo_installed_submodule,
                    "cache_get_dynamo_installed_submodules": guards.InvokeSubgraphCache.get_dynamo_installed_submodules,
                    "cache_add_proxy_dispatch_entry": guards.InvokeSubgraphCache.add_proxy_dispatch_entry,
                    "cache_get_proxy_dispatch_entry": guards.InvokeSubgraphCache.get_proxy_dispatch_entry,
                }
            )

            hop_vars.are_same_graph_modules = _qeff_are_same_graph_modules
            hop_vars.WrapHigherOrderVariable.create_wrapped_node = _qeff_create_wrapped_node
            invoke_vars.InvokeSubgraphHigherOrderVariable.install_subgraph_in_output_graph = (
                _qeff_install_subgraph_in_output_graph
            )
            invoke_vars.InvokeSubgraphHigherOrderVariable._call_function = _qeff_invoke_subgraph_call_function
            guards.InvokeSubgraphCache.add_dynamo_installed_submodule = _qeff_add_dynamo_installed_submodule
            guards.InvokeSubgraphCache.get_dynamo_installed_submodules = _qeff_get_dynamo_installed_submodules
            guards.InvokeSubgraphCache.add_proxy_dispatch_entry = _qeff_add_proxy_dispatch_entry
            guards.InvokeSubgraphCache.get_proxy_dispatch_entry = _qeff_get_proxy_dispatch_entry
        _invoke_subgraph_export_patch_depth += 1


def _disable_invoke_subgraph_export_reuse_hash_patches():
    global _invoke_subgraph_export_patch_depth

    with _INVOKE_SUBGRAPH_EXPORT_PATCH_LOCK:
        if _invoke_subgraph_export_patch_depth == 0:
            return

        _invoke_subgraph_export_patch_depth -= 1
        if _invoke_subgraph_export_patch_depth == 0:
            hop_vars = importlib.import_module("torch._dynamo.variables.higher_order_ops")
            invoke_vars = importlib.import_module("torch._dynamo.variables.invoke_subgraph")
            guards = importlib.import_module("torch._guards")

            hop_vars.are_same_graph_modules = _invoke_subgraph_export_patch_state["are_same_graph_modules"]
            hop_vars.WrapHigherOrderVariable.create_wrapped_node = _invoke_subgraph_export_patch_state[
                "wrap_create_wrapped_node"
            ]
            invoke_cls = invoke_vars.InvokeSubgraphHigherOrderVariable
            invoke_cls.install_subgraph_in_output_graph = _invoke_subgraph_export_patch_state["invoke_install_subgraph"]
            invoke_vars.InvokeSubgraphHigherOrderVariable._call_function = _invoke_subgraph_export_patch_state[
                "invoke_call_function"
            ]
            guards.InvokeSubgraphCache.add_dynamo_installed_submodule = _invoke_subgraph_export_patch_state[
                "cache_add_dynamo_installed_submodule"
            ]
            guards.InvokeSubgraphCache.get_dynamo_installed_submodules = _invoke_subgraph_export_patch_state[
                "cache_get_dynamo_installed_submodules"
            ]
            guards.InvokeSubgraphCache.add_proxy_dispatch_entry = _invoke_subgraph_export_patch_state[
                "cache_add_proxy_dispatch_entry"
            ]
            guards.InvokeSubgraphCache.get_proxy_dispatch_entry = _invoke_subgraph_export_patch_state[
                "cache_get_proxy_dispatch_entry"
            ]
            _invoke_subgraph_export_patch_state.clear()


def apply_torch_patches():
    """Apply monkey patches for ONNX export (TorchScript path)."""
    global _PATCHES_ACTIVE
    if _PATCHES_ACTIVE:
        return

    # Patch onnx_utils (used by both TorchScript and as fallback)
    onnx_utils._setup_trace_module_map = _setup_trace_module_map_patched
    onnx_utils._model_to_graph = _model_to_graph_patched
    if hasattr(onnx_utils, "_get_module_attributes"):
        onnx_utils._get_module_attributes = _get_module_attributes

    # Patch ts_utils (TorchScript-specific exporter utilities, torch >= 2.13 only)
    if _ts_utils_available:
        ts_utils._setup_trace_module_map = _setup_trace_module_map_patched
        if hasattr(ts_utils, "_get_module_attributes"):
            ts_utils._get_module_attributes = _get_module_attributes

    # Patch _C scope-attribute tracker to filter out IValue-incompatible types
    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _track_scope_attributes_patched

    _enable_invoke_subgraph_export_reuse_hash_patches()
    _PATCHES_ACTIVE = True


def undo_torch_patches():
    """Undo monkey patches and restore original functions."""
    global _PATCHES_ACTIVE
    if not _PATCHES_ACTIVE:
        return

    onnx_utils._setup_trace_module_map = _original_setup_trace_module_map
    onnx_utils._model_to_graph = _original_model_to_graph
    if _original_get_module_attributes:
        onnx_utils._get_module_attributes = _original_get_module_attributes

    if _ts_utils_available:
        ts_utils._setup_trace_module_map = _original_ts_setup_trace_module_map
        if _original_ts_get_module_attributes:
            ts_utils._get_module_attributes = _original_ts_get_module_attributes

    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _original_track_scope_attrs

    _disable_invoke_subgraph_export_reuse_hash_patches()
    _PATCHES_ACTIVE = False


@contextmanager
def temporarily_enable_nested_compile_regions(model, target_classes=None):
    """
    Wrap selected module ``forward`` methods with ``nested_compile_region``
    during export so repeated block functions are materialized by dynamo.

    Used when dynamo=True and use_onnx_subfunctions=True. Requires torch >= 2.13.
    """
    target_classes = tuple(target_classes) if target_classes else None
    patched_modules = []

    _enable_invoke_subgraph_export_reuse_hash_patches()
    try:
        for module in model.modules():
            if target_classes and not isinstance(module, target_classes):
                continue

            bound_forward = getattr(module, "forward", None)
            if bound_forward is None:
                continue

            wrapped_forward = getattr(bound_forward, "__func__", bound_forward)
            if getattr(wrapped_forward, "__qualname__", "") == "mark_compile_region.<locals>.wrap.<locals>.inner":
                continue

            previous_forward = module.__dict__.get("forward", _MISSING_INSTANCE_ATTR)
            nested_forward = torch.compiler.nested_compile_region(wrapped_forward)
            setattr(module, "forward", nested_forward.__get__(module, type(module)))
            patched_modules.append((module, previous_forward))

        yield
    finally:
        for module, previous_forward in reversed(patched_modules):
            if previous_forward is _MISSING_INSTANCE_ATTR:
                delattr(module, "forward")
            else:
                setattr(module, "forward", previous_forward)
        _disable_invoke_subgraph_export_reuse_hash_patches()


@contextmanager
def temporarily_disable_nested_compile_regions(model, target_classes=None):
    """
    Replace nested_compile_region-wrapped ``forward`` methods with their original
    underlying functions for the duration of plain dynamo export (flat graph path).

    Used during weight-free export with use_onnx_subfunctions=False so that
    @nested_compile_region boundaries on decoder layer forward() methods do not
    create unwanted subgraph splits during tracing.
    """
    target_classes = tuple(target_classes) if target_classes else None
    patched_modules = []

    try:
        for module in model.modules():
            if target_classes and not isinstance(module, target_classes):
                continue

            bound_forward = getattr(module, "forward", None)
            if bound_forward is None:
                continue

            wrapped_forward = getattr(bound_forward, "__func__", bound_forward)
            if getattr(wrapped_forward, "__qualname__", "") != "mark_compile_region.<locals>.wrap.<locals>.inner":
                continue

            closure = getattr(wrapped_forward, "__closure__", None) or ()
            original_forward = next(
                (cell.cell_contents for cell in closure if inspect.isfunction(cell.cell_contents)),
                None,
            )
            if original_forward is None:
                continue

            previous_forward = module.__dict__.get("forward", _MISSING_INSTANCE_ATTR)
            setattr(module, "forward", original_forward.__get__(module, type(module)))
            patched_modules.append((module, previous_forward))

        yield
    finally:
        for module, previous_forward in reversed(patched_modules):
            if previous_forward is _MISSING_INSTANCE_ATTR:
                delattr(module, "forward")
            else:
                setattr(module, "forward", previous_forward)


_DYNAMO_ENV_LOCK = threading.RLock()
_SUBFUNCTION_SOURCE_PATCH_LOCK = threading.RLock()


@contextmanager
def preserve_subfunction_source_lines():
    """Preserve original FX source metadata while retracing Dynamo subfunctions.

    PyTorch's generated GraphModule code does not preserve the original source
    locations when an ``invoke_subgraph`` GraphModule is retraced. Running the
    GraphModule through ``torch.fx.Interpreter`` under ``preserve_node_meta``
    keeps the metadata available to the ONNX exporter.

    This patches a private PyTorch API for the duration of one export. The
    patch is process-global, so the lock serializes callers and the original
    function is always restored on exit.
    """
    with _SUBFUNCTION_SOURCE_PATCH_LOCK:
        module = importlib.import_module("torch._higher_order_ops.invoke_subgraph")
        original = module.reenter_make_fx

        def retrace(fn, *args, **kwargs):
            if not isinstance(fn, torch.fx.GraphModule):
                return original(fn, *args, **kwargs)

            def interpreted(*operands):
                with torch.fx.traceback.preserve_node_meta():
                    return torch.fx.Interpreter(fn).run(*operands)

            return original(interpreted, *args, **kwargs)

        module.reenter_make_fx = retrace
        try:
            yield
        finally:
            module.reenter_make_fx = original


@contextmanager
def dynamo_invoke_subgraph_fallback_env():
    """Temporarily set TORCH_INVOKE_ALLOW_CREATE_FALLBACK=1 for dynamo export.

    torch.onnx.export's dynamo path (dynamo=True) needs this env var set to
    allow invoke_subgraph placeholders to fall back correctly during tracing.
    Saves and restores whatever value (or absence) the caller's environment
    already had, rather than assuming it was previously unset.
    Uses an RLock so concurrent exports don't race on the env var.
    """
    with _DYNAMO_ENV_LOCK:
        previous = os.environ.get("TORCH_INVOKE_ALLOW_CREATE_FALLBACK")
        os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("TORCH_INVOKE_ALLOW_CREATE_FALLBACK", None)
            else:
                os.environ["TORCH_INVOKE_ALLOW_CREATE_FALLBACK"] = previous
