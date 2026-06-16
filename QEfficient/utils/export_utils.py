# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import contextlib
import copy
import inspect
import os
import re
import warnings
from pathlib import Path
from typing import Dict

import torch

from QEfficient.base.onnx_transforms import CustomOpTransform, RenameFunctionOutputsTransform
from QEfficient.transformers.cache_utils import InvalidIndexProvider
from QEfficient.utils.cache import QEFF_HOME
from QEfficient.utils.hash_utils import create_export_hash
from QEfficient.utils.logging_utils import logger
from QEfficient.utils.torch_patches import apply_torch_patches, undo_torch_patches

_SAFE_ONNX_EXPORT_PASS_NAMES = (
    "_jit_pass_constant_propagation",
    "_jit_pass_dce",
    "_jit_pass_cse",
    "_jit_pass_canonicalize_graph_fuser_ops",
    "_jit_pass_peephole",
    "_jit_pass_fuse_addmm",
    "_jit_pass_onnx_eval_peephole",
    "_jit_pass_onnx_constant_fold",
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects",
    "_jit_pass_canonicalize",
    "_jit_pass_onnx_graph_shape_type_inference",
    "_jit_pass_onnx_deduplicate_initializers",
)


def _qeff_noop(*args, **kwargs):
    return None


def _qeff_false_noop(*args, **kwargs):
    return False


def _qeff_first_arg(*args, **kwargs):
    return args[0] if args else None


def _qeff_second_arg_or_empty_dict(*args, **kwargs):
    if len(args) >= 2:
        return args[1]
    return kwargs.get("params_dict", {})


_SAFE_ONNX_EXPORT_PASS_REPLACEMENTS = {
    "_jit_pass_constant_propagation": _qeff_noop,
    "_jit_pass_dce": _qeff_noop,
    "_jit_pass_cse": _qeff_false_noop,
    "_jit_pass_canonicalize_graph_fuser_ops": _qeff_noop,
    "_jit_pass_peephole": _qeff_noop,
    "_jit_pass_fuse_addmm": _qeff_noop,
    "_jit_pass_onnx_eval_peephole": _qeff_second_arg_or_empty_dict,
    "_jit_pass_onnx_constant_fold": _qeff_second_arg_or_empty_dict,
    "_jit_pass_dce_allow_deleting_nodes_with_side_effects": _qeff_noop,
    "_jit_pass_canonicalize": _qeff_first_arg,
    "_jit_pass_onnx_graph_shape_type_inference": _qeff_noop,
    "_jit_pass_onnx_deduplicate_initializers": _qeff_second_arg_or_empty_dict,
}


def _safe_onnx_export_passes_from_env(default_disable_safe_passes: bool = False) -> tuple[str, ...]:
    disabled_passes = os.environ.get("QEFF_ONNX_DISABLE_SAFE_EXPORT_PASSES")
    if disabled_passes is None:
        return _SAFE_ONNX_EXPORT_PASS_NAMES if default_disable_safe_passes else ()

    normalized_value = disabled_passes.strip().lower()
    if normalized_value in {"", "0", "false", "no", "off"}:
        return ()
    if normalized_value in {"1", "true", "yes", "on", "safe", "all"}:
        return _SAFE_ONNX_EXPORT_PASS_NAMES

    requested_passes = []
    for raw_name in disabled_passes.split(","):
        pass_name = raw_name.strip()
        if not pass_name:
            continue
        if not pass_name.startswith("_jit_pass_"):
            pass_name = f"_jit_pass_{pass_name}"
        if pass_name not in _SAFE_ONNX_EXPORT_PASS_REPLACEMENTS:
            raise ValueError(
                "QEFF_ONNX_DISABLE_SAFE_EXPORT_PASSES can only include safe passes. "
                f"Unknown or unsafe pass: {raw_name!r}. "
                f"Allowed passes: {', '.join(_SAFE_ONNX_EXPORT_PASS_NAMES)}"
            )
        requested_passes.append(pass_name)
    return tuple(dict.fromkeys(requested_passes))


@contextlib.contextmanager
def _disable_safe_onnx_export_passes_from_env(default_disable_safe_passes: bool = False):
    pass_names = _safe_onnx_export_passes_from_env(default_disable_safe_passes=default_disable_safe_passes)
    if not pass_names:
        yield
        return

    originals = []
    for pass_name in pass_names:
        if not hasattr(torch._C, pass_name):
            raise AttributeError(f"torch._C has no ONNX export pass named {pass_name}")
        originals.append((pass_name, getattr(torch._C, pass_name)))
        setattr(torch._C, pass_name, _SAFE_ONNX_EXPORT_PASS_REPLACEMENTS[pass_name])
    logger.info("Disabled safe torch.onnx export passes: %s", ", ".join(pass_names))
    try:
        yield
    finally:
        for pass_name, original in reversed(originals):
            setattr(torch._C, pass_name, original)


def _apply_onnx_export_env_kwargs(export_kwargs: Dict) -> None:
    do_constant_folding = os.environ.get("QEFF_ONNX_DO_CONSTANT_FOLDING")
    if do_constant_folding is None or "do_constant_folding" in export_kwargs:
        return

    normalized_value = do_constant_folding.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        export_kwargs["do_constant_folding"] = True
    elif normalized_value in {"0", "false", "no", "off"}:
        export_kwargs["do_constant_folding"] = False
    else:
        raise ValueError(
            "QEFF_ONNX_DO_CONSTANT_FOLDING must be one of "
            "1/0, true/false, yes/no, or on/off. "
            f"Got: {do_constant_folding!r}"
        )


def export_wrapper(func):
    """
    Decorator for export methods that orchestrates the complete export lifecycle.

    Responsibilities:
    1. Prepare export directory structure
    2. Generate reproducible hash for export configuration
    3. Setup ONNX subfunction environment (if enabled)
    4. Execute the wrapped export function
    5. Cleanup subfunction environment (if enabled)
    6. Save export metadata

    Args:
        func: The export method to wrap (typically _export)

    Returns:
        Wrapped function with complete export lifecycle management
    """

    def wrapper(self, *args, **kwargs):
        cache_probe = kwargs.pop("_layerwise_cache_probe", False)
        # 1. Setup ONNX subfunctions if requested
        if use_onnx_subfunctions := kwargs.pop("use_onnx_subfunctions", False):
            args, kwargs = _setup_onnx_subfunctions(self, args, kwargs)

        # 2. Prepare export directory
        export_dir = _prepare_export_directory(self, kwargs)

        # 3. Generate hash and finalize export directory path
        export_hash, filtered_hash_params = _generate_export_hash(self, args, kwargs, func)
        export_dir = export_dir.with_name(export_dir.name + "-" + export_hash)
        kwargs["export_dir"] = export_dir
        self.export_hash = export_hash
        if cache_probe:
            kwargs["_layerwise_cache_probe"] = True

        # 4. Execute the actual export
        onnx_path = func(self, *args, **kwargs)

        # 5. Save export metadata
        if not cache_probe:
            _save_export_metadata(export_dir, filtered_hash_params)

        # 6. Always cleanup subfunctions if they were setup
        if use_onnx_subfunctions:
            _cleanup_onnx_subfunctions(self)

        return onnx_path

    return wrapper


def _prepare_export_directory(qeff_model, kwargs) -> Path:
    """
    Prepare and return the base export directory path.

    Args:
        qeff_model: The QEff model instance
        kwargs: Keyword arguments containing optional export_dir

    Returns:
        Path object for the base export directory
    """
    export_dir = kwargs.get("export_dir", None)
    parent_dir = qeff_model.model_architecture or qeff_model.model_name
    return Path(export_dir or (QEFF_HOME / parent_dir / qeff_model.model_name))


def _generate_export_hash(qeff_model, args, kwargs, func):
    """
    Generate export hash from model parameters and export arguments.

    The hash ensures reproducibility and prevents conflicts between
    different export configurations.

    Args:
        qeff_model: The QEff model instance
        args: Positional arguments to the export function
        kwargs: Keyword arguments to the export function
        func: The export function being wrapped

    Returns:
        Tuple of (export_hash: str, filtered_hash_params: dict)
    """
    # Extract function signature
    original_sig = inspect.signature(func)
    params = list(original_sig.parameters.values())[1:]  # Skip 'self'
    new_sig = inspect.Signature(params)
    # Bind all arguments
    bound_args = new_sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    all_args = bound_args.arguments
    export_kwargs = dict(all_args.get("export_kwargs") or {})
    is_layerwise_export = func.__name__ == "_export_layerwise"
    if is_layerwise_export:
        export_kwargs["_qeff_layerwise_export"] = True

    do_constant_folding_env = os.environ.get("QEFF_ONNX_DO_CONSTANT_FOLDING")
    if do_constant_folding_env is not None and "do_constant_folding" not in export_kwargs:
        export_kwargs["_qeff_env_onnx_do_constant_folding"] = do_constant_folding_env.strip().lower()

    disabled_safe_passes_env = os.environ.get("QEFF_ONNX_DISABLE_SAFE_EXPORT_PASSES")
    if disabled_safe_passes_env is not None:
        export_kwargs["_qeff_env_onnx_disable_safe_export_passes"] = disabled_safe_passes_env.strip().lower()
    elif is_layerwise_export:
        if all_args.get("prefill_only"):
            export_kwargs["_qeff_layerwise_prefill_default_enable_safe_export_passes"] = True
        else:
            export_kwargs["_qeff_layerwise_default_disable_safe_export_passes"] = True

    if export_kwargs:
        all_args["export_kwargs"] = export_kwargs

    # Use the model's current configuration for hashing to ensure any post-load modifications are captured
    # TODO: Replace with get_model_config property of modeling classes and remove the if-else
    # Determine the config dict to use, preferring .to_diff_dict() if available
    if hasattr(qeff_model.model, "config") and hasattr(qeff_model.model.config, "to_diff_dict"):
        config_val = qeff_model.model.config.to_diff_dict()
    elif hasattr(qeff_model.model, "model") and hasattr(qeff_model.model.model.config, "to_diff_dict"):
        config_val = qeff_model.model.model.config.to_diff_dict()
    else:
        config_val = qeff_model.model.config

    copy_of_hash_params = copy.deepcopy(qeff_model.hash_params)
    copy_of_hash_params.update(
        {
            "config": config_val,
        }
    )
    # Generate hash from relevant parameters
    export_hash, filtered_hash_params = create_export_hash(
        model_params=copy_of_hash_params,
        output_names=all_args.get("output_names"),
        dynamic_axes=all_args.get("dynamic_axes"),
        blocking_kwargs=all_args.get("blocking_kwargs", None),
        export_kwargs=all_args.get("export_kwargs", None),
        onnx_transform_kwargs=all_args.get("onnx_transform_kwargs", None),
    )

    return export_hash, filtered_hash_params


def _setup_onnx_subfunctions(qeff_model, args, kwargs):
    """
    Setup ONNX subfunction export environment.

    This function prepares the model and environment for exporting with
    ONNX subfunctions enabled. It:
    - Applies necessary torch patches
    - Modifies output names for subfunction compatibility
    - Adds subfunction-specific ONNX transforms
    - Updates export kwargs with module classes

    Args:
        qeff_model: The QEff model instance
        kwargs: Export keyword arguments (modified in-place).
    """
    warnings.warn(
        "The subfunction feature is experimental. Please note that using compile "
        "consecutively with and without subfunction may produce inconsistent results."
    )

    # Apply torch patches for subfunction support
    apply_torch_patches()
    InvalidIndexProvider.SUBFUNC_ENABLED = True

    # Transform output names for subfunction compatibility
    if "output_names" in kwargs:
        kwargs["output_names"] = [
            re.sub("_RetainedState", "_InternalRetainedState", name)
            if name.endswith("_RetainedState")
            and ("key" in name or "value" in name or "compressed_kv" in name or "k_pe" in name)
            else name
            for name in kwargs["output_names"]
        ]
    else:
        warnings.warn(
            "ONNX subfunctions are enabled, but no retained-state output names were found to rewrite. "
            "Ensure `output_names` includes key/value retained states if subfunction compatibility is required."
        )

    # Add subfunction-specific ONNX transforms
    qeff_model._onnx_transforms.append(RenameFunctionOutputsTransform)
    qeff_model._onnx_transforms.append(CustomOpTransform)

    submodule_classes = qeff_model.model.get_submodules_for_export()
    if submodule_classes:
        kwargs["export_modules_as_functions"] = submodule_classes
    return args, kwargs


def _cleanup_onnx_subfunctions(qeff_model):
    """
    Cleanup ONNX subfunction export environment.

    Restores the model and environment to pre-subfunction state by:
    - Undoing torch patches
    - Resetting InvalidIndexProvider flag
    - Restoring original ONNX transforms list

    Args:
        qeff_model: The QEff model instance

    Note:
        This function is called in a finally block to ensure cleanup
        even if export fails. Errors during cleanup are logged but
        not re-raised to avoid masking the original exception.
    """
    # Undo torch patches
    undo_torch_patches()
    InvalidIndexProvider.SUBFUNC_ENABLED = False
    qeff_model._onnx_transforms.remove(RenameFunctionOutputsTransform)
    qeff_model._onnx_transforms.remove(CustomOpTransform)


def _save_export_metadata(export_dir: Path, filtered_hash_params: Dict):
    """
    Save export metadata to JSON file for reproducibility.

    Args:
        export_dir: Directory where the export was saved
        filtered_hash_params: Dictionary of parameters used for hashing
    """
    # Import here to avoid circular dependency
    from QEfficient.utils._utils import create_json

    hashed_params_path = export_dir / "hashed_export_params.json"
    create_json(hashed_params_path, filtered_hash_params)
    logger.info("Hashed parameters exported successfully.")
