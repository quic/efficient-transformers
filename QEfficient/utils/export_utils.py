# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import inspect
import re
import warnings
from pathlib import Path
from typing import Dict

from QEfficient.base.onnx_transforms import CustomOpTransform, RenameFunctionOutputsTransform
from QEfficient.transformers.cache_utils import InvalidIndexProvider
from QEfficient.utils.cache import QEFF_HOME
from QEfficient.utils.hash_utils import create_export_hash
from QEfficient.utils.logging_utils import logger
from QEfficient.utils.torch_patches import apply_torch_patches, undo_torch_patches


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
        subfunction_state = None
        # 1. Setup ONNX subfunctions if requested
        if use_onnx_subfunctions := kwargs.pop("use_onnx_subfunctions", False):
            args, kwargs, subfunction_state = _setup_onnx_subfunctions(self, args, kwargs)

        # 2. Prepare export directory
        export_dir = _prepare_export_directory(self, kwargs)

        # 3. Generate hash and finalize export directory path
        export_hash, filtered_hash_params = _generate_export_hash(self, args, kwargs, func)
        export_dir = export_dir.with_name(export_dir.name + "-" + export_hash)
        kwargs["export_dir"] = export_dir
        self.export_hash = export_hash
        if cache_probe:
            kwargs["_layerwise_cache_probe"] = True

        try:
            # 4. Execute the actual export
            onnx_path = func(self, *args, **kwargs)

            # 5. Save export metadata
            if not cache_probe:
                _save_export_metadata(export_dir, filtered_hash_params)
        finally:
            # 6. Always cleanup subfunctions if they were setup
            if use_onnx_subfunctions:
                _cleanup_onnx_subfunctions(self, subfunction_state)

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
    if func.__name__ == "_export_layerwise":
        export_kwargs = dict(all_args.get("export_kwargs") or {})
        export_kwargs["_qeff_layerwise_export"] = True
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
        qeff_model: The QEff model instance.
        args: Positional export arguments.
        kwargs: Export keyword arguments (modified in-place).

    Returns:
        Updated args/kwargs plus cleanup state used to restore the original
        ONNX transform list after export.
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
            and (
                "key" in name
                or "value" in name
                or "compressed_kv" in name
                or "k_pe" in name
                or "conv" in name
                or "recurrent" in name
            )
            else name
            for name in kwargs["output_names"]
        ]
    else:
        warnings.warn(
            "ONNX subfunctions are enabled, but no retained-state output names were found to rewrite. "
            "Ensure `output_names` includes key/value retained states if subfunction compatibility is required."
        )

    # Work on an instance-local copy and restore it in cleanup. Without this,
    # failed or repeated subfunction exports can leak transforms into later exports.
    original_transforms = list(qeff_model._onnx_transforms)
    qeff_model._onnx_transforms = list(original_transforms)
    if RenameFunctionOutputsTransform not in qeff_model._onnx_transforms:
        qeff_model._onnx_transforms.append(RenameFunctionOutputsTransform)
    if CustomOpTransform not in qeff_model._onnx_transforms:
        qeff_model._onnx_transforms.append(CustomOpTransform)

    submodule_classes = qeff_model.model.get_submodules_for_export()
    if submodule_classes:
        kwargs["export_modules_as_functions"] = submodule_classes
    return args, kwargs, {"onnx_transforms": original_transforms}


def _cleanup_onnx_subfunctions(qeff_model, state=None):
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
    if state is not None and "onnx_transforms" in state:
        qeff_model._onnx_transforms = state["onnx_transforms"]


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
