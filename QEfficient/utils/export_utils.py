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
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from QEfficient.base.onnx_transforms import (
    CustomOpTransform,
    PreserveNestedCacheRetainedStateTransform,
    RenameFunctionOutputsTransform,
    RenameRepeatedSubgraphTransform,
)
from QEfficient.transformers.cache_utils import InvalidIndexProvider
from QEfficient.utils.cache import QEFF_HOME
from QEfficient.utils.hash_utils import create_export_hash
from QEfficient.utils.logging_utils import logger
from QEfficient.utils.torch_patches import (
    apply_torch_patches,
    temporarily_disable_nested_compile_regions,
    undo_torch_patches,
)



def _resolve_attr_path(root, attr_path):
    current = root
    for attr in attr_path.split("."):
        if current is None or not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def _extract_repeated_block_class(candidate):
    if isinstance(candidate, (nn.ModuleList, nn.Sequential, list, tuple)):
        modules = [item for item in candidate if isinstance(item, nn.Module)]
        if len(modules) < 2:
            return None
        first_cls = modules[0].__class__
        if all(isinstance(item, first_cls) for item in modules):
            return first_cls
    return None


def _discover_submodule_classes_for_export(model):
    discovered = set()

    attr_paths = [
        "layers",
        "h",
        "model.layers",
        "model.h",
        "decoder.layers",
        "model.decoder.layers",
        "encoder.layer",
        "encoder.layers",
        "model.encoder.layer",
        "model.encoder.layers",
        "transformer.h",
        "transformer.layers",
        "model.transformer.h",
        "model.transformer.layers",
        "language_model.layers",
        "language_model.model.layers",
        "llm.layers",
        "llm.model.layers",
        "vision_model.encoder.layers",
        "vision_model.transformer.layers",
        "model.vision_model.encoder.layers",
        "model.vision_model.transformer.layers",
        "vision_tower.transformer.layers",
        "vision_tower.vision_model.encoder.layers",
        "model.vision_tower.transformer.layers",
        "model.vision_tower.vision_model.encoder.layers",
    ]

    for attr_path in attr_paths:
        candidate = _resolve_attr_path(model, attr_path)
        repeated_cls = _extract_repeated_block_class(candidate)
        if repeated_cls is not None:
            discovered.add(repeated_cls)

    if discovered:
        return discovered

    repeated_suffixes = (
        ".layers",
        ".layer",
        ".h",
        ".blocks",
        ".block",
        ".encoder_layers",
        ".decoder_layers",
    )

    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not module_name.endswith(repeated_suffixes):
            continue
        repeated_cls = _extract_repeated_block_class(module)
        if repeated_cls is None:
            continue
        cls_name = repeated_cls.__name__.lower()
        if any(token in cls_name for token in ("layer", "block", "decoder", "encoder")):
            discovered.add(repeated_cls)

    return discovered


def get_decoder_layer_classes_for_export(model):
    get_submodules_for_export = getattr(model, "get_submodules_for_export", None)
    if get_submodules_for_export is not None:
        try:
            submodule_classes = get_submodules_for_export()
            if submodule_classes:
                return {cls for cls in submodule_classes if inspect.isclass(cls)}
        except Exception as exc:
            logger.warning(
                f"get_submodules_for_export failed for {model.__class__.__name__}: "
                f"{type(exc).__name__}: {exc}. Falling back to auto-discovery."
            )

    discovered = _discover_submodule_classes_for_export(model)
    if discovered:
        logger.info(
            "Auto-discovered repeated submodule classes for export: "
            + ", ".join(sorted(cls.__name__ for cls in discovered))
        )
        return discovered
    return []


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
        # Extract flags
        use_dynamo = kwargs.get("use_dynamo", False)
        use_onnx_subfunctions = kwargs.pop("use_onnx_subfunctions", False)

        # Cache probe flag (used for layerwise inspection runs)
        cache_probe = kwargs.pop("_layerwise_cache_probe", False)

        # Default context managers and state trackers
        export_context = nullcontext()
        subfunction_state = None

        # 1. Setup the requested export mode
        if use_onnx_subfunctions:
            if use_dynamo:
                # Path 4: dynamo + subfunctions.
                # The @nested_compile_region decorator is already statically present
                # on each decoder layer's forward() method — dynamo will naturally
                # emit repeated_subgraph* functions without any dynamic patching.
                # Derive target_classnames directly from get_submodules_for_export()
                # on the model so RenameRepeatedSubgraphTransform can rename them.
                submodule_classes = getattr(self.model, "get_submodules_for_export", lambda: set())()
                target_classnames = sorted(cls.__name__ for cls in (submodule_classes or []))
                args, kwargs, subfunction_state = _setup_onnx_subfunctions(
                    self, args, kwargs, target_classnames=target_classnames
                )
            else:
                # Path 2: TorchScript + subfunctions.
                args, kwargs, subfunction_state = _setup_onnx_subfunctions(self, args, kwargs)

        elif use_dynamo:
            # Path 3: flat dynamo — strip any @nested_compile_region decorators that
            # are statically present on decoder layer forward() methods so they don't
            # create subgraph boundaries during tracing.
            # target_classes=None: the function identifies wrapped methods by qualname,
            # no class filter needed.
            export_context = temporarily_disable_nested_compile_regions(self.model, target_classes=None)

        # 2. Prepare export directory
        export_dir = _prepare_export_directory(self, kwargs)

        # 3. Generate hash and finalize export directory path
        export_hash, filtered_hash_params = _generate_export_hash(self, args, kwargs, func)
        export_dir = export_dir.with_name(export_dir.name + "-" + export_hash)
        kwargs["export_dir"] = export_dir
        self.export_hash = export_hash

        # Re-inject cache probe flag if needed
        if cache_probe:
            kwargs["_layerwise_cache_probe"] = True

        try:
            # 4. Execute the actual export
            # For the dynamo+subfunctions path each decoder layer must be
            # preserved as a repeated_subgraph ONNX function so that
            # PreserveNestedCacheRetainedStateTransform can rename the
            # _RetainedState inputs to plain past_key.X / past_value.X.
            # The default inline_single_use_invoke_subgraph=True would
            # inline single-use layers before the ONNX translation,
            # leaving CtxScatter nodes at top level with _RetainedState
            # input names that the ORT runner doesn't know to feed.
            dynamo_patch = (
                torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
                if use_onnx_subfunctions and use_dynamo
                else nullcontext()
            )
            try:
                with export_context:
                    with dynamo_patch:
                        onnx_path = func(self, *args, **kwargs)
            except Exception as export_exc:
                if use_onnx_subfunctions and use_dynamo:
                    raise RuntimeError(
                        "Export failed with use_dynamo=True and use_onnx_subfunctions=True "
                        "while nested compile regions were enabled for repeated-subgraph "
                        f"extraction ({type(export_exc).__name__}: {export_exc}). "
                        "Retry export with use_onnx_subfunctions=False for this model/runtime."
                    ) from export_exc
                raise

            # 5. Save export metadata (skip when running cache probe)
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
            "use_onnx_subfunctions": getattr(qeff_model, "_use_onnx_subfunctions", False),
            "onnx_transform_version": 1,
            "use_dynamo": all_args.get("use_dynamo", False),
        }
    )
    if getattr(qeff_model, "_use_onnx_subfunctions", False):
        copy_of_hash_params["onnx_subfunction_version"] = 2
    # Generate hash from relevant parameters
    export_hash, filtered_hash_params = create_export_hash(
        model_params=copy_of_hash_params,
        output_names=all_args.get("output_names"),
        dynamic_axes=all_args.get("dynamic_axes"),
        blocking_kwargs=all_args.get("blocking_kwargs", None),
        dynamic_shapes=all_args.get("dynamic_shapes"),
        export_kwargs=all_args.get("export_kwargs", None),
        onnx_transform_kwargs=all_args.get("onnx_transform_kwargs", None),
    )

    return export_hash, filtered_hash_params


def _setup_onnx_subfunctions(qeff_model, args, kwargs, target_classnames=None):
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
    qeff_model._use_onnx_subfunctions = True
    qeff_model.hash_params["use_onnx_subfunctions"] = True
    qeff_model.hash_params["onnx_subfunction_version"] = 2
    use_dynamo = kwargs.get("use_dynamo", False)
    # Apply torch patches for subfunction support
    apply_torch_patches()
    InvalidIndexProvider.SUBFUNC_ENABLED = True

    # For legacy export-modules-as-functions path we keep internal retained-state
    # names and restore them later via RenameFunctionOutputsTransform. For dynamo's
    # native invoke_subgraph/repeated_subgraph path, keep original output names.
    if not use_dynamo:
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

    # Work on an instance-local copy and restore it in cleanup. Without this,
    # failed or repeated subfunction exports can leak transforms into later exports.
    original_transforms = list(qeff_model._onnx_transforms)
    qeff_model._onnx_transforms = list(original_transforms)

    # Add subfunction-specific ONNX transforms based on export path
    if use_dynamo:
        # Dynamo path: use invoke_subgraph/repeated_subgraph native transforms
        if PreserveNestedCacheRetainedStateTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(PreserveNestedCacheRetainedStateTransform)
        if RenameRepeatedSubgraphTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(RenameRepeatedSubgraphTransform)
    else:
        # TorchScript path: use export-modules-as-functions transforms
        if RenameFunctionOutputsTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(RenameFunctionOutputsTransform)
        if CustomOpTransform not in qeff_model._onnx_transforms:
            qeff_model._onnx_transforms.append(CustomOpTransform)

    # TODO: Handle this in the modelling class QEFFTransformersBase, remove from here.
    decoder_layer_classes = get_decoder_layer_classes_for_export(qeff_model.model)
    if decoder_layer_classes:
        if use_dynamo:
            # Pass target classnames to RenameRepeatedSubgraphTransform via onnx_transform_kwargs
            resolved_classnames = target_classnames or sorted(cls.__name__ for cls in decoder_layer_classes)
            qeff_model._subfunction_target_classnames = resolved_classnames
            onnx_transform_kwargs = dict(kwargs.get("onnx_transform_kwargs") or {})
            onnx_transform_kwargs["target_classnames"] = resolved_classnames
            kwargs["onnx_transform_kwargs"] = onnx_transform_kwargs
        else:
            # TorchScript path: pass class objects for export_modules_as_functions
            kwargs["export_modules_as_functions"] = decoder_layer_classes

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
