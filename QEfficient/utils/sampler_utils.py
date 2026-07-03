# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Set

import torch

from QEfficient.utils import constants
from QEfficient.utils.constants import Constants
from QEfficient.utils.logging_utils import logger


def validate_sampler_inputs(
    session_inputs: Set[str], include_sampler: Optional[bool] = None, include_guided_decoding: Optional[bool] = None
) -> bool:
    """
    Validates whether the `QAICInferenceSession` inputs match inputs required for on-device sampling.

    Mandatory Args:
        session_inputs (set[str]): Set of input names from `QAICInferenceSession`.

    Optional Args:
        include_sampler (bool, default=None): Whether the user explicitly requested sampler support.

    Returns:
        True if sampler is supported, False otherwise.

    Raises:
        ValueError if partial support is detected or if user intent conflicts with QPC capabilities.
    """

    sampler_inputs = Constants.SAMPLER_INPUTS | ({"token_bitmasks"} if include_guided_decoding else set())
    count = len(sampler_inputs & session_inputs)

    session_includes_sampler = True
    if count == 0:
        session_includes_sampler = False
    elif count < len(sampler_inputs):
        session_includes_sampler = False
        raise ValueError(
            f"The provided QPC does not have the required number of inputs to run sampling "
            f"on the QAIC device (only {count}/{len(sampler_inputs)} inputs provided). Partial "
            "sampling support is not available. Please check the QPC and try again."
        )

    # Post-validation consistency checks
    if include_sampler and not session_includes_sampler:
        logger.warning(
            "User entered `include_sampler`=True. But the provided QPC is not compiled "
            "to run sampling on the QAIC device. Falling back to the PyTorch backend."
        )
    elif (include_sampler is None or not include_sampler) and session_includes_sampler:
        raise ValueError(
            "The provided QPC is compiled to run sampling on the QAIC device. "
            "But the user did not enter `include_sampler`=True. Please make sure the input "
            "is specified correctly."
        )

    return session_includes_sampler


def get_sampling_inputs_and_outputs(
    example_inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    continuous_batching: bool,
    vocab_size: int,
    qaic_config: Dict,
    dynamic_shapes: Optional[Dict] = None,
):
    """
    Updates the example inputs, output names, and dynamic axes to include
    parameters relevant for on-device sampling during ONNX export.

    Parameters
    ----------
    example_inputs : Dict[str, torch.Tensor]
        Current dictionary of example inputs.
    output_names : List[str]
        Current list of output names.
    dynamic_axes : Dict[str, Dict[int, str]]
        Current dictionary of dynamic axes configurations.
    continuous_batching : bool
        Whether this model will be used for continuous batching in the future.
    vocab_size: int
        Vocabulary size for this model.
    qaic_config : Dict
        QAIC config dictionary.
    dynamic_shapes : Dict, optional
        When provided (dynamo export path), updated in-place with torch.export.Dim
        entries for each new sampler input, mirroring the dynamic_axes additions.

    Returns
    -------
    Tuple[Dict[str, torch.Tensor], List[str], Dict[str, Dict[int, str]], Optional[Dict]]
        Updated example inputs, output names, dynamic axes, and dynamic shapes
        (None if dynamic_shapes was not provided) including sampling-related parameters.
    """
    bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
    fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS
    seq_len: int = example_inputs["input_ids"].shape[-1]

    # Build a registry of Dim objects so identical dim names share the same Dim instance.
    # Seed the registry from any Dim objects already present in dynamic_shapes so that
    # sampler axes stay consistent with the rest of the graph.
    if dynamic_shapes is not None:
        from torch.export import Dim

        dim_registry: Dict[str, object] = {}
        for axes_map in dynamic_shapes.values():
            if not isinstance(axes_map, dict):
                continue
            for dim_obj in axes_map.values():
                if hasattr(dim_obj, "__name__"):
                    dim_registry.setdefault(dim_obj.__name__, dim_obj)

        def _get_dim(name: str, **kwargs) -> object:
            if name not in dim_registry:
                dim_registry[name] = Dim(name, **kwargs)
            return dim_registry[name]

    def _add_dynamic_shape(input_name: str, axes: Dict[int, str]) -> None:
        if dynamic_shapes is None:
            return
        entry = {}
        for axis_idx, dim_name in axes.items():
            if "seq_len" in dim_name:
                entry[axis_idx] = _get_dim(dim_name, min=1, max=513)
            elif dim_name == "batch_size":
                entry[axis_idx] = _get_dim(dim_name, min=1, max=64)
            else:
                entry[axis_idx] = _get_dim(dim_name, min=1, max=4096)
        dynamic_shapes[input_name] = entry

    example_inputs["last_accepted_output_tokens"] = torch.zeros((bs, seq_len), dtype=torch.int64)
    dynamic_axes["last_accepted_output_tokens"] = {0: "batch_size", 1: "seq_len"}
    _add_dynamic_shape("last_accepted_output_tokens", {0: "batch_size", 1: "seq_len"})

    penalty_batch_dim = "full_batch_size" if continuous_batching else "batch_size"

    example_inputs["past_repetition_penalty_buffer"] = torch.zeros(
        (fbs if continuous_batching else bs, vocab_size), dtype=torch.bool
    )
    dynamic_axes["past_repetition_penalty_buffer"] = {0: penalty_batch_dim}
    _add_dynamic_shape("past_repetition_penalty_buffer", {0: penalty_batch_dim})
    output_names.append("past_repetition_penalty_buffer_RetainedState")

    example_inputs["repetition_penalties"] = (
        torch.ones((bs, 1), dtype=torch.float) * constants.ONNX_EXPORT_EXAMPLE_REPETITION_PENALTIES
    )
    dynamic_axes["repetition_penalties"] = {0: "batch_size"}
    _add_dynamic_shape("repetition_penalties", {0: "batch_size"})

    example_inputs["past_presence_penalty_buffer"] = torch.zeros(
        (fbs if continuous_batching else bs, vocab_size), dtype=torch.bool
    )
    dynamic_axes["past_presence_penalty_buffer"] = {0: penalty_batch_dim}
    _add_dynamic_shape("past_presence_penalty_buffer", {0: penalty_batch_dim})
    output_names.append("past_presence_penalty_buffer_RetainedState")

    example_inputs["presence_penalties"] = (
        torch.zeros((bs, 1), dtype=torch.float) + constants.ONNX_EXPORT_EXAMPLE_PRESENCE_PENALTIES
    )
    dynamic_axes["presence_penalties"] = {0: "batch_size"}
    _add_dynamic_shape("presence_penalties", {0: "batch_size"})

    example_inputs["temperatures"] = torch.ones((bs, 1), dtype=torch.float) * constants.ONNX_EXPORT_EXAMPLE_TEMPERATURES
    dynamic_axes["temperatures"] = {0: "batch_size"}
    _add_dynamic_shape("temperatures", {0: "batch_size"})

    max_top_k_ids = qaic_config.get("max_top_k_ids", constants.ONNX_EXPORT_EXAMPLE_MAX_TOP_K_IDS)
    example_inputs["top_ks"] = torch.randint(1, max_top_k_ids, size=(bs, 1)).to(torch.int32)
    dynamic_axes["top_ks"] = {0: "batch_size"}
    _add_dynamic_shape("top_ks", {0: "batch_size"})

    example_inputs["top_ps"] = torch.ones((bs, 1), dtype=torch.float) * constants.ONNX_EXPORT_EXAMPLE_TOP_PS
    dynamic_axes["top_ps"] = {0: "batch_size"}
    _add_dynamic_shape("top_ps", {0: "batch_size"})

    example_inputs["min_ps"] = torch.ones((bs, 1), dtype=torch.float) * constants.ONNX_EXPORT_EXAMPLE_MIN_PS
    dynamic_axes["min_ps"] = {0: "batch_size"}
    _add_dynamic_shape("min_ps", {0: "batch_size"})

    example_inputs["random_numbers"] = torch.rand((bs, max_top_k_ids), dtype=torch.float)
    dynamic_axes["random_numbers"] = {0: "batch_size"}
    _add_dynamic_shape("random_numbers", {0: "batch_size"})

    if qaic_config.get("include_guided_decoding", False):
        example_inputs["token_bitmasks"] = torch.zeros((bs, vocab_size), dtype=torch.bool)
        dynamic_axes["token_bitmasks"] = {0: "batch_size"}
        _add_dynamic_shape("token_bitmasks", {0: "batch_size"})

    return example_inputs, output_names, dynamic_axes, dynamic_shapes
