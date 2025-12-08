# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from contextlib import nullcontext
from typing import ContextManager

import torch


def get_op_verifier_ctx(
    use_op_by_op_verifier: bool,
    device_type: str,
    dump_dir: str,
    step: int,
    ref_device: str = "cpu",
    ref_dtype: torch.dtype = torch.float32,
    atol: float = 1e-1,
    rtol: float = 1e-5,
    use_ref_output_on_mismatch: bool = True,
) -> ContextManager:
    """Get the op-by-op verifier context manager when op-by-op verification is
    enabled. It helps in debuging operator related issues by matching the
    operator execution on qaic v/s cpu. This is meant only for qaic backend.

    Args:
        use_op_by_op_verifier (bool): Boolean flag to enable op-by-op verifier.
        device_type (str): Device on which the model is being executed.
        dump_dir (str): Directory to dump the op-by-op verification results.
        step (int): Step number for which the op-by-op verification is to be performed.
        ref_device (str, optional): Device to use as reference for verification.
            Defaults to "cpu".
        ref_dtype (torch.dtype, optional): Data type to use as reference
            datatype for verification. Defaults to torch.float32.
        atol (float, optional): Absolute tolerance to match the results. Defaults to 1e-1.
        rtol (float, optional): Relative tolerance to match the results. Defaults to 1e-5.
        use_ref_output_on_mismatch (bool, optional): If an operator has a
            mismatch with respect to the reference device, use the reference
            device outputs and continue rest of the verification. Defaults to True.

    Returns:
        ContextManager: Instance of context manager used to verify the operators.
    """
    if (not use_op_by_op_verifier) or ("qaic" in device_type):
        return nullcontext()

    # Lazily imported qaic_debug when it is actually needed.
    import torch_qaic.debug as qaic_debug

    filter_config = qaic_debug.DispatchFilterConfig.default(device_type)
    dump_dir = dump_dir + "/mismatches/step_" + str(step)
    return qaic_debug.OpByOpVerifierMode(
        ref_device=ref_device,
        ref_dtype=ref_dtype,
        atol=atol,
        rtol=rtol,
        use_ref_output_on_mismatch=use_ref_output_on_mismatch,
        filter_config=filter_config,
        dump_root_dir=dump_dir,
    )


def init_qaic_profiling(use_profiler: bool, device_type: str) -> None:
    """Initialize the qaic profiling tool. Note: The profiler is only works
    for qaic backend.

    Args:
        use_profiler (bool): Boolean flag to enable profiler.
        device_type (str): Device on which the model is being executed.
    """
    if (use_profiler) and ("qaic" in device_type):
        # Lazily imported qaic's qaic_profile when it is actually needed.
        import torch_qaic.profile as qaic_profile

        qaic_profile.start_profiling(device_type, 1)


def stop_qaic_profiling(use_profiler: bool, device_type: str) -> None:
    """Stop the qaic profiling tool. Note: The profiler is only works
    for qaic backend.

    Args:
        use_profiler (bool): Boolean flag to enable profiler.
        device_type (str): Device on which the model is being executed.
    """
    if (use_profiler) and ("qaic" in device_type):
        # Lazily imported qaic's qaic_profile when it is actually needed.
        import torch_qaic.profile as qaic_profile

        qaic_profile.stop_profiling(device_type)
