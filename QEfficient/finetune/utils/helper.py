# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from contextlib import nullcontext
from enum import Enum
from typing import Any, ContextManager, List

import torch


class Batching_Strategy(str, Enum):
    PADDING = "padding"
    PACKING = "packing"


class Device(str, Enum):
    QAIC = "qaic"
    CPU = "cpu"
    CUDA = "cuda"


class Peft_Method(str, Enum):
    LORA = "lora"


class Task_Mode(str, Enum):
    GENERATION = "generation"
    SEQ_CLASSIFICATION = "seq_classification"


def enum_names(enum_cls: Enum) -> List[str]:
    """Returns List of Enum members in string format.

    Args:
        enum_cls (Enum): Enum class reference.

    Returns:
        List[str]: List of Enum members in string format.
    """
    return [member.value for member in enum_cls]


def get_rank() -> int:
    """Get the current global rank of the process.

    In DDP, this should correspond to the 'RANK' environment variable set by torchrun.
    In non-DDP use case, returns 0.
    """
    return int(os.getenv("RANK", 0))


def get_local_rank() -> int:
    """Get the current local rank of the process.

    In DDP, this should correspond to the 'LOCAL_RANK' environment variable set by torchrun.
    In non-DDP use case, returns 0.
    """
    return int(os.getenv("LOCAL_RANK", 0))


def get_node_rank() -> int:
    """Get the node rank of the process.

    In DDP, this should correspond to the 'GROUP_RANK' environment variable set by torchrun.
    In non-DDP use case, returns 0.
    """
    return int(os.getenv("GROUP_RANK", 0))


def is_rank_zero() -> bool:
    """Checks whether the current process is in rank-0 in case of DDP. For
    non-DDP use case it will always return True.

    Returns:
        bool: Flag to inidicate whether the current process is in rank-0.
    """
    return get_rank() == 0


def get_world_size() -> int:
    """Get total multiprocesses invoked for DDP setting. For pure DDP use case,
    this will correlate with number of devices being used. For PP+DDP use case,
    this will give number of processes initiated (i.e. number of model replicas).
    In case of non-DDP use case, this will return 1.

    Returns:
        int: Number of DDP devices.
    """
    return int(os.getenv("WORLD_SIZE", 1))


def get_local_world_size() -> int:
    """Get total multiprocesses invoked for DDP setting for that node. For pure DDP use case,
    this will correlate with number of devices being used. For PP+DDP use case,
    this will give number of processes initiated (i.e. number of model replicas).
    In case of non-DDP use case, this will return 1.

    Returns:
        int: Number of DDP devices available on that node.
    """
    return int(os.getenv("LOCAL_WORLD_SIZE", 1))


def get_autocast_ctx(use_autocast: bool, device_type: str, dtype: torch.dtype = torch.float16) -> ContextManager:
    """Get the autocast context manager in case of AMP training. If use_autocast
    is False then nullcontext is returned.

    Args:
        use_autocast (bool): Boolean flag to indicate whether to use autocast.
        device_type (str): Device type to use for autocast.
        dtype (torch.dtype, optional): Autocast data type to be used. Defaults
            to torch.float16.

    Returns:
        ContextManager: Instance of context manager used to autocast.
    """
    return torch.autocast(device_type=device_type, dtype=dtype) if use_autocast else nullcontext()


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
    if (not use_op_by_op_verifier) or (device_type != Device.QAIC):
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


def get_grad_scaler(use_grad_scaler: bool, device_type: str) -> Any:
    """Get the grad scaler for AMP training.

    Args:
        use_grad_scaler (bool): Boolean flag to enable grad scaler.
        device_type (str): Device on which the model is being executed.

    Returns:
        Any: If device is qaic or cuda then the backend specific grad scaler is
            returned. Otherwise None is returned.
    """
    if not use_grad_scaler:
        return None

    if device_type == Device.QAIC:
        # Lazily imported qaic's GradScaler when it is actually needed.
        from torch.qaic.amp import GradScaler as QAicGradScaler

        return QAicGradScaler()
    elif (device_type == Device.CUDA) or (device_type == Device.CPU):
        # Lazily imported torch's GradScaler when it is actually needed.
        from torch.amp import GradScaler

        return GradScaler(device_type)
    else:
        return None


def init_qaic_profiling(use_profiler: bool, device_type: str) -> None:
    """Initialize the qaic profiling tool. Note: The profiler is only works
    for qaic backend.

    Args:
        use_profiler (bool): Boolean flag to enable profiler.
        device_type (str): Device on which the model is being executed.
    """
    if (use_profiler) and (device_type == Device.QAIC):
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
    if (use_profiler) and (device_type == Device.QAIC):
        # Lazily imported qaic's qaic_profile when it is actually needed.
        import torch_qaic.profile as qaic_profile

        qaic_profile.stop_profiling(device_type)


def save_to_json(
    output_filename,
    train_step_loss,
    train_epoch_loss,
    train_step_metric,
    train_epoch_metric,
    val_step_loss,
    val_epoch_loss,
    val_step_metric,
    val_epoch_metric,
):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_metric": train_step_metric,
        "train_epoch_metric": train_epoch_metric,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_metric": val_step_metric,
        "val_epoch_metric": val_epoch_metric,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
