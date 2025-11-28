# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank of the current process, else 0."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Return the local rank of the current process on its node, else 0."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_node_local_rank()


def get_world_size() -> int:
    """Get the total number of processes in distributed training."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    return get_rank() == 0
