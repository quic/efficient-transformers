# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os

TASK_TYPE = ["generation", "seq_classification"]
PEFT_METHOD = ["lora"]
DEVICE = ["qaic", "cpu", "cuda"]
BATCHING_STRATEGY = ["padding", "packing"]


def get_num_ddp_devices():
    return int(os.getenv("WORLD_SIZE", 1))
