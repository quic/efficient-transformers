# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os


def is_rank_zero():
    return int(os.getenv("LOCAL_RANK", 0)) == 0


def print_rank_0(msg):
    if is_rank_zero():
        print(msg)


def get_num_ddp_devices():
    return int(os.getenv("WORLD_SIZE", 1))
