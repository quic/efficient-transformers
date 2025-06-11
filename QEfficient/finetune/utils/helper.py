# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os


def print_rank_0(msg):
    if os.getenv("LOCAL_RANK", None) in [None, 0]:
        print(msg)
