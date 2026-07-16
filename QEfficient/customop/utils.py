# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch


def select_interface(eager_impl, custom_op_impl):
    use_custom_op = torch._dynamo.is_compiling()
    return custom_op_impl if use_custom_op else eager_impl
