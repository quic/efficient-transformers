# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.utils.device_utils import is_multi_qranium_setup_available, is_qpc_size_gt_32gb


def test_is_qpc_size_gt_32gb_mxfp6():
    params = 2**40
    result = is_qpc_size_gt_32gb(params=params, mxfp6=True)
    assert result


def test_is_qpc_size_gt_32gb_other():
    params = 2**30
    result = is_qpc_size_gt_32gb(params=params, mxfp6=False)
    assert result is not True


def test_is_multi_qranium_setup_available():
    _ = is_multi_qranium_setup_available()
    assert True
