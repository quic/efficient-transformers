# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import inspect


def filter_kwargs(func, kwargs):
    """
    Filter a dictionary of keyword arguments to only include the valid arguments of a function.

    Args:
        func: The function to check the arguments for.
        kwargs: The dictionary of keyword arguments to filter.

    Returns:
        A new dictionary containing only the valid keyword arguments.
    """
    valid_args = inspect.signature(func).parameters
    return {key: value for key, value in kwargs.items() if key in valid_args}
