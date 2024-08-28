# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import random

import pytest

from QEfficient.utils.cache import to_hashable


def get_random_string(length: int) -> str:
    return "".join([chr(random.randint(0x20, 0x7E)) for _ in range(length)])


def test_to_hashable_dict():
    dct = {get_random_string(i): i for i in range(5)}
    dct = dict(sorted(dct.items()))
    hash1 = to_hashable(dct)

    dct = dict(reversed(dct.items()))
    hash2 = to_hashable(dct)

    assert hash1 == hash2


def test_to_hashable_set():
    assert to_hashable(set(range(4))) == to_hashable(set(range(4 - 1, -1, -1)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_to_hashable_float_nan(value):
    with pytest.raises(ValueError):
        to_hashable(float("nan"))
