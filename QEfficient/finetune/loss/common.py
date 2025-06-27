# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        pass
