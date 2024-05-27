# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch.nn as nn

from QEfficient.loader import QEFFAutoModel  # noqa: F401
from QEfficient.transformers.modeling_utils import transform as transform_hf


def transform(model: nn.Module, type="Transformers", form_factor="cloud"):
    """Low level apis in library
    model : instance of nn.Module
    type : Transformers | Diffusers, default : Transformers
    """
    if type == "Transformers":
        return transform_hf(model, form_factor)
    else:
        raise NotImplementedError
