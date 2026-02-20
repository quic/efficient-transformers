# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import torch.nn as nn

from QEfficient.base.pytorch_transforms import ProxyModuleMappingTransform
from QEfficient.proxy import QeffProxyEmbedding, QeffProxyLinear


class QeffProxyModuleTransform(ProxyModuleMappingTransform):
    """
    This transform is used to replace the original modules with QEfficient modules.
    """

    _module_mapping = {
        nn.Embedding: QeffProxyEmbedding,
        nn.Linear: QeffProxyLinear,
    }
