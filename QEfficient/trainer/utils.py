# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging


def setup_logging():
    logging.basicConfig(level=logging.DEBUG)
    return logging.getLogger(__name__)
