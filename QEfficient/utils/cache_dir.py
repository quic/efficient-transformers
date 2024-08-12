# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os

QEFF_HOME = None
if "QEFF_HOME" in os.environ:
    QEFF_HOME = os.environ["QEFF_HOME"]
elif "XDG_CACHE_HOME" in os.environ:
    QEFF_HOME = os.path.join(os.environ["XDG_CACHE_HOME"], "qeff_models")
else:
    QEFF_HOME = os.path.expanduser("~/.cache/qeff_models")
