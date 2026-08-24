# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Optional

QEFF_HOME: Path = None
if "QEFF_HOME" in os.environ:
    QEFF_HOME = Path(os.environ["QEFF_HOME"])
elif "XDG_CACHE_HOME" in os.environ:
    QEFF_HOME = Path(os.environ["XDG_CACHE_HOME"]) / "qeff_models"
else:
    QEFF_HOME = Path("~/.cache/qeff_models").expanduser()

# Optional override for where weight-free prepared/converted checkpoints are
# saved. Unset by default — QEfficient/exporter/weight_free/export.py falls
# back to saving next to the source checkpoint under the HF cache when None.
QEFF_CHECKPOINT_HOME: Optional[Path] = (
    Path(os.environ["QEFF_CHECKPOINT_HOME"]) if "QEFF_CHECKPOINT_HOME" in os.environ else None
)
