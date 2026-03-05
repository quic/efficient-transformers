# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from QEfficient.cloud.finetune_experimental import main
from QEfficient.finetune.experimental.examples.custom_dataset import (
    SeqCompletionDataset,  # noqa: F401 - registers CustomDataset
)

if __name__ == "__main__":
    main()
