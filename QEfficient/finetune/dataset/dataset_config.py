# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from functools import partial

from QEfficient.finetune.dataset.alpaca_dataset import (
    InstructionDataset as get_alpaca_dataset,
)
from QEfficient.finetune.dataset.custom_dataset import (
    get_custom_dataset,
    get_data_collator,
)
from QEfficient.finetune.dataset.grammar_dataset import (
    get_dataset as get_grammar_dataset,
)
from QEfficient.finetune.dataset.gsm8k_dataset import get_gsm8k_dataset
from QEfficient.finetune.dataset.imdb_dataset import (
    get_preprocessed_imdb as get_imdb_dataset,
)

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "gsm8k_dataset": get_gsm8k_dataset,
    "custom_dataset": get_custom_dataset,
    "imdb_dataset": get_imdb_dataset,
}
DATALOADER_COLLATE_FUNC = {
    "custom_dataset": get_data_collator,
}
