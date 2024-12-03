from functools import partial

from dataset.alpaca_dataset import (
    InstructionDataset as get_alpaca_dataset,
)
from dataset.custom_dataset import (
    get_custom_dataset,
    get_data_collator,
)
from dataset.grammer_dataset import (
    get_dataset as get_grammar_dataset,
)
from dataset.gsm8k_dataset import get_gsm8k_dataset
from dataset.samsum_dataset import (
    get_preprocessed_samsum as get_samsum_dataset,
)

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "gsm8k_dataset": get_gsm8k_dataset,
    "custom_dataset": get_custom_dataset,
}
DATALOADER_COLLATE_FUNC = {"custom_dataset": get_data_collator}
