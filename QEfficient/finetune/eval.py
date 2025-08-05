# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import os
import random
import warnings

import fire
import numpy as np
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config_utils import generate_dataset_config, update_config
from utils.dataset_utils import get_dataloader
from utils.train_utils import evaluation, print_model_size

from QEfficient.finetune.configs.training import TrainConfig
from QEfficient.finetune.utils.logging_utils import logger

try:
    import torch_qaic  # noqa: F401

    device = "qaic:0"
except ImportError as e:
    logger.log_rank_zero(
        f"Unable to import 'torch_qaic' package due to exception: {e}. Moving ahead without the torch_qaic extension.",
        logging.WARNING,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Suppress all warnings
warnings.filterwarnings("ignore")


def main(**kwargs):
    # update the configuration for the training process
    train_config = TrainConfig()
    update_config(train_config, **kwargs)
    dataset_config = generate_dataset_config(train_config.dataset)
    update_config(dataset_config, **kwargs)

    # Set the seeds for reproducibility
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Load the pre-trained model from latest checkpoint
    trained_weights_path = os.path.join(train_config.output_dir, "trained_weights")
    epoch_max_index = max([int(name.split("_")[-1]) for name in os.listdir(trained_weights_path)])
    epochs_path = os.path.join(trained_weights_path, "epoch_" + str(epoch_max_index))
    step_max_index = max([int(name.split("_")[-1]) for name in os.listdir(epochs_path)])
    save_dir = os.path.join(epochs_path, "step_" + str(step_max_index))

    # Load PEFT model on CPU
    model_peft = AutoPeftModelForCausalLM.from_pretrained(save_dir)

    # Merge LoRA and base model and save
    merged_model = model_peft.merge_and_unload()
    merged_model.save_pretrained(train_config.output_dir, safe_serialization=True)
    model_id = train_config.output_dir

    # Load Model with PEFT adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_cache=False,
        attn_implementation="sdpa",
        torch_dtype=torch.float16 if torch.cuda.is_available() or device == "qaic:0" else None,
    )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.log_rank_zero("Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model)

    if train_config.run_validation:
        eval_dataloader = get_dataloader(tokenizer, dataset_config, train_config, split="test")
        if len(eval_dataloader) == 0:
            logger.raise_error(
                f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)=})",
                ValueError,
            )
        else:
            logger.log_rank_zero(f"Number of Validation Set Batches loaded = {len(eval_dataloader)}")

    model.to(device)
    _ = evaluation(model, train_config, eval_dataloader, None, tokenizer, device)


if __name__ == "__main__":
    fire.Fire(main)
