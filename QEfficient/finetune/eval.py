# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import random
import warnings

import fire
import numpy as np
import torch
from configs.training import train_config as TRAIN_CONFIG
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config_utils import (
    generate_dataset_config,
    get_dataloader_kwargs,
    update_config,
)
from utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)
from utils.train_utils import evaluation, print_model_size

try:
    import torch_qaic  # noqa: F401

    device = "qaic:0"
except ImportError as e:
    print(f"Warning: {e}. Moving ahead without these qaic modules.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Suppress all warnings
warnings.filterwarnings("ignore")


def main(**kwargs):
    # update the configuration for the training process
    train_config = TRAIN_CONFIG()
    update_config(train_config, **kwargs)

    # Set the seeds for reproducibility
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Load the pre-trained model and setup its configuration
    # config = AutoConfig.from_pretrained(train_config.model_name)
    save_dir = "meta-llama-samsum/trained_weights/step_14000"

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
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config)

    # Get the dataset utils
    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_processer = tokenizer

    # Load and preprocess the dataset for training and validation
    dataset_val = get_preprocessed_dataset(
        dataset_processer, dataset_config, split="test", context_length=train_config.context_length
    )

    eval_dataloader = None
    custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
    if train_config.run_validation:
        # TODO: vbaddi enable packing later in entire infra.
        # if train_config.batching_strategy == "packing":
        #    dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, dataset_processer, "val")
        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
        if len(eval_dataloader) == 0:
            raise ValueError(
                f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)=})"
            )
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    model.to(device)
    _ = evaluation(model, train_config, eval_dataloader, None, tokenizer, device)


if __name__ == "__main__":
    fire.Fire(main)
