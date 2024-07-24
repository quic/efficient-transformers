# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import sys
from dataclasses import dataclass, field

from transformers import HfArgumentParser

from .config import QEffConfig, QEffTrainingArguments
from .models import QEfficient


@dataclass
class RefineArguments:
    model_id: str = field(default="facebook/opt-125m", metadata={"help": "Pre-trained model to fine-tune"})
    dataset_name: str = field(
        default="xiyuez/red-dot-design-award-product-description", metadata={"help": "Dataset to fine-tune on"}
    )
    train_frac: float = field(default=0.85, metadata={"help": "Fraction of training dataset to split"})
    lora_r: int = field(default=8, metadata={"help": "LoRA r parameter"})
    max_ctx_len: int = field(default=512, metadata={"help": "Maximum context length for tokenization"})


def finetune_model(model_id, dataset_name, train_frac, lora_r, max_ctx_len, num_train_epochs, batch_size, output_dir):
    config = QEffConfig(
        model_id=model_id,
        dataset_name=dataset_name,
        train_frac=train_frac,
        lora_config={"r": lora_r},
        max_ctx_len=max_ctx_len,
    )

    training_args = QEffTrainingArguments(
        output_dir=output_dir, num_train_epochs=num_train_epochs, per_device_train_batch_size=batch_size
    )

    qefficient = QEfficient(config)
    refined_model, tokenizer = qefficient.refine(training_args)

    return refined_model, tokenizer


def main():
    parser = HfArgumentParser((RefineArguments, QEffTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        args, training_args = parser.parse_args_into_dataclasses()

    refined_model, tokenizer = finetune_model(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        train_frac=args.train_frac,
        lora_r=args.lora_r,
        max_ctx_len=args.max_ctx_len,
        num_train_epochs=training_args.num_train_epochs,
        batch_size=training_args.per_device_train_batch_size,
        output_dir=training_args.output_dir,
    )

    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()

"""
Functional Apis()

from QEfficient import QEfficient, QEffConfig, QEffTrainingArguments

def finetune_model(model_id, dataset_name, train_frac, lora_r, max_ctx_len, num_train_epochs, batch_size, output_dir):
    config = QEffConfig(
        model_id=model_id,
        dataset_name=dataset_name,
        train_frac=train_frac,
        lora_config={"r": lora_r},
        max_ctx_len=max_ctx_len
    )
    
    training_args = QEffTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size
    )
    
    qefficient = QEfficient(config)
    refined_model, tokenizer = qefficient.refine(training_args)
    
    return refined_model, tokenizer

# Usage
model, tokenizer = finetune_model(
    model_id="facebook/opt-125m",
    dataset_name="xiyuez/red-dot-design-award-product-description",
    train_frac=0.85,
    lora_r=8,
    max_ctx_len=512,
    num_train_epochs=3,
    batch_size=8,
    output_dir="./output"
)
"""
