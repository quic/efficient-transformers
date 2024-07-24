# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split


class QEffDataManager:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None

    def prepare_dataset(self):
        rd_df = load_dataset(self.config.dataset_name)["train"].to_pandas()
        instruction = (
            "### Instruction:\n"
            "Create a detailed description for the following product: {product}, belonging to category: {category}\n"
            "### Response:\n"
            "{description}"
        ) + self.tokenizer.eos_token

        ds = [
            self.tokenizer(
                instruction.format(**row),
                text_target=instruction.format(**row),
                return_tensors="pt",
                padding="max_length",
                max_length=self.config.max_ctx_len,
                truncation=True,
            )
            for _, row in rd_df.iterrows()
        ]
        random_generator = torch.Generator().manual_seed(37)
        train_ds, eval_ds = random_split(ds, [self.config.train_frac, 1 - self.config.train_frac], random_generator)
        return train_ds, eval_ds

    def get_dataloader(self, dataset: Dataset, batch_size: int):
        return DataLoader(dataset, batch_size=batch_size, collate_fn=self.collate)

    @staticmethod
    def collate(batch):
        out_batch = {}
        for key in batch[0]:
            out_batch[key] = torch.cat([x[key] for x in batch], 0)
        return out_batch


"""
Custom DataPipeline;

import pandas as pd
from torch.utils.data import Dataset

class CustomProductReviewDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.iloc[idx]['review']
        rating = self.data.iloc[idx]['rating']
        
        # Construct prompt
        prompt = f"Rate this product review on a scale of 1-5:\n\n{review}\n\nRating:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        
        # Prepare target (rating converted to text)
        target = f" {rating}"
        target_encoding = self.tokenizer(target, return_tensors="pt", max_length=10, padding="max_length", truncation=True)
        
        # Combine input and target for training
        combined_input_ids = torch.cat([inputs.input_ids, target_encoding.input_ids], dim=1)
        combined_attention_mask = torch.cat([inputs.attention_mask, target_encoding.attention_mask], dim=1)
        
        return {
            "input_ids": combined_input_ids.squeeze(),
            "attention_mask": combined_attention_mask.squeeze(),
            "labels": combined_input_ids.squeeze()  # For causal language modeling, labels are the same as input_ids
        }

    @classmethod
    def load_data(cls, file_path, tokenizer, max_length):
        return cls(file_path, tokenizer, max_length)

    @classmethod
    def preprocess(cls, raw_data):
        # Any additional preprocessing can be done here
        return raw_data

    @classmethod
    def split(cls, dataset, train_ratio=0.8):
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        return torch.utils.data.random_split(dataset, [train_size, val_size])


        
from torch.utils.data import DataLoader

class QEffDataManager:
    def __init__(self, config, custom_dataset_class=None):
        self.config = config
        self.tokenizer = None
        self.custom_dataset_class = custom_dataset_class or CustomProductReviewDataset

    def prepare_dataset(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        
        # Load and preprocess the data
        raw_data = self.custom_dataset_class.load_data(self.config.dataset_path, self.tokenizer, self.config.max_ctx_len)
        processed_data = self.custom_dataset_class.preprocess(raw_data)
        
        # Split the data
        train_data, val_data = self.custom_dataset_class.split(processed_data, self.config.train_frac)
        
        return train_data, val_data

    def get_dataloader(self, dataset, batch_size, shuffle=True):
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate)

    @staticmethod
    def collate(batch):
        return {key: torch.stack([example[key] for example in batch]) for key in batch[0]}

class QEfficient:
    def __init__(self, config: QEffConfig, custom_dataset_class=None):
        self.config = config
        self.data_manager = QEffDataManager(config, custom_dataset_class)
        self.model_manager = QEffModelManager(config)
        self.trainer = None

    # rest of the class remains the same

# Example
from QEfficient import QEfficient, QEffConfig, QEffTrainingArguments

# Define your custom config
config = QEffConfig(
    model_id="gpt2",
    dataset_path="path/to/your/product_reviews.csv",
    train_frac=0.8,
    max_ctx_len=512
)

# Define your training arguments
training_args = QEffTrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
)

# Initialize QEfficient with your custom dataset
qefficient = QEfficient(config, custom_dataset_class=CustomProductReviewDataset)

# Refine the model
refined_model, tokenizer = qefficient.refine(training_args)

# Now you can use the refined model for inference
prompt = "Rate this product review on a scale of 1-5:\n\nThis product exceeded my expectations. It's durable, efficient, and user-friendly.\n\nRating:"
generated_text = qefficient.generate(prompt, max_length=100)
print(generated_text)
"""
