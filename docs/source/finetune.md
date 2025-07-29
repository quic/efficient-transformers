# Finetune Infra

This repository provides the infrastructure for finetuning models using different hardware accelerators such as QAic.
Same CLI can be used to run finetuning on GPU by changing the value of device flag (for finetuning on GPU, install torch specific to CUDA).

## Installation

Same as QEfficient along with QAIC PyTorch Eager mode.

For QEfficient Library : https://github.com/quic/efficient-transformers

For torch_qaic, assuming QEfficient is already installed,
```bash
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl
```
If qeff-env inside docker is used then torch_qaic and accelerate packages are already installed.

## Finetuning

Export the ENV variables to download and enable private datasets
```bash
export HF_DATASETS_TRUST_REMOTE_CODE=True
```

Export the ENV variables to get the device and HW traces and debugging logs
```bash
export QAIC_DEVICE_LOG_LEVEL=0 # For Device level logs
export QAIC_DEBUG=1 # To understand the CPU fallback ops
```

## Dataset Details

To download the Alpaca dataset, visit this [link](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json). Download the dataset and place it under the **dataset** directory. Make sure to update the training configuration accordingly.
```bash
wget -c https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json -P dataset/
```

To download the grammar dataset, visit this [link](https://github.com/meta-llama/llama-cookbook/blob/main/src/llama_cookbook/datasets/grammar_dataset/grammar_dataset_process.ipynb). Download the dataset and place it under the **datasets_grammar** directory. Make sure to update the training configuration accordingly.


## Usage

### Single SOC finetuning on QAIC

```python
python -m QEfficient.cloud.finetune --device qaic:0 --model_name "meta-llama/Llama-3.2-1B"
```
You can also configure various training parameters. Below is an example command line
```python
python -m QEfficient.cloud.finetune --device qaic:0 --use-peft --output_dir ./meta-sam --num_epochs 2 --context_length 256 
```

For more details on the usage of the training parameters, use the below command:
```python
python -m QEfficient.cloud.finetune -h
```

### Distributed training(DDP) on QAIC

```python
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 -m QEfficient.cloud.finetune --device qaic --enable_ddp  --num_epochs 2  --model_name "meta-llama/Llama-3.2-1B"
```
**nproc-per-node is number of workers(QAIC devices) running locally.

## Visualization

Tensorboard logs are generated inside runs/ directory with date and time stamp.
to visualise the data,

```python
tensorboard --logdir runs/<file> --bind_all
```

## Some features/functionalities of fine-tuning stack:
    1) Gradient accumulation: By default, gradient accumulation happens for 4 steps. To update this value, command line argument gradient_accumulation_steps has to be passed. (Example: '--gradient_accumulation_steps 8')
    2) Gradient Checkpointing: By default, gradient checkpointing is disabled. To enable it, command line argument gradient_accumulation_steps has to be passed.


### 🔧 Steps to Fine-Tune with a Custom Dataset

1. **Launching Fine-Tuning with a Custom Dataset**  
   Use the following command-line arguments to begin fine-tuning:
   ```
   --dataset custom_dataset --dataset_config data_config.json
   ```
   The `data_config.json` file contains essential parameters used during dataset preprocessing.

2. **Specifying the Preprocessing Function**  
   - In `data_config.json`, include a `"preproc_file"` key to define the path to your preprocessing Python file.
   - To specify a custom function within that file, use the format `"filename.py:function_name"`.  
     _Example:_  
     ```json
     "preproc_file": "disc_preproc.py:get_preprocessed_disc"
     ```
   - Your preprocessing function must follow this structure:
     ```python
     def get_custom_dataset(dataset_config, tokenizer, split, context_length=None):
         def apply_prompt_template():
             # Apply prompt formatting to each datapoint

         def tokenize():
             # Tokenize the formatted datapoint

         # Apply functions to dataset using map
         dataset = dataset.map(apply_prompt_template, ...)
         dataset = dataset.map(tokenize, ...)
         
         return dataset
     ```

3. **Custom Collate Function for Batching**  
   - When using a batch size greater than 1, you may override the default collate behavior by including a `"collate_file"` key in `data_config.json`.
   - Use the same `"file.py:function"` format. If omitted, the default Hugging Face `DataCollatorForSeq2Seq` is used, which pads sequences to the longest length in the batch.
   - A custom collate function must have the following signature:
     ```python
     def get_data_collator(tokenizer):
         # Define and return a custom collate_fn here
     ```

4. **Passing Additional Configuration Parameters**  
   You can add custom arguments in `data_config.json`, which will be accessible via the `dataset_config` argument inside your `get_custom_dataset()` function.

5. **Example `data_config.json` File**
   ```json
   {
     "train_split": "train",
     "test_split": "test",
     "test_split_ratio": 0.15,
     "preproc_file": "disc_preprocd.py:get_preprocessed_disc",
     "collate_file": "disc_preprocd.py:get_collate_fn_disc",
     "disc_style": "sarcasm_more"
   }
   ```

6. **Implementing Custom Preprocessing Logic**  
   Within your dataset loader function, define `apply_prompt_template()` to manipulate raw data into desired prompt format, and `tokenize()` to convert it into token IDs using the tokenizer.

7. **Reference for Dataset Utilities**  
   You can refer to existing implementations in the [dataset directory of this repository](https://github.com/quic/efficient-transformers/tree/main/QEfficient/finetune/dataset).

---
