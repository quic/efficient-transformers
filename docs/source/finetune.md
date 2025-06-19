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

## Fine-Tuning on custom dataset

To run fine tuning for any user specific dataset, prepare the dataset using the following steps:

    1. Create a directory named 'dataset' inside efficient-transformers.
    2. Inside this directory, create a file named 'custom_dataset.py'.
    3. Inside the newly created efficient-transformers/dataset/custom_dataset.py, define a function named 'get_custom_dataset'. 
    4. get_custom_dataset() should have following 4 parameters:  dataset_config, tokenizer, split, context_length.  
    5. Inside get_custom_dataset(), user needs to apply prompt and tokenize the dataset accordingly. Please refer the below template on how to define get_custom_dataset().
    6. For examples, please refer python files present in [dataset](https://github.com/quic/efficient-transformers/tree/main/QEfficient/finetune/dataset). In case of Samsum dataset, get_preprocessed_samsum() of efficient-transformers/QEfficient/finetune/dataset/samsum_dataset.py is called. 
    7. In [dataset_config.py](https://github.com/quic/efficient-transformers/blob/main/QEfficient/finetune/configs/dataset_config.py), for custom_dataset class, pass the appropriate value for train_split and test_split. As an alternative, these values can be passed as command line arguments as well with the finetune command. For example "--train_split train".
    8. While running fine tuning, pass argument "-–dataset custom_dataset" to finetune on custom dataset.   

Template for get_custom_dataset() to be defined inside efficient-transformers/dataset/custom_dataset.py is as follows:

```python
def get_custom_dataset(dataset_config, tokenizer, split, context_length=None):

    # load dataset
    # based on split, retrieve only the specific portion of the dataset (train or eval) either here or at the last
    
    def apply_prompt_template():
        # transform the passed datapoint by applying the prompt on it 
    
    def tokenize():
        # tokenize the passed datapoint
    
    # define the prompt
    # call apply_prompt_template() for each data point:
    # dataset = dataset.map(apply_prompt_template ,<other args>)
    # call tokenize() for each data point:
    # dataset = dataset.map(tokenize, <other args>)
    
    return dataset
```
