# Finetune Infra

This repository provides the infrastructure for finetuning models using different hardware accelerators such as QAIC.
Same CLI can be used to run Finetuning on gpu by setting the device flag.(for finetuning on gpu, install torch specific to cuda)

## Installation

Same as QEfficient along with QAIC PyTorch Eager mode.

For QEfficient Library : https://github.com/quic/efficient-transformers

For torch_qaic, assuming QEfficient is already installed,
```bash
pip install /opt/qti-aic/integrations/torch_qaic/py310/torch_qaic-0.1.0-cp310-cp310-linux_x86_64.whl
```

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

To download the grammar dataset, visit this [link](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset_process.ipynb). Download the dataset and place it under the **datasets_grammar** directory. Make sure to update the training configuration accordingly.


## Usage

### Single SOC finetuning on QAIC

```python
python -m QEfficient.cloud.finetune --device qaic:0 --model_name "meta-llama/Llama-3.2-1B"
```
Also, you can configure various training parameters, for more details, checkout: QEfficient/finetune/configs/training.py, Below is example command line
```python
python -m QEfficient.cloud.finetune --device qaic:0 --use-peft --output_dir ./meta-sam --num_epochs 2 --context_length 256 
```

### Distributed training(DDP) on QAIC

```python
QAIC_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 -m QEfficient.cloud.finetune --device qaic --enable_ddp --dist_backend qccl --num_epochs 2  --model_name "meta-llama/Llama-3.2-1B"
```
**nproc-per-node is number of workers(QAIC devices) running locally.

## Visualization

Tensorboard logs are generated inside runs/ directory with date and time stamp.
to visualise the data,

```python
tensorboard --logdir runs/<file> --bind_all
```