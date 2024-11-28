# Finetune Infra

This repository provides the infrastructure for finetuning models using different hardware accelerators such as QAIC, CUDA, and CPU.

## Installation

Same as QEfficient along with QAic Eager mode

## Finetuning
To finetune a model, run the following command:

Export the ENV variables to download and enable private datasets
```bash
export HF_DATASETS_TRUST_REMOTE_CODE=True
```

Export the ENV variables to get the device and HW traces and debugging logs
```bash
export QAIC_DELAY_SEM_WAIT_AT_COPY=1 # For HW profile traces
export QAIC_DEVICE_LOG_LEVEL=0 # For Device level logs
export QAIC_DEBUG=1 # To understand the CPU fallback ops
```

```python
python -m QEfficient.cloud.finetune --device qaic:0
```
You can switch between different hardwares by replacing qaic with cuda or cpu. But remember to install torch specific to CUDA to run on GPU

Also, you can configure various training parameters, for more details, checkout: configs/training.py, Below is example command line

```python
python -m QEfficient.cloud.finetune --device qaic:0 --use-peft --output_dir ./meta-sam --num_epochs 2 --context_length 256 
```

## Dataset Details
To download the Alpaca dataset, visit this [link](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json). Download the dataset and place it under the **dataset** directory. Make sure to update the training configuration accordingly.
```bash
wget -c https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json -P dataset/
```

