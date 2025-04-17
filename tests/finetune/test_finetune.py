# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

import numpy as np
import pytest
import torch.optim as optim
from torch.utils.data import DataLoader

import QEfficient
import QEfficient.cloud.finetune
from QEfficient.cloud.finetune import main as finetune


def clean_up(path):
    if os.path.exists(path):
        shutil.rmtree(path)


configs = [pytest.param("meta-llama/Llama-3.2-1B", 10, 20, 1, None, True, True, "qaic", id="llama_config")]


@pytest.mark.on_qaic
@pytest.mark.parametrize(
    "model_name,max_eval_step,max_train_step,intermediate_step_save,context_length,run_validation,use_peft,device",
    configs,
)
def test_finetune(
    model_name,
    max_eval_step,
    max_train_step,
    intermediate_step_save,
    context_length,
    run_validation,
    use_peft,
    device,
    mocker,
):
    train_config_spy = mocker.spy(QEfficient.cloud.finetune, "TrainConfig")
    generate_dataset_config_spy = mocker.spy(QEfficient.cloud.finetune, "generate_dataset_config")
    generate_peft_config_spy = mocker.spy(QEfficient.cloud.finetune, "generate_peft_config")
    get_dataloader_kwargs_spy = mocker.spy(QEfficient.cloud.finetune, "get_dataloader_kwargs")
    update_config_spy = mocker.spy(QEfficient.cloud.finetune, "update_config")
    get_custom_data_collator_spy = mocker.spy(QEfficient.cloud.finetune, "get_custom_data_collator")
    get_preprocessed_dataset_spy = mocker.spy(QEfficient.cloud.finetune, "get_preprocessed_dataset")
    get_longest_seq_length_spy = mocker.spy(QEfficient.cloud.finetune, "get_longest_seq_length")
    print_model_size_spy = mocker.spy(QEfficient.cloud.finetune, "print_model_size")
    train_spy = mocker.spy(QEfficient.cloud.finetune, "train")

    kwargs = {
        "model_name": model_name,
        "max_eval_step": max_eval_step,
        "max_train_step": max_train_step,
        "intermediate_step_save": intermediate_step_save,
        "context_length": context_length,
        "run_validation": run_validation,
        "use_peft": use_peft,
        "device": device,
    }

    results = finetune(**kwargs)

    assert np.allclose(results["avg_train_prep"], 1.002326, atol=1e-5), "Train perplexity is not matching."
    assert np.allclose(results["avg_train_loss"], 0.00232327, atol=1e-5), "Train loss is not matching."
    assert np.allclose(results["avg_eval_prep"], 1.0193923, atol=1e-5), "Eval perplexity is not matching."
    assert np.allclose(results["avg_eval_loss"], 0.0192067, atol=1e-5), "Eval loss is not matching."
    assert results["avg_epoch_time"] < 60, "Training should complete within 60 seconds."

    train_config_spy.assert_called_once()
    generate_dataset_config_spy.assert_called_once()
    generate_peft_config_spy.assert_called_once()
    get_custom_data_collator_spy.assert_called_once()
    get_longest_seq_length_spy.assert_called_once()
    print_model_size_spy.assert_called_once()
    train_spy.assert_called_once()

    assert update_config_spy.call_count == 2
    assert get_dataloader_kwargs_spy.call_count == 2
    assert get_preprocessed_dataset_spy.call_count == 2

    args, kwargs = train_spy.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    optimizer = args[4]

    batch = next(iter(train_dataloader))
    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert isinstance(optimizer, optim.AdamW)

    assert isinstance(train_dataloader, DataLoader)
    if run_validation:
        assert isinstance(eval_dataloader, DataLoader)
    else:
        assert eval_dataloader is None

    args, kwargs = update_config_spy.call_args_list[0]
    train_config = args[0]
    assert max_train_step >= train_config.gradient_accumulation_steps, (
        "Total training step should be more than 4 which is gradient accumulation steps."
    )

    saved_file = os.path.join(train_config.output_dir, "complete_epoch_1/adapter_model.safetensors")
    assert os.path.isfile(saved_file)

    clean_up(train_config.output_dir)
    clean_up("runs")
    clean_up(train_config.dump_root_dir)
