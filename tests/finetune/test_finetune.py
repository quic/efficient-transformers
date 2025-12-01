# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil

import numpy as np
import pytest
import requests
import torch.optim as optim
from torch.utils.data import DataLoader

import QEfficient
import QEfficient.cloud.finetune
from QEfficient.cloud.finetune import main as finetune
from QEfficient.finetune.utils.helper import Device, Task_Mode, get_rank, get_world_size
from tests.finetune import constants as constant
from tests.finetune import reference_data as ref_data

alpaca_json_path = os.path.join(os.getcwd(), "./dataset/alpaca_data.json")


def clean_up(path):
    if os.path.isdir(path) and os.path.exists(path):
        shutil.rmtree(path)
    if os.path.isfile(path):
        os.remove(path)


def download_alpaca():
    alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json"
    response = requests.get(alpaca_url)
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(alpaca_json_path), exist_ok=True)
    with open(alpaca_json_path, "wb") as f:
        f.write(response.content)


# Define a helper function for comparing lists


def assert_list_close(ref_list, actual_list, atol, name, scenario_key, current_world_size, current_rank):
    """
    Asserts that two lists of floats are numerically close element-wise.
    If not close, reports the step numbers and the differences at those steps.
    """
    # --- Initial Checks ---
    assert actual_list is not None and isinstance(actual_list, list), (
        f"Actual {name} data is missing or not a list for scenario '{scenario_key}'."
    )
    assert len(ref_list) == len(actual_list), (
        f"{name} length mismatch for scenario '{scenario_key}' (WS: {current_world_size}, Rank: {current_rank}). "
        f"Expected {len(ref_list)} elements, but got {len(actual_list)}."
    )

    # --- Convert to NumPy arrays for efficient comparison ---
    ref_arr = np.array(ref_list)
    actual_arr = np.array(actual_list)

    # --- Check if all elements are close using np.allclose ---
    # This is the primary assertion that will fail if any deviation is too large
    if not np.allclose(ref_arr, actual_arr, atol=atol):
        # If not all close, identify the specific deviations
        deviated_indices = np.where(~np.isclose(ref_arr, actual_arr, atol=atol))[0]
        deviation_details = []
        for idx in deviated_indices:
            ref_val = ref_arr[idx]
            actual_val = actual_arr[idx]
            diff = actual_val - ref_val
            deviation_details.append(f"Step {idx}: Ref={ref_val:.6f}, Actual={actual_val:.6f}, Diff={diff:.6f}")

        # Calculate max_diff
        max_diff = np.max(np.abs(ref_arr - actual_arr))

        # --- Report detailed deviation in the AssertionError ---
        error_message = (
            f"{name} deviated too much for scenario '{scenario_key}' "
            f"(WS: {current_world_size}, Rank: {current_rank}).\n"
            f"Max Difference: {max_diff:.6f}, Allowed Tolerance: {atol:.6f}.\n"
            f"Deviations found at {len(deviated_indices)} steps:\n" + "\n".join(deviation_details) + "\n"
            f"Reference (first 10): {ref_list[:10]}...\n"
            f"Actual    (first 10): {actual_list[:10]}..."
        )
        assert False, error_message  # Force the assertion to fail with the custom message
    else:
        # If all close, report success and max_diff for printing
        max_diff = np.max(np.abs(ref_arr - actual_arr))
        print(f"  ✅ {name} PASSED. Max Diff: {max_diff:.6f}")


configs = [
    pytest.param(
        "meta-llama/Llama-3.2-1B",  # model_name
        Task_Mode.GENERATION,  # task_mode
        10,  # max_eval_step
        20,  # max_train_step
        "gsm8k_dataset",  # dataset_name
        None,  # data_path
        10,  # intermediate_step_save
        None,  # context_length
        True,  # run_validation
        True,  # use_peft
        Device.QAIC,  # device
        "llama_3.2_1B_config_gsm8k_single_device",
        id="llama_config_gsm8k",  # config name
    ),
    pytest.param(
        "meta-llama/Llama-3.2-1B",  # model_name
        Task_Mode.GENERATION,  # task_mode
        10,  # max_eval_step
        20,  # max_train_step
        "alpaca_dataset",  # dataset_name
        alpaca_json_path,  # data_path
        10,  # intermediate_step_save
        None,  # context_length
        True,  # run_validation
        True,  # use_peft
        Device.QAIC,  # device
        "llama_3.2_1B_config_alpaca_single_device",
        id="llama_config_alpaca",  # config name
    ),
    pytest.param(
        "google-bert/bert-base-uncased",  # model_name
        Task_Mode.SEQ_CLASSIFICATION,  # task_mode
        10,  # max_eval_step
        20,  # max_train_step
        "imdb_dataset",  # dataset_name
        None,  # data_path
        10,  # intermediate_step_save
        None,  # context_length
        True,  # run_validation
        False,  # use_peft
        Device.QAIC,  # device
        "bert_base_uncased_config_imdb_single_device",
        id="bert_config_imdb",  # config name
    ),
]


def train_function(
    model_name,
    task_mode,
    max_eval_step,
    max_train_step,
    dataset_name,
    data_path,
    intermediate_step_save,
    context_length,
    run_validation,
    use_peft,
    device,
    scenario_key,
    mocker,
):
    train_config_spy = mocker.spy(QEfficient.cloud.finetune, "TrainConfig")
    generate_dataset_config_spy = mocker.spy(QEfficient.cloud.finetune, "generate_dataset_config")
    generate_peft_config_spy = mocker.spy(QEfficient.cloud.finetune, "generate_peft_config")
    get_dataloader_kwargs_spy = mocker.spy(QEfficient.finetune.utils.dataset_utils, "get_dataloader_kwargs")
    update_config_spy = mocker.spy(QEfficient.cloud.finetune, "update_config")
    get_custom_data_collator_spy = mocker.spy(QEfficient.finetune.utils.dataset_utils, "get_custom_data_collator")
    get_preprocessed_dataset_spy = mocker.spy(QEfficient.finetune.utils.dataset_utils, "get_preprocessed_dataset")
    get_longest_seq_length_spy = mocker.spy(QEfficient.cloud.finetune, "get_longest_seq_length")
    print_model_size_spy = mocker.spy(QEfficient.cloud.finetune, "print_model_size")
    train_spy = mocker.spy(QEfficient.cloud.finetune, "train")

    kwargs = {
        "model_name": model_name,
        "task_mode": task_mode,
        "max_eval_step": max_eval_step,
        "max_train_step": max_train_step,
        "dataset": dataset_name,
        "data_path": data_path,
        "intermediate_step_save": intermediate_step_save,
        "context_length": context_length,
        "run_validation": run_validation,
        "use_peft": use_peft,
        "device": device,
    }

    reference_data = ref_data.REFERENCE_DATA.get(scenario_key)
    if reference_data is None:
        pytest.fail(f"Reference data for scenario '{scenario_key}' not found in REFERENCE_DATA.")
    current_world_size = get_world_size()
    current_rank = get_rank()
    if current_world_size > 1:
        rank_reference_data = reference_data.get("rank_data", {}).get(str(current_rank))
        if rank_reference_data is None:
            pytest.fail(f"Reference data for rank {current_rank} not found in distributed scenario '{scenario_key}'.")
        ref_train_losses = rank_reference_data["train_step_losses"]
        ref_eval_losses = rank_reference_data["eval_step_losses"]
        ref_train_metrics = rank_reference_data["train_step_metrics"]
        ref_eval_metrics = rank_reference_data["eval_step_metrics"]
    else:  # Single device or world_size=1
        ref_train_losses = reference_data["train_step_losses"]
        ref_eval_losses = reference_data["eval_step_losses"]
        ref_train_metrics = reference_data["train_step_metrics"]
        ref_eval_metrics = reference_data["eval_step_metrics"]

    if dataset_name == "alpaca_dataset":
        download_alpaca()

    results = finetune(**kwargs)
    all_ref_metrices = {
        "ref_train_losses": ref_train_losses,
        "ref_eval_losses": ref_eval_losses,
        "ref_train_metrics": ref_train_metrics,
        "ref_eval_metrics": ref_eval_metrics,
    }

    all_config_spy = {
        "train_config_spy": train_config_spy,
        "generate_dataset_config_spy": generate_dataset_config_spy,
        "generate_peft_config_spy": generate_peft_config_spy,
        "get_dataloader_kwargs_spy": get_dataloader_kwargs_spy,
        "update_config_spy": update_config_spy,
        "get_custom_data_collator_spy": get_custom_data_collator_spy,
        "get_preprocessed_dataset_spy": get_preprocessed_dataset_spy,
        "get_longest_seq_length_spy": get_longest_seq_length_spy,
        "print_model_size_spy": print_model_size_spy,
        "train_spy": train_spy,
        "current_world_size": current_world_size,
        "current_rank": current_rank,
    }
    return results, all_ref_metrices, all_config_spy


@pytest.mark.cli
@pytest.mark.on_qaic
@pytest.mark.finetune
@pytest.mark.parametrize(
    "model_name,task_mode,max_eval_step,max_train_step,dataset_name,data_path,intermediate_step_save,context_length,run_validation,use_peft,device,scenario_key",  # This parameter will be used to look up reference data
    configs,
)
def test_finetune_functional(
    model_name,
    task_mode,
    max_eval_step,
    max_train_step,
    dataset_name,
    data_path,
    intermediate_step_save,
    context_length,
    run_validation,
    use_peft,
    device,
    scenario_key,
    mocker,
):
    results, all_ref_metrices, all_config_spy = train_function(
        model_name,
        task_mode,
        max_eval_step,
        max_train_step,
        dataset_name,
        data_path,
        intermediate_step_save,
        context_length,
        run_validation,
        use_peft,
        device,
        scenario_key,
        mocker,
    )

    # Assertions for step-level values using the helper function
    assert results["avg_epoch_time"] < 60, "Training should complete within 60 seconds."
    all_config_spy["train_config_spy"].assert_called_once()
    all_config_spy["generate_dataset_config_spy"].assert_called_once()
    if task_mode == Task_Mode.GENERATION:
        all_config_spy["generate_peft_config_spy"].assert_called_once()
    all_config_spy["get_longest_seq_length_spy"].assert_called_once()
    all_config_spy["print_model_size_spy"].assert_called_once()
    all_config_spy["train_spy"].assert_called_once()
    assert all_config_spy["update_config_spy"].call_count == 1
    assert all_config_spy["get_custom_data_collator_spy"].call_count == 2
    assert all_config_spy["get_dataloader_kwargs_spy"].call_count == 2
    assert all_config_spy["get_preprocessed_dataset_spy"].call_count == 2
    args, kwargs = all_config_spy["train_spy"].call_args
    train_dataloader = args[2]
    eval_dataloader = args[3]
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
    args, kwargs = all_config_spy["update_config_spy"].call_args_list[0]
    train_config = args[0]
    assert max_train_step >= train_config.gradient_accumulation_steps, (
        "Total training step should be more than "
        f"{train_config.gradient_accumulation_steps} which is gradient accumulation steps."
    )
    if use_peft:
        saved_file = os.path.join(train_config.output_dir, "complete_epoch_1/adapter_model.safetensors")
    else:
        saved_file = os.path.join(train_config.output_dir, "complete_epoch_1/model.safetensors")
    assert os.path.isfile(saved_file)
    clean_up(train_config.output_dir)
    clean_up("qaic-dumps")

    if dataset_name == "alpaca_dataset":
        clean_up(alpaca_json_path)


@pytest.mark.skip()  # remove when it's clear why diff val_step_loss values are observed in diff runs on existing code (even without PR #478 changes)
@pytest.mark.cli
@pytest.mark.on_qaic
@pytest.mark.finetune
@pytest.mark.parametrize(
    "model_name,task_mode,max_eval_step,max_train_step,dataset_name,data_path,intermediate_step_save,context_length,run_validation,use_peft,device,scenario_key",  # This parameter will be used to look up reference data
    configs,
)
def test_finetune_assert(
    model_name,
    task_mode,
    max_eval_step,
    max_train_step,
    dataset_name,
    data_path,
    intermediate_step_save,
    context_length,
    run_validation,
    use_peft,
    device,
    scenario_key,
    mocker,
):
    results, all_ref_metrices, all_config_spy = train_function(
        model_name,
        task_mode,
        max_eval_step,
        max_train_step,
        dataset_name,
        data_path,
        intermediate_step_save,
        context_length,
        run_validation,
        use_peft,
        device,
        scenario_key,
        mocker,
    )

    # Assertions for step-level values using the helper function
    assert_list_close(
        all_ref_metrices["ref_train_losses"],
        results["train_step_loss"],
        constant.LOSS_ATOL,
        "Train Step Losses",
        scenario_key,
        all_config_spy["current_world_size"],
        all_config_spy["current_rank"],
    )
    assert_list_close(
        all_ref_metrices["ref_eval_losses"],
        results["eval_step_loss"],
        constant.LOSS_ATOL,
        "Eval Step Losses",
        scenario_key,
        all_config_spy["current_world_size"],
        all_config_spy["current_rank"],
    )
    assert_list_close(
        all_ref_metrices["ref_train_metrics"],
        results["train_step_metric"],
        constant.METRIC_ATOL,
        "Train Step Metrics",
        scenario_key,
        all_config_spy["current_world_size"],
        all_config_spy["current_rank"],
    )
    assert_list_close(
        all_ref_metrices["ref_eval_metrics"],
        results["eval_step_metric"],
        constant.METRIC_ATOL,
        "Eval Step Metrics",
        scenario_key,
        all_config_spy["current_world_size"],
        all_config_spy["current_rank"],
    )
    clean_up("qaic-dumps")

    if dataset_name == "alpaca_dataset":
        clean_up(alpaca_json_path)
