# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from contextlib import nullcontext
from enum import Enum
from typing import Dict, List, Tuple

import torch

try:
    import torch_qaic.debug as qaic_debug  # noqa: F401
except ImportError as e:
    print(f"Warning: {e}. Moving ahead without these qaic modules.")


class Batching_Strategy(str, Enum):
    PADDING = "padding"
    PACKING = "packing"


class Device(str, Enum):
    QAIC = "qaic"
    CPU = "cpu"
    CUDA = "cuda"


class Peft_Method(str, Enum):
    LORA = "lora"


class Task_Mode(str, Enum):
    GENERATION = "generation"
    SEQ_CLASSIFICATION = "seq_classification"


def enum_names(enum_cls):
    return [member.value for member in enum_cls]


def is_rank_zero():
    return int(os.getenv("LOCAL_RANK", 0)) == 0


def get_num_ddp_devices():
    return int(os.getenv("WORLD_SIZE", 1))


def get_autocast_ctx(use_autocast, device_type, dtype=torch.float16):
    return torch.autocast(device_type=device_type, dtype=dtype) if use_autocast else nullcontext()


def get_op_verifier_ctx(
    use_op_by_op_verifier,
    train_device,
    dump_dir,
    step,
    ref_device="cpu",
    ref_dtype=torch.float32,
    atol=1e-1,
    rtol=1e-5,
    use_ref_output_on_mismatch=True,
):
    if not use_op_by_op_verifier:
        return nullcontext()

    filter_config = qaic_debug.DispatchFilterConfig.default(train_device)
    dump_dir = dump_dir + "/mismatches/step_" + str(step)
    return qaic_debug.OpByOpVerifierMode(
        ref_device=ref_device,
        ref_dtype=ref_dtype,
        atol=atol,
        rtol=rtol,
        use_ref_output_on_mismatch=use_ref_output_on_mismatch,
        filter_config=filter_config,
        dump_root_dir=dump_dir,
    )


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_to_json(
    output_filename,
    train_step_loss,
    train_epoch_loss,
    train_step_metric,
    train_epoch_metric,
    val_step_loss,
    val_epoch_loss,
    val_step_metric,
    val_epoch_metric,
):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_metric": train_step_metric,
        "train_epoch_metric": train_epoch_metric,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_metric": val_step_metric,
        "val_epoch_metric": val_epoch_metric,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
