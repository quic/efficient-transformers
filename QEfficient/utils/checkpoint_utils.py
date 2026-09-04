# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file


def load_checkpoint_weights(checkpoint_path: str, keys: set[str]) -> dict[str, torch.Tensor]:
    """Read selected weight tensors from a local checkpoint directory or Hugging Face repository."""
    path = Path(checkpoint_path)
    if not path.is_dir():
        path = Path(snapshot_download(checkpoint_path, allow_patterns=["*.safetensors", "pytorch_model*.bin"]))

    found: dict[str, torch.Tensor] = {}
    for safetensors_file in sorted(path.glob("*.safetensors")):
        with safe_open(str(safetensors_file), framework="pt", device="cpu") as checkpoint:
            available_keys = set(checkpoint.keys())
            found.update({key: checkpoint.get_tensor(key) for key in keys & available_keys})
    if not found:
        for pytorch_file in sorted(path.glob("pytorch_model*.bin")):
            state_dict = torch.load(str(pytorch_file), map_location="cpu", weights_only=True)
            found.update({key: state_dict[key] for key in keys if key in state_dict})
    return found


def load_checkpoint(model, checkpoint: str, strict=False, post_process_func=None):
    """load weights ending with `.safetensors` extension
    Args:
        model: model to load wights into
        checkpoint (str): checkpoint path
        strict (bool, optional): strictness of loading weights. Defaults to False.
        post_process_func (optional): Optional post-processing of loaded state dict. Defaults to None.
    Returns:
        model: model with applied weights
    """
    state_dict: dict = load_file(checkpoint)
    if post_process_func is not None:
        state_dict = post_process_func(state_dict)
    model.load_state_dict(state_dict, strict=strict)
    return model
