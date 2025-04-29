# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from safetensors.torch import load_file


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