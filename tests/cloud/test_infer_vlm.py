# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import pytest

from QEfficient.cloud.infer import main as infer


@pytest.mark.on_qaic
@pytest.mark.cli
@pytest.mark.multimodal
@pytest.mark.usefixtures("clean_up_after_test")
def test_vlm_cli(setup, mocker):
    ms = setup
    # Taking some values from setup fixture and assigning other's based on model's requirement.
    # For example, mxint8 is not required for VLM models, so assigning False.
    infer(
        model_name="llava-hf/llava-1.5-7b-hf",
        num_cores=ms.num_cores,
        prompt="Describe the image.",
        prompts_txt_file_path=None,
        aic_enable_depth_first=ms.aic_enable_depth_first,
        mos=ms.mos,
        batch_size=1,
        full_batch_size=None,
        prompt_len=1024,
        ctx_len=2048,
        generation_len=ms.generation_len,
        mxfp6=False,
        mxint8=False,
        local_model_dir=None,
        cache_dir=None,
        hf_token=ms.hf_token,
        enable_qnn=False,
        qnn_config=None,
        image_url="https://i.etsystatic.com/8155076/r/il/0825c2/1594869823/il_fullxfull.1594869823_5x0w.jpg",
    )
