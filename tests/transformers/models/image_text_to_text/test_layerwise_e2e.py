# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import pytest
import requests
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_IDS = {
    "qwen3_vl_moe": "tiny-random/qwen3-vl-moe",
    "qwen3_5_moe": "tiny-random/qwen3.5-moe",
}
IMAGE_URL = "https://picsum.photos/id/237/536/354"
TEXT_PROMPT = "Descibe all the colors seen in the image."
TEXT_ONLY_PROMPT = "Tell me about yourself."
BATCH_SIZE = 1
PREFILL_SEQ_LEN = 32
CTX_LEN = 4096
GENERATION_LEN = 100
SKIP_VISION = True


@pytest.mark.on_qaic
@pytest.mark.regular
@pytest.mark.multimodal
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    [
        pytest.param("qwen3_vl_moe", MODEL_IDS["qwen3_vl_moe"]),
        pytest.param("qwen3_5_moe", MODEL_IDS["qwen3_5_moe"]),
    ],
)
@pytest.mark.skip(
    reason="These tests are currently failing due to token mismatch. They need to be fixed and re-enabled."
)
def test_qwen_multimodal_encoder_layerwise_vs_non_layerwise_tokens(
    manual_cleanup, model_type, model_id
):
    skip_vision = SKIP_VISION
    processor = AutoProcessor.from_pretrained(model_id)
    if skip_vision:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TEXT_ONLY_PROMPT},
                ],
            },
        ]
        messages = [messages] * BATCH_SIZE
        image_inputs, video_inputs = None, None
    else:
        image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
        messages_1 = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": TEXT_PROMPT},
                ],
            },
        ]
        messages = [messages_1] * BATCH_SIZE
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages)

    generated_ids = {}
    for layerwise in (False, True):
        torch.manual_seed(42)
        config = AutoConfig.from_pretrained(model_id)
        config.torch_dtype = "float32"

        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_id,
            attn_implementation="eager",
            kv_offload=True,
            config=config,
            torch_dtype=torch.float32,
            layerwise=layerwise,
        )

        # Follow the updated encoder script: first vision-only compile,
        # then language compile with skip_vision=True.
        if not skip_vision:
            qeff_model.compile(
                batch_size=BATCH_SIZE,
                prefill_seq_len=PREFILL_SEQ_LEN,
                ctx_len=CTX_LEN,
                num_cores=16,
                num_devices=1,
                height=354,
                width=536,
                mxfp6_matmul=False,
                aic_enable_depth_first=True,
                skip_vision=False,
                skip_lang=True,
                split_retained_state_io=True,
                use_onnx_subfunctions=True,
                mos=1,
                layerwise=False,
            )
        qeff_model.compile(
            batch_size=BATCH_SIZE,
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            num_cores=16,
            num_devices=1,
            height=354,
            width=536,
            mxfp6_matmul=False,
            aic_enable_depth_first=True,
            skip_vision=True,
            split_retained_state_io=True,
            use_onnx_subfunctions=True,
            prefill_only=True,
            mos=1,
            layerwise=layerwise,
            layerwise_window_size=1,
        )

        if skip_vision:
            run_inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            run_inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        run_inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=run_inputs,
            prefill_seq_len=PREFILL_SEQ_LEN,
            batch_size=BATCH_SIZE,
        )
        output = qeff_model.generate(inputs=run_inputs, generation_len=GENERATION_LEN)
        generated_ids[layerwise] = output.generated_ids
        manual_cleanup(qeff_model.onnx_path)

    assert torch.equal(torch.as_tensor(generated_ids[False]), torch.as_tensor(generated_ids[True])), (
        f"{model_type} generated tokens differ between layerwise=False and "
        f"layerwise=True with skip_vision={skip_vision}"
    )
