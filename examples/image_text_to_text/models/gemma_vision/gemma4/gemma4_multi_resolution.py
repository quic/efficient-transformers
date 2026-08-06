# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------



import requests
from gemma4_utils import *
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-E2B-it"

# ---------------------------------------------------------------------------
# Sequence-length budget
# ---------------------------------------------------------------------------
PREFILL_SEQ_LEN = 256  # Must be >= longest tokenised vision prompt
CTX_LEN = 2048
GENERATION_LEN = 100
BATCH_SIZE = 1

# ---------------------------------------------------------------------------
# Testing knobs: reduce layers for fast end-to-end validation
# ---------------------------------------------------------------------------
NUM_LANG_HIDDEN_LAYER = 2
NUM_VISION_HIDDEN_LAYER = 2

# ---------------------------------------------------------------------------
# Compiler settings
# ---------------------------------------------------------------------------
NUM_CORES = 16
NUM_DEVICES = 2
NODE_PRECISION_INFO = True  # Auto-generate Gemma4 NPI file for mixed precision

# ---------------------------------------------------------------------------
# Sample inputs ( prompts / images)
# ---------------------------------------------------------------------------
IMAGE_URL = (
    "https://wallup.net/wp-content/uploads/2017/03/28/351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"
)
IMAGE_PROMPT = "Can you Describe this image in detail?"
SYSTEM_PROMPT = "You are a helpful assistant."


def _apply_reduced_layer_config(config, num_lang_layers: int, num_vision_layers: int):
    """Shrink layer counts so the model fits in CPU RAM during testing."""
    config.text_config.num_hidden_layers = num_lang_layers
    config.vision_config.num_hidden_layers = num_vision_layers

    if hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]

    if hasattr(config.text_config, "num_kv_shared_layers"):
        # Avoid invalid first_kv_shared_layer_idx=0 edge case with few layers.
        config.text_config.num_kv_shared_layers = 0

    return config


def main():
    # ------------------------------------------------------------------
    # STEP 1: Processor / tokenizer
    # ------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer
    chat_template = (
        getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE
    )

    # ------------------------------------------------------------------
    # STEP 2: Config (with layer reduction for testing)
    # ------------------------------------------------------------------
    config = AutoConfig.from_pretrained(MODEL_ID)
    # For Testing Purpose Only
    # config = _apply_reduced_layer_config(config, NUM_LANG_HIDDEN_LAYER, NUM_VISION_HIDDEN_LAYER)
    # ------------------------------------------------------------------
    # STEP 3: Height and Width config
    # ------------------------------------------------------------------

    resolutions = [
        {"width": 360, "height": 240},
        {"width": 536, "height": 354},
        {"width": 1024, "height": 1024},
    ]

    widths = [s["width"] for s in resolutions]
    heights = [s["height"] for s in resolutions]
    # ------------------------------------------------------------------
    # STEP 4: Model Loading
    # ------------------------------------------------------------------
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,  # Dual-QPC: vision encoder + LM decoder
        ignore_mismatched_sizes=True,
    )
    remove_fp16clip_transform_if_disabled(qeff_model, effective_fp16clip=True)

    # ------------------------------------------------------------------
    # STEP 5: Compile both QPCs
    # ------------------------------------------------------------------
    qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        batch_size=BATCH_SIZE,
        num_cores=NUM_CORES,
        num_devices=NUM_DEVICES,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        height=heights,
        width=widths,
        aic_enable_depth_first=True,
        mos=1,
        split_model_io=True,
        node_precision_info=NODE_PRECISION_INFO,
        use_onnx_subfunctions=False,
    )

    # ------------------------------------------------------------------
    # STEP 6: Generate
    # ------------------------------------------------------------------
    messages = build_messages(SYSTEM_PROMPT, IMAGE_PROMPT, use_image=True)

    for i, (w, h) in enumerate(zip(widths, heights)):
        image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
        image = image.resize((w, h))
        messages[-1]["content"][0]["url"] = image
        required_soft_tokens = qeff_model.model.choose_gemma4_max_soft_tokens(h, w)
        processor.image_processor.max_soft_tokens = required_soft_tokens
        config.vision_soft_tokens_per_image = required_soft_tokens
        inputs = processor.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        output = qeff_model.generate(
            inputs=inputs,
            generation_len=GENERATION_LEN,
            multi_specs=True,
            tokenizer=tokenizer,
            processor=processor,
        )
        qeff_ids = normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
        generated_texts = tokenizer.batch_decode(qeff_ids, skip_special_tokens=True)

        for text in generated_texts:
            print(f"\n--- Response [{i}] ---")
            print(text)
        i = i + 1

        print("\nExecution info:")
        print(output)


if __name__ == "__main__":
    main()
