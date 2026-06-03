# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Private shared helpers for Qwen3-VL embedding example and tests."""

import os
import unicodedata
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info

DEFAULT_INSTRUCTION = "Represent the user's input."
DEFAULT_MAD_MAX = 1e-2

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR

EXAMPLE_QUERIES = [
    {"text": "A woman playing with her dog on a beach at sunset."},
]

EXAMPLE_DOCUMENTS = [
    {"image": "https://picsum.photos/id/237/536/354"},
]


def resolve_model_source(model_name_or_path: str) -> str:
    """Return a local model path when given an HF repo id."""
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    return snapshot_download(repo_id=model_name_or_path)


def configure_embedding_model_config(
    config,
    num_hidden_layers: int,
    vision_depth: int,
    deepstack_index: Optional[int],
    export_embedding: bool = True,
):
    """Apply Qwen3-VL embedding-specific config adjustments."""
    if hasattr(config, "use_cache"):
        config.use_cache = True
    if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
        config.text_config.use_cache = True
    if hasattr(config, "text_config") and num_hidden_layers > 0:
        config.text_config.num_hidden_layers = num_hidden_layers
    if hasattr(config, "vision_config"):
        if hasattr(config.vision_config, "depth") and vision_depth > 0:
            config.vision_config.depth = vision_depth
        if hasattr(config.vision_config, "deepstack_visual_indexes"):
            max_valid_idx = max(0, config.vision_config.depth - 1)
            if deepstack_index is None:
                default_indexes = [int(idx) for idx in config.vision_config.deepstack_visual_indexes]
                clamped_defaults = [idx for idx in default_indexes if 0 <= idx <= max_valid_idx]
                config.vision_config.deepstack_visual_indexes = (
                    clamped_defaults if clamped_defaults else [max_valid_idx]
                )
            else:
                config.vision_config.deepstack_visual_indexes = [min(max(0, int(deepstack_index)), max_valid_idx)]
    if export_embedding:
        config.export_embedding = True
    return config


def normalize_instruction(instruction: str) -> str:
    """Normalize instruction string and enforce trailing punctuation."""
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    return instruction


def format_model_input(
    text: Optional[str] = None,
    image: Optional[Any] = None,
    video: Optional[Any] = None,
    instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build one chat-style multimodal input for Qwen3-VL embedding."""
    resolved_instruction = normalize_instruction(instruction or DEFAULT_INSTRUCTION)

    content: List[Dict[str, Any]] = []
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": resolved_instruction}]},
        {"role": "user", "content": content},
    ]

    if not text and not image and not video:
        content.append({"type": "text", "text": "NULL"})
        return conversation

    if video:
        raise ValueError("Video input is not supported in this example.")

    if image:
        if isinstance(image, str):
            image_content = image if image.startswith(("http://", "https://", "oss")) else "file://" + image
        else:
            image_content = image
        content.append(
            {
                "type": "image",
                "image": image_content,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS,
            }
        )

    if text:
        content.append({"type": "text", "text": text})

    return conversation


def tokenize_conversation(processor, conversation: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Tokenize one chat conversation with multimodal processing."""
    conversations = [conversation]
    text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

    images, videos, video_kwargs = process_vision_info(
        conversations,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    return inputs
