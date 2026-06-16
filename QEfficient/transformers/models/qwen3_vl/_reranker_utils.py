# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Private shared helpers for Qwen3-VL reranker example and tests."""

import os
from typing import Dict, List, Tuple

import torch
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info


def resolve_model_source(model_name_or_path: str) -> str:
    """Return a local model path when given an HF repo id."""
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    return snapshot_download(repo_id=model_name_or_path)


def get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
    """Resolve tokenizer ids for exact tokens 'yes' and 'no'."""
    vocab = tokenizer.get_vocab()
    if "yes" not in vocab or "no" not in vocab:
        raise ValueError("Could not resolve tokenizer ids for exact tokens 'yes' and 'no'.")
    return vocab["yes"], vocab["no"]


def score_from_logits(logits, yes_token_id: int, no_token_id: int) -> torch.Tensor:
    """Compute sigmoid(logit_yes - logit_no) from model logits."""
    logits_tensor = torch.from_numpy(logits) if hasattr(logits, "shape") and not torch.is_tensor(logits) else logits
    logits_tensor = logits_tensor.detach().to(torch.float32).cpu()
    if logits_tensor.ndim == 3:
        logits_tensor = logits_tensor[:, -1, :]
    elif logits_tensor.ndim != 2:
        raise ValueError(f"Unsupported logits rank for score conversion: {logits_tensor.ndim}")
    return torch.sigmoid(logits_tensor[:, yes_token_id] - logits_tensor[:, no_token_id])


def truncate_tokens_optimized(tokens: List[int], max_length: int, special_tokens: List[int]) -> List[int]:
    """Truncate while preserving all special tokens in sequence order."""
    if len(tokens) <= max_length:
        return tokens

    special_tokens_set = set(special_tokens)
    num_special = sum(1 for token in tokens if token in special_tokens_set)
    num_non_special_to_keep = max_length - num_special

    final_tokens = []
    non_special_kept_count = 0
    for token in tokens:
        if token in special_tokens_set:
            final_tokens.append(token)
        elif non_special_kept_count < num_non_special_to_keep:
            final_tokens.append(token)
            non_special_kept_count += 1
    return final_tokens


def format_mm_content(
    text,
    image,
    video,
    prefix: str,
    min_pixels: int,
    max_pixels: int,
    unsupported_video_error: str,
) -> List[Dict]:
    """Build one multimodal content block."""
    content = [{"type": "text", "text": prefix}]

    if not text and not image and not video:
        content.append({"type": "text", "text": "NULL"})
        return content

    if video:
        raise ValueError(unsupported_video_error)

    if image:
        if isinstance(image, str):
            image_content = image if image.startswith(("http", "oss")) else "file://" + image
        else:
            image_content = image
        content.append(
            {
                "type": "image",
                "image": image_content,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )

    if text:
        content.append({"type": "text", "text": text})

    return content


def format_mm_instruction(
    instruction: str,
    query: Dict,
    document: Dict,
    min_pixels: int,
    max_pixels: int,
    unsupported_video_error: str,
) -> List[Dict]:
    """Create chat payload for one query-document pair."""
    contents = [{"type": "text", "text": "<Instruct>: " + instruction}]

    contents.extend(
        format_mm_content(
            query.get("text"),
            query.get("image"),
            query.get("video"),
            prefix="<Query>:",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            unsupported_video_error=unsupported_video_error,
        )
    )
    contents.extend(
        format_mm_content(
            document.get("text"),
            document.get("image"),
            document.get("video"),
            prefix="\n<Document>:",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            unsupported_video_error=unsupported_video_error,
        )
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Judge whether the Document meets the requirements based on the Query and the Instruct "
                        'provided. Note that the answer can only be "yes" or "no".'
                    ),
                }
            ],
        },
        {"role": "user", "content": contents},
    ]


def tokenize_pair(processor, pair: List[Dict], max_length: int) -> Dict:
    """Tokenize one query-document pair with HF multimodal processor."""
    pairs = [pair]
    text = processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)

    images, videos, video_kwargs = process_vision_info(
        pairs,
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
        truncation=False,
        padding=False,
        do_resize=False,
        **video_kwargs,
    )

    for i, input_ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = (
            truncate_tokens_optimized(
                input_ids[:-5],
                max_length,
                processor.tokenizer.all_special_ids,
            )
            + input_ids[-5:]
        )

    padded = processor.tokenizer.pad(
        {"input_ids": inputs["input_ids"]},
        padding=True,
        return_tensors="pt",
        max_length=max_length,
    )
    for key in padded:
        inputs[key] = padded[key]

    # HF Qwen3-VL processors may return list-based modality ids. Normalize to
    # tensor so downstream boolean masking in model forward works across versions.
    if "mm_token_type_ids" in inputs and not torch.is_tensor(inputs["mm_token_type_ids"]):
        seq_len = int(inputs["input_ids"].shape[1])
        mm_token_type_ids = []
        for token_types in inputs["mm_token_type_ids"]:
            token_types = list(token_types)
            if len(token_types) < seq_len:
                token_types = token_types + [0] * (seq_len - len(token_types))
            else:
                token_types = token_types[:seq_len]
            mm_token_type_ids.append(token_types)
        inputs["mm_token_type_ids"] = torch.tensor(mm_token_type_ids, dtype=torch.int64)

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    return inputs
