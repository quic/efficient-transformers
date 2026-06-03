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

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from QEfficient.generation.cloud_infer import QAICInferenceSession

try:
    from qwen_vl_utils import process_vision_info as _process_vision_info
except ModuleNotFoundError:
    _process_vision_info = None

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
    if _process_vision_info is None:
        raise ModuleNotFoundError(
            "qwen_vl_utils is required for multimodal tokenization. Install it via: pip install 'qwen-vl-utils>=0.0.14'"
        )

    conversations = [conversation]
    text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

    images, videos, video_kwargs = _process_vision_info(
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


class QEffQwen3VLEmbedder:
    """End-to-end AI100 embedding helper for Qwen3-VL.

    This helper owns the runtime flow:
    1) format/tokenize inputs, 2) derive compile specs, 3) run vision+language QPCs,
    and 4) return optional L2-normalized embeddings.
    """

    def __init__(self, processor, model):
        """Store the HF processor and QEff model used by runtime methods."""
        self.processor = processor
        self.model = model

    def format_model_input(
        self,
        text: str = None,
        image: Any = None,
        video: Any = None,
        instruction: str = None,
    ) -> List[Dict[str, Any]]:
        """Create one chat-style multimodal conversation payload."""
        return format_model_input(
            text=text,
            image=image,
            video=video,
            instruction=instruction,
        )

    def _tokenize_conversation(self, conversation: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Tokenize one conversation into model-ready tensors."""
        return tokenize_conversation(self.processor, conversation)

    @staticmethod
    def _prepare_qeff_inputs(qeff_model, tokenized_inputs: Dict[str, torch.Tensor], prefill_seq_len: int):
        """Adapt tokenized inputs to QEff prefill format and validate lengths."""
        runtime_prompt_len = int(tokenized_inputs["input_ids"].shape[1])
        if prefill_seq_len < runtime_prompt_len:
            raise ValueError(
                f"prefill_seq_len ({prefill_seq_len}) must be >= runtime prompt length ({runtime_prompt_len})."
            )

        prepared_inputs = qeff_model.model.prepare_inputs_for_generation(
            inputs=tokenized_inputs,
            prefill_seq_len=prefill_seq_len,
            batch_size=1,
        )

        if "image_grid_thw" in prepared_inputs and prepared_inputs["image_grid_thw"].ndim == 2:
            thw = prepared_inputs["image_grid_thw"][0]
            t, h, w = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
            prepared_inputs["image_grid_thw"] = torch.zeros((1, t, h, w), dtype=thw.dtype)

        if "pixel_values" in prepared_inputs:
            prepared_inputs["pixel_values"] = prepared_inputs["pixel_values"].to(torch.float32)

        return prepared_inputs, runtime_prompt_len

    def _collect_contexts(self, inputs: List[Dict[str, Any]]):
        """Tokenize all entries and gather max prompt/image dimensions."""
        contexts = []
        max_prompt_len = 0
        max_grid_h = 22
        max_grid_w = 34

        for entry in inputs:
            conversation = self.format_model_input(
                text=entry.get("text"),
                image=entry.get("image"),
                video=entry.get("video"),
                instruction=entry.get("instruction"),
            )
            tokenized = self._tokenize_conversation(conversation)
            runtime_prompt_len = int(tokenized["input_ids"].shape[1])

            if "image_grid_thw" in tokenized and tokenized["image_grid_thw"].numel() > 0:
                grid = tokenized["image_grid_thw"]
                max_grid_h = max(max_grid_h, int(grid[..., 1].max().item()))
                max_grid_w = max(max_grid_w, int(grid[..., 2].max().item()))

            contexts.append({"tokenized": tokenized})
            max_prompt_len = max(max_prompt_len, runtime_prompt_len)

        return contexts, max_prompt_len, max_grid_h, max_grid_w

    def get_compile_specs(
        self, inputs: List[Dict[str, Any]], ctx_len: int, prefill_seq_len: int = None
    ) -> Dict[str, int]:
        """Compute compile-time spec values for the current input batch."""
        _, max_prompt_len, max_grid_h, max_grid_w = self._collect_contexts(inputs)
        if max_prompt_len == 0:
            raise ValueError("At least one input is required for compile spec generation.")

        target_prefill_seq_len = max_prompt_len if prefill_seq_len is None else int(prefill_seq_len)
        if target_prefill_seq_len < max_prompt_len:
            raise ValueError(
                f"compile prefill_seq_len ({target_prefill_seq_len}) must be >= max runtime prompt length ({max_prompt_len})."
            )

        patch_size = int(self.model.model.config.vision_config.patch_size)
        height = max_grid_h * patch_size
        width = max_grid_w * patch_size

        return {
            "prefill_seq_len": target_prefill_seq_len,
            "ctx_len": int(ctx_len),
            "img_size": max(height, width),
            "height": height,
            "width": width,
        }

    @staticmethod
    def _zero_vision_outputs(vision_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Build zero-filled vision retained-state buffers with matching shapes."""
        return {name: np.zeros_like(value) for name, value in vision_outputs.items()}

    @staticmethod
    def _run_ai100_vision(vision_qpc_path: str, prepared_inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Execute the vision QPC and return retained-state output buffers."""
        vision_session = QAICInferenceSession(vision_qpc_path)
        vision_outputs = vision_session.run(
            {
                "pixel_values": prepared_inputs["pixel_values"].detach().cpu().numpy().astype(np.float16),
                "image_grid_thw": prepared_inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64),
            }
        )
        vision_session.deactivate()
        return vision_outputs

    @staticmethod
    def _run_ai100_prefill(
        prepared_inputs: Dict[str, torch.Tensor],
        vision_outputs: Dict[str, np.ndarray],
        lang_qpc_path: str,
    ) -> np.ndarray:
        """Execute one language prefill pass and return the embedding row."""
        prefill_len = prepared_inputs["position_ids"].shape[-1]
        input_ids = prepared_inputs["input_ids"]
        if input_ids.shape[1] < prefill_len:
            pad = torch.full(
                (input_ids.shape[0], prefill_len - input_ids.shape[1]),
                1,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
        else:
            input_ids = input_ids[:, :prefill_len]

        position_ids = prepared_inputs["position_ids"][..., :prefill_len]

        lang_session = QAICInferenceSession(lang_qpc_path)
        lang_session.skip_buffers(
            [
                name
                for name in lang_session.input_names + lang_session.output_names
                if name.startswith("past_") or name.endswith("_RetainedState")
            ]
        )
        lang_session.set_buffers(vision_outputs)
        outputs = lang_session.run(
            {
                "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
                "position_ids": position_ids.detach().cpu().numpy().astype(np.int64),
                "image_idx": np.zeros((1, 1), dtype=np.int64),
            }
        )
        lang_session.deactivate()

        if "embedding_output" not in outputs:
            raise KeyError(
                "Missing 'embedding_output' in AI100 decoder outputs. "
                "Ensure export_embedding is enabled in config/qaic_config."
            )

        embedding_output = outputs["embedding_output"]
        if embedding_output.ndim > 2:
            embedding_output = embedding_output.reshape(embedding_output.shape[0], -1)
        return embedding_output

    def process(
        self,
        inputs: List[Dict[str, Any]],
        qpc_paths: Dict[str, str],
        prefill_seq_len: int,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Run AI100 embedding generation for all inputs and return stacked rows."""
        if "vision_qpc_path" not in qpc_paths or "lang_qpc_path" not in qpc_paths:
            raise ValueError("qpc_paths must contain 'vision_qpc_path' and 'lang_qpc_path'.")

        contexts, max_prompt_len, _, _ = self._collect_contexts(inputs)
        if max_prompt_len == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        target_prefill_seq_len = int(prefill_seq_len)
        if target_prefill_seq_len < max_prompt_len:
            raise ValueError(
                f"prefill_seq_len ({target_prefill_seq_len}) must be >= max runtime prompt length ({max_prompt_len})."
            )

        prepared_contexts = []
        vision_template = None
        for ctx in contexts:
            prepared_inputs, _ = self._prepare_qeff_inputs(
                qeff_model=self.model,
                tokenized_inputs=ctx["tokenized"],
                prefill_seq_len=target_prefill_seq_len,
            )
            prepared_contexts.append({"prepared_inputs": prepared_inputs})

            if vision_template is None and "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
                vision_template = self._run_ai100_vision(
                    vision_qpc_path=qpc_paths["vision_qpc_path"],
                    prepared_inputs=prepared_inputs,
                )

        if vision_template is None:
            raise ValueError("At least one input with an image is required to initialize AI100 vision buffers.")

        embedding_rows = []
        for ctx in prepared_contexts:
            prepared_inputs = ctx["prepared_inputs"]
            if "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
                vision_outputs = self._run_ai100_vision(
                    vision_qpc_path=qpc_paths["vision_qpc_path"],
                    prepared_inputs=prepared_inputs,
                )
            else:
                vision_outputs = self._zero_vision_outputs(vision_template)

            embedding_output = self._run_ai100_prefill(
                prepared_inputs=prepared_inputs,
                vision_outputs=vision_outputs,
                lang_qpc_path=qpc_paths["lang_qpc_path"],
            )
            embedding_rows.append(torch.from_numpy(embedding_output).to(torch.float32))

        embeddings = torch.cat(embedding_rows, dim=0)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
