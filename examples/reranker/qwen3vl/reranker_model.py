# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Qwen3-VL-specific reranker helpers for AI100 runtime.

The tokenization/scoring flow is adapted from the official Qwen reference:
https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/blob/main/scripts/qwen3_vl_reranker.py

This module intentionally keeps only Qwen3-VL-specific reranker logic
(prompt construction, multimodal tokenization, yes/no score computation,
and AI100 runtime orchestration with compiled QPC paths).

Model loading (`from_pretrained`) and model compilation (`compile`) are exposed
in `qwen3_vl_reranker.py` so users can directly see QEff API usage.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    format_mm_content,
    format_mm_instruction,
    get_yes_no_token_ids,
    score_from_logits,
    tokenize_pair,
    truncate_tokens_optimized,
)
from QEfficient.transformers.models.qwen3_vl._reranker_utils import (
    resolve_model_source as _resolve_model_source,
)

# Max token budget used by this example's manual truncation/padding flow.
MAX_LENGTH = 8192
# Pixel constraints used by Qwen3-VL preprocessing.
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1280 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1.0


def resolve_model_source(model_name_or_path: str) -> str:
    """Return a local model path when given an HF repo id.

    Some transformers versions can fail when resolving chat templates from
    repo-id mode for this model. Using a local snapshot path avoids that path.
    """
    return _resolve_model_source(model_name_or_path)


class QEffQwen3VLReranker:
    """Qwen3-VL reranker runtime helper for AI100 compiled QPCs."""

    def __init__(self, processor, model, max_length: int = MAX_LENGTH):
        """Initialize helper with preloaded processor and QEff model.

        Parameters
        ----------
        processor:
            HF AutoProcessor for Qwen3-VL reranker.
        model:
            QEFFAutoModelForImageTextToText instance.
        max_length:
            Max token length used by truncation/padding logic.
        """
        self.processor = processor
        self.model = model
        self.max_length = max_length
        self.fps = FPS
        self.yes_token_id, self.no_token_id = self._get_yes_no_token_ids(self.processor.tokenizer)

    @staticmethod
    def _get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
        """Resolve tokenizer ids for the exact tokens 'yes' and 'no'."""
        return get_yes_no_token_ids(tokenizer)

    @staticmethod
    def _score_from_logits(logits, yes_token_id: int, no_token_id: int) -> float:
        """Convert model logits into a reranker relevance score.

        Score formula:
            sigmoid(logit_yes - logit_no)
        """
        score = score_from_logits(logits, yes_token_id, no_token_id)
        return float(score[0].item())

    @staticmethod
    def _truncate_tokens_optimized(tokens: List[int], max_length: int, special_tokens: List[int]) -> List[int]:
        """Truncate while preserving all special tokens in sequence order."""
        return truncate_tokens_optimized(tokens, max_length, special_tokens)

    def _format_mm_content(self, text, image, video, prefix: str) -> List[Dict]:
        """Build one multimodal content block (prefix + optional image + optional text)."""
        return format_mm_content(
            text=text,
            image=image,
            video=video,
            prefix=prefix,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            unsupported_video_error="Video input is not supported in this AI100-only example.",
        )

    def _format_mm_instruction(self, instruction: str, query: Dict, document: Dict) -> List[Dict]:
        """Create the chat payload for one query-document pair."""
        return format_mm_instruction(
            instruction=instruction,
            query=query,
            document=document,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            unsupported_video_error="Video input is not supported in this AI100-only example.",
        )

    def _tokenize_pair(self, pair: List[Dict]) -> Dict:
        """Tokenize a query-document pair with the exact HF multimodal pipeline."""
        return tokenize_pair(self.processor, pair, self.max_length)

    def _prepare_inputs(self, tokenized_inputs: Dict, prefill_seq_len: int):
        """Prepare model inputs for dual-QPC prefill execution."""
        runtime_prompt_len = int(tokenized_inputs["input_ids"].shape[1])
        if prefill_seq_len < runtime_prompt_len:
            raise ValueError(
                f"prefill_seq_len ({prefill_seq_len}) must be >= runtime prompt length ({runtime_prompt_len})."
            )

        prepared_inputs = self.model.model.prepare_inputs_for_generation(
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

        return prepared_inputs

    def _collect_contexts(self, inputs: Dict):
        """Tokenize all docs and collect max prompt/image requirements."""
        instruction = inputs["instruction"]
        query = inputs.get("query", {})
        documents = inputs.get("documents", [])

        prepared_contexts = []
        max_prompt_len = 0
        max_grid_h = 22
        max_grid_w = 34

        for document in documents:
            pair = self._format_mm_instruction(instruction, query, document)
            tokenized = self._tokenize_pair(pair)
            runtime_prompt_len = int(tokenized["input_ids"].shape[1])

            if "image_grid_thw" in tokenized and tokenized["image_grid_thw"].numel() > 0:
                grid = tokenized["image_grid_thw"]
                max_grid_h = max(max_grid_h, int(grid[..., 1].max().item()))
                max_grid_w = max(max_grid_w, int(grid[..., 2].max().item()))

            prepared_contexts.append({"tokenized": tokenized})
            max_prompt_len = max(max_prompt_len, runtime_prompt_len)

        return prepared_contexts, max_prompt_len, max_grid_h, max_grid_w

    def get_compile_specs(self, inputs: Dict, ctx_len: int, prefill_seq_len: int = None) -> Dict[str, int]:
        """Return compile parameters required for this input batch."""
        _, max_prompt_len, max_grid_h, max_grid_w = self._collect_contexts(inputs)
        if max_prompt_len == 0:
            raise ValueError("At least one document is required for compile spec generation.")

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
        """Create zero-valued placeholders matching vision output buffers."""
        return {name: np.zeros_like(value) for name, value in vision_outputs.items()}

    def _run_ai100_vision(self, prepared_inputs: Dict, vision_qpc_path: str) -> Dict[str, np.ndarray]:
        """Run the compiled vision encoder QPC and return retained-state buffers."""
        if "pixel_values" not in prepared_inputs or "image_grid_thw" not in prepared_inputs:
            raise ValueError("Missing pixel_values/image_grid_thw for vision execution.")

        vision_session = QAICInferenceSession(vision_qpc_path)
        vision_outputs = vision_session.run(
            {
                "pixel_values": prepared_inputs["pixel_values"].detach().cpu().numpy().astype(np.float16),
                "image_grid_thw": prepared_inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64),
            }
        )
        vision_session.deactivate()
        return vision_outputs

    def _run_ai100_prefill(
        self,
        prepared_inputs: Dict,
        vision_template: Dict[str, np.ndarray],
        lang_qpc_path: str,
        vision_qpc_path: str,
    ) -> np.ndarray:
        """Run one prefill pass on AI100 language QPC and return logits."""
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

        if "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
            vision_outputs = self._run_ai100_vision(prepared_inputs, vision_qpc_path=vision_qpc_path)
        else:
            vision_outputs = self._zero_vision_outputs(vision_template)

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
        return outputs["logits"]

    def process(self, inputs: Dict, qpc_paths: Dict[str, str], prefill_seq_len: int) -> List[float]:
        """Score all documents for one query on AI100 using precompiled QPCs."""
        prepared_contexts, max_prompt_len, _, _ = self._collect_contexts(inputs)
        if max_prompt_len == 0:
            return []

        target_prefill_seq_len = int(prefill_seq_len)
        if target_prefill_seq_len < max_prompt_len:
            raise ValueError(
                f"prefill_seq_len ({target_prefill_seq_len}) must be >= max runtime prompt length ({max_prompt_len})."
            )

        if "vision_qpc_path" not in qpc_paths or "lang_qpc_path" not in qpc_paths:
            raise ValueError("qpc_paths must contain 'vision_qpc_path' and 'lang_qpc_path'.")

        prepared_contexts_with_prefill = []
        vision_template = None
        for ctx in prepared_contexts:
            prepared_inputs = self._prepare_inputs(ctx["tokenized"], prefill_seq_len=target_prefill_seq_len)
            prepared_contexts_with_prefill.append({"prepared_inputs": prepared_inputs})

            if vision_template is None and "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
                vision_template = self._run_ai100_vision(prepared_inputs, vision_qpc_path=qpc_paths["vision_qpc_path"])

        if vision_template is None:
            raise ValueError("At least one image document is required to initialize AI100 vision buffers.")

        scores = []
        for ctx in prepared_contexts_with_prefill:
            logits = self._run_ai100_prefill(
                ctx["prepared_inputs"],
                vision_template=vision_template,
                lang_qpc_path=qpc_paths["lang_qpc_path"],
                vision_qpc_path=qpc_paths["vision_qpc_path"],
            )
            score = self._score_from_logits(logits, self.yes_token_id, self.no_token_id)
            scores.append(score)

        return scores
