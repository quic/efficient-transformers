# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Qwen3-VL-specific embedding helpers for AI100 runtime.

This module intentionally keeps only Qwen3-VL-specific embedding logic
(prompt construction, multimodal tokenization, and AI100 runtime orchestration
with compiled QPC paths).

Model loading (``from_pretrained``) and model compilation (``compile``) are
exposed in ``qwen3_vl_embedding.py`` so users can directly see QEff API usage.
"""

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.qwen3_vl._embedding_utils import (
    format_model_input as _shared_format_model_input,
)
from QEfficient.transformers.models.qwen3_vl._embedding_utils import (
    resolve_model_source as _shared_resolve_model_source,
)
from QEfficient.transformers.models.qwen3_vl._embedding_utils import (
    tokenize_conversation as _shared_tokenize_conversation,
)


def resolve_model_source(model_name_or_path: str) -> str:
    """Return a local model path when given an HF repo id."""
    return _shared_resolve_model_source(model_name_or_path)


class QEffQwen3VLEmbedder:
    """Qwen3-VL embedding runtime helper for AI100 compiled QPCs."""

    def __init__(self, processor, model):
        """Initialize helper with preloaded processor and QEff model."""
        self.processor = processor
        self.model = model

    def format_model_input(
        self,
        text: str = None,
        image: Any = None,
        video: Any = None,
        instruction: str = None,
    ) -> List[Dict[str, Any]]:
        """Build one chat-style multimodal input for Qwen3-VL embedding."""
        return _shared_format_model_input(
            text=text,
            image=image,
            video=video,
            instruction=instruction,
        )

    def _tokenize_conversation(self, conversation: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Tokenize one chat conversation with multimodal processing."""
        return _shared_tokenize_conversation(self.processor, conversation)

    @staticmethod
    def _prepare_qeff_inputs(qeff_model, tokenized_inputs: Dict[str, torch.Tensor], prefill_seq_len: int):
        """Prepare model inputs for dual-QPC prefill execution."""
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
        """Tokenize all inputs and collect max prompt/image requirements."""
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
        """Return compile parameters required for this input batch."""
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
        """Create zero-valued placeholders matching vision output buffers."""
        return {name: np.zeros_like(value) for name, value in vision_outputs.items()}

    @staticmethod
    def _run_ai100_vision(vision_qpc_path: str, prepared_inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Run the compiled vision encoder QPC and return retained-state buffers."""
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
        """Run one prefill pass on AI100 language QPC and return embedding output."""
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
        """Generate embeddings on AI100 using precompiled QPCs."""
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
