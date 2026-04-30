# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_CTX_LEN = 2048
DEFAULT_NUM_CORES = 16
DEFAULT_NUM_DEVICES = 1
DEFAULT_INSTRUCTION = "Represent the user's input."
DEFAULT_NUM_HIDDEN_LAYERS = 36
DEFAULT_VISION_DEPTH = 27
DEFAULT_DEEPSTACK_INDEX = None

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR


class QEffQwen3VLEmbedder:
    @staticmethod
    def _resolve_model_source(model_name_or_path: str) -> str:
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        return snapshot_download(repo_id=model_name_or_path)

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        ctx_len: int = DEFAULT_CTX_LEN,
        num_cores: int = DEFAULT_NUM_CORES,
        num_devices: int = DEFAULT_NUM_DEVICES,
        mxfp6_matmul: bool = False,
        compile_prefill_seq_len: Optional[int] = None,
        num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS,
        vision_depth: int = DEFAULT_VISION_DEPTH,
        deepstack_index: Optional[int] = DEFAULT_DEEPSTACK_INDEX,
    ):
        self.model_name_or_path = model_name_or_path
        self.model_source = self._resolve_model_source(model_name_or_path)
        self.ctx_len = ctx_len
        self.num_cores = num_cores
        self.num_devices = num_devices
        self.mxfp6_matmul = mxfp6_matmul
        self.compile_prefill_seq_len = compile_prefill_seq_len

        config = AutoConfig.from_pretrained(self.model_source, trust_remote_code=True, padding=True)
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

        # Enable optional hidden-state export from the QEff Qwen3-VL decoder.
        config.export_embedding = True

        self.processor = AutoProcessor.from_pretrained(self.model_source, trust_remote_code=True, padding=True)
        self.model = QEFFAutoModelForImageTextToText.from_pretrained(
            self.model_source,
            kv_offload=True,
            trust_remote_code=True,
            config=config,
            qaic_config={"export_embedding": True},
        )

        self._compiled_qpc_paths = None
        self._compiled_prefill_seq_len = None
        self._compiled_height = None
        self._compiled_width = None

    @staticmethod
    def _normalize_instruction(instruction: str) -> str:
        instruction = instruction.strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction += "."
        return instruction

    def format_model_input(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        resolved_instruction = self._normalize_instruction(instruction or DEFAULT_INSTRUCTION)

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

    def _tokenize_conversation(self, conversation: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        conversations = [conversation]
        text = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

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

        inputs = self.processor(
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

    @staticmethod
    def _prepare_qeff_inputs(qeff_model, tokenized_inputs: Dict[str, torch.Tensor], prefill_seq_len: int):
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

        return prepared_inputs

    @staticmethod
    def _zero_vision_outputs(vision_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {name: np.zeros_like(value) for name, value in vision_outputs.items()}

    @staticmethod
    def _run_ai100_vision(vision_qpc_path: str, prepared_inputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        vision_session = QAICInferenceSession(vision_qpc_path)
        vision_inputs = {
            "pixel_values": prepared_inputs["pixel_values"].detach().cpu().numpy().astype(np.float16),
            "image_grid_thw": prepared_inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64),
        }
        vision_outputs = vision_session.run(vision_inputs)
        vision_session.deactivate()
        return vision_outputs

    @staticmethod
    def _run_ai100_prefill(
        qpc_paths: Dict[str, str],
        prepared_inputs: Dict[str, torch.Tensor],
        vision_outputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        lang_qpc_path = qpc_paths.get("lang_qpc_path")
        if lang_qpc_path is None:
            raise ValueError("Missing lang_qpc_path in compiled QPC outputs.")

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
        lang_inputs = {
            "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
            "position_ids": position_ids.detach().cpu().numpy().astype(np.int64),
            "image_idx": np.zeros((1, 1), dtype=np.int64),
        }

        outputs = lang_session.run(lang_inputs)
        lang_session.deactivate()

        if "embedding_output" not in outputs:
            raise KeyError(
                "Missing 'embedding_output' in AI100 decoder outputs. Ensure export_embedding is enabled in config/qaic_config."
            )
        embedding_output = outputs["embedding_output"]
        if embedding_output.ndim > 2:
            embedding_output = embedding_output.reshape(embedding_output.shape[0], -1)
        return embedding_output

    def _compile_if_needed(self, tokenized_inputs_list: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, str], int]:
        max_prompt_len = 0
        max_grid_h = 22
        max_grid_w = 34

        for tokenized in tokenized_inputs_list:
            max_prompt_len = max(max_prompt_len, int(tokenized["input_ids"].shape[1]))
            if "image_grid_thw" in tokenized and tokenized["image_grid_thw"].numel() > 0:
                grid = tokenized["image_grid_thw"]
                max_grid_h = max(max_grid_h, int(grid[..., 1].max().item()))
                max_grid_w = max(max_grid_w, int(grid[..., 2].max().item()))

        effective_prefill = (
            max_prompt_len if self.compile_prefill_seq_len is None else int(self.compile_prefill_seq_len)
        )
        if effective_prefill < max_prompt_len:
            raise ValueError(
                f"compile_prefill_seq_len ({effective_prefill}) must be >= max runtime prompt length ({max_prompt_len})."
            )

        patch_size = int(self.model.model.config.vision_config.patch_size)
        compile_height = max_grid_h * patch_size
        compile_width = max_grid_w * patch_size

        if (
            self._compiled_qpc_paths is not None
            and self._compiled_prefill_seq_len == effective_prefill
            and self._compiled_height == compile_height
            and self._compiled_width == compile_width
        ):
            return self._compiled_qpc_paths, effective_prefill

        qpc_paths = self.model.compile(
            img_size=max(compile_height, compile_width),
            height=compile_height,
            width=compile_width,
            prefill_seq_len=effective_prefill,
            ctx_len=self.ctx_len,
            num_devices=self.num_devices,
            num_cores=self.num_cores,
            mxfp6_matmul=self.mxfp6_matmul,
        )

        self._compiled_qpc_paths = qpc_paths
        self._compiled_prefill_seq_len = effective_prefill
        self._compiled_height = compile_height
        self._compiled_width = compile_width
        return qpc_paths, effective_prefill

    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        conversations = [
            self.format_model_input(
                text=entry.get("text"),
                image=entry.get("image"),
                video=entry.get("video"),
                instruction=entry.get("instruction"),
            )
            for entry in inputs
        ]

        tokenized_inputs_list = [self._tokenize_conversation(conversation) for conversation in conversations]
        qpc_paths, prefill_seq_len = self._compile_if_needed(tokenized_inputs_list)

        prepared_inputs_list = [
            self._prepare_qeff_inputs(self.model, tokenized_inputs, prefill_seq_len=prefill_seq_len)
            for tokenized_inputs in tokenized_inputs_list
        ]

        vision_template = None
        for prepared_inputs in prepared_inputs_list:
            if "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
                vision_template = self._run_ai100_vision(qpc_paths["vision_qpc_path"], prepared_inputs)
                break

        if vision_template is None:
            raise ValueError("At least one input with an image is required to initialize the vision path.")

        embedding_rows = []
        for prepared_inputs in prepared_inputs_list:
            if "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
                vision_outputs = self._run_ai100_vision(qpc_paths["vision_qpc_path"], prepared_inputs)
            else:
                vision_outputs = self._zero_vision_outputs(vision_template)

            embedding_output = self._run_ai100_prefill(
                qpc_paths=qpc_paths,
                prepared_inputs=prepared_inputs,
                vision_outputs=vision_outputs,
            )
            embedding_rows.append(torch.from_numpy(embedding_output).to(torch.float32))

        embeddings = torch.cat(embedding_rows, dim=0)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL-Embedding AI100 inference")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--num-cores", type=int, default=DEFAULT_NUM_CORES)
    parser.add_argument("--num-devices", type=int, default=DEFAULT_NUM_DEVICES)
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--compile-prefill-seq-len", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=DEFAULT_NUM_HIDDEN_LAYERS)
    parser.add_argument("--vision-depth", type=int, default=DEFAULT_VISION_DEPTH)
    parser.add_argument("--deepstack-index", type=int, default=DEFAULT_DEEPSTACK_INDEX)
    return parser.parse_args()


def main():
    args = parse_args()

    queries = [
        {"text": "A woman playing with her dog on a beach at sunset."},
        {"text": "Pet owner training dog outdoors near water."},
        {"text": "Woman surfing on waves during a sunny day."},
        {"text": "City skyline view from a high-rise building at night."},
    ]

    documents = [
        {
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
            "as the dog offers its paw in a heartwarming display of companionship and trust."
        },
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        {
            "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, "
            "as the dog offers its paw in a heartwarming display of companionship and trust.",
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        },
    ]

    embedder = QEffQwen3VLEmbedder(
        model_name_or_path=args.model_name,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=args.mxfp6_matmul,
        compile_prefill_seq_len=args.compile_prefill_seq_len,
        num_hidden_layers=args.num_hidden_layers,
        vision_depth=args.vision_depth,
        deepstack_index=args.deepstack_index,
    )

    model_inputs = queries + documents
    embeddings = embedder.process(model_inputs)

    q_count = len(queries)
    similarity_scores = embeddings[:q_count] @ embeddings[q_count:].T
    print(similarity_scores.tolist())


if __name__ == "__main__":
    main()
