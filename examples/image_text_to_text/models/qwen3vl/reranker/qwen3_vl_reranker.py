# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_CTX_LEN = 2048
DEFAULT_NUM_CORES = 16
DEFAULT_NUM_DEVICES = 1

# Max token budget used by this example's manual truncation/padding flow.
MAX_LENGTH = 8192
# Pixel constraints used by Qwen3-VL preprocessing.
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1280 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1.0


class QEffQwen3VLReranker:
    @staticmethod
    def _resolve_model_source(model_name_or_path: str) -> str:
        """Return a local model path when given an HF repo id.

        Why:
        Some transformers versions can fail when resolving chat templates from
        repo-id mode for this model. Using a local snapshot path avoids that path.
        """
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
        compile_prefill_seq_len: int = None,
    ):
        """Initialize the AI100-only reranker wrapper.

        This loads:
        - HF config/processor for prompt and multimodal preprocessing.
        - QEFF dual-QPC model wrapper (vision encoder + language decoder).
        - Token ids for "yes"/"no" used to compute reranker scores.

        Parameters
        ----------
        model_name_or_path:
            HF model id or local snapshot path.
        """
        self.model_name_or_path = model_name_or_path
        self.model_source = self._resolve_model_source(model_name_or_path)
        self.ctx_len = ctx_len
        self.num_cores = num_cores
        self.num_devices = num_devices
        self.mxfp6_matmul = mxfp6_matmul
        self.compile_prefill_seq_len = compile_prefill_seq_len
        self.max_length = MAX_LENGTH
        self.fps = FPS

        # Use local snapshot for stable processor/chat-template loading.
        config = AutoConfig.from_pretrained(self.model_source, trust_remote_code=True, padding=True)
        if hasattr(config, "use_cache"):
            config.use_cache = True
        if hasattr(config, "text_config") and hasattr(config.text_config, "use_cache"):
            config.text_config.use_cache = True

        self.processor = AutoProcessor.from_pretrained(self.model_source, trust_remote_code=True, padding=True)
        self.model = QEFFAutoModelForImageTextToText.from_pretrained(
            self.model_source,
            kv_offload=True,
            trust_remote_code=True,
            config=config,
        )

        self.yes_token_id, self.no_token_id = self._get_yes_no_token_ids(self.processor.tokenizer)
        self._compiled_qpc_paths = None
        self._compiled_prefill_seq_len = 0
        self._compiled_height = None
        self._compiled_width = None

    @staticmethod
    def _get_yes_no_token_ids(tokenizer) -> Tuple[int, int]:
        """Resolve tokenizer ids for the exact tokens 'yes' and 'no'."""
        vocab = tokenizer.get_vocab()
        if "yes" not in vocab or "no" not in vocab:
            raise ValueError("Could not resolve tokenizer ids for exact tokens 'yes' and 'no'.")
        return vocab["yes"], vocab["no"]

    @staticmethod
    def _score_from_logits(logits, yes_token_id: int, no_token_id: int) -> float:
        """Convert model logits into a reranker relevance score.

        Score formula:
            sigmoid(logit_yes - logit_no)
        """
        # Convert runtime output to torch and use final-token logits.
        logits_tensor = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits.detach().cpu()
        if logits_tensor.ndim == 3:
            logits_tensor = logits_tensor[:, -1, :]
        # Binary relevance score from yes/no logit gap.
        score = torch.sigmoid(logits_tensor[:, yes_token_id] - logits_tensor[:, no_token_id])
        return float(score[0].item())

    @staticmethod
    def _truncate_tokens_optimized(tokens: List[int], max_length: int, special_tokens: List[int]) -> List[int]:
        """Truncate while preserving all special tokens in sequence order."""
        if len(tokens) <= max_length:
            return tokens

        # Preserve all special/control tokens and trim only non-special tokens.
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

    def _format_mm_content(self, text, image, video, prefix: str) -> List[Dict]:
        """Build one multimodal content block (prefix + optional image + optional text)."""
        # Prefix helps the model distinguish query vs document sections.
        content = [{"type": "text", "text": prefix}]

        if not text and not image and not video:
            content.append({"type": "text", "text": "NULL"})
            return content

        if video:
            raise ValueError("Video input is not supported in this AI100-only example.")

        if image:
            # Convert local paths to file:// URIs for the processor.
            if isinstance(image, str):
                image_content = image if image.startswith(("http", "oss")) else "file://" + image
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

        return content

    def _format_mm_instruction(self, instruction: str, query: Dict, document: Dict) -> List[Dict]:
        """Create the chat payload for one query-document pair."""
        # Prompt shape follows the HF reranker reference format.
        contents = [{"type": "text", "text": "<Instruct>: " + instruction}]

        contents.extend(
            self._format_mm_content(
                query.get("text"),
                query.get("image"),
                query.get("video"),
                prefix="<Query>:",
            )
        )
        contents.extend(
            self._format_mm_content(
                document.get("text"),
                document.get("image"),
                document.get("video"),
                prefix="\n<Document>:",
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

    def _tokenize_pair(self, pair: List[Dict]) -> Dict:
        """Tokenize a query-document pair with the exact HF multimodal pipeline."""
        # Processor expects list-of-conversations.
        pairs = [pair]
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)

        # Build image/video tensors + metadata for processor inputs.
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

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs,
        )

        # Apply custom truncation preserving trailing template control tokens.
        for i, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = (
                self._truncate_tokens_optimized(
                    input_ids[:-5],
                    self.max_length,
                    self.processor.tokenizer.all_special_ids,
                )
                + input_ids[-5:]
            )

        # Re-pad through tokenizer utilities so masks align with token ids.
        padded = self.processor.tokenizer.pad(
            {"input_ids": inputs["input_ids"]},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        for key in padded:
            inputs[key] = padded[key]

        if "pixel_values" in inputs:
            # Keep pixels fp32 before explicit cast to fp16 during vision run.
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

        return inputs

    def _prepare_inputs(self, tokenized_inputs: Dict, prefill_seq_len: int = None):
        """Prepare model inputs for dual-QPC prefill execution."""
        # True prompt length before compile-aligned padding.
        runtime_prompt_len = int(tokenized_inputs["input_ids"].shape[1])
        effective_prefill = runtime_prompt_len if prefill_seq_len is None else prefill_seq_len
        if effective_prefill < runtime_prompt_len:
            raise ValueError(
                f"prefill_seq_len ({effective_prefill}) must be >= runtime prompt length ({runtime_prompt_len})."
            )

        # Let model helper compute position_ids and multimodal placement.
        prepared_inputs = self.model.model.prepare_inputs_for_generation(
            inputs=tokenized_inputs,
            prefill_seq_len=effective_prefill,
            batch_size=1,
        )

        # Normalize image_grid_thw to the shape consumed by compiled path.
        if "image_grid_thw" in prepared_inputs and prepared_inputs["image_grid_thw"].ndim == 2:
            thw = prepared_inputs["image_grid_thw"][0]
            t, h, w = int(thw[0].item()), int(thw[1].item()), int(thw[2].item())
            prepared_inputs["image_grid_thw"] = torch.zeros((1, t, h, w), dtype=thw.dtype)

        if "pixel_values" in prepared_inputs:
            prepared_inputs["pixel_values"] = prepared_inputs["pixel_values"].to(torch.float32)

        return prepared_inputs, runtime_prompt_len

    def _ensure_compiled(self, prefill_seq_len: int, height: int, width: int):
        """Compile QPCs if needed, otherwise reuse cached compiled artifacts."""
        # Reuse previously compiled artifacts whenever shapes are compatible.
        if (
            self._compiled_qpc_paths is not None
            and prefill_seq_len <= self._compiled_prefill_seq_len
            and height == self._compiled_height
            and width == self._compiled_width
        ):
            return

        reuse_vision_qpc = (
            self._compiled_qpc_paths is not None and height == self._compiled_height and width == self._compiled_width
        )

        # Compile one max prefill specialization and optionally skip vision recompile.
        compiled_paths = self.model.compile(
            prefill_seq_len=prefill_seq_len,
            ctx_len=self.ctx_len,
            img_size=max(height, width),
            height=height,
            width=width,
            num_cores=self.num_cores,
            num_devices=self.num_devices,
            mxfp6_matmul=self.mxfp6_matmul,
            # vision_embed_fp32=True,
            skip_vision=reuse_vision_qpc,
        )
        if reuse_vision_qpc:
            compiled_paths["vision_qpc_path"] = self._compiled_qpc_paths["vision_qpc_path"]

        self._compiled_qpc_paths = compiled_paths
        self._compiled_prefill_seq_len = prefill_seq_len
        self._compiled_height = height
        self._compiled_width = width

    @staticmethod
    def _zero_vision_outputs(vision_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create zero-valued placeholders matching vision output buffers."""
        return {name: np.zeros_like(value) for name, value in vision_outputs.items()}

    def _run_ai100_vision(self, prepared_inputs) -> Dict[str, np.ndarray]:
        """Run the compiled vision encoder QPC and return retained-state buffers."""
        if "pixel_values" not in prepared_inputs or "image_grid_thw" not in prepared_inputs:
            raise ValueError("Missing pixel_values/image_grid_thw for vision execution.")

        # Vision session produces retained states consumed by language session.
        vision_session = QAICInferenceSession(self._compiled_qpc_paths["vision_qpc_path"])
        vision_outputs = vision_session.run(
            {
                # Vision qpc expects fp16 pixels + int64 grid coordinates.
                "pixel_values": prepared_inputs["pixel_values"].detach().cpu().numpy().astype(np.float16),
                "image_grid_thw": prepared_inputs["image_grid_thw"].detach().cpu().numpy().astype(np.int64),
            }
        )
        vision_session.deactivate()
        return vision_outputs

    def _run_ai100_prefill(self, prepared_inputs, vision_template: Dict[str, np.ndarray]) -> np.ndarray:
        """Run one prefill pass on AI100 language QPC and return logits."""
        # Match runtime input to compiled prefill length.
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

        # For text-only docs, inject zeroed retained states with matching shapes.
        if "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
            vision_outputs = self._run_ai100_vision(prepared_inputs)
        else:
            vision_outputs = self._zero_vision_outputs(vision_template)

        # Skip past/retained buffers and run only required prefill inputs.
        lang_session = QAICInferenceSession(self._compiled_qpc_paths["lang_qpc_path"])
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
                # image_idx selects the vision buffer slot for this request.
                "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
                "position_ids": position_ids.detach().cpu().numpy().astype(np.int64),
                "image_idx": np.zeros((1, 1), dtype=np.int64),
            }
        )
        lang_session.deactivate()
        return outputs["logits"]

    def process(self, inputs: Dict) -> List[float]:
        """Score all documents for one query on AI100.

        High-level flow:
        1) Build model-ready query-document pairs.
        2) Find max prompt/image shape across all docs.
        3) Compile once at max shape (single stable specialization).
        4) Run prefill per doc and convert logits -> score.
        """
        # Unpack user payload.
        instruction = inputs["instruction"]
        query = inputs.get("query", {})
        documents = inputs.get("documents", [])

        # Collect per-document tokenized contexts first so we can compile once
        # with the largest prompt/image shape required by this request.
        prepared_contexts = []
        max_prompt_len = 0
        max_grid_h = 22
        max_grid_w = 34

        # Build each pair in the exact chat-template format expected by the model.
        for document in documents:
            pair = self._format_mm_instruction(instruction, query, document)
            tokenized = self._tokenize_pair(pair)
            runtime_prompt_len = int(tokenized["input_ids"].shape[1])

            # Track the max image grid (H, W) seen so compile dimensions can
            # handle all documents in this batch.
            if "image_grid_thw" in tokenized and tokenized["image_grid_thw"].numel() > 0:
                grid = tokenized["image_grid_thw"]
                max_grid_h = max(max_grid_h, int(grid[..., 1].max().item()))
                max_grid_w = max(max_grid_w, int(grid[..., 2].max().item()))

            prepared_contexts.append(
                {
                    "tokenized": tokenized,
                    "runtime_prompt_len": runtime_prompt_len,
                }
            )
            max_prompt_len = max(max_prompt_len, runtime_prompt_len)

        # Empty documents list => no scores.
        if max_prompt_len == 0:
            return []

        # Convert max grid to compile-time pixel dimensions using model patch size.
        patch_size = int(self.model.model.config.vision_config.patch_size)
        compile_height = max_grid_h * patch_size
        compile_width = max_grid_w * patch_size

        # Compile/reuse a single language specialization and prepare all requests
        # to that same prefill length to avoid per-document recompiles.
        target_prefill_seq_len = max_prompt_len
        if self.compile_prefill_seq_len is not None:
            if self.compile_prefill_seq_len < max_prompt_len:
                raise ValueError(
                    f"--compile-prefill-seq-len ({self.compile_prefill_seq_len}) must be >= "
                    f"max runtime prompt length ({max_prompt_len})."
                )
            target_prefill_seq_len = self.compile_prefill_seq_len

        self._ensure_compiled(target_prefill_seq_len, compile_height, compile_width)

        # Prepare all documents to the same prefill length used at compile time.
        prepared_contexts_with_prefill = []
        vision_template = None
        for ctx in prepared_contexts:
            prepared_inputs, _ = self._prepare_inputs(ctx["tokenized"], prefill_seq_len=target_prefill_seq_len)
            prepared_contexts_with_prefill.append({"prepared_inputs": prepared_inputs})

            # Capture one real vision-output template so text-only docs can reuse
            # zero-valued buffers with exact matching shapes.
            if vision_template is None and "pixel_values" in prepared_inputs and "image_grid_thw" in prepared_inputs:
                vision_template = self._run_ai100_vision(prepared_inputs)

        # This example currently expects at least one image document to establish
        # retained-state buffer shapes for mixed image/text batches.
        if vision_template is None:
            raise ValueError("At least one image document is required to initialize AI100 vision buffers.")

        # Run language prefill and compute scalar score per document.
        scores = []
        for ctx in prepared_contexts_with_prefill:
            logits = self._run_ai100_prefill(
                ctx["prepared_inputs"],
                vision_template=vision_template,
            )
            # Reranker score = sigmoid(logit_yes - logit_no).
            score = self._score_from_logits(logits, self.yes_token_id, self.no_token_id)
            scores.append(score)

        return scores


def main():
    # Keep CLI simple: just allow model id/path override.
    parser = argparse.ArgumentParser(description="Qwen3-VL reranker example.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN, help="Context length used at compile time.")
    parser.add_argument("--num-cores", type=int, default=DEFAULT_NUM_CORES, help="Number of AI100 cores.")
    parser.add_argument("--num-devices", type=int, default=DEFAULT_NUM_DEVICES, help="Number of AI100 devices.")
    parser.add_argument(
        "--mxfp6-matmul",
        action="store_true",
        help="Enable MXFP6 matmul during compile (default: disabled).",
    )
    parser.add_argument(
        "--compile-prefill-seq-len",
        type=int,
        default=None,
        help=(
            "Optional fixed prefill sequence length for compile/padding. "
            "Must be >= max prompt length of the current request."
        ),
    )
    args = parser.parse_args()

    model = QEffQwen3VLReranker(
        model_name_or_path=args.model_name,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        mxfp6_matmul=args.mxfp6_matmul,
        compile_prefill_seq_len=args.compile_prefill_seq_len,
    )

    # Example input payload matching the HF reranker schema.
    inputs = {
        "instruction": "Retrieve images or text relevant to the user's query.",
        "query": {"text": "A woman playing with her dog on a beach at sunset."},
        "documents": [
            {
                "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."
            },
            {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {
                "text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
        ],
        "fps": 1.0,
    }

    # Print one score per document in the same order as inputs["documents"].
    scores = model.process(inputs)
    print(scores)


if __name__ == "__main__":
    main()
