# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Vision Handler for Vision-Language Models

This module provides the VisionHandler class that encapsulates all vision model
operations, separating them from the main text generation logic.
"""

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils import constants
from QEfficient.utils.logging_utils import logger


class VisionHandler:
    """
    Handles all vision model operations for vision-language models.

    This class encapsulates vision preprocessing, inference, and output handling,
    providing a clean separation between vision and language processing.
    """

    def __init__(
        self,
        qeff_model: Optional[QAICInferenceSession],
        vision_session: Optional[QAICInferenceSession],
        processor: Optional[AutoImageProcessor],
        tokenizer: Optional[AutoTokenizer],
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        lang_session: Optional[QAICInferenceSession] = None,
    ):
        """
        Initialize vision handler

        Args:
            vision_session: QAICInferenceSession for vision model
            processor: AutoImageProcessor for image preprocessing
            tokenizer: AutoTokenizer for text tokenization
            image_height: Desired image height for resizing
            image_width: Desired image width for resizing
            config: Configuration dictionary with vision model parameters
            lang_session: Optional language session for coordination (to avoid resource conflicts)
        """
        self._qeff_model = qeff_model
        self._vision_session = vision_session
        self._processor = processor
        self._tokenizer = tokenizer
        self._image_height = image_height
        self._image_width = image_width
        self._config = config or {}
        self._lang_session = lang_session  # Store language session for coordination

        # Cache for vision output shapes
        self._vision_output_shapes = None

        if self._vision_session and not self._processor:
            logger.warning("Vision session provided but no processor. Vision functionality may be limited.")

    def is_available(self) -> bool:
        """
        Check if vision processing is available

        Returns:
            True if both vision session and processor are available
        """
        return self._vision_session is not None and self._processor is not None

    def prepare_internVL_inputs(self, img_url: str, prompt: str) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for InternVL model

        Args:
            image_url: URL or path to image
            prompt: Text query to process with image
        """
        if not self._tokenizer:
            raise ValueError("Tokenizer is required for InternVL input preparation")
        pixel_values = []
        num_patches_list = []
        questions = []
        img = requests.get(img_url, stream=True)
        image = Image.open(BytesIO(img.content)).convert("RGB")

        if self._image_height and self._image_width:
            image = image.resize((self._image_height, self._image_width))
        else:
            logger.warning("Height and Width not specified. Using default image size for num_patches = 13.")
            image = image.resize((constants.INTERN_IMAGE_HEIGHT, constants.INTERN_IMAGE_WIDTH))

        # preprocess the resized image
        pixel_value = self._processor.load_image(image, max_num=12)
        num_patches_list.append(pixel_value.shape[0])
        pixel_values.append(pixel_value)

        question = "<image>\n" + prompt
        questions.append(question)

        pixel_values = torch.cat(pixel_values, dim=0)

        # Chat Template information for prompt preprocessing
        messages: List[List[str]] = []
        roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
        prompt = self._processor(pixel_values, questions, messages, roles, num_patches_list=num_patches_list)

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs["pixel_values"] = pixel_values.clone()

        # Convert to numpy arrays
        vision_inputs = {}
        for k, v in inputs.items():
            if k in {
                "pixel_values",
                "image_masks",
                "image_input_idx",
                "valid_idx",
                "aspect_ratio_ids",
                "aspect_ratio_mask",
            }:
                vision_inputs[k] = np.array(v)

        # Convert specific inputs to float16
        vision_inputs_fp16 = {"pixel_values", "image_masks"}
        for k in vision_inputs_fp16:
            if k in vision_inputs:
                vision_inputs[k] = vision_inputs[k].astype("float16")

        lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}

        return vision_inputs, lang_inputs

    def prepare_molmo_inputs(self, image_url: str, query: str) -> Dict[str, np.ndarray]:
        """
        Download and preprocess image into model inputs
        Args:
            image_url: URL or path to image
            query: Text query to process with image
        Returns:
            Dictionary of vision model inputs
        Raises:
            ValueError: If vision handler is not properly initialized
            RuntimeError: If image processing fails
        """
        if not self.is_available():
            raise ValueError("Vision handler not properly initialized. Need both vision_session and processor.")

        try:
            # Download image
            if image_url.startswith(("http://", "https://")):
                image = Image.open(requests.get(image_url, stream=True).raw)
            else:
                image = Image.open(image_url)
            image = image.resize((constants.MOLMO_IMAGE_HEIGHT, constants.MOLMO_IMAGE_WIDTH))
            inputs = self._processor.process(images=[image], text=query)
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
            inputs["attention_mask"] = torch.ones((inputs["input_ids"].shape), dtype=torch.int64)
            valid = inputs["image_input_idx"] > 0
            valid = valid.reshape(1, -1)
            inputs["valid_idx"] = torch.nonzero(valid)[:, 1].unsqueeze(0)
            inputs["pixel_values"] = inputs.pop("images")

            # Convert to numpy arrays
            vision_inputs = {}
            for k, v in inputs.items():
                if k in {
                    "pixel_values",
                    "image_masks",
                    "image_input_idx",
                    "valid_idx",
                    "aspect_ratio_ids",
                    "aspect_ratio_mask",
                }:
                    vision_inputs[k] = np.array(v)

            # Convert specific inputs to float16
            vision_inputs_fp16 = {"pixel_values", "image_masks"}
            for k in vision_inputs_fp16:
                if k in vision_inputs:
                    vision_inputs[k] = vision_inputs[k].astype("float16")

            lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}

            return vision_inputs, lang_inputs
        except Exception as e:
            raise RuntimeError(f"Failed to process image {image_url}: {str(e)}")

    def prepare_vlm_inputs(self, image_url: str, query: str, prefill_seq_len: int) -> Dict[str, np.ndarray]:
        """
        Download and preprocess image into model inputs

        Args:
            image_url: URL or path to image
            query: Text query to process with image
            prefill_seq_len: Padded sequence length for language model

        Returns:
            Dictionary of vision model inputs

        Raises:
            ValueError: If vision handler is not properly initialized
            RuntimeError: If image processing fails
        """
        if not self.is_available():
            raise ValueError("Vision handler not properly initialized. Need both vision_session and processor.")

        try:
            # Download image
            if image_url.startswith(("http://", "https://")):
                image = Image.open(requests.get(image_url, stream=True).raw)
            else:
                image = Image.open(image_url)

            if self._image_height and self._image_width:
                image = image.resize((self._image_width, self._image_height))
            else:
                logger.warning("Height and Width not specified. Using default image size.")
                if "mistral3" in self._qeff_model.model.config.model_type:
                    image = image.resize((constants.MISTRAL3_IMAGE_HEIGHT, constants.MISTRAL3_IMAGE_WIDTH))
                if "llava_next" in self._qeff_model.model.config.model_type:
                    image = image.resize(
                        (constants.GRANITEVISION_IMG_SIZE_HEIGHT, constants.GRANITEVISION_IMG_SIZE_WIDTH)
                    )

            # Prepare conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image"},
                    ],
                },
            ]

            # Apply chat template
            prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Process image and text
            inputs = self._processor(images=image, text=prompt, return_tensors="pt")

            if (
                hasattr(self._qeff_model.model.config, "model_type")
                and self._qeff_model.model.config.model_type == "qwen2_5_vl"
            ):
                inputs = self._qeff_model.model.prepare_inputs_for_generation(
                    inputs=inputs, prefill_seq_len=prefill_seq_len, batch_size=inputs["input_ids"].shape[0]
                )

            # Convert to float32 if needed
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

            # Convert to numpy arrays
            vision_inputs = {}
            for k, v in inputs.items():
                if k in {
                    "pixel_values",
                    "image_masks",
                    "image_input_idx",
                    "valid_idx",
                    "aspect_ratio_ids",
                    "aspect_ratio_mask",
                }:
                    vision_inputs[k] = np.array(v)

            # Convert specific inputs to float16
            vision_inputs_fp16 = {"pixel_values", "image_masks"}
            for k in vision_inputs_fp16:
                if k in vision_inputs:
                    vision_inputs[k] = vision_inputs[k].astype("float16")

            lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}

            return vision_inputs, lang_inputs

        except Exception as e:
            raise RuntimeError(f"Failed to process image {image_url}: {str(e)}")

    def run_vision_inference(self, vision_inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute vision model inference with session coordination

        Args:
            vision_inputs: Preprocessed vision inputs

        Returns:
            Vision embeddings and metadata

        Raises:
            ValueError: If vision session is not available
            RuntimeError: If inference fails
        """
        if not self._vision_session:
            raise ValueError("Vision session not available")

        lang_was_active = False
        try:
            # Coordinate with language session to avoid resource conflicts
            if self._lang_session and self._lang_session.is_active:
                logger.debug("Deactivating language session before vision inference")
                self._lang_session.deactivate()
                lang_was_active = True

            # Activate vision session
            logger.debug("Activating vision session for inference")
            self._vision_session.activate()

            # Run inference
            vision_outputs = self._vision_session.run(vision_inputs)

            # Deactivate vision session
            logger.debug("Deactivating vision session after inference")
            self._vision_session.deactivate()

            # Reactivate language session if it was active before
            if lang_was_active and self._lang_session:
                logger.debug("Reactivating language session after vision inference")
                self._lang_session.activate()

            return vision_outputs

        except Exception as e:
            # Ensure proper cleanup on error
            if self._vision_session:
                try:
                    self._vision_session.deactivate()
                except Exception:
                    logger.warning("Deactivating vision session failed")

            # Restore language session if needed
            if lang_was_active and self._lang_session:
                try:
                    self._lang_session.activate()
                except Exception:
                    logger.warning("Deactivating language session failed")

            raise RuntimeError(f"Vision inference failed: {str(e)}")

    def get_vision_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get vision output dimensions from config or session

        Returns:
            Dictionary mapping output names to shapes
        """
        if self._vision_output_shapes is not None:
            return self._vision_output_shapes

        # Try to get from config first
        if self._config and "vision_output_shapes" in self._config:
            self._vision_output_shapes = self._config["vision_output_shapes"]
            return self._vision_output_shapes

        # Try to derive from vision session
        if self._vision_session:
            try:
                shapes = {}
                for output_name in self._vision_session.output_names:
                    if (
                        hasattr(self._vision_session, "bindings")
                        and output_name in self._vision_session.binding_index_map
                    ):
                        binding_idx = self._vision_session.binding_index_map[output_name]
                        if hasattr(self._vision_session.bindings[binding_idx], "dims"):
                            shapes[output_name] = tuple(self._vision_session.bindings[binding_idx].dims)

                if shapes:
                    self._vision_output_shapes = shapes
                    return shapes
            except Exception as e:
                logger.warning(f"Could not derive vision output shapes from session: {e}")

        # Fallback to default shapes (these were hard-coded in original implementation)
        default_shapes = {
            "vision_embeds": (2448, 5120)  # This should be derived from model config
        }

        logger.warning("Using default vision output shapes. Consider providing shapes in config.")
        self._vision_output_shapes = default_shapes
        return default_shapes

    def setup_vision_buffers(self):
        """
        Configure vision model output buffers

        Raises:
            ValueError: If vision session is not available
        """
        if not self._vision_session:
            raise ValueError("Vision session not available")

        try:
            shapes = self.get_vision_output_shapes()

            # Set up output buffers
            buffers = {}
            for output_name, shape in shapes.items():
                # Create placeholder with appropriate dtype
                if "vision_embeds" in output_name:
                    buffers[output_name] = np.zeros(shape, dtype=np.float16)
                else:
                    buffers[output_name] = np.zeros(shape, dtype=np.float32)

            self._vision_session.set_buffers(buffers)

        except Exception as e:
            raise RuntimeError(f"Failed to setup vision buffers: {str(e)}")

    def prepare_complete_vision_language_inputs(
        self, image_url: str, query: str
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Complete pipeline: prepare inputs and run vision inference

        Args:
            image_url: URL or path to image
            query: Text query

        Returns:
            Tuple of (vision_inputs, vision_outputs)
        """
        # Prepare vision inputs
        vision_inputs = self.prepare_vision_inputs(image_url, query)

        # Setup buffers
        self.setup_vision_buffers()

        # Run vision inference
        vision_outputs = self.run_vision_inference(vision_inputs)

        return vision_inputs, vision_outputs

    def get_processed_inputs(
        self, image_url: str, query: str, prefill_seq_len: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Process vision inputs and prepare language model inputs

        Args:
            image_url: URL or path to image
            query: Text query
            padded_len: Padded sequence length for language model

        Returns:
            Tuple of (language_inputs, vision_outputs)
        """
        if not self.is_available():
            raise ValueError("Vision handler not properly initialized")

        try:
            ## Get vlm inputs ##
            if (
                hasattr(self._qeff_model.model.config, "model_type")
                and self._qeff_model.model.config.model_type == "internvl_chat"
            ):
                vision_inputs, lang_inputs = self.prepare_internVL_inputs(image_url, query)
            elif (
                hasattr(self._qeff_model.model.config, "model_type")
                and self._qeff_model.model.config.model_type == "molmo"
            ):
                vision_inputs, lang_inputs = self.prepare_molmo_inputs(image_url, query)
            else:
                vision_inputs, lang_inputs = self.prepare_vlm_inputs(image_url, query, prefill_seq_len)

            # Handle padding for language model
            pad_token_id = 1
            input_ids_length = lang_inputs["input_ids"].shape[1]
            num_chunks = -(input_ids_length // -prefill_seq_len)
            padded_len = num_chunks * prefill_seq_len

            lang_inputs["input_ids"] = torch.nn.functional.pad(
                lang_inputs["input_ids"],
                (0, padded_len - input_ids_length),
                "constant",
                pad_token_id,
            )
            lang_inputs["attention_mask"] = torch.nn.functional.pad(
                lang_inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
            )

            if "cross_attention_mask" in lang_inputs:
                lang_inputs["cross_attention_mask"] = torch.nn.functional.pad(
                    lang_inputs["cross_attention_mask"], (0, 0, 0, 0, 0, padded_len - input_ids_length)
                )

            for k, v in lang_inputs.items():
                lang_inputs[k] = np.array(v)

            vision_outputs = {}
            if vision_inputs:
                self.setup_vision_buffers()
                vision_outputs = self.run_vision_inference(vision_inputs)

            if "position_ids" in lang_inputs:
                lang_inputs.pop("attention_mask")
            else:
                lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)

            lang_inputs["image_idx"] = np.array([[0]])

            return lang_inputs, vision_outputs, num_chunks

        except Exception as e:
            raise RuntimeError(f"Failed to process vision-language inputs: {str(e)}")
