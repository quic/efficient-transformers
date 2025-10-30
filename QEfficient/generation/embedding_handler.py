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

from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.logging_utils import logger


class VisionHandler:
    """
    Handles all vision model operations for vision-language models.

    This class encapsulates vision preprocessing, inference, and output handling,
    providing a clean separation between vision and language processing.
    """

    def __init__(
        self,
        vision_session: Optional[QAICInferenceSession],
        processor: Optional[AutoImageProcessor],
        config: Optional[Dict[str, Any]] = None,
        lang_session: Optional[QAICInferenceSession] = None,
    ):
        """
        Initialize vision handler

        Args:
            vision_session: QAICInferenceSession for vision model
            processor: AutoImageProcessor for image preprocessing
            config: Configuration dictionary with vision model parameters
            lang_session: Optional language session for coordination (to avoid resource conflicts)
        """
        self._vision_session = vision_session
        self._processor = processor
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

    def prepare_vision_inputs(self, image_url: str, query: str) -> Dict[str, np.ndarray]:
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

            return vision_inputs

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

    def get_language_inputs_from_vision_processing(
        self, image_url: str, query: str, padded_len: int
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
            # Download and process image
            if image_url.startswith(("http://", "https://")):
                image = Image.open(requests.get(image_url, stream=True).raw)
            else:
                image = Image.open(image_url)

            # Prepare conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image"},
                    ],
                },
            ]

            prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self._processor(images=image, text=prompt, return_tensors="pt")

            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

            # Handle padding for language model
            pad_token_id = 1
            input_ids_length = inputs["input_ids"].shape[1]

            inputs["input_ids"] = torch.nn.functional.pad(
                inputs["input_ids"],
                (0, padded_len - input_ids_length),
                "constant",
                pad_token_id,
            )
            inputs["attention_mask"] = torch.nn.functional.pad(
                inputs["attention_mask"], (0, padded_len - input_ids_length), "constant", 0
            )

            if "cross_attention_mask" in inputs:
                inputs["cross_attention_mask"] = torch.nn.functional.pad(
                    inputs["cross_attention_mask"], (0, 0, 0, 0, 0, padded_len - input_ids_length)
                )

            # Convert to numpy
            for k, v in inputs.items():
                inputs[k] = np.array(v)

            # Separate vision and language inputs
            vision_inputs = {
                k: v
                for k, v in inputs.items()
                if k
                in {
                    "pixel_values",
                    "image_masks",
                    "image_input_idx",
                    "valid_idx",
                    "aspect_ratio_ids",
                    "aspect_ratio_mask",
                }
            }

            # Convert vision inputs to appropriate dtypes
            vision_inputs_fp16 = {"pixel_values", "image_masks"}
            for k in vision_inputs_fp16:
                if k in vision_inputs:
                    vision_inputs[k] = vision_inputs[k].astype("float16")

            # Run vision inference if we have vision inputs
            vision_outputs = {}
            if vision_inputs:
                self.setup_vision_buffers()
                vision_outputs = self.run_vision_inference(vision_inputs)

            # Prepare language inputs
            lang_inputs = {k: v for k, v in inputs.items() if k not in vision_inputs}
            lang_inputs["position_ids"] = np.where(lang_inputs.pop("attention_mask"), np.arange(padded_len), -1)
            lang_inputs["image_idx"] = np.array([[0]])

            return lang_inputs, vision_outputs

        except Exception as e:
            raise RuntimeError(f"Failed to process vision-language inputs: {str(e)}")
