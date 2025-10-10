# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import warnings
from pathlib import Path


class CustomIOGenerator:
    """
    Abstract base class for generating custom IO mappings for different model types.

    Args:
        model (object): The model instance for which IO mappings are to be generated.
        cache_dir (str): Directory path where the generated YAML files will be saved.
        mxint8_kv_cache (bool): If True, use 'mxint8' precision for KV cache; otherwise, use 'float16'.
    """

    def __init__(self, model, cache_dir=".", mxint8_kv_cache=False):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
        self.dtype_suffix = "int8" if mxint8_kv_cache else "fp16"

    def dump(self, custom_io: dict, suffix: str):
        """
        Writes the custom IO mapping to a YAML file.

        Args:
            custom_io (dict): Dictionary containing IO names and their precision types.
            suffix (str): Suffix to append to the output filename.
        """
        custom_io_yaml = self.cache_dir / f"custom_io_{suffix}.yaml"
        with open(custom_io_yaml, "w") as fp:
            for io_name, dtype in custom_io.items():
                fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")

    def generate(self) -> dict:
        """
        Abstract method to generate custom IO mappings.

        Returns:
            dict: A dictionary of IO names and their precision types.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")


class CausalLMIOGenerator(CustomIOGenerator):
    """
    IO generator for causal language models.
    """

    def generate(self) -> dict:
        """
        Generates IO mappings for past key/value states in causal language models.

        Returns:
            dict: Mapping of IO names to precision types.
        """
        custom_io = {}
        num_layers = getattr(self.model, "num_layers", 12)
        for suffix in ["", "_RetainedState"]:
            for i in range(num_layers):
                for kv in ["key", "value"]:
                    custom_io[f"past_{kv}.{i}{suffix}"] = self.kv_cache_dtype
        self.dump(custom_io, self.dtype_suffix)
        return custom_io


class DualQPCIOGenerator(CustomIOGenerator):
    """
    IO generator for dual QPC models (e.g., vision-language models).
    """

    def generate(self) -> dict:
        """
        Generates IO mappings for both vision and language components.

        Returns:
            dict: Combined mapping of IO names to precision types for vision and language outputs.
        """
        output_names = self.model.model.get_output_names()
        custom_io_vision = {
            name: self.kv_cache_dtype if name.startswith("past_") else "float16"
            for name in output_names.get("vision", [])
        }

        custom_io_lang = {}
        for name in output_names.get("lang", []):
            if name.endswith("_RetainedState"):
                base = name[: -len("_RetainedState")]
                dtype = "float16" if "vision_embeds" in name else self.kv_cache_dtype
                custom_io_lang[base] = dtype
                custom_io_lang[name] = dtype

        self.dump(custom_io_vision, f"{self.dtype_suffix}_vision")
        self.dump(custom_io_lang, f"{self.dtype_suffix}_lang")
        warnings.warn(f"Unsupported model class via CLI: {type(self.model).__name__}", UserWarning)
        return {**custom_io_vision, **custom_io_lang}


class SingleQPCIOGenerator(CustomIOGenerator):
    """
    IO generator for single QPC models.
    """

    def generate(self) -> dict:
        """
        Generates IO mappings for retained states in single QPC models.

        Returns:
            dict: Mapping of IO names to precision types.
        """
        output_names = self.model.model.get_output_names()
        custom_io = {}
        for name in output_names:
            if name.endswith("_RetainedState"):
                base = name[: -len("_RetainedState")]
                dtype = "float16" if "pixel_values" in name else self.kv_cache_dtype
                custom_io[base] = dtype
                custom_io[name] = dtype
        self.dump(custom_io, self.dtype_suffix)
        return custom_io


class SpeechSeq2SeqIOGenerator(CustomIOGenerator):
    """
    IO generator for speech sequence-to-sequence models.
    """

    def generate(self) -> dict:
        """
        Generates IO mappings for input features and retained states in speech models.

        Returns:
            dict: Mapping of IO names to precision types.
        """
        output_names = self.model.model.get_output_names()
        custom_io = {"input_features": self.kv_cache_dtype}
        for name in output_names:
            if name.endswith("_RetainedState"):
                base = name[: -len("_RetainedState")]
                custom_io[base] = self.kv_cache_dtype
                custom_io[name] = self.kv_cache_dtype
        self.dump(custom_io, self.dtype_suffix)
        return custom_io


class UnsupportedModelIOGenerator(CustomIOGenerator):
    """
    Fallback IO generator for unsupported model types.
    """

    def generate(self) -> dict:
        """
        Emits a warning for unsupported model types.

        Returns:
            dict: Empty dictionary.
        """
        warnings.warn(f"Unsupported model class: {type(self.model).__name__}", UserWarning)
        return {}


class CustomIOFactory:
    """
    Factory class to instantiate the appropriate IO generator based on model type.
    """

    @staticmethod
    def get_generator(model, cache_dir=".", mxint8_kv_cache=False) -> CustomIOGenerator:
        """
        Returns the appropriate IO generator instance for the given model.

        Args:
            model (object): The model instance.
            cache_dir (str): Directory to store YAML files.
            mxint8_kv_cache (bool): Flag to use 'mxint8' precision.

        Returns:
            CustomIOGenerator: An instance of the appropriate subclass.
        """
        model_class_name = type(model).__name__
        mapping = {
            "QEFFAutoModelForCausalLM": CausalLMIOGenerator,
            "_QEFFAutoModelForImageTextToTextDualQPC": DualQPCIOGenerator,
            "_QEFFAutoModelForImageTextToTextSingleQPC": SingleQPCIOGenerator,
            "QEFFAutoModelForSpeechSeq2Seq": SpeechSeq2SeqIOGenerator,
        }
        generator_class = mapping.get(model_class_name, UnsupportedModelIOGenerator)
        return generator_class(model, cache_dir, mxint8_kv_cache)


def generate_custom_io(qeff_model, cache_dir=".", mxint8_kv_cache=False) -> dict:
    """
    Generates and returns custom IO mappings for the given QEFF model.

    Args:
        qeff_model (object): The model instance.
        cache_dir (str): Directory to store YAML files.
        mxint8_kv_cache (bool): Flag to use 'mxint8' precision.

    Returns:
        dict: Custom IO mapping generated by the appropriate generator.
    """
    generator = CustomIOFactory.get_generator(qeff_model, cache_dir, mxint8_kv_cache)
    return generator.generate()
