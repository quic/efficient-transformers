# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING
from transformers.quantizers.quantizer_awq import AwqQuantizer
from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from transformers.utils.quantization_config import AwqConfig, CompressedTensorsConfig, GPTQConfig

from QEfficient.transformers.quantizers.quantizer_awq import QEffAwqConfig, QEffAwqQuantizer
from QEfficient.transformers.quantizers.quantizer_compressed_tensors import (
    QEffCompressedTensorsConfig,
    QEffCompressedTensorsFP8Quantizer,
    QEffFP8Config,
    QEffFP8Quantizer,
)
from QEfficient.transformers.quantizers.quantizer_gptq import QEffGPTQConfig, QEffGPTQQuantizer

QEFF_AUTO_QUANTIZER_MAPPING = {
    "awq": QEffAwqQuantizer,
    "gptq": QEffGPTQQuantizer,
    "compressed-tensors": QEffCompressedTensorsFP8Quantizer,
    "fp8": QEffFP8Quantizer,
}
QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "awq": QEffAwqConfig,
    "gptq": QEffGPTQConfig,
    "compressed-tensors": QEffCompressedTensorsConfig,
    "fp8": QEffFP8Config,
}
DUPLICATE_AUTO_QUANTIZER_MAPPING = {
    "awq": AwqQuantizer,
    "gptq": GptqHfQuantizer,
    "compressed-tensors": CompressedTensorsHfQuantizer,
    "fp8": None,
}
DUPLICATE_AUTO_QUANTIZATION_CONFIG_MAPPING = {
    "awq": AwqConfig,
    "gptq": GPTQConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "fp8": None,
}


def with_replaced_quantizers(func):
    def wrapper(*args, **kwargs):
        transformers_replaced_quantization_config_mapping = dict()
        transformers_replaced_quantizer_mapping = dict()

        for k in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            # Replace quantization config
            transformers_replaced_quantization_config_mapping[k] = AUTO_QUANTIZATION_CONFIG_MAPPING.get(k, None)
            AUTO_QUANTIZATION_CONFIG_MAPPING[k] = QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING[k]

            # Replace quantizer
            transformers_replaced_quantizer_mapping[k] = AUTO_QUANTIZER_MAPPING.get(k, None)
            AUTO_QUANTIZER_MAPPING[k] = QEFF_AUTO_QUANTIZER_MAPPING[k]

        # Call the function for loading quantized models here
        out = func(*args, **kwargs)

        # Put back quantization config and quantizer
        for k in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            AUTO_QUANTIZATION_CONFIG_MAPPING[k] = transformers_replaced_quantization_config_mapping[k]
            AUTO_QUANTIZER_MAPPING[k] = transformers_replaced_quantizer_mapping[k]

        return out

    return wrapper


def replace_transformers_quantizers():
    """
    This method lets you import AWQ/GPTQ models on CPU without bypassing the
    rule of transformers of need to GPU.
    Just call this method before using
    `transformer.AutoModelForCausalLM.from_pretrained` and any AWQ/GPTQ model
    that can be supported by QEfficient will be loaded using CPU.
    """
    AUTO_QUANTIZER_MAPPING.update(QEFF_AUTO_QUANTIZER_MAPPING)
    AUTO_QUANTIZATION_CONFIG_MAPPING.update(QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING)


# TODO: Make this a fixture? Or better, always update the quantizer and config in transformers.
# When a user imports QEfficient, these are always available.
def undo_transformers_quantizers():
    """
    This method is used to undo the effects on method `replace_transformers_quantizers`.
    After this is called, the transformers library will be used for loading AWQ/GPTQ models.
    """
    AUTO_QUANTIZER_MAPPING.update(DUPLICATE_AUTO_QUANTIZER_MAPPING)
    AUTO_QUANTIZATION_CONFIG_MAPPING.update(DUPLICATE_AUTO_QUANTIZATION_CONFIG_MAPPING)
