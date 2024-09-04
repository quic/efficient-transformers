# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING

from QEfficient.transformers.quantizers.quantizer_awq import QEffAwqConfig, QEffAwqQuantizer

QEFF_AUTO_QUANTIZER_MAPPING = {"awq": QEffAwqQuantizer}

QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING = {"awq": QEffAwqConfig}


def with_replaced_quantizers(func):
    def wrapper(*args, **kwargs):
        transformers_replaced_quantization_config_mapping = dict()
        transformers_replaced_quantizer_mapping = dict()

        for k in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            # Replace quantization config
            transformers_replaced_quantization_config_mapping[k] = AUTO_QUANTIZATION_CONFIG_MAPPING[k]
            AUTO_QUANTIZATION_CONFIG_MAPPING[k] = QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING[k]

            # Replace quantizer
            transformers_replaced_quantizer_mapping[k] = AUTO_QUANTIZER_MAPPING[k]
            AUTO_QUANTIZER_MAPPING[k] = QEFF_AUTO_QUANTIZER_MAPPING[k]

        # Call the function for loading quantized models here
        out = func(*args, **kwargs)

        # Put back quantization config and quantizer
        for k in QEFF_AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            AUTO_QUANTIZATION_CONFIG_MAPPING[k] = transformers_replaced_quantization_config_mapping[k]
            AUTO_QUANTIZER_MAPPING[k] = transformers_replaced_quantizer_mapping[k]

        return out

    return wrapper
