# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import tqdm
from transformers.quantizers.quantizer_gptq import HfQuantizer
from transformers.utils.quantization_config import GPTQConfig

from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from QEfficient.transformers.quantizers.qunatizer_utils import (
    find_layers,
    get_keys_to_not_convert,
    replace_linear_layer_with_target_layer,
)
from QEfficient.utils.logging_utils import logger


class QEffGPTQConfig(GPTQConfig):
    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.bits != 4:
            raise ValueError(f"Only 4-bit quantization is supported, got bits={self.bits}")
        if self.desc_act:
            raise ValueError("Only GPTQ model without decreasing activation size supported.")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")

        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")


class QEffGPTQQuantizer(HfQuantizer):
    """
    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    `auto_gptq` package. Quantization is done under the hood for users if they load a non-prequantized model.
    """

    target_cls = QuantLinearGPTQ

    def __init__(self, quantization_config: QEffGPTQConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        return True

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype not in [None, torch.float32]:
            logger.warning(f"Requested dtype {torch_dtype} is not supported, overriding to None")
        return None

    def _process_model_before_weight_loading(self, model, **kwargs):
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")
        if not self.pre_quantized:
            raise RuntimeError("Model is not quantized")

        self.modules_to_not_convert = get_keys_to_not_convert(model)

        model, has_been_replaced = replace_linear_layer_with_target_layer(
            model,
            target_cls=self.target_cls,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )
        if not has_been_replaced:
            logger.warning(
                "You are loading an GPTQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        qlayers = find_layers(model, [QuantLinearGPTQ])
        for _, qlayer in tqdm.tqdm(qlayers.items(), desc="Repacking AutoGPTQ qzeros..."):
            qlayer.handle()
        print("Done")

    @property
    def is_trainable(self):
        return False

    @property
    def is_serializable(self):
        return True
