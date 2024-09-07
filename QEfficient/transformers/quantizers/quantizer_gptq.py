# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import tqdm
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from transformers.utils.quantization_config import GPTQConfig

from QEfficient.transformers.quantizers.qunatizer_utils import (
    get_keys_to_not_convert,
    replace_linear_layer_with_target_layer
)
from QEfficient.transformers.quantizers.gptq import QuantLinearGPTQ
from torch import nn

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():  # noqa:SIM108
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):  # noqa:B006
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def select_quant_linear(pack_mode: str, wbits:int, method:str):
    from QEfficient.customop.matmulnbits import QuantLinearORT
    
    pack_mode = pack_mode.upper()
    method = method.lower()
    if pack_mode == "ORT":
        target_layer = QuantLinearORT
    else:
        target_layer = QuantLinearGPTQ
    return target_layer

def repack_to_new_mode(model, new_pack_mode):
    old_pack_mode = "GPTQ"
    if old_pack_mode == new_pack_mode:
        return model
    bits = model.config.quantization_config.bits
    source_layer = select_quant_linear(old_pack_mode, bits, 'gptq')
    target_layer = select_quant_linear(new_pack_mode, bits, 'gptq')
    if source_layer == target_layer:
        return model
    # model.quant_config.version = new_pack_mode
    qlayers = find_layers(model, [source_layer])
    for module_name, qlayer in tqdm.tqdm(qlayers.items(),
            desc=f"repacking model from pack_mode=`{old_pack_mode}` to `{new_pack_mode}`"):
        fp16_weight, scales, zeros = qlayer.unpack()    
        qlayer.weight = fp16_weight
        new_module = target_layer(qlayer.bits, qlayer.groupsize, qlayer.infeatures, qlayer.outfeatures, qlayer.bias is not None)
        new_module.bias = qlayer.bias if qlayer.bias is not None else None
        set_op_by_name(model, module_name, new_module)
        new_module.pack(qlayer, scales.T, zeros.T, qlayer.g_idx)
        qlayer.to('cpu')
        new_module.to('cpu')
    del qlayers
    torch.cuda.empty_cache()
    return model

from QEfficient.utils.logging_utils import logger

class QEffGPTQConfig(GPTQConfig):
    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.bits != 4:
            raise ValueError(f"Only 4-bit quantization is supported, got bits={self.bits}")
        
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")


class QEffGPTQQuantizer(GptqHfQuantizer):
    """
    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    `auto_gptq` package. Quantization is done under the hood for users if they load a non-prequantized model.
    """
    
    target_cls=QuantLinearGPTQ
    
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
            model, target_cls=self.target_cls, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
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
