# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy

import torch
import torch.nn as nn
from transformers.integrations.awq import AWQ_SCALES_MAPPINGS
from transformers.quantizers.quantizer_awq import AwqQuantizer
from transformers.utils.quantization_config import AwqBackendPackingMethod, AwqConfig, AWQLinearVersion

from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.utils.logging_utils import logger


class QEffAwqConfig(AwqConfig):
    def post_init(self):
        """
        Safety checker that arguments are correct
        """

        if self.backend not in [AwqBackendPackingMethod.AUTOAWQ]:
            raise ValueError(
                f"Only quantization backend {AwqBackendPackingMethod.AUTOAWQ} is supported - not recognized backend {self.backend}"
            )

        self.version = AWQLinearVersion.from_str(self.version)
        if self.version not in [AWQLinearVersion.GEMM]:
            raise ValueError(
                f"Only {AWQLinearVersion.GEMM} version in supported - not recognized version {self.version}"
            )

        if self.do_fuse or self.fuse_max_seq_len is not None:
            raise ValueError(
                f"fused modules are not supported, got do_fuse={self.do_fuse}, fuse_max_seq_len={self.fuse_max_seq_len}"
            )

        if self.bits != 4:
            raise ValueError(f"Only 4-bit AWQ quantization is supported, got bits={self.bits}")


class QEffAwqQuantizer(AwqQuantizer):
    def __init__(self, quantization_config: QEffAwqConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, device_map, **kwargs):
        # No need to validate as we will always use pytorch CPU version.
        return True

    @property
    def is_trainable(self):
        return False

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype not in [None, torch.float32]:
            logger.warning(f"Requested dtype {torch_dtype} is not supported, overriding to None")
        return None

    def _process_model_before_weight_loading(self, model, **kwargs):
        self.modules_to_not_convert = get_keys_to_not_convert(model)

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model, has_been_replaced = replace_linear_layer_with_awq_gemm(
            model, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
        )

        model = replace_quantization_scales(model, model.config.model_type)
        if not has_been_replaced:
            logger.warning(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1)


def replace_quantization_scales(model, model_type):
    if model_type not in AWQ_SCALES_MAPPINGS:
        return model
    for name, module in model.named_children():
        act_name = AWQ_SCALES_MAPPINGS[model_type]["act"]
        layer_before_act_name = AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"]
        if name == act_name and hasattr(model, layer_before_act_name):
            layer_before_act = getattr(model, AWQ_SCALES_MAPPINGS[model_type]["layer_before_act"])
            size = layer_before_act.out_features
            scale_like = torch.ones(size)
            model._modules[name] = ScaledActivation(module, scale_like)
            replace_quantization_scales(module, model_type)
    return model


def replace_linear_layer_with_awq_gemm(
    model: torch.nn.Module,
    quantization_config=None,
    modules_to_not_convert=None,
    current_key_name=None,
    has_been_replaced=False,
):
    modules_to_not_convert = modules_to_not_convert if modules_to_not_convert else []

    for name, module in model.named_children():
        current_key_name = current_key_name if current_key_name else []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                model._modules[name] = WQLinear_GEMM(
                    w_bit=quantization_config.bits,
                    group_size=quantization_config.group_size,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                )
                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_linear_layer_with_awq_gemm(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )

        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def get_keys_to_not_convert(model):
    tied_model = copy.deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()
    tied_params = find_tied_parameters(tied_model)
    tied_keys = sum(tied_params, [])

    # If there is not tied weights, we want to keep the lm_head（output_embedding) in full precision
    if len(tied_keys) == 0:
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            list_last_module = [name for name, module in model.named_modules() if id(module) == id(output_emb)]
            return list_last_module

    # otherwise, no tied weights, no output embedding defined, simply keep the last module in full precision
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]
    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = list(set(tied_keys)) + list(intersection)

    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names


def find_tied_parameters(model: nn.Module, named_parameters=None, prefix="", result={}):
    if named_parameters is None:
        named_parameters = {n: p for n, p in model.named_parameters()}
    else:
        for name, parameter in model.named_parameters():
            full_name = name if prefix == "" else f"{prefix}.{name}"
            if full_name not in named_parameters:
                # When we find one, it has to be one of the existing parameters.
                for new_name, new_param in named_parameters.items():
                    if new_param is parameter:
                        if new_name not in result:
                            result[new_name] = []
                        result[new_name].append(full_name)

    # Once we have treated direct parameters, we move to the child modules.
    for name, child in model.named_children():
        child_name = name if prefix == "" else f"{prefix}.{name}"
        find_tied_parameters(child, named_parameters=named_parameters, prefix=child_name, result=result)

    return [sorted([weight] + list(set(tied))) for weight, tied in result.items()]
