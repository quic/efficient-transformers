# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass

import torch
from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
from transformers.utils.quantization_config import CompressedTensorsConfig, QuantizationMethod

from QEfficient.utils.logging_utils import logger

FP8_DTYPE = torch.float8_e4m3fn


@dataclass
class FP8QuantizationScheme:
    dynamic: bool
    num_bits: int
    strategy: str
    symmetric: bool
    type: str

    def __post_init__(self):
        if self.num_bits != 8 or self.type != "float" or self.strategy not in ["tensor", "channel", "token"]:
            raise NotImplementedError(
                f"Only FP8 compressed-tensors supported, got num_bits={self.num_bits}, type={self.type}, strategy={self.strategy}"
            )


class CompressedFP8Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weights_quant_scheme: FP8QuantizationScheme,
        input_activations_quant_scheme: FP8QuantizationScheme,
        bias: bool = False,
    ):
        super().__init__()
        self.weights_quantization_scheme = weights_quant_scheme
        self.input_activations_quantization_scheme = input_activations_quant_scheme
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.empty(
                (out_features, in_features), dtype=FP8_DTYPE
            ),  # This is fixed for now and only e4m3fn quantization is prominent
        )

        if self.weights_quantization_scheme.dynamic:
            raise NotImplementedError(
                f"Expected statically quantized weights but got weights quantization scheme dynamic = {self.weights_quantization_scheme.dynamic}"
            )

        if self.weights_quantization_scheme.strategy == "tensor":
            self.register_buffer("weight_scale", torch.zeros((1), dtype=torch.float32))
        elif self.weights_quantization_scheme.strategy == "channel":
            self.register_buffer("weight_scale", torch.zeros((out_features, 1), dtype=torch.float32))
        else:
            raise NotImplementedError(
                f"Unknown weights quantization strategy {self.weights_quantization_scheme.strategy}, ['channel' or 'tensor'] strategy supported."
            )

        if not self.input_activations_quantization_scheme.dynamic:
            if self.input_activations_quantization_scheme.strategy == "tensor":
                self.register_buffer("input_scale", torch.zeros((1), dtype=torch.float32))
            elif self.input_activations_quant_scheme.strategy == "token":
                self.register_buffer("input_scale", torch.zeros((1, in_features), dtype=torch.float32))
            else:
                raise NotImplementedError(
                    f"Unknown input activations quantization strategy {self.input_activations_quantization_scheme.strategy}, ['token' or 'tensor'] strategy supported."
                )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                ),
            )
        else:
            self.bias = None

    def forward(self, x):
        # Only inference supported
        with torch.no_grad():
            dequantized_weights = self.weight.to(torch.float32) * self.weight_scale
            out = torch.matmul(x.float(), dequantized_weights.T)
            out = out + self.bias if self.bias is not None else out

        return out


class QEffCompressedTensorsConfig(CompressedTensorsConfig):
    def __init__(
        self,
        config_groups=None,
        format="dense",
        quantization_status="initialized",
        kv_cache_scheme=None,
        global_compression_ratio=None,
        ignore=None,
        sparsity_config=None,
        quant_method="compressed-tensors",
        **kwargs,
    ):
        self.config_groups = config_groups
        self.quant_method = quant_method
        self.kv_cache_scheme = kv_cache_scheme
        self.format = format
        self.quantization_status = quantization_status
        self.global_compression_ratio = global_compression_ratio
        self.ignore = ignore

        # Validate configuration
        if len(self.config_groups) != 1:
            raise NotImplementedError(
                "Currently only single quantization group is supported, please raise an issue with model details for support!"
            )

        if quantization_status != "frozen":
            raise NotImplementedError(f"expected quantization_status=`frozen`, got {quantization_status}")

        if kv_cache_scheme:
            raise NotImplementedError(f"Expected kv_cache_scheme=None, got {kv_cache_scheme}")

        if format != "naive-quantized":
            raise NotImplementedError(f"Expected quantization format =`naive_quantized` got {format}")

        if sparsity_config:
            raise NotImplementedError(f"Expected sparsity_config to be None, got {sparsity_config}")

        if quant_method != "compressed-tensors":
            raise NotImplementedError("Only compressed-tensors quant_method is supported for now!")

        group_0 = self.config_groups.get("group_0")
        activations_quantization_config = group_0.get("input_activations")
        weights_quantization_config = group_0.get("weights")
        output_activation_quantization_config = group_0.get("output_activations")
        self.targets = group_0.get("targets")

        if self.targets != ["Linear"]:
            raise NotImplementedError(f"Only linear targets are supported, got {self.targets}")

        if output_activation_quantization_config:
            raise NotImplementedError(
                f"output_activations quantization is not supported got {output_activation_quantization_config}"
            )

        if (
            activations_quantization_config.get("block_structure")
            or activations_quantization_config.get("group_size")
            or weights_quantization_config.get("block_structure")
            or weights_quantization_config.get("group_size")
        ):
            raise NotImplementedError(f"group_size and block_structure not supported got {group_0}")

        self.weights_quantization_scheme = FP8QuantizationScheme(
            weights_quantization_config.get("dynamic"),
            weights_quantization_config.get("num_bits"),
            weights_quantization_config.get("strategy"),
            weights_quantization_config.get("symmetric"),
            weights_quantization_config.get("type"),
        )
        self.input_activations_quantization_scheme = FP8QuantizationScheme(
            activations_quantization_config.get("dynamic"),
            activations_quantization_config.get("num_bits"),
            activations_quantization_config.get("strategy"),
            activations_quantization_config.get("symmetric"),
            activations_quantization_config.get("type"),
        )

        self.quant_method = QuantizationMethod.COMPRESSED_TENSORS

    def to_dict(self):
        return {
            "quantization_config": {
                "config_groups": self.config_groups,
                "weights_quantization_scheme": self.weights_quantization_scheme.__dict__,
                "activations_quantization_scheme": self.input_activations_quantization_scheme.__dict__,
                "quant_method": self.quant_method,
                "kv_cache_scheme": self.kv_cache_scheme,
                "format": self.format,
                "quantization_status": self.quantization_status,
                "global_compression_ratio": self.global_compression_ratio,
                "ignore": self.ignore,
                "targets": self.targets,
            },
            "sparsity_config": None,
        }


class QEffFP8Quantizer(CompressedTensorsHfQuantizer):
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        # TODO: check if more checks are required
        if not isinstance(quantization_config, QEffCompressedTensorsConfig):
            raise TypeError(
                f"Only {QEffCompressedTensorsConfig} is supported for initialization got {type(quantization_config)}"
            )

        self.quantization_config = quantization_config

        # -- Handle extra kwargs below --
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.pre_quantized = kwargs.pop("pre_quantized", True)

        if not self.pre_quantized and self.requires_calibration:
            raise ValueError(
                f"The quantization method {quantization_config.quant_method} does require the model to be pre-quantized."
                f" You explicitly passed `pre_quantized=False` meaning your model weights are not quantized. Make sure to "
                f"pass `pre_quantized=True` while knowing what you are doing."
            )

    def validate_environment(self, *args, **kwargs):
        return True

    def update_torch_dtype(self, torch_dtype):
        if torch_dtype not in [None, torch.float32]:
            logger.warning(f"Requested dtype {torch_dtype} is not supported, overriding to None")
        return None

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.quantization_config.targets != ["Linear"]:
            raise NotImplementedError(
                f"Only Linear layer with FP8 quantization are supported got targets = {self.quantization_config.targets}"
            )

        # -- Defining local method as it uses lot of local variables --
        def replace_linear_layer_with_compressed_fp8_layer(module):
            for name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear) and name not in self.quantization_config.ignore:
                    compressed_fp8_layer = CompressedFP8Linear(
                        child_module.in_features,
                        child_module.out_features,
                        self.quantization_config.weights_quantization_scheme,
                        self.quantization_config.input_activations_quantization_scheme,
                        child_module.bias,
                    )
                    setattr(module, name, compressed_fp8_layer)
                else:
                    replace_linear_layer_with_compressed_fp8_layer(child_module)

        replace_linear_layer_with_compressed_fp8_layer(model)
