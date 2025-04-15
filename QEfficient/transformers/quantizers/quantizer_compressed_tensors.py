# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from enum import Enum
from typing import List

import torch
from transformers.quantizers.quantizer_compressed_tensors import CompressedTensorsHfQuantizer
from transformers.utils.quantization_config import CompressedTensorsConfig, QuantizationConfigMixin, QuantizationMethod

from QEfficient.transformers.quantizers.quantizer_utils import get_keys_to_not_convert
from QEfficient.utils.logging_utils import logger

FP8_DTYPE = torch.float8_e4m3fn


class QEffExtendedQuantizationMethod(str, Enum):
    FP8 = "fp8"


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


class FP8DeQuantLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.empty(
                (out_features, in_features), dtype=FP8_DTYPE
            ),  # This is fixed for now and only e4m3fn quantization is prominent
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

    @classmethod
    def for_compressed_tensors_fp8_layer(
        cls,
        in_features: int,
        out_features: int,
        weights_quant_scheme: FP8QuantizationScheme,
        input_activations_quant_scheme: FP8QuantizationScheme,
        bias: bool = False,
    ):
        fp8_dequant_layer = cls(in_features, out_features, bias)
        fp8_dequant_layer.weights_quantization_scheme = weights_quant_scheme
        fp8_dequant_layer.input_activations_quantization_scheme = input_activations_quant_scheme

        if fp8_dequant_layer.weights_quantization_scheme.dynamic:
            raise NotImplementedError(
                f"Expected statically quantized weights but got weights quantization scheme dynamic = {fp8_dequant_layer.weights_quantization_scheme.dynamic}"
            )

        if fp8_dequant_layer.weights_quantization_scheme.strategy == "tensor":
            fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((1), dtype=torch.float32))
        elif fp8_dequant_layer.weights_quantization_scheme.strategy == "channel":
            fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((out_features, 1), dtype=torch.float32))
        else:
            raise NotImplementedError(
                f"Unknown weights quantization strategy {fp8_dequant_layer.weights_quantization_scheme.strategy}, ['channel' or 'tensor'] strategy supported."
            )

        if not fp8_dequant_layer.input_activations_quantization_scheme.dynamic:
            if fp8_dequant_layer.input_activations_quantization_scheme.strategy == "tensor":
                fp8_dequant_layer.register_buffer("input_scale", torch.zeros((1), dtype=torch.float32))
            elif fp8_dequant_layer.input_activations_quant_scheme.strategy == "token":
                fp8_dequant_layer.register_buffer("input_scale", torch.zeros((1, in_features), dtype=torch.float32))
            else:
                raise NotImplementedError(
                    f"Unknown input activations quantization strategy {fp8_dequant_layer.input_activations_quantization_scheme.strategy}, ['token' or 'tensor'] strategy supported."
                )

        return fp8_dequant_layer

    @classmethod
    def for_fp8_layer(cls, in_features, out_features, activation_quantization_strategy, bias):
        fp8_dequant_layer = cls(in_features, out_features, bias)

        # -- Always per tensor quantization assumed --
        fp8_dequant_layer.register_buffer("weight_scale", torch.zeros((), dtype=torch.float32))

        if activation_quantization_strategy == "static":
            fp8_dequant_layer.register_buffer("input_scale", torch.zeros((), dtype=torch.float32))

        return fp8_dequant_layer

    def forward(self, x):
        # Only inference supported
        with torch.no_grad():
            dequantized_weights = self.weight.to(torch.float32) * self.weight_scale
            out = torch.matmul(x.float(), dequantized_weights.T)
            out = out + self.bias if self.bias is not None else out

        return out


class QEffFP8Config(QuantizationConfigMixin):
    def __init__(
        self,
        quant_method: str,
        activation_scheme: str,
        ignored_layers: List[str] = None,
        kv_cache_scheme: str = None,
        run_compressed: bool = True,
    ):
        self.quant_method = quant_method
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers
        self.kv_cache_scheme = kv_cache_scheme
        self.run_compressed = run_compressed
        self.quantization_config = None
        self.sparsity_config = None
        if kv_cache_scheme:
            logger.warning(
                f"kv_cache_scheme={kv_cache_scheme} will be ignored please use `mxint8_kv_cache=True` during compile call if you want to keep kv cache in int8 at runtime on Cloud AI 100"
            )

        if quant_method != "fp8" or activation_scheme not in ["static", "dynamic", None]:
            raise NotImplementedError(
                f"Expected FP8 quantization with static/dynamic/None activation quantization, go quant_method={quant_method}, activation_scheme={activation_scheme}"
            )

        self.quant_method = QEffExtendedQuantizationMethod.FP8


class QEffFP8Quantizer(CompressedTensorsHfQuantizer):
    def __init__(self, quantization_config, **kwargs):
        # TODO: check if more checks are required
        if not isinstance(quantization_config, QEffFP8Config):
            raise TypeError(f"Only {QEffFP8Config} is supported for initialization got {type(quantization_config)}")

        self.quantization_config = quantization_config
        self.run_compressed = quantization_config.run_compressed
        # -- Handle extra kwargs below --
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.modules_to_not_convert = list(
            set(self.modules_to_not_convert if self.modules_to_not_convert else [])
            | set(self.quantization_config.ignored_layers if self.quantization_config.ignored_layers else [])
        )
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
        if not self.modules_to_not_convert or "lm_head" not in self.modules_to_not_convert:
            self.modules_to_not_convert.extend(get_keys_to_not_convert(model))

        logger.warning(
            f"activations quantization strategy = {self.quantization_config.activation_scheme}, will be ignored and the layers will be run with de-quantized weights"
        )

        # -- Defining local method as it uses lot of local variables --
        def replace_linear_with_fp8_dequant_layer(module):
            for name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear) and name not in self.modules_to_not_convert:
                    compressed_fp8_layer = FP8DeQuantLinear.for_fp8_layer(
                        child_module.in_features,
                        child_module.out_features,
                        self.quantization_config.activation_scheme,
                        child_module.bias is not None,
                    )
                    setattr(module, name, compressed_fp8_layer)
                else:
                    replace_linear_with_fp8_dequant_layer(child_module)

        replace_linear_with_fp8_dequant_layer(model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        pass


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
        run_compressed: bool = False,
        **kwargs,
    ):
        self.config_groups = config_groups
        self.quant_method = quant_method
        self.kv_cache_scheme = kv_cache_scheme
        self.format = format
        self.quantization_status = quantization_status
        self.global_compression_ratio = global_compression_ratio
        self.ignore = ignore

        self.quantization_config = None
        self.sparsity_config = None

        self.run_compressed = run_compressed
        # Validate configuration
        if len(self.config_groups) != 1:
            raise NotImplementedError(
                "Currently only single quantization group is supported, please raise an issue with model details for support!"
            )

        if quantization_status not in {"compressed", "frozen"}:
            raise NotImplementedError(f"expected quantization_status=`frozen`, got {quantization_status}")

        if kv_cache_scheme:
            raise NotImplementedError(f"Expected kv_cache_scheme=None, got {kv_cache_scheme}")

        if format not in {"naive-quantized", "float-quantized"}:
            raise NotImplementedError(
                f"Expected quantization format in ['naive_quantized', 'float-quantized']  got {format}"
            )

        if sparsity_config:
            raise NotImplementedError(f"Expected sparsity_config to be None, got {sparsity_config}")

        if quant_method != "compressed-tensors":
            raise NotImplementedError("Only compressed-tensors quant_method is supported for now!")

        if "lm_head" not in self.ignore:
            raise AttributeError(f"Expected `lm_head` to be present in non-quantized layers got ignore={self.ignore}")

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


class QEffCompressedTensorsFP8Quantizer(CompressedTensorsHfQuantizer):
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        # TODO: check if more checks are required
        if not isinstance(quantization_config, QEffCompressedTensorsConfig):
            raise TypeError(
                f"Only {QEffCompressedTensorsConfig} is supported for initialization got {type(quantization_config)}"
            )
        self.run_compressed = quantization_config.run_compressed
        self.quantization_config = quantization_config

        # -- Handle extra kwargs below --
        self.modules_to_not_convert = kwargs.pop("modules_to_not_convert", [])
        self.modules_to_not_convert = list(
            set(self.modules_to_not_convert if self.modules_to_not_convert else [])
            | set(self.quantization_config.ignore if self.quantization_config.ignore else [])
        )
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

        logger.warning(
            f"activations quantization scheme = {self.quantization_config.input_activations_quantization_scheme.__dict__}, will be ignored and the layers will be run with de-quantized weights"
        )

        # -- Defining local method as it uses lot of local variables --
        def replace_linear_with_fp8_dequant_layer(module):
            for name, child_module in module.named_children():
                if isinstance(child_module, torch.nn.Linear) and name not in self.modules_to_not_convert:
                    compressed_fp8_layer = FP8DeQuantLinear.for_compressed_tensors_fp8_layer(
                        child_module.in_features,
                        child_module.out_features,
                        self.quantization_config.weights_quantization_scheme,
                        self.quantization_config.input_activations_quantization_scheme,
                        child_module.bias is not None,
                    )
                    setattr(module, name, compressed_fp8_layer)
                else:
                    replace_linear_with_fp8_dequant_layer(child_module)

        replace_linear_with_fp8_dequant_layer(model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        pass
