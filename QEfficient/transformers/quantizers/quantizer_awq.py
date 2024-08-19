import torch
from transformers.integrations import replace_quantization_scales
from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING
from transformers.quantizers.quantizer_awq import AwqQuantizer
from transformers.utils.quantization_config import AwqBackendPackingMethod, AwqConfig, AWQLinearVersion

from QEfficient.transformers.quantizers.awq import get_keys_to_not_convert, replace_linear_layer_with_awq_gemm
from QEfficient.utils.logging_utils import logger


class QEffAwqConfig(AwqConfig):
    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """

        if self.backend not in [AwqBackendPackingMethod.AUTOAWQ]:
            raise ValueError(
                f"Only supported quantization backends in {AwqBackendPackingMethod.AUTOAWQ} - not recognized backend {self.backend}"
            )

        self.version = AWQLinearVersion.from_str(self.version)
        if self.version not in [AWQLinearVersion.GEMM]:
            raise ValueError(
                f"Only supported versions are in [AWQLinearVersion.GEMM] - not recognized version {self.version}"
            )

        if self.do_fuse or self.fuse_max_seq_len is not None:
            raise ValueError("fused modules are not supported")

        if self.bits != 4:
            raise ValueError(f"Only 4-bit quantization is supported, got bits={self.bits}")


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


def replace_transformers_quantizers():
    AUTO_QUANTIZER_MAPPING.update({"awq": QEffAwqQuantizer})
    AUTO_QUANTIZATION_CONFIG_MAPPING.update({"awq": QEffAwqConfig})
