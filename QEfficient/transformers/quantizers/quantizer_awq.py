# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers.quantizers.quantizer_awq import AwqQuantizer
from transformers.utils.quantization_config import AwqConfig

try:
    from transformers.utils.quantization_config import AwqBackend, AwqFormat
except ImportError:
    from transformers.utils.quantization_config import AwqBackendPackingMethod as AwqBackend
    from transformers.utils.quantization_config import AWQLinearVersion as AwqFormat

from QEfficient.transformers.quantizers.awq import WQLinear_GEMM
from QEfficient.transformers.quantizers.quantizer_utils import (
    get_keys_to_not_convert,
    replace_linear_layer_with_target_layer,
    replace_quantization_scales,
)
from QEfficient.utils.logging_utils import logger


class QEffAwqConfig(AwqConfig):
    def post_init(self):
        """
        Safety checker that arguments are correct
        """

        auto_backend = getattr(AwqBackend, "AUTOAWQ", None)
        if auto_backend is None:
            auto_backend = getattr(AwqBackend, "AUTO", None)

        if self.backend not in [auto_backend]:
            raise ValueError(
                f"Only quantization backend {auto_backend} is supported - not recognized backend {self.backend}"
            )

        fmt = getattr(self, "format", getattr(self, "version", None))
        if isinstance(fmt, str):
            normalized_fmt = fmt.lower()
        else:
            normalized_fmt = getattr(fmt, "value", fmt)

        gemm_format = getattr(AwqFormat, "GEMM", None)
        if normalized_fmt != getattr(gemm_format, "value", gemm_format):
            raise ValueError(f"Only {gemm_format} version in supported - not recognized version {fmt}")

        do_fuse = getattr(self, "do_fuse", None)
        fuse_max_seq_len = getattr(self, "fuse_max_seq_len", None)
        if do_fuse or fuse_max_seq_len is not None:
            raise ValueError(
                f"fused modules are not supported, got do_fuse={do_fuse}, fuse_max_seq_len={fuse_max_seq_len}"
            )

        if self.bits != 4:
            raise ValueError(f"Only 4-bit AWQ quantization is supported, got bits={self.bits}")


class QEffAwqQuantizer(AwqQuantizer):
    target_cls = WQLinear_GEMM

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

    def update_dtype(self, dtype):
        return self.update_torch_dtype(dtype)

    def _process_model_before_weight_loading(self, model, **kwargs):
        self.modules_to_not_convert = get_keys_to_not_convert(model)

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model, has_been_replaced = replace_linear_layer_with_target_layer(
            model,
            target_cls=self.target_cls,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )

        model = replace_quantization_scales(model, model.config.model_type)
        if not has_been_replaced:
            logger.warning(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is a bug."
            )
