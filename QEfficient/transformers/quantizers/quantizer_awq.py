# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
from transformers.quantizers.quantizer_awq import AwqQuantizer
# v5: AwqBackendPackingMethod -> AwqBackend, AWQLinearVersion -> AwqFormat
from transformers.utils.quantization_config import AwqBackend, AwqConfig, AwqFormat

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

        # Coerce backend to the new enum if it was provided as a string (e.g. "autoawq")
        if isinstance(self.backend, str):
            try:
                self.backend = AwqBackend(self.backend)
            except ValueError:
                pass  # keep as-is and let the check below fail with a clear message

        if self.backend not in [AwqBackend.LEGACY_AWQ]:
            raise ValueError(
                f"Only quantization backend {AwqBackend.LEGACY_AWQ} is supported - not recognized backend {self.backend}"
            )

        # In v5 AwqFormat values are lowercase ("gemm", "gemv", ...).
        # Accept both enum and string inputs (e.g. "GEMM"/"gemm")
        if isinstance(self.version, str):
            try:
                self.version = AwqFormat(self.version)
            except ValueError:
                self.version = AwqFormat(self.version.lower())

        if self.version not in [AwqFormat.GEMM]:
            raise ValueError(
                f"Only {AwqFormat.GEMM} version in supported - not recognized version {self.version}"
            )

        if self.do_fuse or self.fuse_max_seq_len is not None:
            raise ValueError(
                f"fused modules are not supported, got do_fuse={self.do_fuse}, fuse_max_seq_len={self.fuse_max_seq_len}"
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
