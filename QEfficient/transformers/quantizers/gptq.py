import math

import torch
from torch import nn

from QEfficient.transformers.quantizers.qunatizer_utils import dequantize_gptq


class QuantLinearGPTQ(nn.Module):
    """
    A quantized linear layer using GPTQ (Generalized Post-Training Quantization).
    This class supports only 4-bit quantization and is compatible with QuantLinearORT.

    Attributes:
        infeatures (int): The number of input features.
        outfeatures (int): The number of output features.
        bits (int): The number of bits used for quantization (must be 4).
        act_order (None or bool): The activation order.
        orig_fp_weight (None or torch.Tensor): The original floating-point weights.
        maxq (int): The maximum quantization value.
        groupsize (int): The group size for quantization.
        pack_mode (str): The packing mode, set to "GPTQ".
        qweight (torch.Tensor): The quantized weight tensor.
        qzeros (torch.Tensor): The quantized zeros tensor.
        scales (torch.Tensor): The scales tensor.
        g_idx (torch.Tensor): The group index tensor.
        bias (torch.Tensor or None): The bias tensor, if applicable.
    """

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits != 4:
            raise NotImplementedError("Only 4 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.act_order = None
        self.orig_fp_weight = None
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.pack_mode = "GPTQ"

        # For compatibility with QuantLinearORT
        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32),
        )
        self.register_buffer(
            "scales",
            torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((outfeatures), dtype=torch.float16),
            )
        else:
            self.bias = None

    def handle(self):
        shifts = torch.arange(0, 32, self.bits, dtype=torch.int32, device=self.qzeros.device).unsqueeze(0)
        izeros = torch.bitwise_right_shift(self.qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int32  # smallest dtype available
        )
        izeros = torch.bitwise_and(izeros[0], (2**self.bits) - 1).view(-1, 1, izeros[0].size(1) * izeros[0].size(2))
        izeros = izeros.view(izeros.shape[0], -1)
        izeros += 1
        device = "cpu"
        qzeros = self.qzeros.to(device)
        qzeros.mul_(0)
        if qzeros.shape[0] == izeros.shape[0]:
            qzeros = qzeros.T
            izeros = izeros.T
        compress_ratio = 32 // self.bits
        i = 0
        row = 0
        while row < qzeros.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + compress_ratio):
                    qzeros[row:] |= izeros[j::compress_ratio] << (self.bits * (j - i))
                break
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")
        qzeros = qzeros.T
        self.qzeros = qzeros.to("cpu", non_blocking=True)

    def forward(self, x):
        # Only Inference supported
        if self.act_order is None:
            self.act_order = self.g_idx[: self.groupsize].sum() != 0
        out, _, _ = dequantize_gptq(self.qweight.T, self.qzeros, self.scales, self.bits, self.g_idx)
        out = torch.matmul(x.float(), out.float())
        out = out + self.bias if self.bias is not None else out

        return out
