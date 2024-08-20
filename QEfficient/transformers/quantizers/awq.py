# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn

# def unpack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
#     need_transpose = False
#     if pack_tensor.shape[0] != ori_int_tensor.shape[0]:
#         need_transpose = True
#         pack_tensor = pack_tensor.T
#     wf = torch.arange(0, 32, bits).to(pack_tensor.device).unsqueeze(0)
#     out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, 2), wf.unsqueeze(0))
#     out = out.reshape(pack_tensor.shape[0], -1)
#     torch.bitwise_and(out, (2**bits) - 1, out=out).int()
#     if need_transpose:
#         out = out.T.contiguous()

#     ori_int_tensor.copy_(out)


# def general_unpack_on_row(pack_tensor, ori_int32_tensor, bits: int):
#     assert pack_tensor.shape[0] == ori_int32_tensor.shape[0] or pack_tensor.shape[1] == ori_int32_tensor.shape[1], ""
#     ori_int32_tensor.mul_(0)
#     if bits in [2, 4, 8]:
#         return unpack_on_row_fast_248bit(pack_tensor, ori_int32_tensor, bits)
#     else:
#         raise NotImplementedError()


class WQLinear_GEMM(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()

        if w_bit != 4:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features

        # quick sanity check (make sure alignment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        self.g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    def forward(self, x):
        with torch.no_grad():
            out_shape = x.shape[:-1] + (self.out_features,)

            out = dequantize_gemm(self.qweight, self.qzeros, self.scales, self.w_bit, self.group_size)
            out = torch.matmul(x.float(), out.float())

            out = out + self.bias if self.bias is not None else out
            out = out.reshape(out_shape)

        return out

    # def _dequant_weight(self, intweight, scales, zeros, g_idx):
    #     scale_zeros = zeros * scales
    #     scale_mat = scales[g_idx]
    #     scale_zeros_mat = scale_zeros[g_idx]
    #     qdq_weight_T = intweight * scale_mat - scale_zeros_mat.half()

    #     return qdq_weight_T

    # def reverse_reorder_int_tensor(self, int_tensor):
    #     int_tensor = int_tensor.T.contiguous()
    #     compress_ratio = 32 // self.w_bit
    #     assert int_tensor.shape[-1] % compress_ratio == 0
    #     if self.w_bit == 4:
    #         order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    #     else:
    #         raise NotImplementedError("Only 4-bit are supported for now.")
    #     order_tensor = torch.tensor(order_map, dtype=torch.int32, device=int_tensor.device).reshape(1, -1)
    #     order_tensor = order_tensor.repeat(int_tensor.shape[1] // compress_ratio, 1)
    #     order_tensor = order_tensor + torch.arange(
    #         0, int_tensor.shape[1], compress_ratio, dtype=torch.int32, device=int_tensor.device
    #     ).reshape(-1, 1)
    #     order_tensor = order_tensor.reshape(-1)

    #     reverse_order_tensor = torch.arange(order_tensor.shape[0]).to(int_tensor.device)[order_tensor]
    #     reverse_order_tensor = reverse_order_tensor[order_tensor]
    #     int_tensor = int_tensor[:, reverse_order_tensor]
    #     return int_tensor

    # def unpack_qzeros(self, device):
    #     qzeros = self.qzeros.to(device)
    #     zeros = torch.zeros((self.in_features // self.group_size, self.out_features), dtype=torch.int32, device=device)
    #     general_unpack_on_row(qzeros, zeros, self.w_bit)
    #     zeros = zeros.T.contiguous()
    #     zeros = self.reverse_reorder_int_tensor(zeros)
    #     return zeros

    # def unpack_qweight(self, device):
    #     qweight = self.qweight.to(device)
    #     # weight_dim0 = self.infeatures

    #     qweight = qweight.T.contiguous()
    #     weight_dim0 = self.out_features

    #     weight = torch.zeros((weight_dim0, qweight.shape[1]), dtype=torch.int32, device=device)
    #     general_unpack_on_row(qweight, weight, self.w_bit)
    #     weight = self.reverse_reorder_int_tensor(weight)

    #     return weight

    # def unpack(self):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     scales = self.scales.to(device)

    #     zeros = self.unpack_qzeros(device)
    #     weight = self.unpack_qweight(device)

    #     fp16_weight = self._dequant_weight(weight, scales, zeros, self.g_idx.to(device)).T
    #     # free memory
    #     weight = weight.to("cpu", non_blocking=True)
    #     # weight = (scales * (weight - zeros))
    #     # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    #     fp16_weight = fp16_weight.to("cpu", non_blocking=True)
    #     zeros = zeros.to("cpu", non_blocking=True)
    #     scales = scales.to("cpu", non_blocking=True)
    #     return (fp16_weight, scales, zeros)


def unpack_awq_weights_and_zeros(qweight: torch.Tensor, qzeros: torch.Tensor, bits: int):
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    # unpacking columnwise
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8  # smallest dtype available
    )
    iweights = iweights.view(iweights.shape[0], -1)

    # unpacking columnwise
    if qzeros is not None:
        izeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int8  # smallest dtype available
        )
        izeros = izeros.view(izeros.shape[0], -1)
    else:
        izeros = qzeros

    return iweights, izeros


def reverse_awq_order(iweights: torch.Tensor, izeros: torch.Tensor, bits: int):
    reverse_order_tensor = torch.arange(
        iweights.shape[-1],
        dtype=torch.int32,
        device=izeros.device,
    )
    reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
    reverse_order_tensor = reverse_order_tensor[:, [0, 4, 1, 5, 2, 6, 3, 7]]
    reverse_order_tensor = reverse_order_tensor.view(-1)

    if izeros is not None:
        izeros = izeros[:, reverse_order_tensor]
    iweights = iweights[:, reverse_order_tensor]

    return iweights, izeros


def dequantize_gemm(qweight, qzeros, scales, bits, group_size):
    # Unpack the qweight and qzeros tensors
    scales, iweight, izeros = unpack_awq_weights(qweight, qzeros, scales, bits, group_size)

    # fp16 weights
    scales = scales.repeat_interleave(group_size, dim=0)
    izeros = izeros.repeat_interleave(group_size, dim=0)

    iweight = (iweight - izeros) * scales

    return iweight


def unpack_awq_weights(qweight, qzeros, scales, bits, group_size):
    iweight, izeros = unpack_awq_weights_and_zeros(qweight, qzeros, bits)
    # Reverse the order of the iweight and izeros tensors
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    # overflow checks
    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    return scales, iweight, izeros
