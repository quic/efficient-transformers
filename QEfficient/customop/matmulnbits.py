import math

import torch
from torch import nn


def pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
    if pack_tensor.shape[0] == ori_int_tensor.shape[0]:
        ori_int_tensor = ori_int_tensor.T
        pack_tensor = pack_tensor.T
    compress_ratio = 32 // bits
    i = 0
    row = 0
    while row < pack_tensor.shape[0]:
        if bits in [2, 4, 8]:
            for j in range(i, i + compress_ratio):
                pack_tensor[row:] |= ori_int_tensor[j::compress_ratio] << (bits * (j - i))
            break
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")


def general_pack_on_row(pack_tensor, ori_int32_tensor, bits):
    assert pack_tensor.shape[0] == ori_int32_tensor.shape[0] or pack_tensor.shape[1] == ori_int32_tensor.shape[1], ""
    pack_tensor.mul_(0)
    if bits in [2, 4, 8]:
        return pack_on_row_fast_248bit(pack_tensor, ori_int32_tensor, bits)
    else:
        raise NotImplementedError()


class QuantLinearTorchFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, qself_qweight, qself_scales, qself_qzeros, g_idx, bits, groupsize, in_features, out_features):
        input_tuple = (x, qself_qweight, qself_scales, qself_qzeros)
        input_tuple += (g_idx,) if g_idx is not None else ()
        # import ipdb; ipdb.set_trace()
        return g.op(
            "com.microsoft::MatMulNBits",
            *input_tuple,
            outputs=1,
            K_i=in_features,
            N_i=out_features,
            bits_i=bits,
            block_size_i=groupsize,
        )

    @staticmethod
    def forward(ctx, x, qself_qweight, qself_scales, qself_qzeros, g_idx, bits, groupsize, in_features, out_features):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(x.shape[:-1] + (out_features,), dtype=x.dtype, device=x.device).float()
        # qself_qweight=qself_qweight.reshape(qself_qweight.shape[0], -1)
        fp_weight = dequantize_blockwise_4bits(
            qself_qweight, qself_scales, qself_qzeros, g_idx, in_features, out_features
        )[0].float()

        return torch.matmul(x.float(), fp_weight.T.float())


def QuantLinearTorchFunction_forward(
    inputs, qweight, scales, qzeros, g_idx, bits, groupsize, in_features, out_features
):
    assert bits == 4, "Only 4 bits are supported."
    out = QuantLinearTorchFunction().apply(
        inputs, qweight, scales, qzeros, g_idx, bits, groupsize, in_features, out_features
    )
    return out


def dequantize_blockwise_4bits(quant_values, scale, zero_point, g_idx, rows, cols):
    expand_quant_value = (
        quant_values.unsqueeze(-1) >> torch.tensor([[[[0, 4]]]], dtype=torch.int32, device=quant_values.device)
    ) & 0x0F
    expand_quant_value = expand_quant_value.reshape(*quant_values.shape[:-1], -1)
    aligned_scale = scale.reshape(*quant_values.shape[:-1], 1)
    if zero_point.dtype == scale.dtype:
        expand_zero_point = zero_point.reshape(*quant_values.shape[:-1], -1)
    else:
        expand_zero_point = (
            zero_point.unsqueeze(-1) >> torch.tensor([[[[0, 4]]]], dtype=torch.int32, device=quant_values.device)
        ) & 0x0F
        try:
            expand_zero_point = expand_zero_point.reshape(*quant_values.shape[:-1], -1)
        # FIXME: remove try-except
        except RuntimeError:
            expand_zero_point = expand_zero_point.reshape(quant_values.shape[0], -1, 1)
            expand_zero_point = expand_zero_point[:, : quant_values.shape[1]]
    if g_idx is not None and g_idx[:32].sum().item() != 0:
        float_values = (
            (expand_quant_value.reshape(expand_quant_value.shape[0], -1) - expand_zero_point[:, g_idx, 0])
            * aligned_scale[:, g_idx, 0]
        ).to(scale.dtype)
    else:
        float_values = ((expand_quant_value - expand_zero_point) * aligned_scale).to(scale.dtype)
    float_values = float_values.reshape(cols, -1)
    if rows != float_values.shape[-1]:
        float_values = float_values[:, :rows]
        expand_zero_point = expand_zero_point[:, :rows]
    if expand_zero_point.ndim == 3:
        expand_zero_point = expand_zero_point.squeeze(-1)
    if aligned_scale.ndim == 3:
        aligned_scale = aligned_scale.squeeze(-1)

    return float_values, expand_zero_point, aligned_scale


class QuantLinearORT(nn.Module):
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 3, 4, 5, 6, 7, 8]:
            raise NotImplementedError("Only 2,4,5,6,7,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.orig_fp_weight = None
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.act_order = None
        self.pack_mode = "ORT"
        q_rows = infeatures // self.groupsize
        self.register_buffer(
            "qweight",
            torch.zeros((outfeatures, q_rows, self.groupsize // (8 // bits)), dtype=torch.uint8),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros((q_rows + (q_rows & 1)) * (outfeatures // 8 * self.bits), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales", torch.zeros((math.ceil(infeatures / self.groupsize) * outfeatures), dtype=torch.float16)
        )
        self.register_buffer("g_idx", torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def _quant_weight(self, weight, scales, zeros, g_idx, need_transpose=True):
        scale_zeros = zeros * scales
        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        intweight_T = torch.round(((weight + scale_zeros_mat) / scale_mat).float()).to(torch.int)
        return intweight_T

    def pack_on_device(self, intweight_gpu, intzeros_T):
        self.act_order = self.g_idx[: self.groupsize // self.bits].sum().item() != 0
        assert self.bits == 4, "only 4bit is supported by ONNXRUNTIME for now."
        intzeros_pt = intzeros_T.T if intzeros_T.dtype == self.scales.dtype else intzeros_T.T.byte()
        scales_pt = self.scales.T.to(intweight_gpu.device)
        intweight_pt = intweight_gpu.byte()
        block_size = self.groupsize

        rows, cols = intweight_pt.shape
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            intweight_pt = torch.nn.functional.pad(intweight_pt, (0, 0, 0, pad_len), "constant", 0)
        intzeros_pt = torch.nn.functional.pad(intzeros_pt, (0, intzeros_pt.shape[-1] & 1, 0, 0), "constant", 0)

        if intzeros_T.dtype != self.scales.dtype:
            intzeros_pt = (intzeros_pt[:, 0::2]) | (intzeros_pt[:, 1::2] << 4)
            intzeros_pt = intzeros_pt.reshape(-1)

        intweight_pt_T = intweight_gpu.T
        intweight_pt_T = (intweight_pt_T[:, 0::2]) | (intweight_pt_T[:, 1::2] << 4)
        intweight_pt_T = intweight_pt_T.reshape(cols, k_blocks, blob_size)

        scales_pt = scales_pt.reshape(-1)

        assert self.qweight.shape == intweight_pt_T.shape
        assert self.qzeros.shape == intzeros_pt.shape or self.qzeros.dtype != intzeros_pt.dtype

        self.scales = scales_pt.float().contiguous()
        self.qweight = intweight_pt_T.contiguous().byte()
        if intzeros_T.dtype != self.scales.dtype:
            self.qzeros = intzeros_pt.contiguous().byte()
        else:
            self.qzeros = intzeros_pt.contiguous()

    def accelerate_pack_on_device(self, layer_weight, scales, zeros, g_idx=None, device="cuda"):
        self.scales = scales.T.contiguous().half().to("cpu", non_blocking=True)
        if g_idx is None:
            g_idx = self.g_idx.to(device) if g_idx is None else g_idx
        else:
            self.g_idx = g_idx.clone().to("cpu")

        intweight_gpu = self._quant_weight(layer_weight.T, scales.T, zeros.T, g_idx)

        qzeros = zeros.T.contiguous()

        return self.pack_on_device(intweight_gpu, qzeros)

    def pack(self, linear, scales, zeros, g_idx=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scales = scales.to(device)
        zeros = zeros.to(device)
        layer_weight = linear.weight.data.to(device)
        return self.accelerate_pack_on_device(layer_weight, scales, zeros, g_idx, device)

    def forward(self, x):
        out = QuantLinearTorchFunction_forward(
            x,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx if self.act_order else None,
            self.bits,
            self.groupsize,
            self.infeatures,
            self.outfeatures,
        )
        out = out + self.bias if self.bias is not None else out
        return out
