from torch import nn
from typing import Optional, Union
import math 
import torch
import os
import tqdm
BLOCK_PATTERNS = [
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "model.layers",
]
from QEfficient.transformers.quantizers.qunatizer_utils import dequantize_gptq

def DequantizeLinearBlockWise(qweight, scales, qzeros, groupsize, bits, in_features, g_idx):
    # scales = scales.reshape(-1, 1, scales.shape[-1])
    # import ipdb; ipdb.set_trace()
    
    scales = scales.view(-1, 1, scales.size(-1))
    #creates a 2D tensor with a single row containing a sequence of integers,
    #which will be used later in the function for bitwise operations.
    wf = torch.arange(0, 32, bits, dtype=torch.int32, device=qweight.device).unsqueeze(0)
    # expand is removed as torch will auto broadcast to relavant dimension

    zeros = torch.bitwise_right_shift(qzeros.unsqueeze(2), wf).to(torch.int8)
    zeros = torch.bitwise_and(zeros, (2 ** bits) - 1).view(-1, 1, zeros.size(1) * zeros.size(2))
    # expand is removed as torch will auto broadcast to relavant dimension
    weight = torch.bitwise_right_shift(qweight.unsqueeze(1), wf.unsqueeze(-1)).to(torch.int8)
    # import ipdb; ipdb.set_trace()
    print(weight.shape)
    weight = torch.bitwise_and(weight, (2 ** bits) - 1)
    
    scale_zeros = zeros * scales
    weight = weight.reshape(-1, groupsize, weight.shape[-1])
    weight = (scales * weight - scale_zeros.float())
    weight = weight.reshape(-1, weight.shape[2])
    return weight

class QuantLinearGPTQ(nn.Module):

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
        self.register_buffer(
            'qweight', 
            torch.zeros
            (
                (infeatures // 32 * self.bits, outfeatures),
                dtype=torch.int32
            ),
        )
        self.register_buffer(
            'qzeros',
            torch.zeros(
                (math.ceil(infeatures / self.groupsize),outfeatures // 32 * self.bits),
                dtype=torch.int32
            ),
        )
        self.register_buffer(
            'scales',
            torch.zeros
            (
                (math.ceil(infeatures / self.groupsize), outfeatures),
                dtype=torch.float16
            ),
        )
        self.register_buffer(
            'g_idx',
            torch.tensor(
                [i // self.groupsize for i in range(infeatures)],
                dtype=torch.int32
            ),
        )
        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(
                    (outfeatures),
                    dtype=torch.float16
                ),
            )
        else:
            self.bias = None

    def handle(self):
        
        shifts = torch.arange(0, 32, self.bits, dtype=torch.int32, device=self.qzeros.device).unsqueeze(0)
        izeros = torch.bitwise_right_shift(self.qzeros[:, :, None], shifts[None, None, :]).to(
            torch.int32  # smallest dtype available
        )
        izeros = torch.bitwise_and(izeros[0], (2 ** self.bits) - 1).view(-1, 1, izeros[0].size(1) * izeros[0].size(2))
        izeros = izeros.view(izeros.shape[0], -1)
        izeros+=1
        device = "cpu"
        qzeros = self.qzeros.to(device)
        qzeros.mul_(0)
        if qzeros.shape[0] == izeros.shape[0]:
            qzeros=qzeros.T
            izeros=izeros.T
        compress_ratio = (32 // self.bits)
        i = 0
        row = 0
        while row < qzeros.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + compress_ratio):
                    qzeros[row:] |= izeros[j::compress_ratio] << (
                        self.bits * (j - i))
                break
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")
        qzeros=qzeros.T
        self.qzeros = qzeros.to("cpu", non_blocking=True)
        
    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        if self.act_order is None:
            self.act_order = self.g_idx[:self.groupsize].sum() != 0
        out1 = dequantize_gptq(self.qweight, self.qzeros, self.scales, self.bits,self.g_idx)
        out=DequantizeLinearBlockWise(self.qweight, self.scales,self.qzeros, self.groupsize, self.bits,self.infeatures,self.outfeatures)
        print(torch.equal(out,out1))
        print(torch.equal(out,out1.T))
        # out=torch.transpose(out,0,1)
        # import ipdb; ipdb.set_trace()
        out =torch.matmul(x.float(), out.float())
        out = out + self.bias if self.bias is not None else out
       
        return out
