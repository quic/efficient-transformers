from torch import nn

from QEfficient.base.pytorch_transforms import ModuleMutatorTransform
from QEfficient.transformers.quantizers.awq import WQLinear_GEMM

class AwqToOnnxTransform(ModuleMutatorTransform):
    _match_class = WQLinear_GEMM
    
    @classmethod
    def mutate(cls, original_module: nn.Module, parent_module: nn.Module):
        fp16_weight, scales, zeros = unpack(original_module)
