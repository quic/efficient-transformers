import transformers
from torch import nn
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenForCausalLM,
    CodeGenModel,
)
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconForCausalLM,
    FalconModel,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJForCausalLM, GPTJModel
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaModel, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
    MistralModel,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptForCausalLM, MptModel
from transformers.models.phi.modeling_phi import PhiAttention, PhiForCausalLM, PhiModel
from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3ForCausalLM, Phi3Model, Phi3RMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2ForCausalLM, Qwen2Model, Qwen2RMSNorm
from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2ForCausalLM,
    Starcoder2Model,
)

from QEfficient.base.pytorch_transforms import ModuleMapping
from QEfficient.customop import CustomRMSNormAIC
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.models.codegen.modeling_codegen import (
    QEffCodeGenAttention,
    QEffCodeGenForCausalLM,
    QEffCodeGenModel,
)
from QEfficient.transformers.models.falcon.modeling_falcon import (
    QEffFalconAttention,
    QEffFalconForCausalLM,
    QEffFalconModel,
)
from QEfficient.transformers.models.gpt2.modeling_gpt2 import (
    QEffGPT2Attention,
    QEffGPT2Block,
    QEffGPT2LMHeadModel,
    QEffGPT2Model,
)
from QEfficient.transformers.models.gptj.modeling_gptj import QEffGPTJAttention, QEffGPTJForCausalLM, QEffGPTJModel
from QEfficient.transformers.models.llama.modeling_llama import (
    QEffLlamaAttention,
    QEffLlamaForCausalLM,
    QEffLlamaModel,
)
from QEfficient.transformers.models.mistral.modeling_mistral import (
    QEffMistralAttention,
    QEffMistralForCausalLM,
    QEffMistralModel,
)
from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import (
    QEffMixtralAttention,
    QEffMixtralForCausalLM,
    QEffMixtralModel,
    QEffMixtralSparseMoeBlock,
)
from QEfficient.transformers.models.mpt.modeling_mpt import (
    QEffMptAttention,
    QEffMptBlock,
    QEffMptForCausalLM,
    QEFfMptModel,
)
from QEfficient.transformers.models.phi.modeling_phi import QEffPhiAttention, QEffPhiForCausalLM, QEffPhiModel
from QEfficient.transformers.models.phi3.modeling_phi3 import QEffPhi3Attention, QEffPhi3ForCausalLM, QEffPhi3Model
from QEfficient.transformers.models.qwen2.modeling_qwen2 import QEffQwen2Attention, QEffQwen2ForCausalLM, QEffQwen2Model
from QEfficient.transformers.models.starcoder2.modeling_starcoder2 import (
    QEffStarcoder2Attention,
    QEffStarcoder2ForCausalLM,
    QEffStarcoder2Model,
)


class CustomOpsTransform(ModuleMapping):
    _module_mapping = {
        LlamaRMSNorm: CustomRMSNormAIC,
        MistralRMSNorm: CustomRMSNormAIC,
        MixtralRMSNorm: CustomRMSNormAIC,
        Phi3RMSNorm: CustomRMSNormAIC,
        Qwen2RMSNorm: CustomRMSNormAIC,
    }


class KVCacheTransform(ModuleMapping):
    _module_mapping = {
        # CodeGen
        CodeGenAttention: QEffCodeGenAttention,
        CodeGenModel: QEffCodeGenModel,
        CodeGenForCausalLM: QEffCodeGenForCausalLM,
        # Falcon
        FalconAttention: QEffFalconAttention,
        FalconModel: QEffFalconModel,
        FalconForCausalLM: QEffFalconForCausalLM,
        # GPT2
        GPT2Attention: QEffGPT2Attention,
        GPT2Block: QEffGPT2Block,
        GPT2Model: QEffGPT2Model,
        GPT2LMHeadModel: QEffGPT2LMHeadModel,
        # GPTJ
        GPTJAttention: QEffGPTJAttention,
        GPTJModel: QEffGPTJModel,
        GPTJForCausalLM: QEffGPTJForCausalLM,
        # Llama
        LlamaAttention: QEffLlamaAttention,
        LlamaModel: QEffLlamaModel,
        LlamaForCausalLM: QEffLlamaForCausalLM,
        # Mistral
        MistralAttention: QEffMistralAttention,
        MistralModel: QEffMistralModel,
        MistralForCausalLM: QEffMistralForCausalLM,
        # Mixtral
        MixtralAttention: QEffMixtralAttention,
        MixtralSparseMoeBlock: QEffMixtralSparseMoeBlock,
        MixtralModel: QEffMixtralModel,
        MixtralForCausalLM: QEffMixtralForCausalLM,
        # Mpt
        MptAttention: QEffMptAttention,
        MptBlock: QEffMptBlock,
        MptModel: QEFfMptModel,
        MptForCausalLM: QEffMptForCausalLM,
        # Phi3
        Phi3Attention: QEffPhi3Attention,
        Phi3Model: QEffPhi3Model,
        Phi3ForCausalLM: QEffPhi3ForCausalLM,
        # Phi
        PhiAttention: QEffPhiAttention,
        PhiModel: QEffPhiModel,
        PhiForCausalLM: QEffPhiForCausalLM,
        # Qwen2
        Qwen2Attention: QEffQwen2Attention,
        Qwen2Model: QEffQwen2Model,
        Qwen2ForCausalLM: QEffQwen2ForCausalLM,
        # Starcoder2
        Starcoder2Attention: QEffStarcoder2Attention,
        Starcoder2Model: QEffStarcoder2Model,
        Starcoder2ForCausalLM: QEffStarcoder2ForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module):
        model, transformed = super().apply(model)
        # FIXME: see if we can merge into _module_mapping dict
        transformers.cache_utils.DynamicCache.update = QEffDynamicCache.update
        return model, transformed
