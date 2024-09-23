# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Tuple

import transformers
from torch import nn
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenBlock,
    CodeGenForCausalLM,
    CodeGenModel,
)
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconDecoderLayer,
    FalconForCausalLM,
    FalconModel,
)
from transformers.models.gemma.modeling_gemma import (
    GemmaAttention,
    GemmaDecoderLayer,
    GemmaForCausalLM,
    GemmaModel,
    GemmaRMSNorm,
)
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
    Gemma2Model,
    Gemma2RMSNorm,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    GPTBigCodeAttention,
    GPTBigCodeBlock,
    GPTBigCodeForCausalLM,
    GPTBigCodeModel,
)
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJForCausalLM, GPTJModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralModel,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptForCausalLM, MptModel
from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer, PhiForCausalLM, PhiModel
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    Phi3DecoderLayer,
    Phi3ForCausalLM,
    Phi3Model,
    Phi3RMSNorm,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2RMSNorm,
)
from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2DecoderLayer,
    Starcoder2ForCausalLM,
    Starcoder2Model,
)

from QEfficient.base.pytorch_transforms import ModuleMappingTransform
from QEfficient.customop import CustomRMSNormAIC, GemmaCustomRMSNormAIC
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.models.codegen.modeling_codegen import (
    QEffCodeGenAttention,
    QeffCodeGenBlock,
    QEffCodeGenForCausalLM,
    QEffCodeGenModel,
)
from QEfficient.transformers.models.falcon.modeling_falcon import (
    QEffFalconAttention,
    QEffFalconDecoderLayer,
    QEffFalconForCausalLM,
    QEffFalconModel,
)
from QEfficient.transformers.models.gemma.modeling_gemma import (
    QEffGemmaAttention,
    QEffGemmaDecoderLayer,
    QEffGemmaForCausalLM,
    QEffGemmaModel,
)
from QEfficient.transformers.models.gemma2.modeling_gemma2 import (
    QEffGemma2Attention,
    QEffGemma2DecoderLayer,
    QEffGemma2ForCausalLM,
    QEffGemma2Model,
)
from QEfficient.transformers.models.gpt2.modeling_gpt2 import (
    QEffGPT2Attention,
    QEffGPT2Block,
    QEffGPT2LMHeadModel,
    QEffGPT2Model,
)
from QEfficient.transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    QEffGPTBigCodeAttention,
    QEffGPTBigCodeBlock,
    QEffGPTBigCodeForCausalLM,
    QEffGPTBigCodeModel,
)
from QEfficient.transformers.models.gptj.modeling_gptj import (
    QEffGPTJAttention,
    QEffGPTJBlock,
    QEffGPTJForCausalLM,
    QEffGPTJModel,
)
from QEfficient.transformers.models.llama.modeling_llama import (
    QEffLlamaAttention,
    QEffLlamaDecoderLayer,
    QEffLlamaForCausalLM,
    QEffLlamaModel,
)
from QEfficient.transformers.models.mistral.modeling_mistral import (
    QEffMistralAttention,
    QEffMistralDecoderLayer,
    QEffMistralForCausalLM,
    QEffMistralModel,
)
from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import (
    QEffMixtralAttention,
    QeffMixtralDecoderLayer,
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
from QEfficient.transformers.models.phi.modeling_phi import (
    QEffPhiAttention,
    QEffPhiDecoderLayer,
    QEffPhiForCausalLM,
    QEffPhiModel,
)
from QEfficient.transformers.models.phi3.modeling_phi3 import (
    QEffPhi3Attention,
    QEffPhi3DecoderLayer,
    QEffPhi3ForCausalLM,
    QEffPhi3Model,
)
from QEfficient.transformers.models.qwen2.modeling_qwen2 import (
    QEffQwen2Attention,
    QEffQwen2DecoderLayer,
    QEffQwen2ForCausalLM,
    QEffQwen2Model,
)
from QEfficient.transformers.models.starcoder2.modeling_starcoder2 import (
    QEffStarcoder2Attention,
    QEFFStarcoder2DecoderLayer,
    QEffStarcoder2ForCausalLM,
    QEffStarcoder2Model,
)


class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {
        GemmaRMSNorm: GemmaCustomRMSNormAIC,
        Gemma2RMSNorm: GemmaCustomRMSNormAIC,
        LlamaRMSNorm: CustomRMSNormAIC,
        MistralRMSNorm: CustomRMSNormAIC,
        MixtralRMSNorm: CustomRMSNormAIC,
        Phi3RMSNorm: CustomRMSNormAIC,
        Qwen2RMSNorm: CustomRMSNormAIC,
    }


class KVCacheTransform(ModuleMappingTransform):
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
        # Gemma
        GemmaAttention: QEffGemmaAttention,
        GemmaModel: QEffGemmaModel,
        GemmaForCausalLM: QEffGemmaForCausalLM,
        # Gemma2
        Gemma2Attention: QEffGemma2Attention,
        Gemma2Model: QEffGemma2Model,
        Gemma2ForCausalLM: QEffGemma2ForCausalLM,
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
        # GptBigcode
        GPTBigCodeAttention: QEffGPTBigCodeAttention,
        GPTBigCodeBlock: QEffGPTBigCodeBlock,
        GPTBigCodeModel: QEffGPTBigCodeModel,
        GPTBigCodeForCausalLM: QEffGPTBigCodeForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        # FIXME: see if we can merge into _module_mapping dict
        transformers.cache_utils.DynamicCache.update = QEffDynamicCache.update
        return model, transformed


class CBTransform(KVCacheTransform):
    _module_mapping = {
        **KVCacheTransform._module_mapping,  # Unpack existing KV mapping
        # Llama
        LlamaDecoderLayer: QEffLlamaDecoderLayer,
        # Gemma
        GemmaDecoderLayer: QEffGemmaDecoderLayer,
        # Gemma2
        Gemma2DecoderLayer: QEffGemma2DecoderLayer,
        # Mistral
        MistralDecoderLayer: QEffMistralDecoderLayer,
        # Mixtral
        MixtralDecoderLayer: QeffMixtralDecoderLayer,
        # Starcoder2
        Starcoder2DecoderLayer: QEFFStarcoder2DecoderLayer,
        # Phi
        PhiDecoderLayer: QEffPhiDecoderLayer,
        # Phi3
        Phi3DecoderLayer: QEffPhi3DecoderLayer,
        # Qwen2
        Qwen2DecoderLayer: QEffQwen2DecoderLayer,
        # Codegen
        CodeGenBlock: QeffCodeGenBlock,
        # Falcon
        FalconDecoderLayer: QEffFalconDecoderLayer,
        # GPT-J
        GPTJBlock: QEffGPTJBlock,
    }
