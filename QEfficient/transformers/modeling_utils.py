# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from collections import namedtuple
from typing import Dict, Type

import torch.nn as nn
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenBlock,
    CodeGenForCausalLM,
    CodeGenModel,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model
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
    MistralRotaryEmbedding,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralBLockSparseTop2MLP,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
    MixtralSparseMoeBlock,
)
from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptForCausalLM, MptModel

from QEfficient.customop import CustomRMSNormAIC

from .models.codegen.modeling_codegen import (
    QEffCodeGenAttention,
    QEffCodeGenBlock,
    QEffCodeGenForCausalLM,
    QEffCodeGenModel,
)
from .models.gpt2.modeling_gpt2 import QEffGPT2Attention, QEffGPT2Block, QEffGPT2LMHeadModel, QEffGPT2Model
from .models.llama.modeling_llama import (
    QEffLlamaAttention,
    QEffLlamaDecoderLayer,
    QEffLlamaForCausalLM,
    QEffLlamaModel,
)
from .models.mistral.modeling_mistral import (
    QEffMistralAttention,
    QEffMistralDecoderLayer,
    QEffMistralForCausalLM,
    QEffMistralModel,
    QEffMistralRotaryEmbedding,
)
from .models.mixtral_moe.modeling_mixtral import (
    QEffMixtralAttention,
    QEffMixtralBLockSparseTop2MLP,
    QEffMixtralDecoderLayer,
    QEffMixtralForCausalLM,
    QEffMixtralModel,
    QEffMixtralRotaryEmbedding,
    QEffMixtralSparseMoeBlock,
)
from .models.mpt.modeling_mpt import QEffMptAttention, QEffMptBlock, QEffMptForCausalLM, QEFfMptModel

# Define a named tuple for ModelArchitectures
# Required for the Automation tool
ModelArchitectures = namedtuple("ModelArchitectures", ["architectures"])
# Create an instance of the named tuple
my_architectures = ModelArchitectures(
    [
        GPT2LMHeadModel.__name__,
        MptForCausalLM.__name__,
        CodeGenForCausalLM.__name__,
        LlamaForCausalLM.__name__,
        MistralForCausalLM.__name__,
        MixtralForCausalLM.__name__,
    ]
)

# Define a transformers layers to QEff layers dictionary
# While onboarding new models make sure to add the new layer maps to this dictionary.
TransformersToQEffModulesDict: Dict[Type[nn.Module], Type[nn.Module]] = {
    # GPT model layers
    GPT2Model: QEffGPT2Model,
    GPT2Block: QEffGPT2Block,
    GPT2Attention: QEffGPT2Attention,
    GPT2LMHeadModel: QEffGPT2LMHeadModel,
    # Llama model layers
    LlamaModel: QEffLlamaModel,
    LlamaAttention: QEffLlamaAttention,
    LlamaForCausalLM: QEffLlamaForCausalLM,
    LlamaDecoderLayer: QEffLlamaDecoderLayer,
    LlamaRMSNorm: CustomRMSNormAIC,
    # MPT model layers
    MptAttention: QEffMptAttention,
    MptBlock: QEffMptBlock,
    MptModel: QEFfMptModel,
    MptForCausalLM: QEffMptForCausalLM,
    # CodeGen model layers
    CodeGenAttention: QEffCodeGenAttention,
    CodeGenBlock: QEffCodeGenBlock,
    CodeGenModel: QEffCodeGenModel,
    CodeGenForCausalLM: QEffCodeGenForCausalLM,
    # Mistral model layers
    MistralAttention: QEffMistralAttention,
    MistralModel: QEffMistralModel,
    MistralDecoderLayer: QEffMistralDecoderLayer,
    MistralForCausalLM: QEffMistralForCausalLM,
    MistralRotaryEmbedding: QEffMistralRotaryEmbedding,
    MistralRMSNorm: CustomRMSNormAIC,
    # Mixtral model layers
    MixtralAttention: QEffMixtralAttention,
    MixtralModel: QEffMixtralModel,
    MixtralDecoderLayer: QEffMixtralDecoderLayer,
    MixtralForCausalLM: QEffMixtralForCausalLM,
    MixtralRotaryEmbedding: QEffMixtralRotaryEmbedding,
    MixtralRMSNorm: CustomRMSNormAIC,
    MixtralSparseMoeBlock: QEffMixtralSparseMoeBlock,
    MixtralBLockSparseTop2MLP:QEffMixtralBLockSparseTop2MLP,
}
