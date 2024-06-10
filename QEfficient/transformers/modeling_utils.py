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
    CodeGenForCausalLM,
    CodeGenModel,
)
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    FalconForCausalLM,
    FalconModel,
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
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptForCausalLM, MptModel
from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3ForCausalLM, Phi3Model, Phi3RMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2ForCausalLM, Qwen2Model, Qwen2RMSNorm
from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2ForCausalLM,
    Starcoder2Model,
)

from QEfficient.customop import CustomRMSNormAIC

from .models.codegen.modeling_codegen import (
    QEffCodeGenAttention,
    QEffCodeGenForCausalLM,
    QEffCodeGenModel,
)
from .models.falcon.modeling_falcon import (
    QEffFalconAttention,
    QEffFalconForCausalLM,
    QEffFalconModel,
)
from .models.gpt2.modeling_gpt2 import QEffGPT2Attention, QEffGPT2Block, QEffGPT2LMHeadModel, QEffGPT2Model
from .models.llama.modeling_llama import QEffLlamaAttention, QEffLlamaDecoderLayer, QEffLlamaForCausalLM, QEffLlamaModel
from .models.mistral.modeling_mistral import (
    QEffMistralAttention,
    QEffMistralDecoderLayer,
    QEffMistralForCausalLM,
    QEffMistralModel,
)
from .models.mixtral_moe.modeling_mixtral import (
    QEffMixtralAttention,
    QEffMixtralForCausalLM,
    QEffMixtralModel,
    QEffMixtralSparseMoeBlock,
)
from .models.mpt.modeling_mpt import QEffMptAttention, QEffMptBlock, QEffMptForCausalLM, QEFfMptModel
from .models.phi3.modeling_phi3 import QEffPhi3Attention, QEffPhi3ForCausalLM, QEffPhi3Model
from .models.qwen2.modeling_qwen2 import QEffQwen2Attention, QEffQwen2ForCausalLM, QEffQwen2Model
from .models.starcoder2.modeling_starcoder2 import (
    QEffStarcoder2Attention,
    QEffStarcoder2ForCausalLM,
    QEffStarcoder2Model,
)

# Define a named tuple for ModelArchitectures
# Required for the Automation tool
ModelArchitectures = namedtuple("ModelArchitectures", ["architectures"])
# Create an instance of the named tuple
my_architectures = ModelArchitectures(
    [
        LlamaForCausalLM.__name__,
        MistralForCausalLM.__name__,
        # add post the changes for CB in all the model categories
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
    CodeGenModel: QEffCodeGenModel,
    CodeGenForCausalLM: QEffCodeGenForCausalLM,
    # Mistral model layers
    MistralAttention: QEffMistralAttention,
    MistralModel: QEffMistralModel,
    MistralForCausalLM: QEffMistralForCausalLM,
    MistralDecoderLayer: QEffMistralDecoderLayer,
    MistralRMSNorm: CustomRMSNormAIC,
    # Mixtral model layers
    MixtralAttention: QEffMixtralAttention,
    MixtralModel: QEffMixtralModel,
    MixtralForCausalLM: QEffMixtralForCausalLM,
    MixtralRMSNorm: CustomRMSNormAIC,
    MixtralSparseMoeBlock: QEffMixtralSparseMoeBlock,
    # Phi3 model layers
    Phi3Attention: QEffPhi3Attention,
    Phi3Model: QEffPhi3Model,
    Phi3ForCausalLM: QEffPhi3ForCausalLM,
    Phi3RMSNorm: CustomRMSNormAIC,
    # Falcon model layers
    FalconAttention: QEffFalconAttention,
    FalconForCausalLM: QEffFalconForCausalLM,
    FalconModel: QEffFalconModel,
    # Qwen2 model layers
    Qwen2Attention: QEffQwen2Attention,
    Qwen2ForCausalLM: QEffQwen2ForCausalLM,
    Qwen2Model: QEffQwen2Model,
    Qwen2RMSNorm: CustomRMSNormAIC,
    # Starcoder2 model layers
    Starcoder2Attention: QEffStarcoder2Attention,
    Starcoder2ForCausalLM: QEffStarcoder2ForCausalLM,
    Starcoder2Model: QEffStarcoder2Model,
}
