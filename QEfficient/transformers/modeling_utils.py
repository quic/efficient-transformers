# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import hashlib
from collections import namedtuple

import torch.nn as nn
import transformers
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention,
    CodeGenBlock,
    CodeGenForCausalLM,
    CodeGenModel,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model
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
    MistralRotaryEmbedding,
)
from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptForCausalLM, MptModel

from QEfficient.customop import CustomRMSNormAIC
from QEfficient.utils.logging_utils import logger

from .modeling_attn_mask_utils import (
    QEffAttentionMaskConverter,
    _qeff_prepare_4d_attention_mask,
    _qeff_prepare_4d_causal_attention_mask,
)
from .modeling_outputs import (
    QEffBaseModelOutputWithPast,
    QEffBaseModelOutputWithPastAndCrossAttentions,
    QEffCausalLMOutputWithCrossAttentions,
    QEffCausalLMOutputWithPast,
)
from .models.codegen.modeling_codegen import (
    QEffCodeGenAttention,
    QEffCodeGenBlock,
    QEffCodeGenForCausalLM,
    QEffCodeGenModel,
)
from .models.gpt2.modeling_gpt2 import QEffGPT2Attention, QEffGPT2Block, QEffGPT2LMHeadModel, QEffGPT2Model
from .models.gptj.modeling_gptj import QEffGPTJAttention, QEffGPTJBlock, QEffGPTJForCausalLM, QEffGPTJModel
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
from .models.mpt.modeling_mpt import QEffMptAttention, QEffMptBlock, QEffMptForCausalLM, QEFfMptModel

# Define a named tuple for ModelArchitectures
# Required for the Automation tool
ModelArchitectures = namedtuple("ModelArchitectures", ["architectures"])
# Create an instance of the named tuple
my_architectures = ModelArchitectures(
    [
        GPT2LMHeadModel.__name__,
        GPTJForCausalLM.__name__,
        MptForCausalLM.__name__,
        CodeGenForCausalLM.__name__,
        LlamaForCausalLM.__name__,
        MistralForCausalLM.__name__,
    ]
)

# Define a transformers layers to QEff layers dictionary
# While onboarding new models make sure to add the new layer maps to this dictionary.
TransformersToQEffModulesDict = {
    # GPT2 model layers
    GPT2Model: QEffGPT2Model,
    GPT2Block: QEffGPT2Block,
    GPT2Attention: QEffGPT2Attention,
    GPT2LMHeadModel: QEffGPT2LMHeadModel,
    # GPTJ model layers
    GPTJModel: QEffGPTJModel,
    GPTJBlock: QEffGPTJBlock,
    GPTJAttention: QEffGPTJAttention,
    GPTJForCausalLM: QEffGPTJForCausalLM,
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
}


def get_params_hash(model: nn.Module) -> str:
    """
    Creates a Hash of all the parameters values i.e. weights using SHA256 algo.
    --------
    :param model: torch.nn.Module. Base PyTorch model.
    :returns: str. Hash string
    """
    hasher = hashlib.sha256()
    for _, params in model.named_parameters():
        hasher.update(params.data.numpy().tobytes())

    return hasher.hexdigest()


def replace_module_with_qeff_layers(model: nn.Module) -> None:
    """
    Replaces the transformers nn.Module classes with optmized QEff classes in place.
    ----------
    :param model: torch.nn.Module. Base PyTorch model.
    """
    # Replace if module class is registed in TransformersToQEffModulesDict
    target_module = TransformersToQEffModulesDict.get(model.__class__)
    if target_module is not None:
        model.__class__ = target_module

    # Iterate over child modules
    for _, module in model.named_children():
        replace_module_with_qeff_layers(module)


def transform(model: nn.Module, form_factor: str = "cloud") -> nn.Module:
    """
    Replaces some Transformers' methods for equivalent methods optimized for AI 100.
    ---------
    Args:
    param model (torch.nn.Module): PyTorch model.
    form_factor(str): form factor configuration for optmizing the model, available options=["cloud", "edge"].

    Returns:
    torch.nn.Module: PyTorch Module with replaced QEff layers.
    """

    # Introducnig qeff_transformed attribue in model to check status of transform
    if getattr(model, "qeff_transformed", False):
        print("Model is already transformed")
        return model

    if form_factor == "cloud":
        # Get Hash of all params for checking later
        prior_params_hash = get_params_hash(model)
        logger.warning(f"The model {model.__class__} layers has been upadted to QEff layers in-place")
        # Replace with QEff layers
        replace_module_with_qeff_layers(model)

        # Check with new params hash
        later_params_hash = get_params_hash(model)
        assert (
            prior_params_hash == later_params_hash
        ), "Weights were changed in the transform process, please report an issue"

        # Replace the modeling output classes
        transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions = (
            QEffBaseModelOutputWithPastAndCrossAttentions
        )
        transformers.modeling_outputs.CausalLMOutputWithCrossAttentions = QEffCausalLMOutputWithCrossAttentions
        transformers.modeling_outputs.BaseModelOutputWithPast = QEffBaseModelOutputWithPast
        transformers.modeling_outputs.CausalLMOutputWithPast = QEffCausalLMOutputWithPast

        # Replace the modeling attn util classes and functions
        transformers.modeling_attn_mask_utils.AttentionMaskConverter = QEffAttentionMaskConverter
        transformers.modeling_attn_mask_utils._prepare_4d_attention_mask = _qeff_prepare_4d_attention_mask
        transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask = _qeff_prepare_4d_causal_attention_mask

        setattr(model, "qeff_transformed", True)
        return model.eval()

    elif form_factor == "edge":
        # Add changes for the edge usecase
        raise NotImplementedError("We currently only support cloud form factor!")
