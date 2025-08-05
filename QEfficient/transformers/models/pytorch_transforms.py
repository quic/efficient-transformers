# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import warnings
from functools import partial
from types import MethodType
from typing import Callable, Optional, Tuple, Union

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
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3RMSNorm,
    Gemma3TextModel,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2LMHeadModel, GPT2Model
from transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
    GPTBigCodeAttention,
    GPTBigCodeBlock,
    GPTBigCodeForCausalLM,
    GPTBigCodeModel,
)
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssForCausalLM,
    GptOssMLP,
    GptOssModel,
    GptOssRMSNorm,
)
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJBlock, GPTJForCausalLM, GPTJModel
from transformers.models.granite.modeling_granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteModel,
    GraniteRMSNorm,
)
from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeAttention,
    GraniteMoeForCausalLM,
    GraniteMoeModel,
    GraniteMoeMoE,
    GraniteMoeParallelExperts,
    GraniteMoeRMSNorm,
    GraniteMoeRotaryEmbedding,
    GraniteMoeTopKGating,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4ForCausalLM,
    Llama4ForConditionalGeneration,
    Llama4Router,
    Llama4TextAttention,
    Llama4TextDecoderLayer,
    Llama4TextExperts,
    Llama4TextModel,
    Llama4TextMoe,
    Llama4TextRMSNorm,
    Llama4VisionAttention,
    Llama4VisionModel,
)
from transformers.models.llava.modeling_llava import (
    LlavaForConditionalGeneration,
)
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralForCausalLM,
    MistralModel,
    MistralRMSNorm,
)
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3ForConditionalGeneration,
    Mistral3Model,
    Mistral3RMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralModel,
    MixtralRMSNorm,
    MixtralSparseMoeBlock,
)
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaForCausalLM,
    MllamaForConditionalGeneration,
    MllamaModel,
    MllamaRotaryEmbedding,
    MllamaSelfAttentionDecoderLayer,
    MllamaTextCrossAttention,
    MllamaTextModel,
    MllamaTextRMSNorm,
    MllamaTextSelfAttention,
    MllamaVisionModel,
)
from transformers.models.mpt.modeling_mpt import MptAttention, MptBlock, MptForCausalLM, MptModel
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2Model,
    Olmo2RMSNorm,
)
from transformers.models.phi.modeling_phi import PhiAttention, PhiDecoderLayer, PhiForCausalLM, PhiModel
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    Phi3DecoderLayer,
    Phi3ForCausalLM,
    Phi3Model,
    Phi3RMSNorm,
)
from transformers.models.pixtral.modeling_pixtral import PixtralRMSNorm, PixtralVisionModel
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2RMSNorm,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLAttention,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLVisionAttention,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2RMSNorm as Qwen2_5RMSNorm,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3RMSNorm,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
)
from transformers.models.starcoder2.modeling_starcoder2 import (
    Starcoder2Attention,
    Starcoder2DecoderLayer,
    Starcoder2ForCausalLM,
    Starcoder2Model,
)
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperEncoder,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperPositionalEmbedding,
)

from QEfficient.base.pytorch_transforms import ExternalModuleMapperTransform, ModuleMappingTransform
from QEfficient.customop import CustomRMSNormAIC, GemmaCustomRMSNormAIC
from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP, PooledModel, validate_user_pooling_function
from QEfficient.transformers.models.codegen.modeling_codegen import (
    QEffCodeGenAttention,
    QEffCodeGenBlock,
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
from QEfficient.transformers.models.gemma3.modeling_gemma3 import (
    QEffGemma3Attention,
    QEffGemma3CustomRMSNormAIC,
    QEffGemma3DecoderLayer,
    QEffGemma3ForCausalLMModel,
    QEffGemma3ForConditionalGeneration,
    QEffGemma3TextModel,
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
from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
    QEffGptOssAttention,
    QEffGptOssDecoderLayer,
    QEffGptOssExperts,
    QEffGptOssForCausalLM,
    QEffGptOssMLP,
    QEffGptOssModel,
)
from QEfficient.transformers.models.gptj.modeling_gptj import (
    QEffGPTJAttention,
    QEffGPTJBlock,
    QEffGPTJForCausalLM,
    QEffGPTJModel,
)
from QEfficient.transformers.models.granite.modeling_granite import (
    QEffGraniteAttention,
    QEffGraniteDecoderLayer,
    QEffGraniteForCausalLM,
    QEffGraniteModel,
)
from QEfficient.transformers.models.granitemoe.modeling_granitemoe import (
    QEffGraniteMoeAttention,
    QEffGraniteMoeForCausalLM,
    QEffGraniteMoeModel,
    QEffGraniteMoeMoE,
    QEffGraniteMoeParallelExperts,
    QEffGraniteMoeRotaryEmbedding,
    QEffGraniteMoeTopKGating,
)
from QEfficient.transformers.models.grok_1.modeling_grok1 import (
    QEFFGrok1CustomRMSNormAIC,
    QEffGrok1DecoderLayer,
    QEffGrok1Model,
    QEffGrok1ModelForCausalLM,
    QEffGrok1MoeBlock,
    QEffGrok1MultiHeadAttention,
)
from QEfficient.transformers.models.internvl.modeling_internvl import (
    QEffInternVisionEmbeddings,
    QEffInternVLModel,
)
from QEfficient.transformers.models.llama.modeling_llama import (
    QEffLlamaAttention,
    QEffLlamaDecoderLayer,
    QEffLlamaForCausalLM,
    QEffLlamaModel,
    QEffLlamaRotaryEmbedding,
)
from QEfficient.transformers.models.llama4.modeling_llama4 import (
    QEffLlama4ForCausalLM,
    QEffLlama4ForConditionalGeneration,
    QEffLlama4Router,
    QEffLlama4TextAttention,
    QEffLlama4TextDecoderLayer,
    QEffLlama4TextExperts,
    QEffLlama4TextModel,
    QEffLlama4TextMoe,
    QEffLlama4VisionAttention,
    QEffLlama4VisionModel,
)
from QEfficient.transformers.models.llava.modeling_llava import (
    QEffLlavaForConditionalGeneration,
)
from QEfficient.transformers.models.llava_next.modeling_llava_next import (
    QEffLlavaNextForConditionalGeneration,
)
from QEfficient.transformers.models.mistral.modeling_mistral import (
    QEffMistralAttention,
    QEffMistralDecoderLayer,
    QEffMistralForCausalLM,
    QEffMistralModel,
)
from QEfficient.transformers.models.mistral3.modeling_mistral3 import (
    QEffMistral3ForConditionalGeneration,
    QEffMistral3Model,
    QEffPixtralVisionModel,
)
from QEfficient.transformers.models.mixtral_moe.modeling_mixtral import (
    QEffMixtralAttention,
    QeffMixtralDecoderLayer,
    QEffMixtralForCausalLM,
    QEffMixtralModel,
    QEffMixtralSparseMoeBlock,
)
from QEfficient.transformers.models.mllama.modeling_mllama import (
    QEffMllamaCrossAttentionDecoderLayer,
    QEffMllamaForCausalLM,
    QEffMllamaForConditionalGeneration,
    QEffMllamaModel,
    QEffMllamaRotaryEmbedding,
    QEffMllamaSelfAttentionDecoderLayer,
    QEffMllamaTextCrossAttentionSingleQPC,
    QEffMllamaTextCrossAttentionTwoQPC,
    QEffMllamaTextModel,
    QEffMllamaTextSelfAttention,
    QEffMllamaVisionModel,
)
from QEfficient.transformers.models.molmo.modeling_molmo import (
    QEffMolmo,
    QEffMolmoBlock,
    QEffMolmoModel,
    QEffMolmoSequentialBlock,
    QEffMultiHeadDotProductAttention,
)
from QEfficient.transformers.models.mpt.modeling_mpt import (
    QEffMptAttention,
    QEffMptBlock,
    QEffMptForCausalLM,
    QEFfMptModel,
)
from QEfficient.transformers.models.olmo2.modeling_olmo2 import (
    QEffOlmo2Attention,
    QEffOlmo2DecoderLayer,
    QEffOlmo2ForCausalLM,
    QEffOlmo2Model,
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
from QEfficient.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    QEffQwen2_5_VisionTransformerPretrainedModel,
    QEffQwen2_5_VLAttention,
    QEffQwen2_5_VLDecoderLayer,
    QEffQwen2_5_VLModel,
    QEffQwen2_5_VLTextModel,
    QEffQwen2_5_VLVisionAttention,
    QEffQwen_2_5_vl_ForConditionalGeneration,
)
from QEfficient.transformers.models.qwen3.modeling_qwen3 import (
    QEffQwen3Attention,
    QEffQwen3DecoderLayer,
    QEffQwen3ForCausalLM,
    QEffQwen3Model,
)
from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import (
    QEffQwen3MoeAttention,
    QEffQwen3MoeDecoderLayer,
    QEffQwen3MoeForCausalLM,
    QEffQwen3MoeModel,
    QEffQwen3MoeRotaryEmbedding,
    QEffQwen3MoeSparseMoeBlock,
)
from QEfficient.transformers.models.starcoder2.modeling_starcoder2 import (
    QEffStarcoder2Attention,
    QEFFStarcoder2DecoderLayer,
    QEffStarcoder2ForCausalLM,
    QEffStarcoder2Model,
)
from QEfficient.transformers.models.whisper.modeling_whisper import (
    QEffWhisperAttention,
    QEffWhisperDecoder,
    QEffWhisperDecoderLayer,
    QEffWhisperEncoder,
    QEffWhisperForConditionalGeneration,
    QEffWhisperModel,
    QEffWhisperPositionalEmbedding,
)
from QEfficient.transformers.post_processing import build_and_attach_mlp, model_type_registry
from QEfficient.transformers.sampler.sampler import sampler_forward
from QEfficient.transformers.spd.spd_transform_forward import tlm_forward

SPD_TARGET = "target"


class CustomOpsTransform(ModuleMappingTransform):
    _module_mapping = {
        GemmaRMSNorm: GemmaCustomRMSNormAIC,
        Gemma2RMSNorm: GemmaCustomRMSNormAIC,
        GptOssRMSNorm: CustomRMSNormAIC,
        LlamaRMSNorm: CustomRMSNormAIC,
        Llama4TextRMSNorm: CustomRMSNormAIC,
        MistralRMSNorm: CustomRMSNormAIC,
        Mistral3RMSNorm: CustomRMSNormAIC,
        MixtralRMSNorm: CustomRMSNormAIC,
        Phi3RMSNorm: CustomRMSNormAIC,
        Qwen2RMSNorm: CustomRMSNormAIC,
        Qwen3RMSNorm: CustomRMSNormAIC,
        Qwen2_5RMSNorm: CustomRMSNormAIC,
        MllamaTextRMSNorm: CustomRMSNormAIC,
        GraniteRMSNorm: CustomRMSNormAIC,
        PixtralRMSNorm: CustomRMSNormAIC,
        GraniteMoeRMSNorm: CustomRMSNormAIC,
        Qwen3MoeRMSNorm: CustomRMSNormAIC,
        Gemma3RMSNorm: QEffGemma3CustomRMSNormAIC,
        Olmo2RMSNorm: CustomRMSNormAIC,
    }


class KVCacheTransform(ModuleMappingTransform):
    _module_mapping = {
        # CodeGen
        CodeGenAttention: QEffCodeGenAttention,
        CodeGenBlock: QEffCodeGenBlock,
        CodeGenModel: QEffCodeGenModel,
        CodeGenForCausalLM: QEffCodeGenForCausalLM,
        # Falcon
        FalconAttention: QEffFalconAttention,
        FalconDecoderLayer: QEffFalconDecoderLayer,
        FalconModel: QEffFalconModel,
        FalconForCausalLM: QEffFalconForCausalLM,
        # GPT2
        GPT2Attention: QEffGPT2Attention,
        GPT2Block: QEffGPT2Block,
        GPT2Model: QEffGPT2Model,
        GPT2LMHeadModel: QEffGPT2LMHeadModel,
        # GPTJ
        GPTJAttention: QEffGPTJAttention,
        GPTJBlock: QEffGPTJBlock,
        GPTJModel: QEffGPTJModel,
        GPTJForCausalLM: QEffGPTJForCausalLM,
        # Llama
        LlamaAttention: QEffLlamaAttention,
        LlamaDecoderLayer: QEffLlamaDecoderLayer,
        LlamaModel: QEffLlamaModel,
        LlamaForCausalLM: QEffLlamaForCausalLM,
        LlamaRotaryEmbedding: QEffLlamaRotaryEmbedding,
        # Llama4
        Llama4TextAttention: QEffLlama4TextAttention,
        Llama4ForCausalLM: QEffLlama4ForCausalLM,
        Llama4TextDecoderLayer: QEffLlama4TextDecoderLayer,
        Llama4TextModel: QEffLlama4TextModel,
        Llama4TextMoe: QEffLlama4TextMoe,
        Llama4ForConditionalGeneration: QEffLlama4ForConditionalGeneration,
        Llama4VisionAttention: QEffLlama4VisionAttention,
        Llama4VisionModel: QEffLlama4VisionModel,
        Llama4TextExperts: QEffLlama4TextExperts,
        Llama4Router: QEffLlama4Router,
        # Llava
        LlavaForConditionalGeneration: QEffLlavaForConditionalGeneration,
        # Llava Next
        LlavaNextForConditionalGeneration: QEffLlavaNextForConditionalGeneration,
        # Gemma
        GemmaAttention: QEffGemmaAttention,
        GemmaDecoderLayer: QEffGemmaDecoderLayer,
        GemmaModel: QEffGemmaModel,
        GemmaForCausalLM: QEffGemmaForCausalLM,
        # Qwen3Moe
        Qwen3MoeForCausalLM: QEffQwen3MoeForCausalLM,
        Qwen3MoeModel: QEffQwen3MoeModel,
        Qwen3MoeDecoderLayer: QEffQwen3MoeDecoderLayer,
        Qwen3MoeAttention: QEffQwen3MoeAttention,
        Qwen3MoeRotaryEmbedding: QEffQwen3MoeRotaryEmbedding,
        Qwen3MoeSparseMoeBlock: QEffQwen3MoeSparseMoeBlock,
        # Gemma2
        Gemma2Attention: QEffGemma2Attention,
        Gemma2DecoderLayer: QEffGemma2DecoderLayer,
        Gemma2Model: QEffGemma2Model,
        Gemma2ForCausalLM: QEffGemma2ForCausalLM,
        # Gemma3
        Gemma3Attention: QEffGemma3Attention,
        Gemma3DecoderLayer: QEffGemma3DecoderLayer,
        Gemma3TextModel: QEffGemma3TextModel,
        Gemma3ForCausalLM: QEffGemma3ForCausalLMModel,
        Gemma3ForConditionalGeneration: QEffGemma3ForConditionalGeneration,
        # GPT_OSS
        GptOssAttention: QEffGptOssAttention,
        GptOssDecoderLayer: QEffGptOssDecoderLayer,
        GptOssModel: QEffGptOssModel,
        GptOssForCausalLM: QEffGptOssForCausalLM,
        GptOssMLP: QEffGptOssMLP,
        GptOssExperts: QEffGptOssExperts,
        # Granite
        GraniteModel: QEffGraniteModel,
        GraniteForCausalLM: QEffGraniteForCausalLM,
        GraniteAttention: QEffGraniteAttention,
        GraniteDecoderLayer: QEffGraniteDecoderLayer,
        # GraniteMoe
        GraniteMoeModel: QEffGraniteMoeModel,
        GraniteMoeForCausalLM: QEffGraniteMoeForCausalLM,
        GraniteMoeAttention: QEffGraniteMoeAttention,
        GraniteMoeRotaryEmbedding: QEffGraniteMoeRotaryEmbedding,
        GraniteMoeParallelExperts: QEffGraniteMoeParallelExperts,
        GraniteMoeTopKGating: QEffGraniteMoeTopKGating,
        GraniteMoeMoE: QEffGraniteMoeMoE,
        # mllama
        MllamaTextRMSNorm: CustomRMSNormAIC,
        MllamaTextSelfAttention: QEffMllamaTextSelfAttention,
        MllamaSelfAttentionDecoderLayer: QEffMllamaSelfAttentionDecoderLayer,
        MllamaModel: QEffMllamaModel,
        MllamaCrossAttentionDecoderLayer: QEffMllamaCrossAttentionDecoderLayer,
        MllamaRotaryEmbedding: QEffMllamaRotaryEmbedding,
        MllamaVisionModel: QEffMllamaVisionModel,
        MllamaTextModel: QEffMllamaTextModel,
        MllamaForCausalLM: QEffMllamaForCausalLM,
        MllamaForConditionalGeneration: QEffMllamaForConditionalGeneration,
        # Mistral
        MistralAttention: QEffMistralAttention,
        MistralDecoderLayer: QEffMistralDecoderLayer,
        MistralModel: QEffMistralModel,
        MistralForCausalLM: QEffMistralForCausalLM,
        # Mistral3
        Mistral3ForConditionalGeneration: QEffMistral3ForConditionalGeneration,
        Mistral3Model: QEffMistral3Model,
        # Mixtral
        MixtralAttention: QEffMixtralAttention,
        MixtralSparseMoeBlock: QEffMixtralSparseMoeBlock,
        MixtralDecoderLayer: QeffMixtralDecoderLayer,
        MixtralModel: QEffMixtralModel,
        MixtralForCausalLM: QEffMixtralForCausalLM,
        # Mpt
        MptAttention: QEffMptAttention,
        MptBlock: QEffMptBlock,
        MptModel: QEFfMptModel,
        MptForCausalLM: QEffMptForCausalLM,
        # Phi3
        Phi3Attention: QEffPhi3Attention,
        Phi3DecoderLayer: QEffPhi3DecoderLayer,
        Phi3Model: QEffPhi3Model,
        Phi3ForCausalLM: QEffPhi3ForCausalLM,
        # Phi
        PhiAttention: QEffPhiAttention,
        PhiDecoderLayer: QEffPhiDecoderLayer,
        PhiModel: QEffPhiModel,
        PhiForCausalLM: QEffPhiForCausalLM,
        # Pixtral
        PixtralVisionModel: QEffPixtralVisionModel,
        # Qwen2
        Qwen2Attention: QEffQwen2Attention,
        Qwen2DecoderLayer: QEffQwen2DecoderLayer,
        Qwen2Model: QEffQwen2Model,
        Qwen2ForCausalLM: QEffQwen2ForCausalLM,
        # Qwen3
        Qwen3Attention: QEffQwen3Attention,
        Qwen3DecoderLayer: QEffQwen3DecoderLayer,
        Qwen3Model: QEffQwen3Model,
        Qwen3ForCausalLM: QEffQwen3ForCausalLM,
        # Qwen2.5 VL
        Qwen2_5_VLForConditionalGeneration: QEffQwen_2_5_vl_ForConditionalGeneration,
        Qwen2_5_VLModel: QEffQwen2_5_VLModel,
        Qwen2_5_VLAttention: QEffQwen2_5_VLAttention,
        Qwen2_5_VLDecoderLayer: QEffQwen2_5_VLDecoderLayer,
        Qwen2_5_VisionTransformerPretrainedModel: QEffQwen2_5_VisionTransformerPretrainedModel,
        Qwen2_5_VLVisionAttention: QEffQwen2_5_VLVisionAttention,
        Qwen2_5_VLTextModel: QEffQwen2_5_VLTextModel,
        # Starcoder2
        Starcoder2Attention: QEffStarcoder2Attention,
        Starcoder2DecoderLayer: QEFFStarcoder2DecoderLayer,
        Starcoder2Model: QEffStarcoder2Model,
        Starcoder2ForCausalLM: QEffStarcoder2ForCausalLM,
        # GptBigcode
        GPTBigCodeAttention: QEffGPTBigCodeAttention,
        GPTBigCodeBlock: QEffGPTBigCodeBlock,
        GPTBigCodeModel: QEffGPTBigCodeModel,
        GPTBigCodeForCausalLM: QEffGPTBigCodeForCausalLM,
        # Olmo2
        Olmo2Attention: QEffOlmo2Attention,
        Olmo2DecoderLayer: QEffOlmo2DecoderLayer,
        Olmo2Model: QEffOlmo2Model,
        Olmo2ForCausalLM: QEffOlmo2ForCausalLM,
        # Whisper encoder and decoder layers
        WhisperPositionalEmbedding: QEffWhisperPositionalEmbedding,
        WhisperAttention: QEffWhisperAttention,
        WhisperDecoderLayer: QEffWhisperDecoderLayer,
        WhisperEncoder: QEffWhisperEncoder,
        WhisperDecoder: QEffWhisperDecoder,
        WhisperModel: QEffWhisperModel,
        WhisperForConditionalGeneration: QEffWhisperForConditionalGeneration,
    }

    @classmethod
    def apply(cls, model: nn.Module) -> Tuple[nn.Module, bool]:
        model, transformed = super().apply(model)
        return model, transformed


class SpDTransform:
    """
    Apply generic QEffForCausalLM forward pass to extract `num_speculative_tokens+1` hidden states before computing logits during decode phase and extract last predicted token during prefill.
    This is only needed if user is exporting Target Language Model (TLM) for Speculative Decoding to validate output logits
    against the speculated tokens from a smaller model.
    Other than the computed logits, there should be no difference between the SpD Transformed model and its corresponding cunterpart.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model.

    Returns:
        :model (nn.Module): PyTorch model.
        :transformed (bool): whether transformation was applied successfully.
    """

    # supported architectures
    _module_mapping = {
        # Llama
        QEffLlamaForCausalLM,
        QEffQwen2ForCausalLM,
        QEffQwen3ForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: Optional[dict] = None, **kwargs) -> Tuple[nn.Module, bool]:
        transformed = False
        pretrained_model_name_or_path_temp = kwargs.pop("pretrained_model_name_or_path", None)
        if qaic_config is None or (speculative_model_type := qaic_config.get("speculative_model_type")) is None:
            return model, transformed
        elif speculative_model_type not in (
            supported_spd_model_types := [SPD_TARGET] + list(model_type_registry.keys())
        ):
            raise ValueError(
                f"Specualtive model type {speculative_model_type} is not supported. we currently only support {supported_spd_model_types}"
            )
        elif (model_class := model.__class__) in cls._module_mapping:
            model.forward = MethodType(tlm_forward, model)
            if speculative_model_type != SPD_TARGET:
                # build and attach draft mlp
                pretrained_model_name_or_path = qaic_config["pretrained_model_name_or_path"]
                model = build_and_attach_mlp(
                    model, pretrained_model_name_or_path, speculative_model_type=speculative_model_type, **kwargs
                )
            transformed = True
        else:
            raise NotImplementedError(
                f"model class {model_class} does not yet support returning multiple logits to keep."
            )
        kwargs["pretrained_model_name_or_path"] = pretrained_model_name_or_path_temp
        return model, transformed


class SamplerTransform:
    """
    Add nodes at the output of any generic QEffForCausalLM model to enable the
    sampling of next tokens at the device (instead of the host) and return the
    next tokens and/or probability distributions.

    Note: To achieve this, the generic QEffForCausalLM model must provide the
    logits as output.

    ``Mandatory`` Args:
        :model (nn.Module): PyTorch model.

    Returns:
        :model (nn.Module): PyTorch model.
        :transformed (bool): whether transformation was applied successfully.
    """

    # supported architectures
    _module_mapping = {
        QEffFalconForCausalLM,
        QEffGemmaForCausalLM,
        QEffGPT2LMHeadModel,
        QEffGPTJForCausalLM,
        QEffGraniteForCausalLM,
        QEffGraniteMoeForCausalLM,
        QEffLlamaForCausalLM,
        QEffMptForCausalLM,
        QEffPhi3ForCausalLM,
        QEffQwen2ForCausalLM,
    }

    @classmethod
    def apply(cls, model: nn.Module, qaic_config: Optional[dict] = None, **kwargs) -> Tuple[nn.Module, bool]:
        transformed = False
        if qaic_config is None or not qaic_config.get("include_sampler", False):
            return model, transformed
        elif (model_class := model.__class__) in cls._module_mapping:
            model.old_forward = model.forward
            model.forward = MethodType(sampler_forward, model)
            transformed = True
        else:
            raise NotImplementedError(f"Model class {model_class} does not support on device sampling.")
        return model, transformed


class VlmKVOffloadTransform(ModuleMappingTransform):
    # supported architectures
    _module_mapping = {
        # Llama
        MllamaTextCrossAttention: QEffMllamaTextCrossAttentionTwoQPC,
    }


class VlmNoKVOffloadTransform(ModuleMappingTransform):
    # supported architectures
    _module_mapping = {
        # Llama
        MllamaTextCrossAttention: QEffMllamaTextCrossAttentionSingleQPC,
    }


class KVCacheExternalModuleMapperTransform(ExternalModuleMapperTransform):
    _match_string_replace_method = {
        "InternVLChatModel": {
            "forward": QEffInternVLModel.forward,
            "get_dummy_inputs": QEffInternVLModel.get_dummy_inputs,
            "get_specializations": QEffInternVLModel.get_specializations,
            "get_onnx_dynamic_axes": QEffInternVLModel.get_onnx_dynamic_axes,
            "get_output_names": QEffInternVLModel.get_output_names,
            "get_inputs_info": QEffInternVLModel.get_inputs_info,
            "get_qeff_vision_encoder": QEffInternVLModel.get_qeff_vision_encoder,
            "get_qeff_language_decoder": QEffInternVLModel.get_qeff_language_decoder,
        },
        "InternVisionEmbeddings": {"forward": QEffInternVisionEmbeddings.forward},
        # Mapping for Molmo
        "MolmoForCausalLM": {
            "forward": QEffMolmoModel.forward,
            "get_qeff_vision_encoder": QEffMolmoModel.get_qeff_vision_encoder,
            "get_qeff_language_decoder": QEffMolmoModel.get_qeff_language_decoder,
            "get_specializations": QEffMolmoModel.get_specializations,
            "get_onnx_dynamic_axes": QEffMolmoModel.get_onnx_dynamic_axes,
            "get_output_names": QEffMolmoModel.get_output_names,
            "get_dummy_inputs": QEffMolmoModel.get_dummy_inputs,
            "get_inputs_info": QEffMolmoModel.get_inputs_info,
        },
        "RMSLayerNorm": {"forward": CustomRMSNormAIC.forward},
        # "MolmoForCausalLM": {"forward": QEffMolmoForCausalLM.forward},
        "Molmo": {"forward": QEffMolmo.forward},
        "MolmoSequentialBlock": {
            "forward": QEffMolmoSequentialBlock.forward,
            "attention": QEffMolmoBlock.attention,
            "__qeff_init__": QEffMolmoBlock.__qeff_init__,
        },
        "MolmoBlock": {
            "attention": QEffMolmoBlock.attention,
            "__qeff_init__": QEffMolmoBlock.__qeff_init__,
        },
        "MultiHeadDotProductAttention": {
            "forward": QEffMultiHeadDotProductAttention.forward,
        },
        # Mapping for grok1 model
        "Grok1ModelForCausalLM": {"forward": QEffGrok1ModelForCausalLM.forward},
        "Grok1Model": {
            "forward": QEffGrok1Model.forward,
            "__qeff_init__": QEffGrok1Model.__qeff_init__,
        },
        "DecoderLayer": {
            "forward": QEffGrok1DecoderLayer.forward,
            "__qeff_init__": QEffGrok1DecoderLayer.__qeff_init__,
        },
        "MoeBlock": {"forward": QEffGrok1MoeBlock.forward},
        "MultiHeadAttention": {
            "forward": QEffGrok1MultiHeadAttention.forward,
        },
        "RMSNorm": {
            "forward": QEFFGrok1CustomRMSNormAIC.forward,
        },
    }

    _match_class_replace_method = {}


class PoolingTransform:
    """
    Apply a pooling transformation to the model. This transformation appends a pooling layer to the model, allowing for the reduction of spatial dimensions in the output.
    The pooling layer can be configured to use different pooling methods, such as max pooling or average pooling.
    """

    @classmethod
    def apply(cls, model: nn.Module, pooling: Union[str, Callable]) -> Tuple[nn.Module, bool]:
        transformed = False
        pooling_method = (
            POOLING_MAP[pooling]
            if isinstance(pooling, str) and pooling in POOLING_MAP
            else validate_user_pooling_function(pooling)
        )
        model = PooledModel(model, pooling_method)
        warnings.warn("Pooling is applied to the model.")
        return model, transformed


def get_decoder_layer_classes_for_export(model: nn.Module) -> set:
    """
    Dynamically determine which DecoderLayer classes should be exported as functions
    based on the model's architecture using the existing KVCacheTransform mapping.
    """
    # Define patterns that identify decoder layer classes
    DECODER_LAYER_PATTERNS = ["DecoderLayer", "Block", "Layer"]

    # Get all QEff classes that are decoder layers from the existing mapping
    decoder_layer_classes = set()

    for original_class, qeff_class in KVCacheTransform._module_mapping.items():
        # Check if the QEff class name contains decoder layer patterns
        qeff_class_name = qeff_class.__name__
        if any(pattern in qeff_class_name for pattern in DECODER_LAYER_PATTERNS):
            decoder_layer_classes.add(qeff_class)

    # Filter to only include classes that are actually used in the current model
    model_decoder_classes = set()
    for module in model.modules():
        if module.__class__ in decoder_layer_classes:
            model_decoder_classes.add(module.__class__)

    return model_decoder_classes


class BlockedKVAttentionTransform:
    _module_mapping = {
        QEffLlamaAttention,
        QEffQwen2_5_VLAttention,
    }

    @classmethod
    def apply(cls, model: nn.Module, num_kv_blocks) -> Tuple[nn.Module, bool]:
        transformed = False
        for module in model.modules():
            if type(module) in cls._module_mapping:
                repl_module = type(module)
                module.__class__ = repl_module
                module.forward = MethodType(partial(repl_module.forward, num_kv_blocks=num_kv_blocks), module)
                transformed = True  # Set to True if at least one transformation occurs
            elif module.__class__.__name__.endswith("Attention") and type(module) not in cls._module_mapping:
                warnings.warn(f"KV blocking is not yet supported for {type(module)}.")
        return model, transformed
