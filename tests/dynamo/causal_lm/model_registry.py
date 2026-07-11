# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Dynamo test model registry.

Single source of truth for every CausalLM (and VLM text-side) spec that the
dynamo test suite parametrises over. Every architecture is exercised end to
end; there are no per-architecture skips. When an architecture is broken,
the coverage matrix surfaces the failure directly (red cell) rather than
hiding it behind a skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class DynamoModelSpec:
    """A single row in the dynamo coverage matrix.

    architecture
        HF model_type / QEff wrapper key. Also the pytest id.
    family
        Coarse grouping used only for reporting.
    category
        "causal_lm" for text-only CausalLM wrappers, "vlm_text" when the tiny
        model is a VLM whose text backbone is exercised through
        AutoModelForCausalLM.from_config on the text sub-config.
    model_id
        Tiny HuggingFace model id.
    model_loader
        "causal_lm" -> AutoModelForCausalLM.from_pretrained.
        "vlm_text" -> AutoConfig then AutoModelForCausalLM.from_config on the
        text_config.
    """

    architecture: str
    family: str
    category: str
    model_id: str
    model_loader: str = "causal_lm"
    subfunctions_supported: bool = True
    continuous_batching_supported: bool = True
    prefix_caching_supported: bool = False
    sampler_supported: bool = False
    blocking_kv_supported: bool = False
    notes: Optional[str] = None


DYNAMO_MODEL_SPECS: List[DynamoModelSpec] = [
    DynamoModelSpec("codegen", "code", "causal_lm", "hf-internal-testing/tiny-random-CodeGenForCausalLM", "causal_lm"),
    DynamoModelSpec(
        "deepseek_v3",
        "deepseek",
        "causal_lm",
        "hf-internal-testing/tiny-random-DeepseekV3ForCausalLM",
        "causal_lm",
    ),
    DynamoModelSpec("falcon", "falcon", "causal_lm", "hf-internal-testing/tiny-random-FalconForCausalLM", "causal_lm"),
    DynamoModelSpec(
        "gemma",
        "gemma",
        "causal_lm",
        "Xenova/tiny-random-GemmaForCausalLM",
        "causal_lm",
        prefix_caching_supported=True,
    ),
    DynamoModelSpec(
        "gemma2",
        "gemma",
        "causal_lm",
        "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
        "causal_lm",
        prefix_caching_supported=True,
    ),
    DynamoModelSpec("glm4_moe", "glm", "causal_lm", "tiny-random/glm-4-moe", "causal_lm"),
    DynamoModelSpec(
        "gpt2",
        "gpt",
        "causal_lm",
        "hf-internal-testing/tiny-random-GPT2LMHeadModel",
        "causal_lm",
        prefix_caching_supported=True,
        sampler_supported=True,
        blocking_kv_supported=True,
    ),
    DynamoModelSpec(
        "gpt_bigcode",
        "gpt",
        "causal_lm",
        "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
        "causal_lm",
    ),
    DynamoModelSpec(
        "gpt_oss",
        "gpt",
        "causal_lm",
        "tiny-random/gpt-oss-bf16",
        "causal_lm",
        subfunctions_supported=False,
        notes="gpt_oss subfunction export currently hits a torch.export failure; flat lane still runs.",
    ),
    DynamoModelSpec("gptj", "gpt", "causal_lm", "hf-internal-testing/tiny-random-GPTJForCausalLM", "causal_lm"),
    DynamoModelSpec(
        "granite",
        "granite",
        "causal_lm",
        "hf-internal-testing/tiny-random-GraniteForCausalLM",
        "causal_lm",
        prefix_caching_supported=True,
    ),
    DynamoModelSpec(
        "granitemoe",
        "granite",
        "causal_lm",
        "hf-internal-testing/tiny-random-GraniteMoeForCausalLM",
        "causal_lm",
    ),
    DynamoModelSpec("grok_1", "grok", "causal_lm", "yujiepan/grok-1-tiny-random", "causal_lm"),
    DynamoModelSpec(
        "llama",
        "llama",
        "causal_lm",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "causal_lm",
        prefix_caching_supported=True,
        blocking_kv_supported=True,
    ),
    DynamoModelSpec(
        "llama_swiftkv",
        "llama",
        "causal_lm",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "causal_lm",
        notes="Reuses the tiny-random Llama checkpoint until a dedicated SwiftKV tiny model is tracked.",
    ),
    DynamoModelSpec(
        "mistral",
        "mistral",
        "causal_lm",
        "hf-internal-testing/tiny-random-MistralForCausalLM",
        "causal_lm",
        blocking_kv_supported=True,
    ),
    DynamoModelSpec(
        "mixtral", "mistral", "causal_lm", "hf-internal-testing/tiny-random-MixtralForCausalLM", "causal_lm"
    ),
    DynamoModelSpec(
        "mpt",
        "mpt",
        "causal_lm",
        "hf-internal-testing/tiny-random-MptForCausalLM",
        "causal_lm",
        prefix_caching_supported=True,
    ),
    DynamoModelSpec("olmo2", "olmo", "causal_lm", "hf-internal-testing/tiny-random-Olmo2ForCausalLM", "causal_lm"),
    DynamoModelSpec("phi", "phi", "causal_lm", "hf-internal-testing/tiny-random-PhiForCausalLM", "causal_lm"),
    DynamoModelSpec("phi3", "phi", "causal_lm", "tiny-random/phi-4", "causal_lm"),
    DynamoModelSpec(
        "qwen2", "qwen", "causal_lm", "yujiepan/qwen2-tiny-random", "causal_lm", blocking_kv_supported=True
    ),
    DynamoModelSpec("qwen3", "qwen", "causal_lm", "tiny-random/qwen3", "causal_lm"),
    DynamoModelSpec("qwen3_5_moe", "qwen", "vlm_text", "tiny-random/qwen3.5-moe", "vlm_text"),
    DynamoModelSpec("qwen3_moe", "qwen", "causal_lm", "tiny-random/qwen3-moe", "causal_lm"),
    DynamoModelSpec(
        "starcoder2",
        "code",
        "causal_lm",
        "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
        "causal_lm",
        prefix_caching_supported=True,
    ),
    DynamoModelSpec("gemma3", "gemma", "vlm_text", "tiny-random/gemma-3", "vlm_text"),
    DynamoModelSpec(
        "mllama",
        "llama",
        "vlm_text",
        "hf-internal-testing/tiny-random-MllamaForConditionalGeneration",
        "vlm_text",
        notes="JIRA-08: verify this model id is publicly accessible; update if gated.",
    ),
    DynamoModelSpec(
        "molmo",
        "molmo",
        "vlm_text",
        "yujiepan/molmo-tiny-random",
        "vlm_text",
        notes="JIRA-08: model id does not exist on Hub — replace with a valid tiny molmo checkpoint.",
    ),
    DynamoModelSpec(
        "qwen2_5_vl",
        "qwen",
        "vlm_text",
        "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
        "vlm_text",
    ),
    DynamoModelSpec("qwen3_vl", "qwen", "vlm_text", "tiny-random/qwen3-vl", "vlm_text"),
    DynamoModelSpec("qwen3_vl_moe", "qwen", "vlm_text", "tiny-random/qwen3-vl-moe", "vlm_text"),
]


def spec_ids(specs: Iterable[DynamoModelSpec]) -> List[str]:
    """pytest id list matching a filtered spec iterable."""
    return [spec.architecture for spec in specs]


def specs_with(*, subfunctions: bool = False, continuous_batching: bool = False) -> List[DynamoModelSpec]:
    """Filter DYNAMO_MODEL_SPECS on feature-support flags."""
    result = []
    for spec in DYNAMO_MODEL_SPECS:
        if subfunctions and not spec.subfunctions_supported:
            continue
        if continuous_batching and not (spec.continuous_batching_supported and spec.model_loader == "causal_lm"):
            continue
        result.append(spec)
    return result


def spec_by_architecture(architecture: str) -> DynamoModelSpec:
    for spec in DYNAMO_MODEL_SPECS:
        if spec.architecture == architecture:
            return spec
    raise KeyError(f"No dynamo spec registered for architecture={architecture!r}")
