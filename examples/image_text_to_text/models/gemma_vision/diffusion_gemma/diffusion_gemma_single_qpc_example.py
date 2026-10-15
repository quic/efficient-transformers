# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""DiffusionGemma unified, split, or staged-prefill execution on Cloud AI 100."""

import argparse

from transformers import AutoConfig, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.transformers.models.diffusion_gemma_single_qpc_example_utils import (
    build_step_callback,
    clean_diffusion_text,
    compile_split_qpcs,
    compile_unified_qpc,
    diffusion_gemma_prefill_split_qpc_chunked,
    prepare_prompt_inputs,
)

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
CTX_LEN = 1024 # * 25
CANVAS_LENGTH = 256
DIFFUSION_STEPS = 48
NUM_CORES = 16
NUM_DEVICES = 4
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
IMAGE_PROMPT = "Describe this image in detail in not less than 512 tokens"
# TEXT_PROMPT = "What is the capital city of Zimbabwe? Answer in one sentence."
# TEXT_PROMPT = "What are the seven continents? Answer in one sentence."
# TEXT_PROMPT = "Can you summarize Romeo and Juliet story by William Shakespere? Explain in more detail with answer no less than 1280 tokens"
TEXT_PROMPT = "What is diffusion based generative learning?"
# TEXT_PROMPT = "How to make pizza?"
# TEXT_PROMPT = "What is diffusion based generative learning? Answer in one sentence."

TEXT_PROMPT_big = """
You are an expert researcher specializing in large language models, probabilistic generative modeling,
diffusion processes, scalable Transformer architectures, and long-context evaluation methodologies.

You are also deeply familiar with autoregressive decoding, diffusion-based generative systems,
sampling efficiency trade-offs, positional encoding techniques, and failure modes
in large-scale sequence generation.

Below is a long technical document. Read it carefully in its entirety.
The document intentionally contains repetition, reinforcement of concepts,
rephrasing across sections, and overlapping explanations.
All of this content is relevant and should be treated as authoritative.

After reading the document, answer the questions at the end.
Base your answers strictly and exclusively on the document content.
Do not rely on outside knowledge or assumptions.

==================== DOCUMENT ====================

Large Language Diffusion Models (LLaDA) propose an alternative paradigm to traditional
autoregressive language modeling approaches. While autoregressive models such as GPT,
GPT‑Neo, LLaMA, and similar architectures generate sequences token by token using
a strict left‑to‑right ordering and causal masking, diffusion-based language models
treat text generation as a gradual denoising process.

In autoregressive language modeling, generation proceeds sequentially.
At each step, the model predicts the probability distribution of the next token,
conditioned on all previously generated tokens.
Once a token is generated, it becomes fixed and cannot be revised.
This irreversible commitment is a defining characteristic of autoregressive decoding.

By contrast, Large Language Diffusion Models (LLaDA) operate via a forward and reverse process.
In the forward diffusion process, a clean text sequence is progressively corrupted.
This corruption is performed by randomly masking tokens according to a timestep variable t,
where t is sampled uniformly from the interval [0, 1].

At t = 0, the sequence remains fully intact and uncorrupted.
As t increases, a growing fraction of tokens are masked.
At t = 1, the sequence is entirely masked, containing no observable tokens.

The reverse diffusion process aims to recover the original sequence.
Starting from a fully or partially masked input, the model repeatedly predicts
denoised token distributions until a clean sequence is obtained.
This reverse process is implemented using a Transformer network with bidirectional
self-attention rather than causal attention.

Because the Transformer uses bidirectional attention,
each token can attend to tokens both to its left and to its right.
This architectural choice allows the model to leverage global context
when predicting or revising any part of the sequence.

One major conceptual advantage of diffusion-based language models is their ability
to revise previously generated tokens.
Unlike autoregressive models, which commit to each generated token permanently,
diffusion models can adjust earlier predictions during later denoising steps.
This enables correction of earlier errors, improved global consistency,
and better performance on tasks requiring holistic understanding of the sequence.

Examples of such tasks include text infilling, where missing spans inside a document
must be reconstructed; document editing, where existing text must be modified;
and long-context reasoning, where conclusions depend on interactions between
distant parts of the input.

Despite these advantages, diffusion-based language models introduce new challenges.
A primary challenge is sampling efficiency.
Diffusion models typically require many denoising steps,
each involving a full forward pass through a large Transformer.
As a result, naive diffusion sampling can be substantially slower
than autoregressive decoding, which generates one token per step.

To mitigate this inefficiency, LLaDA introduces block-wise diffusion
and semi-autoregressive decoding strategies.
In block-wise diffusion, the output sequence is divided into fixed-length blocks,
such as blocks containing tens or hundreds of tokens.

Each block initially starts in a fully masked state.
Blocks are then denoised independently while being conditioned
on all previously generated blocks.
This creates a hybrid structure: autoregressive at the block level,
but diffusion-based within each block.

Block-wise diffusion significantly reduces computational cost.
Instead of performing diffusion over the entire sequence at once,
the model only needs to denoise a limited number of tokens per block.
This preserves many of the benefits of diffusion modeling,
including local revision and context-aware correction,
while improving practical decoding speed.

However, block-wise diffusion does not fully eliminate all limitations.
Early versions of LLaDA exhibit instability during very long,
free-form generation tasks.
This instability often arises late in the generation process
and manifests in several observable ways.

Common failure modes include repetitive token patterns,
looping phrases, semantic drift, and collapse into low-information output.
These phenomena suggest that uncertainty can accumulate
across successive denoising steps and blocks.

Later versions of the model, specifically LLaDA‑2.0 and LLaDA‑2.1,
introduce additional architectural and algorithmic improvements
to mitigate these instability issues.

One such improvement is explicit token editing.
This mechanism allows the model to selectively revisit
and modify previously generated tokens, even across block boundaries.

Another improvement is confidence-aware parallel decoding.
In this approach, the model estimates its own uncertainty
and adapts the denoising strategy accordingly,
allocating more refinement steps to uncertain regions.

Additionally, re-editing mechanisms are introduced,
allowing earlier blocks to be revisited and refined
when inconsistencies or contradictions are detected later in the sequence.
These mechanisms help maintain global coherence
over long generation horizons.

Positional encoding is another critical component of LLaDA.
The model uses Rotary Positional Embeddings (RoPE)
to encode relative token positions.
RoPE enables efficient attention over sequences
by embedding positional information directly into attention computations.

To extend the usable context length,
scaling factors can be applied to the RoPE theta parameters.
This permits attention over much longer contexts,
such as 16K or 24K tokens, exceeding the model’s original training window.

However, extending context length via positional encoding
does not automatically guarantee stable long-range generation.
Even when the attention mechanism technically spans long contexts,
the iterative diffusion process can amplify small prediction errors
over many denoising steps.

As a result, positional encoding extensions alone
are insufficient to ensure coherent and stable long-form output.
Additional architectural safeguards are required,
such as re-editing, confidence modeling, and constrained decoding.

Large Language Diffusion Models demonstrate strong performance
on tasks that benefit from global context and holistic optimization.
These include structured reasoning, document revision,
controlled generation, and infilling tasks.

At the same time, traditional autoregressive models
remain highly effective for tasks requiring very long,
uninterrupted text generation at high speed.
Their simplicity, stability, and efficient sampling
make them well-suited for many production scenarios.

The comparison between diffusion-based models and autoregressive models
highlights a fundamental trade-off.
Diffusion models emphasize flexibility, global optimization,
and revision capability.
Autoregressive models emphasize efficiency, determinism,
and stable incremental generation.

[The remainder of this document intentionally restates and reinforces
the above concepts through paraphrasing, elaboration, and repetition.
This redundancy is deliberate and should not be ignored.]

Large Language Diffusion Models (LLaDA) again differ from
traditional causal language models in both training objective
and inference procedure.
The diffusion objective focuses on denoising masked tokens,
whereas causal language modeling focuses on next-token prediction.

In practice, these objective differences lead to divergent behaviors.
Diffusion models can improve earlier tokens at later stages,
while autoregressive models cannot.
This allows diffusion systems to correct mistakes,
but also introduces additional complexity and instability risks.

Block-wise diffusion serves as a compromise between
full-sequence diffusion and purely autoregressive decoding.
By structuring generation into blocks,
LLaDA aims to balance computational efficiency
with the flexibility of diffusion-based revision.

The instability observed in early LLaDA versions
underscores that long-context modeling is not solely
a function of attention span.
Stability depends on interaction between sampling strategy,
model confidence, revision mechanisms, and architectural constraints.

Later versions of LLaDA explicitly acknowledge this
and incorporate mechanisms designed to regulate uncertainty,
limit error propagation, and enforce global consistency.

In conclusion, LLaDA represents a promising alternative
to conventional autoregressive language models,
offering new capabilities alongside new challenges.
Its evolution across versions highlights the complexity
of designing stable, efficient, and flexible generative systems.

Large Language Diffusion Models (LLaDA) propose an alternative paradigm to traditional
autoregressive language modeling approaches. While autoregressive models such as GPT,
GPT‑Neo, LLaMA, and similar architectures generate sequences token by token using
a strict left‑to‑right ordering and causal masking, diffusion-based language models
treat text generation as a gradual denoising process.

In autoregressive language modeling, generation proceeds sequentially.
At each step, the model predicts the probability distribution of the next token,
conditioned on all previously generated tokens.
Once a token is generated, it becomes fixed and cannot be revised.
This irreversible commitment is a defining characteristic of autoregressive decoding.

By contrast, Large Language Diffusion Models (LLaDA) operate via a forward and reverse process.
In the forward diffusion process, a clean text sequence is progressively corrupted.
This corruption is performed by randomly masking tokens according to a timestep variable t,
where t is sampled uniformly from the interval [0, 1].

At t = 0, the sequence remains fully intact and uncorrupted.
As t increases, a growing fraction of tokens are masked.
At t = 1, the sequence is entirely masked, containing no observable tokens.

The reverse diffusion process aims to recover the original sequence.
Starting from a fully or partially masked input, the model repeatedly predicts
denoised token distributions until a clean sequence is obtained.
This reverse process is implemented using a Transformer network with bidirectional
self-attention rather than causal attention.

Because the Transformer uses bidirectional attention,
each token can attend to tokens both to its left and to its right.
This architectural choice allows the model to leverage global context
when predicting or revising any part of the sequence.

One major conceptual advantage of diffusion-based language models is their ability
to revise previously generated tokens.
Unlike autoregressive models, which commit to each generated token permanently,
diffusion models can adjust earlier predictions during later denoising steps.
This enables correction of earlier errors, improved global consistency,
and better performance on tasks requiring holistic understanding of the sequence.

Examples of such tasks include text infilling, where missing spans inside a document
must be reconstructed; document editing, where existing text must be modified;
and long-context reasoning, where conclusions depend on interactions between
distant parts of the input.

Despite these advantages, diffusion-based language models introduce new challenges.
A primary challenge is sampling efficiency.
Diffusion models typically require many denoising steps,
each involving a full forward pass through a large Transformer.
As a result, naive diffusion sampling can be substantially slower
than autoregressive decoding, which generates one token per step.

To mitigate this inefficiency, LLaDA introduces block-wise diffusion
and semi-autoregressive decoding strategies.
In block-wise diffusion, the output sequence is divided into fixed-length blocks,
such as blocks containing tens or hundreds of tokens.

Each block initially starts in a fully masked state.
Blocks are then denoised independently while being conditioned
on all previously generated blocks.
This creates a hybrid structure: autoregressive at the block level,
but diffusion-based within each block.

Block-wise diffusion significantly reduces computational cost.
Instead of performing diffusion over the entire sequence at once,
the model only needs to denoise a limited number of tokens per block.
This preserves many of the benefits of diffusion modeling,
including local revision and context-aware correction,
while improving practical decoding speed.

However, block-wise diffusion does not fully eliminate all limitations.
Early versions of LLaDA exhibit instability during very long,
free-form generation tasks.
This instability often arises late in the generation process
and manifests in several observable ways.

Common failure modes include repetitive token patterns,
looping phrases, semantic drift, and collapse into low-information output.
These phenomena suggest that uncertainty can accumulate
across successive denoising steps and blocks.

Later versions of the model, specifically LLaDA‑2.0 and LLaDA‑2.1,
introduce additional architectural and algorithmic improvements
to mitigate these instability issues.

One such improvement is explicit token editing.
This mechanism allows the model to selectively revisit
and modify previously generated tokens, even across block boundaries.

Another improvement is confidence-aware parallel decoding.
In this approach, the model estimates its own uncertainty
and adapts the denoising strategy accordingly,
allocating more refinement steps to uncertain regions.

Additionally, re-editing mechanisms are introduced,
allowing earlier blocks to be revisited and refined
when inconsistencies or contradictions are detected later in the sequence.
These mechanisms help maintain global coherence
over long generation horizons.

Positional encoding is another critical component of LLaDA.
The model uses Rotary Positional Embeddings (RoPE)
to encode relative token positions.
RoPE enables efficient attention over sequences
by embedding positional information directly into attention computations.

To extend the usable context length,
scaling factors can be applied to the RoPE theta parameters.
This permits attention over much longer contexts,
such as 16K or 24K tokens, exceeding the model’s original training window.

However, extending context length via positional encoding
does not automatically guarantee stable long-range generation.
Even when the attention mechanism technically spans long contexts,
the iterative diffusion process can amplify small prediction errors
over many denoising steps.

As a result, positional encoding extensions alone
are insufficient to ensure coherent and stable long-form output.
Additional architectural safeguards are required,
such as re-editing, confidence modeling, and constrained decoding.

Large Language Diffusion Models demonstrate strong performance
on tasks that benefit from global context and holistic optimization.
These include structured reasoning, document revision,
controlled generation, and infilling tasks.

At the same time, traditional autoregressive models
remain highly effective for tasks requiring very long,
uninterrupted text generation at high speed.
Their simplicity, stability, and efficient sampling
make them well-suited for many production scenarios.

The comparison between diffusion-based models and autoregressive models
highlights a fundamental trade-off.
Diffusion models emphasize flexibility, global optimization,
and revision capability.
Autoregressive models emphasize efficiency, determinism,
and stable incremental generation.

[The remainder of this document intentionally restates and reinforces
the above concepts through paraphrasing, elaboration, and repetition.
This redundancy is deliberate and should not be ignored.]

Large Language Diffusion Models (LLaDA) again differ from
traditional causal language models in both training objective
and inference procedure.
The diffusion objective focuses on denoising masked tokens,
whereas causal language modeling focuses on next-token prediction.

In practice, these objective differences lead to divergent behaviors.
Diffusion models can improve earlier tokens at later stages,
while autoregressive models cannot.
This allows diffusion systems to correct mistakes,
but also introduces additional complexity and instability risks.

Block-wise diffusion serves as a compromise between
full-sequence diffusion and purely autoregressive decoding.
By structuring generation into blocks,
LLaDA aims to balance computational efficiency
with the flexibility of diffusion-based revision.

The instability observed in early LLaDA versions
underscores that long-context modeling is not solely
a function of attention span.
Stability depends on interaction between sampling strategy,
model confidence, revision mechanisms, and architectural constraints.

Later versions of LLaDA explicitly acknowledge this
and incorporate mechanisms designed to regulate uncertainty,
limit error propagation, and enforce global consistency.

In conclusion, LLaDA represents a promising alternative
to conventional autoregressive language models,
offering new capabilities alongside new challenges.
Its evolution across versions highlights the complexity
of designing stable, efficient, and flexible generative systems.

Large Language Diffusion Models (LLaDA) propose an alternative paradigm to traditional
autoregressive language modeling approaches. While autoregressive models such as GPT,
GPT‑Neo, LLaMA, and similar architectures generate sequences token by token using
a strict left‑to‑right ordering and causal masking, diffusion-based language models
treat text generation as a gradual denoising process.

In autoregressive language modeling, generation proceeds sequentially.
At each step, the model predicts the probability distribution of the next token,
conditioned on all previously generated tokens.
Once a token is generated, it becomes fixed and cannot be revised.
This irreversible commitment is a defining characteristic of autoregressive decoding.

By contrast, Large Language Diffusion Models (LLaDA) operate via a forward and reverse process.
In the forward diffusion process, a clean text sequence is progressively corrupted.
This corruption is performed by randomly masking tokens according to a timestep variable t,
where t is sampled uniformly from the interval [0, 1].

At t = 0, the sequence remains fully intact and uncorrupted.
As t increases, a growing fraction of tokens are masked.
At t = 1, the sequence is entirely masked, containing no observable tokens.

The reverse diffusion process aims to recover the original sequence.
Starting from a fully or partially masked input, the model repeatedly predicts
denoised token distributions until a clean sequence is obtained.
This reverse process is implemented using a Transformer network with bidirectional
self-attention rather than causal attention.

Because the Transformer uses bidirectional attention,
each token can attend to tokens both to its left and to its right.
This architectural choice allows the model to leverage global context
when predicting or revising any part of the sequence.

One major conceptual advantage of diffusion-based language models is their ability
to revise previously generated tokens.
Unlike autoregressive models, which commit to each generated token permanently,
diffusion models can adjust earlier predictions during later denoising steps.
This enables correction of earlier errors, improved global consistency,
and better performance on tasks requiring holistic understanding of the sequence.

Examples of such tasks include text infilling, where missing spans inside a document
must be reconstructed; document editing, where existing text must be modified;
and long-context reasoning, where conclusions depend on interactions between
distant parts of the input.

Despite these advantages, diffusion-based language models introduce new challenges.
A primary challenge is sampling efficiency.
Diffusion models typically require many denoising steps,
each involving a full forward pass through a large Transformer.
As a result, naive diffusion sampling can be substantially slower
than autoregressive decoding, which generates one token per step.

To mitigate this inefficiency, LLaDA introduces block-wise diffusion
and semi-autoregressive decoding strategies.
In block-wise diffusion, the output sequence is divided into fixed-length blocks,
such as blocks containing tens or hundreds of tokens.

Each block initially starts in a fully masked state.
Blocks are then denoised independently while being conditioned
on all previously generated blocks.
This creates a hybrid structure: autoregressive at the block level,
but diffusion-based within each block.

Block-wise diffusion significantly reduces computational cost.
Instead of performing diffusion over the entire sequence at once,
the model only needs to denoise a limited number of tokens per block.
This preserves many of the benefits of diffusion modeling,
including local revision and context-aware correction,
while improving practical decoding speed.

However, block-wise diffusion does not fully eliminate all limitations.
Early versions of LLaDA exhibit instability during very long,
free-form generation tasks.
This instability often arises late in the generation process
and manifests in several observable ways.

Common failure modes include repetitive token patterns,
looping phrases, semantic drift, and collapse into low-information output.
These phenomena suggest that uncertainty can accumulate
across successive denoising steps and blocks.

Later versions of the model, specifically LLaDA‑2.0 and LLaDA‑2.1,
introduce additional architectural and algorithmic improvements
to mitigate these instability issues.

One such improvement is explicit token editing.
This mechanism allows the model to selectively revisit
and modify previously generated tokens, even across block boundaries.

Another improvement is confidence-aware parallel decoding.
In this approach, the model estimates its own uncertainty
and adapts the denoising strategy accordingly,
allocating more refinement steps to uncertain regions.

Additionally, re-editing mechanisms are introduced,
allowing earlier blocks to be revisited and refined
when inconsistencies or contradictions are detected later in the sequence.
These mechanisms help maintain global coherence
over long generation horizons.

Positional encoding is another critical component of LLaDA.
The model uses Rotary Positional Embeddings (RoPE)
to encode relative token positions.
RoPE enables efficient attention over sequences
by embedding positional information directly into attention computations.

To extend the usable context length,
scaling factors can be applied to the RoPE theta parameters.
This permits attention over much longer contexts,
such as 16K or 24K tokens, exceeding the model’s original training window.

However, extending context length via positional encoding
does not automatically guarantee stable long-range generation.
Even when the attention mechanism technically spans long contexts,
the iterative diffusion process can amplify small prediction errors
over many denoising steps.

As a result, positional encoding extensions alone
are insufficient to ensure coherent and stable long-form output.
Additional architectural safeguards are required,
such as re-editing, confidence modeling, and constrained decoding.

Large Language Diffusion Models demonstrate strong performance
on tasks that benefit from global context and holistic optimization.
These include structured reasoning, document revision,
controlled generation, and infilling tasks.

At the same time, traditional autoregressive models
remain highly effective for tasks requiring very long,
uninterrupted text generation at high speed.
Their simplicity, stability, and efficient sampling
make them well-suited for many production scenarios.

The comparison between diffusion-based models and autoregressive models
highlights a fundamental trade-off.
Diffusion models emphasize flexibility, global optimization,
and revision capability.
Autoregressive models emphasize efficiency, determinism,
and stable incremental generation.

[The remainder of this document intentionally restates and reinforces
the above concepts through paraphrasing, elaboration, and repetition.
This redundancy is deliberate and should not be ignored.]

Large Language Diffusion Models (LLaDA) again differ from
traditional causal language models in both training objective
and inference procedure.
The diffusion objective focuses on denoising masked tokens,
whereas causal language modeling focuses on next-token prediction.

In practice, these objective differences lead to divergent behaviors.
Diffusion models can improve earlier tokens at later stages,
while autoregressive models cannot.
This allows diffusion systems to correct mistakes,
but also introduces additional complexity and instability risks.

Block-wise diffusion serves as a compromise between
full-sequence diffusion and purely autoregressive decoding.
By structuring generation into blocks,
LLaDA aims to balance computational efficiency
with the flexibility of diffusion-based revision.

The instability observed in early LLaDA versions
underscores that long-context modeling is not solely
a function of attention span.
Stability depends on interaction between sampling strategy,
model confidence, revision mechanisms, and architectural constraints.

Later versions of LLaDA explicitly acknowledge this
and incorporate mechanisms designed to regulate uncertainty,
limit error propagation, and enforce global consistency.

In conclusion, LLaDA represents a promising alternative
to conventional autoregressive language models,
offering new capabilities alongside new challenges.
Its evolution across versions highlights the complexity
of designing stable, efficient, and flexible generative systems.

Large Language Diffusion Models (LLaDA) propose an alternative paradigm to traditional
autoregressive language modeling approaches. While autoregressive models such as GPT,
GPT‑Neo, LLaMA, and similar architectures generate sequences token by token using
a strict left‑to‑right ordering and causal masking, diffusion-based language models
treat text generation as a gradual denoising process.

In autoregressive language modeling, generation proceeds sequentially.
At each step, the model predicts the probability distribution of the next token,
conditioned on all previously generated tokens.
Once a token is generated, it becomes fixed and cannot be revised.
This irreversible commitment is a defining characteristic of autoregressive decoding.

By contrast, Large Language Diffusion Models (LLaDA) operate via a forward and reverse process.
In the forward diffusion process, a clean text sequence is progressively corrupted.
This corruption is performed by randomly masking tokens according to a timestep variable t,
where t is sampled uniformly from the interval [0, 1].

At t = 0, the sequence remains fully intact and uncorrupted.
As t increases, a growing fraction of tokens are masked.
At t = 1, the sequence is entirely masked, containing no observable tokens.

The reverse diffusion process aims to recover the original sequence.
Starting from a fully or partially masked input, the model repeatedly predicts
denoised token distributions until a clean sequence is obtained.
This reverse process is implemented using a Transformer network with bidirectional
self-attention rather than causal attention.

Because the Transformer uses bidirectional attention,
each token can attend to tokens both to its left and to its right.
This architectural choice allows the model to leverage global context
when predicting or revising any part of the sequence.

One major conceptual advantage of diffusion-based language models is their ability
to revise previously generated tokens.
Unlike autoregressive models, which commit to each generated token permanently,
diffusion models can adjust earlier predictions during later denoising steps.
This enables correction of earlier errors, improved global consistency,
and better performance on tasks requiring holistic understanding of the sequence.

Examples of such tasks include text infilling, where missing spans inside a document
must be reconstructed; document editing, where existing text must be modified;
and long-context reasoning, where conclusions depend on interactions between
distant parts of the input.

Despite these advantages, diffusion-based language models introduce new challenges.
A primary challenge is sampling efficiency.
Diffusion models typically require many denoising steps,
each involving a full forward pass through a large Transformer.
As a result, naive diffusion sampling can be substantially slower
than autoregressive decoding, which generates one token per step.

To mitigate this inefficiency, LLaDA introduces block-wise diffusion
and semi-autoregressive decoding strategies.
In block-wise diffusion, the output sequence is divided into fixed-length blocks,
such as blocks containing tens or hundreds of tokens.

Each block initially starts in a fully masked state.
Blocks are then denoised independently while being conditioned
on all previously generated blocks.
This creates a hybrid structure: autoregressive at the block level,
but diffusion-based within each block.

Block-wise diffusion significantly reduces computational cost.
Instead of performing diffusion over the entire sequence at once,
the model only needs to denoise a limited number of tokens per block.
This preserves many of the benefits of diffusion modeling,
including local revision and context-aware correction,
while improving practical decoding speed.

However, block-wise diffusion does not fully eliminate all limitations.
Early versions of LLaDA exhibit instability during very long,
free-form generation tasks.
This instability often arises late in the generation process
and manifests in several observable ways.

Common failure modes include repetitive token patterns,
looping phrases, semantic drift, and collapse into low-information output.
These phenomena suggest that uncertainty can accumulate
across successive denoising steps and blocks.

Later versions of the model, specifically LLaDA‑2.0 and LLaDA‑2.1,
introduce additional architectural and algorithmic improvements
to mitigate these instability issues.

One such improvement is explicit token editing.
This mechanism allows the model to selectively revisit
and modify previously generated tokens, even across block boundaries.

Another improvement is confidence-aware parallel decoding.
In this approach, the model estimates its own uncertainty
and adapts the denoising strategy accordingly,
allocating more refinement steps to uncertain regions.

Additionally, re-editing mechanisms are introduced,
allowing earlier blocks to be revisited and refined
when inconsistencies or contradictions are detected later in the sequence.
These mechanisms help maintain global coherence
over long generation horizons.

Positional encoding is another critical component of LLaDA.
The model uses Rotary Positional Embeddings (RoPE)
to encode relative token positions.
RoPE enables efficient attention over sequences
by embedding positional information directly into attention computations.

To extend the usable context length,
scaling factors can be applied to the RoPE theta parameters.
This permits attention over much longer contexts,
such as 16K or 24K tokens, exceeding the model’s original training window.

However, extending context length via positional encoding
does not automatically guarantee stable long-range generation.
Even when the attention mechanism technically spans long contexts,
the iterative diffusion process can amplify small prediction errors
over many denoising steps.

As a result, positional encoding extensions alone
are insufficient to ensure coherent and stable long-form output.
Additional architectural safeguards are required,
such as re-editing, confidence modeling, and constrained decoding.

Large Language Diffusion Models demonstrate strong performance
on tasks that benefit from global context and holistic optimization.
These include structured reasoning, document revision,
controlled generation, and infilling tasks.

At the same time, traditional autoregressive models
remain highly effective for tasks requiring very long,
uninterrupted text generation at high speed.
Their simplicity, stability, and efficient sampling
make them well-suited for many production scenarios.

The comparison between diffusion-based models and autoregressive models
highlights a fundamental trade-off.
Diffusion models emphasize flexibility, global optimization,
and revision capability.
Autoregressive models emphasize efficiency, determinism,
and stable incremental generation.

[The remainder of this document intentionally restates and reinforces
the above concepts through paraphrasing, elaboration, and repetition.
This redundancy is deliberate and should not be ignored.]

Large Language Diffusion Models (LLaDA) again differ from
traditional causal language models in both training objective
and inference procedure.
The diffusion objective focuses on denoising masked tokens,
whereas causal language modeling focuses on next-token prediction.

In practice, these objective differences lead to divergent behaviors.
Diffusion models can improve earlier tokens at later stages,
while autoregressive models cannot.
This allows diffusion systems to correct mistakes,
but also introduces additional complexity and instability risks.

Block-wise diffusion serves as a compromise between
full-sequence diffusion and purely autoregressive decoding.
By structuring generation into blocks,
LLaDA aims to balance computational efficiency
with the flexibility of diffusion-based revision.

The instability observed in early LLaDA versions
underscores that long-context modeling is not solely
a function of attention span.
Stability depends on interaction between sampling strategy,
model confidence, revision mechanisms, and architectural constraints.

Later versions of LLaDA explicitly acknowledge this
and incorporate mechanisms designed to regulate uncertainty,
limit error propagation, and enforce global consistency.

In conclusion, LLaDA represents a promising alternative
to conventional autoregressive language models,
offering new capabilities alongside new challenges.
Its evolution across versions highlights the complexity
of designing stable, efficient, and flexible generative systems.
Large Language Diffusion Models (LLaDA) propose an alternative paradigm to traditional
autoregressive language modeling approaches. While autoregressive models such as GPT,
GPT‑Neo, LLaMA, and similar architectures generate sequences token by token using
a strict left‑to‑right ordering and causal masking, diffusion-based language models
treat text generation as a gradual denoising process.

In autoregressive language modeling, generation proceeds sequentially.
At each step, the model predicts the probability distribution of the next token,
conditioned on all previously generated tokens.
Once a token is generated, it becomes fixed and cannot be revised.
This irreversible commitment is a defining characteristic of autoregressive decoding.

By contrast, Large Language Diffusion Models (LLaDA) operate via a forward and reverse process.
In the forward diffusion process, a clean text sequence is progressively corrupted.
This corruption is performed by randomly masking tokens according to a timestep variable t,
where t is sampled uniformly from the interval [0, 1].

At t = 0, the sequence remains fully intact and uncorrupted.
As t increases, a growing fraction of tokens are masked.
At t = 1, the sequence is entirely masked, containing no observable tokens.

The reverse diffusion process aims to recover the original sequence.
Starting from a fully or partially masked input, the model repeatedly predicts
denoised token distributions until a clean sequence is obtained.
This reverse process is implemented using a Transformer network with bidirectional
self-attention rather than causal attention.

Because the Transformer uses bidirectional attention,
each token can attend to tokens both to its left and to its right.
This architectural choice allows the model to leverage global context
when predicting or revising any part of the sequence.

One major conceptual advantage of diffusion-based language models is their ability
to revise previously generated tokens.
Unlike autoregressive models, which commit to each generated token permanently,
diffusion models can adjust earlier predictions during later denoising steps.
This enables correction of earlier errors, improved global consistency,
and better performance on tasks requiring holistic understanding of the sequence.

Examples of such tasks include text infilling, where missing spans inside a document
must be reconstructed; document editing, where existing text must be modified;
and long-context reasoning, where conclusions depend on interactions between
distant parts of the input.

Despite these advantages, diffusion-based language models introduce new challenges.
A primary challenge is sampling efficiency.
Diffusion models typically require many denoising steps,
each involving a full forward pass through a large Transformer.
As a result, naive diffusion sampling can be substantially slower
than autoregressive decoding, which generates one token per step.

To mitigate this inefficiency, LLaDA introduces block-wise diffusion
and semi-autoregressive decoding strategies.
In block-wise diffusion, the output sequence is divided into fixed-length blocks,
such as blocks containing tens or hundreds of tokens.

Each block initially starts in a fully masked state.
Blocks are then denoised independently while being conditioned
on all previously generated blocks.
This creates a hybrid structure: autoregressive at the block level,
but diffusion-based within each block.

Block-wise diffusion significantly reduces computational cost.
Instead of performing diffusion over the entire sequence at once,
the model only needs to denoise a limited number of tokens per block.
This preserves many of the benefits of diffusion modeling,
including local revision and context-aware correction,
while improving practical decoding speed.

However, block-wise diffusion does not fully eliminate all limitations.
Early versions of LLaDA exhibit instability during very long,
free-form generation tasks.
This instability often arises late in the generation process
and manifests in several observable ways.

Common failure modes include repetitive token patterns,
looping phrases, semantic drift, and collapse into low-information output.
These phenomena suggest that uncertainty can accumulate
across successive denoising steps and blocks.

Later versions of the model, specifically LLaDA‑2.0 and LLaDA‑2.1,
introduce additional architectural and algorithmic improvements
to mitigate these instability issues.

One such improvement is explicit token editing.
This mechanism allows the model to selectively revisit
and modify previously generated tokens, even across block boundaries.

Another improvement is confidence-aware parallel decoding.
In this approach, the model estimates its own uncertainty
and adapts the denoising strategy accordingly,
allocating more refinement steps to uncertain regions.

Additionally, re-editing mechanisms are introduced,
allowing earlier blocks to be revisited and refined
when inconsistencies or contradictions are detected later in the sequence.
These mechanisms help maintain global coherence
over long generation horizons.

Positional encoding is another critical component of LLaDA.
The model uses Rotary Positional Embeddings (RoPE)
to encode relative token positions.
RoPE enables efficient attention over sequences
by embedding positional information directly into attention computations.

To extend the usable context length,
scaling factors can be applied to the RoPE theta parameters.
This permits attention over much longer contexts,
such as 16K or 24K tokens, exceeding the model’s original training window.

However, extending context length via positional encoding
does not automatically guarantee stable long-range generation.
Even when the attention mechanism technically spans long contexts,
the iterative diffusion process can amplify small prediction errors
over many denoising steps.

As a result, positional encoding extensions alone
are insufficient to ensure coherent and stable long-form output.
Additional architectural safeguards are required,
such as re-editing, confidence modeling, and constrained decoding.

Large Language Diffusion Models demonstrate strong performance
on tasks that benefit from global context and holistic optimization.
These include structured reasoning, document revision,
controlled generation, and infilling tasks.

At the same time, traditional autoregressive models
remain highly effective for tasks requiring very long,
uninterrupted text generation at high speed.
Their simplicity, stability, and efficient sampling
make them well-suited for many production scenarios.

The comparison between diffusion-based models and autoregressive models
highlights a fundamental trade-off.
Diffusion models emphasize flexibility, global optimization,
and revision capability.
Autoregressive models emphasize efficiency, determinism,
and stable incremental generation.

[The remainder of this document intentionally restates and reinforces
the above concepts through paraphrasing, elaboration, and repetition.
This redundancy is deliberate and should not be ignored.]

Large Language Diffusion Models (LLaDA) again differ from
traditional causal language models in both training objective
and inference procedure.
The diffusion objective focuses on denoising masked tokens,
whereas causal language modeling focuses on next-token prediction.

In practice, these objective differences lead to divergent behaviors.
Diffusion models can improve earlier tokens at later stages,
while autoregressive models cannot.
This allows diffusion systems to correct mistakes,
but also introduces additional complexity and instability risks.

Block-wise diffusion serves as a compromise between
full-sequence diffusion and purely autoregressive decoding.
By structuring generation into blocks,
LLaDA aims to balance computational efficiency
with the flexibility of diffusion-based revision.

The instability observed in early LLaDA versions
underscores that long-context modeling is not solely
a function of attention span.
Stability depends on interaction between sampling strategy,
model confidence, revision mechanisms, and architectural constraints.

Later versions of LLaDA explicitly acknowledge this
and incorporate mechanisms designed to regulate uncertainty,
limit error propagation, and enforce global consistency.

In conclusion, LLaDA represents a promising alternative
to conventional autoregressive language models,
offering new capabilities alongside new challenges.
Its evolution across versions highlights the complexity
of designing stable, efficient, and flexible generative systems.

Large Language Diffusion Models (LLaDA) propose an alternative paradigm to traditional
autoregressive language modeling approaches. While autoregressive models such as GPT,
GPT‑Neo, LLaMA, and similar architectures generate sequences token by token using
a strict left‑to‑right ordering and causal masking, diffusion-based language models
treat text generation as a gradual denoising process.

In autoregressive language modeling, generation proceeds sequentially.
At each step, the model predicts the probability distribution of the next token,
conditioned on all previously generated tokens.
Once a token is generated, it becomes fixed and cannot be revised.
This irreversible commitment is a defining characteristic of autoregressive decoding.

By contrast, Large Language Diffusion Models (LLaDA) operate via a forward and reverse process.
In the forward diffusion process, a clean text sequence is progressively corrupted.
This corruption is performed by randomly masking tokens according to a timestep variable t,
where t is sampled uniformly from the interval [0, 1].

At t = 0, the sequence remains fully intact and uncorrupted.
As t increases, a growing fraction of tokens are masked.
At t = 1, the sequence is entirely masked, containing no observable tokens.

The reverse diffusion process aims to recover the original sequence.
Starting from a fully or partially masked input, the model repeatedly predicts
denoised token distributions until a clean sequence is obtained.
This reverse process is implemented using a Transformer network with bidirectional
self-attention rather than causal attention.

Because the Transformer uses bidirectional attention,
each token can attend to tokens both to its left and to its right.
This architectural choice allows the model to leverage global context
when predicting or revising any part of the sequence.

One major conceptual advantage of diffusion-based language models is their ability
to revise previously generated tokens.
Unlike autoregressive models, which commit to each generated token permanently,
diffusion models can adjust earlier predictions during later denoising steps.
This enables correction of earlier errors, improved global consistency,
and better performance on tasks requiring holistic understanding of the sequence.

Examples of such tasks include text infilling, where missing spans inside a document
must be reconstructed; document editing, where existing text must be modified;
and long-context reasoning, where conclusions depend on interactions between
distant parts of the input.

Despite these advantages, diffusion-based language models introduce new challenges.
A primary challenge is sampling efficiency.
Diffusion models typically require many denoising steps,
each involving a full forward pass through a large Transformer.
As a result, naive diffusion sampling can be substantially slower
than autoregressive decoding, which generates one token per step.

To mitigate this inefficiency, LLaDA introduces block-wise diffusion
and semi-autoregressive decoding strategies.
In block-wise diffusion, the output sequence is divided into fixed-length blocks,
such as blocks containing tens or hundreds of tokens.

Each block initially starts in a fully masked state.
Blocks are then denoised independently while being conditioned
on all previously generated blocks.
This creates a hybrid structure: autoregressive at the block level,
but diffusion-based within each block.

Block-wise diffusion significantly reduces computational cost.
Instead of performing diffusion over the entire sequence at once,
the model only needs to denoise a limited number of tokens per block.
This preserves many of the benefits of diffusion modeling,
including local revision and context-aware correction,
while improving practical decoding speed.

However, block-wise diffusion does not fully eliminate all limitations.
Early versions of LLaDA exhibit instability during very long,
free-form generation tasks.
This instability often arises late in the generation process
and manifests in several observable ways.

Common failure modes include repetitive token patterns,
looping phrases, semantic drift, and collapse into low-information output.
These phenomena suggest that uncertainty can accumulate
across successive denoising steps and blocks.

Later versions of the model, specifically LLaDA‑2.0 and LLaDA‑2.1,
introduce additional architectural and algorithmic improvements
to mitigate these instability issues.

One such improvement is explicit token editing.
This mechanism allows the model to selectively revisit
and modify previously generated tokens, even across block boundaries.

Another improvement is confidence-aware parallel decoding.
In this approach, the model estimates its own uncertainty
and adapts the denoising strategy accordingly,
allocating more refinement steps to uncertain regions.

Additionally, re-editing mechanisms are introduced,
allowing earlier blocks to be revisited and refined
when inconsistencies or contradictions are detected later in the sequence.
These mechanisms help maintain global coherence
over long generation horizons.

Positional encoding is another critical component of LLaDA.
The model uses Rotary Positional Embeddings (RoPE)
to encode relative token positions.
RoPE enables efficient attention over sequences
by embedding positional information directly into attention computations.

To extend the usable context length,
scaling factors can be applied to the RoPE theta parameters.
This permits attention over much longer contexts,
such as 16K or 24K tokens, exceeding the model’s original training window.

However, extending context length via positional encoding
does not automatically guarantee stable long-range generation.
Even when the attention mechanism technically spans long contexts,
the iterative diffusion process can amplify small prediction errors
over many denoising steps.

As a result, positional encoding extensions alone
are insufficient to ensure coherent and stable long-form output.
Additional architectural safeguards are required,
such as re-editing, confidence modeling, and constrained decoding.

Large Language Diffusion Models demonstrate strong performance
on tasks that benefit from global context and holistic optimization.
These include structured reasoning, document revision,
controlled generation, and infilling tasks.

At the same time, traditional autoregressive models
remain highly effective for tasks requiring very long,
uninterrupted text generation at high speed.
Their simplicity, stability, and efficient sampling
make them well-suited for many production scenarios.

The comparison between diffusion-based models and autoregressive models
highlights a fundamental trade-off.
Diffusion models emphasize flexibility, global optimization,
and revision capability.
Autoregressive models emphasize efficiency, determinism,
and stable incremental generation.

[The remainder of this document intentionally restates and reinforces
the above concepts through paraphrasing, elaboration, and repetition.
This redundancy is deliberate and should not be ignored.]

Large Language Diffusion Models (LLaDA) again differ from
traditional causal language models in both training objective
and inference procedure.
The diffusion objective focuses on denoising masked tokens,
whereas causal language modeling focuses on next-token prediction.

In practice, these objective differences lead to divergent behaviors.
Diffusion models can improve earlier tokens at later stages,
while autoregressive models cannot.
This allows diffusion systems to correct mistakes,
but also introduces additional complexity and instability risks.

Block-wise diffusion serves as a compromise between
full-sequence diffusion and purely autoregressive decoding.
By structuring generation into blocks,
LLaDA aims to balance computational efficiency
with the flexibility of diffusion-based revision.

The instability observed in early LLaDA versions
underscores that long-context modeling is not solely
a function of attention span.
Stability depends on interaction between sampling strategy,
model confidence, revision mechanisms, and architectural constraints.

Later versions of LLaDA explicitly acknowledge this
and incorporate mechanisms designed to regulate uncertainty,
limit error propagation, and enforce global consistency.

In conclusion, LLaDA represents a promising alternative
to conventional autoregressive language models,
offering new capabilities alongside new challenges.
Its evolution across versions highlights the complexity
of designing stable, efficient, and flexible generative systems.

Large Language Diffusion Models (LLaDA) propose an alternative paradigm to traditional
autoregressive language modeling approaches. While autoregressive models such as GPT,
GPT‑Neo, LLaMA, and similar architectures generate sequences token by token using
a strict left‑to‑right ordering and causal masking, diffusion-based language models
treat text generation as a gradual denoising process.

In autoregressive language modeling, generation proceeds sequentially.
At each step, the model predicts the probability distribution of the next token,
conditioned on all previously generated tokens.
Once a token is generated, it becomes fixed and cannot be revised.
This irreversible commitment is a defining characteristic of autoregressive decoding.

By contrast, Large Language Diffusion Models (LLaDA) operate via a forward and reverse process.
In the forward diffusion process, a clean text sequence is progressively corrupted.
This corruption is performed by randomly masking tokens according to a timestep variable t,
where t is sampled uniformly from the interval [0, 1].

At t = 0, the sequence remains fully intact and uncorrupted.
As t increases, a growing fraction of tokens are masked.
At t = 1, the sequence is entirely masked, containing no observable tokens.

The reverse diffusion process aims to recover the original sequence.
Starting from a fully or partially masked input, the model repeatedly predicts
denoised token distributions until a clean sequence is obtained.
This reverse process is implemented using a Transformer network with bidirectional
self-attention rather than causal attention.

Because the Transformer uses bidirectional attention,
each token can attend to tokens both to its left and to its right.
This architectural choice allows the model to leverage global context
when predicting or revising any part of the sequence.


==================== END DOCUMENT ====================

Questions:

1. Summarize the fundamental differences between diffusion-based language models and autoregressive language models.
2. Explain the motivation behind introducing block-wise diffusion in LLaDA.
3. Why does extending context length via positional encoding not guarantee stable long-form generation?
4. What specific architectural or algorithmic changes were introduced in LLaDA‑2.0 and LLaDA‑2.1 to mitigate instability?
5. Based on the document, discuss the trade-offs between flexibility and efficiency in diffusion versus autoregressive models.

"""


_BLOCKING_MODE_OPTIONS = ("kv", "q", "h", "qkv", "hq", "hkv", "hqkv", "bhqkv")
_BLOCKING_MODE_ARGS = {
    "kv": {"num_kv_blocks"},
    "q": {"num_q_blocks"},
    "h": {"head_block_size"},
    "qkv": {"num_kv_blocks", "num_q_blocks"},
    "hq": {"head_block_size", "num_q_blocks"},
    "hkv": {"head_block_size", "num_kv_blocks"},
    "hqkv": {"head_block_size", "num_kv_blocks", "num_q_blocks"},
    "bhqkv": {"head_block_size", "num_kv_blocks", "num_q_blocks", "num_batch_blocks"},
}


def _apply_reduced_layer_config(config, num_lang_layers: int):
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        config.text_config.num_hidden_layers = num_lang_layers
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = num_lang_layers
    if hasattr(config, "text_config") and hasattr(config.text_config, "layer_types") and config.text_config.layer_types:
        config.text_config.layer_types = config.text_config.layer_types[:num_lang_layers]
    return config


def load_model_and_processor(model_id: str, canvas_length: int, num_lang_layers: int = None):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if num_lang_layers is not None:
        config = _apply_reduced_layer_config(config, num_lang_layers=num_lang_layers)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        config=config,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=False,
        ignore_mismatched_sizes=num_lang_layers is not None,
    )
    qeff_model.model.config.canvas_length = canvas_length
    return processor, qeff_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run DiffusionGemma unified, split, or staged-prefill inference.")
    parser.add_argument("--runtime-mode", choices=("unified", "split", "prefill"), default="unified")
    parser.add_argument("--text-only", action="store_true", help="Run without image tokens.")
    parser.add_argument("--prompt", help="Override the default image or text prompt.")
    parser.add_argument("--seed", type=int, default=1234, help="Use -1 for an unseeded sampler.")
    parser.add_argument("--ctx-len", type=int, default=CTX_LEN, help="Compiled retained-KV context length.")
    parser.add_argument("--canvas-length", type=int, default=CANVAS_LENGTH, help="Tokens per denoising canvas.")
    parser.add_argument(
        "--prefill-stages",
        type=int,
        default=1,
        help="Encoder prefill pipeline partitions for split and prefill modes.",
    )
    parser.add_argument(
        "--enable-blocking",
        action="store_true",
        help="Enable generic attention blocking for split encoder prefill.",
    )
    parser.add_argument(
        "--blocking-mode",
        choices=_BLOCKING_MODE_OPTIONS,
        default="kv",
        help="Generic blocking mode for split encoder prefill.",
    )
    parser.add_argument("--num-kv-blocks", type=int, help="KV blocks for split encoder prefill.")
    parser.add_argument("--num-q-blocks", type=int, help="Query blocks for split encoder prefill.")
    parser.add_argument("--head-block-size", type=int, help="Heads per block for split encoder prefill.")
    parser.add_argument("--num-batch-blocks", type=int, help="Batch blocks for split encoder prefill.")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Total generated tokens.")
    parser.add_argument("--diffusion-steps", type=int, default=DIFFUSION_STEPS, help="Steps per canvas.")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Use a reduced number of language layers; defaults to the full model.",
    )
    parser.add_argument(
        "--sampler",
        choices=("local", "hf"),
        default="local",
        help="Cumulative local freezing or Hugging Face per-step re-noising.",
    )
    parser.add_argument("--no-stop-on-eos", action="store_true", help="Do not stop at the first EOS token.")
    parser.add_argument("--truncate-first-sentence", action="store_true", help="Return the first sentence only.")
    parser.add_argument("--verbose-steps", action="store_true", help="Decode a preview after each step.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.canvas_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Canvas length and max new tokens must be positive.")
    if args.prefill_stages <= 0:
        raise ValueError("Prefill stages must be positive.")
    if args.num_layers is not None and args.num_layers <= 0:
        raise ValueError("Number of layers must be positive.")
    if args.enable_blocking and args.runtime_mode == "unified":
        raise ValueError("Blocking is supported only for split encoder prefill.")
    blocking_option_values = {
        "num_kv_blocks": args.num_kv_blocks,
        "num_q_blocks": args.num_q_blocks,
        "head_block_size": args.head_block_size,
        "num_batch_blocks": args.num_batch_blocks,
    }
    for option_name, option_value in blocking_option_values.items():
        if option_value is not None and option_value <= 0:
            raise ValueError(f"{option_name} must be positive when provided.")
        if option_value is not None and option_name not in _BLOCKING_MODE_ARGS[args.blocking_mode]:
            raise ValueError(f"{option_name} is not compatible with blocking_mode={args.blocking_mode!r}.")
    if args.runtime_mode == "prefill" and not args.text_only:
        raise ValueError("Prefill-only mode requires --text-only.")

    encoder_blocking_config = None
    if args.enable_blocking:
        encoder_blocking_config = {
            "enable_blocking": True,
            "blocking_mode": args.blocking_mode,
            **{name: value for name, value in blocking_option_values.items() if value is not None},
        }

    device_ids = None  # [int(device_id) for device_id in os.environ.get("DG", "4,5,6,7").split(",")]
    processor, qeff_model = load_model_and_processor(
        MODEL_ID,
        args.canvas_length,
        num_lang_layers=args.num_layers,
    )
    print(f"Compiling a {qeff_model.model.config.text_config.num_hidden_layers}-layer DiffusionGemma model.")
    if args.runtime_mode == "unified":
        qpc_path = compile_unified_qpc(
            qeff_model.model,
            prefill_seq_len=args.canvas_length,
            ctx_len=args.ctx_len,
            canvas_length=args.canvas_length,
            num_devices=NUM_DEVICES,
            num_cores=NUM_CORES,
        )
        qpc_kwargs = {"qpc_path": qpc_path}
    else:
        encoder_qpc_path, decoder_qpc_path = compile_split_qpcs(
            qeff_model.model,
            prefill_seq_len=args.canvas_length,
            ctx_len=args.ctx_len,
            canvas_length=args.canvas_length,
            num_devices=NUM_DEVICES,
            num_cores=NUM_CORES,
            prefill_stages=args.prefill_stages,
            text_only=args.text_only,
            encoder_blocking_config=encoder_blocking_config,
        )
        qpc_kwargs = {
            "encoder_qpc_path": encoder_qpc_path,
            "decoder_qpc_path": decoder_qpc_path,
        }

    prompt = args.prompt or (TEXT_PROMPT if args.text_only else IMAGE_PROMPT)
    inputs = prepare_prompt_inputs(
        processor=processor,
        qeff_model=qeff_model.model,
        model_id=MODEL_ID,
        prompt=prompt,
        text_only=args.text_only,
        image_url=IMAGE_URL,
    )
    print(f"Canvas length is {CANVAS_LENGTH} and input ids is of size {inputs['input_ids'].shape[1]}")
    if args.runtime_mode == "prefill":
        result = diffusion_gemma_prefill_split_qpc_chunked(
            qeff_model=qeff_model,
            inputs=inputs,
            encoder_qpc_path=encoder_qpc_path,
            decoder_qpc_path=decoder_qpc_path,
            device_ids=device_ids,
            seed=args.seed,
            prefill_stages=args.prefill_stages,
        )
        print(f"Prompt length: {result.prompt_length}")
        print(f"Next position: {result.next_position}")
        print(f"Prefill time: {result.total_time:.3f}s")
        print(f"Encoder outputs: {list(result.encoder_outputs)}")
        print(f"Retained KV buffers: {len(result.kv_caches)}")
        for index, cache in enumerate(result.kv_caches):
            print(f"  [{index}] shape={cache.shape} dtype={cache.dtype}")
        print(f"ENCODER_QPC={encoder_qpc_path}")
        print(f"DECODER_QPC={decoder_qpc_path}")
        return

    result = qeff_model.cloud_ai_100_diffusion_generate(
        inputs=inputs,
        generation_len=args.max_new_tokens,
        runtime_mode=args.runtime_mode,
        device_ids=device_ids,
        ctx_len=args.ctx_len,
        max_denoising_steps=args.diffusion_steps,
        sampler=args.sampler,
        seed=args.seed,
        stop_on_eos=not args.no_stop_on_eos,
        step_callback=build_step_callback(processor.tokenizer, args.verbose_steps),
        **({"prefill_stages": args.prefill_stages} if args.runtime_mode == "split" else {}),
        **qpc_kwargs,
    )

    raw_output = processor.tokenizer.decode(result.generated_ids[0].tolist(), skip_special_tokens=True)
    output_text = clean_diffusion_text(
        raw_output,
        truncate_first_sentence=(result.generated_ids.shape[1] <= args.canvas_length or args.truncate_first_sentence),
    )
    clean_token_count = len(processor.tokenizer(output_text, add_special_tokens=False).input_ids)
    print(f"\nOutput:\n{output_text}")
    print(f"\nTTFT: {result.ttft:.2f}s ({result.retained_kv_buffers} KV buffers retained)")
    print(
        f"Average number of steps: {result.total_steps / result.executed_blocks}\n "
        f"Tokens per second: {clean_token_count / result.total_time:.1f}\n"
        f"Tokens per second(TTFT excluded): {clean_token_count / (result.total_time - result.ttft):.1f}\n"
    )
    for name, path in qpc_kwargs.items():
        print(f"\n{name.upper()}={path}")


if __name__ == "__main__":
    main()
