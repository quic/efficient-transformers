# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Canonical text-generation example for QEfficient.

The script wires ``QEFFAutoModelForCausalLM.from_pretrained -> .compile -> .generate``
in reading order. The three shapes of run people actually care about all sit on
top of the same three calls:

    Dense, single prompt (default):
        python basic_inference.py --model-name Qwen/Qwen2-1.5B-Instruct

    Continuous batching:
        python basic_inference.py --model-name meta-llama/Llama-3.1-8B \
            --continuous-batching --full-batch-size 4 \
            --prompt "A" "B" "C" "D"

    MoE with expert-blocked chunked prefill + ONNX subfunctions:
        python basic_inference.py --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
            --use-onnx-subfunctions --enable-chunking --stage prefill

Everything else (disaggregated compile, blocked attention, MDP knobs, GGUF, CCL,
speculative decoding, on-device sampler) is a flag away; ``--help-advanced``
prints the full list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `examples._common` importable when this file is run directly, i.e.
# `python examples/text_generation/basic_inference.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from examples._common import args as A  # noqa: E402
from QEfficient import QEFFAutoModelForCausalLM  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Text generation on Qualcomm Cloud AI 100 via QEFFAutoModelForCausalLM.",
        parents=[
            A.model_group(),
            A.compile_group(),
            A.ccl_group(),
            A.disagg_group(),
            A.blocking_group(),
            A.speculative_group(),
            A.sampler_group(),
            A.runtime_group(),
            A.meta_group(),
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def main() -> None:
    parser = build_parser()
    ns = parser.parse_args()
    A.validate_args(ns, parser.error)

    if ns.print_resolved:
        A.print_namespace(ns)
    if ns.dry_run:
        return

    tokenizer = AutoTokenizer.from_pretrained(
        ns.tokenizer_name or ns.model_name,
        gguf_file=ns.gguf_file,
        trust_remote_code=True,
    )

    from_pretrained_kwargs = {"continuous_batching": ns.continuous_batching}
    if ns.gguf_file:
        from_pretrained_kwargs["gguf_file"] = ns.gguf_file
    if ns.max_seq_len_cached is not None:
        from_pretrained_kwargs["max_seq_len_cached"] = ns.max_seq_len_cached
    if ns.layerwise:
        # Route through the meta-device load path so very large models
        # (e.g. 8x DeepSeek-R1, Qwen3.5-MoE 671B) never materialize the
        # full checkpoint into host RAM before compile()'s layerwise
        # driver runs (modeling_auto.py: from_pretrained ``layerwise`` kwarg).
        from_pretrained_kwargs["layerwise"] = True
    qaic_config = A.build_qaic_config(ns)
    if qaic_config is not None:
        from_pretrained_kwargs["qaic_config"] = qaic_config
    A.apply_num_layers_override(from_pretrained_kwargs, ns)

    model = QEFFAutoModelForCausalLM.from_pretrained(ns.model_name, **from_pretrained_kwargs)

    qpc_path = model.compile(
        onnx_path=ns.onnx_path,
        compile_dir=ns.compile_dir,
        prefill_seq_len=ns.prefill_seq_len,
        ctx_len=ns.ctx_len,
        comp_ctx_lengths_prefill=ns.comp_ctx_lengths_prefill,
        comp_ctx_lengths_decode=ns.comp_ctx_lengths_decode,
        batch_size=ns.batch_size,
        full_batch_size=ns.full_batch_size,
        kv_cache_batch_size=ns.kv_cache_batch_size,
        num_devices=A.resolve_num_devices(ns),
        num_cores=ns.num_cores,
        mxfp6_matmul=ns.mxfp6_matmul,
        mxint8_kv_cache=ns.mxint8_kv_cache,
        num_speculative_tokens=A.num_speculative_tokens(ns),
        prefill_only=A.resolve_prefill_only(ns),
        use_onnx_subfunctions=ns.use_onnx_subfunctions,
        offload_pt_weights=ns.offload_pt_weights,
        enable_chunking=ns.enable_chunking,
        moe_prefill_packed_chunk_size=ns.moe_prefill_packed_chunk_size,
        retain_full_kv=ns.retain_full_kv or None,
        layerwise=ns.layerwise,
        layerwise_window_size=ns.layerwise_window_size,
        kv_cache_prefix=ns.kv_cache_prefix,
        **A.compiler_options(ns),
    )
    print(f"Compiled QPC: {qpc_path}")

    if ns.stage == "prefill":
        return

    prompts = A.resolve_prompts(ns)
    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_id=ns.device_group,
        generation_len=ns.generation_len,
        iteration=ns.iteration,
        write_io=ns.write_io,
        automation=ns.automation,
    )
    for prompt, text in zip(prompts, exec_info.generated_texts):
        print(f"\nPrompt: {prompt}\nGenerated: {text}")
    print(exec_info)


if __name__ == "__main__":
    main()
