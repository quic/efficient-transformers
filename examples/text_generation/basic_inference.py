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
import os
import sys
from collections import deque
from pathlib import Path
from time import perf_counter

# Make `examples._common` importable when this file is run directly, i.e.
# `python examples/text_generation/basic_inference.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples._common import args as A  # noqa: E402


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
            A.dflash_group(),
            A.runtime_group(),
            A.meta_group(),
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )


def _stage_compiler_options(ns: argparse.Namespace, stage: str | None = None) -> dict:
    options = A.compiler_options(ns)
    stage_npi = getattr(ns, f"{stage}_node_precision_info", None) if stage else None
    if stage_npi is not None:
        options["node_precision_info"] = stage_npi
    for option in ("aic_enable_depth_first", "user_tiled"):
        stage_value = getattr(ns, f"{stage}_{option}", None) if stage else None
        if stage_value is not None:
            if stage_value:
                options[option] = True
            else:
                options.pop(option, None)
    return options


def _stage_compile_dir(ns: argparse.Namespace, stage: str) -> str | None:
    if ns.compile_dir is None:
        return None
    stage_compile_dir = Path(ns.compile_dir) / stage
    stage_compile_dir.mkdir(parents=True, exist_ok=True)
    return str(stage_compile_dir)


def _stage_device_group(ns: argparse.Namespace, stage: str):
    stage_device_group = getattr(ns, f"{stage}_device_group")
    if stage_device_group is not None:
        return stage_device_group
    if getattr(ns, f"{stage}_num_devices") is None:
        return ns.device_group
    return None


def _select_ccl_length(ccl_lengths, required_length: int, ctx_len: int) -> int | None:
    if not ccl_lengths:
        return None
    candidates = sorted(set(ccl_lengths) | {ctx_len})
    return next((length for length in candidates if length >= required_length), candidates[-1])


def _build_default_sampling_params(ns: argparse.Namespace, batch_size: int, vocab_size: int) -> dict:
    """Build neutral, deterministic runtime inputs for the on-device sampler."""
    import numpy as np

    max_top_k_ids = min(ns.max_top_k_ids, vocab_size)
    params = {
        "repetition_penalties": np.ones((batch_size, 1), dtype=np.float32),
        "presence_penalties": np.zeros((batch_size, 1), dtype=np.float32),
        "temperatures": np.zeros((batch_size, 1), dtype=np.float32),
        "top_ks": np.ones((batch_size, 1), dtype=np.int32),
        "top_ps": np.ones((batch_size, 1), dtype=np.float32),
        "min_ps": np.zeros((batch_size, 1), dtype=np.float32),
        "random_numbers": np.zeros((batch_size, max_top_k_ids), dtype=np.float32),
    }
    if ns.include_guided_decoding:
        params["token_bitmasks"] = np.ones((batch_size, vocab_size), dtype=np.bool_)
    return params


def compile_disaggregated(model, ns: argparse.Namespace) -> tuple[str, str]:
    """Compile decode and pipeline-parallel prefill QPCs for DMA KV handoff."""
    shared = {
        "ctx_len": ns.ctx_len,
        "full_batch_size": ns.full_batch_size,
        "kv_cache_batch_size": ns.kv_cache_batch_size,
        "num_cores": ns.num_cores,
        "mxfp6_matmul": ns.mxfp6_matmul,
        "mxint8_kv_cache": ns.mxint8_kv_cache,
        "num_speculative_tokens": None,
        "split_retained_state_io": True,
        "retain_full_kv": True,
        "use_onnx_subfunctions": ns.use_onnx_subfunctions,
        "kv_cache_prefix": ns.kv_cache_prefix,
        "artifacts": ns.artifacts,
    }

    decode_qpc_path = model.compile(
        compile_dir=_stage_compile_dir(ns, "decode"),
        prefill_seq_len=1,
        comp_ctx_lengths_decode=ns.comp_ctx_lengths_decode,
        num_devices=A.resolve_decode_num_devices(ns),
        prefill_only=False,
        offload_pt_weights=False,
        qaic_config=A.build_decode_qaic_config(ns),
        **shared,
        **_stage_compiler_options(ns, "decode"),
    )
    if ns.comp_ctx_lengths_decode is not None:
        ns.comp_ctx_lengths_decode = getattr(model, "comp_ctx_lengths_decode", ns.comp_ctx_lengths_decode)
    label = "compiler artifacts" if ns.artifacts else "QPC"
    print(f"Decode {label}: {decode_qpc_path}")

    prefill_qpc_path = model.compile(
        compile_dir=_stage_compile_dir(ns, "prefill"),
        prefill_seq_len=ns.prefill_seq_len,
        comp_ctx_lengths_prefill=ns.comp_ctx_lengths_prefill,
        num_devices=A.resolve_prefill_num_devices(ns),
        prefill_only=True,
        offload_pt_weights=ns.offload_pt_weights,
        enable_chunking=True,
        qaic_config=A.build_prefill_qaic_config(ns),
        mdp_num_partitions=ns.mdp_num_partitions,
        mdp_strategy=ns.mdp_strategy,
        **shared,
        **_stage_compiler_options(ns, "prefill"),
    )
    if ns.comp_ctx_lengths_prefill is not None:
        ns.comp_ctx_lengths_prefill = getattr(model, "comp_ctx_lengths_prefill", ns.comp_ctx_lengths_prefill)
    print(f"Prefill {label}: {prefill_qpc_path}")
    return prefill_qpc_path, decode_qpc_path


def run_disaggregated(
    tokenizer,
    prefill_qpc_path: str,
    decode_qpc_path: str,
    prompts: list[str],
    ns: argparse.Namespace,
) -> dict:
    """Run continuous-batching text generation with zero-copy host KV handoff."""
    import numpy as np

    from QEfficient.generation.cloud_infer import QAICInferenceSession
    from QEfficient.utils import padding_check_and_fix

    padding_check_and_fix(tokenizer)

    prefill_session = QAICInferenceSession(
        prefill_qpc_path,
        device_ids=_stage_device_group(ns, "prefill"),
        kv_dma_share=True,
        stages=ns.mdp_num_partitions,
        full_batch_size=ns.full_batch_size,
        cluster_id="prefill",
    )
    decode_session = QAICInferenceSession(
        decode_qpc_path,
        device_ids=_stage_device_group(ns, "decode"),
        kv_dma_share=True,
        full_batch_size=ns.full_batch_size,
        cluster_id="decode",
    )
    if "batch_index" not in decode_session.binding_index_map:
        raise ValueError("The decode QPC must expose a batch_index input for continuous batching.")

    kv_caches = [np.zeros(shape, dtype=dtype) for shape, dtype in decode_session.kv_cache_info]
    if not kv_caches or kv_caches[0].shape[0] != ns.full_batch_size:
        actual_batch = kv_caches[0].shape[0] if kv_caches else None
        raise ValueError(f"Decode KV batch dimension {actual_batch} != full batch size {ns.full_batch_size}.")
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    def prepare_prompt(prompt: str):
        encoded = tokenizer(prompt, return_tensors="np", padding=True)
        prompt_len = encoded["input_ids"].shape[1]
        num_chunks = -(prompt_len // -ns.prefill_seq_len)
        padded_len = num_chunks * ns.prefill_seq_len
        if padded_len > ns.ctx_len:
            raise ValueError(f"Prompt requires {padded_len} padded tokens, which exceeds --ctx-len {ns.ctx_len}.")
        encoded = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
        lang_inputs = {"input_ids": encoded["input_ids"]}
        attention_mask = encoded["attention_mask"]
        token_positions = np.cumsum(attention_mask, axis=-1) - 1
        lang_inputs["position_ids"] = np.where(attention_mask, token_positions, -1)
        return lang_inputs, num_chunks

    def prefill_slot(lang_inputs, num_chunks: int, slot: int):
        chunk_inputs = {"batch_index": np.array([[slot]], dtype=np.int64)}
        slot_kv_view = [kv_cache[slot : slot + 1] for kv_cache in kv_caches]
        exec_idx = None
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * ns.prefill_seq_len
            chunk_end = (chunk_idx + 1) * ns.prefill_seq_len
            chunk_inputs["input_ids"] = lang_inputs["input_ids"][:, chunk_start:chunk_end]
            chunk_inputs["position_ids"] = lang_inputs["position_ids"][:, chunk_start:chunk_end]
            ccl_length = _select_ccl_length(ns.comp_ctx_lengths_prefill, chunk_end, ns.ctx_len)
            if ccl_length is not None:
                chunk_inputs["comp_ctx_lengths"] = np.zeros(ccl_length, dtype=np.int64)
            last_chunk = chunk_idx == num_chunks - 1
            exec_idx = prefill_session.np_run_pipeline(
                chunk_inputs,
                last_chunk=last_chunk,
                kv_cache_buffers=slot_kv_view if last_chunk else None,
            )
            prefill_session.complete_inf(exec_idx, is_prefill=True)

        prefill_output = prefill_session.get_outputs(index=exec_idx)
        first_token = int(np.argmax(prefill_output["logits"]))
        next_position = int(np.max(lang_inputs["position_ids"])) + 1
        return first_token, next_position

    generation_len = ns.generation_len or ns.ctx_len
    ongoing = [False] * ns.full_batch_size
    last_token = [0] * ns.full_batch_size
    position = [0] * ns.full_batch_size
    generated_count = [0] * ns.full_batch_size
    slot_prompt_index = [-1] * ns.full_batch_size
    slot_tokens = [None] * ns.full_batch_size
    results = [None] * len(prompts)

    def seed_slot(slot: int, prompt_index: int, first_token: int, next_position: int) -> bool:
        slot_prompt_index[slot] = prompt_index
        slot_tokens[slot] = [first_token]
        generated_count[slot] = 1
        last_token[slot] = first_token
        position[slot] = next_position
        ongoing[slot] = first_token != tokenizer.eos_token_id and generation_len > 1 and next_position < ns.ctx_len
        if not ongoing[slot]:
            results[prompt_index] = slot_tokens[slot]
        return ongoing[slot]

    prompt_queue = deque(enumerate(prompts))

    def fill_slot(slot: int) -> None:
        while prompt_queue:
            prompt_index, prompt = prompt_queue.popleft()
            lang_inputs, num_chunks = prepare_prompt(prompt)
            first_token, next_position = prefill_slot(lang_inputs, num_chunks, slot)
            if seed_slot(slot, prompt_index, first_token, next_position):
                return
        ongoing[slot] = False

    prefill_start = perf_counter()
    for slot in range(ns.full_batch_size):
        fill_slot(slot)
    print(f"Initial prefill time: {perf_counter() - prefill_start:.2f} sec")

    def build_decode_inputs():
        input_ids = np.full((ns.full_batch_size, 1), -1, dtype=np.int64)
        position_ids = np.full((ns.full_batch_size, 1), -1, dtype=np.int64)
        batch_index = np.full((ns.full_batch_size, 1), -1, dtype=np.int64)
        for slot in range(ns.full_batch_size):
            if ongoing[slot]:
                input_ids[slot, 0] = last_token[slot]
                position_ids[slot, 0] = position[slot]
                batch_index[slot, 0] = slot
        decode_inputs = {"input_ids": input_ids, "position_ids": position_ids, "batch_index": batch_index}
        active_positions = [position[slot] for slot in range(ns.full_batch_size) if ongoing[slot]]
        required_ccl_length = max(active_positions, default=-1) + 1
        ccl_length = _select_ccl_length(ns.comp_ctx_lengths_decode, required_ccl_length, ns.ctx_len)
        if ccl_length is not None:
            decode_inputs["comp_ctx_lengths"] = np.zeros(ccl_length, dtype=np.int64)
        return decode_inputs

    decode_start = perf_counter()
    decode_steps = 0
    while any(ongoing):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(build_decode_inputs(), is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        output = decode_session.get_outputs(index=exec_idx)
        decode_steps += 1

        logits = output["logits"].reshape(ns.full_batch_size, -1, output["logits"].shape[-1])[:, -1, :]
        next_tokens = np.argmax(logits, axis=-1)
        for slot in range(ns.full_batch_size):
            if not ongoing[slot]:
                continue
            token = int(next_tokens[slot])
            slot_tokens[slot].append(token)
            generated_count[slot] += 1
            if token != tokenizer.eos_token_id:
                last_token[slot] = token
                position[slot] += 1
            reached_limit = generated_count[slot] >= generation_len or position[slot] >= ns.ctx_len
            if token == tokenizer.eos_token_id or reached_limit:
                results[slot_prompt_index[slot]] = slot_tokens[slot]
                fill_slot(slot)

    decode_time = perf_counter() - decode_start
    total_tokens = sum(len(tokens) for tokens in results if tokens)
    print(f"Decode steps: {decode_steps}; throughput: {total_tokens / max(decode_time, 1e-9):.2f} tok/sec")
    for prompt_index, prompt in enumerate(prompts):
        tokens = results[prompt_index] or []
        print(f"\nPrompt: {prompt}\nGenerated: {tokenizer.decode(tokens)}")
    return {"tokens": results}


def main() -> None:
    parser = build_parser()
    ns = parser.parse_args()
    A.validate_dflash_args(ns, parser, sys.argv[1:])
    A.validate_args(ns, parser.error)

    if ns.print_resolved:
        A.print_namespace(ns)
    if ns.dry_run:
        return

    prompts = A.resolve_prompts(ns)
    if ns.dflash:
        from examples.performance.dflash.basic_inference_text import run_text_dflash

        run_text_dflash(
            model_name=ns.model_name,
            prompt=prompts[0],
            ctx_len=ns.ctx_len,
            prefill_seq_len=ns.prefill_seq_len,
            generation_len=ns.generation_len or ns.ctx_len,
            iteration=ns.iteration,
            tlm_devices=ns.tlm_devices or ns.device_group or [0],
            dlm_devices=ns.dlm_devices or ns.device_group or [0],
            tlm_cores=ns.tlm_cores,
            dlm_cores=ns.dlm_cores,
            tlm_hf_path=ns.tlm_hf_path,
            tlm_qpc=ns.tlm_qpc,
            dlm_qpc=ns.dlm_qpc,
            category=ns.category,
            format_prompt=ns.format_prompt,
            hf_token=os.environ.get("HF_TOKEN"),
            compile_only=ns.compile_only,
            compile_dir=ns.compile_dir,
        )
        return

    import torch
    from transformers import AutoTokenizer

    from QEfficient import QEFFAutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        ns.tokenizer_name or ns.model_name,
        gguf_file=ns.gguf_file,
        trust_remote_code=True,
    )

    from_pretrained_kwargs = {"continuous_batching": ns.continuous_batching or ns.disaggregated}
    if ns.dtype is not None:
        from_pretrained_kwargs["dtype"] = getattr(torch, ns.dtype)
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

    if ns.disaggregated:
        prefill_qpc_path, decode_qpc_path = compile_disaggregated(model, ns)
        if ns.compile_only:
            return
        run_disaggregated(tokenizer, prefill_qpc_path, decode_qpc_path, prompts, ns)
        return

    compile_options = _stage_compiler_options(ns)
    if ns.stage == "prefill":
        compile_options.update(mdp_num_partitions=ns.mdp_num_partitions, mdp_strategy=ns.mdp_strategy)

    compile_path = model.compile(
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
        qaic_config=(
            A.build_prefill_qaic_config(ns)
            if ns.stage == "prefill"
            else A.build_decode_qaic_config(ns)
            if ns.stage == "decode"
            else qaic_config
        ),
        retain_full_kv=ns.retain_full_kv or None,
        layerwise=ns.layerwise,
        layerwise_window_size=ns.layerwise_window_size,
        kv_cache_prefix=ns.kv_cache_prefix,
        dynamo=ns.dynamo,
        artifacts=ns.artifacts,
        **compile_options,
    )
    label = "Compiler artifacts written to" if ns.artifacts else "Compiled QPC"
    print(f"{label}: {compile_path}")

    if ns.compile_only or ns.stage == "prefill":
        return

    execution_batch_size = ns.full_batch_size or ns.batch_size
    sampling_params = (
        _build_default_sampling_params(ns, execution_batch_size, model.vocab_size) if ns.include_sampler else None
    )
    runtime_options = {
        "include_sampler": ns.include_sampler,
        "return_pdfs": ns.return_pdfs,
        "include_guided_decoding": ns.include_guided_decoding,
        "sampling_params": sampling_params,
    }
    if ns.write_io and not ns.artifacts:
        io_dir = model.generate(tokenizer=tokenizer, prompts=prompts, artifacts=True, **runtime_options)
        print(f"Runner inputs written to: {io_dir}")

    exec_info = model.generate(
        tokenizer=tokenizer,
        prompts=prompts,
        device_ids=ns.device_group,
        generation_len=ns.generation_len,
        iteration=ns.iteration,
        automation=ns.automation,
        artifacts=ns.artifacts,
        **runtime_options,
    )
    if ns.artifacts:
        print(f"Runner inputs written to: {exec_info}")
        return
    for prompt, text in zip(prompts, exec_info.generated_texts):
        print(f"\nPrompt: {prompt}\nGenerated: {text}")
    print(exec_info)


if __name__ == "__main__":
    main()
