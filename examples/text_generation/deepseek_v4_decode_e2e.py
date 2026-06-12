# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""DeepSeek-V4 retained compression decode smoke/e2e.

This script exercises the QEff DeepSeek-V4 decode path with device-retained
compression states:

* sliding shared-KV cache (`past_key.*`),
* HCA compressor buffers/compressed KV,
* CSA compressor overlap buffers/compressed KV,
* CSA indexer buffers/overlap/compressed KV.

Default model: ``silence09/DeepSeek-V4-Pro-Tiny``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.deepseek_v4 import (
    extract_deepseek_v4_compression_states,
    get_deepseek_v4_compression_state_initializers,
    get_deepseek_v4_compression_state_names,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

DEFAULT_MODEL_ID = "silence09/DeepSeek-V4-Pro-Tiny"


def _legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple((layer.keys, layer.values) for layer in cache.layers)


def _build_decode_state(qeff_model, config, ctx_len: int, tokens: list[int]):
    state_names = get_deepseek_v4_compression_state_names(config)
    comp_states = get_deepseek_v4_compression_state_initializers(config, 1, ctx_len, dtype=torch.float32)
    pkv = tuple(
        (
            torch.zeros(1, 1, ctx_len, config.head_dim),
            torch.zeros(1, 1, ctx_len, config.head_dim),
        )
        for _ in range(config.num_hidden_layers)
    )

    with torch.no_grad():
        for position, token in enumerate(tokens[:ctx_len]):
            outputs = qeff_model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                position_ids=torch.tensor([[position]], dtype=torch.long),
                past_key_values=pkv,
                deepseek_v4_compression_states=comp_states,
                use_cache=True,
            )
            comp_states = extract_deepseek_v4_compression_states(outputs.past_key_values)
            pkv = _legacy_cache(outputs.past_key_values)

        decode_token = tokens[ctx_len]
        pt_outputs = qeff_model(
            input_ids=torch.tensor([[decode_token]], dtype=torch.long),
            position_ids=torch.tensor([[ctx_len]], dtype=torch.long),
            past_key_values=tuple((key.clone(), value.clone()) for key, value in pkv),
            deepseek_v4_compression_states=[[state.clone() for state in layer] for layer in comp_states],
            use_cache=True,
        )

    return state_names, pkv, comp_states, pt_outputs.logits.detach().cpu().numpy()


def _make_ort_feeds(session_inputs, state_names, pkv, comp_states, decode_token: int, position: int, dtype=np.float32):
    input_names = {inp.name for inp in session_inputs}
    feeds = {
        "input_ids": np.array([[decode_token]], dtype=np.int64),
        "position_ids": np.array([[position]], dtype=np.int64),
    }
    for layer_idx, (key, _value) in enumerate(pkv):
        name = f"past_key.{layer_idx}"
        if name in input_names:
            feeds[name] = key.detach().cpu().numpy().astype(dtype)
    for layer_idx, layer_states in enumerate(comp_states):
        for state_name, state in zip(state_names[layer_idx], layer_states):
            name = f"deepseek_v4_{state_name}.{layer_idx}"
            if name in input_names:
                feeds[name] = state.detach().cpu().numpy().astype(dtype)
    return feeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--compile-dir", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run-qaic", action="store_true")
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--num-cores", type=int, default=1)
    args = parser.parse_args()

    if len(args.tokens) <= args.ctx_len:
        raise ValueError("Provide at least ctx_len + 1 tokens so one decode step can be checked.")

    model_hf = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32, low_cpu_mem_usage=False).eval()
    qeff_auto = QEFFAutoModelForCausalLM(model_hf)
    qeff_model = qeff_auto.model.eval()

    state_names, pkv, comp_states, pt_logits = _build_decode_state(
        qeff_model, model_hf.config, args.ctx_len, args.tokens
    )

    export_dir = args.export_dir or tempfile.mkdtemp(prefix="qeff_dsv4_decode_e2e_export_")
    onnx_path = Path(
        qeff_auto.export(
            export_dir,
            prefill_seq_len=1,
            ctx_len=args.ctx_len,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
            offload_pt_weights=False,
        )
    )
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    scatter_count = sum(node.op_type == "ScatterElements" for node in onnx_model.graph.node)
    retained_count = sum(output.name.endswith("_RetainedState") for output in onnx_model.graph.output)
    print(f"onnx_path={onnx_path}")
    print(f"scatter_elements={scatter_count}")
    print(f"retained_outputs={retained_count}")

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_feeds = _make_ort_feeds(
        ort_session.get_inputs(), state_names, pkv, comp_states, args.tokens[args.ctx_len], args.ctx_len
    )
    ort_names = [output.name for output in ort_session.get_outputs()]
    ort_outputs = dict(zip(ort_names, ort_session.run(None, ort_feeds)))
    ort_logits = ort_outputs["logits"]
    print(f"qeff_pytorch_vs_ort_max_abs_diff={np.max(np.abs(pt_logits - ort_logits))}")

    if not args.compile and not args.run_qaic:
        return

    compile_dir = args.compile_dir or tempfile.mkdtemp(prefix="qeff_dsv4_decode_e2e_compile_")
    qpc_path = qeff_auto.compile(
        onnx_path=str(onnx_path),
        compile_dir=compile_dir,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        num_cores=args.num_cores,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        aic_enable_depth_first=True,
    )
    print(f"qpc_path={qpc_path}")

    if args.run_qaic:
        qaic_session = QAICInferenceSession(qpc_path)
        qaic_inputs = set(qaic_session.input_names)
        qaic_feeds = {
            name: (value.astype(np.float16) if value.dtype.kind == "f" else value)
            for name, value in ort_feeds.items()
            if name in qaic_inputs
        }
        qaic_outputs = qaic_session.run(qaic_feeds)
        qaic_logits = qaic_outputs["logits"]
        print(
            f"qaic_vs_ort_max_abs_diff={np.max(np.abs(qaic_logits.astype(np.float32) - ort_logits.astype(np.float32)))}"
        )
        qaic_session.deactivate()


if __name__ == "__main__":
    main()
