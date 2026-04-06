# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Compare dense Gemma4 decoder-layer prefill outputs between ORT and QAic.

This script promotes the exported decoder-layer handoff tensors to graph
outputs, compiles that debug ONNX, and reports the first layer where QAic
drifts from ORT.
"""

import argparse
import copy
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import TensorProto, helper
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.generate_inputs import InputHandler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="tiny-random/gemma-4-dense")
    parser.add_argument("--prompt", default="hello world hello world")
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--probe-layer-idx", type=int, default=0)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _export_debug_onnx(qeff_model, export_dir: Path) -> tuple[Path, list[dict]]:
    onnx_path = Path(
        qeff_model.export(
            export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    model = onnx.load(str(onnx_path), load_external_data=False)

    value_info = {info.name: info for info in model.graph.value_info}
    existing_outputs = {output.name for output in model.graph.output}
    layer_outputs = []

    for layer_idx, node in enumerate(model.graph.node):
        if not node.op_type.startswith("QEffGemma4TextDecoderLayer"):
            continue

        non_retained = [name for name in node.output if not name.endswith("_RetainedState")]
        if not non_retained:
            continue

        hidden_state_output = non_retained[-1]
        layer_outputs.append(
            {
                "layer_idx": len(layer_outputs),
                "node_name": node.name,
                "output_name": hidden_state_output,
            }
        )

        if hidden_state_output in existing_outputs:
            continue

        if hidden_state_output in value_info:
            model.graph.output.append(copy.deepcopy(value_info[hidden_state_output]))
        else:
            model.graph.output.append(helper.make_tensor_value_info(hidden_state_output, TensorProto.FLOAT, None))

    debug_onnx_path = onnx_path.with_name(f"{onnx_path.stem}_layer_debug.onnx")
    onnx.save(model, str(debug_onnx_path), save_as_external_data=False)
    return debug_onnx_path, layer_outputs


def _promote_internal_probes(debug_onnx_path: Path, probe_layer_idx: int) -> list[dict]:
    model = onnx.load(str(debug_onnx_path), load_external_data=False)
    layer_nodes = [node for node in model.graph.node if node.op_type.startswith("QEffGemma4TextDecoderLayer")]
    if probe_layer_idx < 0 or probe_layer_idx >= len(layer_nodes):
        raise ValueError(f"probe_layer_idx={probe_layer_idx} out of range for {len(layer_nodes)} decoder layers")

    layer_node = layer_nodes[probe_layer_idx]
    layer_function = next(function for function in model.functions if function.name == layer_node.op_type)
    node_by_output = {}
    consumers = defaultdict(list)

    for node in layer_function.node:
        for output_name in node.output:
            node_by_output[output_name] = node
        for input_name in node.input:
            consumers[input_name].append(node)

    def find_output(semantic_name: str):
        for output_name in node_by_output:
            if output_name == semantic_name or output_name.startswith(f"{semantic_name}."):
                return output_name
        return None

    def find_consumer(input_name: str | None, op_type: str):
        if input_name is None:
            return None
        for consumer in consumers.get(input_name, []):
            if consumer.op_type == op_type:
                return consumer
        return None

    def first_node_after(start_node, predicate):
        if start_node is None:
            return None
        start_idx = list(layer_function.node).index(start_node)
        for node in layer_function.node[start_idx + 1 :]:
            if predicate(node):
                return node
        return None

    def follow_cast_chain(output_name: str | None):
        cast_outputs = []
        current_output = output_name
        while current_output is not None:
            cast_node = find_consumer(current_output, "Cast")
            if cast_node is None:
                break
            current_output = cast_node.output[0]
            cast_outputs.append(current_output)
        return cast_outputs

    probe_map = []

    def add_probe(label: str, local_output_name: str | None):
        if local_output_name is None:
            return
        probe_map.append((label, local_output_name))

    query_states = find_output("query_states")
    key_states = find_output("key_states")
    value_states = find_output("value_states")
    retained_key = find_output("past_key.0_InternalRetainedState")
    retained_value = find_output("past_value.0_InternalRetainedState")
    gathered_key = find_output("key")
    gathered_value = find_output("value")
    masked_attn_logits = find_output("attn_weights")

    add_probe("query_states", query_states)
    add_probe("key_states", key_states)
    add_probe("value_states", value_states)
    add_probe("retained_key", retained_key)
    add_probe("retained_value", retained_value)
    add_probe("gathered_key", gathered_key)
    add_probe("gathered_value", gathered_value)

    qk_logits_node = find_consumer(query_states, "MatMul")
    qk_logits = qk_logits_node.output[0] if qk_logits_node is not None else None
    add_probe("qk_logits", qk_logits)

    scaled_attn_logits_node = find_consumer(qk_logits, "Mul")
    scaled_attn_logits = scaled_attn_logits_node.output[0] if scaled_attn_logits_node is not None else None
    add_probe("scaled_attn_logits", scaled_attn_logits)

    attention_mask_cast = next(
        (node.output[0] for node in layer_function.node if node.op_type == "Cast" and "attention_mask" in node.input),
        None,
    )
    add_probe("attention_mask_cast", attention_mask_cast)
    add_probe("masked_attn_logits", masked_attn_logits)

    softmax_node = find_consumer(masked_attn_logits, "Softmax")
    softmax_probs = softmax_node.output[0] if softmax_node is not None else None
    add_probe("softmax_probs", softmax_probs)

    softmax_cast_node = find_consumer(softmax_probs, "Cast")
    softmax_probs_cast = softmax_cast_node.output[0] if softmax_cast_node is not None else None
    add_probe("softmax_probs_cast", softmax_probs_cast)

    attention_probs = softmax_probs_cast or softmax_probs
    attention_probs_cast_node = find_consumer(softmax_probs_cast, "Cast") if softmax_probs_cast is not None else None
    if attention_probs_cast_node is not None:
        attention_probs = attention_probs_cast_node.output[0]
    add_probe("attention_probs", attention_probs)

    context_pre_transpose_node = find_consumer(attention_probs, "MatMul")
    context_pre_transpose = context_pre_transpose_node.output[0] if context_pre_transpose_node is not None else None
    add_probe("context_pre_transpose", context_pre_transpose)

    context_transposed_node = find_consumer(context_pre_transpose, "Transpose")
    context_transposed = context_transposed_node.output[0] if context_transposed_node is not None else None
    add_probe("context_transposed", context_transposed)

    context_reshaped_node = find_consumer(context_transposed, "Reshape")
    context_reshaped = context_reshaped_node.output[0] if context_reshaped_node is not None else None
    add_probe("context_reshaped", context_reshaped)

    attention_output_node = find_consumer(context_reshaped, "MatMul")
    attention_output = attention_output_node.output[0] if attention_output_node is not None else None
    add_probe("attention_output", attention_output)

    post_attention_residual_node = first_node_after(
        attention_output_node,
        lambda node: node.op_type == "Add" and any(inp.startswith("residual") for inp in node.input),
    )
    post_attention_residual_preclip = post_attention_residual_node.output[0] if post_attention_residual_node else None
    add_probe("post_attention_residual_preclip", post_attention_residual_preclip)

    post_attention_clip_node = find_consumer(post_attention_residual_preclip, "Clip")
    post_attention_residual = (
        follow_cast_chain(post_attention_clip_node.output[0])[-1]
        if post_attention_clip_node is not None and follow_cast_chain(post_attention_clip_node.output[0])
        else (post_attention_clip_node.output[0] if post_attention_clip_node is not None else None)
    )
    add_probe("post_attention_residual", post_attention_residual)

    mlp_input = post_attention_residual
    add_probe("mlp_input", mlp_input)

    mlp_gate_input_node = first_node_after(
        post_attention_residual_node,
        lambda node: node.op_type == "MatMul",
    )
    mlp_gate_input = mlp_gate_input_node.input[0] if mlp_gate_input_node is not None else None
    mlp_gate_proj = mlp_gate_input_node.output[0] if mlp_gate_input_node is not None else None
    add_probe("mlp_gate_input", mlp_gate_input)
    add_probe("mlp_gate_proj", mlp_gate_proj)

    mlp_output_node = first_node_after(
        mlp_gate_input_node,
        lambda node: node.op_type == "Cast" and any(inp.startswith("onnx::Cast_") for inp in node.input),
    )
    mlp_output = mlp_output_node.output[0] if mlp_output_node is not None else None

    final_layer_residual = next(
        (output_name for output_name in reversed(layer_function.output) if not output_name.endswith("_RetainedState")),
        None,
    )

    if final_layer_residual is not None:
        for node in layer_function.node:
            if final_layer_residual in node.output:
                final_layer_node = node
                break
        else:
            final_layer_node = None

        final_residual_add_node = first_node_after(
            post_attention_residual_node,
            lambda node: node.op_type == "Add" and any(inp.startswith("residual") for inp in node.input),
        )
        final_residual_preclip = final_residual_add_node.output[0] if final_residual_add_node is not None else None
        add_probe("final_residual_preclip", final_residual_preclip)

        final_residual_clip_node = find_consumer(final_residual_preclip, "Clip")
        final_residual_clipped = final_residual_clip_node.output[0] if final_residual_clip_node is not None else None
        add_probe("final_residual_clipped", final_residual_clipped)

        post_mlp_norm = None
        if final_residual_add_node is not None:
            post_mlp_norm = next(
                (name for name in final_residual_add_node.input if not name.startswith("residual")), None
            )
        add_probe("post_mlp_norm", post_mlp_norm)

    add_probe("mlp_output", mlp_output)
    add_probe("final_layer_residual", final_layer_residual)

    probes = []
    for label, local_output_name in probe_map:
        debug_output_name = f"/debug/layer{probe_layer_idx}/{label}"
        if local_output_name in layer_function.output:
            output_idx = list(layer_function.output).index(local_output_name)
            debug_output_name = layer_node.output[output_idx]
        else:
            layer_function.output.append(local_output_name)
            layer_node.output.append(debug_output_name)
            model.graph.output.append(helper.make_tensor_value_info(debug_output_name, TensorProto.FLOAT, None))
        probes.append(
            {
                "label": label,
                "local_output_name": local_output_name,
                "debug_output_name": debug_output_name,
                "node_name": layer_node.name,
                "layer_idx": probe_layer_idx,
            }
        )

    onnx.save(model, str(debug_onnx_path), save_as_external_data=False)
    return probes


def _load_text_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

    full_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(torch.float32)
    text_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True).text_config
    text_model = AutoModelForCausalLM.from_config(
        text_config,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(torch.float32)
    text_model.model.load_state_dict(full_model.model.language_model.state_dict())
    text_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
    text_model.eval()
    return tokenizer, text_model


def _prepare_prefill_inputs(tokenizer, config, prompt: str, prompt_len: int, ctx_len: int):
    handler = InputHandler(
        batch_size=1,
        tokenizer=tokenizer,
        config=config,
        prompt=[prompt],
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=None,
    )
    return handler.prepare_ort_inputs()


def _run_ort_prefill(onnx_path: Path, inputs: dict, output_names: list[str]):
    session = onnxruntime.InferenceSession(str(onnx_path))
    feed = {name: value for name, value in inputs.items() if name in {x.name for x in session.get_inputs()}}
    values = session.run(output_names, feed)
    return dict(zip(output_names, values))


def _run_qaic_prefill(qpc_path: Path, inputs: dict):
    session = QAICInferenceSession(qpc_path)
    feed = {}
    for name, value in inputs.items():
        if name not in session.input_names:
            continue
        binding = session.bindings[session.binding_index_map[name]]
        expected_dtype = session.aic_to_np_dtype_mapping[binding.type]
        feed[name] = value.astype(expected_dtype, copy=False)

    matched_allowed_shape = None
    for allowed_shape in session.allowed_shapes or []:
        matches = True
        for input_name, value in feed.items():
            expected_dims = allowed_shape[session.binding_index_map[input_name]][1]
            if list(value.shape) != expected_dims:
                matches = False
                break
        if matches:
            matched_allowed_shape = allowed_shape
            break

    output_buffers = {}
    for output_name in session.output_names:
        binding = session.bindings[session.binding_index_map[output_name]]
        output_dtype = session.aic_to_np_dtype_mapping[binding.type]
        if matched_allowed_shape is not None:
            output_shape = matched_allowed_shape[session.binding_index_map[output_name]][1]
        else:
            output_shape = list(binding.dims)
        output_buffers[output_name] = np.zeros(output_shape, dtype=output_dtype)
    session.set_buffers(output_buffers)

    return session.run(feed)


def _summarize_diff(name: str, ort_value: np.ndarray, qaic_value: np.ndarray, atol: float, rtol: float):
    ort_float = ort_value.astype(np.float32)
    qaic_float = qaic_value.astype(np.float32)
    diff = np.abs(ort_float - qaic_float)
    return {
        "name": name,
        "shape": list(ort_value.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose": bool(np.allclose(ort_float, qaic_float, atol=atol, rtol=rtol)),
        "ort_sample": ort_float.reshape(-1)[:8].tolist(),
        "qaic_sample": qaic_float.reshape(-1)[:8].tolist(),
    }


def main():
    args = parse_args()
    tokenizer, text_model = _load_text_model(args.model_id)
    qeff_model = QEFFAutoModelForCausalLM(copy.deepcopy(text_model), pretrained_model_name_or_path=args.model_id)

    export_dir = Path(tempfile.mkdtemp()) / "onnx"
    debug_onnx_path, layer_outputs = _export_debug_onnx(qeff_model, export_dir)
    layer_probes = _promote_internal_probes(debug_onnx_path, args.probe_layer_idx)
    prefill_inputs = _prepare_prefill_inputs(tokenizer, text_model.config, args.prompt, args.prompt_len, args.ctx_len)

    requested_outputs = (
        ["logits"]
        + [item["output_name"] for item in layer_outputs]
        + [item["debug_output_name"] for item in layer_probes]
    )
    ort_outputs = _run_ort_prefill(debug_onnx_path, prefill_inputs, requested_outputs)

    compile_dir = Path(tempfile.mkdtemp()) / "compile"
    qpc_path = qeff_model.compile(
        onnx_path=str(debug_onnx_path),
        compile_dir=str(compile_dir),
        prefill_seq_len=args.prompt_len,
        ctx_len=args.ctx_len,
        use_onnx_subfunctions=True,
    )
    qaic_outputs = _run_qaic_prefill(qpc_path, prefill_inputs)

    results = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "debug_onnx_path": str(debug_onnx_path),
        "generated_npi_path": str(debug_onnx_path.with_name(f"{debug_onnx_path.stem}_gemma4_npi.yaml")),
        "qpc_path": str(qpc_path),
        "prefill_logits_argmax": {
            "ort": np.asarray(ort_outputs["logits"]).argmax(-1).tolist(),
            "qaic": np.asarray(qaic_outputs["logits"]).argmax(-1).tolist(),
        },
        "layers": [],
        "first_drifting_layer": None,
        "probe_layer_idx": args.probe_layer_idx,
        "layer_probes": [],
        "first_drifting_probe": None,
    }

    for layer in layer_outputs:
        diff_summary = _summarize_diff(
            layer["output_name"],
            np.asarray(ort_outputs[layer["output_name"]]),
            np.asarray(qaic_outputs[layer["output_name"]]),
            atol=args.atol,
            rtol=args.rtol,
        )
        diff_summary["layer_idx"] = layer["layer_idx"]
        diff_summary["node_name"] = layer["node_name"]
        results["layers"].append(diff_summary)
        if results["first_drifting_layer"] is None and not diff_summary["allclose"]:
            results["first_drifting_layer"] = diff_summary

    for probe in layer_probes:
        diff_summary = _summarize_diff(
            probe["debug_output_name"],
            np.asarray(ort_outputs[probe["debug_output_name"]]),
            np.asarray(qaic_outputs[probe["debug_output_name"]]),
            atol=args.atol,
            rtol=args.rtol,
        )
        diff_summary["label"] = probe["label"]
        diff_summary["local_output_name"] = probe["local_output_name"]
        diff_summary["node_name"] = probe["node_name"]
        diff_summary["layer_idx"] = probe["layer_idx"]
        results["layer_probes"].append(diff_summary)
        if results["first_drifting_probe"] is None and not diff_summary["allclose"]:
            results["first_drifting_probe"] = diff_summary

    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output_json is not None:
        args.output_json.write_text(payload)


if __name__ == "__main__":
    main()
