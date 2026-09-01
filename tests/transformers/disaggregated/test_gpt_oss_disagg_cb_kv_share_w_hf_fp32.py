# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

"""Continuous-batching parity + KV-handoff check for the gpt-oss disaggregated

Two checks, in one on-device session:
  1. KV-handoff correctness: the shared host ``kv_caches`` arrays are inspected
     directly right after each slot's chunked prefill (DMA write) and right after
     the first decode step for both slots (DMA read + write-back), to prove the
     prefill-written rows land in the right slot with no cross-slot contamination
     and that decode only appends the new position without disturbing prior KV.
  2. Token-level parity: full decoded output per CB slot vs HF PyTorch fp32
     ``generate(do_sample=False)`` on the same prompt.

pytest -m "on_qaic and disagg_dma" tests/transformers/disaggregated/test_gpt_oss_disagg_cb_kv_share_w_hf_fp32.py

Run the nightly full-model HF/ORT/QAIC three-way parity test with:
    pytest -m "nightly_disagg" \
        tests/transformers/disaggregated/test_gpt_oss_disagg_cb_kv_share_w_hf_fp32.py
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from tests.transformers.disaggregated._disagg_dma_config import disagg_dma_configs
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    assert_three_way_tokens_match as _assert_three_way_tokens_match,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    ensure_session_inputs as _ensure_session_inputs,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    get_next_token_ids as _get_next_token_ids,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    session_input_names as _session_input_names,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    session_output_names as _session_output_names,
)
from tests.transformers.disaggregated._disagg_ort_test_utils import (
    update_state_from_outputs as _update_state_from_outputs,
)
from tests.transformers.disaggregated._nightly_disagg_config import nightly_disagg_configs

MODEL_NAME = "openai/gpt-oss-20b"
NUM_HIDDEN_LAYERS = 4
PREFILL_SEQ_LEN = 32
CTX_LEN = 256
BATCH_SIZE = 1
GENERATION_LEN = 40
FULL_BATCH_SIZE = 2
TEXT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
]

NUM_CORES = 16
MOE_PREFILL_PACKED_CHUNK_SIZE = 16


def _assert_onnx_path(onnx_path, label: str) -> Path:
    assert onnx_path is not None, f"{label} compile did not set an ONNX path"
    onnx_path = Path(onnx_path)
    assert onnx_path.is_file(), f"{label} ONNX path does not exist: {onnx_path}"
    assert onnx_path.suffix == ".onnx", f"{label} path is not an ONNX file: {onnx_path}"
    return onnx_path.resolve()


def _build_config(dtype: str = "float32", full_model: bool = False, model_name: str = MODEL_NAME):
    """Load the real config; optionally truncate depth for the regular CB test."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not full_model:
        config.num_hidden_layers = NUM_HIDDEN_LAYERS
        config.layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(NUM_HIDDEN_LAYERS)]
    config.dtype = dtype
    config.torch_dtype = getattr(torch, dtype)
    return config


def _load_hf_model(config, model_name: str = MODEL_NAME) -> AutoModelForCausalLM:
    torch.manual_seed(42)
    if getattr(config, "num_hidden_layers", None) != NUM_HIDDEN_LAYERS:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            attn_implementation="eager",
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
        ).eval()

    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    # Scale weights down so fp32 activations stay small; keeps HF and QAIC numerics close.
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(0.02)
    return model.eval()


def _run_hf_torch_fp32(model, tokenizer, prompt: str) -> np.ndarray:
    model = model.to(dtype=torch.float32).eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=GENERATION_LEN,
            min_new_tokens=GENERATION_LEN,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    prompt_len = input_ids.shape[-1]
    return outputs[0, prompt_len:].detach().cpu().numpy()


def _prepare_prompt(tokenizer, prompt: str):
    """Tokenise + pad to a multiple of PREFILL_SEQ_LEN; -1 position_ids at pad positions."""
    enc = tokenizer(prompt, return_tensors="np", padding=True)
    prompt_len = enc["input_ids"].shape[1]
    num_chunks = -(prompt_len // -PREFILL_SEQ_LEN)  # ceil divide without float
    padded_len = num_chunks * PREFILL_SEQ_LEN

    enc = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    input_ids = enc["input_ids"]
    position_ids = np.where(enc["attention_mask"], np.arange(padded_len), -1)
    return input_ids, position_ids.astype(np.int64), num_chunks, prompt_len


def _next_token_ids_from_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim == 2:
        return logits.argmax(axis=-1).astype(np.int64)
    return _get_next_token_ids(logits)


def _prefill_slot(prefill_session, input_ids, position_ids, num_chunks, slot: int, slot_kv_view):
    """Chunked prefill of one prompt into KV ``slot``. Returns (first_token, next_pos)."""
    chunk_inputs = {"batch_index": np.array([[slot]], dtype=np.int64)}
    exec_idx = None
    for i in range(num_chunks):
        chunk_inputs["input_ids"] = input_ids[:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = position_ids[:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
        last_chunk = i == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=slot_kv_view if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    first_token = int(_next_token_ids_from_logits(prefill_out["logits"])[0])
    next_pos = int(np.max(position_ids)) + 1
    return first_token, next_pos


def _patch_custom_rmsnorm_for_ort(path: Path) -> Path:
    """Patch exported local functions so ORT can execute the QPC ONNX in fp32."""
    try:
        import onnx
        from onnx import TensorProto, helper
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnx is required for ORT parity test.") from exc

    model = onnx.load(str(path), load_external_data=False)
    changed = False
    int32_max = 2147483647

    def _make_const_i64_1d(name: str, values: list):
        return helper.make_node(
            "Constant",
            [],
            [name],
            value=helper.make_tensor(name, TensorProto.INT64, [len(values)], values),
        )

    all_graph_nodes = list(model.graph.node)

    def _formal_param_for(function, call_arg: str) -> str | None:
        for node in all_graph_nodes:
            if node.op_type == function.name:
                for idx, inp in enumerate(node.input):
                    if inp == call_arg and idx < len(function.input):
                        return function.input[idx]
        for caller in model.functions:
            for node in caller.node:
                if node.op_type == function.name:
                    for idx, inp in enumerate(node.input):
                        if inp == call_arg and idx < len(function.input):
                            return function.input[idx]
        return None

    for function in model.functions:
        function_changed = False
        new_nodes = []

        if function.name == "CustomRMSNorm":
            for node in function.node:
                if node.op_type == "Cast" and list(node.input) == ["weight"] and list(node.output) == ["weight_0"]:
                    node = helper.make_node(
                        "CastLike",
                        ["weight", "hidden_states"],
                        ["weight_0"],
                        name=node.name or "CastLike_weight",
                    )
                    function_changed = True
                if node.op_type == "Expand" and list(node.output) == ["epsilon_2"]:
                    node.output[0] = "epsilon_2_pre_cast"
                    new_nodes.append(node)
                    new_nodes.append(
                        helper.make_node(
                            "CastLike",
                            ["epsilon_2_pre_cast", "variance"],
                            ["epsilon_2"],
                            name="CastLike_epsilon",
                        )
                    )
                    function_changed = True
                    continue
                new_nodes.append(node)

        elif function.name.startswith("CtxScatter"):
            pos_param = _formal_param_for(function, "position_ids") or next(
                (param for param in function.input if "pos" in param.lower()), None
            )
            if pos_param is not None:
                new_nodes.extend(
                    [
                        helper.make_node("Shape", ["data"], ["_sc_data_shape"]),
                        _make_const_i64_1d("_sc_dim1_idx", [1]),
                        helper.make_node("Gather", ["_sc_data_shape", "_sc_dim1_idx"], ["_sc_ctx_dim"]),
                        _make_const_i64_1d("_sc_one", [1]),
                        helper.make_node("Sub", ["_sc_ctx_dim", "_sc_one"], ["_sc_last_i64"]),
                        helper.make_node("CastLike", ["_sc_last_i64", pos_param], ["_sc_last"]),
                        _make_const_i64_1d("_sc_inv_i64", [int32_max]),
                        helper.make_node("CastLike", ["_sc_inv_i64", pos_param], ["_sc_inv"]),
                        helper.make_node("Equal", [pos_param, "_sc_inv"], ["_sc_inv_mask"]),
                        helper.make_node("Where", ["_sc_inv_mask", "_sc_last", pos_param], ["_sc_safe_pos"]),
                        helper.make_node(
                            "Cast",
                            ["_sc_safe_pos"],
                            ["_sc_pos_i64"],
                            name="Cast_sc_pos_i64",
                            to=TensorProto.INT64,
                        ),
                    ]
                )
                function_changed = True

            for node in function.node:
                if pos_param is not None:
                    for idx, inp in enumerate(node.input):
                        if inp == pos_param:
                            node.input[idx] = "_sc_pos_i64"
                if node.op_type == "ScatterND":
                    indices_in = node.input[1]
                    if not indices_in.endswith("_i64"):
                        cast_out = indices_in + "_i64"
                        new_nodes.append(
                            helper.make_node(
                                "Cast",
                                [indices_in],
                                [cast_out],
                                name=f"Cast_{indices_in}_i64",
                                to=TensorProto.INT64,
                            )
                        )
                        node.input[1] = cast_out
                        function_changed = True
                if node.op_type == "Cast" and list(node.output) == ["batch_idx_3"]:
                    for attr in node.attribute:
                        if attr.name == "to":
                            attr.i = TensorProto.INT64
                    function_changed = True
                if node.op_type == "Expand" and list(node.output) == ["ctx_idx"]:
                    node.output[0] = "ctx_idx_pre_i64"
                    new_nodes.append(node)
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            ["ctx_idx_pre_i64"],
                            ["ctx_idx"],
                            name="Cast_ctx_idx_i64",
                            to=TensorProto.INT64,
                        )
                    )
                    function_changed = True
                    continue
                new_nodes.append(node)

        elif function.name.startswith("CtxGather"):
            ctx_indices_param = next((param for param in function.input if "ctx_indices" in param.lower()), None)
            pos_param = next((param for param in function.input if "pos" in param.lower()), None)
            clamp_target = pos_param or ctx_indices_param
            if clamp_target is not None:
                new_nodes.extend(
                    [
                        _make_const_i64_1d("_gc_inv_i64", [int32_max]),
                        helper.make_node("CastLike", ["_gc_inv_i64", clamp_target], ["_gc_inv"]),
                        helper.make_node("Equal", [clamp_target, "_gc_inv"], ["_gc_inv_mask"]),
                        _make_const_i64_1d("_gc_zero_i64", [0]),
                        helper.make_node("CastLike", ["_gc_zero_i64", clamp_target], ["_gc_zero"]),
                        helper.make_node("Where", ["_gc_inv_mask", "_gc_zero", clamp_target], ["_gc_safe_target"]),
                        helper.make_node(
                            "Cast",
                            ["_gc_safe_target"],
                            ["_gc_target_i64"],
                            name="Cast_gc_target_i64",
                            to=TensorProto.INT64,
                        ),
                    ]
                )
                function_changed = True

            inserted_ctx_indices_cast = False
            for node in function.node:
                if clamp_target is not None:
                    for idx, inp in enumerate(node.input):
                        if inp == clamp_target:
                            node.input[idx] = "_gc_target_i64"
                if (
                    not inserted_ctx_indices_cast
                    and ctx_indices_param is not None
                    and ctx_indices_param != clamp_target
                    and node.op_type in {"Expand", "Unsqueeze", "GatherND"}
                    and ctx_indices_param in node.input
                ):
                    cast_name = f"{ctx_indices_param}_i64"
                    new_nodes.append(
                        helper.make_node(
                            "Cast",
                            [ctx_indices_param],
                            [cast_name],
                            name=f"Cast_{ctx_indices_param}_i64",
                            to=TensorProto.INT64,
                        )
                    )
                    inserted_ctx_indices_cast = True
                    function_changed = True
                    for idx, inp in enumerate(node.input):
                        if inp == ctx_indices_param:
                            node.input[idx] = cast_name
                new_nodes.append(node)

        else:
            continue

        if function_changed:
            del function.node[:]
            function.node.extend(new_nodes)
            changed = True

    if not changed:
        return path

    patched_path = path.with_name(f"{path.stem}.ort.onnx")
    onnx.save(model, str(patched_path))
    return patched_path


def _make_ort_session(path: Path):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("onnxruntime is required for ORT parity test.") from exc

    ort_path = _patch_custom_rmsnorm_for_ort(path)
    if ort_path != path:
        print(f"ORT patched ONNX: {path} -> {ort_path}")
    return ort.InferenceSession(str(ort_path), providers=["CPUExecutionProvider"])


def _prepare_inputs(tokenizer) -> dict:
    """Tokenize the nightly prompt, right-pad to a multiple of PREFILL_SEQ_LEN, build position_ids."""
    ids = tokenizer(TEXT_PROMPTS[0], return_tensors="pt")["input_ids"]
    input_len = ids.shape[1]
    num_chunks = -(input_len // -PREFILL_SEQ_LEN)
    padded_len = num_chunks * PREFILL_SEQ_LEN
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = np.full((BATCH_SIZE, padded_len), pad_id, dtype=np.int64)
    input_ids[:, :input_len] = ids.numpy()
    attention_mask = np.zeros((BATCH_SIZE, padded_len), dtype=np.int64)
    attention_mask[:, :input_len] = 1
    position_ids = np.where(attention_mask, np.arange(padded_len), -1)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids.astype(np.int64),
        "attention_mask": attention_mask,
        "num_chunks": num_chunks,
        "input_len": input_len,
    }


def _run_ort_generation(onnx_paths: dict[str, Path], tokenizer) -> np.ndarray:
    """Run the non-CB disaggregated prefill -> decode loop with ORT using QPC ONNX graphs."""
    prefill_session = _make_ort_session(onnx_paths["prefill"])
    decode_session = _make_ort_session(onnx_paths["decode"])

    print(f"[ORT] prefill_inputs : {_session_input_names(prefill_session)}")
    print(f"[ORT] decode_inputs  : {_session_input_names(decode_session)}")

    prepared = _prepare_inputs(tokenizer)
    input_ids = prepared["input_ids"]
    position_ids = prepared["position_ids"]
    num_chunks = prepared["num_chunks"]

    state: dict = {}
    prefill_outputs: dict = {}
    for chunk_idx in range(num_chunks):
        start = chunk_idx * PREFILL_SEQ_LEN
        end = (chunk_idx + 1) * PREFILL_SEQ_LEN
        provided = {
            "input_ids": input_ids[:, start:end],
            "position_ids": position_ids[:, start:end],
        }
        feed = _ensure_session_inputs(prefill_session, provided, state, PREFILL_SEQ_LEN, BATCH_SIZE, CTX_LEN)
        prefill_outputs = dict(zip(_session_output_names(prefill_session), prefill_session.run(None, feed)))
        _update_state_from_outputs(state, prefill_outputs)

    first_token = _next_token_ids_from_logits(prefill_outputs["logits"])
    generated_ids = [first_token]
    pos = np.max(position_ids, axis=-1, keepdims=True) + 1

    decode_inputs = {
        "input_ids": first_token.reshape(BATCH_SIZE, 1),
        "position_ids": pos,
    }
    for _ in range(GENERATION_LEN - 1):
        feed = _ensure_session_inputs(decode_session, decode_inputs, state, 1, BATCH_SIZE, CTX_LEN)
        decode_outputs = dict(zip(_session_output_names(decode_session), decode_session.run(None, feed)))
        _update_state_from_outputs(state, decode_outputs)
        token = _next_token_ids_from_logits(decode_outputs["logits"])
        generated_ids.append(token)
        pos = pos + 1
        decode_inputs = {
            "input_ids": token.reshape(BATCH_SIZE, 1),
            "position_ids": pos,
        }

    return np.stack(generated_ids, axis=1).astype(np.int64)


def _run_disagg_kv_share_qaic_generation(
    tokenizer,
    prefill_session: QAICInferenceSession,
    decode_session: QAICInferenceSession,
) -> np.ndarray:
    prepared = _prepare_inputs(tokenizer)
    num_chunks = prepared["num_chunks"]
    input_ids = prepared["input_ids"]
    position_ids = prepared["position_ids"]
    kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
    chunk_inputs = {}
    exec_idx = None
    for chunk_idx in range(num_chunks):
        chunk_inputs["input_ids"] = input_ids[:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]
        chunk_inputs["position_ids"] = position_ids[:, chunk_idx * PREFILL_SEQ_LEN : (chunk_idx + 1) * PREFILL_SEQ_LEN]
        last_chunk = chunk_idx == num_chunks - 1
        exec_idx = prefill_session.np_run_pipeline(
            chunk_inputs,
            last_chunk=last_chunk,
            kv_cache_buffers=kv_caches if last_chunk else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)

    prefill_out = prefill_session.get_outputs(index=exec_idx)
    generated_ids = [_next_token_ids_from_logits(prefill_out["logits"])]
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map
    pos = np.max(position_ids, axis=-1, keepdims=True) + 1
    decode_inputs = {
        "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
        "position_ids": pos,
    }
    for _ in range(GENERATION_LEN - 1):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_outputs = decode_session.get_outputs(index=exec_idx)
        generated_ids.append(_next_token_ids_from_logits(decode_outputs["logits"]))
        pos = pos + 1
        decode_inputs = {
            "input_ids": generated_ids[-1].reshape(BATCH_SIZE, 1),
            "position_ids": pos,
        }

    return np.stack(generated_ids, axis=1)


def _compile_disagg_qpcs(
    qeff_model,
    sessions: list,
    compiled_onnx_paths: dict,
    *,
    prefill_num_devices: int,
    decode_num_devices: int,
    stages: int,
):
    decode_qpc_path = qeff_model.compile(
        prefill_seq_len=1,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        num_devices=decode_num_devices,
        mos=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        offload_pt_weights=False,
        split_retained_state_io=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
    )
    compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.onnx_path, "decode")

    prefill_qpc_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=NUM_CORES,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
        num_devices=prefill_num_devices,
        mdp_num_partitions=stages,
        split_retained_state_io=True,
        mos=1,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        aic_enable_depth_first=False,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
    )
    compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.onnx_path, "prefill")
    print(f"Disagg ONNX paths: {compiled_onnx_paths}")

    prefill_session = QAICInferenceSession(prefill_qpc_path, kv_dma_share=True, stages=stages)
    decode_session = QAICInferenceSession(decode_qpc_path, kv_dma_share=True)
    sessions.extend([prefill_session, decode_session])
    return prefill_session, decode_session


@pytest.mark.skip()
@pytest.mark.parametrize("nightly_config", nightly_disagg_configs("gpt_oss"))
def test_gpt_oss_disagg_kv_share_qaic_vs_ort_vs_hf_fp32(manual_cleanup, nightly_config):
    """Non-CB three-way parity: HF fp32 == ORT on QPC ONNX == QAIC disagg DMA."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    torch.manual_seed(42)

    model_id = nightly_config["model_id"]
    config = _build_config(dtype="float32", full_model=True, model_name=model_id)
    hf_model = _load_hf_model(config, model_name=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_tokens = _run_hf_torch_fp32(hf_model, tokenizer, TEXT_PROMPTS[0]).reshape(BATCH_SIZE, GENERATION_LEN)
    qeff_model = QEFFAutoModelForCausalLM(hf_model)

    sessions = []
    compiled_onnx_paths = {}
    try:
        prefill_session, decode_session = _compile_disagg_qpcs(
            qeff_model,
            sessions,
            compiled_onnx_paths,
            prefill_num_devices=nightly_config["prefill_num_devices"],
            decode_num_devices=nightly_config["decode_num_devices"],
            stages=nightly_config["stages"],
        )
        ort_tokens = _run_ort_generation(compiled_onnx_paths, tokenizer)
        qaic_tokens = _run_disagg_kv_share_qaic_generation(tokenizer, prefill_session, decode_session)
    finally:
        for session in sessions:
            session.deactivate()
        cleanup_paths = list(compiled_onnx_paths.values()) or [getattr(qeff_model, "onnx_path", None)]
        manual_cleanup([path for path in cleanup_paths if path is not None])

    hf_text = tokenizer.batch_decode(hf_tokens, skip_special_tokens=True)
    ort_text = tokenizer.batch_decode(ort_tokens, skip_special_tokens=True)
    qaic_text = tokenizer.batch_decode(qaic_tokens, skip_special_tokens=True)
    print(f"HF   tokens : {hf_tokens.tolist()}")
    print(f"ORT  tokens : {ort_tokens.tolist()}")
    print(f"QAIC tokens : {qaic_tokens.tolist()}")
    print(f"HF   text   : {hf_text}")
    print(f"ORT  text   : {ort_text}")
    print(f"QAIC text   : {qaic_text}")

    _assert_three_way_tokens_match(hf_tokens, ort_tokens, qaic_tokens, BATCH_SIZE, GENERATION_LEN)


@pytest.mark.on_qaic
@pytest.mark.disagg_dma
@pytest.mark.parametrize("dma_config", disagg_dma_configs("gpt_oss_reduced"))
def test_gpt_oss_disagg_cb_kv_handoff_and_hf_parity(manual_cleanup, dma_config):
    torch.manual_seed(42)

    model_id = dma_config["model_id"]
    use_onnx_subfunctions = dma_config.get("use_onnx_subfunctions", True)

    config = _build_config(dtype="float32", model_name=model_id)
    hf_model = _load_hf_model(config, model_name=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_tokens = [_run_hf_torch_fp32(hf_model, tokenizer, prompt) for prompt in TEXT_PROMPTS]

    qeff_model = QEFFAutoModelForCausalLM(hf_model, continuous_batching=True)

    sessions = []
    compiled_onnx_paths = {}
    try:
        decode_qpc_path = qeff_model.compile(
            prefill_seq_len=1,
            ctx_len=CTX_LEN,
            full_batch_size=FULL_BATCH_SIZE,
            num_cores=NUM_CORES,
            num_devices=dma_config["decode_num_devices"],
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            offload_pt_weights=False,
            split_retained_state_io=True,
            retain_full_kv=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
        compiled_onnx_paths["decode"] = _assert_onnx_path(qeff_model.onnx_path, "decode")

        prefill_qpc_path = qeff_model.compile(
            prefill_seq_len=PREFILL_SEQ_LEN,
            ctx_len=CTX_LEN,
            full_batch_size=FULL_BATCH_SIZE,
            num_cores=NUM_CORES,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": MOE_PREFILL_PACKED_CHUNK_SIZE}},
            num_devices=dma_config["prefill_num_devices"],
            mdp_num_partitions=dma_config["stages"],
            split_retained_state_io=True,
            mos=1,
            mxfp6_matmul=False,
            mxint8_kv_cache=False,
            aic_enable_depth_first=True,
            num_speculative_tokens=None,
            prefill_only=True,
            enable_chunking=True,
            retain_full_kv=True,
            use_onnx_subfunctions=use_onnx_subfunctions,
        )
        compiled_onnx_paths["prefill"] = _assert_onnx_path(qeff_model.onnx_path, "prefill")
        print(f"Disagg CB ONNX paths: {compiled_onnx_paths}")

        prefill_session = QAICInferenceSession(prefill_qpc_path, kv_dma_share=True, full_batch_size=FULL_BATCH_SIZE)
        decode_session = QAICInferenceSession(decode_qpc_path, kv_dma_share=True, full_batch_size=FULL_BATCH_SIZE)
        sessions.extend([prefill_session, decode_session])

        assert "batch_index" in decode_session.binding_index_map, "batch_index not a compiled decode input binding"

        kv_caches = [np.zeros(shape, dtype=dtype) for (shape, dtype) in decode_session.kv_cache_info]
        assert kv_caches[0].shape[0] == FULL_BATCH_SIZE, (
            f"decode KV batch dim {kv_caches[0].shape[0]} != full_batch_size {FULL_BATCH_SIZE}"
        )
        decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

        # -------------------- Chunked prefill into both slots --------------------
        first_tokens = [None] * FULL_BATCH_SIZE
        next_pos = [None] * FULL_BATCH_SIZE
        prompt_len = [None] * FULL_BATCH_SIZE
        for slot, prompt in enumerate(TEXT_PROMPTS):
            input_ids, position_ids, num_chunks, plen = _prepare_prompt(tokenizer, prompt)
            prompt_len[slot] = plen

            # Pre-condition: this slot's row is still all-zero before its prefill runs.
            assert all(np.all(kv[slot] == 0) for kv in kv_caches), f"slot {slot} KV row is not zero before prefill"

            slot_kv_view = [kv[slot : slot + 1] for kv in kv_caches]
            ft, npos = _prefill_slot(prefill_session, input_ids, position_ids, num_chunks, slot, slot_kv_view)
            first_tokens[slot] = ft
            next_pos[slot] = npos

            # Post-condition: the DMA write landed in row `slot` (real prefix non-zero)
            # and did NOT touch any other slot's row (no cross-slot contamination).
            written = [kv[slot, :, : prompt_len[slot], :] for kv in kv_caches]
            assert all(np.any(w != 0) for w in written), (
                f"slot {slot} KV row is still zero after prefill -- DMA handoff did not write it"
            )
            for other in range(FULL_BATCH_SIZE):
                if other == slot or first_tokens[other] is None:
                    continue
                assert np.any(kv_caches[0][other] != 0), (
                    f"slot {other} KV row went to zero after prefilling slot {slot} -- cross-slot corruption"
                )

        # Snapshot the prefill-written region before decode touches anything.
        pre_decode_kv = [kv.copy() for kv in kv_caches]

        # -------------------- First decode step, both slots --------------------
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        input_ids = np.array([[first_tokens[s]] for s in range(FULL_BATCH_SIZE)], dtype=np.int64)
        position_ids = np.array([[next_pos[s]] for s in range(FULL_BATCH_SIZE)], dtype=np.int64)
        batch_index = np.array([[s] for s in range(FULL_BATCH_SIZE)], dtype=np.int64)
        decode_inputs = {"input_ids": input_ids, "position_ids": position_ids, "batch_index": batch_index}
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        decode_out = decode_session.get_outputs(index=exec_idx)
        decode_logits = decode_out["logits"].reshape(FULL_BATCH_SIZE, -1, decode_out["logits"].shape[-1])[:, -1, :]
        second_tokens = np.argmax(decode_logits, axis=-1)

        # Post-condition: decode's write-back only appends the new position; the
        # prefill-written prefix (input KV the decode step read from) is untouched.
        for slot in range(FULL_BATCH_SIZE):
            for kv_before, kv_after in zip(pre_decode_kv, kv_caches):
                prefix_before = kv_before[slot, :, : prompt_len[slot], :]
                prefix_after = kv_after[slot, :, : prompt_len[slot], :]
                assert np.array_equal(prefix_before, prefix_after), (
                    f"slot {slot}: decode step overwrote prefill-written KV prefix "
                    f"(positions 0..{prompt_len[slot]}) -- handoff/write-back is wiring the wrong offset"
                )
                new_pos_after = kv_after[slot, :, next_pos[slot], :]
                assert np.any(new_pos_after != 0), (
                    f"slot {slot}: decode step did not write KV at the new position {next_pos[slot]} "
                    "-- write-back side of the handoff is not wired"
                )

        # -------------------- Continue decoding to full length --------------------
        gen_tokens = [[first_tokens[s], int(second_tokens[s])] for s in range(FULL_BATCH_SIZE)]
        pos = position_ids + 1
        last_token = second_tokens.reshape(FULL_BATCH_SIZE, 1)
        for _ in range(GENERATION_LEN - 2):
            decode_session.set_data_for_kv_handoff(
                kv_caches + kv_caches,
                [("batch_index", 0), ("ctx_start", 0)],
                index=decode_session.decode_execObj_idx,
                buff_map=decode_kv_map,
            )
            decode_inputs = {
                "input_ids": last_token.astype(np.int64),
                "position_ids": pos.astype(np.int64),
                "batch_index": batch_index,
            }
            exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
            decode_session.complete_inf(exec_idx, is_prefill=False)
            out = decode_session.get_outputs(index=exec_idx)
            logits = out["logits"].reshape(FULL_BATCH_SIZE, -1, out["logits"].shape[-1])[:, -1, :]
            next_tokens = np.argmax(logits, axis=-1)
            for s in range(FULL_BATCH_SIZE):
                gen_tokens[s].append(int(next_tokens[s]))
            last_token = next_tokens.reshape(FULL_BATCH_SIZE, 1)
            pos = pos + 1
    finally:
        for session in sessions:
            session.deactivate()

    for slot in range(FULL_BATCH_SIZE):
        qaic_tokens = np.array(gen_tokens[slot], dtype=np.int64)
        ref_tokens = hf_tokens[slot]
        matches = ref_tokens == qaic_tokens
        num_matched = int(np.cumprod(matches).sum())
        print(f"\nslot[{slot}] prompt: {TEXT_PROMPTS[slot]}")
        print(f"HF Torch fp32 tokens   : {ref_tokens.tolist()}")
        print(f"Disagg CB QAIC tokens  : {qaic_tokens.tolist()}")
        print(f"Matched leading tokens : {num_matched}/{GENERATION_LEN}")
        if not matches.all():
            first_mismatch = int(np.argmin(matches))
            raise AssertionError(
                f"slot {slot}: tokens don't match HF Torch fp32 output; "
                f"first mismatch at token index {first_mismatch} "
                f"(matched {num_matched}/{GENERATION_LEN} leading tokens): "
                f"HF={ref_tokens[first_mismatch]} vs QAIC={qaic_tokens[first_mismatch]}"
            )
