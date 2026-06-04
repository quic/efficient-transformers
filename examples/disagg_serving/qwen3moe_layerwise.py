# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import functools
import os
import time
from pathlib import Path

import transformers
from transformers import AutoConfig, AutoTokenizer

import QEfficient
from QEfficient import QEFFAutoModelForCausalLM

model_id = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # weights are not required to convert to fp32
# model_id = "yujiepan/qwen3-moe-tiny-random"
prompt = """
Explain quantum computing in simple terms.
"""
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
PREFILL_SEQ_LEN = 4
CTX_LEN = 128


def _ensure_pretrained_window_attrs():
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_start"):
        transformers.modeling_utils.PreTrainedModel._start = 0
    if not hasattr(transformers.modeling_utils.PreTrainedModel, "_end"):
        transformers.modeling_utils.PreTrainedModel._end = 0


def _build_layer_windows(total_layers: int, window_size: int):
    if total_layers <= 0:
        raise ValueError(f"Invalid total_layers={total_layers}. Expected: total_layers > 0.")
    if window_size <= 0:
        raise ValueError(f"Invalid window_size={window_size}. Expected: window_size > 0.")

    windows = []
    end = total_layers
    while end > 0:
        start = max(0, end - window_size)
        windows.append((start, end))
        end = start

    return windows


def _null_outside_window_layers(model):
    start = int(getattr(transformers.modeling_utils.PreTrainedModel, "_start", 0))
    end = int(getattr(transformers.modeling_utils.PreTrainedModel, "_end", 0))
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for idx, _ in enumerate(layers):
        if idx < start or idx >= end:
            layers[idx] = None


def _install_window_patch(model_cls):
    if getattr(model_cls, "_window_patch_installed", False):
        return

    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _null_outside_window_layers(self)

    model_cls.__init__ = patched_init
    model_cls._window_patch_installed = True


def _resolve_export_root(onnx_path: Path) -> Path:
    parts = list(onnx_path.parts)
    if "onnx_layerwise_tmp" in parts:
        marker_idx = parts.index("onnx_layerwise_tmp")
        return Path(*parts[:marker_idx])
    return onnx_path.parent


def _install_shard_window_patch():
    if getattr(transformers.modeling_utils, "_window_shard_patch_installed", False):
        return

    original_get_checkpoint_shard_files = transformers.modeling_utils.get_checkpoint_shard_files

    @functools.wraps(original_get_checkpoint_shard_files)
    def patched_get_checkpoint_shard_files(*args, **kwargs):
        shard_files, metadata = original_get_checkpoint_shard_files(*args, **kwargs)
        weight_map = metadata.get("weight_map")
        if not weight_map:
            return shard_files, metadata

        start = int(getattr(transformers.modeling_utils.PreTrainedModel, "_start", 0))
        end = int(getattr(transformers.modeling_utils.PreTrainedModel, "_end", 0))
        if end <= start:
            return shard_files, metadata

        selected_prefixes = tuple(f"model.layers.{layer_idx}." for layer_idx in range(start, end))
        filtered_weight_map = {}
        for checkpoint_key, shard_name in weight_map.items():
            if checkpoint_key.startswith("model.layers."):
                if checkpoint_key.startswith(selected_prefixes):
                    filtered_weight_map[checkpoint_key] = shard_name
                continue
            filtered_weight_map[checkpoint_key] = shard_name

        if not filtered_weight_map:
            return shard_files, metadata

        shard_name_to_path = {path.split("/")[-1]: path for path in shard_files}
        filtered_shard_names = sorted(set(filtered_weight_map.values()))
        filtered_shard_files = [shard_name_to_path[name] for name in filtered_shard_names if name in shard_name_to_path]
        if not filtered_shard_files:
            return shard_files, metadata

        metadata["weight_map"] = filtered_weight_map
        metadata["all_checkpoint_keys"] = list(filtered_weight_map.keys())
        return filtered_shard_files, metadata

    transformers.modeling_utils.get_checkpoint_shard_files = patched_get_checkpoint_shard_files
    transformers.modeling_utils._window_shard_patch_installed = True


_ensure_pretrained_window_attrs()
_install_shard_window_patch()
text_config = getattr(config, "text_config", config)
resolved_total_layers = getattr(text_config, "num_hidden_layers", None)
if resolved_total_layers is None:
    raise ValueError("Could not resolve `num_hidden_layers` from config.")

# Layerwise window size. `1` keeps only one decoder layer active per window.
window_size = 1
total_layers = 2  # resolved_total_layers # config.num_hidden_layers = 1
windows = _build_layer_windows(total_layers=total_layers, window_size=window_size)
qeff_model = None
first_onnx_path = None
export_start = time.perf_counter()

os.environ["LAYERWISE_EXPORT"] = "True"
for start, end in windows:
    transformers.modeling_utils.PreTrainedModel._start = start
    transformers.modeling_utils.PreTrainedModel._end = end
    transformers.modeling_utils.PreTrainedModel._total_layers = total_layers
    QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe.QEffQwen3MoeModel._start = start
    QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe.QEffQwen3MoeModel._end = end
    QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe.QEffQwen3MoeModel._total_layers = total_layers
    QEfficient.base.modeling_qeff.QEFFBaseModel._start = start
    QEfficient.base.modeling_qeff.QEFFBaseModel._end = end
    QEfficient.base.modeling_qeff.QEFFBaseModel._total_layers = total_layers
    _install_window_patch(transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeForCausalLM)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, config=config)
    if hasattr(qeff_model, "model"):
        _null_outside_window_layers(qeff_model.model)

    # Following command errors out by default, the user is supposed to run the printed command and provide the generated qpc path as prefill_qpc_path commenting out lines 55-68

    # prefill_qpc_path = ""
    ################################# prefill

    onnx_path = qeff_model.compile(
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=CTX_LEN,
        num_cores=16,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        num_devices=1,
        split_retained_state_io=True,
        mos=1,
        aic_enable_depth_first=True,
        num_speculative_tokens=None,
        prefill_only=True,
        enable_chunking=True,
        use_onnx_subfunctions=True,
    )

    ################################# decode
    # onnx_path = qeff_model.compile(
    #     prefill_seq_len=PREFILL_SEQ_LEN,
    #     ctx_len=CTX_LEN,
    #     num_cores=16,
    #     mxfp6_matmul=True,
    #     mxint8_kv_cache=True,
    #     num_devices=1,
    #     split_retained_state_io=True,
    #     mos=1,
    #     aic_enable_depth_first=True,
    #     num_speculative_tokens=None,
    #     prefill_only=False,
    #     use_onnx_subfunctions=True,
    # )
    if first_onnx_path is None:
        first_onnx_path = Path(onnx_path)

if first_onnx_path is None:
    raise RuntimeError("No ONNX path produced during compilation.")
export_root = _resolve_export_root(first_onnx_path)
final_onnx_path = QEfficient.utils.layerwise_pipeline(str(export_root))
print(f"Layer-wise language export completed. Final artifact/root: {final_onnx_path}")
os.environ["LAYERWISE_EXPORT"] = "False"
qpc_path = qeff_model.compile(
    onnx_path=final_onnx_path,
    prefill_seq_len=PREFILL_SEQ_LEN,
    ctx_len=CTX_LEN,
    num_cores=16,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    num_devices=1,
    split_retained_state_io=True,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    prefill_only=True,
    enable_chunking=True,
    use_onnx_subfunctions=True,
)

print(f"QPC path: {qpc_path}")

# inputs = tokenizer(prompt, return_tensors="np", padding=True)
# position_ids = inputs["attention_mask"].sum(1, keepdims=True)
# generation_len = CTX_LEN - position_ids.max()
# padded_len = inputs["input_ids"].shape[1]
# num_chunks = -(padded_len // -PREFILL_SEQ_LEN)  # ceil divide without float
# padded_len = num_chunks * PREFILL_SEQ_LEN  # Convert to a multiple of prompt_len
# inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=padded_len)
# inputs["position_ids"] = np.where(inputs.pop("attention_mask"), np.arange(padded_len), -1)
# inputs.pop("token_type_ids", None)
# inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
# inputs.pop("past_key_values", None)
# inputs = {k: v.detach().numpy() for k, v in inputs.items()}


# prefill_session = QAICInferenceSession(prefill_qpc_path)


# all_outputs = []
# for i in range(num_chunks):
#     chunk_inputs = inputs.copy()
#     chunk_inputs["input_ids"] = inputs["input_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
#     chunk_inputs["position_ids"] = inputs["position_ids"][:, i * PREFILL_SEQ_LEN : (i + 1) * PREFILL_SEQ_LEN]
#     ins = time.time()
#     qpc_out = prefill_session.run(chunk_inputs)
#     print(f"time for this run={time.time() - ins}")
#     for i in range(config.num_hidden_layers):
#         inputs[f"past_key.{i}"] = qpc_out[f"past_key.{i}_RetainedState"]
#         inputs[f"past_value.{i}"] = qpc_out[f"past_value.{i}_RetainedState"]

# all_outputs.append(np.argmax(qpc_out["logits"]))
# print(all_outputs)
# print(">>>>>>>> export for prefill is done <<<<<<<<<<<")
# ###########################

# decode_qpc_path = qeff_model.compile(
#     prefill_seq_len=1,
#     ctx_len=CTX_LEN,
#     num_cores=16,
#     mxfp6_matmul=True,
#     mxint8_kv_cache=True,
#     num_devices=1,
#     mos=1,
#     aic_enable_depth_first=True,
#     num_speculative_tokens=None,
#     offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation in the next step
#     retain_full_kv=True,
# )
# decode_session = QAICInferenceSession(decode_qpc_path)

# decode_inputs = {
#     "input_ids": np.argmax(qpc_out["logits"]).reshape(1, 1),
#     "position_ids": np.max(inputs["position_ids"]).reshape(1, 1) + 1,
# }
# for i in range(config.num_hidden_layers):
#     decode_inputs[f"past_key.{i}"] = qpc_out[f"past_key.{i}_RetainedState"]
#     decode_inputs[f"past_value.{i}"] = qpc_out[f"past_value.{i}_RetainedState"]

# st = time.time()
# decode_out = decode_session.run(decode_inputs)
# print(f"time for first run of decode with KV as input = {time.time() - st} sec\n")
# all_outputs.append(np.argmax(decode_out["logits"]))
# pos_id = np.max(decode_inputs["position_ids"]).reshape(1, 1) + 1
# loop_decode_inputs = {
#     "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
#     "position_ids": pos_id,
# }

# for i in range(config.num_hidden_layers):
#     loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
#     loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]

# st = time.time()
# for i in range(generation_len - 2):
#     decode_out = decode_session.run(loop_decode_inputs)
#     all_outputs.append(np.argmax(decode_out["logits"]))
#     pos_id += 1
#     for i in range(config.num_hidden_layers):
#         loop_decode_inputs[f"past_key.{i}"] = decode_out[f"past_key.{i}_RetainedState"]
#         loop_decode_inputs[f"past_value.{i}"] = decode_out[f"past_value.{i}_RetainedState"]

#     loop_decode_inputs.update(
#         {
#             "input_ids": np.argmax(decode_out["logits"]).reshape(1, 1),
#             "position_ids": pos_id,
#         }
#     )
# ft = time.time()

# print(f"decode tok/sec={(generation_len - 2) / (ft - st)}")
# print(f"input\n{prompt}\noutput\n{tokenizer.decode(all_outputs)}")
