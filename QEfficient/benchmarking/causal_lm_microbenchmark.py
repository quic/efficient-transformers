# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.generation.text_generation_inference import write_io_files
from QEfficient.transformers.cache_utils import QEffDynamicCache, QEffHybridCacheForGPTOSS
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.models.gpt_oss.modeling_gpt_oss import (
    QEffGptOssExperts,
)
from QEfficient.transformers.models.llama.modeling_llama import (
    QEffLlamaAttention,
    QEffLlamaDecoderLayer,
    QEffLlamaRotaryEmbedding,
)
from QEfficient.transformers.transform import replace_module_with_qeff_layers
from QEfficient.utils._utils import get_padding_shape_from_config

if TYPE_CHECKING:
    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM


SUPPORTED_CAUSAL_RUNTIME_MODEL_IDS = {
    "gpt2": "hf-internal-testing/tiny-random-GPT2LMHeadModel",
    "codegen": "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    "falcon": "hf-internal-testing/tiny-random-FalconForCausalLM",
    "gptj": "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "mistral": "hf-internal-testing/tiny-random-MistralForCausalLM",
    "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "mpt": "hf-internal-testing/tiny-random-MptForCausalLM",
    "phi": "hf-internal-testing/tiny-random-PhiForCausalLM",
    "phi3": "tiny-random/phi-4",
    "qwen2": "yujiepan/qwen2-tiny-random",
    "starcoder2": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "granite": "hf-internal-testing/tiny-random-GraniteForCausalLM",
    "olmo2": "hf-internal-testing/tiny-random-Olmo2ForCausalLM",
    "gpt_oss": "tiny-random/gpt-oss-bf16",
}

BENCHMARK_TYPES = ("attention", "mlp", "moe")
BENCHMARK_MODES = ("prefill", "decode", "both")


@dataclass
class RuntimeStats:
    iterations: int
    mean_ms: float
    min_ms: float
    max_ms: float
    total_ms: float
    p50_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    throughput_ips: Optional[float] = None


@dataclass
class BenchmarkSummary:
    benchmark_type: str
    module_name: str
    mode: str
    model_name: str
    model_id: str
    architecture: str
    layer_index: int
    batch_size: int
    seq_len: int
    ctx_len: int
    resolved_dims: Dict[str, int]
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    onnx_path: str
    qpc_path: Optional[str]
    prefill_runtime: Optional[RuntimeStats]
    seed_prefill_ms: Optional[float]
    first_decode_ms: Optional[float]
    decode_runtime: Optional[RuntimeStats]
    io_dir: Optional[str] = None
    io_manifest_path: Optional[str] = None
    export_error: Optional[str] = None


@dataclass
class BenchmarkManifest:
    prefill_only: Optional[bool]
    enable_chunking: bool
    batch_size: int
    seq_len: int
    ctx_len: int
    num_cores: int
    num_devices: int
    warmup_runs: int
    benchmark_runs: int
    summaries: List[BenchmarkSummary]
    blocking_config: Optional[AttentionBlockingConfig] = None


@dataclass
class BenchmarkModuleSpec:
    benchmark_type: str
    module_name: str
    mode: str
    layer_index: int
    wrapper: nn.Module
    output_name: str


def resolve_model_id(model_name_or_path: str) -> Tuple[str, str]:
    resolved = SUPPORTED_CAUSAL_RUNTIME_MODEL_IDS.get(model_name_or_path, model_name_or_path)
    return model_name_or_path, resolved


def _build_position_ids(batch_size: int, seq_len: int, start: int = 0) -> np.ndarray:
    position_ids = np.arange(start, start + seq_len, dtype=np.int64).reshape(1, seq_len)
    return np.repeat(position_ids, batch_size, axis=0)


def _zeros_kv_cache(config, batch_size: int, ctx_len: int) -> Tuple[np.ndarray, np.ndarray]:
    kv_shape = get_padding_shape_from_config(config, batch_size, ctx_len)
    return np.zeros(kv_shape, dtype=np.float32), np.zeros(kv_shape, dtype=np.float32)


def _timed_session_runs(
    session: QAICInferenceSession,
    build_inputs,
    warmup_runs: int,
    benchmark_runs: int,
) -> RuntimeStats:
    for _ in range(warmup_runs):
        _ = _run_session(session, build_inputs())

    timings_ms = []
    for _ in range(benchmark_runs):
        inputs = build_inputs()
        start = perf_counter()
        _ = _run_session(session, inputs)
        timings_ms.append((perf_counter() - start) * 1000.0)

    total_ms = float(sum(timings_ms))
    timings_array = np.asarray(timings_ms, dtype=np.float64)
    return RuntimeStats(
        iterations=benchmark_runs,
        mean_ms=total_ms / benchmark_runs,
        min_ms=float(min(timings_ms)),
        max_ms=float(max(timings_ms)),
        total_ms=total_ms,
        p50_ms=float(np.percentile(timings_array, 50)),
        p99_ms=float(np.percentile(timings_array, 99)),
        throughput_ips=(benchmark_runs / (total_ms / 1000.0)) if total_ms else None,
    )


def _cast_inputs_for_session(session: QAICInferenceSession, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    casted_inputs = {}
    for input_name, value in inputs.items():
        if input_name not in session.binding_index_map:
            continue
        binding = session.bindings[session.binding_index_map[input_name]]
        dtype = session.aic_to_np_dtype_mapping[binding.type]
        casted_inputs[input_name] = np.ascontiguousarray(value.astype(dtype, copy=False))
    return casted_inputs


def _matching_allowed_shape_index(session: QAICInferenceSession, inputs: Dict[str, np.ndarray]) -> Optional[int]:
    if not session.allowed_shapes:
        return None
    for shape_index, allowed_shape in enumerate(session.allowed_shapes):
        matches = True
        for binding in session.bindings:
            if binding.name not in inputs:
                continue
            if list(inputs[binding.name].shape) != allowed_shape[binding.index][1]:
                matches = False
                break
        if matches:
            return shape_index
    return None


def _run_session(session: QAICInferenceSession, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    casted_inputs = _cast_inputs_for_session(session, inputs)
    allowed_shape_index = _matching_allowed_shape_index(session, casted_inputs)
    if allowed_shape_index is not None:
        output_buffers = {}
        allowed_shape = session.allowed_shapes[allowed_shape_index]
        for output_name in session.output_names:
            binding_index = session.binding_index_map[output_name]
            binding = session.bindings[binding_index]
            dtype = session.aic_to_np_dtype_mapping[binding.type]
            output_shape = tuple(allowed_shape[binding_index][1])
            output_buffers[output_name] = np.empty(output_shape, dtype=dtype)
        session.set_buffers(output_buffers)
    return session.run(casted_inputs)


class BenchmarkWrapperBase(nn.Module):
    benchmark_input_kind = "hidden"

    def build_example_inputs(self, batch_size: int, seq_len: int, ctx_len: int) -> Dict[str, object]:
        raise NotImplementedError

    def dynamic_axes(self, output_name: str) -> Dict[str, Dict[int, str]]:
        raise NotImplementedError

    def numpy_inputs(self, batch_size: int, seq_len: int, ctx_len: int, seed: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def input_shapes(self, batch_size: int, seq_len: int, ctx_len: int) -> Dict[str, List[int]]:
        raise NotImplementedError

    def output_shapes(self, batch_size: int, seq_len: int, ctx_len: int, output_name: str) -> Dict[str, List[int]]:
        raise NotImplementedError

    def specialization_values(self, batch_size: int, seq_len: int, ctx_len: int, mode: str) -> Dict[str, int]:
        return {"batch_size": batch_size, "seq_len": seq_len, "ctx_len": ctx_len}

    def build_decode_inputs(self, outputs: Dict[str, np.ndarray], position_ids: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError


def _benchmark_cache_root(summary: BenchmarkSummary) -> Path:
    onnx_path = Path(summary.onnx_path)
    return onnx_path.parent


def _manifest_cache_path(qeff_model: "QEFFAutoModelForCausalLM", manifest: BenchmarkManifest) -> Path:
    summary = manifest.summaries[0]
    payload = {
        "model_id": summary.model_id,
        "model_name": summary.model_name,
        "prefill_only": manifest.prefill_only,
        "enable_chunking": manifest.enable_chunking,
        "batch_size": manifest.batch_size,
        "seq_len": manifest.seq_len,
        "ctx_len": manifest.ctx_len,
        "num_cores": manifest.num_cores,
        "num_devices": manifest.num_devices,
        "modules": [module_summary.module_name for module_summary in manifest.summaries],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return _benchmark_cache_root(summary) / f"benchmark_manifest_{digest}.json"


def _report_cache_path(qeff_model: "QEFFAutoModelForCausalLM", summaries: List[BenchmarkSummary]) -> Path:
    summary = summaries[0]
    payload = {
        "model_id": summary.model_id,
        "model_name": summary.model_name,
        "batch_size": summary.batch_size,
        "seq_len": summary.seq_len,
        "ctx_len": summary.ctx_len,
        "modules": [module_summary.module_name for module_summary in summaries],
        "modes": [module_summary.mode for module_summary in summaries],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return _benchmark_cache_root(summary) / f"benchmark_report_{digest}.json"


def _benchmark_io_cache_dir(summary: BenchmarkSummary) -> Path:
    safe_module_name = summary.module_name.replace("/", "_")
    return (
        _benchmark_cache_root(summary)
        / "benchmark_io"
        / f"layer{summary.layer_index}_{summary.mode}_{safe_module_name}"
    )


def _contiguous_io_map(values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(array) for name, array in values.items()}


def _save_benchmark_io_artifacts(
    summary: BenchmarkSummary,
    phase_ios: List[Tuple[str, Dict[str, np.ndarray]]],
) -> Tuple[str, str]:
    io_dir = _benchmark_io_cache_dir(summary)
    io_dir.mkdir(parents=True, exist_ok=True)
    for index, (phase_name, inputs) in enumerate(phase_ios):
        write_io_files(
            _contiguous_io_map(inputs),
            {},
            str(io_dir),
            phase_name,
            "aic_batch_io",
            include_dims=True,
            reset=index == 0,
        )
    return str(io_dir), str(io_dir / "aic_batch_io.json")


def save_benchmark_manifest(
    qeff_model: "QEFFAutoModelForCausalLM",
    manifest: BenchmarkManifest,
) -> str:
    manifest_path = _manifest_cache_path(qeff_model, manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    qeff_model._benchmark_manifest_path = str(manifest_path)
    return str(manifest_path)


def save_benchmark_report(
    qeff_model: "QEFFAutoModelForCausalLM",
    summaries: List[BenchmarkSummary],
) -> str:
    report_path = _report_cache_path(qeff_model, summaries)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps([_summary_to_dict(summary) for summary in summaries], indent=2))
    qeff_model._benchmark_report_path = str(report_path)
    return str(report_path)


def _build_torch_position_ids(batch_size: int, seq_len: int, start: int = 0) -> torch.Tensor:
    return torch.arange(start, start + seq_len, dtype=torch.int64).view(1, seq_len).repeat(batch_size, 1)


def _prepare_direct_qeff_module(module: nn.Module) -> nn.Module:
    replace_module_with_qeff_layers(module)
    experts = getattr(module, "experts", None)
    if experts is not None and experts.__class__.__name__ == "GptOssExperts":
        experts.__class__ = QEffGptOssExperts
    for child in module.modules():
        if hasattr(child, "__qeff_init__"):
            child.__qeff_init__()
    return module.eval()


def _build_minimal_gpt_oss_qeff_bundle(config, *, layer_type: str, mode: str, enable_chunking: bool):
    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

    cfg = copy.deepcopy(config)
    cfg.num_hidden_layers = 1
    cfg.layer_types = [layer_type]
    hf_model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager", trust_remote_code=True)
    qeff_model = QEFFAutoModelForCausalLM(hf_model)
    if mode == "prefill":
        qeff_model.prefill(enable=True, enable_chunking=enable_chunking)
    else:
        qeff_model.prefill(enable=False)
    qeff_model.model.float()
    source_model = qeff_model.model.model
    return cfg, source_model, source_model.layers[0]


def _build_torch_causal_mask(
    position_ids: torch.Tensor,
    target_length: int,
    *,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    return _create_causal_mask(position_ids=position_ids, target_length=target_length, sliding_window=sliding_window)


def _build_numpy_causal_mask(
    position_ids: np.ndarray,
    target_length: int,
    *,
    sliding_window: Optional[int] = None,
) -> np.ndarray:
    position_ids_t = torch.from_numpy(position_ids.astype(np.int64, copy=False))
    return _build_torch_causal_mask(position_ids_t, target_length, sliding_window=sliding_window).cpu().numpy()


def _next_position_ids(position_ids: np.ndarray) -> np.ndarray:
    return (np.max(position_ids, axis=1, keepdims=True) + 1).astype(np.int64)


class LlamaCacheModuleBenchmarkWrapper(BenchmarkWrapperBase):
    benchmark_input_kind = "cache"
    past_key_input_name = "past_key.0"
    past_value_input_name = "past_value.0"

    def __init__(self, config, cos_cached: torch.Tensor, sin_cached: torch.Tensor):
        super().__init__()
        self.config = config
        self.register_buffer("cos_cached", cos_cached.detach().clone(), persistent=False)
        self.register_buffer("sin_cached", sin_cached.detach().clone(), persistent=False)

    def _kv_shape(self, batch_size: int, ctx_len: int) -> Tuple[int, ...]:
        return get_padding_shape_from_config(self.config, batch_size, ctx_len)

    def _example_cache(self, batch_size: int, ctx_len: int):
        kv_shape = self._kv_shape(batch_size, ctx_len)
        return [[torch.zeros(kv_shape, dtype=torch.float32), torch.zeros(kv_shape, dtype=torch.float32)]]

    def _numpy_cache(self, batch_size: int, ctx_len: int) -> Tuple[np.ndarray, np.ndarray]:
        return _zeros_kv_cache(self.config, batch_size, ctx_len)

    def dynamic_axes(self, output_name: str) -> Dict[str, Dict[int, str]]:
        return {
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 2: "seq_len", 3: "ctx_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "past_key.0": {0: "batch_size", 2: "ctx_len"},
            "past_value.0": {0: "batch_size", 2: "ctx_len"},
            output_name: {0: "batch_size", 1: "seq_len"},
            "past_key_RetainedState": {0: "batch_size", 2: "ctx_len"},
            "past_value_RetainedState": {0: "batch_size", 2: "ctx_len"},
        }

    def build_example_inputs(self, batch_size: int, seq_len: int, ctx_len: int) -> Dict[str, object]:
        position_ids = _build_torch_position_ids(batch_size, seq_len)
        return {
            "hidden_states": torch.zeros((batch_size, seq_len, self.config.hidden_size), dtype=torch.float32),
            "attention_mask": _build_torch_causal_mask(position_ids, ctx_len),
            "position_ids": position_ids,
            "past_key_values": self._example_cache(batch_size, ctx_len),
        }

    def numpy_inputs(self, batch_size: int, seq_len: int, ctx_len: int, seed: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        position_ids = _build_position_ids(batch_size, seq_len)
        past_key, past_value = self._numpy_cache(batch_size, ctx_len)
        return {
            "hidden_states": rng.standard_normal((batch_size, seq_len, self.config.hidden_size), dtype=np.float32),
            "attention_mask": _build_numpy_causal_mask(position_ids, ctx_len),
            "position_ids": position_ids,
            "past_key.0": past_key,
            "past_value.0": past_value,
        }

    def input_shapes(self, batch_size: int, seq_len: int, ctx_len: int) -> Dict[str, List[int]]:
        kv_shape = list(self._kv_shape(batch_size, ctx_len))
        return {
            "hidden_states": [batch_size, seq_len, self.config.hidden_size],
            "attention_mask": [batch_size, 1, seq_len, ctx_len],
            "position_ids": [batch_size, seq_len],
            "past_key.0": kv_shape,
            "past_value.0": kv_shape,
        }

    def build_decode_inputs(self, outputs: Dict[str, np.ndarray], position_ids: np.ndarray) -> Dict[str, np.ndarray]:
        next_position_ids = _next_position_ids(position_ids)
        next_ctx_len = outputs["past_key_RetainedState"].shape[2]
        return {
            "hidden_states": outputs["attention_output"][:, -1:, :],
            "attention_mask": _build_numpy_causal_mask(next_position_ids, next_ctx_len),
            "position_ids": next_position_ids,
            self.past_key_input_name: outputs["past_key_RetainedState"],
            self.past_value_input_name: outputs["past_value_RetainedState"],
        }

    def output_shapes(self, batch_size: int, seq_len: int, ctx_len: int, output_name: str) -> Dict[str, List[int]]:
        kv_shape = list(self._kv_shape(batch_size, ctx_len))
        return {
            output_name: [batch_size, seq_len, self.config.hidden_size],
            "past_key_RetainedState": kv_shape,
            "past_value_RetainedState": kv_shape,
        }


class LlamaAttentionBenchmarkWrapper(LlamaCacheModuleBenchmarkWrapper):
    def __init__(self, attention_module: nn.Module, cos_cached: torch.Tensor, sin_cached: torch.Tensor, config):
        super().__init__(config=config, cos_cached=cos_cached, sin_cached=sin_cached)
        self.attention = attention_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values,
    ):
        past_key_value = QEffDynamicCache.from_legacy_cache(past_key_values)
        attention_output, _ = self.attention(
            hidden_states=hidden_states,
            position_embeddings=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=True,
            cos_cached=self.cos_cached,
            sin_cached=self.sin_cached,
        )
        present_key, present_value = past_key_value.to_legacy_cache()[0]
        return attention_output, present_key, present_value


class DenseMlpBenchmarkWrapper(BenchmarkWrapperBase):
    benchmark_input_kind = "hidden"

    def __init__(self, mlp_module: nn.Module, config, *, returns_tuple: bool = False, output_name: str = "mlp_output"):
        super().__init__()
        self.mlp = mlp_module
        self.config = config
        self.returns_tuple = returns_tuple
        self._output_name = output_name

    def _normalize_hidden_output(self, hidden_output: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_output.dim() == 2:
            hidden_output = hidden_output.view(hidden_states.shape)
        return hidden_output

    def forward(self, hidden_states: torch.Tensor):
        outputs = self.mlp(hidden_states)
        if self.returns_tuple:
            return self._normalize_hidden_output(outputs[0], hidden_states)
        return self._normalize_hidden_output(outputs, hidden_states)

    def build_example_inputs(self, batch_size: int, seq_len: int, ctx_len: int) -> Dict[str, object]:
        return {"hidden_states": torch.zeros((batch_size, seq_len, self.config.hidden_size), dtype=torch.float32)}

    def dynamic_axes(self, output_name: str) -> Dict[str, Dict[int, str]]:
        return {
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            output_name: {0: "batch_size", 1: "seq_len"},
        }

    def numpy_inputs(self, batch_size: int, seq_len: int, ctx_len: int, seed: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {"hidden_states": rng.standard_normal((batch_size, seq_len, self.config.hidden_size), dtype=np.float32)}

    def input_shapes(self, batch_size: int, seq_len: int, ctx_len: int) -> Dict[str, List[int]]:
        return {"hidden_states": [batch_size, seq_len, self.config.hidden_size]}

    def output_shapes(self, batch_size: int, seq_len: int, ctx_len: int, output_name: str) -> Dict[str, List[int]]:
        return {output_name: [batch_size, seq_len, self.config.hidden_size]}

    def build_decode_inputs(self, outputs: Dict[str, np.ndarray], position_ids: np.ndarray) -> Dict[str, np.ndarray]:
        hidden_output = outputs[self._output_name]
        if hidden_output.ndim == 2:
            hidden_output = hidden_output[:, None, :]
        return {"hidden_states": hidden_output[:, -1:, :]}


class GptOssCacheModuleBenchmarkWrapper(BenchmarkWrapperBase):
    benchmark_input_kind = "cache"
    past_key_input_name = "past_key"
    past_value_input_name = "past_value"

    def __init__(
        self,
        attention_module: nn.Module,
        config,
        cos_cached: torch.Tensor,
        sin_cached: torch.Tensor,
        ctx_len: int,
        cache_len: Optional[int] = None,
        layer_index: int = 0,
    ):
        super().__init__()
        self.attention = attention_module
        self.config = config
        self.ctx_len = ctx_len
        self.cache_len = cache_len if cache_len is not None else ctx_len
        self.layer_index = layer_index
        self.register_buffer("cos_cached", cos_cached.detach().clone(), persistent=False)
        self.register_buffer("sin_cached", sin_cached.detach().clone(), persistent=False)

    @property
    def sliding_window_len(self) -> int:
        return self.cache_len

    @property
    def uses_sliding_window(self) -> bool:
        return getattr(self.attention, "sliding_window", None) is not None

    @property
    def cache_axis_name(self) -> str:
        if not self.uses_sliding_window:
            return "ctx_len"
        return "ctx_len" if self.cache_len != self.config_sliding_window else "sliding_window"

    @property
    def config_sliding_window(self) -> int:
        return int(getattr(self.attention, "sliding_window", None) or getattr(self.config, "sliding_window", 0))

    def _build_cache(self, past_key: torch.Tensor, past_value: torch.Tensor):
        cache = QEffHybridCacheForGPTOSS(
            self.config,
            batch_size=past_key.shape[0],
            max_cache_len=self.cache_len,
            sliding_window_len=self.sliding_window_len,
        )
        for _ in range(self.layer_index):
            cache.key_cache.append(torch.zeros_like(past_key))
            cache.value_cache.append(torch.zeros_like(past_value))
        cache.key_cache.append(past_key)
        cache.value_cache.append(past_value)
        return cache

    def _attention_mask_shape(self, batch_size: int, seq_len: int) -> List[int]:
        return [batch_size, 1, seq_len, self.cache_len]

    def _sliding_mask_shape(self, batch_size: int, seq_len: int) -> List[int]:
        return [batch_size, 1, seq_len, self.sliding_window_len]

    def build_example_inputs(self, batch_size: int, seq_len: int, ctx_len: int) -> Dict[str, object]:
        position_ids = _build_torch_position_ids(batch_size, seq_len)
        inputs = {
            "hidden_states": torch.zeros((batch_size, seq_len, self.config.hidden_size), dtype=torch.float32),
            "attention_mask": _build_torch_causal_mask(position_ids, self.cache_len),
            "position_ids": position_ids,
            "past_key": torch.zeros(
                get_padding_shape_from_config(self.config, batch_size, self.sliding_window_len), dtype=torch.float32
            ),
            "past_value": torch.zeros(
                get_padding_shape_from_config(self.config, batch_size, self.sliding_window_len), dtype=torch.float32
            ),
        }
        if self.uses_sliding_window:
            inputs["sliding_mask"] = _build_torch_causal_mask(
                position_ids,
                self.sliding_window_len,
                sliding_window=self.sliding_window_len,
            )
        return inputs

    def dynamic_axes(self, output_name: str) -> Dict[str, Dict[int, str]]:
        axes = {
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 2: "seq_len", 3: self.cache_axis_name},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "past_key": {0: "batch_size", 2: self.cache_axis_name},
            "past_value": {0: "batch_size", 2: self.cache_axis_name},
            output_name: {0: "batch_size", 1: "seq_len"},
            "past_key_RetainedState": {0: "batch_size", 2: self.cache_axis_name},
            "past_value_RetainedState": {0: "batch_size", 2: self.cache_axis_name},
        }
        if self.uses_sliding_window:
            axes["sliding_mask"] = {0: "batch_size", 2: "seq_len", 3: self.cache_axis_name}
        return axes

    def numpy_inputs(self, batch_size: int, seq_len: int, ctx_len: int, seed: int) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        position_ids = _build_position_ids(batch_size, seq_len)
        past_key, past_value = _zeros_kv_cache(self.config, batch_size, self.sliding_window_len)
        inputs = {
            "hidden_states": rng.standard_normal((batch_size, seq_len, self.config.hidden_size), dtype=np.float32),
            "attention_mask": _build_numpy_causal_mask(position_ids, self.cache_len),
            "position_ids": position_ids,
            "past_key": past_key,
            "past_value": past_value,
        }
        if self.uses_sliding_window:
            inputs["sliding_mask"] = _build_numpy_causal_mask(
                position_ids,
                self.sliding_window_len,
                sliding_window=self.sliding_window_len,
            )
        return inputs

    def input_shapes(self, batch_size: int, seq_len: int, ctx_len: int) -> Dict[str, List[int]]:
        kv_shape = list(get_padding_shape_from_config(self.config, batch_size, self.sliding_window_len))
        shapes = {
            "hidden_states": [batch_size, seq_len, self.config.hidden_size],
            "attention_mask": self._attention_mask_shape(batch_size, seq_len),
            "position_ids": [batch_size, seq_len],
            "past_key": kv_shape,
            "past_value": kv_shape,
        }
        if self.uses_sliding_window:
            shapes["sliding_mask"] = self._sliding_mask_shape(batch_size, seq_len)
        return shapes

    def output_shapes(self, batch_size: int, seq_len: int, ctx_len: int, output_name: str) -> Dict[str, List[int]]:
        kv_shape = list(get_padding_shape_from_config(self.config, batch_size, self.sliding_window_len))
        return {
            output_name: [batch_size, seq_len, self.config.hidden_size],
            "past_key_RetainedState": kv_shape,
            "past_value_RetainedState": kv_shape,
        }

    def specialization_values(self, batch_size: int, seq_len: int, ctx_len: int, mode: str) -> Dict[str, int]:
        specializations = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "ctx_len": self.cache_len,
        }
        if self.uses_sliding_window:
            specializations["sliding_window"] = self.config_sliding_window
        return specializations

    def build_decode_inputs(self, outputs: Dict[str, np.ndarray], position_ids: np.ndarray) -> Dict[str, np.ndarray]:
        next_position_ids = _next_position_ids(position_ids)
        inputs = {
            "hidden_states": outputs["attention_output"][:, -1:, :],
            "attention_mask": _build_numpy_causal_mask(next_position_ids, self.cache_len),
            "position_ids": next_position_ids,
            self.past_key_input_name: outputs["past_key_RetainedState"],
            self.past_value_input_name: outputs["past_value_RetainedState"],
        }
        if self.uses_sliding_window:
            inputs["sliding_mask"] = _build_numpy_causal_mask(
                next_position_ids,
                outputs["past_key_RetainedState"].shape[2],
                sliding_window=outputs["past_key_RetainedState"].shape[2],
            )
        return inputs


class GptOssAttentionBenchmarkWrapper(GptOssCacheModuleBenchmarkWrapper):
    def __init__(
        self,
        attention_module: nn.Module,
        config,
        cos_cached: torch.Tensor,
        sin_cached: torch.Tensor,
        ctx_len: int,
        cache_len: Optional[int] = None,
        layer_index: int = 0,
    ):
        super().__init__(attention_module, config, cos_cached, sin_cached, ctx_len, cache_len, layer_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        sliding_mask: Optional[torch.Tensor] = None,
    ):
        past_key_value = self._build_cache(past_key, past_value)
        results = self.attention(
            hidden_states=hidden_states,
            position_embeddings=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=True,
            sliding_mask=sliding_mask,
            sin_cached=self.sin_cached,
            cos_cached=self.cos_cached,
        )
        attn_output = results[0]
        return attn_output, past_key_value.key_cache[self.layer_index], past_key_value.value_cache[self.layer_index]


class LlamaArchitectureAdapter:
    benchmarkable_types = {"attention", "mlp"}
    supports_combined_prefill_decode = True

    @staticmethod
    def matches(qeff_model: "QEFFAutoModelForCausalLM") -> bool:
        return getattr(qeff_model.model.config, "model_type", None) == "llama"

    @staticmethod
    def resolved_dims(qeff_model: "QEFFAutoModelForCausalLM") -> Dict[str, int]:
        config = qeff_model.model.config
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // num_attention_heads)
        return {
            "hidden_size": config.hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": getattr(config, "intermediate_size", 0),
        }

    @staticmethod
    def list_specs(
        qeff_model: "QEFFAutoModelForCausalLM",
        mode: str,
        layer_index: int,
        seq_len: int,
        ctx_len: int,
        enable_chunking: bool = False,
        blocking_config: Optional[AttentionBlockingConfig] = None,
    ) -> List[BenchmarkModuleSpec]:
        config = qeff_model.model.config
        rotary_emb = QEffLlamaRotaryEmbedding(config)
        blocking_suffix = (
            f"_blocked_{blocking_config.mode.value}"
            if blocking_config and blocking_config.mode != BlockingMode.NONE
            else ""
        )
        specs = []
        if mode in {"prefill", "decode", "both"}:
            attn_module = _prepare_direct_qeff_module(QEffLlamaAttention(config, layer_index))
            if blocking_config and blocking_config.mode != BlockingMode.NONE:
                attn_module.attn_blocking_config = blocking_config
            specs.append(
                BenchmarkModuleSpec(
                    benchmark_type="attention",
                    module_name=f"attention{blocking_suffix}",
                    mode=mode,
                    layer_index=layer_index,
                    wrapper=LlamaAttentionBenchmarkWrapper(
                        attention_module=attn_module,
                        cos_cached=rotary_emb.cos_cached,
                        sin_cached=rotary_emb.sin_cached,
                        config=config,
                    ),
                    output_name="attention_output",
                )
            )
            specs.append(
                BenchmarkModuleSpec(
                    benchmark_type="mlp",
                    module_name="mlp",
                    mode=mode,
                    layer_index=layer_index,
                    wrapper=DenseMlpBenchmarkWrapper(
                        mlp_module=_prepare_direct_qeff_module(QEffLlamaDecoderLayer(config, layer_index)).mlp,
                        config=config,
                        output_name="mlp_output",
                    ),
                    output_name="mlp_output",
                )
            )
        return specs


class GptOssArchitectureAdapter:
    benchmarkable_types = {"attention", "mlp", "moe"}
    supports_combined_prefill_decode = True

    @staticmethod
    def matches(qeff_model: "QEFFAutoModelForCausalLM") -> bool:
        return getattr(qeff_model.model.config, "model_type", None) == "gpt_oss"

    @staticmethod
    def resolved_dims(qeff_model: "QEFFAutoModelForCausalLM") -> Dict[str, int]:
        config = qeff_model.model.config
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // num_attention_heads)
        return {
            "hidden_size": config.hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": getattr(config, "intermediate_size", 0),
            "sliding_window": getattr(config, "sliding_window", 0),
            "num_local_experts": getattr(config, "num_local_experts", 0),
        }

    @staticmethod
    def list_specs(
        qeff_model: "QEFFAutoModelForCausalLM",
        mode: str,
        layer_index: int,
        seq_len: int,
        ctx_len: int,
        enable_chunking: bool = False,
        blocking_config: Optional[AttentionBlockingConfig] = None,
    ) -> List[BenchmarkModuleSpec]:
        config = qeff_model.model.config
        blocking_suffix = (
            f"_blocked_{blocking_config.mode.value}"
            if blocking_config and blocking_config.mode != BlockingMode.NONE
            else ""
        )

        layer_variants = []
        for variant_index, variant_name in [
            ("sliding_attention", "swa_attention"),
            ("full_attention", "full_attention"),
        ]:
            match_index = None
            for i, layer_type in enumerate(getattr(config, "layer_types", [])):
                if layer_type == variant_index:
                    match_index = i
                    break
            if match_index is None:
                continue
            layer_variants.append((match_index, variant_name))

        if not layer_variants:
            layer_variants.append((layer_index, "attention"))

        specs = []
        bundle_cache = {}
        for variant_layer_index, variant_name in layer_variants:
            layer_type = "sliding_attention" if variant_name == "swa_attention" else "full_attention"
            bundle_mode = "decode" if mode == "both" else mode
            cache_key = (layer_type, bundle_mode, bool(enable_chunking and mode == "prefill"))
            if cache_key not in bundle_cache:
                bundle_cache[cache_key] = _build_minimal_gpt_oss_qeff_bundle(
                    config,
                    layer_type=layer_type,
                    mode=bundle_mode,
                    enable_chunking=enable_chunking if bundle_mode == "prefill" else False,
                )
            variant_config, source_model, layer = bundle_cache[cache_key]
            if mode == "prefill":
                effective_cache_len = seq_len + getattr(config, "sliding_window", 0) if enable_chunking else seq_len
            elif variant_name == "swa_attention":
                effective_cache_len = getattr(config, "sliding_window", ctx_len)
            else:
                effective_cache_len = ctx_len
            if mode == "prefill" and enable_chunking:
                attn_name = f"prefill_chunked_{variant_name}"
            elif mode == "prefill":
                attn_name = f"prefill_{variant_name}"
            else:
                attn_name = variant_name

            attn_module = layer.self_attn
            if blocking_config and blocking_config.mode != BlockingMode.NONE and variant_name != "swa_attention":
                attn_module.attn_blocking_config = blocking_config
                attn_name = f"{attn_name}{blocking_suffix}"

            specs.append(
                BenchmarkModuleSpec(
                    benchmark_type="attention",
                    module_name=attn_name,
                    mode=mode,
                    layer_index=variant_layer_index,
                    wrapper=GptOssAttentionBenchmarkWrapper(
                        attention_module=attn_module,
                        config=variant_config,
                        cos_cached=source_model.cos_cached,
                        sin_cached=source_model.sin_cached,
                        ctx_len=ctx_len,
                        cache_len=effective_cache_len,
                        layer_index=0,
                    ),
                    output_name="attention_output",
                )
            )
        if mode == "prefill" and enable_chunking:
            mlp_name = "prefill_chunked_moe"
        elif mode == "prefill":
            mlp_name = "prefill_moe"
        else:
            mlp_name = "moe"
        first_layer_type = "sliding_attention" if layer_variants[0][1] == "swa_attention" else "full_attention"
        first_cache_key = (first_layer_type, bundle_mode, bool(enable_chunking and mode == "prefill"))
        if first_cache_key not in bundle_cache:
            bundle_cache[first_cache_key] = _build_minimal_gpt_oss_qeff_bundle(
                config,
                layer_type=first_layer_type,
                mode=bundle_mode,
                enable_chunking=enable_chunking if bundle_mode == "prefill" else False,
            )
        variant_config, _, mlp_layer = bundle_cache[first_cache_key]
        specs.append(
            BenchmarkModuleSpec(
                benchmark_type="moe",
                module_name=mlp_name,
                mode=mode,
                layer_index=layer_variants[0][0],
                wrapper=DenseMlpBenchmarkWrapper(
                    mlp_module=mlp_layer.mlp,
                    config=variant_config,
                    returns_tuple=True,
                    output_name="mlp_output",
                ),
                output_name="mlp_output",
            )
        )
        return specs


class CausalLMModuleBenchmarkModel(QEFFBaseModel):
    _pytorch_transforms: List = []

    def __init__(
        self, model: BenchmarkWrapperBase, output_name: str, model_name: str, model_id: str, module_name: str = ""
    ):
        self._benchmark_model_name = model_name
        self._benchmark_model_id = model_id
        self._output_name = output_name
        self._module_name = module_name
        super().__init__(model)
        self.model_architecture = model_name.removesuffix("Model") if model_name.endswith("LMModel") else model_name
        self.hash_params["benchmark_output_name"] = output_name
        self.hash_params["benchmark_model_name"] = model_name
        self.hash_params["benchmark_model_id"] = model_id
        if module_name:
            self.hash_params["benchmark_module_name"] = module_name

    @property
    def model_name(self) -> str:
        if self._module_name:
            return self._module_name
        return super().model_name

    @property
    def get_model_config(self) -> Dict:
        return self.model.config.to_dict()

    def export(
        self,
        export_dir: Optional[str] = None,
        *,
        batch_size: int = 1,
        seq_len: int = 32,
        ctx_len: int = 128,
        offload_pt_weights: bool = False,
    ) -> str:
        example_inputs = self.model.build_example_inputs(batch_size=batch_size, seq_len=seq_len, ctx_len=ctx_len)
        output_names = [self._output_name]
        if self.model.benchmark_input_kind == "cache":
            output_names.extend(["past_key_RetainedState", "past_value_RetainedState"])
        return self._export(
            example_inputs=example_inputs,
            output_names=output_names,
            dynamic_axes=self.model.dynamic_axes(self._output_name),
            export_dir=export_dir,
            offload_pt_weights=offload_pt_weights,
        )

    def compile(
        self,
        onnx_path: Optional[str] = None,
        compile_dir: Optional[str] = None,
        *,
        batch_size: int = 1,
        seq_len: int = 32,
        ctx_len: int = 128,
        mode: str = "both",
        num_devices: int = 1,
        num_cores: int = 16,
        mxint8_kv_cache: bool = False,
        **compiler_options,
    ) -> str:
        specializations = []
        prefill_specialization = self.model.specialization_values(batch_size, seq_len, ctx_len, "prefill")
        decode_specialization = self.model.specialization_values(batch_size, 1, ctx_len, "decode")

        if mode == "prefill":
            specializations.append(prefill_specialization)
        elif mode == "decode":
            specializations.append(decode_specialization)
        else:
            specializations.extend([prefill_specialization, decode_specialization])
        specializations = _dedupe_specializations(specializations)

        custom_io = None
        if self.model.benchmark_input_kind == "cache":
            kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
            custom_io = {
                self.model.past_key_input_name: kv_cache_dtype,
                self.model.past_value_input_name: kv_cache_dtype,
                "past_key_RetainedState": kv_cache_dtype,
                "past_value_RetainedState": kv_cache_dtype,
            }
        return self._compile(
            onnx_path=onnx_path,
            compile_dir=compile_dir,
            compile_only=True,
            retained_state=self.model.benchmark_input_kind == "cache",
            specializations=specializations,
            convert_to_fp16=True,
            custom_io=custom_io,
            mdp_ts_num_devices=num_devices,
            aic_num_cores=num_cores,
            mxint8_kv_cache=mxint8_kv_cache,
            **compiler_options,
        )


def _resolve_layers(qeff_model):
    model = qeff_model.model
    for model_attr in ("model", "transformer"):
        inner = getattr(model, model_attr, None)
        if inner is None:
            continue
        for layer_attr in ("layers", "h", "blocks"):
            layers = getattr(inner, layer_attr, None)
            if layers is not None and len(layers) > 0:
                return inner, layers
    return None, None


class GenericArchitectureAdapter:
    benchmarkable_types = {"attention", "mlp"}
    supports_combined_prefill_decode = True

    @staticmethod
    def matches(qeff_model):
        _, layers = _resolve_layers(qeff_model)
        if layers is None:
            return False
        return hasattr(layers[0], "self_attn") or hasattr(layers[0], "attn")

    @staticmethod
    def resolved_dims(qeff_model):
        config = qeff_model.model.config
        num_attention_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", 0))
        num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // max(num_attention_heads, 1))
        return {
            "hidden_size": config.hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": getattr(config, "intermediate_size", 0),
            "sliding_window": getattr(config, "sliding_window", 0),
        }

    @staticmethod
    @staticmethod
    def list_specs(
        qeff_model,
        mode,
        layer_index,
        seq_len,
        ctx_len,
        enable_chunking=False,
        blocking_config=None,
    ):
        config = qeff_model.model.config
        _, layers = _resolve_layers(qeff_model)
        has_sliding = getattr(config, "sliding_window", None) is not None
        blocking_suffix = (
            f"_blocked_{blocking_config.mode.value}"
            if blocking_config and blocking_config.mode != BlockingMode.NONE
            else ""
        )
        cos_dummy = torch.ones(1, 1, 1, getattr(config, "head_dim", 32))
        sin_dummy = torch.zeros(1, 1, 1, getattr(config, "head_dim", 32))

        attn_variants = []
        if has_sliding:
            seen_sliding, seen_full = False, False
            for idx, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
                if attn is None:
                    continue
                is_sliding = getattr(attn, "is_sliding", False)
                if is_sliding and not seen_sliding:
                    attn_variants.append((idx, "swa_attention", attn))
                    seen_sliding = True
                elif not is_sliding and not seen_full:
                    attn_variants.append((idx, "full_attention", attn))
                    seen_full = True
                if seen_sliding and seen_full:
                    break
        if not attn_variants:
            layer = layers[layer_index]
            attn = getattr(layer, "self_attn", getattr(layer, "attn", None))
            if attn is not None:
                attn_variants.append((layer_index, "attention", attn))

        specs = []
        for variant_idx, variant_name, attn_module in attn_variants:
            apply_blocking = (
                blocking_config and blocking_config.mode != BlockingMode.NONE and variant_name != "swa_attention"
            )
            if apply_blocking:
                attn_module.attn_blocking_config = blocking_config
                name = f"{variant_name}{blocking_suffix}"
            else:
                name = variant_name
            if mode == "prefill":
                name = f"prefill_{name}"

            if variant_name == "swa_attention":
                effective_cache_len = (
                    getattr(config, "sliding_window", ctx_len)
                    if mode != "prefill"
                    else seq_len + getattr(config, "sliding_window", 0)
                )
            else:
                effective_cache_len = ctx_len if mode != "prefill" else seq_len

            specs.append(
                BenchmarkModuleSpec(
                    benchmark_type="attention",
                    module_name=name,
                    mode=mode,
                    layer_index=variant_idx,
                    wrapper=GptOssAttentionBenchmarkWrapper(
                        attention_module=attn_module,
                        config=config,
                        cos_cached=cos_dummy,
                        sin_cached=sin_dummy,
                        ctx_len=ctx_len,
                        cache_len=effective_cache_len,
                        layer_index=0,
                    ),
                    output_name="attention_output",
                )
            )

        mlp_module = getattr(layers[layer_index], "mlp", None)
        if mlp_module is not None:
            mlp_name = "mlp" if mode != "prefill" else "prefill_mlp"
            specs.append(
                BenchmarkModuleSpec(
                    benchmark_type="mlp",
                    module_name=mlp_name,
                    mode=mode,
                    layer_index=layer_index,
                    wrapper=DenseMlpBenchmarkWrapper(
                        mlp_module=mlp_module,
                        config=config,
                        output_name="mlp_output",
                    ),
                    output_name="mlp_output",
                )
            )
        return specs


def _resolve_adapter(qeff_model):
    for adapter in (LlamaArchitectureAdapter, GptOssArchitectureAdapter, GenericArchitectureAdapter):
        if adapter.matches(qeff_model):
            return adapter
    raise NotImplementedError(
        f"Microbenchmarking: could not find decoder layers for model_type="
        f"{getattr(qeff_model.model.config, 'model_type', None)}."
    )


def _resolve_benchmark_modes(adapter, requested_mode: str) -> Tuple[str, ...]:
    if requested_mode != "both":
        return (requested_mode,)
    if getattr(adapter, "supports_combined_prefill_decode", False):
        return ("both",)
    return ("prefill", "decode")


def _dedupe_specializations(specializations: List[Dict[str, int]]) -> List[Dict[str, int]]:
    deduped = []
    seen = set()
    for spec in specializations:
        key = tuple(sorted(spec.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def get_benchmark_module_specs(
    qeff_model: "QEFFAutoModelForCausalLM",
    *,
    mode: str = "decode",
    layer_index: int = 0,
    seq_len: int = 32,
    ctx_len: int = 128,
    enable_chunking: bool = False,
    blocking_config: Optional[AttentionBlockingConfig] = None,
) -> List[BenchmarkModuleSpec]:
    if mode not in {"prefill", "decode", "both"}:
        raise ValueError("get_benchmark_module_specs supports `prefill`, `decode`, or `both`.")
    adapter = _resolve_adapter(qeff_model)
    return adapter.list_specs(
        qeff_model=qeff_model,
        mode=mode,
        layer_index=layer_index,
        seq_len=seq_len,
        ctx_len=ctx_len,
        enable_chunking=enable_chunking,
        blocking_config=blocking_config,
    )


def export_benchmark_modules(
    qeff_model: "QEFFAutoModelForCausalLM",
    *,
    mode: str = "both",
    benchmark_type: Optional[str] = None,
    batch_size: int = 1,
    seq_len: int = 32,
    ctx_len: int = 128,
    layer_index: int = 0,
    export_dir: Optional[str] = None,
    enable_chunking: bool = False,
    blocking_config: Optional[AttentionBlockingConfig] = None,
) -> List[BenchmarkSummary]:
    model_name = getattr(qeff_model, "benchmark_model_name", qeff_model.model_name)
    model_id = qeff_model.hash_params.get("pretrained_model_name_or_path", model_name)
    adapter = _resolve_adapter(qeff_model)
    concrete_modes = _resolve_benchmark_modes(adapter, mode)
    summaries = []

    for concrete_mode in concrete_modes:
        specs = get_benchmark_module_specs(
            qeff_model,
            mode=concrete_mode,
            layer_index=layer_index,
            seq_len=seq_len,
            ctx_len=ctx_len,
            enable_chunking=enable_chunking,
            blocking_config=blocking_config,
        )
        for spec in specs:
            if benchmark_type and spec.benchmark_type != benchmark_type:
                continue
            benchmark_model = CausalLMModuleBenchmarkModel(
                model=spec.wrapper,
                output_name=spec.output_name,
                model_name=model_name,
                model_id=model_id,
                module_name=spec.module_name,
            )
            try:
                onnx_path = benchmark_model.export(
                    export_dir=export_dir,
                    batch_size=batch_size,
                    seq_len=1 if spec.mode == "decode" else seq_len,
                    ctx_len=ctx_len,
                    offload_pt_weights=False,
                )
            except Exception as exc:
                summaries.append(
                    BenchmarkSummary(
                        benchmark_type=spec.benchmark_type,
                        module_name=spec.module_name,
                        mode=concrete_mode,
                        model_name=model_name,
                        model_id=model_id,
                        architecture=getattr(qeff_model.model.config, "model_type", "unknown"),
                        layer_index=spec.layer_index,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        ctx_len=ctx_len,
                        resolved_dims={},
                        input_shapes={},
                        output_shapes={},
                        onnx_path="",
                        qpc_path=None,
                        prefill_runtime=None,
                        seed_prefill_ms=None,
                        first_decode_ms=None,
                        decode_runtime=None,
                        export_error=str(exc),
                    )
                )
                continue
            summaries.append(
                BenchmarkSummary(
                    benchmark_type=spec.benchmark_type,
                    module_name=spec.module_name,
                    mode=concrete_mode,
                    model_name=model_name,
                    model_id=model_id,
                    architecture=getattr(qeff_model.model.config, "model_type", "unknown"),
                    layer_index=spec.layer_index,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    ctx_len=ctx_len,
                    resolved_dims=adapter.resolved_dims(qeff_model),
                    input_shapes=spec.wrapper.input_shapes(batch_size, seq_len, ctx_len),
                    output_shapes=spec.wrapper.output_shapes(batch_size, seq_len, ctx_len, spec.output_name),
                    onnx_path=str(onnx_path),
                    qpc_path=None,
                    prefill_runtime=None,
                    seed_prefill_ms=None,
                    first_decode_ms=None,
                    decode_runtime=None,
                )
            )
    return summaries


def compile_benchmark_modules(
    qeff_model: "QEFFAutoModelForCausalLM",
    *,
    prefill_only: Optional[bool] = None,
    batch_size: int = 1,
    seq_len: int = 32,
    ctx_len: int = 128,
    layer_index: int = 0,
    num_cores: int = 16,
    num_devices: int = 1,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    export_dir: Optional[str] = None,
    compile_dir: Optional[str] = None,
    export_only: bool = False,
    benchmark_type: Optional[str] = None,
    mxint8_kv_cache: bool = False,
    seed: int = 13,
    enable_chunking: bool = False,
    blocking_config: Optional[AttentionBlockingConfig] = None,
    **compiler_options,
) -> BenchmarkManifest:
    adapter = _resolve_adapter(qeff_model)
    effective_prefill_only = False if prefill_only is None and seq_len == 1 else prefill_only
    if effective_prefill_only is True:
        concrete_modes = ("prefill",)
    elif effective_prefill_only is False:
        concrete_modes = ("decode",)
    else:
        concrete_modes = _resolve_benchmark_modes(adapter, "both")

    summaries = []
    for concrete_mode in concrete_modes:
        specs = get_benchmark_module_specs(
            qeff_model,
            mode=concrete_mode,
            layer_index=layer_index,
            seq_len=seq_len,
            ctx_len=ctx_len,
            enable_chunking=enable_chunking if concrete_mode == "prefill" else False,
            blocking_config=blocking_config,
        )
        for spec in specs:
            if benchmark_type and spec.benchmark_type != benchmark_type:
                continue
            benchmark_model = CausalLMModuleBenchmarkModel(
                model=spec.wrapper,
                output_name=spec.output_name,
                model_name=getattr(qeff_model, "benchmark_model_name", qeff_model.model_name),
                model_id=qeff_model.hash_params.get("pretrained_model_name_or_path", qeff_model.model_name),
                module_name=spec.module_name,
            )
            try:
                onnx_path = benchmark_model.export(
                    export_dir=export_dir,
                    batch_size=batch_size,
                    seq_len=1 if spec.mode == "decode" else seq_len,
                    ctx_len=ctx_len,
                    offload_pt_weights=False,
                )
            except Exception as exc:
                summaries.append(
                    BenchmarkSummary(
                        benchmark_type=spec.benchmark_type,
                        module_name=spec.module_name,
                        mode=spec.mode,
                        model_name=getattr(qeff_model, "benchmark_model_name", qeff_model.model_name),
                        model_id=qeff_model.hash_params.get("pretrained_model_name_or_path", qeff_model.model_name),
                        architecture=getattr(qeff_model.model.config, "model_type", "unknown"),
                        layer_index=spec.layer_index,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        ctx_len=ctx_len,
                        resolved_dims={},
                        input_shapes={},
                        output_shapes={},
                        onnx_path="",
                        qpc_path=None,
                        prefill_runtime=None,
                        seed_prefill_ms=None,
                        first_decode_ms=None,
                        decode_runtime=None,
                        export_error=str(exc),
                    )
                )
                continue
            qpc_path = None
            if not export_only:
                qpc_path = benchmark_model.compile(
                    onnx_path=onnx_path,
                    compile_dir=compile_dir,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    ctx_len=ctx_len,
                    mode=spec.mode,
                    num_cores=num_cores,
                    num_devices=num_devices,
                    mxint8_kv_cache=mxint8_kv_cache,
                    **compiler_options,
                )
            summaries.append(
                BenchmarkSummary(
                    benchmark_type=spec.benchmark_type,
                    module_name=spec.module_name,
                    mode=spec.mode,
                    model_name=getattr(qeff_model, "benchmark_model_name", qeff_model.model_name),
                    model_id=qeff_model.hash_params.get("pretrained_model_name_or_path", qeff_model.model_name),
                    architecture=getattr(qeff_model.model.config, "model_type", "unknown"),
                    layer_index=spec.layer_index,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    ctx_len=ctx_len,
                    resolved_dims=adapter.resolved_dims(qeff_model),
                    input_shapes=spec.wrapper.input_shapes(batch_size, seq_len, ctx_len),
                    output_shapes=spec.wrapper.output_shapes(batch_size, seq_len, ctx_len, spec.output_name),
                    onnx_path=str(onnx_path),
                    qpc_path=str(qpc_path) if qpc_path else None,
                    prefill_runtime=None,
                    seed_prefill_ms=None,
                    first_decode_ms=None,
                    decode_runtime=None,
                )
            )

    manifest = BenchmarkManifest(
        prefill_only=effective_prefill_only,
        enable_chunking=enable_chunking,
        batch_size=batch_size,
        seq_len=seq_len,
        ctx_len=ctx_len,
        num_cores=num_cores,
        num_devices=num_devices,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        summaries=summaries,
        blocking_config=blocking_config,
    )
    qeff_model._benchmark_manifest = manifest
    save_benchmark_manifest(qeff_model, manifest)
    return manifest


def _run_decode_benchmark(
    session: QAICInferenceSession,
    wrapper: BenchmarkWrapperBase,
    seed_outputs: Dict[str, np.ndarray],
    seed_position_ids: np.ndarray,
    warmup_runs: int,
    benchmark_runs: int,
) -> Tuple[float, RuntimeStats]:
    decode_inputs = wrapper.build_decode_inputs(seed_outputs, seed_position_ids)
    start = perf_counter()
    outputs = _run_session(session, decode_inputs)
    first_decode_ms = (perf_counter() - start) * 1000.0
    next_position_ids = decode_inputs.get("position_ids", seed_position_ids)
    decode_inputs = wrapper.build_decode_inputs(outputs, next_position_ids)

    for _ in range(warmup_runs):
        outputs = _run_session(session, decode_inputs)
        next_position_ids = decode_inputs.get("position_ids", next_position_ids)
        decode_inputs = wrapper.build_decode_inputs(outputs, next_position_ids)

    timings_ms = []
    for _ in range(benchmark_runs):
        start = perf_counter()
        outputs = _run_session(session, decode_inputs)
        timings_ms.append((perf_counter() - start) * 1000.0)
        next_position_ids = decode_inputs.get("position_ids", next_position_ids)
        decode_inputs = wrapper.build_decode_inputs(outputs, next_position_ids)

    total_ms = float(sum(timings_ms))
    timings_array = np.asarray(timings_ms, dtype=np.float64)
    stats = RuntimeStats(
        iterations=benchmark_runs,
        mean_ms=total_ms / benchmark_runs,
        min_ms=float(min(timings_ms)),
        max_ms=float(max(timings_ms)),
        total_ms=total_ms,
        p50_ms=float(np.percentile(timings_array, 50)),
        p99_ms=float(np.percentile(timings_array, 99)),
        throughput_ips=(benchmark_runs / (total_ms / 1000.0)) if total_ms else None,
    )
    return first_decode_ms, stats


def _run_prefill_and_decode_benchmark(
    session: QAICInferenceSession,
    wrapper: BenchmarkWrapperBase,
    *,
    batch_size: int,
    seq_len: int,
    ctx_len: int,
    seed: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> Tuple[RuntimeStats, float, float, RuntimeStats]:
    prefill_runtime = _timed_session_runs(
        session=session,
        build_inputs=lambda: wrapper.numpy_inputs(batch_size, seq_len, ctx_len, seed),
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )
    raw_seed_inputs = wrapper.numpy_inputs(batch_size, seq_len, ctx_len, seed)
    seed_position_ids = raw_seed_inputs.get("position_ids", _build_position_ids(batch_size, seq_len))
    start = perf_counter()
    seed_outputs = _run_session(session, raw_seed_inputs)
    seed_prefill_ms = (perf_counter() - start) * 1000.0
    first_decode_ms, decode_runtime = _run_decode_benchmark(
        session=session,
        wrapper=wrapper,
        seed_outputs=seed_outputs,
        seed_position_ids=seed_position_ids,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )
    return prefill_runtime, seed_prefill_ms, first_decode_ms, decode_runtime


def _collect_phase_io_artifacts(
    summary: BenchmarkSummary,
    session: QAICInferenceSession,
    wrapper: BenchmarkWrapperBase,
    *,
    batch_size: int,
    seq_len: int,
    ctx_len: int,
    seed: int,
) -> Tuple[str, str]:
    phase_ios: List[Tuple[str, Dict[str, np.ndarray]]] = []

    if summary.mode == "prefill":
        prefill_inputs = wrapper.numpy_inputs(batch_size, seq_len, ctx_len, seed)
        phase_ios.append(("prefill", prefill_inputs))
    elif summary.mode == "decode":
        seed_inputs = wrapper.numpy_inputs(batch_size, 1, ctx_len, seed)
        seed_outputs = _run_session(session, seed_inputs)
        phase_ios.append(("seed", seed_inputs))
        seed_position_ids = seed_inputs.get("position_ids", _build_position_ids(batch_size, 1))
        decode_inputs = wrapper.build_decode_inputs(seed_outputs, seed_position_ids)
        phase_ios.append(("decode", decode_inputs))
    else:
        prefill_inputs = wrapper.numpy_inputs(batch_size, seq_len, ctx_len, seed)
        prefill_outputs = _run_session(session, prefill_inputs)
        phase_ios.append(("prefill", prefill_inputs))
        seed_position_ids = prefill_inputs.get("position_ids", _build_position_ids(batch_size, seq_len))
        decode_inputs = wrapper.build_decode_inputs(prefill_outputs, seed_position_ids)
        phase_ios.append(("decode", decode_inputs))

    return _save_benchmark_io_artifacts(summary, phase_ios)


def benchmark_module_spec(
    qeff_model: "QEFFAutoModelForCausalLM",
    spec: BenchmarkModuleSpec,
    *,
    batch_size: int,
    seq_len: int,
    ctx_len: int,
    num_cores: int = 16,
    num_devices: int = 1,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    export_dir: Optional[str] = None,
    compile_dir: Optional[str] = None,
    mxint8_kv_cache: bool = False,
    seed: int = 13,
    **compiler_options,
) -> BenchmarkSummary:
    adapter = _resolve_adapter(qeff_model)
    model_name = getattr(qeff_model, "benchmark_model_name", qeff_model.model_name)
    model_id = qeff_model.hash_params.get("pretrained_model_name_or_path", model_name)

    benchmark_model = CausalLMModuleBenchmarkModel(
        model=spec.wrapper,
        output_name=spec.output_name,
        model_name=model_name,
        model_id=model_id,
        module_name=spec.module_name,
    )
    onnx_path = benchmark_model.export(
        export_dir=export_dir,
        batch_size=batch_size,
        seq_len=seq_len,
        ctx_len=ctx_len,
        offload_pt_weights=False,
    )
    qpc_path = benchmark_model.compile(
        onnx_path=onnx_path,
        compile_dir=compile_dir,
        batch_size=batch_size,
        seq_len=seq_len,
        ctx_len=ctx_len,
        mode=spec.mode,
        num_cores=num_cores,
        num_devices=num_devices,
        mxint8_kv_cache=mxint8_kv_cache,
        **compiler_options,
    )

    session = QAICInferenceSession(qpc_path)
    prefill_runtime = None
    seed_prefill_ms = None
    first_decode_ms = None
    decode_runtime = None

    if spec.mode == "prefill":
        prefill_runtime = _timed_session_runs(
            session=session,
            build_inputs=lambda: spec.wrapper.numpy_inputs(batch_size, seq_len, ctx_len, seed),
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )
    elif spec.mode == "decode":
        raw_seed_inputs = spec.wrapper.numpy_inputs(batch_size, 1, ctx_len, seed)
        seed_position_ids = raw_seed_inputs.get("position_ids", _build_position_ids(batch_size, 1))
        start = perf_counter()
        seed_outputs = _run_session(session, raw_seed_inputs)
        seed_prefill_ms = (perf_counter() - start) * 1000.0
        first_decode_ms, decode_runtime = _run_decode_benchmark(
            session=session,
            wrapper=spec.wrapper,
            seed_outputs=seed_outputs,
            seed_position_ids=seed_position_ids,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )
    else:
        prefill_runtime, seed_prefill_ms, first_decode_ms, decode_runtime = _run_prefill_and_decode_benchmark(
            session=session,
            wrapper=spec.wrapper,
            batch_size=batch_size,
            seq_len=seq_len,
            ctx_len=ctx_len,
            seed=seed,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
        )

    return BenchmarkSummary(
        benchmark_type=spec.benchmark_type,
        module_name=spec.module_name,
        mode=spec.mode,
        model_name=model_name,
        model_id=model_id,
        architecture=getattr(qeff_model.model.config, "model_type", "unknown"),
        layer_index=spec.layer_index,
        batch_size=batch_size,
        seq_len=seq_len,
        ctx_len=ctx_len,
        resolved_dims=adapter.resolved_dims(qeff_model),
        input_shapes=spec.wrapper.input_shapes(batch_size, seq_len, ctx_len),
        output_shapes=spec.wrapper.output_shapes(batch_size, seq_len, ctx_len, spec.output_name),
        onnx_path=str(onnx_path),
        qpc_path=str(qpc_path),
        prefill_runtime=prefill_runtime,
        seed_prefill_ms=seed_prefill_ms,
        first_decode_ms=first_decode_ms,
        decode_runtime=decode_runtime,
    )


def benchmark_modules(
    qeff_model: "QEFFAutoModelForCausalLM",
    *,
    mode: str = "both",
    benchmark_type: Optional[str] = None,
    batch_size: int = 1,
    seq_len: int = 32,
    ctx_len: int = 128,
    layer_index: int = 0,
    num_cores: int = 16,
    num_devices: int = 1,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    export_dir: Optional[str] = None,
    compile_dir: Optional[str] = None,
    mxint8_kv_cache: bool = False,
    seed: int = 13,
    enable_chunking: bool = False,
    **compiler_options,
) -> List[BenchmarkSummary]:
    adapter = _resolve_adapter(qeff_model)
    concrete_modes = _resolve_benchmark_modes(adapter, mode)
    summaries = []
    for concrete_mode in concrete_modes:
        specs = get_benchmark_module_specs(
            qeff_model,
            mode=concrete_mode,
            layer_index=layer_index,
            seq_len=seq_len,
            ctx_len=ctx_len,
            enable_chunking=enable_chunking,
        )
        for spec in specs:
            if benchmark_type and spec.benchmark_type != benchmark_type:
                continue
            summaries.append(
                benchmark_module_spec(
                    qeff_model,
                    spec,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    ctx_len=ctx_len,
                    num_cores=num_cores,
                    num_devices=num_devices,
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                    export_dir=export_dir,
                    compile_dir=compile_dir,
                    mxint8_kv_cache=mxint8_kv_cache,
                    seed=seed,
                    **compiler_options,
                )
            )
    return summaries


def generate_benchmark_report(
    qeff_model: "QEFFAutoModelForCausalLM",
    *,
    warmup_runs: Optional[int] = None,
    benchmark_runs: Optional[int] = None,
    seed: int = 13,
    device_id: Optional[List[int]] = None,
) -> List[BenchmarkSummary]:
    manifest = getattr(qeff_model, "_benchmark_manifest", None)
    if manifest is None or not manifest.summaries:
        raise TypeError("Please run compile(export_only=False) first in benchmark mode.")

    if any(summary.qpc_path is None for summary in manifest.summaries if not summary.export_error):
        raise TypeError("Benchmark manifest contains ONNX-only exports. Re-run compile with export_only=False.")

    adapter = _resolve_adapter(qeff_model)
    resolved = []
    warmup_runs = manifest.warmup_runs if warmup_runs is None else warmup_runs
    benchmark_runs = manifest.benchmark_runs if benchmark_runs is None else benchmark_runs

    summary_map = {(summary.mode, summary.module_name): summary for summary in manifest.summaries}
    if manifest.prefill_only is True:
        concrete_modes = ("prefill",)
    elif manifest.prefill_only is False:
        concrete_modes = ("decode",)
    else:
        concrete_modes = _resolve_benchmark_modes(adapter, "both")

    for concrete_mode in concrete_modes:
        specs = get_benchmark_module_specs(
            qeff_model,
            mode=concrete_mode,
            layer_index=manifest.summaries[0].layer_index,
            seq_len=manifest.seq_len,
            ctx_len=manifest.ctx_len,
            enable_chunking=manifest.enable_chunking if concrete_mode == "prefill" else False,
            blocking_config=manifest.blocking_config,
        )
        for spec in specs:
            key = (spec.mode, spec.module_name)
            if key not in summary_map:
                continue
            summary = summary_map[key]
            if summary.export_error:
                resolved.append(summary)
                continue
            session = QAICInferenceSession(summary.qpc_path, device_ids=device_id)

            if spec.mode == "prefill":
                prefill_runtime = _timed_session_runs(
                    session=session,
                    build_inputs=lambda w=spec.wrapper: w.numpy_inputs(
                        manifest.batch_size,
                        manifest.seq_len,
                        manifest.ctx_len,
                        seed,
                    ),
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                )
                summary.prefill_runtime = prefill_runtime
            elif spec.mode == "decode":
                raw_seed_inputs = spec.wrapper.numpy_inputs(manifest.batch_size, 1, manifest.ctx_len, seed)
                seed_position_ids = raw_seed_inputs.get("position_ids", _build_position_ids(manifest.batch_size, 1))
                start = perf_counter()
                seed_outputs = _run_session(session, raw_seed_inputs)
                seed_runtime_ms = (perf_counter() - start) * 1000.0
                summary.seed_prefill_ms = seed_runtime_ms
                first_decode_ms, decode_runtime = _run_decode_benchmark(
                    session=session,
                    wrapper=spec.wrapper,
                    seed_outputs=seed_outputs,
                    seed_position_ids=seed_position_ids,
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                )
                summary.first_decode_ms = first_decode_ms
                summary.decode_runtime = decode_runtime
            else:
                (
                    summary.prefill_runtime,
                    summary.seed_prefill_ms,
                    summary.first_decode_ms,
                    summary.decode_runtime,
                ) = _run_prefill_and_decode_benchmark(
                    session=session,
                    wrapper=spec.wrapper,
                    batch_size=manifest.batch_size,
                    seq_len=manifest.seq_len,
                    ctx_len=manifest.ctx_len,
                    seed=seed,
                    warmup_runs=warmup_runs,
                    benchmark_runs=benchmark_runs,
                )
            summary.resolved_dims = adapter.resolved_dims(qeff_model)
            summary.io_dir, summary.io_manifest_path = _collect_phase_io_artifacts(
                summary,
                session,
                spec.wrapper,
                batch_size=manifest.batch_size,
                seq_len=manifest.seq_len,
                ctx_len=manifest.ctx_len,
                seed=seed,
            )
            resolved.append(summary)

    qeff_model._benchmark_manifest.summaries = resolved
    return resolved


def _summary_to_dict(summary: BenchmarkSummary) -> Dict[str, object]:
    result = asdict(summary)
    for key in ("prefill_runtime", "decode_runtime"):
        if result[key] is None:
            continue
        result[key] = {
            metric_name: round(metric_value, 4) if isinstance(metric_value, float) else metric_value
            for metric_name, metric_value in result[key].items()
        }
    for key in ("seed_prefill_ms", "first_decode_ms"):
        if result[key] is not None:
            result[key] = round(result[key], 4)
    return result


def _print_summaries(summaries: List[BenchmarkSummary], as_json: bool) -> None:
    summary_dicts = [_summary_to_dict(summary) for summary in summaries]
    if as_json:
        print(json.dumps(summary_dicts, indent=2))
        return
    print(format_benchmark_table(summaries))


def format_benchmark_table(summaries: List[BenchmarkSummary]) -> str:
    headers = ["Mode", "Module", "Type", "Prefill ms", "Seed ms", "Decode ms"]
    rows = []
    for summary in summaries:
        if summary.export_error:
            rows.append([summary.mode, summary.module_name, summary.benchmark_type, "EXPORT_FAILED", "-", "-"])
        else:
            prefill_ms = f"{summary.prefill_runtime.mean_ms:.4f}" if summary.prefill_runtime else "-"
            seed_ms = f"{summary.seed_prefill_ms:.4f}" if summary.seed_prefill_ms is not None else "-"
            decode_ms = f"{summary.decode_runtime.mean_ms:.4f}" if summary.decode_runtime else "-"
            rows.append([summary.mode, summary.module_name, summary.benchmark_type, prefill_ms, seed_ms, decode_ms])

    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def render_row(values):
        return " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(values))

    divider = "-+-".join("-" * width for width in widths)
    lines = [render_row(headers), divider]
    lines.extend(render_row(row) for row in rows)

    for summary in summaries:
        if summary.export_error:
            lines.append("")
            lines.append(f"{summary.mode} | {summary.module_name} | {summary.benchmark_type}")
            lines.append(f"  EXPORT FAILED: {summary.export_error}")
            continue
        lines.append("")
        lines.append(f"{summary.mode} | {summary.module_name} | {summary.benchmark_type}")
        lines.append(f"inputs:  {json.dumps(summary.input_shapes, sort_keys=True)}")
        lines.append(f"outputs: {json.dumps(summary.output_shapes, sort_keys=True)}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Micro-benchmark isolated QEff causal LM modules on QAIC hardware.")
    parser.add_argument("--model", required=True, help="HF model id or a known tiny-model alias such as `llama`.")
    parser.add_argument("--benchmark-type", choices=BENCHMARK_TYPES)
    parser.add_argument("--mode", default="both", choices=BENCHMARK_MODES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    parser.add_argument("--export-dir")
    parser.add_argument("--compile-dir")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--enable-chunking", action="store_true")
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    parser.add_argument("--mos", type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--action",
        default="benchmark",
        choices=("list", "export", "benchmark"),
        help="List module inventory, export all selected module wrappers, or compile and benchmark them.",
    )
    return parser


def run_benchmark(
    *,
    model_name_or_path: str,
    benchmark_type: Optional[str] = None,
    mode: str = "both",
    batch_size: int = 1,
    seq_len: int = 32,
    ctx_len: int = 128,
    layer_index: int = 0,
    num_cores: int = 16,
    num_devices: int = 1,
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
    export_dir: Optional[str] = None,
    compile_dir: Optional[str] = None,
    mxint8_kv_cache: bool = False,
    seed: int = 13,
    enable_chunking: bool = False,
    action: str = "benchmark",
    **compiler_options,
) -> List[BenchmarkSummary]:
    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

    model_name, model_id = resolve_model_id(model_name_or_path)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, enable_benchmark=True)
    qeff_model.benchmark_model_name = model_name

    if action == "list":
        summaries = []
        adapter = _resolve_adapter(qeff_model)
        concrete_modes = _resolve_benchmark_modes(adapter, mode)
        for concrete_mode in concrete_modes:
            specs = get_benchmark_module_specs(
                qeff_model,
                mode=concrete_mode,
                layer_index=layer_index,
                seq_len=seq_len,
                ctx_len=ctx_len,
                enable_chunking=enable_chunking,
            )
            for spec in specs:
                if benchmark_type and spec.benchmark_type != benchmark_type:
                    continue
                summaries.append(
                    BenchmarkSummary(
                        benchmark_type=spec.benchmark_type,
                        module_name=spec.module_name,
                        mode=spec.mode,
                        model_name=model_name,
                        model_id=model_id,
                        architecture=getattr(qeff_model.model.config, "model_type", "unknown"),
                        layer_index=spec.layer_index,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        ctx_len=ctx_len,
                        resolved_dims=adapter.resolved_dims(qeff_model),
                        input_shapes=spec.wrapper.input_shapes(batch_size, seq_len, ctx_len),
                        output_shapes=spec.wrapper.output_shapes(batch_size, seq_len, ctx_len, spec.output_name),
                        onnx_path="",
                        qpc_path=None,
                        prefill_runtime=None,
                        seed_prefill_ms=None,
                        first_decode_ms=None,
                        decode_runtime=None,
                    )
                )
        return summaries

    if action == "export":
        return export_benchmark_modules(
            qeff_model,
            mode=mode,
            benchmark_type=benchmark_type,
            batch_size=batch_size,
            seq_len=seq_len,
            ctx_len=ctx_len,
            layer_index=layer_index,
            export_dir=export_dir,
            enable_chunking=enable_chunking,
        )

    return benchmark_modules(
        qeff_model,
        mode=mode,
        benchmark_type=benchmark_type,
        batch_size=batch_size,
        seq_len=seq_len,
        ctx_len=ctx_len,
        layer_index=layer_index,
        num_cores=num_cores,
        num_devices=num_devices,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        export_dir=export_dir,
        compile_dir=compile_dir,
        mxint8_kv_cache=mxint8_kv_cache,
        seed=seed,
        enable_chunking=enable_chunking,
        **compiler_options,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    compiler_options = {}
    if args.aic_enable_depth_first:
        compiler_options["aic_enable_depth_first"] = True
    if args.mos is not None:
        compiler_options["mos"] = args.mos

    summaries = run_benchmark(
        model_name_or_path=args.model,
        benchmark_type=args.benchmark_type,
        mode=args.mode,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        ctx_len=args.ctx_len,
        layer_index=args.layer_index,
        num_cores=args.num_cores,
        num_devices=args.num_devices,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        export_dir=args.export_dir,
        compile_dir=args.compile_dir,
        mxint8_kv_cache=args.mxint8_kv_cache,
        seed=args.seed,
        enable_chunking=args.enable_chunking,
        action=args.action,
        **compiler_options,
    )
    _print_summaries(summaries, as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
