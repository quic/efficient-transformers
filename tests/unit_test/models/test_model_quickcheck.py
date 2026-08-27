# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Fast CPU regression coverage across the main model families supported by QEfficient.

This file intentionally uses two coverage tiers:

1. Runtime parity:
   - Exact token or tensor parity across HF PyTorch, transformed PyTorch, and ORT
   - Used where the repo already has a stable CPU verification path
2. Export smoke:
   - Used for model families or architectures that are supported by export today,
     but do not yet have a stable CPU runtime parity path in the consolidated test
"""

import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    LlamaConfig,
    Qwen2Config,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)

from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForImageTextToText,
    QEFFAutoModelForSequenceClassification,
    QEFFAutoModelForSpeechSeq2Seq,
)
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils._utils import _infer_specialization_name, to_named_specializations
from QEfficient.utils.run_utils import ApiRunner

ort.set_default_logger_severity(3)
logging.getLogger("QEfficient").setLevel(logging.ERROR)
logging.getLogger("QEfficient.base.modeling_qeff").setLevel(logging.ERROR)


CAUSAL_RUNTIME_MODEL_IDS = {
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

#   In PyTorch ≤2.3 (used with transformers v4.57.3), torch.onnx.export with
#   export_modules_as_functions created one ONNX function definition per module instance — so a Mixtral
#   model with 2 decoder layers produced 2 separate QeffMixtralDecoderLayer function definitions in the
#   ONNX.
#   In PyTorch 2.7 (used with transformers v5.5.4), the same export creates one shared function
#   definition per module class, called once per instance. So 2 decoder layers → 1 function definition
#   called 2 times.
CAUSAL_MULTI_SUBFUNCTION_MODEL_TYPES = {
    "codegen",
    "phi",
    "starcoder2",
    "gpt_oss",
    # "granitemoe" is intentionally not listed in CAUSAL_RUNTIME_MODEL_IDS yet.
}

VLM_TEXT_RUNTIME_MODEL_ID = "tiny-random/gemma-3"
VLM_EXPORT_MODEL_IDS = {
    "gemma3": "tiny-random/gemma-3",
    "qwen2_5_vl": "optimum-intel-internal-testing/tiny-random-qwen2.5-vl",
    "internvl2": "optimum-intel-internal-testing/tiny-random-internvl2",
}
TINY_TEXT_EMBEDDING_MODEL_ID = "hf-internal-testing/tiny-random-BertModel"
TINY_AUDIO_CTC_MODEL_ID = "hf-internal-testing/tiny-random-wav2vec2"
TINY_WHISPER_MODEL_ID = "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
TINY_SEQ_CLASSIFICATION_MODEL_ID = "ydshieh/tiny-random-BertForSequenceClassification"
TINY_AWQ_MODEL_ID = "optimum-intel-internal-testing/tiny-mixtral-AWQ-4bit"

QWEN_QUICKCHECK_MODEL_TYPES = (
    "qwen3",
    "qwen3_5",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5_moe",
)
QWEN_MOE_QUICKCHECK_MODEL_TYPES = ("qwen3_moe", "qwen3_vl_moe", "qwen3_5_moe")
QWEN_EXPECTED_DECODER_LAYERS = {
    "qwen3": "QEffQwen3DecoderLayer",
    "qwen3_5": "QEffQwen3_5DecoderLayer",
    "qwen3_moe": "QEffQwen3MoeDecoderLayer",
    "qwen3_vl": "QEffQwen3VLTextDecoderLayer",
    "qwen3_vl_moe": "QEffQwen3VLMoeTextDecoderLayer",
    "qwen3_5_moe": "QEffQwen3_5MoeDecoderLayer",
}

TINY_MOE_PREFILL_SUBFUNCTION_CONFIGS = {
    "glm4_moe": dict(
        max_position_embeddings=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        vocab_size=127,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        n_group=1,
        topk_group=1,
        head_dim=16,
    ),
    "qwen3_moe": dict(
        max_position_embeddings=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        vocab_size=127,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
    ),
    "gpt_oss": dict(
        max_position_embeddings=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=32,
        intermediate_size=32,
        vocab_size=127,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        head_dim=16,
        sliding_window=128,
        rope_scaling=None,
    ),
}

MODEL_KWARGS = {"attn_implementation": "eager"}
PREFIX_CACHING_MODEL_ID = "hf-internal-testing/tiny-random-GPT2LMHeadModel"


def _per_test_thread_budget() -> int:
    override = os.environ.get("QEFF_NUM_THREADS")
    if override:
        return max(1, int(override))
    total = os.cpu_count() or 1
    workers = max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    return max(1, total // workers)


def _configure_torch_threads() -> None:
    threads = _per_test_thread_budget()
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads)))


def _ort_session(onnx_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    threads = _per_test_thread_budget()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    onnx_model = onnx.load(onnx_path, load_external_data=True)
    int32_max = torch.iinfo(torch.int32).max
    graph_was_patched = False
    added_initializers = {}

    def _zero_int32_max_tensor_attr(attr) -> bool:
        if attr.type != onnx.AttributeProto.TENSOR:
            return False
        np_tensor = onnx.numpy_helper.to_array(attr.t, os.path.dirname(onnx_path))
        if np.issubdtype(np_tensor.dtype, np.integer) and np.any(np_tensor == int32_max):
            patched_tensor = np.where(np_tensor == int32_max, np.array(0, dtype=np_tensor.dtype), np_tensor)
            attr.t.CopyFrom(onnx.numpy_helper.from_array(patched_tensor.astype(np_tensor.dtype, copy=False)))
            return True
        return False

    def _patch_constant_nodes(nodes, *, allow_initializers: bool = False) -> bool:
        patched = False
        for node in nodes:
            if node.op_type in {"Constant", "ConstantOfShape"}:
                for attr in node.attribute:
                    if _zero_int32_max_tensor_attr(attr):
                        patched = True
                        if allow_initializers and node.op_type == "Constant":
                            np_tensor = onnx.numpy_helper.to_array(attr.t, os.path.dirname(onnx_path))
                            added_initializers[node.output[0]] = ort.OrtValue.ortvalue_from_numpy(np_tensor)
        return patched

    def _patch_ort_index_inputs(nodes) -> bool:
        patched = False
        for index, node in list(enumerate(nodes)):
            if node.op_type not in {"GatherND", "ScatterND"} or len(node.input) < 2:
                continue
            indices_name = node.input[1]
            cast_output = f"{indices_name}__ort_int64"
            cast_node = onnx.helper.make_node(
                "Cast",
                inputs=[indices_name],
                outputs=[cast_output],
                name=f"{node.name or node.op_type}_IndicesCastToInt64",
                to=onnx.TensorProto.INT64,
            )
            nodes.insert(index, cast_node)
            node.input[1] = cast_output
            patched = True
        return patched

    graph_was_patched |= _patch_constant_nodes(onnx_model.graph.node, allow_initializers=True)
    graph_was_patched |= _patch_ort_index_inputs(onnx_model.graph.node)
    for function_proto in onnx_model.functions:
        graph_was_patched |= _patch_constant_nodes(function_proto.node)
        graph_was_patched |= _patch_ort_index_inputs(function_proto.node)

    for name, value in added_initializers.items():
        options.add_initializer(name, value)
    if graph_was_patched:
        return ort.InferenceSession(onnx_model.SerializeToString(), sess_options=options)
    return ort.InferenceSession(str(onnx_path), sess_options=options)


def _ort_session_zeroing_int32_max_constants(onnx_path: Path):
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    options = ort.SessionOptions()
    added_initializers = {}
    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(onnx_path))
            if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                added_initializers[node.output[0]] = ort.OrtValue.ortvalue_from_numpy(np.array(0, np_tensor.dtype))
    for name, value in added_initializers.items():
        options.add_initializer(name, value)
    return ort.InferenceSession(str(onnx_path), sess_options=options), added_initializers


_configure_torch_threads()


def _cleanup_stale_tmp_exports() -> None:
    tmp_root = Path(tempfile.gettempdir())
    for pattern in ("qeff_*", "*qeff*", "*onnx*", "*qnn*"):
        for path in tmp_root.glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.is_file():
                    path.unlink(missing_ok=True)
            except OSError:
                # Best-effort cleanup only.
                pass


@pytest.fixture(scope="session", autouse=True)
def _clean_tmp_exports_before_quickcheck():
    # Avoid concurrent cleanup from all xdist workers.
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker not in (None, "gw0"):
        return
    _cleanup_stale_tmp_exports()


@contextmanager
def _suppress_native_output():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _exported_onnx_path(export_result) -> Path:
    if isinstance(export_result, (list, tuple)):
        export_result = export_result[-1]
    onnx_path = Path(export_result)
    assert onnx_path.is_file()
    return onnx_path


def _count_decoder_block_subfunctions(onnx_model, qeff_model) -> int:
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    if not callable(get_submodules):
        return 0

    submodules = get_submodules()
    if not submodules:
        return 0

    if not isinstance(submodules, (set, list, tuple)):
        submodules = [submodules]

    block_names = {module.__name__ for module in submodules if hasattr(module, "__name__")}
    return sum(any(block_name in func.name for block_name in block_names) for func in onnx_model.functions)


def _decoder_block_subfunction_names(onnx_model, qeff_model) -> Set[str]:
    get_submodules = getattr(qeff_model.model, "get_submodules_for_export", None)
    assert callable(get_submodules)

    submodules = get_submodules()
    assert submodules

    if not isinstance(submodules, (set, list, tuple)):
        submodules = [submodules]

    block_names = {module.__name__ for module in submodules if hasattr(module, "__name__")}
    assert block_names
    return {func.name for func in onnx_model.functions if any(block_name in func.name for block_name in block_names)}


def _function_op_types(onnx_model, function_names: Set[str]) -> Set[str]:
    return {
        node.op_type
        for function_proto in onnx_model.functions
        if function_proto.name in function_names
        for node in function_proto.node
    }


def _assert_has_retained_state_outputs(onnx_path: Path) -> None:
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    retained_outputs = [output.name for output in onnx_model.graph.output if output.name.endswith("_RetainedState")]
    assert retained_outputs


def _run_embedding_ort(onnx_path: Path, inputs: Dict[str, torch.Tensor]) -> np.ndarray:
    session = _ort_session(onnx_path)
    input_names = {item.name for item in session.get_inputs()}
    ort_inputs = {name: tensor.detach().numpy() for name, tensor in inputs.items() if name in input_names}
    return session.run(None, ort_inputs)[0]


def _run_whisper_export_smoke(qeff_model: QEFFAutoModelForSpeechSeq2Seq, out_dir: Path) -> Path:
    onnx_path = _exported_onnx_path(qeff_model.export(out_dir))
    _assert_has_retained_state_outputs(onnx_path)
    return onnx_path


def _assert_proxy_only_onnx_transform_policy(
    qeff_model, enable_proxy: bool, always_on_transforms: Optional[Set[str]] = None
) -> None:
    pytorch_transform_names = {transform.__name__ for transform in qeff_model._pytorch_transforms}
    transform_names = {transform.__name__ for transform in qeff_model._onnx_transforms}
    proxy_pytorch_transform = "QeffProxyModuleTransform"
    proxy_only_transforms = {"FP16ClipTransform", "SplitTensorsTransform"}
    always_on_transforms = always_on_transforms or set()
    conditional_proxy_transforms = proxy_only_transforms - always_on_transforms

    if enable_proxy:
        assert proxy_pytorch_transform in pytorch_transform_names
        assert proxy_only_transforms.issubset(transform_names)
    else:
        assert proxy_pytorch_transform not in pytorch_transform_names
        assert conditional_proxy_transforms.isdisjoint(transform_names)
        assert always_on_transforms.issubset(transform_names)


def _assert_dual_qpc_vlm_proxy_transform_policy(qeff_model, enable_proxy: bool) -> None:
    _assert_proxy_only_onnx_transform_policy(qeff_model.vision_model, enable_proxy=enable_proxy)
    _assert_proxy_only_onnx_transform_policy(qeff_model.lang_model, enable_proxy=enable_proxy)


def _skip_on_model_fetch_error(exc: Exception, model_id: str) -> None:
    pytest.skip(
        f"Skipping {model_id}: model unavailable or unsupported in this environment ({type(exc).__name__}: {exc})"
    )


def _export_vlm_with_text_fallback(model_id: str, out_dir: Path) -> Path:
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        use_text_only_first = model_type in {"qwen2_5_vl", "qwen2_5_vl_text", "internvl_chat"}

        if not use_text_only_first:
            try:
                vlm_model = QEFFAutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True)
                return _exported_onnx_path(vlm_model.export(out_dir / "full-vlm"))
            except Exception:
                pass

        try:
            if model_type in {"qwen2_5_vl", "qwen2_5_vl_text"} and getattr(config, "text_config", None) is not None:
                qwen2_cfg_dict = config.text_config.to_dict()
                qwen2_cfg_dict["model_type"] = "qwen2"
                qwen2_allowed_keys = set(Qwen2Config().to_dict().keys())
                qwen2_cfg = Qwen2Config(**{k: v for k, v in qwen2_cfg_dict.items() if k in qwen2_allowed_keys})
                text_model = AutoModelForCausalLM.from_config(qwen2_cfg, trust_remote_code=True, **MODEL_KWARGS)
                text_model = text_model.to(torch.float32)
                text_model.eval()
                qeff_text_model = QEFFAutoModelForCausalLM(text_model)
                return _exported_onnx_path(qeff_text_model.export(out_dir / "text-fallback"))

            text_configs = [getattr(config, "text_config", None), getattr(config, "llm_config", None)]
            for text_config in text_configs:
                if text_config is None:
                    continue
                try:
                    text_model = AutoModelForCausalLM.from_config(
                        text_config,
                        trust_remote_code=True,
                        **MODEL_KWARGS,
                    )
                    text_model = text_model.to(torch.float32)
                    text_model.eval()
                    qeff_text_model = QEFFAutoModelForCausalLM(text_model)
                    return _exported_onnx_path(qeff_text_model.export(out_dir / "text-fallback"))
                except Exception:
                    continue
            raise RuntimeError(f"No text fallback config path available for {model_id}")
        except Exception as text_exc:
            _skip_on_model_fetch_error(text_exc, model_id)
    except Exception as cfg_exc:
        _skip_on_model_fetch_error(cfg_exc, model_id)


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        dtype="float32",
    )


def _tiny_qwen3_moe_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=256,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        mlp_only_layers=[],
        dtype="float32",
    )


def _tiny_qwen3_5_text_config(*, moe: bool = False) -> Qwen3_5TextConfig | Qwen3_5MoeTextConfig:
    kwargs = dict(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["full_attention", "linear_attention"],
        dtype="float32",
    )
    if moe:
        kwargs.update(
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=2,
            num_experts_per_tok=1,
        )
        return Qwen3_5MoeTextConfig(**kwargs)

    kwargs.update(
        intermediate_size=32,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
    )
    return Qwen3_5TextConfig(**kwargs)


def _tiny_qwen_vision_config(config_cls, *, deepstack: bool = False):
    kwargs = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=4,
        temporal_patch_size=1,
        spatial_merge_size=1,
        out_hidden_size=16,
        num_position_embeddings=64,
        dtype="float32",
    )
    if deepstack:
        kwargs["deepstack_visual_indexes"] = []
    return config_cls(**kwargs)


def _tiny_qwen3_vl_config() -> Qwen3VLConfig:
    text_config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_scaling={"rope_type": "default", "mrope_section": [11, 11, 10]},
        dtype="float32",
    )
    return Qwen3VLConfig(
        text_config=text_config,
        vision_config=_tiny_qwen_vision_config(Qwen3VLVisionConfig, deepstack=True),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _tiny_qwen3_vl_moe_config() -> Qwen3VLMoeConfig:
    text_config = Qwen3VLMoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_scaling={"rope_type": "default", "mrope_section": [11, 11, 10]},
        dtype="float32",
    )
    return Qwen3VLMoeConfig(
        text_config=text_config,
        vision_config=_tiny_qwen_vision_config(Qwen3VLMoeVisionConfig, deepstack=True),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _tiny_qwen3_5_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config=_tiny_qwen3_5_text_config(),
        vision_config=_tiny_qwen_vision_config(Qwen3_5VisionConfig),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _tiny_qwen3_5_moe_config() -> Qwen3_5MoeConfig:
    return Qwen3_5MoeConfig(
        text_config=_tiny_qwen3_5_text_config(moe=True),
        vision_config=_tiny_qwen_vision_config(Qwen3_5MoeVisionConfig),
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
    )


def _tiny_qwen_config(model_type: str):
    config_builders = {
        "qwen3": _tiny_qwen3_config,
        "qwen3_5": _tiny_qwen3_5_config,
        "qwen3_moe": _tiny_qwen3_moe_config,
        "qwen3_vl": _tiny_qwen3_vl_config,
        "qwen3_vl_moe": _tiny_qwen3_vl_moe_config,
        "qwen3_5_moe": _tiny_qwen3_5_moe_config,
    }
    return config_builders[model_type]()


def _tiny_qwen_qeff_model(model_type: str):
    config = _tiny_qwen_config(model_type)
    torch.manual_seed(0)
    if model_type in {"qwen3", "qwen3_moe"}:
        hf_model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS).eval()
        return QEFFAutoModelForCausalLM(hf_model, continuous_batching=False)

    hf_model = AutoModelForImageTextToText.from_config(config).eval()
    return QEFFAutoModelForImageTextToText(hf_model, kv_offload=True)


def _qwen_decoder_qeff_model(qeff_model):
    return qeff_model if isinstance(qeff_model, QEFFAutoModelForCausalLM) else qeff_model.lang_model


def _qwen_decoder_export_model(qeff_model):
    return _qwen_decoder_qeff_model(qeff_model).model


def _qwen_decoder_subfunction_names(qeff_model) -> Set[str]:
    return {module_cls.__name__ for module_cls in _qwen_decoder_export_model(qeff_model).get_submodules_for_export()}


def _tiny_qwen_hf_model(model_type: str):
    config = _tiny_qwen_config(model_type)
    torch.manual_seed(0)
    if model_type in {"qwen3", "qwen3_moe"}:
        return AutoModelForCausalLM.from_config(config, **MODEL_KWARGS).eval()

    return AutoModelForImageTextToText.from_config(config).eval()


def _tiny_qwen_checkpoint_path(model_type: str, tmp_path) -> Path:
    checkpoint_dir = tmp_path / f"{model_type}-tiny-checkpoint"
    if checkpoint_dir.is_dir():
        return checkpoint_dir

    hf_model = _tiny_qwen_hf_model(model_type)
    assert _tiny_qwen_text_config(model_type).num_hidden_layers == 2
    hf_model.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def _tiny_qwen_text_config(model_type: str):
    config = _tiny_qwen_config(model_type)
    return getattr(config, "text_config", config)


def _qwen_causal_inputs(config, *, seq_len: int = 4, ctx_len: int = 8):
    input_ids = (torch.arange(seq_len, dtype=torch.int64).view(1, seq_len) % (config.vocab_size - 1)) + 1
    position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    past_key_values = tuple(
        (
            torch.zeros((1, config.num_key_value_heads, ctx_len, head_dim), dtype=torch.float32),
            torch.zeros((1, config.num_key_value_heads, ctx_len, head_dim), dtype=torch.float32),
        )
        for _ in range(config.num_hidden_layers)
    )
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
    }


def _qwen_vlm_lang_inputs(qeff_model, *, prefill_seq_len: int = 4):
    try:
        inputs = qeff_model.model.get_dummy_inputs(kv_offload=True, prefill_seq_len=prefill_seq_len)["lang"]
    except TypeError:
        inputs = qeff_model.model.get_dummy_inputs(kv_offload=True)["lang"]
    inputs = deepcopy(inputs)
    if "input_ids" in inputs:
        inputs["input_ids"].fill_(1)
    return inputs


def _qwen_export_io(qeff_model, *, prefill_seq_len: int = 4):
    if isinstance(qeff_model, QEFFAutoModelForCausalLM):
        return None, None, None

    inputs = qeff_model.model.get_dummy_inputs(kv_offload=True, prefill_seq_len=prefill_seq_len)
    dynamic_axes = qeff_model.model.get_onnx_dynamic_axes(kv_offload=True)
    output_names = qeff_model.model.get_output_names(kv_offload=True)
    lang_inputs = deepcopy(inputs["lang"])
    if "input_ids" in lang_inputs:
        lang_inputs["input_ids"].fill_(1)
    return lang_inputs, output_names["lang"], dynamic_axes["lang"]


def _extract_qwen_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits.detach().float().numpy()
    if isinstance(outputs, (tuple, list)):
        return outputs[0].detach().float().numpy()
    return outputs.detach().float().numpy()


def _as_hf_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    return tuple(tuple(tensor.detach().clone() for tensor in layer) for layer in past_key_values)


def _hf_qwen_vlm_text_logits(hf_model, inputs):
    language_model = (
        hf_model.model.language_model if hasattr(hf_model.model, "language_model") else hf_model.language_model
    )
    outputs = language_model(
        input_ids=inputs["input_ids"],
        position_ids=inputs["position_ids"],
        use_cache=True,
    )
    position_ids = inputs["position_ids"]
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    logit_index = text_position_ids.to(torch.int32).argmax(1, keepdim=True)
    hidden_states = outputs.last_hidden_state[torch.arange(text_position_ids.shape[0]).view(-1, 1), logit_index]
    return hf_model.lm_head(hidden_states).detach().float().numpy()


def _hf_qwen_logits(model_type: str, hf_model, inputs):
    with torch.no_grad():
        if model_type in {"qwen3", "qwen3_moe"}:
            hf_inputs = {
                "input_ids": inputs["input_ids"],
                "position_ids": inputs["position_ids"],
                "use_cache": True,
            }
            logits = _extract_qwen_logits(hf_model(**hf_inputs))
            logit_index = inputs["position_ids"].to(torch.int32).argmax(1, keepdim=True).detach().cpu().numpy()
            batch_index = np.arange(inputs["position_ids"].shape[0]).reshape(-1, 1)
            return logits[batch_index, logit_index]
        return _hf_qwen_vlm_text_logits(hf_model, inputs)


def _qeff_qwen_logits(qeff_model, inputs):
    with torch.no_grad():
        return _extract_qwen_logits(_qwen_decoder_qeff_model(qeff_model).model(**inputs))


def _flatten_qwen_ort_inputs(inputs, session_inputs, qeff_export_model):
    flat_inputs = {name: value.detach().numpy() for name, value in inputs.items() if torch.is_tensor(value)}
    past_key_values = inputs.get("past_key_values")
    if past_key_values is not None:
        for layer_idx, layer_state in enumerate(past_key_values):
            if hasattr(qeff_export_model, "get_onnx_past_key_value_names"):
                names = qeff_export_model.get_onnx_past_key_value_names(layer_idx, layer_state)
            elif len(layer_state) == 2:
                names = [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
            else:
                names = [f"past_state.{layer_idx}.{state_idx}" for state_idx in range(len(layer_state))]
            for name, tensor in zip(names, layer_state):
                flat_inputs[name] = tensor.detach().numpy()
    return {name: flat_inputs[name] for name in session_inputs if name in flat_inputs}


def _ort_qwen_logits(onnx_path: Path, qeff_model, inputs):
    session = _ort_session(onnx_path)
    session_inputs = [item.name for item in session.get_inputs()]
    session_outputs = [item.name for item in session.get_outputs()]
    ort_inputs = _flatten_qwen_ort_inputs(inputs, session_inputs, _qwen_decoder_export_model(qeff_model))
    outputs = dict(zip(session_outputs, session.run(session_outputs, ort_inputs)))
    return outputs["logits"].astype(np.float32)


def _kimi_k25_prompt_inputs(processor):
    from PIL import Image

    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": image},
                {"type": "text", "text": "Describe."},
            ],
        },
    ]
    inputs = processor(
        messages=messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors="pt",
    )
    return {name: (value.to("cpu") if torch.is_tensor(value) else value) for name, value in inputs.items()}


def _kimi_k25_qeff_lang_inputs(qeff_model, inputs, vision_embeds):
    seq_len = inputs["input_ids"].shape[1]
    lang_inputs = deepcopy(qeff_model.model.get_dummy_inputs(kv_offload=True, prefill_seq_len=seq_len)["lang"])
    lang_inputs["input_ids"] = inputs["input_ids"].clone()
    lang_inputs["position_ids"] = torch.where(
        inputs["attention_mask"] > 0,
        torch.arange(seq_len, dtype=torch.int64).view(1, seq_len),
        -1,
    )
    lang_inputs["vision_embeds"] = vision_embeds
    lang_inputs["image_idx"] = torch.zeros((1, 1), dtype=torch.int64)
    return lang_inputs


class _KimiSyntheticQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        if linear.in_features % 2 != 0:
            raise ValueError("Synthetic int4 fixture requires an even input dimension")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = 4
        self.group_size = 1
        self.act_order = None
        self.register_buffer(
            "qweight", torch.full((linear.out_features, linear.in_features // 2), 0x99, dtype=torch.uint8)
        )
        self.register_buffer(
            "qzeros", torch.full((linear.out_features, linear.in_features // 2), 0x88, dtype=torch.uint8)
        )
        self.register_buffer("scales", linear.weight.detach().clone().to(torch.float32))
        self.register_buffer("g_idx", torch.arange(linear.in_features, dtype=torch.int32))
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone().to(torch.float32))

    def forward(self, inputs):
        output = torch.matmul(inputs.float(), self.scales.transpose(0, 1).float())
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output.to(inputs.dtype)


def _install_kimi_synthetic_quant_experts(model):
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        for expert in experts:
            for projection_name in ("gate_proj", "up_proj", "down_proj"):
                projection = getattr(expert, projection_name, None)
                if isinstance(projection, nn.Linear):
                    setattr(expert, projection_name, _KimiSyntheticQuantLinear(projection))


@pytest.mark.llm_model
def test_kimi_k25_quickcheck_hf_qeff_vision_logits_parity():
    from tests.utils.load_kimi_utils import get_kimi_k25_test_config, load_kimi_k25_model_from_config

    model_id = "moonshotai/Kimi-K2.5"
    config_path = Path(__file__).parents[2] / "configs" / "image_text_model_configs.json"
    model_configs = json.loads(config_path.read_text())["image_text_models"]
    model_config_dict = {model["model_name"]: model for model in model_configs}

    try:
        config = get_kimi_k25_test_config(model_id, model_config_dict)
        model_hf, _, processor = load_kimi_k25_model_from_config(config)
        _install_kimi_synthetic_quant_experts(model_hf)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    inputs = _kimi_k25_prompt_inputs(processor)
    with torch.no_grad():
        hf_outputs = model_hf(**inputs, use_cache=False, return_dict=True)
        hf_logits = hf_outputs.logits[:, -1:, :].detach().float().numpy()

    qeff_model = QEFFAutoModelForImageTextToText(
        deepcopy(model_hf),
        kv_offload=True,
        config=model_hf.config,
        torch_dtype=torch.float32,
    )
    grid_thws = inputs["grid_thws"].to(torch.int64)
    h_shape = torch.ones((int(grid_thws[0, 1].item()),), dtype=torch.int64)
    w_shape = torch.ones((int(grid_thws[0, 2].item()),), dtype=torch.int64)

    with torch.no_grad():
        vision_embeds = qeff_model.vision_model.model(inputs["pixel_values"], h_shape, w_shape)
        lang_inputs = _kimi_k25_qeff_lang_inputs(qeff_model, inputs, vision_embeds)
        qeff_logits = qeff_model.lang_model.model(**lang_inputs)[0].detach().float().numpy()

    assert qeff_logits.shape == hf_logits.shape
    assert np.allclose(hf_logits, qeff_logits, atol=1e-4, rtol=1e-4)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_lm_cpu_runtime_parity_with_api_runner(model_type, model_id, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    prompt = ["hello world"]
    prompt_len = 8
    ctx_len = 12

    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id,
        **MODEL_KWARGS,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model_hf.eval()

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompt,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0))
    assert np.array_equal(kv_tokens, ort_tokens)


@pytest.mark.llm_model
def test_vlm_text_side_runtime_parity_and_full_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(VLM_TEXT_RUNTIME_MODEL_ID, trust_remote_code=True)
    config = AutoConfig.from_pretrained(VLM_TEXT_RUNTIME_MODEL_ID, trust_remote_code=True)
    text_config = config.text_config

    text_model = AutoModelForCausalLM.from_config(text_config, trust_remote_code=True, **MODEL_KWARGS)
    text_model.eval()

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=text_model.config,
        prompt=["hello world"],
        prompt_len=4,
        ctx_len=8,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(text_model)
    qeff_text_model = QEFFAutoModelForCausalLM(text_model)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_text_model.model)
    onnx_path = _exported_onnx_path(qeff_text_model.export(tmp_path / "vlm-text"))
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0))
    assert np.array_equal(kv_tokens, ort_tokens)

    vlm_model = QEFFAutoModelForImageTextToText.from_pretrained(VLM_TEXT_RUNTIME_MODEL_ID, trust_remote_code=True)
    vlm_onnx_path = _exported_onnx_path(vlm_model.export(tmp_path / "vlm-full"))
    assert vlm_onnx_path.name.endswith(".onnx")


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("vlm_name", "model_id"),
    sorted(VLM_EXPORT_MODEL_IDS.items()),
    ids=sorted(VLM_EXPORT_MODEL_IDS),
)
def test_vlm_export_smoke_additional_models(vlm_name, model_id, tmp_path):
    vlm_onnx_path = _export_vlm_with_text_fallback(model_id, tmp_path / f"vlm-{vlm_name}")
    assert vlm_onnx_path.name.endswith(".onnx")


@pytest.mark.llm_model
def test_text_embedding_cpu_parity_and_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID)
    model_hf = AutoModel.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID, **MODEL_KWARGS)
    model_hf.eval()

    inputs = tokenizer("hello world", return_tensors="pt")
    hf_outputs = model_hf(**inputs).last_hidden_state.detach().numpy()

    qeff_model = QEFFAutoModel(model_hf)
    qeff_outputs = qeff_model.generate(inputs=inputs, runtime_ai100=False).last_hidden_state.detach().numpy()
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_outputs = _run_embedding_ort(onnx_path, inputs)

    assert np.allclose(hf_outputs, qeff_outputs, atol=1e-5)
    assert np.allclose(hf_outputs, ort_outputs, atol=1e-5)


@pytest.mark.llm_model
def test_text_embedding_fp16_clip_transform_and_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID)
    qeff_model = QEFFAutoModel.from_pretrained(TINY_TEXT_EMBEDDING_MODEL_ID)
    transform_names = {transform.__name__ for transform in qeff_model._onnx_transforms}

    assert "FP16ClipTransform" in transform_names
    assert "SplitTensorsTransform" in transform_names

    inputs = tokenizer("hello world", return_tensors="pt")
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "embedding-ai100"))
    ort_outputs = _run_embedding_ort(onnx_path, inputs)
    assert ort_outputs.shape[0] == inputs["input_ids"].shape[0]
    assert ort_outputs.shape[1] == inputs["input_ids"].shape[1]


@pytest.mark.llm_model
def test_audio_embedding_ctc_cpu_parity_and_export(tmp_path):
    processor = AutoTokenizer.from_pretrained(TINY_AUDIO_CTC_MODEL_ID)
    del processor
    replace_transformers_quantizers()
    model_hf = AutoModelForCTC.from_pretrained(TINY_AUDIO_CTC_MODEL_ID, **MODEL_KWARGS, low_cpu_mem_usage=False)
    model_hf.eval()

    from transformers import AutoProcessor

    audio_processor = AutoProcessor.from_pretrained(TINY_AUDIO_CTC_MODEL_ID)
    input_values = audio_processor(
        np.zeros(400, dtype=np.float32), return_tensors="pt", sampling_rate=16000
    ).input_values

    hf_logits = model_hf(input_values=input_values).logits.detach().numpy()
    qeff_model = QEFFAutoModelForCTC(model_hf, pretrained_model_name_or_path=TINY_AUDIO_CTC_MODEL_ID)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_session = _ort_session(onnx_path)
    ort_logits = ort_session.run(None, {"input_values": input_values.detach().numpy()})[0]

    assert np.allclose(hf_logits, ort_logits, atol=1e-5)


@pytest.mark.llm_model
def test_seq_classification_cpu_parity_and_export(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TINY_SEQ_CLASSIFICATION_MODEL_ID, trust_remote_code=True)
    model_hf = AutoModelForSequenceClassification.from_pretrained(
        TINY_SEQ_CLASSIFICATION_MODEL_ID,
        trust_remote_code=True,
    )
    model_hf.eval()

    inputs = tokenizer("quick classification check", return_tensors="pt")
    hf_logits = model_hf(**inputs).logits.detach().numpy()

    qeff_model = QEFFAutoModelForSequenceClassification(model_hf)
    qeff_logits = qeff_model.model(**inputs).logits.detach().numpy()
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
    ort_session = _ort_session(onnx_path)
    input_names = {item.name for item in ort_session.get_inputs()}
    ort_logits = ort_session.run(
        None,
        {name: tensor.detach().numpy() for name, tensor in inputs.items() if name in input_names},
    )[0]

    assert np.allclose(hf_logits, qeff_logits, atol=1e-5)
    assert np.allclose(hf_logits, ort_logits, atol=1e-5)


@pytest.mark.llm_model
def test_whisper_export_smoke(tmp_path):
    model_hf = AutoModelForSpeechSeq2Seq.from_pretrained(
        TINY_WHISPER_MODEL_ID,
        **MODEL_KWARGS,
        low_cpu_mem_usage=False,
    )
    model_hf.eval()

    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model_hf, pretrained_model_name_or_path=TINY_WHISPER_MODEL_ID)
    onnx_path = _run_whisper_export_smoke(qeff_model, tmp_path / "whisper")

    assert onnx_path.name.endswith(".onnx")


@pytest.mark.llm_model
def test_causal_subfunction_export_smoke(tmp_path):
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    with_subfunctions_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "with-subfunctions", use_onnx_subfunctions=True, offload_pt_weights=False)
    )
    without_subfunctions_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "without-subfunctions", use_onnx_subfunctions=False)
    )

    with_subfunctions_model = onnx.load(with_subfunctions_path, load_external_data=False)
    without_subfunctions_model = onnx.load(without_subfunctions_path, load_external_data=False)
    with_names = [func.name for func in with_subfunctions_model.functions]
    without_names = [func.name for func in without_subfunctions_model.functions]
    assert any("QEffGPT2Block" in name for name in with_names)
    assert not any("QEffGPT2Block" in name for name in without_names)


@pytest.mark.llm_model
def test_causal_subfunction_export_uses_semantic_weight_and_node_names(tmp_path):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model_hf = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS).eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    onnx_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "semantic-wsub-names", use_onnx_subfunctions=True, offload_pt_weights=False)
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)

    initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}
    assert "model.layers.0.self_attn.q_proj.weight" in initializer_names
    assert not any(name.startswith("onnx::MatMul_") for name in initializer_names)

    function_node_names = {node.name for function in onnx_model.functions for node in function.node}
    assert "self_attn/q_proj/MatMul" in function_node_names
    assert "mlp/down_proj/MatMul" in function_node_names


@pytest.mark.llm_model
def test_gemma3_vlm_export_parity_with_and_without_subfunctions(tmp_path):
    """Gemma3 VLM export keeps retained-state/output signatures stable across subfunction toggles.

    This guards the Gemma3 cache-bridging export patch on the supported VLM path:
    the historical working non-subfunction export remains the reference, and
    enabling subfunctions must preserve the exported graph interface.
    """
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        VLM_TEXT_RUNTIME_MODEL_ID,
        trust_remote_code=True,
        kv_offload=True,
    )

    with_subfunctions_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "gemma3-vlm-with-subfunctions",
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )
    without_subfunctions_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / "gemma3-vlm-without-subfunctions",
            use_onnx_subfunctions=False,
            offload_pt_weights=False,
        )
    )

    with_model = onnx.load(with_subfunctions_path, load_external_data=False)
    without_model = onnx.load(without_subfunctions_path, load_external_data=False)

    # Ensure the subfunction export path really emits decoder-block subfunctions
    # for the Gemma3 language decoder, not just a graph with matching IO names.
    with_subfunction_names = _decoder_block_subfunction_names(with_model, qeff_model.lang_model)
    without_subfunction_names = _decoder_block_subfunction_names(without_model, qeff_model.lang_model)
    assert with_subfunction_names, "Expected Gemma3 decoder-layer subfunctions with use_onnx_subfunctions=True"
    assert not without_subfunction_names, (
        "Did not expect Gemma3 decoder-layer subfunctions with use_onnx_subfunctions=False"
    )

    assert [value.name for value in with_model.graph.input] == [value.name for value in without_model.graph.input]
    assert [value.name for value in with_model.graph.output] == [value.name for value in without_model.graph.output]


@pytest.mark.llm_model
def test_repeat_kv_quickcheck_hf_qeff_ort_parity(tmp_path):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model_hf = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS).eval()
    qeff_model = QEFFAutoModelForCausalLM(deepcopy(model_hf), qaic_config={"replicate_kv_heads": True})

    input_ids = torch.arange(1, 5, dtype=torch.int64).view(1, 4)
    position_ids = torch.arange(4, dtype=torch.int64).view(1, 4)
    head_dim = config.hidden_size // config.num_attention_heads
    past_key_values = tuple(
        (
            torch.zeros((1, config.num_attention_heads, 8, head_dim), dtype=torch.float32),
            torch.zeros((1, config.num_attention_heads, 8, head_dim), dtype=torch.float32),
        )
        for _ in range(config.num_hidden_layers)
    )
    inputs = {"input_ids": input_ids, "position_ids": position_ids, "past_key_values": past_key_values}
    with torch.no_grad():
        hf_logits = model_hf(input_ids=input_ids, position_ids=position_ids).logits[:, -1:, :].detach().numpy()

    qeff_model.transform(ctx_len=8, seq_len=4, bs=1, num_devices=4, qaic_config=qeff_model.model.qaic_config)
    with torch.no_grad():
        qeff_logits = qeff_model.model(**inputs).logits.detach().numpy()

    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "repeat-kv"))
    ort_session, added_initializers = _ort_session_zeroing_int32_max_constants(onnx_path)
    assert added_initializers is not None
    flat_inputs = {name: value.detach().numpy() for name, value in inputs.items() if torch.is_tensor(value)}
    for layer_idx, layer_state in enumerate(past_key_values):
        flat_inputs[f"past_key.{layer_idx}"] = layer_state[0].detach().numpy()
        flat_inputs[f"past_value.{layer_idx}"] = layer_state[1].detach().numpy()
    ort_inputs = {item.name: flat_inputs[item.name] for item in ort_session.get_inputs()}
    ort_logits = ort_session.run(None, ort_inputs)[0]

    assert np.allclose(hf_logits, qeff_logits, atol=1e-4, rtol=1e-4)
    assert np.allclose(qeff_logits, ort_logits, atol=1e-4, rtol=1e-4)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "config_kwargs"),
    sorted(TINY_MOE_PREFILL_SUBFUNCTION_CONFIGS.items()),
    ids=sorted(TINY_MOE_PREFILL_SUBFUNCTION_CONFIGS),
)
def test_moe_prefill_subfunction_export_uses_einsum_reductions(model_type, config_kwargs, tmp_path):
    config = AutoConfig.for_model(model_type, **config_kwargs)
    model_hf = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf, continuous_batching=False)
    qeff_model.transform(
        ctx_len=64,
        seq_len=64,
        bs=1,
        prefill_only=True,
        enable_chunking=True,
        num_cores=2,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 32}},
        prefill_seq_len=64,
    )

    onnx_path = _exported_onnx_path(
        qeff_model.export(
            tmp_path / f"{model_type}-prefill-subfunctions",
            prefill_only=True,
            enable_chunking=True,
            prefill_seq_len=64,
            num_cores=2,
            qaic_config={"moe_config": {"expert_parallel_chunk_size": 32}},
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    decoder_function_names = _decoder_block_subfunction_names(onnx_model, qeff_model)
    decoder_op_types = _function_op_types(onnx_model, decoder_function_names)

    assert len(decoder_function_names) == config.num_hidden_layers
    assert "Einsum" in decoder_op_types
    assert "CtxGather3D" in decoder_op_types
    assert "CtxScatter3D" in decoder_op_types
    assert "CtxScatter3DInt" in decoder_op_types


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_compile_with_subfunctions_all_models(model_type, model_id, tmp_path):
    del model_type
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    try:
        qpc = qeff_model.compile(
            prefill_seq_len=8,
            ctx_len=32,
            use_onnx_subfunctions=True,
            compile_dir=tmp_path / "compile-with-subfunctions",
        )
    except Exception as exc:
        pytest.skip(
            f"Skipping compile for {model_id}: compile backend unavailable or unsupported in this environment "
            f"({type(exc).__name__}: {exc})"
        )

    qpc_path = Path(qpc)
    assert qpc_path.name == "qpc"
    assert qpc_path.is_dir()


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_subfunction_export_smoke_all_models(model_type, model_id, tmp_path):
    if model_type == "gpt_oss":
        pytest.skip("Subfunction runtime parity is currently excluded for gpt_oss in quickcheck.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model_hf = AutoModelForCausalLM.from_pretrained(
            model_id,
            **MODEL_KWARGS,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    prompt = ["hello world"]
    prompt_len = 8
    ctx_len = 12

    model_hf.eval()

    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=prompt,
        prompt_len=prompt_len,
        ctx_len=ctx_len,
        full_batch_size=None,
    )

    hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
    qeff_model = QEFFAutoModelForCausalLM(model_hf)
    kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "with-subfunctions-all", use_onnx_subfunctions=True))
    ort_tokens = api_runner.run_kv_model_on_ort(str(onnx_path))

    assert np.array_equal(hf_tokens, kv_tokens.squeeze(0))
    assert np.array_equal(kv_tokens, ort_tokens)


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("model_type", "model_id"),
    sorted(CAUSAL_RUNTIME_MODEL_IDS.items()),
    ids=sorted(CAUSAL_RUNTIME_MODEL_IDS),
)
def test_causal_subfunction_count_with_onnx_subfunctions(model_type, model_id, tmp_path):
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    onnx_path = _exported_onnx_path(
        qeff_model.export(tmp_path / f"subfunction-count-{model_type}", use_onnx_subfunctions=True)
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    subfunction_count = _count_decoder_block_subfunctions(onnx_model, qeff_model)

    if model_type in CAUSAL_MULTI_SUBFUNCTION_MODEL_TYPES:
        assert subfunction_count > 1, (
            f"{model_type} expected multiple decoder-block subfunctions (>1), but found {subfunction_count}"
        )
    else:
        assert subfunction_count == 1, (
            f"{model_type} expected a single decoder-block subfunction (1), but found {subfunction_count}"
        )


@pytest.mark.llm_model
def test_causal_subfunction_and_proxy_export_smoke_gpt2(tmp_path):
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            enable_proxy=True,
            torch_dtype=torch.float32,
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_model, enable_proxy=True)
    onnx_path = _exported_onnx_path(
        qeff_model.export(tmp_path / "with-subfunctions-and-proxy", use_onnx_subfunctions=True)
    )
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    assert any("QEffGPT2Block" in func.name for func in onnx_model.functions)


def test_subfunction_export_restores_onnx_transforms_on_failure():
    from QEfficient.base.onnx_transforms import CustomOpTransform
    from QEfficient.transformers.cache_utils import InvalidIndexProvider
    from QEfficient.utils.export_utils import export_wrapper

    class DummyInner:
        config = None

        def get_submodules_for_export(self):
            return set()

    class DummyQEff:
        model = DummyInner()
        model_architecture = None
        model_name = "DummyQEff"
        hash_params = {}
        export_hash = None
        _onnx_transforms = []

        @export_wrapper
        def export(self, export_dir=None, **kwargs):
            raise RuntimeError("forced export failure")

    qeff_model = DummyQEff()

    with pytest.raises(RuntimeError, match="forced export failure"):
        qeff_model.export(use_onnx_subfunctions=True)

    assert qeff_model._onnx_transforms == []
    assert CustomOpTransform not in DummyQEff._onnx_transforms
    assert InvalidIndexProvider.SUBFUNC_ENABLED is False


@pytest.mark.llm_model
def test_prefix_caching_continuous_batching_export_and_ort_smoke(tmp_path):
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        PREFIX_CACHING_MODEL_ID, continuous_batching=True, torch_dtype=torch.float32
    )
    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "prefix-caching"))
    onnx_model = onnx.load(onnx_path, load_external_data=False)

    input_names = {inp.name for inp in onnx_model.graph.input}
    output_names = {out.name for out in onnx_model.graph.output}
    op_types = {node.op_type for node in onnx_model.graph.node}
    assert "batch_index" in input_names
    assert "CtxScatterCB" in op_types
    assert "CtxGatherCB" in op_types
    assert any(name.endswith("_RetainedState") for name in output_names)


@pytest.mark.llm_model
def test_awq_export_smoke(tmp_path):
    replace_transformers_quantizers()
    try:
        model_hf = AutoModelForCausalLM.from_pretrained(
            TINY_AWQ_MODEL_ID, low_cpu_mem_usage=False, torch_dtype=torch.float32
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, TINY_AWQ_MODEL_ID)
    model_hf.eval()

    qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=TINY_AWQ_MODEL_ID)
    with _suppress_native_output():
        onnx_path = _exported_onnx_path(qeff_model.export(tmp_path))
        onnx_model = onnx.load(onnx_path, load_external_data=False)

    assert any(node.op_type == "MatMulNBits" for node in onnx_model.graph.node)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_causal_lm():
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    try:
        qeff_default = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float32
        )
        qeff_proxy = QEFFAutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, enable_proxy=True, torch_dtype=torch.float32
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_embedding():
    model_id = TINY_TEXT_EMBEDDING_MODEL_ID
    try:
        qeff_default = QEFFAutoModel.from_pretrained(model_id)
        qeff_proxy = QEFFAutoModel.from_pretrained(model_id, enable_proxy=True)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(
        qeff_default, enable_proxy=False, always_on_transforms={"FP16ClipTransform", "SplitTensorsTransform"}
    )
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_sequence_classification():
    model_id = TINY_SEQ_CLASSIFICATION_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
        qeff_proxy = QEFFAutoModelForSequenceClassification.from_pretrained(
            model_id, trust_remote_code=True, enable_proxy=True
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_whisper():
    model_id = TINY_WHISPER_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True)
        qeff_proxy = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True, enable_proxy=True)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_ctc():
    model_id = TINY_AUDIO_CTC_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForCTC.from_pretrained(model_id, trust_remote_code=True)
        qeff_proxy = QEFFAutoModelForCTC.from_pretrained(model_id, trust_remote_code=True, enable_proxy=True)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


@pytest.mark.llm_model
def test_proxy_toggle_onnx_transform_policy_for_vlm():
    model_id = VLM_TEXT_RUNTIME_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, kv_offload=True
        )
        qeff_proxy = QEFFAutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, enable_proxy=True, kv_offload=True
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_dual_qpc_vlm_proxy_transform_policy(qeff_default, enable_proxy=False)
    _assert_dual_qpc_vlm_proxy_transform_policy(qeff_proxy, enable_proxy=True)


class TestCausalLMFlagDiagnostics:
    """Unsupported/ignored CausalLM flags should fail or warn clearly."""

    def _tiny_llama(self):
        try:
            return QEFFAutoModelForCausalLM.from_pretrained(
                CAUSAL_RUNTIME_MODEL_IDS["llama"],
                continuous_batching=True,
                num_hidden_layers=2,
                **MODEL_KWARGS,
            )
        except Exception as exc:
            _skip_on_model_fetch_error(exc, CAUSAL_RUNTIME_MODEL_IDS["llama"])

    def test_export_decode_only_rejected_for_standard_causal_lm(self, tmp_path):
        qeff_model = self._tiny_llama()

        with pytest.raises(NotImplementedError, match="decode_only=True is not supported"):
            qeff_model.export(tmp_path / "decode-only", decode_only=True)

    def test_compile_retain_full_kv_ignored_for_llama(self, tmp_path, monkeypatch, caplog):
        qeff_model = self._tiny_llama()
        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake")
        captured_kwargs = {}

        def fake_compile(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "qpc"

        monkeypatch.setattr(qeff_model, "_compile", fake_compile)
        caplog.set_level(logging.WARNING, logger="QEfficient")

        qeff_model.compile(
            onnx_path=str(onnx_path),
            compile_dir=str(tmp_path),
            prefill_seq_len=1,
            ctx_len=128,
            full_batch_size=1,
            prefill_only=False,
            retain_full_kv=True,
        )

        assert captured_kwargs["retain_full_kv"] is False
        assert captured_kwargs["specializations"][0]["_graph_name"] == "Decode"
        assert "retain_full_kv=True is only supported" in caplog.text


# ---------------------------------------------------------------------------
# Tests for the named-specializations format (backend team feature request)
# ---------------------------------------------------------------------------


class TestInferSpecializationName:
    """Unit tests for _infer_specialization_name."""

    # --- _graph_name tag (authoritative, set at creation time) ---

    def test_graph_name_tag_takes_priority(self):
        """_graph_name in spec overrides all heuristics."""
        spec = {"_graph_name": "Vision", "batch_size": "1", "seq_len": "128", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 0) == "Vision"

    def test_graph_name_tag_whisper_encoder(self):
        spec = {
            "_graph_name": "Encoder",
            "batch_size": "1",
            "seq_len": "1",
            "encoder_ctx_len": "1500",
            "feature_len": "3000",
        }
        assert _infer_specialization_name(spec, 0) == "Encoder"

    def test_graph_name_tag_whisper_decode(self):
        spec = {
            "_graph_name": "Decode",
            "batch_size": "1",
            "seq_len": "1",
            "encoder_ctx_len": "1500",
            "feature_len": "1",
        }
        assert _infer_specialization_name(spec, 1) == "Decode"

    def test_graph_name_tag_prefill(self):
        spec = {"_graph_name": "Prefill", "batch_size": "1", "seq_len": "128", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 0) == "Prefill"

    def test_graph_name_tag_decode(self):
        spec = {"_graph_name": "Decode", "batch_size": "1", "seq_len": "1", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 1) == "Decode"

    def test_graph_name_tag_embedding_with_seq_len(self):
        spec = {"_graph_name": "Embedding", "batch_size": "1", "seq_len": "128"}
        assert _infer_specialization_name(spec, 0) == "Embedding"

    # --- module_name hint (diffusers path) ---

    def test_module_name_used_when_no_tag(self):
        spec = {"batch_size": "1", "seq_len": "77"}
        assert _infer_specialization_name(spec, 0, module_name="text_encoder") == "text_encoder"

    def test_module_name_with_model_type(self):
        spec = {"batch_size": "1", "model_type": 1}
        assert _infer_specialization_name(spec, 0, module_name="transformer") == "transformer_model_type_1"

    # --- seq_len heuristic fallback (plain causal LM raw dicts) ---

    def test_prefill_detected_by_seq_len_gt_1(self):
        spec = {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 0) == "Prefill"

    def test_decode_detected_by_seq_len_1_string(self):
        spec = {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"}
        assert _infer_specialization_name(spec, 1) == "Decode"

    def test_decode_detected_by_seq_len_1_int(self):
        spec = {"batch_size": 1, "seq_len": 1, "ctx_len": 4096}
        assert _infer_specialization_name(spec, 1) == "Decode"

    def test_encoder_detected_by_encoder_ctx_len_no_seq_len(self):
        spec = {"batch_size": "1", "encoder_ctx_len": "1500"}
        assert _infer_specialization_name(spec, 0) == "Encoder"

    def test_legacy_embedding_detected_by_sequence_length(self):
        spec = {"batch_size": "1", "sequence_length": "128"}
        assert _infer_specialization_name(spec, 0) == "Embedding"

    def test_generic_fallback_for_unknown_shape(self):
        spec = {"custom_dim": "42"}
        assert _infer_specialization_name(spec, 3) == "Graph_3"


class TestToNamedSpecializations:
    """Unit tests for to_named_specializations — the main serialization helper."""

    def test_llm_prefill_decode_pair(self):
        """Standard LLM: two specializations → Prefill + Decode."""
        flat = [
            {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
        ]
        result = to_named_specializations(flat)
        assert len(result) == 2
        assert result[0] == {
            "name": "Prefill",
            "symbols": {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
        }
        assert result[1] == {
            "name": "Decode",
            "symbols": {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
        }

    def test_llm_continuous_batching_with_full_batch_size(self):
        """Continuous-batching LLM: full_batch_size present in both entries."""
        flat = [
            {"batch_size": "1", "full_batch_size": "16", "seq_len": "128", "ctx_len": "4096"},
            {"batch_size": "16", "full_batch_size": "16", "seq_len": "1", "ctx_len": "4096"},
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Prefill"
        assert result[1]["name"] == "Decode"
        assert result[0]["symbols"]["full_batch_size"] == "16"
        assert result[1]["symbols"]["batch_size"] == "16"

    def test_prefill_only_single_entry(self):
        """When prompt_len == 1 the decode entry is dropped; only one entry remains."""
        flat = [{"batch_size": "1", "seq_len": "1", "ctx_len": "128"}]
        result = to_named_specializations(flat)
        assert len(result) == 1
        assert result[0]["name"] == "Decode"

    def test_vlm_vision_specialization(self):
        """VLM vision encoder: _graph_name tag set at source → 'Vision', tag stripped from symbols."""
        flat = [
            {
                "_graph_name": "Vision",
                "batch_size": "1",
                "vision_size": "247",
                "grid_height": "988",
                "grid_width": "1176",
                "grid_h": "26",
                "grid_w": "38",
            }
        ]
        result = to_named_specializations(flat)
        assert len(result) == 1
        assert result[0]["name"] == "Vision"
        assert result[0]["symbols"]["vision_size"] == "247"
        assert "_graph_name" not in result[0]["symbols"]

    def test_vlm_lang_prefill_decode(self):
        """VLM language side: prefill + decode with vision_batch_size."""
        flat = [
            {
                "batch_size": "1",
                "ctx_len": "4096",
                "seq_len": "128",
                "vision_batch_size": "1",
                "vision_size": "247",
            },
            {
                "batch_size": "1",
                "ctx_len": "4096",
                "seq_len": "1",
                "vision_batch_size": "1",
                "vision_size": "247",
            },
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Prefill"
        assert result[1]["name"] == "Decode"
        assert result[0]["symbols"]["vision_size"] == "247"
        assert result[1]["symbols"]["vision_size"] == "247"

    def test_all_values_coerced_to_strings(self):
        """Integer values in the flat dict must be stringified in symbols."""
        flat = [{"batch_size": 1, "seq_len": 32, "ctx_len": 128}]
        result = to_named_specializations(flat)
        symbols = result[0]["symbols"]
        assert all(isinstance(v, str) for v in symbols.values())

    def test_output_structure_keys(self):
        """Every entry must have exactly 'name' and 'symbols' keys."""
        flat = [
            {"batch_size": "1", "seq_len": "64", "ctx_len": "256"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "256"},
        ]
        result = to_named_specializations(flat)
        for entry in result:
            assert set(entry.keys()) == {"name", "symbols"}

    def test_whisper_encoder_specialization_simplified(self):
        """Simplified Whisper spec: encoder_ctx_len, no seq_len → 'Encoder' via heuristic."""
        flat = [
            {"batch_size": "1", "encoder_ctx_len": "1500"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "448"},
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Encoder"
        assert result[1]["name"] == "Decode"
        assert result[0]["symbols"]["encoder_ctx_len"] == "1500"

    def test_whisper_encoder_specialization_real_spec(self):
        """Real Whisper spec: _graph_name tags set at source distinguish Encoder from Decode."""
        flat = [
            {
                "_graph_name": "Encoder",
                "batch_size": "1",
                "seq_len": "1",
                "encoder_ctx_len": "1500",
                "decoder_ctx_len": "150",
                "feature_len": "3000",
            },
            {
                "_graph_name": "Decode",
                "batch_size": "1",
                "seq_len": "1",
                "encoder_ctx_len": "1500",
                "decoder_ctx_len": "150",
                "feature_len": "1",
            },
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Encoder"
        assert result[1]["name"] == "Decode"
        assert result[0]["symbols"]["feature_len"] == "3000"
        assert result[1]["symbols"]["feature_len"] == "1"
        assert "_graph_name" not in result[0]["symbols"]
        assert "_graph_name" not in result[1]["symbols"]

    def test_graph_name_tag_stripped_from_symbols(self):
        """_graph_name must never appear in the serialized symbols dict."""
        flat = [{"_graph_name": "Prefill", "batch_size": "1", "seq_len": "128", "ctx_len": "4096"}]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Prefill"
        assert "_graph_name" not in result[0]["symbols"]

    def test_text_embedding_specialization_actual_compile_path(self):
        """Text embedding compile path: _graph_name + seq_len should serialize as Embedding."""
        flat = [{"_graph_name": "Embedding", "batch_size": "1", "seq_len": "128"}]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Embedding"
        assert result[0]["symbols"]["seq_len"] == "128"

    def test_legacy_text_embedding_specialization(self):
        """Legacy BERT-like raw dict: sequence_length fallback still resolves to Embedding."""
        flat = [{"batch_size": "1", "sequence_length": "128"}]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Embedding"
        assert result[0]["symbols"]["sequence_length"] == "128"

    def test_roundtrip_via_json(self):
        """Serializing to JSON and back must preserve the named format exactly."""
        import json

        flat = [
            {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
        ]
        named = to_named_specializations(flat)
        payload = {"specializations": named}
        roundtripped = json.loads(json.dumps(payload))
        assert roundtripped["specializations"][0]["name"] == "Prefill"
        assert roundtripped["specializations"][1]["name"] == "Decode"
        assert roundtripped["specializations"][0]["symbols"]["seq_len"] == "128"

    def test_compile_helper_create_and_dump_specializations(self, tmp_path):
        """create_and_dump_specializations must write the new named format to disk."""
        import json

        from QEfficient.compile.compile_helper import create_and_dump_specializations

        out_path = tmp_path / "specializations.json"
        create_and_dump_specializations(
            batch_size=1,
            prompt_len=128,
            ctx_len=4096,
            path=str(out_path),
        )
        data = json.loads(out_path.read_text())
        specs = data["specializations"]
        assert len(specs) == 2
        assert specs[0]["name"] == "Prefill"
        assert specs[1]["name"] == "Decode"
        assert "symbols" in specs[0]
        assert specs[0]["symbols"]["seq_len"] == "128"
        assert specs[1]["symbols"]["seq_len"] == "1"

    def test_compile_helper_prefill_only_when_prompt_len_1(self, tmp_path):
        """When prompt_len==1 and no full_batch_size, only one entry is written."""
        import json

        from QEfficient.compile.compile_helper import create_and_dump_specializations

        out_path = tmp_path / "specializations_prefill_only.json"
        create_and_dump_specializations(
            batch_size=1,
            prompt_len=1,
            ctx_len=128,
            path=str(out_path),
        )
        data = json.loads(out_path.read_text())
        specs = data["specializations"]
        assert len(specs) == 1
        assert specs[0]["name"] == "Decode"

    def test_compile_helper_continuous_batching(self, tmp_path):
        """With full_batch_size, both entries carry full_batch_size in symbols."""
        import json

        from QEfficient.compile.compile_helper import create_and_dump_specializations

        out_path = tmp_path / "specializations_cb.json"
        create_and_dump_specializations(
            batch_size=1,
            prompt_len=128,
            ctx_len=4096,
            path=str(out_path),
            full_batch_size=16,
        )
        data = json.loads(out_path.read_text())
        specs = data["specializations"]
        assert len(specs) == 2
        assert specs[0]["name"] == "Prefill"
        assert specs[1]["name"] == "Decode"
        assert specs[0]["symbols"]["full_batch_size"] == "16"
        assert specs[1]["symbols"]["full_batch_size"] == "16"
        assert specs[1]["symbols"]["batch_size"] == "16"


class TestGetCompilationDims:
    """Verify get_compilation_dims handles both flat (legacy) and named (new) formats."""

    def _write_spec(self, tmp_path, payload):
        import json

        spec_dir = tmp_path / "qpc-hash"
        spec_dir.mkdir()
        qpc_dir = spec_dir / "qpc"
        qpc_dir.mkdir()
        (spec_dir / "specializations.json").write_text(json.dumps(payload))
        return str(qpc_dir)

    def test_new_named_format(self, tmp_path):
        from QEfficient.generation.text_generation_inference import get_compilation_dims

        qpc_path = self._write_spec(
            tmp_path,
            {
                "specializations": [
                    {"name": "Prefill", "symbols": {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"}},
                    {"name": "Decode", "symbols": {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"}},
                ]
            },
        )
        bs, ctx, fbs = get_compilation_dims(qpc_path)
        assert bs == 1
        assert ctx == 4096
        assert fbs is None

    def test_new_named_format_with_full_batch_size(self, tmp_path):
        from QEfficient.generation.text_generation_inference import get_compilation_dims

        qpc_path = self._write_spec(
            tmp_path,
            {
                "specializations": [
                    {
                        "name": "Prefill",
                        "symbols": {"batch_size": "1", "full_batch_size": "16", "seq_len": "128", "ctx_len": "4096"},
                    },
                    {
                        "name": "Decode",
                        "symbols": {"batch_size": "16", "full_batch_size": "16", "seq_len": "1", "ctx_len": "4096"},
                    },
                ]
            },
        )
        bs, ctx, fbs = get_compilation_dims(qpc_path)
        assert bs == 1
        assert ctx == 4096
        assert fbs == 16

    def test_legacy_flat_format_still_works(self, tmp_path):
        from QEfficient.generation.text_generation_inference import get_compilation_dims

        qpc_path = self._write_spec(
            tmp_path,
            {
                "specializations": [
                    {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
                    {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
                ]
            },
        )
        bs, ctx, fbs = get_compilation_dims(qpc_path)
        assert bs == 1
        assert ctx == 4096
        assert fbs is None


class TestDiffusersNamedSpecializations:
    """Named-specialization format for diffusers pipeline modules via _graph_name tag."""

    def _tag(self, specs, module_name):
        """Simulate what pipeline_utils does: tag each spec with _graph_name."""
        return [
            {**s, "_graph_name": f"{module_name}_model_type_{s['model_type']}" if "model_type" in s else module_name}
            for s in specs
        ]

    def test_flux_text_encoder(self):
        flat = self._tag([{"batch_size": 1, "seq_len": 77}], "text_encoder")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "text_encoder"
        assert result[0]["symbols"] == {"batch_size": "1", "seq_len": "77"}
        assert "_graph_name" not in result[0]["symbols"]

    def test_flux_text_encoder_2(self):
        flat = self._tag([{"batch_size": 1, "seq_len": 256}], "text_encoder_2")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "text_encoder_2"

    def test_flux_transformer(self):
        flat = self._tag([{"batch_size": 1, "seq_len": 256, "steps": 1}], "transformer")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "transformer"

    def test_flux_vae_decoder(self):
        flat = self._tag([{"batch_size": 1, "channels": 16}], "vae_decoder")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "vae_decoder"

    def test_wan_transformer_model_type_naming(self):
        """Wan transformer: two model_type entries → transformer_model_type_1 / _2."""
        flat = self._tag(
            [
                {"batch_size": "1", "num_channels": "16", "steps": "1", "sequence_length": "512", "model_type": 1},
                {"batch_size": "1", "num_channels": "16", "steps": "1", "sequence_length": "512", "model_type": 2},
            ],
            "transformer",
        )
        result = to_named_specializations(flat)
        assert result[0]["name"] == "transformer_model_type_1"
        assert result[1]["name"] == "transformer_model_type_2"

    def test_wan_vae_decoder(self):
        flat = self._tag([{"batch_size": 1, "num_channels": 16}], "vae_decoder")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "vae_decoder"

    def test_wan_i2v_vae_encoder(self):
        flat = self._tag([{"batch_size": 1, "num_channels": 16}], "vae_encoder")
        result = to_named_specializations(flat)
        assert result[0]["name"] == "vae_encoder"

    def test_graph_name_tag_stripped_from_symbols(self):
        flat = self._tag([{"batch_size": 1, "seq_len": 77}], "text_encoder")
        result = to_named_specializations(flat)
        assert "_graph_name" not in result[0]["symbols"]

    def test_idempotent_already_named_entries(self):
        already = [{"name": "transformer", "symbols": {"batch_size": "1", "steps": "1"}}]
        result = to_named_specializations(already)
        assert result == already

    def test_no_tag_falls_back_to_lm_rules(self):
        """Without _graph_name tag, seq_len heuristic applies."""
        flat = [
            {"batch_size": "1", "seq_len": "128", "ctx_len": "4096"},
            {"batch_size": "1", "seq_len": "1", "ctx_len": "4096"},
        ]
        result = to_named_specializations(flat)
        assert result[0]["name"] == "Prefill"
        assert result[1]["name"] == "Decode"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_resolve_torch_dtype_normalizes_dtype_alias(caplog):
    """transformers-v5 ``dtype`` alias must be honored and kept in sync with ``torch_dtype``.

    Regression guard: passing ``dtype=float16`` used to be ignored, leaving the
    model forced to float32 - breaking the Qwen3.5 float16 export path.
    """
    from QEfficient.transformers.models.modeling_auto import _resolve_torch_dtype

    # float16 supplied via the v5 alias must survive and populate torch_dtype.
    kwargs = {"dtype": torch.float16}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.float16
    assert kwargs["dtype"] == torch.float16

    # float16 via the legacy name still works.
    kwargs = {"torch_dtype": torch.float16}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.float16

    # bfloat16 is kept as-is on ai100 (regardless of which name is used) so export/compile
    # still run in bfloat16, with a warning that on-device generation is expected to fail.
    caplog.set_level(logging.WARNING, logger="QEfficient")
    kwargs = {"dtype": torch.bfloat16}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.bfloat16
    assert kwargs["dtype"] == torch.bfloat16
    assert "on-device generation is expected to fail" in caplog.text

    # Regression guard: when torch_dtype/dtype is not set at all, default to float32 on
    # ai100 so models whose config.json declares bfloat16 (e.g. Mistral-7B-v0.1) aren't
    # silently loaded in bfloat16, which breaks KV-cache dtype (float32) parity.
    kwargs = {}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.float32


def test_qwen3_5_moe_gated_norm_preserves_float16():
    """GatedDeltaNet RMSNorm must keep the input dtype so the gated output feeds
    the float16 out_proj without a dtype mismatch (Qwen3.5 float16 export)."""
    import torch.nn as nn

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeGatedDeltaNetCustomRMSNormAIC,
    )

    norm = QEffQwen3_5MoeGatedDeltaNetCustomRMSNormAIC()
    norm.weight = nn.Parameter(torch.ones(16, dtype=torch.float16))
    norm.eps = 1e-6

    out = norm(torch.randn(4, 16, dtype=torch.float16), torch.randn(4, 16, dtype=torch.float16))
    assert out.dtype == torch.float16


def test_qwen3_5_moe_get_submodules_for_export_keeps_decoder_layer_for_mixed_layer_types():
    """Mixed full/linear attention configs must still expose decoder layer subfunctions."""
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeDecoderLayer,
        QEffQwen3_5MoeDecoderWrapper,
        QEffQwen3_5MoeForCausalLM,
    )

    causal_lm = QEffQwen3_5MoeForCausalLM.__new__(QEffQwen3_5MoeForCausalLM)
    causal_lm.config = SimpleNamespace(layer_types=["full_attention", "linear_attention"])
    assert causal_lm.get_submodules_for_export() == {QEffQwen3_5MoeDecoderLayer}

    wrapper = QEffQwen3_5MoeDecoderWrapper.__new__(QEffQwen3_5MoeDecoderWrapper)
    wrapper.config = SimpleNamespace(text_config=SimpleNamespace(layer_types=["full_attention", "linear_attention"]))
    assert wrapper.get_submodules_for_export() == {QEffQwen3_5MoeDecoderLayer}


def test_qwen3_5_moe_get_specializations_supports_multi_resolution():
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeForConditionalGeneration

    model = QEffQwen3_5MoeForConditionalGeneration.__new__(QEffQwen3_5MoeForConditionalGeneration)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=16, temporal_patch_size=1))

    specs, _ = model.get_specializations(
        batch_size=1,
        prefill_seq_len=64,
        ctx_len=4096,
        height=[448, 896],
        width=[448, 896],
        num_frames=[1, 2],
        kv_offload=True,
    )

    vision_specs = specs["vision"]
    lang_specs = specs["lang"]

    assert len(vision_specs) == 2
    expected_vision_size = max(spec["vision_size"] * frames for spec, frames in zip(vision_specs, [1, 2]))
    assert all(spec["vision_size"] == expected_vision_size for spec in lang_specs)
    assert all(spec["vision_batch_size"] == 1 for spec in lang_specs)


def test_qwen3_5_moe_get_specializations_strips_vision_symbols_for_comp_ctx_variants():
    from types import SimpleNamespace

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import QEffQwen3_5MoeForConditionalGeneration

    model = QEffQwen3_5MoeForConditionalGeneration.__new__(QEffQwen3_5MoeForConditionalGeneration)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=16, temporal_patch_size=1))

    lang_specs, _ = model.get_specializations(
        batch_size=1,
        prefill_seq_len=64,
        ctx_len=4096,
        height=[448, 672],
        width=[448, 672],
        num_frames=[1, 1],
        comp_ctx_lengths_prefill=[32, 48],
        comp_ctx_lengths_decode=[96, 128],
        kv_offload=False,
    )

    assert [spec["seq_len"] for spec in lang_specs] == [64, 64, 1, 1]
    assert all("vision_size" not in spec for spec in lang_specs)
    assert all("vision_batch_size" not in spec for spec in lang_specs)


def test_moe_prefill_transform_does_not_require_enable_chunking():
    from QEfficient.transformers.models.glm4_moe.modeling_glm4_moe import QEffGlm4MoeMoE
    from QEfficient.transformers.models.pytorch_transforms import (
        PrefillOnlyChunkedTransform,
        PrefillOnlyTransform,
        RevertPrefillKeepAttentionTransform,
        RevertPrefillOnlyTransform,
    )
    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeSparseMoeBlock,
    )
    from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import (
        QEffQwen3MoeSparseMoeBlock,
    )
    from QEfficient.transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        QEffQwen3VLMoeTextSparseMoeBlock,
    )
    from QEfficient.transformers.moe import MoEFlavour

    for moe_cls in (
        QEffGlm4MoeMoE,
        QEffQwen3MoeSparseMoeBlock,
        QEffQwen3VLMoeTextSparseMoeBlock,
        QEffQwen3_5MoeSparseMoeBlock,
    ):
        assert moe_cls not in PrefillOnlyTransform._module_mapping
        assert moe_cls not in PrefillOnlyChunkedTransform._module_mapping
        assert moe_cls not in RevertPrefillKeepAttentionTransform._module_mapping
        assert moe_cls not in RevertPrefillOnlyTransform._module_mapping
        assert MoEFlavour.EXPERT_PARALLEL in moe_cls.supported_moe_flavours


@pytest.mark.llm_model
def test_subfunction_compile_io_names_use_internal_retained_state():
    from QEfficient.transformers.models.modeling_auto import _compile_io_name, _state_input_name

    output_name = _compile_io_name("past_value.1_RetainedState", use_onnx_subfunctions=True)

    assert output_name == "past_value.1_InternalRetainedState"
    assert _state_input_name(output_name) == "past_value.1"
    assert _compile_io_name("vision_embeds_RetainedState", use_onnx_subfunctions=True) == "vision_embeds_RetainedState"


@pytest.mark.llm_model
def test_runtime_aliases_internal_retained_state_outputs():
    from QEfficient.generation.cloud_infer import _add_basename_binding_aliases, _public_retained_state_name

    assert _public_retained_state_name("past_key.0_InternalRetainedState") == "past_key.0_RetainedState"
    assert _public_retained_state_name("past_value.1_InternalRetainedState") == "past_value.1_RetainedState"
    assert _public_retained_state_name("logits") is None

    binding_map = {"layer_0/input_ids": 3}
    bindings = [type("Binding", (), {"name": "layer_0/input_ids", "index": 3})()]
    _add_basename_binding_aliases(binding_map, bindings)
    assert binding_map["input_ids"] == 3


# ---------------------------------------------------------------------------
# Tests for the optional KV-cache buffer-name prefix (vLLM disaggregated KV transfer)
# ---------------------------------------------------------------------------


def _retained_state_outputs(onnx_path: Path) -> Set[str]:
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    return {out.name for out in onnx_model.graph.output if out.name.endswith("_RetainedState")}


def _kv_input_names(onnx_path: Path) -> Set[str]:
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    return {
        inp.name
        for inp in onnx_model.graph.input
        if inp.name.startswith(
            ("past_key.", "past_value.", "compressed_kv.", "k_pe.", "conv_state.", "recurrent_state.")
        )
    }


class TestApplyKvCachePrefixHelper:
    """Unit tests for the apply_kv_cache_prefix / validate_kv_cache_prefix helpers."""

    def test_flat_list_kv_only(self):
        from QEfficient.utils import apply_kv_cache_prefix

        names = ["logits", "past_key.0_RetainedState", "past_value.0_RetainedState", "vision_embeds_RetainedState"]
        result = apply_kv_cache_prefix(names, "VLLM")
        assert result == [
            "logits",
            "past_key.0_VLLM_RetainedState",
            "past_value.0_VLLM_RetainedState",
            "vision_embeds_RetainedState",  # vision buffer untouched
        ]

    def test_compressed_kv_and_k_pe(self):
        from QEfficient.utils import apply_kv_cache_prefix

        names = ["compressed_kv.0_RetainedState", "k_pe.0_RetainedState"]
        assert apply_kv_cache_prefix(names, "P") == ["compressed_kv.0_P_RetainedState", "k_pe.0_P_RetainedState"]

    def test_conv_recurrent_state_prefixed(self):
        """Hybrid linear-attention state (conv_state / recurrent_state on Qwen3.5 hybrid models)
        must receive the same KV-cache prefix as classical attention KV. vision/multimodal
        retained buffers must remain untouched on the same call."""
        from QEfficient.utils import apply_kv_cache_prefix

        names = [
            "logits",
            "past_key.0_RetainedState",
            "past_value.0_RetainedState",
            "conv_state.1_RetainedState",
            "recurrent_state.1_RetainedState",
            "vision_embeds_RetainedState",
            "image_idx_output",
            "pixel_values_RetainedState",
        ]
        result = apply_kv_cache_prefix(names, "VLLM")
        assert result == [
            "logits",
            "past_key.0_VLLM_RetainedState",
            "past_value.0_VLLM_RetainedState",
            "conv_state.1_VLLM_RetainedState",
            "recurrent_state.1_VLLM_RetainedState",
            "vision_embeds_RetainedState",
            "image_idx_output",
            "pixel_values_RetainedState",
        ]

    def test_align_inputs_handles_conv_recurrent(self):
        """The input rename map must pair conv_state/recurrent_state inputs with their
        prefixed retained-state outputs, exactly like past_key/past_value."""
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = [
            "input_ids",
            "past_key.0",
            "past_value.0",
            "conv_state.1",
            "recurrent_state.1",
        ]
        output_names = [
            "logits",
            "past_key.0_VLLM_RetainedState",
            "past_value.0_VLLM_RetainedState",
            "conv_state.1_VLLM_RetainedState",
            "recurrent_state.1_VLLM_RetainedState",
        ]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == [
            "input_ids",
            "past_key.0_VLLM",
            "past_value.0_VLLM",
            "conv_state.1_VLLM",
            "recurrent_state.1_VLLM",
        ]

    def test_dict_form_lang_only(self):
        from QEfficient.utils import apply_kv_cache_prefix

        names = {
            "vision": ["vision_embeds", "past_key.0_RetainedState"],  # vision side never rewritten
            "lang": ["logits", "vision_embeds_RetainedState", "past_key.0_RetainedState"],
        }
        result = apply_kv_cache_prefix(names, "VLLM")
        assert result["vision"] == ["vision_embeds", "past_key.0_RetainedState"]
        assert result["lang"] == ["logits", "vision_embeds_RetainedState", "past_key.0_VLLM_RetainedState"]

    def test_noop_when_prefix_absent(self):
        from QEfficient.utils import apply_kv_cache_prefix

        names = ["logits", "past_key.0_RetainedState"]
        assert apply_kv_cache_prefix(names, None) == names
        assert apply_kv_cache_prefix(names, "") == names

    def test_align_inputs_to_retained_outputs(self):
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = ["input_ids", "past_key.0", "past_value.0"]
        output_names = ["logits", "past_key.0_VLLM_RetainedState", "past_value.0_VLLM_RetainedState"]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == [
            "input_ids",
            "past_key.0_VLLM",
            "past_value.0_VLLM",
        ]

    def test_align_inputs_to_internal_retained_outputs(self):
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = ["input_ids", "past_key.0", "past_value.0"]
        output_names = ["logits", "past_key.0_VLLM_InternalRetainedState", "past_value.0_VLLM_InternalRetainedState"]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == [
            "input_ids",
            "past_key.0_VLLM",
            "past_value.0_VLLM",
        ]

    def test_align_inputs_noop_without_prefix(self):
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = ["input_ids", "past_key.0", "past_value.0"]
        output_names = ["logits", "past_key.0_RetainedState", "past_value.0_RetainedState"]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == input_names

    def test_on_device_sampler_buffers_not_prefixed(self):
        """Sampler retained-state buffers (past_repetition_penalty_buffer / past_presence_penalty_buffer)
        must NOT receive the KV-cache prefix on either the output or paired input side."""
        from QEfficient.utils import align_kv_input_names_to_retained_outputs, apply_kv_cache_prefix

        outputs = [
            "logits",
            "probs",
            "next_tokens",
            "past_key.0_RetainedState",
            "past_value.0_RetainedState",
            "past_repetition_penalty_buffer_RetainedState",
            "past_presence_penalty_buffer_RetainedState",
        ]
        prefixed = apply_kv_cache_prefix(outputs, "VLLM")
        assert "past_repetition_penalty_buffer_RetainedState" in prefixed
        assert "past_presence_penalty_buffer_RetainedState" in prefixed
        # And no sampler-buffer name accidentally carries the infix.
        assert not any("penalty_buffer_VLLM_RetainedState" in name for name in prefixed)

        inputs = [
            "input_ids",
            "past_key.0",
            "past_value.0",
            "past_repetition_penalty_buffer",
            "past_presence_penalty_buffer",
        ]
        aligned = align_kv_input_names_to_retained_outputs(inputs, prefixed)
        assert "past_repetition_penalty_buffer" in aligned
        assert "past_presence_penalty_buffer" in aligned
        assert not any(name.endswith("penalty_buffer_VLLM") for name in aligned)

    @pytest.mark.parametrize("bad", ["", "a_b", "a.b", "a b", 123, "past-key"])
    def test_validation_rejects_bad_prefix(self, bad):
        from QEfficient.utils import validate_kv_cache_prefix

        with pytest.raises(ValueError):
            validate_kv_cache_prefix(bad)

    def test_validation_accepts_alnum_and_none(self):
        from QEfficient.utils import validate_kv_cache_prefix

        assert validate_kv_cache_prefix("VLLM") == "VLLM"
        assert validate_kv_cache_prefix("vllm0") == "vllm0"
        assert validate_kv_cache_prefix(None) is None


@pytest.mark.llm_model
def test_causal_export_with_kv_cache_prefix(tmp_path):
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "prefixed", kv_cache_prefix="VLLM"))

    retained = _retained_state_outputs(onnx_path)
    assert retained, "expected retained-state outputs"
    # Every KV retained output carries the infix; the suffix is preserved.
    kv_retained = {name for name in retained if name.startswith(("past_key.", "past_value."))}
    assert kv_retained
    assert all(name.endswith("_VLLM_RetainedState") for name in kv_retained)

    # The matching device input buffer exists (output minus _RetainedState).
    kv_inputs = _kv_input_names(onnx_path)
    for out_name in kv_retained:
        stripped = out_name[: -len("_RetainedState")]
        assert stripped in kv_inputs, f"missing paired input buffer for {out_name}"
        assert stripped.endswith("_VLLM")


@pytest.mark.llm_model
def test_causal_export_default_names_unchanged(tmp_path):
    """Without the flag, retained-state names must remain byte-for-byte identical to today."""
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf.eval()
    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    onnx_path = _exported_onnx_path(qeff_model.export(tmp_path / "default"))
    retained = _retained_state_outputs(onnx_path)
    kv_retained = {name for name in retained if name.startswith(("past_key.", "past_value."))}
    assert kv_retained
    assert all(name.endswith("_RetainedState") and "_VLLM_" not in name for name in kv_retained)
    # Inputs use the plain names.
    assert "past_key.0" in _kv_input_names(onnx_path)


@pytest.mark.llm_model
def test_causal_export_prefix_changes_hash_dir(tmp_path):
    """Prefixed and unprefixed exports must land in distinct hashed dirs (no cache collision)."""
    model_id = CAUSAL_RUNTIME_MODEL_IDS["gpt2"]
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf.eval()

    plain_path = _exported_onnx_path(QEFFAutoModelForCausalLM(model_hf).export(tmp_path / "p"))

    model_hf2 = AutoModelForCausalLM.from_pretrained(
        model_id, **MODEL_KWARGS, low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model_hf2.eval()
    prefixed_path = _exported_onnx_path(
        QEFFAutoModelForCausalLM(model_hf2).export(tmp_path / "p", kv_cache_prefix="VLLM")
    )

    assert plain_path.parent != prefixed_path.parent


@pytest.mark.llm_model
def test_causal_compile_custom_io_carries_prefix(tmp_path, monkeypatch):
    """The compile custom_io must pair prefixed input/output KV buffers."""
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            CAUSAL_RUNTIME_MODEL_IDS["llama"],
            num_hidden_layers=2,
            **MODEL_KWARGS,
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, CAUSAL_RUNTIME_MODEL_IDS["llama"])

    captured = {}

    def fake_compile(**kwargs):
        captured.update(kwargs)
        return tmp_path / "qpc"

    monkeypatch.setattr(qeff_model, "_compile", fake_compile)
    qeff_model.compile(prefill_seq_len=8, ctx_len=32, compile_dir=str(tmp_path), kv_cache_prefix="VLLM")

    custom_io = captured["custom_io"]
    assert custom_io, "expected non-empty custom_io"
    assert "past_key.0_VLLM" in custom_io  # input buffer
    assert "past_key.0_VLLM_RetainedState" in custom_io  # paired retained output
    assert all("_VLLM" in k for k in custom_io if k.startswith(("past_key.", "past_value.")))
    assert captured.get("kv_cache_prefix") == "VLLM"

    # Critical safety check: kv_cache_prefix must NOT appear in compiler_options.
    # The _compile signature has kv_cache_prefix as an explicit named param so Python never places
    # it in **compiler_options — if it did, the compiler would see "-kv-cache-prefix=VLLM" and fail.
    # We verify by reconstructing the known explicit params and confirming the remainder (what would
    # become **compiler_options in the real _compile) does not contain kv_cache_prefix.
    _known_explicit_params = {
        "onnx_path",
        "compile_dir",
        "mxint8_kv_cache",
        "specializations",
        "custom_io",
        "mdp_ts_num_devices",
        "num_speculative_tokens",
        "enable_qnn",
        "qnn_config",
        "use_onnx_subfunctions",
        "prefill_only",
        "offload_pt_weights",
        "enable_chunking",
        "retain_full_kv",
        "qaic_config",
        "specialization_module_name",
        "kv_cache_prefix",
        "retained_state",
        "convert_to_fp16",
        "mxfp6_matmul",
        # compile-time args added by causal compile():
        "aic_num_cores",
    }
    implicit_compiler_options = {k: v for k, v in captured.items() if k not in _known_explicit_params}
    assert "kv_cache_prefix" not in implicit_compiler_options, (
        "kv_cache_prefix leaked into compiler_options — would produce an invalid compiler flag"
    )


@pytest.mark.llm_model
def test_vlm_export_prefix_lang_only(tmp_path):
    """VLM export with prefix: lang KV buffers prefixed, vision retained buffers untouched."""
    try:
        vlm_model = QEFFAutoModelForImageTextToText.from_pretrained(
            VLM_TEXT_RUNTIME_MODEL_ID, trust_remote_code=True, kv_offload=True
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, VLM_TEXT_RUNTIME_MODEL_ID)

    vlm_model.export(tmp_path / "vlm-prefixed", kv_cache_prefix="VLLM")
    lang_onnx = Path(vlm_model.lang_model.onnx_path)
    retained = _retained_state_outputs(lang_onnx)

    kv_retained = {name for name in retained if name.startswith(("past_key.", "past_value."))}
    assert kv_retained
    assert all(name.endswith("_VLLM_RetainedState") for name in kv_retained)

    # Vision/multimodal retained buffers on the lang graph must NOT be prefixed.
    for name in retained:
        if name.startswith(("vision_embeds", "pixel_values", "deepstack_features")):
            assert "_VLLM_" not in name


def _tiny_qwen3_moe_causal():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    import QEfficient

    cfg = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=64,
        decoder_sparse_step=1,
        norm_topk_prob=True,
    )
    cfg.torch_dtype = "float32"
    torch.manual_seed(0)
    hf = Qwen3MoeForCausalLM(cfg).eval()
    return QEfficient.QEFFAutoModelForCausalLM(hf, continuous_batching=False), cfg.num_hidden_layers


def test_kimi_k25_get_specializations_supports_multi_resolution_grid_sizes():
    """Kimi K2.5 accepts list-valued image sizes for multi-resolution specs."""
    from types import SimpleNamespace

    from QEfficient.transformers.models.kimi_k25.modeling_kimi_k25 import QEffKimiK25ForConditionalGeneration

    model = QEffKimiK25ForConditionalGeneration.__new__(QEffKimiK25ForConditionalGeneration)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14, merge_kernel_size=(2, 2)))

    specs, _ = model.get_specializations(
        batch_size=1,
        prefill_seq_len=64,
        ctx_len=4096,
        image_height=[512, 448],
        image_width=[910, 448],
        num_frames=[1, 1],
        kv_offload=True,
    )
    assert specs["vision"] == [
        {"num_patches": 2508, "grid_h": 38, "grid_w": 66, "num_image_tokens": 627},
        {"num_patches": 1024, "grid_h": 32, "grid_w": 32, "num_image_tokens": 256},
    ]
    assert all(spec["num_image_tokens"] == 627 for spec in specs["lang"])

    with pytest.raises(ValueError, match="image_height and image_width"):
        model.get_specializations(
            batch_size=1,
            prefill_seq_len=64,
            ctx_len=4096,
            h=[30, 32],
            w=[80, 64],
            num_frames=[1, 2],
            kv_offload=True,
        )

    with pytest.raises(ValueError, match="num_patches"):
        model.get_specializations(
            batch_size=1,
            prefill_seq_len=64,
            ctx_len=4096,
            image_height=512,
            image_width=910,
            num_patches=2508,
            kv_offload=True,
        )
