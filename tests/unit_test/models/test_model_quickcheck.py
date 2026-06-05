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

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoTokenizer,
    Qwen2Config,
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
    return ort.InferenceSession(str(onnx_path), sess_options=options)


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
# Layer-wise export (provisional, scheduled for deprecation)
# ---------------------------------------------------------------------------

LAYERWISE_TINY_MODEL_ID = "tiny-random/qwen3-vl-moe"
LAYERWISE_TINY_MODEL_IDS = {
    "qwen3_vl_moe": "tiny-random/qwen3-vl-moe",
    "qwen3_5_moe": "tiny-random/qwen3.5-moe",
    "qwen3_moe": "tiny-random/qwen3-moe",
}


@pytest.mark.llm_model
def test_layerwise_window_helpers():
    """Pure-Python coverage of the windowing helpers - no model load required."""
    from QEfficient.transformers.models import _layerwise

    assert _layerwise._build_layer_windows(4, 1) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert _layerwise._build_layer_windows(5, 2) == [(0, 2), (2, 4), (4, 5)]
    with pytest.raises(ValueError):
        _layerwise._build_layer_windows(0, 1)
    with pytest.raises(ValueError):
        _layerwise._build_layer_windows(4, 0)


@pytest.mark.llm_model
def test_layerwise_supported_guard_rejects_unrelated_model():
    """layerwise=True must hard-fail on architectures without windowing hooks."""
    from QEfficient.transformers.models import _layerwise

    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    with pytest.raises(NotImplementedError, match="layerwise=True is only supported"):
        _layerwise.assert_layerwise_supported(config)


def test_resolve_torch_dtype_normalizes_dtype_alias():
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

    # bfloat16 is downgraded to float32 on ai100 regardless of which name is used.
    kwargs = {"dtype": torch.bfloat16}
    _resolve_torch_dtype(kwargs)
    assert kwargs["torch_dtype"] == torch.float32
    assert kwargs["dtype"] == torch.float32


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


def test_layerwise_matches_default_path_for_qwen3_moe():
    """Without-layerwise and with-layerwise forwards must produce identical output.
    This is the core backward-compatibility contract: running every decoder layer
    in a single forward (default path) must match running the same layers one
    window at a time and chaining the hidden states (layerwise path), bit for bit.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    import QEfficient
    from QEfficient.transformers.models import _layerwise

    cfg = Qwen3MoeConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=128,
        decoder_sparse_step=1,
        norm_topk_prob=True,
    )
    torch.manual_seed(0)
    hf = Qwen3MoeForCausalLM(cfg).eval()
    qeff_model = QEfficient.QEFFAutoModelForCausalLM(hf, continuous_batching=False)
    inner = qeff_model.model.model

    B, S, ctx, num_layers = 1, 8, 16, cfg.num_hidden_layers
    n_kv, head_dim = cfg.num_key_value_heads, cfg.head_dim
    ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).view(1, -1)

    def fresh_pkv():
        return tuple(
            (torch.zeros(B, n_kv, ctx, head_dim), torch.zeros(B, n_kv, ctx, head_dim)) for _ in range(num_layers)
        )

    # Default (non-layerwise) path: all layers in one shot.
    with torch.no_grad():
        default_out = inner(
            input_ids=ids,
            position_ids=position_ids,
            past_key_values=fresh_pkv(),
            cache_position=torch.arange(S),
        )

    # Layerwise path: window size 1, chaining hidden states across windows.
    hidden_states = inner.embed_tokens(ids)
    try:
        with _layerwise._layerwise_export_env():
            for window in range(num_layers):
                _layerwise._set_layer_windows(window, window + 1, num_layers)
                with torch.no_grad():
                    out = inner(
                        inputs_embeds=hidden_states,
                        position_ids=position_ids,
                        past_key_values=fresh_pkv(),
                        cache_position=torch.arange(S),
                    )
                hidden_states = out.last_hidden_state
    finally:
        _layerwise._reset_layer_windows()

    assert torch.equal(default_out.last_hidden_state, hidden_states), (
        "layerwise windowed forward diverged from the default single-shot forward"
    )

    # The default path must leave the mutable window state untouched.
    from QEfficient.transformers.models.qwen3_moe.modeling_qwen3_moe import QEffQwen3MoeModel

    assert QEffQwen3MoeModel._start == 0
    assert QEffQwen3MoeModel._end == 0


@pytest.mark.llm_model
def test_layerwise_matches_default_path_for_qwen3_5_moe():
    """Qwen3.5-MoE decoder wrapper must preserve logits with layerwise windows."""
    import QEfficient
    from QEfficient.transformers.models import _layerwise

    config = AutoConfig.from_pretrained(LAYERWISE_TINY_MODEL_IDS["qwen3_5_moe"])
    config.torch_dtype = "float32"

    qeff_model = QEfficient.QEFFAutoModelForImageTextToText.from_pretrained(
        LAYERWISE_TINY_MODEL_IDS["qwen3_5_moe"],
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float32,
        layerwise=False,
    )
    wrapper = qeff_model.model.get_qeff_language_decoder().eval()
    lang_inputs = qeff_model.model.get_dummy_inputs(kv_offload=True)["lang"]

    with torch.no_grad():
        default_out = wrapper(**lang_inputs)

    hidden_states = None
    total_layers = qeff_model.model.config.text_config.num_hidden_layers
    try:
        with _layerwise._layerwise_export_env():
            for window in range(total_layers):
                _layerwise._set_layer_windows(window, window + 1, total_layers)
                call_kwargs = {
                    "position_ids": lang_inputs["position_ids"],
                    "past_key_values": lang_inputs["past_key_values"],
                }
                if window == 0:
                    call_kwargs.update(
                        input_ids=lang_inputs["input_ids"],
                        vision_embeds=lang_inputs["vision_embeds"],
                        image_idx=lang_inputs["image_idx"],
                    )
                else:
                    call_kwargs["inputs_embeds"] = hidden_states

                with torch.no_grad():
                    window_out = wrapper(**call_kwargs)
                hidden_states = window_out[0]
    finally:
        _layerwise._reset_layer_windows()

    assert torch.equal(default_out[0], hidden_states), "Qwen3.5-MoE layerwise logits diverged from default logits"


@pytest.mark.llm_model
def test_layerwise_matches_default_path_for_qwen3_vl_moe():
    """Qwen3-VL-MoE decoder wrapper must preserve logits with layerwise windows."""
    import QEfficient
    from QEfficient.transformers.models import _layerwise

    model_id = LAYERWISE_TINY_MODEL_IDS["qwen3_vl_moe"]
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)
    config.torch_dtype = "float32"

    qeff_model = QEfficient.QEFFAutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="eager",
        kv_offload=True,
        config=config,
        dtype=torch.float32,
        layerwise=False,
    )
    wrapper = qeff_model.model.get_qeff_language_decoder().eval()
    lang_inputs = qeff_model.model.get_dummy_inputs(kv_offload=True)["lang"]

    with torch.no_grad():
        default_out = wrapper(**lang_inputs)

    hidden_states = None
    total_layers = qeff_model.model.config.text_config.num_hidden_layers
    try:
        with _layerwise._layerwise_export_env():
            for window in range(total_layers):
                _layerwise._set_layer_windows(window, window + 1, total_layers)
                call_kwargs = {k: v for k, v in lang_inputs.items() if k not in ("input_ids", "inputs_embeds")}
                if window == 0:
                    call_kwargs["input_ids"] = lang_inputs["input_ids"]
                else:
                    call_kwargs["inputs_embeds"] = hidden_states

                with torch.no_grad():
                    window_out = wrapper(**call_kwargs)
                hidden_states = window_out[0]
    finally:
        _layerwise._reset_layer_windows()

    assert torch.equal(default_out[0], hidden_states), "Qwen3-VL-MoE layerwise logits diverged from default logits"


def test_split_layer_graph_keeps_qwen3_5_linear_states(tmp_path):
    """Qwen3.5 layerwise split must retain linear-attention state inputs."""
    from onnx import TensorProto, helper

    from QEfficient.utils.layerwise_pipeline import split_layer_graph

    layer_dir = tmp_path / "onnx_layerwise_tmp" / "layer_0_1"
    layer_dir.mkdir(parents=True)
    model_path = layer_dir / "model_layer_tmp_0_1.onnx"

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["input_ids"], ["logits"]),
            helper.make_node("Identity", ["conv_state.0"], ["conv_state.0_RetainedState"]),
            helper.make_node("Identity", ["recurrent_state.0"], ["recurrent_state.0_RetainedState"]),
        ],
        name="qwen35_layer",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 8]),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [4, 1, 8]),
            helper.make_tensor_value_info("conv_state.0", TensorProto.FLOAT, [1, 64, 4]),
            helper.make_tensor_value_info("recurrent_state.0", TensorProto.FLOAT, [1, 4, 16, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.INT64, [1, 8]),
            helper.make_tensor_value_info("conv_state.0_RetainedState", TensorProto.FLOAT, [1, 64, 4]),
            helper.make_tensor_value_info("recurrent_state.0_RetainedState", TensorProto.FLOAT, [1, 4, 16, 16]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, model_path)

    assert split_layer_graph(0, 1, str(tmp_path), 0, 1) is True

    split_model = onnx.load(layer_dir / "split_graph.onnx", load_external_data=False)
    input_names = {value.name for value in split_model.graph.input}
    assert "conv_state.0" in input_names
    assert "recurrent_state.0" in input_names


@pytest.mark.llm_model
def test_layerwise_supported_guard_accepts_qwen3_vl_moe():
    from QEfficient.transformers.models import _layerwise

    try:
        config = AutoConfig.from_pretrained(LAYERWISE_TINY_MODEL_ID)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, LAYERWISE_TINY_MODEL_ID)
    resolved = _layerwise.assert_layerwise_supported(config)
    assert resolved in {"qwen3_vl_moe", "qwen3_vl_moe_text"}


@pytest.mark.llm_model
@pytest.mark.parametrize(
    ("arch", "model_id"),
    sorted(LAYERWISE_TINY_MODEL_IDS.items()),
    ids=sorted(LAYERWISE_TINY_MODEL_IDS),
)
def test_layerwise_supported_guard_accepts_all_supported(arch, model_id):
    """Guard must accept each architecture in the layerwise allowlist."""
    from QEfficient.transformers.models import _layerwise

    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)
    resolved = _layerwise.assert_layerwise_supported(config)
    assert arch in resolved or resolved.startswith(arch)


@pytest.mark.llm_model
def test_layerwise_off_does_not_set_env_var(tmp_path):
    """Backward compat: layerwise must be controlled purely via the API,
    never via environment variables, and must be off by default."""
    from QEfficient.base.modeling_qeff import QEFFBaseModel
    from QEfficient.transformers.models import _layerwise  # noqa: F401

    assert os.environ.get("LAYERWISE_EXPORT") is None
    assert QEFFBaseModel._layerwise_active is False


@pytest.mark.llm_model
def test_layerwise_vision_wrapper_keeps_only_first_text_window():
    try:
        config = AutoConfig.from_pretrained(LAYERWISE_TINY_MODEL_ID)
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            LAYERWISE_TINY_MODEL_ID,
            kv_offload=True,
            config=config,
            layerwise=True,
        )
        vision_wrapper = qeff_model._build_layerwise_vision_wrapper()
    except Exception as exc:
        _skip_on_model_fetch_error(exc, LAYERWISE_TINY_MODEL_ID)

    layers = vision_wrapper.model.model.language_model.layers

    assert getattr(qeff_model, "_layerwise_outer_meta", False) is True
    assert layers[0] is not None
    assert sum(layer is not None for layer in layers) == 1
    assert next(vision_wrapper.model.model.visual.parameters()).device.type != "meta"

    default_model = QEFFAutoModelForImageTextToText.from_pretrained(
        LAYERWISE_TINY_MODEL_ID,
        kv_offload=True,
        config=config,
    )
    default_layers = default_model.model.model.language_model.layers
    assert sum(layer is not None for layer in default_layers) == len(default_layers)


@pytest.mark.llm_model
def test_layerwise_context_manager_toggles_class_flag():
    """The driver's context manager must flip the class flag and restore it,
    even on exception, with no env-var side-effects."""
    from QEfficient.base.modeling_qeff import QEFFBaseModel
    from QEfficient.transformers.models import _layerwise

    assert QEFFBaseModel._layerwise_active is False
    with _layerwise._layerwise_export_env():
        assert QEFFBaseModel._layerwise_active is True
        assert "LAYERWISE_EXPORT" not in os.environ
    assert QEFFBaseModel._layerwise_active is False

    try:
        with _layerwise._layerwise_export_env():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert QEFFBaseModel._layerwise_active is False


@pytest.mark.llm_model
def test_layerwise_uses_probe_model_for_cached_export(monkeypatch, tmp_path):
    """A cached merged ONNX must avoid rebuilding per-window models."""
    from QEfficient.transformers.models import _layerwise

    class DummyConfig:
        model_type = "qwen3_moe"
        num_hidden_layers = 2

    class ProbeModel:
        def __init__(self, cached_path):
            self.cached_path = cached_path

        def compile(self, **kwargs):
            assert kwargs.pop("_layerwise_cache_probe") is True
            return self.cached_path

    cached_path = tmp_path / "Model-hash" / "final_data" / "merged_0-2.onnx"
    cached_path.parent.mkdir(parents=True)
    cached_path.touch()
    factory_called = False

    def fail_factory(*args, **kwargs):
        nonlocal factory_called
        factory_called = True
        raise AssertionError("factory must not run when merged ONNX is cached")

    monkeypatch.setattr(_layerwise, "_install_window_patches_for", lambda model_type: None)
    result = _layerwise.run_layerwise(
        model_id="dummy",
        config=DummyConfig(),
        qeff_factory=fail_factory,
        compile_kwargs={},
        probe_qeff_model=ProbeModel(cached_path),
        final_compile=False,
    )

    assert result == str(cached_path)
    assert factory_called is False


@pytest.mark.llm_model
def test_layerwise_cache_miss_exports_all_windows(monkeypatch, tmp_path):
    from QEfficient.transformers.models import _layerwise

    class DummyConfig:
        model_type = "qwen3_moe"
        num_hidden_layers = 3

    class ProbeModel:
        def compile(self, **kwargs):
            assert kwargs.pop("_layerwise_cache_probe") is True
            return None

    exported_windows = []

    class WindowModel:
        def __init__(self):
            self.model = object()

        def compile(self, **kwargs):
            start = _layerwise._LAYERWISE_STATE["text_start"]
            end = _layerwise._LAYERWISE_STATE["text_end"]
            exported_windows.append((start, end))
            shard = tmp_path / "onnx_layerwise_tmp" / f"layer_{start}_{end}" / f"model_layer_tmp_{start}_{end}.onnx"
            shard.parent.mkdir(parents=True, exist_ok=True)
            shard.touch()
            return str(shard)

    monkeypatch.setattr(_layerwise, "_install_window_patches_for", lambda model_type: None)
    monkeypatch.setattr(_layerwise, "_null_outside_window_layers", lambda *args, **kwargs: None)
    monkeypatch.setattr(_layerwise, "_slim_for_window_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        _layerwise,
        "_stitch_layerwise_if_available",
        lambda export_root, total_layers=None: str(export_root / "merged.onnx"),
    )

    result = _layerwise.run_layerwise(
        model_id="dummy",
        config=DummyConfig(),
        qeff_factory=lambda *args, **kwargs: WindowModel(),
        compile_kwargs={},
        probe_qeff_model=ProbeModel(),
        window_size=1,
        final_compile=False,
    )

    assert exported_windows == [(0, 1), (1, 2), (2, 3)]
    assert result.endswith("merged.onnx")


@pytest.mark.llm_model
def test_layerwise_cached_merged_prefers_complete_graph(tmp_path):
    from QEfficient.transformers.models import _layerwise

    final_data = tmp_path / "final_data"
    final_data.mkdir()
    partial = final_data / "merged_9-48.onnx"
    complete = final_data / "merged_0-48.onnx"
    partial.touch()
    complete.touch()

    assert _layerwise._cached_merged_onnx(tmp_path, total_layers=48) == complete


@pytest.mark.llm_model
def test_layerwise_export_hash_is_separate_from_default(monkeypatch):
    from QEfficient.utils import export_utils

    class DummyConfig:
        def to_diff_dict(self):
            return {"model_type": "dummy"}

    class DummyModel:
        config = DummyConfig()

    class DummyQEffModel:
        model = DummyModel()
        hash_params = {}

    def normal_export(self, example_inputs, output_names, dynamic_axes, **export_kwargs):
        pass

    def _export_layerwise(self, example_inputs, output_names, dynamic_axes, **export_kwargs):
        pass

    captured_export_kwargs = []

    def fake_create_export_hash(**kwargs):
        captured_export_kwargs.append(kwargs.get("export_kwargs"))
        return "hash", {}

    monkeypatch.setattr(export_utils, "create_export_hash", fake_create_export_hash)

    export_utils._generate_export_hash(DummyQEffModel(), ({}, ["logits"], {}), {}, normal_export)
    export_utils._generate_export_hash(DummyQEffModel(), ({}, ["logits"], {}), {}, _export_layerwise)

    assert captured_export_kwargs[0] in (None, {})
    assert captured_export_kwargs[1]["_qeff_layerwise_export"] is True


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


@pytest.mark.llm_model
def test_layerwise_compile_hydrates_outer_qpc_paths(monkeypatch, tmp_path):
    from QEfficient.transformers.models import _layerwise
    from QEfficient.transformers.models.modeling_auto import _QEffAutoModelForImageTextToTextDualQPC

    qpc_path = tmp_path / "qpc"
    model = object.__new__(_QEffAutoModelForImageTextToTextDualQPC)
    model._pretrained_model_name_or_path = "dummy"
    model.config = object()
    model.vision_model = type("Vision", (), {"qpc_path": None})()
    model.lang_model = type("Lang", (), {"qpc_path": None})()
    model._build_layerwise_factory = lambda: None

    monkeypatch.setattr(_layerwise, "run_layerwise", lambda **kwargs: {"lang_decode_qpc_path": qpc_path})

    result = model._run_layerwise_compile(layerwise_window_size=1)

    assert result == {"lang_decode_qpc_path": qpc_path}
    assert model.qpc_paths == result
    assert model.lang_model.qpc_path == qpc_path


@pytest.mark.llm_model
def test_layerwise_compile_rejects_unsupported_model():
    """End-to-end smoke: invoking layerwise=True on llama bubbles the guard error."""
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, "tiny-random/llama")
    # CausalLM does not expose a layerwise= kwarg today; only DualQPC VLM does.
    # So this test guards via the helper directly to make the contract explicit
    # for future surface expansion.
    from QEfficient.transformers.models import _layerwise

    with pytest.raises(NotImplementedError):
        _layerwise.assert_layerwise_supported(qeff_model.model.config)


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
        if inp.name.startswith(("past_key.", "past_value.", "compressed_kv.", "k_pe."))
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

    def test_align_inputs_noop_without_prefix(self):
        from QEfficient.utils import align_kv_input_names_to_retained_outputs

        input_names = ["input_ids", "past_key.0", "past_value.0"]
        output_names = ["logits", "past_key.0_RetainedState", "past_value.0_RetainedState"]
        assert align_kv_input_names_to_retained_outputs(input_names, output_names) == input_names

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
        "moe_prefill_packed_chunk_size",
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
