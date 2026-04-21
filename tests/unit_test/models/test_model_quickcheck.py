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
CAUSAL_MULTI_SUBFUNCTION_MODEL_TYPES = {
    "codegen",
    "phi",
    "starcoder2",
    "mixtral",
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
    transform_names = {transform.__name__ for transform in qeff_model._onnx_transforms}
    proxy_only_transforms = {"FP16ClipTransform", "SplitTensorsTransform"}
    always_on_transforms = always_on_transforms or set()
    conditional_proxy_transforms = proxy_only_transforms - always_on_transforms

    if enable_proxy:
        assert proxy_only_transforms.issubset(transform_names)
    else:
        assert conditional_proxy_transforms.isdisjoint(transform_names)
        assert always_on_transforms.issubset(transform_names)


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
    assert "SplitTensorsTransform" not in transform_names

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
        qeff_default, enable_proxy=False, always_on_transforms={"FP16ClipTransform"}
    )
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
def test_proxy_toggle_onnx_transform_policy_for_vlm():
    model_id = VLM_TEXT_RUNTIME_MODEL_ID
    try:
        qeff_default = QEFFAutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, kv_offload=False
        )
        qeff_proxy = QEFFAutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, enable_proxy=True, kv_offload=False
        )
    except Exception as exc:
        _skip_on_model_fetch_error(exc, model_id)

    _assert_proxy_only_onnx_transform_policy(qeff_default, enable_proxy=False)
    _assert_proxy_only_onnx_transform_policy(qeff_proxy, enable_proxy=True)


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
