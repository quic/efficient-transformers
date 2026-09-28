# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import AutoModelForCausalLM

from QEfficient.generation import cloud_infer
from QEfficient.generation.text_generation_inference import TextGeneration
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import hf_download
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants

configs = [pytest.param("gpt2", 2, None, 32, id="gpt2_config")]


@pytest.fixture
def profiling_runtime(monkeypatch):
    """Provide a runtime whose profiling enum predates device KPI support."""
    runtime = SimpleNamespace(
        Context=Mock(),
        Queue=Mock(),
        Qpc=Mock(return_value=SimpleNamespace(getIoDescriptor=Mock(return_value=(0, b"")))),
        QAicProgramProperties=SimpleNamespace,
        Program=Mock(return_value=SimpleNamespace(load=Mock(return_value=0), activate=Mock())),
        ExecObj=Mock(),
        BufferDimensionsVecRef=list,
        QStatus=SimpleNamespace(QS_SUCCESS=0),
        QAicProfilingTypeEnum=SimpleNamespace(
            QAIC_PROFILING_INFERENCE_LATENCY_TYPE=1,
            QAIC_PROFILING_INFERENCE_TRACE_TYPE=2,
            QAIC_PROFILING_INFERENCE_RAW_DEVICE_STATS_TYPE=3,
        ),
        ProfilingHandle=Mock(return_value=SimpleNamespace(start=Mock(return_value=0), stop=Mock(return_value=0))),
    )
    dtype_names = (
        "FLOAT_TYPE",
        "FLOAT_16_TYPE",
        "INT8_Q_TYPE",
        "UINT8_Q_TYPE",
        "INT16_Q_TYPE",
        "INT32_Q_TYPE",
        "INT32_I_TYPE",
        "INT64_I_TYPE",
        "INT8_TYPE",
    )
    aicapi = SimpleNamespace(
        **{name: index for index, name in enumerate(dtype_names)},
        IoDesc=Mock(
            return_value=SimpleNamespace(
                ParseFromString=Mock(), allowed_shapes=[], selected_set=SimpleNamespace(bindings=[])
            )
        ),
    )
    monkeypatch.setattr(cloud_infer, "qaicrt", runtime, raising=False)
    monkeypatch.setattr(cloud_infer, "aicapi", aicapi, raising=False)
    monkeypatch.setattr(cloud_infer, "is_qaicrt_imported", True)
    monkeypatch.setattr(cloud_infer, "is_aicapi_imported", True, raising=False)
    return runtime


@pytest.mark.parametrize("has_profiling_api", [True, False])
def test_session_profiling_disabled_on_older_sdk(profiling_runtime, tmp_path, has_profiling_api):
    if not has_profiling_api:
        del profiling_runtime.QAicProfilingTypeEnum
        del profiling_runtime.ProfilingHandle

    session = cloud_infer.QAICInferenceSession(tmp_path / "model.qpc")

    assert session.profiling_handle is None
    assert session.is_active
    profiling_runtime.ExecObj.assert_called_once_with(session.context, session.program)
    if has_profiling_api:
        profiling_runtime.ProfilingHandle.assert_not_called()
    assert not (tmp_path / "profiling_output").exists()


@pytest.mark.parametrize(
    "profiling_type,enum_name,enum_value",
    [
        ("latency", "QAIC_PROFILING_INFERENCE_LATENCY_TYPE", 1),
        ("trace", "QAIC_PROFILING_INFERENCE_TRACE_TYPE", 2),
        ("raw_device_stats", "QAIC_PROFILING_INFERENCE_RAW_DEVICE_STATS_TYPE", 3),
        ("stats", "QAIC_PROFILING_INFERENCE_DEV_KPI_TYPE", 4),
    ],
)
def test_session_profiling_resolves_only_requested_mode(
    profiling_runtime, tmp_path, profiling_type, enum_name, enum_value
):
    profiling_runtime.QAicProfilingTypeEnum = SimpleNamespace(**{enum_name: enum_value})
    output_dir = tmp_path / "reports"

    session = cloud_infer.QAICInferenceSession(
        tmp_path / "model.qpc",
        profiling_type=profiling_type,
        profiling_output_dir=output_dir,
        profiling_file_prefix="test-profiling",
    )

    profiling_runtime.ProfilingHandle.assert_called_once_with(
        programs=[session.program],
        type=enum_value,
        fileNamePrefix="test-profiling",
        outputDirectory=str(output_dir),
    )
    assert output_dir.is_dir()
    with session.profile():
        session.profiling_handle.start.assert_called_once_with()
        session.profiling_handle.stop.assert_not_called()
    session.profiling_handle.stop.assert_called_once_with()


@pytest.mark.parametrize(
    "profiling_type,missing_api",
    [("stats", None), ("latency", "QAicProfilingTypeEnum"), ("latency", "ProfilingHandle")],
)
def test_session_profiling_unavailable_before_device_allocation(
    profiling_runtime, tmp_path, profiling_type, missing_api
):
    if missing_api is not None:
        delattr(profiling_runtime, missing_api)

    with pytest.raises(RuntimeError, match=f"profiling_type '{profiling_type}'.*installed QAIC SDK"):
        cloud_infer.QAICInferenceSession(tmp_path / "model.qpc", profiling_type=profiling_type)

    profiling_runtime.Context.assert_not_called()
    profiling_runtime.Qpc.assert_not_called()
    profiling_runtime.Program.assert_not_called()
    assert not (tmp_path / "profiling_output").exists()


def test_session_profiling_invalid_mode_before_device_allocation(profiling_runtime, tmp_path):
    del profiling_runtime.QAicProfilingTypeEnum
    del profiling_runtime.ProfilingHandle

    with pytest.raises(ValueError, match="Unsupported profiling_type 'invalid'"):
        cloud_infer.QAICInferenceSession(tmp_path / "model.qpc", profiling_type="invalid")

    profiling_runtime.Context.assert_not_called()
    profiling_runtime.Qpc.assert_not_called()
    profiling_runtime.Program.assert_not_called()


def load_causal_lm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=True,
        num_hidden_layers=model_config["n_layer"],
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


# Use @pytest.mark.parametrize to apply the configurations
@pytest.mark.on_qaic
@pytest.mark.llm_model
@pytest.mark.parametrize("model_name, n_layer, full_batch_size, max_gen_len", configs)
def test_generate_text_stream(
    model_name: str,
    n_layer: int,
    full_batch_size: int,
    max_gen_len: int,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :prompt_len (int): Prompt length for the model to compile.
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
    """
    model_config = {"model_name": model_name, "n_layer": n_layer}
    model_hf, _ = load_causal_lm_model(model_config)

    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)

    qeff_model = QEFFAutoModelForCausalLM(model_hf)

    qeff_model.export()

    qpc_path = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=14,
        mxfp6=False,
        aic_enable_depth_first=False,
        full_batch_size=full_batch_size,
    )

    exec_info = qeff_model.generate(tokenizer, prompts=Constants.INPUT_STR, generation_len=max_gen_len)
    cloud_ai_100_tokens = exec_info.generated_ids[0]  # Because we always run for single input and single batch size
    cloud_ai_100_output = [tokenizer.decode(token, skip_special_tokens=True) for token in cloud_ai_100_tokens[0]]

    text_generator = TextGeneration(
        tokenizer=tokenizer,
        qpc_path=qpc_path,
        ctx_len=ctx_len,
        full_batch_size=full_batch_size,
    )
    stream_tokens = []
    for decoded_tokens in text_generator.generate_stream_tokens(Constants.INPUT_STR, generation_len=max_gen_len):
        stream_tokens.extend(decoded_tokens)

    assert cloud_ai_100_output == stream_tokens, (
        f"Deviation in output observed while comparing regular execution and streamed output: {cloud_ai_100_output} != {stream_tokens}"
    )
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))
