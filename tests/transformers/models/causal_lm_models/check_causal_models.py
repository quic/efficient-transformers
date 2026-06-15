# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import copy
import os
from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils.test_utils import ModelConfig, load_hf_causal_lm_model

from ..check_model_results import dump_and_compare_results


def _tiny_lane_active() -> bool:
    """True under per-PR profile (random-weight tinies). Token-equality on
    flat logits is an unreliable proxy here — fp16↔fp32 ULP drift flips argmax
    constantly. Real-weight runs (full_layers_model / unset) keep strict checks.
    """
    return os.environ.get("QEFF_TEST_PROFILE", "").strip() in {"dummy_layers_model", "few_layers_model"}


def _tokens_match_or_first_token(reference, candidate, *, label: str, length: int = 24, vlm: bool = False) -> None:
    """Token-equality check that softens to first-token-match under the tiny lane.

    Why: under random weights the post-softmax distribution is nearly flat, so
    a fp16↔fp32 ULP drift in any later step flips argmax — a false-positive bug
    signal. The first token (right after prefill, with the largest logit margin)
    still carries enough signal to catch real export/compile breakage.

    The ``vlm`` parameter is currently treated identically to non-vlm: VLM
    relaxation produced regressions (long-running generation, additional
    runtime failures further into the decode loop), so we keep the strict
    first-token check until per-VLM mitigations are in.

    Under full_layers_model (nightly) this is the original ``[:length]`` equality.
    """
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if _tiny_lane_active():
        # Require the first generated token to match. Catches export/compile
        # breakage (whole logit distribution shifted) without flagging fp drift.
        assert ref.shape[-1] >= 1 and cand.shape[-1] >= 1, f"{label}: empty token sequence"
        assert (ref[..., :1] == cand[..., :1]).all(), (
            f"{label}: first-token mismatch under tiny lane (ref={ref[..., :1].tolist()} cand={cand[..., :1].tolist()})"
        )
        return
    assert (ref[..., :length] == cand[..., :length]).all(), label


def get_custom_n_layers(model_name):
    """
    Function to set number layers of the variuos types of models such as swiftkv models and others
    --------

    :model_name: str

    :return n_layer
    """
    if model_name in {"microsoft/Phi-3-mini-4k-instruct", "neuralmagic/Qwen2-0.5B-Instruct-FP8", "openai/gpt-oss-20b"}:
        return 2
    elif model_name in ModelConfig.SWIFTKV_MODELS:
        return -1
    return 1


def check_causal_lm_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    manual_cleanup: callable,
    num_devices: int = 1,
    continuous_batching: bool = False,
    prompt_len: int = Constants.PROMPT_LEN,
    ctx_len: int = Constants.CTX_LEN,
    n_layer: int = -1,
    num_speculative_tokens: Optional[int] = None,
    prefill_only: Optional[bool] = None,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    config: Optional[AutoConfig] = None,
    pytorch_hf_tokens: Optional[list] = None,
    qaic_config: Optional[dict] = None,
    retain_full_kv: Optional[bool] = None,
    compare_results: bool = False,
):
    torch.manual_seed(42)
    replace_transformers_quantizers()
    model_hf = load_hf_causal_lm_model(model_name, num_hidden_layers=n_layer, config=config)
    tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
    config = model_hf.config
    batch_size = len(Constants.INPUT_STR)
    prompts = Constants.INPUT_STR * 4 if continuous_batching else Constants.INPUT_STR
    full_batch_size = 4
    gen_len = 24
    is_tlm = False if num_speculative_tokens is None else True
    pytorch_hf_tokens = None
    pytorch_kv_tokens = None
    ort_tokens = None

    api_runner = ApiRunner(
        batch_size,
        tokenizer,
        config,
        prompts,
        Constants.PROMPT_LEN,
        Constants.CTX_LEN,
        full_batch_size if continuous_batching else None,
    )
    qeff_model = QEFFAutoModelForCausalLM(
        copy.deepcopy(model_hf),
        is_tlm=is_tlm,
        pretrained_model_name_or_path=model_name,
        continuous_batching=continuous_batching,
        qaic_config=qaic_config,
    )
    qeff_model.transform(
        ctx_len=ctx_len,
        seq_len=prompt_len,
        batch_size=full_batch_size if continuous_batching else batch_size,
        num_devices=num_devices,
        qaic_config=qaic_config,
    )
    if continuous_batching is False:
        pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    if model_name not in ModelConfig.SWIFTKV_MODELS and model_name not in ModelConfig.EXTERNAL_MODELS:
        if continuous_batching:
            pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch_CB(model_hf)
            pytorch_hf_tokens = np.vstack(pytorch_hf_tokens)
        else:
            pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    onnx_model_path = qeff_model.export()
    if continuous_batching is False:
        ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=is_tlm)
        gen_len = ort_tokens.shape[-1]

    if pytorch_hf_tokens is not None and ort_tokens is not None:
        assert (pytorch_hf_tokens == ort_tokens).all(), (
            "Tokens don't match for HF PyTorch model output and ONNXRT output."
        )

    if pytorch_kv_tokens is not None and ort_tokens is not None:
        assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    compiler_options = {}
    if continuous_batching and prompt_len == 1:
        prefill_spec = {
            "batch_size": batch_size,
            "seq_len": 1,
            "ctx_len": ctx_len,
            "full_batch_size": full_batch_size,
            "sliding_window": 128,
        }
        decode_spec = {
            "batch_size": full_batch_size,
            "seq_len": 1,
            "ctx_len": ctx_len,
            "full_batch_size": full_batch_size,
            "sliding_window": 128,
        }
        compiler_options["specializations"] = [prefill_spec, decode_spec]

    qpc_path = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_devices=num_devices,
        mxfp6=False,
        aic_enable_depth_first=False,
        num_speculative_tokens=num_speculative_tokens,
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
        retain_full_kv=retain_full_kv,
        prefill_only=prefill_only,
        batch_size=batch_size if continuous_batching else 1,
        full_batch_size=full_batch_size if continuous_batching else None,
        **compiler_options,
    )
    assert os.path.isfile(os.path.join(os.path.dirname(qpc_path), "qconfig.json"))

    # Generate
    exec_info = qeff_model.generate(tokenizer, prompts=prompts)

    if continuous_batching:
        cloud_ai_100_tokens = exec_info.generated_ids
        if cloud_ai_100_tokens is not None and ort_tokens is not None:
            for ort_token, cloud_token in zip(ort_tokens, cloud_ai_100_tokens):
                _tokens_match_or_first_token(
                    ort_token,
                    cloud_token,
                    label="Tokens don't match for ONNXRT output and Cloud AI 100 output.",
                )
        if pytorch_hf_tokens is not None and cloud_ai_100_tokens is not None:
            for pt_token, cloud_token in zip(pytorch_hf_tokens, cloud_ai_100_tokens):
                _tokens_match_or_first_token(
                    pt_token,
                    cloud_token,
                    label="Tokens don't match for HF PyTorch model output and Cloud AI 100 output.",
                )
    else:
        cloud_ai_100_tokens = exec_info.generated_ids[0][:, :gen_len]
        if prefill_only:
            assert (ort_tokens[0][0] == cloud_ai_100_tokens[0][0]).all(), (
                "prefill run output tokens don't match for ONNXRT output and Cloud AI 100 output."
            )
        else:
            _tokens_match_or_first_token(
                ort_tokens,
                cloud_ai_100_tokens,
                label="Tokens don't match for ONNXRT output and Cloud AI 100 output.",
            )

    manual_cleanup(onnx_model_path)  # Clean up the model files after the tests are done.
    if compare_results is False:
        return
    # Compare results for full model only.
    compile_params = {
        "prefill_seq_len": prompt_len,
        "ctx_len": ctx_len,
        "num_devices": num_devices,
        "mxfp6": False,
        "aic_enable_depth_first": False,
        "num_speculative_tokens": num_speculative_tokens,
        "enable_qnn": enable_qnn,
        "qnn_config": qnn_config,
        "retain_full_kv": retain_full_kv,
        "prefill_only": prefill_only,
        "batch_size": batch_size if continuous_batching else 1,
        "full_batch_size": full_batch_size if continuous_batching else None,
        "compiler_options": compiler_options,
    }
    assert dump_and_compare_results(
        model_name,
        compile_params,
        "causal_lm_model_results.json",
        cloud_ai_100_tokens,
        exec_info,
        pytorch_hf_tokens,
        pytorch_kv_tokens,
        ort_tokens,
    )
