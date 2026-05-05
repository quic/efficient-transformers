# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Standalone example for checking Gemma4 dense token parity between ONNX Runtime
and QAic.

This script:
1. Loads `tiny-random/gemma-4-dense` text weights from the multimodal HF model.
2. Exports the QEff text path with ONNX subfunctions enabled.
3. Compiles for QAic using the generated Gemma4 NPI automatically.
4. Runs greedy decode on ORT and QAic for the same prompt.
5. Prints token IDs, decoded text, and parity status.

Default prompt is the currently verified parity case: "hello world".
"""

import argparse
import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

MODEL_KWARGS = {"attn_implementation": "eager"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/gemma-4-31B-it")
    parser.add_argument("--prompt", default="hello world")
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--ctx-len", type=int, default=8)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device-group", nargs="+", type=int, default=None)
    parser.add_argument("--disable-npi", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def load_gemma4_text_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

    full_model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
        **MODEL_KWARGS,
    ).to(torch.float32)
    full_model.eval()

    text_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True).text_config
    text_config.num_hidden_layers = 6
    text_model = AutoModelForCausalLM.from_config(
        text_config,
        trust_remote_code=True,
        **MODEL_KWARGS,
    ).to(torch.float32)
    text_model.model.load_state_dict(full_model.model.language_model.state_dict())
    text_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
    text_model.eval()
    return tokenizer, text_model


def main():
    args = parse_args()
    tokenizer, model_hf = load_gemma4_text_model(args.model_id)
    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=model_hf.config,
        prompt=[args.prompt],
        prompt_len=args.prompt_len,
        ctx_len=args.ctx_len,
        full_batch_size=None,
    )

    qeff_model = QEFFAutoModelForCausalLM(copy.deepcopy(model_hf), pretrained_model_name_or_path=args.model_id)

    export_dir = Path(tempfile.mkdtemp()) / "onnx"
    onnx_path = Path(
        qeff_model.export(
            export_dir,
            use_onnx_subfunctions=True,
            offload_pt_weights=False,
        )
    )

    compile_dir = Path(tempfile.mkdtemp()) / "compile"
    if args.disable_npi:
        kv_cache_dtype = "float16"
        custom_io = {}
        for suffix in ("", "_RetainedState"):
            for i in range(qeff_model.num_layers):
                for kv in ("key", "value"):
                    custom_io[f"past_{kv}.{i}{suffix}"] = kv_cache_dtype

        specializations = [
            qeff_model.build_prefill_specialization(
                prefill_seq_len=args.prompt_len,
                ctx_len=args.ctx_len,
                batch_size=1,
                kv_cache_batch_size=1,
                full_batch_size=None,
            )
        ]
        if args.prompt_len != 1:
            decode_spec = qeff_model.build_decode_specialization(
                prefill_seq_len=args.prompt_len,
                ctx_len=args.ctx_len,
                batch_size=1,
                kv_cache_batch_size=1,
                full_batch_size=None,
            )
            if decode_spec:
                specializations.append(decode_spec)

        qpc_path = Path(
            qeff_model._compile(
                onnx_path=str(onnx_path),
                compile_dir=compile_dir,
                compile_only=True,
                retained_state=True,
                specializations=specializations,
                convert_to_fp16=True,
                custom_io=custom_io,
                aic_num_cores=16,
                use_onnx_subfunctions=True,
            )
        )
    else:
        qpc_path = Path(
            qeff_model.compile(
                onnx_path=str(onnx_path),
                compile_dir=compile_dir,
                prefill_seq_len=args.prompt_len,
                ctx_len=args.ctx_len,
                use_onnx_subfunctions=True,
            )
        )

    ort_tokens = np.asarray(api_runner.run_kv_model_on_ort(str(onnx_path))).reshape(-1)
    qaic_tokens_full = np.asarray(api_runner.run_kv_model_on_cloud_ai_100(str(qpc_path), args.device_group)).reshape(-1)

    # QAic generation output is padded to the compiled context length.
    qaic_tokens = qaic_tokens_full[: api_runner.gen_len]
    parity_match = bool(np.array_equal(ort_tokens, qaic_tokens))

    result = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "prompt_len": args.prompt_len,
        "ctx_len": args.ctx_len,
        "disable_npi": args.disable_npi,
        "generation_len": api_runner.gen_len,
        "onnx_path": str(onnx_path),
        "generated_npi_path": None
        if args.disable_npi
        else str(onnx_path.with_name(f"{onnx_path.stem}_gemma4_npi.yaml")),
        "qpc_path": str(qpc_path),
        "ort_tokens": ort_tokens.tolist(),
        "qaic_tokens_prefix": qaic_tokens.tolist(),
        "qaic_tokens_full": qaic_tokens_full.tolist(),
        "ort_text": tokenizer.decode(ort_tokens.tolist(), skip_special_tokens=True),
        "qaic_text": tokenizer.decode(qaic_tokens.tolist(), skip_special_tokens=True),
        "match": parity_match,
    }

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output_json is not None:
        args.output_json.write_text(payload)

    if args.fail_on_mismatch and not parity_match:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
