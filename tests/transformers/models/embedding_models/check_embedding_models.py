# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
import os
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP
from QEfficient.transformers.models.modeling_auto import QEFFAutoModel, QEFFAutoModelForImageTextToText
from QEfficient.transformers.models.qwen3_vl._embedding_utils import (
    EXAMPLE_DOCUMENTS,
    EXAMPLE_QUERIES,
    QEffQwen3VLEmbedder,
    configure_embedding_model_config,
    resolve_model_source,
)
from QEfficient.utils.constants import Constants
from QEfficient.utils.test_utils import load_vlm_model

from ..check_model_results import dump_and_compare_results


def _compute_cpu_embeddings(model_hf, embedder, model_inputs: List[Dict[str, Any]]) -> torch.Tensor:
    embedding_rows = []
    for entry in model_inputs:
        conversation = embedder.format_model_input(
            text=entry.get("text"),
            image=entry.get("image"),
            video=entry.get("video"),
            instruction=entry.get("instruction"),
        )
        tokenized = embedder._tokenize_conversation(conversation)
        hf_inputs = {}
        for key, value in tokenized.items():
            hf_inputs[key] = value.to(next(model_hf.parameters()).device) if torch.is_tensor(value) else value

        with torch.no_grad():
            last_hidden_state = model_hf.model(**hf_inputs).last_hidden_state

        last_idx = tokenized["input_ids"].shape[1] - 1
        row = last_hidden_state[:, last_idx : last_idx + 1, :].reshape(last_hidden_state.shape[0], -1)
        embedding_rows.append(row.detach().cpu().to(torch.float32))

    embeddings = torch.cat(embedding_rows, dim=0)
    return F.normalize(embeddings, p=2, dim=-1)


def check_qwen3_vl_embedding_cpu_vs_ai100_mad_parity(model_name, export_compile_only=False):
    torch.manual_seed(42)

    ctx_len = 2048
    num_layers = -1
    vision_depth = 9
    deepstack_index = 8
    compile_prefill_seq_len = None
    mad_max_threshold = 0.002

    model_source = resolve_model_source(model_name)

    config = AutoConfig.from_pretrained(model_source, trust_remote_code=True, padding=True)
    # Keep parity runs lightweight by default (reduced text/vision depth from
    # test config). To validate full-layer quality, update the config entry.
    configure_embedding_model_config(
        config=config,
        num_hidden_layers=num_layers,
        vision_depth=vision_depth,
        deepstack_index=deepstack_index,
        export_embedding=False,
    )

    model_hf = load_vlm_model(config)
    model_hf.eval()

    qeff_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True, padding=True)
    configure_embedding_model_config(
        config=qeff_config,
        num_hidden_layers=num_layers,
        vision_depth=vision_depth,
        deepstack_index=deepstack_index,
        export_embedding=True,
    )

    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True, padding=True)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_source,
        kv_offload=True,
        trust_remote_code=True,
        config=qeff_config,
        qaic_config={"export_embedding": True},
    )

    embedder = QEffQwen3VLEmbedder(
        processor=processor,
        model=qeff_model,
    )

    model_inputs = EXAMPLE_QUERIES + EXAMPLE_DOCUMENTS
    compile_specs = embedder.get_compile_specs(
        inputs=model_inputs,
        ctx_len=ctx_len,
        prefill_seq_len=compile_prefill_seq_len,
    )
    qpc_paths = qeff_model.compile(
        prefill_seq_len=compile_specs["prefill_seq_len"],
        ctx_len=compile_specs["ctx_len"],
        img_size=compile_specs["img_size"],
        height=compile_specs["height"],
        width=compile_specs["width"],
        num_devices=1,
        num_cores=16,
        mxfp6_matmul=False,
    )

    if export_compile_only:
        return

    cpu_embeddings = _compute_cpu_embeddings(model_hf=model_hf, embedder=embedder, model_inputs=model_inputs)
    ai100_embeddings = embedder.process(
        inputs=model_inputs,
        qpc_paths=qpc_paths,
        prefill_seq_len=compile_specs["prefill_seq_len"],
        normalize=True,
    )

    diff = torch.abs(cpu_embeddings - ai100_embeddings)
    mad_mean = float(diff.mean().item())
    mad_max = float(diff.max().item())
    threshold = float(mad_max_threshold)

    print(f"[MAD] CPU vs AI100 mean={mad_mean:.6e}, max={mad_max:.6e}")
    assert mad_max <= threshold, (
        f"CPU vs AI100 MAD max {mad_max:.6e} exceeds threshold {threshold:.6e}. "
        f"Check prompt formatting, tokenization, prompt-length handling, and AI100 compile args."
    )


def load_embedding_model(model_name: str, n_layer: int = -1):
    """Load a pre-trained embedding model."""
    kwargs = {"attn_implementation": "eager", "trust_remote_code": True}
    if n_layer > 0:
        kwargs["num_hidden_layers"] = n_layer
    pt_model = AutoModel.from_pretrained(
        model_name,
        **kwargs,
    )
    pt_model.eval()
    return pt_model


def check_embed_pytorch_vs_ort_vs_ai100(
    model_name: str,
    seq_len: int = Constants.CTX_LEN,
    n_layer: int = -1,
    enable_qnn: Optional[bool] = False,
    qnn_config: Optional[str] = None,
    pooling: Optional[str] = None,
    compare_results: Optional[bool] = False,
    export_compile_only: Optional[bool] = False,
):
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("My name is", return_tensors="pt")

    pt_model = load_embedding_model(model_name, n_layer)
    # Original PyTorch model output
    pt_outputs = pt_model(**inputs)
    pooling_method = POOLING_MAP[pooling] if pooling else None
    pt_embeddings = (
        pooling_method(pt_outputs.last_hidden_state, inputs["attention_mask"])
        if pooling
        else pt_outputs.last_hidden_state
    )

    # QEff transformed PyTorch model
    qeff_model = QEFFAutoModel(pt_model, pretrained_model_name_or_path=model_name, pooling=pooling)

    # QEff transformed PyTorch model output
    qeff_pt_outputs = qeff_model.generate(inputs=inputs, runtime_ai100=False)
    qeff_pt_embeddings = qeff_pt_outputs if pooling else qeff_pt_outputs[0]

    mad = torch.mean(torch.abs(pt_embeddings - qeff_pt_embeddings))
    print("Mad for PyTorch and PyTorch transformed qeff_model is ", mad)
    assert mad <= 0, f"MAD is too high for onnx and Pytorch: {mad}"

    # ONNX session load
    onnx_model = qeff_model.export()
    ort_session = ort.InferenceSession(str(onnx_model))

    # Prepare the inputs for ONNX Runtime
    input_ids = np.array(inputs["input_ids"])
    attention_mask = np.array(inputs["attention_mask"])

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # Run inference
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # Compare Transformed PyTorch and ONNX outputs
    mad = torch.mean(torch.abs(pt_embeddings - torch.tensor(onnx_outputs[0])))
    print("Mad for onnx and PyTorch is ", mad)
    assert mad <= 10**-5, f"MAD is too high for onnx and Pytorch: {mad}"

    qeff_model.compile(
        enable_qnn=enable_qnn,
        qnn_config=qnn_config,
    )

    if export_compile_only:
        return

    ai100_output = qeff_model.generate(inputs=inputs)
    qeff_ai100_embeddings = (
        ai100_output["output"] if pooling else ai100_output["output"][:, : inputs["input_ids"].shape[1], :]
    )

    # Compare ONNX and AI 100 outputs
    mad = np.mean(np.abs(qeff_ai100_embeddings - onnx_outputs[0]))
    print("Mad for onnx and AI 100 output is ", mad)
    assert mad <= 10**-2, f"MAD is too high for onnx and Pytorch: {mad}"
    assert os.path.isfile(os.path.join(os.path.dirname(qeff_model.qpc_path), "qconfig.json"))

    if compare_results is False:
        return

    compile_params = {"enable_qnn": enable_qnn, "qnn_config": qnn_config, "pooling": pooling, "seq_len": seq_len}
    assert dump_and_compare_results(
        model_name,
        compile_params,
        "embedding_model_results.json",
        qeff_ai100_embeddings,
        pytorch_hf_tokens=pt_embeddings,
        pytorch_kv_tokens=qeff_pt_embeddings,
        ort_tokens=onnx_outputs[0],
    )
