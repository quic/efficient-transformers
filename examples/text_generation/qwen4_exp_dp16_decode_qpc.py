#!/usr/bin/env python3
"""Build and run a decode-only Qwen3.8-Flash-Next model.

This is a decode-only DP16/CP1 flow.  With no ``--checkpoint``, it loads the
model ID through QEfficient's Hugging Face loader; ``--layers 48`` runs the
full text model with the same export and runtime path.

Example::

    HF_HUB_CACHE=/home/huggingface_hub \
    /home/ochougul/envs/qwen38f/bin/python \
        examples/text_generation/qwen4_exp_dp16_decode_qpc.py \
        --model-name-or-path Qwen/Qwen3.8-Flash-Next \
        --layers 48 \

Use ``--checkpoint`` to reuse an existing local checkpoint, or ``--qpc`` to
skip export/compile and replay an existing QPC.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from QEfficient.transformers.models.qwen4_exp.host_ngram import HostNGramHistory, ShardedNGramLookup


NUM_CORES = 4
BATCH_SIZE = 16
DEFAULT_LAYERS = 4
CONTEXT_LENGTH = 65_536


def _window_config_from_source(model_name_or_path: str | Path, layers: int):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen4_exp_decode_parity import (
        _window_config,
    )

    model_path = Path(model_name_or_path)
    if model_path.is_dir():
        return _window_config(model_path, layers, CONTEXT_LENGTH, dtype="bfloat16")

    config = AutoConfig.from_pretrained(str(model_name_or_path))
    raw = config.text_config.to_dict()
    raw["dtype"] = "bfloat16"
    raw["num_hidden_layers"] = layers
    raw["layer_types"] = raw["layer_types"][:layers]
    raw["ple_layer_ids"] = [layer_id for layer_id in raw["ple_layer_ids"] if layer_id <= layers]
    raw["max_position_embeddings"] = CONTEXT_LENGTH

    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(**raw)


def build_model(model_name_or_path: str | Path, checkpoint: Path | None, layers: int, num_devices: int):
    qaic_config = {
        "indexer_dp": num_devices,
        "indexer_cp": 1,
        "attn_dp": num_devices,
        "attn_cp": 1,
        "indexer_num_blocks": 4,
        "indexer_ql_chunk": 512,
        "indexer_q_block_size": 512,
        "indexer_topk_blocking": 4,
        "sparse_q_block_size": 16,
        "sparse_kv_num_blocks": 8,
        "num_cores_per_device": NUM_CORES,
        "ffn_blocking_mode": "token",
        "ffn_token_block_size": 4,
    }
    config = _window_config_from_source(model_name_or_path, layers)
    from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

    model_source = str(checkpoint) if checkpoint is not None else str(model_name_or_path)
    model = QEFFAutoModelForCausalLM.from_pretrained(
        model_source,
        config=config,
        qaic_config=qaic_config,
        weight_free=True,
        torch_dtype=torch.bfloat16,
    )
    model.model.eval()
    return model


def _load_window_config(snapshot: Path, layers: int):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qwen4_exp_decode_parity import _window_config

    return _window_config(snapshot, layers, CONTEXT_LENGTH, dtype="bfloat16")


class Qwen4ExpQAICDecodeRunner:
    """Single-token Qwen4-Exp runner that supports the compiled DP16 batch."""

    def __init__(self, qpc_path: Path, device_ids: list[int] | None = None):
        from QEfficient.generation.cloud_infer import QAICInferenceSession

        self.session = QAICInferenceSession(qpc_path, device_ids=device_ids)
        skipped_names = [
            name for name in (*self.session.input_names, *self.session.output_names) if is_retained_state(name)
        ]
        self.session.skip_buffers(skipped_names)

        session_inputs = {name.rsplit("/", 1)[-1]: name for name in self.session.input_names}
        required_inputs = ("input_ids", "ngram_embeddings", "position_ids")
        missing_inputs = set(required_inputs) - set(session_inputs)
        if missing_inputs:
            raise ValueError(f"QPC is missing Qwen host inputs: {sorted(missing_inputs)}")
        self.host_input_names = {name: session_inputs[name] for name in required_inputs}
        self.host_input_shapes = {
            name: tuple(self.session.bindings[self.session.binding_index_map[binding_name]].dims)
            for name, binding_name in self.host_input_names.items()
        }
        if self.host_input_shapes["input_ids"][1:] != (1,):
            raise ValueError(f"Decode-only QPC must accept input_ids with sequence length 1")
        if self.host_input_shapes["ngram_embeddings"][1:] != (1, 2560):
            raise ValueError(f"Unexpected Qwen n-gram embedding shape: {self.host_input_shapes['ngram_embeddings']}")

        self.logits_name = next(
            (name for name in self.session.output_names if name.rsplit("/", 1)[-1] == "logits"),
            None,
        )
        if self.logits_name is None:
            raise ValueError(f"QPC is missing logits output: {self.session.output_names}")
        self.logits_shape = tuple(self.session.bindings[self.session.binding_index_map[self.logits_name]].dims)
        logits_dtype = np.float16 if self.session.binding_is_bfloat16(self.logits_name) else np.float32
        self.logits_is_bfloat16 = self.session.binding_is_bfloat16(self.logits_name)
        self.session.set_buffers({self.logits_name: np.zeros(self.logits_shape, dtype=logits_dtype)})

    def run_step(self, input_ids: np.ndarray, ngram_embeddings: torch.Tensor, position_ids: np.ndarray) -> np.ndarray:
        inputs = {
            self.host_input_names["input_ids"]: np.asarray(input_ids, dtype=np.int64),
            self.host_input_names["ngram_embeddings"]: self._embedding_numpy(ngram_embeddings),
            self.host_input_names["position_ids"]: np.asarray(position_ids, dtype=np.int64),
        }
        for name, binding_name in self.host_input_names.items():
            if tuple(inputs[binding_name].shape) != self.host_input_shapes[name]:
                raise ValueError(f"{name} must be {self.host_input_shapes[name]}")
        outputs = self.session.run(inputs)
        logits_output = next(
            (output for name, output in outputs.items() if name.rsplit("/", 1)[-1] == "logits"),
            None,
        )
        if logits_output is None:
            raise ValueError(f"QAIC run did not return logits: {sorted(outputs)}")
        logits = np.asarray(logits_output)
        if self.logits_is_bfloat16:
            logits = (logits.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            logits = logits.astype(np.float32, copy=False)
        if tuple(logits.shape) != self.logits_shape:
            raise ValueError(f"Unexpected QAIC logits shape: {tuple(logits.shape)}")
        return logits

    def _embedding_numpy(self, ngram_embeddings: torch.Tensor) -> np.ndarray:
        """Pack host embeddings according to the QPC n-gram binding dtype."""
        embeddings = ngram_embeddings.detach().cpu()
        binding_name = self.host_input_names["ngram_embeddings"]
        if self.session.binding_is_bfloat16(binding_name):
            # NumPy has no BF16 dtype. Preserve BF16 bit patterns in a 2-byte
            # float16 view, matching the QAIC runtime host representation.
            return embeddings.to(torch.bfloat16).view(torch.int16).numpy().view(np.float16)
        return embeddings.float().numpy()


def is_retained_state(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.startswith(("gdn_", "ple_", "qsa_"))


def run_one_decode_step(qpc_path: Path) -> None:
    runner = Qwen4ExpQAICDecodeRunner(qpc_path)
    input_ids = np.zeros(runner.host_input_shapes["input_ids"], dtype=np.int64)
    ngram_embeddings = torch.zeros(
        runner.host_input_shapes["ngram_embeddings"],
        dtype=torch.bfloat16
        if runner.session.binding_is_bfloat16(runner.host_input_names["ngram_embeddings"])
        else torch.float32,
    )
    position_ids = np.zeros(runner.host_input_shapes["position_ids"], dtype=np.int64)
    logits = runner.run_step(input_ids, ngram_embeddings, position_ids)
    print(f"executed one decode step; logits shape={logits.shape}; next token={int(np.argmax(logits))}")


def run_decode_generation(
    qpc_runner: Qwen4ExpQAICDecodeRunner,
    lookup: ShardedNGramLookup,
    input_ids: np.ndarray,
    generation_len: int,
    tokenizer,
) -> tuple[np.ndarray, list[str]]:
    """Run prompt and greedy generation through the decode-only QPC.

    The prompt is consumed one token at a time because this QPC has only a
    single-token input specialization.  The host PLE lookup is performed before
    every QPC call, including generated tokens.  ``-1`` is treated as prompt
    padding and does not advance the host history or position.
    """
    if generation_len < 1:
        raise ValueError("generation_len must be at least 1")

    prompt_ids = np.asarray(input_ids, dtype=np.int64)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [prompt_length] or [batch_size, prompt_length]")

    input_shape = qpc_runner.host_input_shapes["input_ids"]
    batch_size, decode_seq_len = input_shape
    if decode_seq_len != 1:
        raise ValueError(f"Decode-only QPC must accept one token, got input_ids shape {input_shape}")
    if prompt_ids.shape[0] == 1:
        prompt_ids = np.repeat(prompt_ids, batch_size, axis=0)
    elif prompt_ids.shape[0] != batch_size:
        raise ValueError(f"input_ids batch must be 1 or the compiled batch size {batch_size}")

    config = lookup.config
    eos_token_id = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
    history = HostNGramHistory(config.ngram_size, eos_token_id)
    position_shape = qpc_runner.host_input_shapes["position_ids"]
    last_logits = None
    position = 0
    prompt_steps = 0

    for prompt_position in range(prompt_ids.shape[1]):
        token_ids = prompt_ids[:, prompt_position : prompt_position + 1]
        if np.all(token_ids == -1):
            continue
        if np.any(token_ids == -1):
            raise ValueError("Each prompt position must be valid for every batch row or uniformly -1")

        token_tensor = torch.from_numpy(token_ids)
        ngram_embeddings = lookup.lookup(token_tensor, history)
        position_ids = np.full(position_shape, position, dtype=np.int64)
        last_logits = qpc_runner.run_step(token_ids, ngram_embeddings, position_ids)
        position += 1
        prompt_steps += 1

    if last_logits is None:
        raise ValueError("input_ids must contain at least one non-negative prompt token")

    generated_tokens = [np.argmax(last_logits, axis=-1).astype(np.int64)]
    for _ in range(1, generation_len):
        token_ids = generated_tokens[-1]
        ngram_embeddings = lookup.lookup(torch.from_numpy(token_ids), history)
        position_ids = np.full(position_shape, position, dtype=np.int64)
        logits = qpc_runner.run_step(token_ids, ngram_embeddings, position_ids)
        generated_tokens.append(np.argmax(logits, axis=-1).astype(np.int64))
        position += 1

    generated_ids = np.concatenate(generated_tokens, axis=1)
    generated_texts = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
    print(f"prompt_decode_steps={prompt_steps}")
    print(f"generation_steps={generation_len}")
    print(f"generated_token_ids={generated_ids.tolist()}")
    for batch_index, generated_text in enumerate(generated_texts):
        print(f"generated_text[{batch_index}]={generated_text}")
    return generated_ids, generated_texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="local checkpoint directory or Hugging Face model ID",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional local checkpoint; omitted uses --model-name-or-path through QEfficient.from_pretrained",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=DEFAULT_LAYERS,
        help="number of leading text layers to export; use 48 for the full model",
    )
    parser.add_argument("--num-devices", type=int, default=16, help="TS device count for DP and compilation")
    parser.add_argument(
        "--compile-dir",
        type=Path,
        default=Path("attention_benchmarks/qsa_4layer_dp16"),
        help="directory for exported ONNX and QPC artifacts",
    )
    parser.add_argument("--qpc", type=Path, help="reuse an existing QPC and skip model export/compile")
    parser.add_argument("--prompt", help="optional prompt to run through the QEfficient generate API")
    parser.add_argument("--generation-len", type=int, default=8, help="maximum tokens for --prompt generation")
    args = parser.parse_args()
    args.model_name_or_path = str(args.model_name_or_path)
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    args.compile_dir = args.compile_dir.resolve()

    qeff_model = None
    if args.qpc is None:
        os.environ.setdefault("QEFF_HOME", str(args.compile_dir))
        if args.checkpoint is not None and not args.checkpoint.is_dir():
            raise FileNotFoundError(f"Four-layer checkpoint directory not found: {args.checkpoint}")
        qeff_model = build_model(args.model_name_or_path, args.checkpoint, args.layers, args.num_devices)
        qeff_model.model.eval()
        qpc_path = Path(
            qeff_model.compile(
                compile_dir=str(args.compile_dir),
                batch_size=BATCH_SIZE,
                ctx_len=CONTEXT_LENGTH,
                prefill_seq_len=1,
                num_devices=args.num_devices,
                num_cores=NUM_CORES,
                mxfp6_matmul=False,
                offload_pt_weights=False,
                dynamo=True,
                use_onnx_subfunctions=False,
                aic_hw_version="ai200",
                mxint8_kv_cache=True,
                allow_mxint8_mdp_io=True,
            )
        )
    else:
        qpc_path = args.qpc

    if not (qpc_path / "programqpc.bin").is_file():
        raise FileNotFoundError(f"QPC binary not found: {qpc_path / 'programqpc.bin'}")
    print(f"running QPC: {qpc_path}")
    run_one_decode_step(qpc_path)

    if args.prompt is not None:
        snapshot = Path(args.model_name_or_path)
        if not snapshot.is_dir():
            raise ValueError(
                "--prompt requires --model-name-or-path to be a local checkpoint directory because "
                "the host n-gram lookup reads its safetensors shards directly"
            )
        snapshot = snapshot.resolve()
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot))
        prompt_input_ids = tokenizer(
            args.prompt,
            return_tensors="np",
            add_special_tokens=True,
        )["input_ids"]
        qpc_runner = Qwen4ExpQAICDecodeRunner(qpc_path)
        config = _load_window_config(snapshot, args.layers)
        lookup = ShardedNGramLookup(config, snapshot)
        run_decode_generation(
            qpc_runner=qpc_runner,
            lookup=lookup,
            input_ids=prompt_input_ids,
            generation_len=args.generation_len,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()
