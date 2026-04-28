from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer

from QEfficient.generation.cloud_infer import QAICInferenceSession

LAYER_DIR_RE = re.compile(r"layer_(\d+)_(\d+)$")


def discover_qpc_paths(base_path: Path) -> List[Path]:
    layer_dirs = []
    for child in base_path.iterdir():
        if not child.is_dir():
            continue
        match = LAYER_DIR_RE.match(child.name)
        if not match:
            continue
        layer_dirs.append((int(match.group(1)), int(match.group(2)), child))

    if not layer_dirs:
        raise FileNotFoundError(f"No layer directories found under: {base_path}")

    layer_dirs.sort(key=lambda x: (x[0], x[1]))
    qpc_paths: List[Path] = []
    for _, _, layer_dir in layer_dirs:
        candidates = sorted(p for p in layer_dir.glob("**/qpcs") if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No qpcs directory found in: {layer_dir}")
        qpc_paths.append(candidates[0])
    return qpc_paths


def pick_token_input_name(session: QAICInferenceSession) -> Optional[str]:
    if "input_ids" in session.input_names:
        return "input_ids"
    for name in session.input_names:
        if "input_ids" in name:
            return name
    return None


def pick_hidden_input_name(session: QAICInferenceSession) -> Optional[str]:
    for preferred in ("inputs_embeds", "input_embeds"):
        if preferred in session.input_names:
            return preferred
    for name in session.input_names:
        if name == "position_ids":
            continue
        if "compressed_kv" in name or "k_pe" in name:
            continue
        if "input_ids" in name:
            continue
        return name
    return None


def pick_pos_input_name(session: QAICInferenceSession) -> Optional[str]:
    if "position_ids" in session.input_names:
        return "position_ids"
    for name in session.input_names:
        if "position" in name:
            return name
    return None


def pick_main_output_name(session: QAICInferenceSession) -> str:
    candidates = [name for name in session.output_names]
    if not candidates:
        raise RuntimeError(f"No usable output name found for session outputs: {session.output_names}")
    if "logits" in candidates:
        return "logits"
    return candidates[-1]


def output_placeholder(session: QAICInferenceSession, output_name: str) -> np.ndarray:
    idx = session.binding_index_map[output_name]
    binding = session.bindings[idx]
    dtype = session.aic_to_np_dtype_mapping[binding.type]
    shape = tuple(max(1, int(dim)) for dim in binding.dims)
    return np.zeros(shape, dtype=dtype)


def inference_pipeline(
    base_path: str | Path,
    model_name: str = "moonshotai/Kimi-K2.5",
    prompt: str = "Help",
    max_len: int = 32,
    device_start: Optional[int] = None,
) -> List[int]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        trust_remote_code=True,
    )
    prompt_ids = tokenizer(prompt, return_tensors="np", add_special_tokens=True)["input_ids"][0].tolist()
    all_ids = list(prompt_ids)

    qpc_paths = discover_qpc_paths(Path(base_path + "/onnx_layerwise_tmp"))
    print(f"[LOAD] Found {len(qpc_paths)} layer sessions")

    sessions: List[Dict[str, object]] = []
    for i, qpc in enumerate(qpc_paths):
        device_ids = [device_start + i] if device_start is not None else None
        session = QAICInferenceSession(str(qpc), device_ids=device_ids)
        session.skip_buffers(
            [n for n in session.input_names + session.output_names if "compressed_kv" in n or "k_pe" in n]
        )

        out_name = pick_main_output_name(session)
        session.set_buffers({out_name: output_placeholder(session, out_name)})

        sessions.append(
            {
                "session": session,
                "token_input": pick_token_input_name(session),
                "hidden_input": pick_hidden_input_name(session),
                "pos_input": pick_pos_input_name(session),
                "out_name": out_name,
            }
        )
        print(f"[LOAD] layer {i}: {qpc} -> out={out_name}")

    if not sessions:
        raise RuntimeError("No sessions loaded")
    if sessions[0]["token_input"] is None:
        raise RuntimeError(f"First layer has no token input. inputs={sessions[0]['session'].input_names}")

    logits = None

    # Prefill: pass each prompt token through all layers
    for pos, token_id in enumerate(prompt_ids):
        hidden = None
        for i, info in enumerate(sessions):
            session = info["session"]
            run_inputs: Dict[str, np.ndarray] = {}
            if i == 0:
                run_inputs[info["token_input"]] = np.array([[token_id]], dtype=np.int64)
            else:
                if hidden is None:
                    raise RuntimeError("Missing hidden state while executing intermediate layer")
                if info["hidden_input"] is None:
                    raise RuntimeError(f"Layer {i} has no hidden-state input. inputs={session.input_names}")
                run_inputs[info["hidden_input"]] = hidden

            if info["pos_input"] is not None:
                run_inputs[info["pos_input"]] = np.array([[pos]], dtype=np.int64)

            outputs = session.run(run_inputs)
            hidden = outputs[info["out_name"]]
        logits = hidden

    if logits is None:
        raise RuntimeError("Prompt produced no logits")

    # Decode
    generated_ids: List[int] = []
    while len(all_ids) < max_len:
        next_token_id = int(np.argmax(logits, axis=-1)[0, 0])
        generated_ids.append(next_token_id)
        all_ids.append(next_token_id)

        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break

        pos = len(all_ids) - 1
        hidden = None
        for i, info in enumerate(sessions):
            session = info["session"]
            run_inputs: Dict[str, np.ndarray] = {}
            if i == 0:
                run_inputs[info["token_input"]] = np.array([[next_token_id]], dtype=np.int64)
            else:
                if hidden is None:
                    raise RuntimeError("Missing hidden state while decoding intermediate layer")
                if info["hidden_input"] is None:
                    raise RuntimeError(f"Layer {i} has no hidden-state input. inputs={session.input_names}")
                run_inputs[info["hidden_input"]] = hidden

            if info["pos_input"] is not None:
                run_inputs[info["pos_input"]] = np.array([[pos]], dtype=np.int64)

            outputs = session.run(run_inputs)
            hidden = outputs[info["out_name"]]
        logits = hidden

    print("Generated token ids:")
    print(generated_ids)
    print("Generated text:")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    return generated_ids


def inference_pipelines(base_path: str | Path) -> List[int]:
    # Backward-compatible wrapper used by some local scripts.
    return inference_pipeline(base_path=base_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run layerwise QAIC prefill + decode from a base path.")
    parser.add_argument("base_path", type=Path, help="Path to onnx layer wise without onnx_layerwise_tmp ")
    parser.add_argument("--model-name", default="moonshotai/Kimi-K2.5")
    parser.add_argument("--prompt", default="Help")
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument(
        "--device-start",
        type=int,
        default=None,
        help="Optional starting device id. If set, layer i uses device_start + i.",
    )
    args = parser.parse_args()
    inference_pipeline(
        base_path=args.base_path,
        model_name=args.model_name,
        prompt=args.prompt,
        max_len=args.max_len,
        device_start=args.device_start,
    )


if __name__ == "__main__":
    main()
