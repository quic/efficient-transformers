#!/usr/bin/env python3
"""
CohereLabs/cohere-transcribe-03-2026 — QPC inference on a single audio file.

Uses the demo clip from the model repo by default; accepts any local wav/mp3.

    python examples/audio/cohere_asr_infer.py \
        --qpc-dir <qpc_dir> --num-devices 4 \
        --token-file ~/huggingface/token
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, GenerationConfig

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
ENCODE_FEATURE_LEN = 5000  # QPC Encode spec: feature_len=5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--qpc-dir", required=True, help="Path to compiled QPC directory")
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--audio-file", default=None, help="Local path to a wav/mp3 file; defaults to the demo clip from the model repo")
    parser.add_argument("--generation-len", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.3, help="Penalise repeated tokens (1.0=off, 1.3 default)")
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--token-file", default=None, help="Path to file containing HF token")
    args = parser.parse_args()

    t0 = time.time()

    token = args.token or None
    if token is None and args.token_file:
        token = Path(args.token_file).read_text().strip() or None
    if token is None:
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    # ── Resolve audio file ────────────────────────────────────────────────────
    if args.audio_file:
        audio_path = args.audio_file
    else:
        print(f"[{time.time()-t0:.1f}s] Downloading demo clip from model repo...")
        from huggingface_hub import hf_hub_download
        audio_path = hf_hub_download(
            repo_id=args.model_id,
            filename="demo/voxpopuli_test_en_demo.wav",
            token=token,
            cache_dir=args.hf_cache,
        )
        print(f"[{time.time()-t0:.1f}s] Demo clip: {audio_path}")

    # ── Load audio ────────────────────────────────────────────────────────────
    from transformers.audio_utils import load_audio
    audio = load_audio(audio_path, sampling_rate=16000)

    # ── Processor ─────────────────────────────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, token=token, cache_dir=args.hf_cache)

    # Extract features (time-major: bsz, frames, mel)
    feats = processor(audio, sampling_rate=16000, language="en", return_tensors="pt")
    x = feats["input_features"]  # (1, frames, mel)
    pad_len = ENCODE_FEATURE_LEN - x.shape[1]
    if pad_len > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
    elif pad_len < 0:
        x = x[:, :ENCODE_FEATURE_LEN, :]
    # Transpose to mel-bins-major for QPC: (1, mel, frames)
    qpc_features = x.transpose(1, 2).numpy().astype(np.float16)

    prefix_ids = feats["decoder_input_ids"][0].tolist()
    print(f"[{time.time()-t0:.1f}s] Decoder prefix ({len(prefix_ids)} tokens): {prefix_ids}")

    # ── Load QPC ──────────────────────────────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] Loading model and QPC...")
    hw_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, token=token, cache_dir=args.hf_cache)

    # Backfill decoder_start_token_id from generation_config if missing on config
    if getattr(hw_model.model.config, "decoder_start_token_id", None) is None:
        try:
            gen_cfg = GenerationConfig.from_pretrained(args.model_id, token=token, cache_dir=args.hf_cache)
            hw_model.model.config.decoder_start_token_id = gen_cfg.decoder_start_token_id
        except Exception:
            hw_model.model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    hw_model.qpc_path = Path(args.qpc_dir)
    device_ids = list(range(args.num_devices))

    from QEfficient.generation.cloud_infer import QAICInferenceSession, is_retained_state_name

    if hw_model.qpc_session is None:
        hw_model.qpc_session = QAICInferenceSession(str(hw_model.qpc_path), device_ids)
        hw_model.batch_size = hw_model.qpc_session.bindings[0].dims[0]

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] Running QPC inference on {args.num_devices} devices...")

    inp = {
        "input_features": qpc_features,
        "input_ids": np.array([[prefix_ids[0]]], dtype=np.int64),
        "position_ids": np.array([[0]], dtype=np.int64),
    }

    hw_model.qpc_session.skip_buffers(
        [x for x in hw_model.qpc_session.input_names + hw_model.qpc_session.output_names
         if is_retained_state_name(x)]
    )
    hw_model.qpc_session.set_buffers({
        "logits": np.zeros((hw_model.batch_size, 1, hw_model.model.config.vocab_size), dtype=np.float32)
    })

    # Encode run — timed separately (TTFT numerator)
    encode_start = time.time()
    outputs = hw_model.qpc_session.run(inp)
    encode_elapsed = time.time() - encode_start

    # Switch to Decode spec
    inp["input_features"] = np.zeros(
        (hw_model.batch_size, hw_model.model.config.num_mel_bins, 1), dtype=np.float16
    )

    # Feed remaining prefix tokens (forced context)
    for pos, forced_tok in enumerate(prefix_ids[1:], start=1):
        inp["input_ids"] = np.array([[forced_tok]], dtype=np.int64)
        inp["position_ids"] = np.array([[pos]], dtype=np.int64)
        outputs = hw_model.qpc_session.run(inp)

    def _apply_repetition_penalty(logits_1d, generated, penalty):
        """Divide logits of already-generated tokens by penalty (>1 discourages repeats)."""
        if penalty == 1.0 or not generated:
            return logits_1d
        logits = logits_1d.copy()
        for tok in set(generated):
            if logits[tok] > 0:
                logits[tok] /= penalty
            else:
                logits[tok] *= penalty
        return logits

    # Greedy decode
    token_ids = list(prefix_ids)
    pos = len(prefix_ids)
    next_token = outputs["logits"].argmax(-1)
    token_ids.append(int(next_token[0][0]))

    decode_start = time.time()
    for step in range(args.generation_len):
        if int(next_token[0][0]) == hw_model.model.config.eos_token_id:
            print(f"  EOS at step {step}")
            break
        inp["input_ids"] = next_token
        inp["position_ids"] = np.array([[pos]], dtype=np.int64)
        pos += 1
        outputs = hw_model.qpc_session.run(inp)
        logits_1d = outputs["logits"][0][0]
        if args.repetition_penalty != 1.0:
            logits_1d = _apply_repetition_penalty(logits_1d, token_ids[len(prefix_ids):], args.repetition_penalty)
        next_token = np.array([[logits_1d.argmax()]], dtype=np.int64)
        token_ids.append(int(next_token[0][0]))
    decode_elapsed = time.time() - decode_start

    new_tokens = token_ids[len(prefix_ids):]
    text = processor.batch_decode([token_ids], skip_special_tokens=True)[0].strip()

    print(f"\n[{time.time()-t0:.1f}s] === RESULT ===")
    print(f"  Transcription : {text!r}")
    print(f"  New tokens    : {len(new_tokens)}")
    print(f"  Encode (TTFT) : {encode_elapsed*1000:.1f} ms")
    if len(new_tokens) > 1:
        tok_per_sec = (len(new_tokens) - 1) / decode_elapsed
        print(f"  Decode speed  : {tok_per_sec:.1f} tok/s  ({decode_elapsed*1000:.0f} ms total)")
    print(f"  Total elapsed : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
