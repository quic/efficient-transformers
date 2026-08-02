#!/usr/bin/env python3
"""
CohereLabs/cohere-transcribe-03-2026 hardware parity validation.

Run with QAIC device access (qaic group membership or sudo):

    python examples/audio/cohere_asr_hw_parity.py --qpc-dir <qpc_dir>

or let it compile automatically:

    python examples/audio/cohere_asr_hw_parity.py --compile
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--qpc-dir", default=None, help="Path to existing compiled QPC directory")
    parser.add_argument("--compile", action="store_true", help="Compile from scratch if no QPC given")
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--generation-len", type=int, default=100)
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    t0 = time.time()
    token = args.token or None
    hf_cache = args.hf_cache or None

    processor = AutoProcessor.from_pretrained(args.model_id, token=token, cache_dir=hf_cache)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation",
                      cache_dir=hf_cache)
    samples = [ds[i] for i in range(args.num_samples)]

    def get_hf_features(sample):
        audio, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        return processor(audio, "en", sampling_rate=sr, return_tensors="pt")

    def get_qeff_features(sample):
        feats = get_hf_features(sample)
        # Transpose from time-major (bsz, frames, mel) to mel-bins-major (bsz, mel, frames)
        feats["input_features"] = feats["input_features"].transpose(1, 2)
        return feats

    # ── HF reference ───────────────────────────────────────────────────────────
    print(f"[{time.time()-t0:.1f}s] HF reference inference...")
    hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, token=token,
        cache_dir=hf_cache, dtype=torch.float32)
    hf_model.eval()

    hf_texts = []
    for i, sample in enumerate(samples):
        features = get_hf_features(sample)
        with torch.no_grad():
            out = hf_model.generate(
                input_features=features["input_features"],
                decoder_input_ids=features["decoder_input_ids"],
                max_new_tokens=args.generation_len,
            )
        text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        hf_texts.append(text)
        print(f"  HF [{i}]: {text}")
    del hf_model

    # ── QPC inference ──────────────────────────────────────────────────────────
    hw_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, token=token)

    if args.qpc_dir:
        hw_model.qpc_path = Path(args.qpc_dir)
        # Try to infer onnx_path from QPC directory structure
        onnx_dir = Path(args.qpc_dir).parent.parent
        onnx_candidates = list(onnx_dir.glob("*.onnx"))
        if onnx_candidates:
            hw_model.onnx_path = onnx_candidates[0]
        print(f"[{time.time()-t0:.1f}s] Using existing QPC: {args.qpc_dir}")
    elif args.compile:
        print(f"[{time.time()-t0:.1f}s] Exporting and compiling...")
        hw_model.export()
        hw_model.compile(num_devices=args.num_devices, num_cores=16)
    else:
        print("ERROR: Provide --qpc-dir <path> or --compile")
        sys.exit(1)

    print(f"[{time.time()-t0:.1f}s] Running QPC inference (device_ids=[0..{args.num_devices-1}])...")
    device_ids = list(range(args.num_devices))

    hw_texts = []
    for i, sample in enumerate(samples):
        features = get_qeff_features(sample)
        exec_info = hw_model.generate(
            inputs={"input_features": features["input_features"]},
            generation_len=args.generation_len,
            device_ids=device_ids,
        )
        text = processor.batch_decode(exec_info.generated_ids, skip_special_tokens=True)[0].strip()
        hw_texts.append(text)
        print(f"  QPC [{i}]: {text}")

    # ── Parity summary ─────────────────────────────────────────────────────────
    print(f"\n[{time.time()-t0:.1f}s] === PARITY ===")
    all_pass = True
    for i in range(len(samples)):
        match = hf_texts[i].lower() == hw_texts[i].lower()
        if not match:
            all_pass = False
        print(f"  [{i}] {'PASS' if match else 'FAIL'}: HF={hf_texts[i]!r}  HW={hw_texts[i]!r}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nOverall: {verdict} ({sum(1 for h, w in zip(hf_texts, hw_texts) if h.lower()==w.lower())}/{len(samples)} samples match)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
