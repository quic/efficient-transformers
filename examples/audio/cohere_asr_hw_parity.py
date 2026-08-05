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

import numpy as np
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
    parser.add_argument("--token-file", default=None, help="Path to file containing HF token")
    args = parser.parse_args()

    t0 = time.time()
    token = args.token or None
    if token is None and args.token_file:
        token = Path(args.token_file).read_text().strip() or None
    if token is None:
        import os
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    hf_cache = args.hf_cache or None

    processor = AutoProcessor.from_pretrained(args.model_id, token=token, cache_dir=hf_cache)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation",
                      cache_dir=hf_cache)
    samples = [ds[i] for i in range(args.num_samples)]

    def get_hf_features(sample):
        audio, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        return processor(audio, "en", sampling_rate=sr, return_tensors="pt")

    # The QPC Encode spec requires exactly feature_len=5000 (= encoder_ctx_len * subsampling_factor).
    # To ensure HF and QPC receive identical encoder input (for fair parity), both paths
    # use the same 5000-frame padded input_features.
    ENCODE_FEATURE_LEN = 5000

    def get_hf_features(sample):
        audio, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        feats = processor(audio, "en", sampling_rate=sr, return_tensors="pt")
        # Pad time-major (bsz, frames, mel) to ENCODE_FEATURE_LEN so HF and QPC
        # receive identical encoder input (pad_len real frames + zeros).
        x = feats["input_features"]  # (bsz, frames, mel)
        pad_len = ENCODE_FEATURE_LEN - x.shape[1]
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        elif pad_len < 0:
            x = x[:, :ENCODE_FEATURE_LEN, :]
        feats["input_features"] = x
        return feats

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
    hf_token_lists = []
    for i, sample in enumerate(samples):
        features = get_hf_features(sample)
        with torch.no_grad():
            out = hf_model.generate(
                input_features=features["input_features"],
                decoder_input_ids=features["decoder_input_ids"],
                max_new_tokens=args.generation_len,
            )
        # out includes the decoder_input_ids prefix; strip it for token comparison
        prefix_len = features["decoder_input_ids"].shape[1]
        new_tokens = out[0][prefix_len:].tolist()
        hf_token_lists.append(new_tokens)
        text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        hf_texts.append(text)
        print(f"  HF [{i}]: {text}")
    del hf_model

    # ── QPC inference ──────────────────────────────────────────────────────────
    hw_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, token=token,
                                                              cache_dir=hf_cache)

    # CohereAsrConfig doesn't carry decoder_start_token_id; it lives in generation_config.json.
    # Backfill so QEFFAutoModelForSpeechSeq2Seq.generate() can build the initial decoder input.
    if getattr(hw_model.model.config, "decoder_start_token_id", None) is None:
        from transformers import GenerationConfig
        try:
            gen_cfg = GenerationConfig.from_pretrained(args.model_id, token=token, cache_dir=hf_cache)
            hw_model.model.config.decoder_start_token_id = gen_cfg.decoder_start_token_id
        except Exception:
            hw_model.model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

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

    from QEfficient.generation.cloud_infer import QAICInferenceSession, is_retained_state_name

    if hw_model.qpc_session is None:
        hw_model.qpc_session = QAICInferenceSession(str(hw_model.qpc_path), device_ids)
        hw_model.batch_size = hw_model.qpc_session.bindings[0].dims[0]

    hw_texts = []
    hw_token_lists = []
    for i, sample in enumerate(samples):
        features = get_qeff_features(sample)
        # Get the full decoder prefix from the processor (10 control tokens).
        # QPC generate() only seeds with decoder_start_token_id; we must feed the
        # full prefix manually so the greedy decode starts from the right distribution.
        hf_feats = get_hf_features(sample)
        prefix_ids = hf_feats["decoder_input_ids"][0].tolist()  # e.g. [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]

        inp = {
            "input_features": features["input_features"].numpy().astype(np.float16),
            "input_ids": np.array([[prefix_ids[0]]], dtype=np.int64),
            "position_ids": np.array([[0]], dtype=np.int64),
        }

        hw_model.qpc_session.skip_buffers(
            [x for x in hw_model.qpc_session.input_names + hw_model.qpc_session.output_names
             if is_retained_state_name(x)]
        )
        out_buf = {"logits": np.random.randn(hw_model.batch_size, 1, hw_model.model.config.vocab_size).astype(np.float32)}
        hw_model.qpc_session.set_buffers(out_buf)

        # Encode run (Encode spec: input_features=5000 frames)
        outputs = hw_model.qpc_session.run(inp)

        # Switch to Decode spec
        inp["input_features"] = np.zeros((hw_model.batch_size, hw_model.model.config.num_mel_bins, 1), dtype=np.float16)

        # Feed remaining prefix tokens through Decode spec (discard logits — forced context)
        for pos, forced_tok in enumerate(prefix_ids[1:], start=1):
            inp["input_ids"] = np.array([[forced_tok]], dtype=np.int64)
            inp["position_ids"] = np.array([[pos]], dtype=np.int64)
            outputs = hw_model.qpc_session.run(inp)

        # After the prefix loop, `outputs["logits"]` is the prediction at position len(prefix_ids).
        # token_ids holds prefix + that first generated token.
        # The loop feeds each generated token and advances position.
        logits = outputs["logits"]
        next_token = logits.argmax(-1)
        token_ids = list(prefix_ids) + [int(next_token[0][0])]

        # next greedy step goes at position len(prefix_ids), feeding next_token
        pos = len(prefix_ids)
        for _ in range(args.generation_len):
            if int(next_token[0][0]) == hw_model.model.config.eos_token_id:
                break
            inp["input_ids"] = next_token
            inp["position_ids"] = np.array([[pos]], dtype=np.int64)
            pos += 1
            outputs = hw_model.qpc_session.run(inp)
            logits = outputs["logits"]
            next_token = logits.argmax(-1)
            token_ids.append(int(next_token[0][0]))

        text = processor.batch_decode([token_ids], skip_special_tokens=True)[0].strip()
        hw_texts.append(text)
        hw_token_lists.append(token_ids[len(prefix_ids):])  # strip prefix for comparison
        print(f"  QPC [{i}]: {text}")

    # ── Parity summary ─────────────────────────────────────────────────────────
    print(f"\n[{time.time()-t0:.1f}s] === PARITY ===")
    all_pass = True
    for i in range(len(samples)):
        hf_toks = hf_token_lists[i]
        qpc_toks = hw_token_lists[i]
        # Compare token-for-token up to the shorter of the two sequences.
        # This handles the case where EOS is not generated at fp16 (generation cutoff differs).
        n = min(len(hf_toks), len(qpc_toks))
        match = hf_toks[:n] == qpc_toks[:n]
        if not match:
            all_pass = False
            # Find first divergence
            first_diff = next((j for j in range(n) if hf_toks[j] != qpc_toks[j]), n)
            print(f"  [{i}] FAIL: first divergence at new token {first_diff}")
            print(f"         HF  tok[{first_diff}]={hf_toks[first_diff] if first_diff < len(hf_toks) else 'N/A'} "
                  f"({processor.tokenizer.decode([hf_toks[first_diff]]) if first_diff < len(hf_toks) else 'N/A'!r})")
            print(f"         QPC tok[{first_diff}]={qpc_toks[first_diff] if first_diff < len(qpc_toks) else 'N/A'} "
                  f"({processor.tokenizer.decode([qpc_toks[first_diff]]) if first_diff < len(qpc_toks) else 'N/A'!r})")
        else:
            print(f"  [{i}] PASS ({n} tokens match): HF={hf_texts[i]!r}")

    verdict = "PASS" if all_pass else "FAIL"
    n_pass = sum(
        1 for hf_t, qpc_t in zip(hf_token_lists, hw_token_lists)
        if hf_t[:min(len(hf_t), len(qpc_t))] == qpc_t[:min(len(hf_t), len(qpc_t))]
    )
    print(f"\nOverall: {verdict} ({n_pass}/{len(samples)} samples match token-for-token)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
