# Qwen3-ASR Changes

## Overview

Added a standalone `qwen_asr_onefile.py` workflow for Qwen3-ASR model compilation
and Cloud AI 100 inference.

## Added

- Qwen3-ASR model support using `Qwen/Qwen3-ASR-0.6B-hf`.
- One-file `compile`, `run`, `compile-run`, and `benchmark` commands.
- QPC compilation with configurable chunk length, decoder context length,
  core count, and device IDs.
- WAV, FLAC, OGG, AIFF, and optional video/audio conversion through `ffmpeg`.
- Qwen ASR chat-template processor input construction.
- Prefill padding and retained-state carry-over for decode iterations.
- EOS-based generation stopping, with an option for fixed-length decoding.
- Runtime chunking with configurable overlap.
- JSON and CSV benchmark output.
- Explicit rejection of device ID `43`.

## Usage

Compile and run:

```bash
python qwen_asr_onefile.py compile-run \
  --audio-file /path/to/audio.flac \
  --chunk-seconds 30 \
  --ctx-len 512 \
  --generation-len 128 \
  --num-cores 8 \
  --device-ids 0
```

Run an existing QPC:

```bash
python qwen_asr_onefile.py run \
  --manifest /path/to/qpc_manifest.json \
  --audio-file /path/to/audio.flac \
  --generation-len 128 \
  --device-ids 0
```

## Generated artifacts

Compilation writes a manifest, QPC path file, and QEfficient build output under
the selected output directory. QPC binaries are generated artifacts and should
not be committed to the source repository.

## Verification note

The script has passed syntax and runtime smoke checks on Cloud AI 100. A smoke
run or coherent transcript is not a numerical parity result; correctness claims
require the appropriate HF/QEfficient/ORT/QPC parity evidence.
