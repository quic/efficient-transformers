# Cohere ASR on Cloud AI 100

`cohere_asr_inference.py` compiles a Cohere ASR QPC when no QPC is supplied,
then transcribes a local audio file. It can also reuse an existing QPC without
compiling again.

## Prerequisites

- A Cloud AI 100 host with the Qualcomm AI SDK installed and `qaic-compile` on
  `PATH`.
- A QEfficient Python environment.
- An accessible AI 100 device. The examples below use device `0`.
- Access to `CohereLabs/cohere-transcribe-03-2026` on Hugging Face.

Clone the fork containing Cohere ASR support and activate the QEff environment:

```bash
git clone <efficient-transformers-fork-url>
cd efficient-transformers
git checkout feature/add-cohere_asr

source <qeff-venv>/bin/activate
export QEFF_HOME="$PWD/.qeff_cohere"
```

## Compile and transcribe

This command exports the model, compiles a batch-1 QPC, and transcribes the
specified WAV or FLAC audio file. The generated QPC path is printed as `qpc=`.

```bash
python examples/audio/cohere_asr_inference.py \
  --model-name CohereLabs/cohere-transcribe-03-2026 \
  --language en \
  --audio-path /path/to/audio.wav \
  --compile-dir "$PWD/cohere_asr_compile" \
  --batch-size 1 \
  --num-cores 16 \
  --encoder-ctx-len 438 \
  --ctx-len 512 \
  --generation-len 500
```

The compile configuration above uses one device, batch size 1, 16 QPC cores,
a 438-token encoder context, and a 512-token decoder context. Audio longer
than one processor window is processed as sequential batch-1 chunks.

## Reuse an existing QPC

Pass the QPC directory printed by the compile command to skip export and
compilation:

```bash
python examples/audio/cohere_asr_inference.py \
  --model-name CohereLabs/cohere-transcribe-03-2026 \
  --language en \
  --audio-path /path/to/audio.wav \
  --qpc-path /path/to/generated/qpc \
  --generation-len 500
```

## Optional transcript comparison

Provide a known transcript to print exact and normalized text-match results:

```bash
  --reference-text "expected transcript text"
```

The script prints the QPC path, audio path, processor chunk count, generated
transcript, EOS status, and generated token IDs.
