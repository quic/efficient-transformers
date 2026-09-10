# Qwen3-ASR Transformers Changes

This PR adds the Transformers integration needed to compile and execute
`Qwen/Qwen3-ASR-0.6B-hf` through QEfficient.

## Transformers-side changes

- Loads Qwen3-ASR with `AutoModelForSpeechSeq2Seq`.
- Uses `QEFFAutoModelForSpeechSeq2Seq` to create the QEfficient model wrapper.
- Loads the matching `AutoProcessor` and `AutoConfig` from the same model ID.
- Builds processor input with the Qwen ASR audio chat template:
  `processor.apply_chat_template(..., add_generation_prompt=True)`.
- Uses eager attention during model loading for the Qwen3-ASR export path.
- Uses float32 model weights during export and applies QEfficient's Transformers
  quantizer replacement before compilation.
- Converts processor position IDs and input-feature masks to the integer types
  expected by the QEfficient/QPC graph.
- Handles the model's audio-feature context and audio-token sizing when setting
  the compiled encoder and decoder context lengths.

## Compatibility

- The model ID is `Qwen/Qwen3-ASR-0.6B-hf`.
- The environment must provide a Transformers version compatible with the
  installed QEfficient checkout and the Qwen3-ASR architecture.
- No upstream Transformers source files were modified by this PR; the integration
  is implemented through the standalone `qwen_asr_onefile.py` workflow.

## Verification note

Syntax and Cloud AI 100 smoke checks passed. These checks do not establish
numerical parity; correctness requires HF/QEfficient/ORT/QPC parity evidence.
