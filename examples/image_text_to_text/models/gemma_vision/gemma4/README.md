# Gemma4 README

This folder provides runnable Gemma4 vision-language examples using QEfficient.

## Files

- `gemma4_example.py`
  - Standard end-to-end flow with `QEFFAutoModelForImageTextToText`.
  - Good starting point for most users.
- `gemma4_diss.py`
  - Disaggregated flow (separate vision and language prefill and decode sessions).
- `gemma4_utils.py`
  - Shared prompt/chat-template helpers used by both scripts.


## Run

```bash
python examples/image_text_to_text/models/gemma_vision/gemma4/gemma4_example.py
python examples/image_text_to_text/models/gemma_vision/gemma4/gemma4_diss.py
```


## Common knobs

You can adjust these values directly in the scripts:

- Model selection: `MODEL_ID` / `model_id`
- Prompt settings: `SYSTEM_PROMPT`, `TEXT_PROMPT`, `IMAGE_PROMPT`, `IMAGE_URL`
- Text-only vs vision+text mode: `SKIP_VISION` / `skip_vision`
- Sequence and generation settings: `PREFILL_SEQ_LEN`, `CTX_LEN`, `GENERATION_LEN`
- Compile/runtime settings: number of cores/devices and related compile arguments

## Notes

- First run can take longer due to compile/export.
- Keep prompt length and generation length consistent with `CTX_LEN`.
- If model access fails, verify your Hugging Face authentication and model permissions.
