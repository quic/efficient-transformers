from gemma4_utils import (
    CHAT_TEMPLATE,
    build_compile_kwargs,
    build_messages,
    effective_lens,
    normalize_generated_ids,
    remove_fp16clip_transform_if_disabled,
    resolve_npi_mode,
)
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "google/gemma-4-E2B-it"
SYSTEM_PROMPT = "You are a helpful assistant."
TEXT_PROMPT = "Tell me about Taj Mahal?"
IMAGE_PROMPT = "Can you Describe this image in detail?"
IMAGE_URL = "https://wallup.net/wp-content/uploads/2017/03/28/351036-San_Francisco-USA-bridge-sunset-Golden_Gate_Bridge-lights.jpg"

SKIP_VISION = False
ENABLE_NPI = True
DISABLE_NPI = False
ENABLE_FP16_CLIP = True

PREFILL_SEQ_LEN = 128
CTX_LEN = 2048
GENERATION_LEN = 1920

NUM_CORES = 16
NUM_DEVICES = 2
MOS = 1

USE_ONNX_SUBFUNCTIONS = False
VISION_USE_ONNX_SUBFUNCTIONS = False
LANG_USE_ONNX_SUBFUNCTIONS = False

MXFP6_MATMUL = True
MXINT8_KV_CACHE = True
AIC_ENABLE_DEPTH_FIRST = True
compiler_kwargs = {
    "NUM_CORES": NUM_CORES,
    "NUM_DEVICES": NUM_DEVICES,
    "MXFP6_MATMUL": MXFP6_MATMUL,
    "MXINT8_KV_CACHE": MXINT8_KV_CACHE,
    "AIC_ENABLE_DEPTH_FIRST": AIC_ENABLE_DEPTH_FIRST,
    "MOS": MOS,
    "USE_ONNX_SUBFUNCTIONS": USE_ONNX_SUBFUNCTIONS,
    "VISION_USE_ONNX_SUBFUNCTIONS": VISION_USE_ONNX_SUBFUNCTIONS,
    "LANG_USE_ONNX_SUBFUNCTIONS": LANG_USE_ONNX_SUBFUNCTIONS,
}


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer
    chat_template = (
        getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None) or CHAT_TEMPLATE
    )

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,
        skip_vision=SKIP_VISION,
    )

    if SKIP_VISION:
        messages = build_messages(SYSTEM_PROMPT, TEXT_PROMPT, use_image=False)
        text_inputs = processor.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        remove_fp16clip_transform_if_disabled(qeff_model, ENABLE_FP16_CLIP)
        npi_mode = resolve_npi_mode(ENABLE_NPI, DISABLE_NPI)
        prompt_len = int(text_inputs["input_ids"].shape[1])
        effective_prefill_seq_len, effective_ctx_len = effective_lens(
            qeff_model,
            PREFILL_SEQ_LEN,
            CTX_LEN,
            prompt_len,
            GENERATION_LEN,
            SKIP_VISION,
        )

        compile_kwargs = build_compile_kwargs(
            effective_prefill_seq_len=effective_prefill_seq_len,
            effective_ctx_len=effective_ctx_len,
            skip_vision=SKIP_VISION,
            npi_mode=npi_mode,
            **compiler_kwargs,
        )
        qeff_model.compile(**compile_kwargs)

        output = qeff_model.generate(inputs=text_inputs, generation_len=GENERATION_LEN)
        qeff_ids = normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
        print(tokenizer.batch_decode(qeff_ids, skip_special_tokens=True))
        print(output)
        return
    remove_fp16clip_transform_if_disabled(qeff_model, ENABLE_FP16_CLIP)
    npi_mode = resolve_npi_mode(ENABLE_NPI, DISABLE_NPI)
    messages = build_messages(SYSTEM_PROMPT, IMAGE_PROMPT, use_image=True)
    messages[-1]["content"][0]["url"] = IMAGE_URL

    inputs = processor.apply_chat_template(
        messages,
        chat_template=chat_template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_len = int(inputs["input_ids"].shape[1])
    effective_prefill_seq_len, effective_ctx_len = effective_lens(
        qeff_model,
        PREFILL_SEQ_LEN,
        CTX_LEN,
        prompt_len,
        GENERATION_LEN,
        SKIP_VISION,
    )

    compile_kwargs = build_compile_kwargs(
        effective_prefill_seq_len=effective_prefill_seq_len,
        effective_ctx_len=effective_ctx_len,
        skip_vision=SKIP_VISION,
        npi_mode=npi_mode,
        **compiler_kwargs,
    )
    qeff_model.compile(**compile_kwargs)

    output = qeff_model.generate(
        inputs=inputs,
        generation_len=GENERATION_LEN,
    )
    qeff_ids = normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
    print(tokenizer.batch_decode(qeff_ids, skip_special_tokens=True))
    print(output)


if __name__ == "__main__":
    main()
