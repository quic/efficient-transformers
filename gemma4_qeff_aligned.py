from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

CHAT_TEMPLATE = """
{%- for message in messages %}
    {%- if loop.index0 == 0 %}
        {{- bos_token }}
    {%- endif %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {%- if message['content'] is string %}
        {{- message['content'] }}
    {%- else %}
        {%- for content in message['content'] %}
            {%- if content['type'] == 'image' %}
                {{- '<|image|>' }}
            {%- elif content['type'] == 'text' %}
                {{- content['text'] }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""

MODEL_ID = "google/gemma-4-31b"
SYSTEM_PROMPT = "You are a helpful assistant."
TEXT_PROMPT = "Tell me about Taj Mahal?"
IMAGE_PROMPT = "Can you Describe this image in detail?"
IMAGE_URL = (
    "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/"
    "Demos/sample-data/GoldenGate.png"
)

SKIP_VISION = False
ENABLE_NPI = True
DISABLE_NPI = True
ENABLE_FP16_CLIP = True

PREFILL_SEQ_LEN = 128
CTX_LEN = 2048
GENERATION_LEN = 1920

NUM_CORES = 16
NUM_DEVICES = 2
DEVICE_IDS = [0, 1]
MOS = 1

USE_ONNX_SUBFUNCTIONS = True
VISION_USE_ONNX_SUBFUNCTIONS = False
LANG_USE_ONNX_SUBFUNCTIONS = True

MXFP6_MATMUL = True
MXINT8_KV_CACHE = True
AIC_ENABLE_DEPTH_FIRST = True


def build_messages(system_prompt: str, user_prompt: str, use_image: bool):
    messages = []
    if system_prompt and not use_image:
        messages.append({"role": "system", "content": system_prompt})

    if use_image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": user_prompt})

    return messages


def resolve_npi_mode(enable_npi: bool, disable_npi: bool) -> str:
    return "enabled" if enable_npi else "disabled" if disable_npi else "auto"


def build_compile_kwargs(
    *,
    effective_prefill_seq_len: int,
    effective_ctx_len: int,
    skip_vision: bool,
    npi_mode: str,
):
    kwargs = {
        "prefill_seq_len": effective_prefill_seq_len,
        "ctx_len": effective_ctx_len,
        "num_cores": NUM_CORES,
        "num_devices": NUM_DEVICES,
        "mxfp6_matmul": MXFP6_MATMUL,
        "mxint8_kv_cache": MXINT8_KV_CACHE,
        "aic_enable_depth_first": AIC_ENABLE_DEPTH_FIRST,
        "mos": MOS,
        "use_onnx_subfunctions": USE_ONNX_SUBFUNCTIONS,
        "vision_use_onnx_subfunctions": VISION_USE_ONNX_SUBFUNCTIONS,
        "lang_use_onnx_subfunctions": LANG_USE_ONNX_SUBFUNCTIONS,
    }

    if skip_vision:
        kwargs["skip_vision"] = True
        kwargs["convert_to_fp16"] = True

    if npi_mode == "enabled":
        kwargs["node_precision_info"] = True
    elif npi_mode == "disabled":
        kwargs["node_precision_info"] = False
        if not skip_vision:
            kwargs["vision_node_precision_info"] = False

    return kwargs


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = processor.tokenizer

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype="float32",
        kv_offload=True,
        skip_vision=SKIP_VISION,
    )
    qeff_model.model.remove_fp16clip_transform_if_disabled(ENABLE_FP16_CLIP)
    npi_mode = resolve_npi_mode(ENABLE_NPI, DISABLE_NPI)

    if SKIP_VISION:
        messages = build_messages(SYSTEM_PROMPT, TEXT_PROMPT, use_image=False)
        text_inputs = processor.apply_chat_template(
            messages,
            chat_template=CHAT_TEMPLATE,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        prompt_len = int(text_inputs["input_ids"].shape[1])
        effective_prefill_seq_len, effective_ctx_len = qeff_model.model.effective_lens(
            PREFILL_SEQ_LEN,
            CTX_LEN,
            prompt_len,
            GENERATION_LEN,
            SKIP_VISION,
        )

        compile_kwargs = build_compile_kwargs(
            effective_prefill_seq_len=effective_prefill_seq_len,
            effective_ctx_len=effective_ctx_len,
            skip_vision=True,
            npi_mode=npi_mode,
        )
        qeff_model.compile(**compile_kwargs)

        output = qeff_model.generate(inputs=text_inputs, generation_len=GENERATION_LEN)
        qeff_ids = qeff_model.model.normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
        print(tokenizer.batch_decode(qeff_ids, skip_special_tokens=True))
        print(output)
        return

    messages = build_messages(SYSTEM_PROMPT, IMAGE_PROMPT, use_image=True)
    messages[-1]["content"][0]["url"] = IMAGE_URL

    inputs = processor.apply_chat_template(
        messages,
        chat_template=CHAT_TEMPLATE,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_len = int(inputs["input_ids"].shape[1])
    effective_prefill_seq_len, effective_ctx_len = qeff_model.model.effective_lens(
        PREFILL_SEQ_LEN,
        CTX_LEN,
        prompt_len,
        GENERATION_LEN,
        SKIP_VISION,
    )

    compile_kwargs = build_compile_kwargs(
        effective_prefill_seq_len=effective_prefill_seq_len,
        effective_ctx_len=effective_ctx_len,
        skip_vision=False,
        npi_mode=npi_mode,
    )
    qeff_model.compile(**compile_kwargs)
    
    output = qeff_model.generate(
        inputs=inputs,
        generation_len=GENERATION_LEN,
        device_ids=DEVICE_IDS,
    )
    qeff_ids = qeff_model.model.normalize_generated_ids(output.generated_ids)[:, :GENERATION_LEN]
    print(tokenizer.batch_decode(qeff_ids, skip_special_tokens=True))
    print(output)


if __name__ == "__main__":
    main()
