import pytest
import requests
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants

test_models = [
    "llava-hf/llava-1.5-7b-hf",
]


def load_image_text_to_text_model(model_config):
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForImageTextToText.from_pretrained(
        model_path,
        _attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # TODO:# Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def generate_runtime_inputs(model_name):
    generate_func = generate_runtime_func_map.get(model_name)
    if not generate_func:
        raise ValueError(f"Input generation function for model {model_name} not found.")

    return generate_func(model_name)


def generate_hf_inputs(model_name, model, processor=None):
    generate_func = generate_hf_inputs_func_map.get(model_name)
    if not generate_func:
        raise ValueError(f"Input generation function for model {model_name} not found.")

    return generate_func(model_name, model, processor)


def generate_hf_inputs_llava(model_name, model, processor=None):
    img = Image.open(requests.get(Constants.BASE_URL_LLAVA, stream=True).raw)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": Constants.PROMPT_LLAVA}, {"type": "image"}]}],
        add_generation_prompt=True,
    )
    inputs = processor(images=img, text=prompt, return_tensors="pt")
    inputs["processor"] = processor
    return inputs


def generate_runtime_inputs_llava(model_name):
    processor = AutoProcessor.from_pretrained(model_name, padding_side="right")
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": Constants.PROMPT_LLAVA,
                    },
                    {"type": "image"},
                ],
            },
        ],
        add_generation_prompt=True,
    )
    image = Image.open(requests.get(Constants.BASE_URL_LLAVA, stream=True).raw)
    inputs = processor(images=image, text=prompt, return_tensors="np", padding=True)
    inputs["prompt"] = prompt
    inputs["processor"] = processor
    streamer = TextStreamer(tokenizer)
    return inputs, streamer


generate_runtime_func_map = {
    "llava-hf/llava-1.5-7b-hf": generate_runtime_inputs_llava,
}
generate_hf_inputs_func_map = {
    "llava-hf/llava-1.5-7b-hf": generate_hf_inputs_llava,
}


def check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    prompt_len: int = Constants.PL_VLM,
    ctx_len: int = Constants.CTX_LEN_VLM,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
        :prompt_len (int): Prompt length for the model to compile.
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
    """
    model_config = {"model_name": model_name}
    # TODO:single layer
    # model_config["n_layer"] = n_layer
    breakpoint()
    model_hf, _ = load_image_text_to_text_model(model_config)
    # Load processor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, padding=True)
    inputs, streamer = generate_runtime_inputs(model_name)
    qeff_model = QEFFAutoModelForImageTextToText(model_hf, processor, is_tlm=False)
    onnx_model_path = qeff_model.export()
    _ = qeff_model.compile(
        prefill_seq_len=prompt_len,
        onnx_path=onnx_model_path,
        ctx_len=ctx_len,
        num_cores=14,
        mxfp6=False,
        aic_enable_depth_first=False,
        num_speculative_tokens=None,
    )

    exec_info = qeff_model.generate(inputs, streamer, device_ids=None, runtime_ai100=True)
    cloud_ai_100_tokens = exec_info[0]
    inputs_hf = generate_hf_inputs(model_name, model_hf, processor)
    pytorch_hf_tokens = run_model_hf_on_pytorch(model_hf, inputs_hf)
    assert cloud_ai_100_tokens == pytorch_hf_tokens, "Tokens do not match between cloud AIC and PyTorch."
    return


def run_model_hf_on_pytorch(model, inputs):
    processor = inputs.pop("processor") if "processor" in inputs else None
    output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    generated_texts = processor.decode(output[0][2:], skip_special_tokens=True)
    # FIX:TODO
    # breakpoint()
    # batch_size = inputs['input_ids'].shape[0]
    # generation_len = Constants.CTX_LEN_VLM - inputs["input_ids"].shape[1]
    # generated_ids = torch.full((batch_size, generation_len + 1), processor.tokenizer.pad_token_id)
    # streamer = TextStreamer(processor.tokenizer)
    # outputs = model(**inputs)
    # inputs["input_ids"] = torch.argmax(outputs[0][:,-1,:], dim=-1).unsqueeze(0)
    # generated_ids[:, 0] = inputs["input_ids"].squeeze(1)
    # finished_sequences = inputs["input_ids"] == processor.tokenizer.eos_token_id
    # inputs["position_ids"] = inputs["position_ids"].max(1, keepdim=True).values + 1
    # streamer.put(inputs["input_ids"][0])
    # inputs['pixel_values'] = None
    # for num_token in range(10):
    #     outputs = model(**inputs)
    #     inputs["input_ids"] = torch.argmax(outputs[0][:,-1,:], dim=-1).unsqueeze(0)
    #     inputs["position_ids"] += 1
    #     streamer.put(inputs["input_ids"][0])
    #     generated_ids[:,num_token] = inputs["input_ids"].squeeze(1)
    #     finished_sequences |= inputs["input_ids"] == processor.tokenizer.eos_token_id
    #     if finished_sequences.all():
    #         break
    # generated_texts = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # streamer.end()
    # print("text",generated_texts)
    return generated_texts


@pytest.mark.on_qaic
@pytest.mark.parametrize("model_name", test_models)
def test_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name):
    """
    Test function to validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``gpt2``
    """

    check_image_text_to_text_pytorch_vs_kv_vs_ort_vs_ai100(model_name=model_name)
