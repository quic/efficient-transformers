import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, TextStreamer

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForImageTextToText
from QEfficient.utils import hf_download


def load_vlm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=True,
        num_hidden_layers=model_config["n_layer"],
        _attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params


def _generate_inputs(model, processor):
    ## PREPROCESSING THE MULTI-MODAL INPUTS
    images = []
    placeholder = ""

    # Note: if OOM, you might consider reduce number of frames in this example.
    for i in range(1, 2):
        url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
        images.append(Image.open(requests.get(url, stream=True).raw))
        placeholder += f"<|image_{1}|>\n"

    messages = [
        {"role": "user", "content": placeholder + "Summarize the deck of slides."},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # padding="max_length",
        # max_length=seq_len
    )
    model.config.hidden_size // model.config.num_attention_heads
    # ctx_len = 1280  # FIXME: Pass it a ssome arguement later on
    inputs = dict(processor(images=images, text=prompt, return_tensors="pt"))
    inputs["position_ids"] = inputs.pop("attention_mask").cumsum(1)
    # inputs["past_key_values"] = []
    # for i in range(model.config.num_hidden_layers):
    #     inputs["past_key_values"].append((
    #         torch.zeros(1, model.config.num_key_value_heads, ctx_len, head_dim),
    #         torch.zeros(1, model.config.num_key_value_heads, ctx_len, head_dim),
    #     ))
    return inputs


def check_vlm_pytorch_vs_kv_vs_ort_vs_ai100(
    model_name: str,
    prompt_len: int = 1024,
    ctx_len: int = 1280,
    n_layer: int = 32,
    # num_speculative_tokens: Optional[int] = None,
):
    """
    Validate the PyTorch model, the PyTorch model after KV changes, the ONNX model, and the Cloud AI 100 model, both with and without continuous batching.
    ``Mandatory`` Args:
        :model_name (str): Hugging Face Model Card name, Example: ``Phi-3.5-vision-instruct``
        :prompt_len (int): Prompt length for the model to compile.
        :ctx_len (int): Maximum context length to compile the model.
        :n_layers (int): Number of layers for the Model.
    """
    # replace_transformers_quantizers()
    model_config = {"model_name": model_name}
    model_config["n_layer"] = n_layer

    model_hf, _ = load_vlm_model(model_config)
    # Load processor instead
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    # config = model_hf.config
    # batch_size = len(Constants.INPUT_STR)
    streamer = TextStreamer(processor)
    # Testing for Phi-3.5 only atm
    inputs = _generate_inputs(model_hf, processor)
    # Original PyTorch model
    pt_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        num_hidden_layers=n_layer,
        _attn_implementation="eager",
        trust_remote_code=True,
        # Check if this works
        rope_scaling=None,
    )
    # TODO: Don't use API RUNNER CLASS BUT directly define those functions here
    # pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)

    # is_tlm = False if num_speculative_tokens is None else True
    qeff_model = QEFFAutoModelForImageTextToText(pt_model, processor, is_tlm=False)
    breakpoint()
    # pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

    # assert (
    #     pytorch_hf_tokens == pytorch_kv_tokens
    # ).all(), "Tokens don't match for HF PyTorch model output and KV PyTorch model output"

    qeff_model.export()

    # ort_tokens = api_runner.run_kv_model_on_ort(onnx_model_path, is_tlm=is_tlm)

    # assert (pytorch_kv_tokens == ort_tokens).all(), "Tokens don't match for ONNXRT output and PyTorch output."

    # if not get_available_device_id():
    #     pytest.skip("No available devices to run model on Cloud AI 100")
    breakpoint()
    _ = qeff_model.compile(
        prefill_seq_len=prompt_len,
        ctx_len=ctx_len,
        num_cores=14,
        mxfp6=False,
        aic_enable_depth_first=False,
        # num_speculative_tokens=num_speculative_tokens,
    )
    exec_info = qeff_model.generate(inputs, streamer, device_ids=None, runtime_ai100=True)
    exec_info[0]  # Because we always run for single input and single batch size
    # gen_len = ort_tokens.shape[-1]
    # assert (
    #     ort_tokens == cloud_ai_100_tokens[:, :gen_len]
    # ).all(), "Tokens don't match for ONNXRT output and Cloud AI 100 output."


check_vlm_pytorch_vs_kv_vs_ort_vs_ai100("microsoft/Phi-3.5-vision-instruct", 1024)
