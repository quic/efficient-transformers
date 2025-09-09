from pathlib import Path
import warnings

def dump_custom_io(custom_io, cache_dir, dtype_suffix):
    custom_io_yaml = Path(cache_dir) / f"custom_io_{dtype_suffix}.yaml"
    with open(custom_io_yaml, "w") as fp:
        for io_name, dtype in custom_io.items():
            fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")

def generate_custom_io(qeff_model, cache_dir=".", mxint8_kv_cache=False):
    model_class_name = type(qeff_model).__name__
    if not model_class_name == "QEFFAutoModelForCausalLM":
        output_names = qeff_model.model.get_output_names()
    kv_cache_dtype = "mxint8" if mxint8_kv_cache else "float16"
    dtype_suffix = "int8" if mxint8_kv_cache else "fp16"

    custom_io = {}

    # if model_class_name in [
    #     "QEffCausalLMForTextImageToTextModel",
    #     "QEffVisionEncoderForTextImageToTextModel"
    # ]:
    #     dump_custom_io(custom_io, cache_dir, dtype_suffix)
    #     warnings.warn(
    #         f"custom_io generated for these '{model_class_name}' class is empty.",
    #         UserWarning
    #     )

    # Dual QPC: generate two YAML files
    if model_class_name == "_QEFFAutoModelForImageTextToTextDualQPC":
        custom_io_vision = {}
        for output_name in output_names.get("vision", []):
            custom_io_vision[output_name] = kv_cache_dtype if output_name.startswith("past_") else "float16"

        custom_io_lang = {}
        for output_name in output_names.get("lang", []):
            if output_name.endswith("_RetainedState"):
                base_name = output_name[: -len("_RetainedState")]
                custom_io_lang[base_name] = "float16" if "vision_embeds" in output_name else kv_cache_dtype
                custom_io_lang[output_name] = "float16" if "vision_embeds" in output_name else kv_cache_dtype

        dump_custom_io(custom_io_vision, cache_dir, f'{dtype_suffix}_vision')
        dump_custom_io(custom_io_lang, cache_dir, f'{dtype_suffix}_lang')
        return {**custom_io_vision, **custom_io_lang}

    # Single QPC
    elif model_class_name == "_QEFFAutoModelForImageTextToTextSingleQPC":
        for input_name in output_names:
            if input_name.endswith("_RetainedState"):
                custom_io[input_name[: -len("_RetainedState")]] = (
                    "float16" if "pixel_values" in input_name else kv_cache_dtype
                )
        for output_name in output_names:
            if output_name.endswith("_RetainedState"):
                custom_io[output_name] = "float16" if "pixel_values" in output_name else kv_cache_dtype

    # Causal LM
    elif model_class_name == "QEFFAutoModelForCausalLM":
        for suffix in ["", "_RetainedState"]:
            num_layers = getattr(qeff_model, "num_layers", 12)
            for i in range(num_layers):
                for kv in ["key", "value"]:
                    custom_io[f"past_{kv}.{i}{suffix}"] = kv_cache_dtype

    # Speech Seq2Seq
    elif model_class_name == "QEFFAutoModelForSpeechSeq2Seq":
        custom_io["input_features"] = kv_cache_dtype
        for output_name in output_names:
            if output_name.endswith("_RetainedState"):
                custom_io[output_name[: -len("_RetainedState")]] = kv_cache_dtype
                custom_io[output_name] = kv_cache_dtype
    else:
        warnings.warn(f"Unsupported model class: {model_class_name}", UserWarning)
    
    dump_custom_io(custom_io, cache_dir, dtype_suffix)
    return custom_io