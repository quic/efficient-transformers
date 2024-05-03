# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil
from typing import Optional, Tuple

import torch
from huggingface_hub import login
from transformers import AutoTokenizer

from QEfficient.exporter.export_utils import export_onnx, fix_onnx_fp16, generate_input_files, run_model_on_ort
from QEfficient.transformers.modeling_utils import transform
from QEfficient.utils import hf_download
from QEfficient.utils.constants import QEFF_MODELS_DIR, Constants
from QEfficient.utils.logging_utils import logger


def convert_to_cloud_bertstyle(
    model_name: str,
    model_class: type = None,
    tokenizer=None,
    onnx_dir_path=None,
    hf_token: str = None,
    seq_len: int = Constants.seq_length,
    input_str: str = Constants.input_str,
    return_path: bool = False,
    save_fp32_onnx: bool = False,
    save_fp16_onnx: bool = True,
):
    """
    Function to convert the model to Bertstyle approach.
    Bertstyle Approach:
        1. No Prefill/Decode sepeartly compiled
        2. No KV retaintion logic.
        3. KV is everytime computed for all the tokens until EOS/max_length

    Args:
        model_name (str): The name of the model to be used.
        model_class (type): The class of the model.
        tokenizer (HF AutoTokenizer): Tokenzier to prepare inputs.
        model_path (str, optional): The path where the model is stored. If None, the model is loaded from the default location.
        hf_token (str): If hf_token passed, it will be used for authentication for gated. Default is None.
        seq_len (int, optional): The length of the sequence. Default is 128.
        input_str (str): The input string to be processed.
        return_path (bool): If True, return the base path for models and exported onnx model path
        save_fp32_onnx (bool); If True, fp32 unclipped version of ONNX will be saved. Default is False.
        save_fp16_onnx (bool); If false, generation of fp32 clipped version of ONNX will be skipped. Default is True.

    """
    # todo (amitraj) Optimize the onnx export
    if onnx_dir_path is None:
        model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
        onnx_dir_path = os.path.join(model_card_dir, "onnx_bertstyle")

    if os.path.exists(onnx_dir_path):
        logger.warning(f"Overriding {onnx_dir_path}")
        shutil.rmtree(onnx_dir_path)

    if not (save_fp32_onnx or save_fp16_onnx):
        raise AttributeError("save_fp32_onnx and save_fp16_onnx can't be false")

    seq_len = Constants.seq_length
    input_str = Constants.input_str

    # Load tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    else:
        if tokenizer.padding_side != "left":
            logger.warning(f"Please use padding_side='left' while initializing the tokenizer")
            tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        if hf_token:
            login(hf_token)
        model_hf_path = hf_download(
            repo_id=model_name,
            cache_dir=Constants.CACHE_DIR,
            ignore_pattrens=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf"],
        )
        model = model_class.from_pretrained(model_hf_path, cache_dir=Constants.CACHE_DIR, use_cache=True)
    except Exception as e:
        print(f"Failed to download the {model_name} model from Huggingface:%s", e)
    model.eval()

    # Decide path for saving exported ONNX files.
    model_base_name = model_name.replace("/", "_") + "_bertstyle"
    os.makedirs(onnx_dir_path, exist_ok=True)

    # Preprocess inputs
    if seq_len > 0:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(
            input_str,
            return_tensors="pt",
            padding="max_length",
            max_length=seq_len,
        )
    else:
        inputs = tokenizer(input_str, return_tensors="pt")
    if model.config.is_encoder_decoder:
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        inputs["decoder_input_ids"] = torch.full((1, 1), model.generation_config.decoder_start_token_id)

    # Run PyTorch inference
    try:
        pt_outputs = model(**inputs)
        output_names = list(pt_outputs.keys())
    except Exception as e:
        print(f"Model {model_name} Execution failed in pytorch:%s", e)

    # Add pkv into output_names
    pkv = tuple([(key.detach(), value.detach()) for key, value in pt_outputs.past_key_values])
    pkv_idx = output_names.index("past_key_values")
    key_value_names = [f"past_{x}.{i}" for i in range(len(pkv)) for x in ["key", "value"]]
    output_names[pkv_idx : pkv_idx + 1] = [x for x in key_value_names]

    pt_outputs = dict(pt_outputs)
    pkv_out = pt_outputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv_out):
        pt_outputs[f"past_key.{i}"] = key
        pt_outputs[f"past_value.{i}"] = value

    # Export the model to Onnx.
    try:
        fp32_model_name = export_onnx(
            pt_model=model,
            inputs=inputs,
            output_names=output_names,
            gen_models_path=onnx_dir_path,
            model_base_name=model_base_name,
        )
    except Exception as e:
        print(f"Model {model_name} failed to export in Onnx:%s", e)

    # Run onnxrt inference
    input_names, ort_outputs = run_model_on_ort(
        onnx_path=os.path.join(onnx_dir_path, f"{fp32_model_name}.onnx"),
        inputs=inputs,
        output_names=output_names,
        pt_outputs=pt_outputs,
    )

    # Fix onnx for fp16
    # Clip the values to fp16 ranges to avoid over/under flow in AI 100
    if save_fp16_onnx:
        fp16_model_name = fix_onnx_fp16(
            inputs=inputs,
            output_names=output_names,
            ort_outputs=ort_outputs,
            gen_models_path=onnx_dir_path,
            model_base_name=fp32_model_name,
            pt_outputs=pt_outputs,
            save_fp32_onnx=save_fp32_onnx,
        )

    # Generate inputFiles
    # todo(ochougul):rename to bert_style_input_list.txt
    input_list_file = os.path.join(onnx_dir_path, "input_list.txt")
    generate_input_files(
        input_files_path=os.path.join(onnx_dir_path, "inputFiles"),
        input_names=input_names,
        inputs=inputs,
        input_list_file=input_list_file,
    )

    # return the model path for automation.
    if return_path:
        if save_fp16_onnx:
            return onnx_dir_path, os.path.join(onnx_dir_path, f"{fp16_model_name}.onnx")
        else:
            return onnx_dir_path, os.path.join(onnx_dir_path, f"{fp32_model_name}.onnx")
    else:
        return


def convert_to_cloud_kvstyle(
    model_name: str,
    model_class: type = None,
    model_kv: torch.nn.Module = None,
    tokenizer=None,
    onnx_dir_path=None,
    hf_token: str = None,
    seq_len: int = Constants.seq_length,
    input_str: str = Constants.input_str,
    return_path: bool = False,
    save_fp32_onnx: bool = False,
    save_fp16_onnx: bool = True,
):
    """
    Function Modeling changes for kv retention and export to Onnx.
    KV Style Approach:
        1. This architecture is particularly suitable for autoregressive tasks
        2. where sequence generation involves processing one token at a time
        3. And contextual information from earlier tokens is crucial for predicting the next token.
        4. The inclusion of a kV cache enhances the efficiency of the decoding process, making it more computationally efficient.

    Args:
        model_name (str): The name of the model to be used.
        model_class (type): The class of the model.
        model_kv (torch.nn.Module): Transformed KV torch model to be used
        tokenizer (HF AutoTokenizer): Tokenzier to prepare inputs.
        onnx_dir_path (str, optional): The path where the model is stored. If None, the model is loaded from the default location.
        hf_token (str): If hf_token passed, it will be used for authentication for gated. Default is None.
        seq_len (int, optional): The length of the sequence. Default is 128.
        input_str (str): The input string to be processed.
        return_path (bool): If True, return the base path for models and exported onnx model path
        save_fp32_onnx (bool); If True, fp32 unclipped version of ONNX will be saved. Default is False.
        save_fp16_onnx (bool); If false, generation of fp32 clipped version of ONNX will be skipped. Default is True.

    """
    if onnx_dir_path is None:
        model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
        onnx_dir_path = os.path.join(model_card_dir, "onnx")

    if os.path.exists(onnx_dir_path):
        logger.warning(f"Overriding {onnx_dir_path}")
        shutil.rmtree(onnx_dir_path)

    if not (save_fp32_onnx or save_fp16_onnx):
        raise AttributeError("save_fp32_onnx and save_fp16_onnx can't be false")

    if model_class is None and model_kv is None:
        raise AttributeError("model_class and model_kv both can't be None")

    if model_kv is not None:
        if not getattr(model_kv, "qeff_transformed", False):
            raise AttributeError(
                "Model is not transformed, Please first use QEfficient.transform to tranform the model."
            )
        model = model_kv
    else:
        try:
            if hf_token:
                login(hf_token)
            model_hf_path = hf_download(
                repo_id=model_name,
                cache_dir=Constants.CACHE_DIR,
                ignore_pattrens=["*.txt", "*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf"],
            )
            model = model_class.from_pretrained(model_hf_path, cache_dir=Constants.CACHE_DIR, use_cache=True)
        except Exception as e:
            print(f"Failed to download the {model_name} model from Huggingface:%s", e)
        transform(model, form_factor="cloud")

    # Decide path for saving exported ONNX files.
    model_base_name = model_name.replace("/", "_") + "_kv"
    os.makedirs(onnx_dir_path, exist_ok=True)

    # Load tokenizer
    if tokenizer is None:
        # todo(ochougul): use cache dir from snapshot download
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    else:
        if tokenizer.padding_side != "left":
            logger.warning(f"Please use padding_side='left' while initializing the tokenizer")
            tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Disabling requires_grad on all parameters
    for j, p in enumerate(model.parameters()):
        p.requires_grad_(False)

    # Preprocess inputs
    input_str = ["My name is Sarah.", "I live in London."]
    if seq_len > 0:
        inputs = tokenizer(input_str, return_tensors="pt", padding=True)
        batch_size, prompt_len = inputs["input_ids"].shape
        inputs["input_ids"] = torch.concat(
            [
                inputs["input_ids"],
                torch.full((batch_size, seq_len - prompt_len), tokenizer.pad_token_id),
            ],
            1,
        )
        inputs["attention_mask"] = torch.concat(
            [
                inputs["attention_mask"],
                torch.zeros((batch_size, seq_len - prompt_len), dtype=torch.int64),
            ],
            1,
        )
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(1) - 1) * inputs["attention_mask"]
    else:
        inputs = tokenizer(input_str, return_tensors="pt")

    try:
        pt_outputs = model(**inputs)
        output_names = list(pt_outputs.keys())
    except Exception as e:
        print(f"Model {model_name} Execution failed in pytorch:%s", e)

    # Raise error if expected outputs are not present
    assert "logits" in output_names, "logits not found in output"
    assert "past_key_values" in output_names, "past_key_values not found in output"

    # Build inputs for next iteration from outputs
    cache_index = torch.tensor(prompt_len)
    inputs["input_ids"] = tokenizer(["I have"] * 2, return_tensors="pt").input_ids[:, -2:]
    inputs["position_ids"] = inputs["attention_mask"].sum(1, keepdim=True)
    inputs["position_ids"] = inputs["position_ids"].repeat(1, 2) + torch.arange(2).view(1, 2)
    inputs["attention_mask"] = inputs["attention_mask"].bool()
    inputs["cache_index"] = cache_index

    # Add past_key_values into inputs
    inputs["past_key_values"] = tuple([(key.detach(), value.detach()) for key, value in pt_outputs.past_key_values])

    # Run PyTorch inference with past
    try:
        pt_outputs = model(**inputs)
        output_names = list(pt_outputs.keys())
    except Exception as e:
        print(f"Model {model_name} Execution failed in pytorch:%s", e)

    # Add pkv into output_names
    pkv = tuple([(key.detach(), value.detach()) for key, value in pt_outputs.past_key_values])
    pkv_idx = output_names.index("past_key_values")
    key_value_names = [f"past_{x}.{i}" for i in range(len(pkv)) for x in ["key", "value"]]
    output_names[pkv_idx : pkv_idx + 1] = [x + "_RetainedState" for x in key_value_names]

    # Replace nested past_key_values outputs with separate KV tensors
    pt_outputs = dict(pt_outputs)
    pkv_out = pt_outputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv_out):
        pt_outputs[f"past_key.{i}_RetainedState"] = key
        pt_outputs[f"past_value.{i}_RetainedState"] = value

    # Export and simplify ONNX model
    fp32_model_name = export_onnx(
        pt_model=model,
        inputs=inputs,
        output_names=output_names,
        gen_models_path=onnx_dir_path,
        model_base_name=model_base_name,
    )

    # Replace nested past_key_values inputs with separate KV tensors
    inputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv):
        inputs[f"past_key.{i}"] = key
        inputs[f"past_value.{i}"] = value

    # Run onnxrt inference
    input_names, ort_outputs = run_model_on_ort(
        onnx_path=os.path.join(onnx_dir_path, f"{fp32_model_name}.onnx"),
        inputs=inputs,
        output_names=output_names,
        pt_outputs=pt_outputs,
    )

    # Fix onnx for FP16
    if save_fp16_onnx:
        fp16_model_name = fix_onnx_fp16(
            inputs=inputs,
            output_names=output_names,
            ort_outputs=ort_outputs,
            gen_models_path=onnx_dir_path,
            model_base_name=fp32_model_name,
            pt_outputs=pt_outputs,
            save_fp32_onnx=save_fp32_onnx,
        )

    # Generate custom-IO files
    with open(os.path.join(onnx_dir_path, "custom_io.yaml"), "w") as fp:
        fp.write("# Model Inputs\n\n")
        for input_name in key_value_names:
            fp.write(f" - IOName: {input_name}\n   Precision: float16\n\n")
            inputs[input_name] = inputs[input_name].to(torch.float16)
        fp.write("# Model Outputs\n\n")
        for output_name in key_value_names:
            fp.write(f" - IOName: {output_name}_RetainedState\n   Precision: float16\n\n")

    # Generate inputFiles
    input_list_file = os.path.join(onnx_dir_path, "input_list.txt")
    generate_input_files(
        input_files_path=os.path.join(onnx_dir_path, "inputFiles"),
        input_names=input_names,
        inputs=inputs,
        input_list_file=input_list_file,
    )

    # return the model path for automation.
    if return_path:
        if save_fp16_onnx:
            return onnx_dir_path, os.path.join(onnx_dir_path, f"{fp16_model_name}.onnx")
        else:
            return onnx_dir_path, os.path.join(onnx_dir_path, f"{fp32_model_name}.onnx")
    else:
        return


def convert_to_edge(self) -> None:
    # [TODO]: Apply the class transformation to make changes for the KV models in edge use cases
    # model = QEfficient.transform(model_hf, type="Transformers", form_factor="edge")
    # model.eval()
    raise NotImplementedError("Oops...reached too far!!")


def qualcomm_efficient_converter(
    model_name: str,
    model_class: type = None,
    model_kv: torch.nn.Module = None,
    tokenizer=None,
    onnx_dir_path=None,
    hf_token: str = "",
    seq_length: int = Constants.seq_length,
    input_str: str = Constants.input_str,
    kv: bool = True,
    return_path: bool = False,
    form_factor="cloud",
    save_fp32_onnx: bool = False,
    save_fp16_onnx: bool = True,
) -> Optional[Tuple[str, str]]:
    """
    Function to convert the input string using the specified model and returns the result.

    Args:
        model_name (str): The name of the model to be used.
        model_class (type): The class of the model.
        model_kv (torch.nn.Module): Transformed KV torch model to be used
        tokenizer (HF AutoTokenizer): Tokenzier to prepare inputs.
        onnx_dir_path (str, optional): The path where the model is stored. If None, the model is loaded from the default location.
        token (bool): If True, an authentication token will be used. Default is False.
        seq_len (int, optional): The length of the sequence. Default is 128.
        input_str (str): The input string to be processed.
        kv (bool): If True, key-value pairs will be used. Default is True.
        return_path (bool): If True, return the base path for models and exported onnx model path
        save_fp32_onnx (bool); If True, fp32 unclipped version of ONNX will be saved. Default is False.
        save_fp16_onnx (bool); If false, fp32 unclipped version of ONNX will be deleted. Default is False.

    Returns:
        None, if automation is False, else path to exported Onnx file

    """
    if model_kv is not None and not kv:
        raise AttributeError("For Transformed model kv must be True")

    if form_factor == "cloud":
        if kv:
            return convert_to_cloud_kvstyle(
                model_name=model_name,
                model_class=model_class,
                model_kv=model_kv,
                onnx_dir_path=onnx_dir_path,
                tokenizer=tokenizer,
                hf_token=hf_token,
                seq_len=seq_length,
                input_str=input_str,
                return_path=return_path,
                save_fp32_onnx=save_fp32_onnx,
                save_fp16_onnx=save_fp16_onnx,
            )
        else:
            return convert_to_cloud_bertstyle(
                model_name=model_name,
                model_class=model_class,
                tokenizer=tokenizer,
                onnx_dir_path=onnx_dir_path,
                hf_token=hf_token,
                seq_len=seq_length,
                input_str=input_str,
                return_path=return_path,
                save_fp32_onnx=save_fp32_onnx,
                save_fp16_onnx=save_fp16_onnx,
            )
    else:
        return convert_to_edge()
