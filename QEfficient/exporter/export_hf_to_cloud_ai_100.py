# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil
import warnings
from typing import Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

import QEfficient
from QEfficient.base.common import AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP, QEFF_MODEL_TYPE, QEFFCommonLoader
from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.exporter.export_utils import export_onnx, fix_onnx_fp16, generate_input_files, run_model_on_ort
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import load_hf_tokenizer
from QEfficient.utils.constants import QEFF_MODELS_DIR, Constants
from QEfficient.utils.generate_inputs import InputHandler
from QEfficient.utils.logging_utils import logger


def convert_to_cloud_bertstyle(
    model_name: str,
    qeff_model: QEFFAutoModelForCausalLM,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_len: int,
) -> str:
    """
    Function to convert the model to Bertstyle approach.
    Bertstyle Approach:
        1. No Prefill/Decode sepeartly compiled
        2. No KV retaintion logic.
        3. KV is everytime computed for all the tokens until EOS/max_length

    Args:
        model_name (str): The name of the model to be used.
        qeff_model (QEFFBaseModel): Transformed KV torch model to be used
        tokenizer (HF AutoTokenizer): Tokenzier to prepare inputs.
        onnx_dir_path (str, optional): The path where the model is stored. If None, the model is loaded from the default location.
        seq_len (int, optional): The length of the sequence. Default is 128.
    """
    if os.path.exists(onnx_dir_path):
        logger.warning(f"Overriding {onnx_dir_path}")
        shutil.rmtree(onnx_dir_path)

    # Decide path for saving exported ONNX files.
    model_name = export_bertstyle_model_to_onnx(model_name, qeff_model.model, tokenizer, onnx_dir_path, seq_len)  # type: ignore

    # return the model path for automation.
    return os.path.join(onnx_dir_path, f"{model_name}.onnx")


def export_bertstyle_model_to_onnx(model_name, model, tokenizer, onnx_dir_path, seq_len) -> str:
    model_base_name = model_name.replace("/", "_") + "_bertstyle"
    os.makedirs(onnx_dir_path, exist_ok=True)

    input_str = Constants.INPUT_STR
    # Preprocess inputs
    if seq_len > 0:
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
        model_name = export_onnx(
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
        onnx_path=os.path.join(onnx_dir_path, f"{model_name}.onnx"),
        inputs=inputs,
        output_names=output_names,
        pt_outputs=pt_outputs,
    )

    # Fix onnx for fp16
    # Clip the values to fp16 ranges to avoid over/under flow in AI 100
    model_name = fix_onnx_fp16(
        inputs=inputs,
        output_names=output_names,
        ort_outputs=ort_outputs,
        gen_models_path=onnx_dir_path,
        model_base_name=model_name,
        pt_outputs=pt_outputs,
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

    return model_name


def convert_to_cloud_kvstyle(
    model_name: str,
    qeff_model: QEFFAutoModelForCausalLM,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_len: int,
) -> str:
    """
    Function Modeling changes for kv retention and export to Onnx.
    KV Style Approach:
        1. This architecture is particularly suitable for autoregressive tasks
        2. where sequence generation involves processing one token at a time
        3. And contextual information from earlier tokens is crucial for predicting the next token.
        4. The inclusion of a kV cache enhances the efficiency of the decoding process, making it more computationally efficient.

    Args:
        model_name (str): The name of the model to be used.
        qeff_model (QEFFBaseModel): Transformed KV torch model to be used
        tokenizer (HF AutoTokenizer): Tokenzier to prepare inputs.
        onnx_dir_path (str, optional): The path where the model is stored. If None, the model is loaded from the default location.
        seq_len (int, optional): The length of the sequence. Default is 128.
    """
    warnings.warn(
        "\033[93mThis function will be deprecated soon, use QEfficient.export instead\033[0m",
        DeprecationWarning,
        stacklevel=2,
    )
    if os.path.exists(onnx_dir_path):
        logger.warning(f"Overriding {onnx_dir_path}")
        shutil.rmtree(onnx_dir_path)

    assert qeff_model.is_transformed, f"please pass the {qeff_model.__class__.__name__} after transform API"

    # Decide path for saving exported ONNX files.
    model_name = export_kvstyle_transformed_model_to_onnx(
        model_name, qeff_model.model, tokenizer, onnx_dir_path, seq_len
    )  # type: ignore

    # return the model path for automation.
    return os.path.join(onnx_dir_path, f"{model_name}.onnx")


def export_kvstyle_transformed_model_to_onnx(
    model_name: str,
    transformed_model: torch.nn.Module,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_len: int,
) -> str:
    # Disabling requires_grad on all parameters
    for _, p in enumerate(transformed_model.parameters()):
        p.requires_grad_(False)

    assert seq_len > 0, "Need seq_len to be greater than zero"

    # Preprocess inputs
    # Build inputs for prefill
    input_handler = InputHandler(tokenizer=tokenizer,
                                 config=transformed_model.config,
                                 prompt=Constants.INPUT_STR,
                                 prompt_len=Constants.PROMPT_LEN,
                                 ctx_len=seq_len)
    inputs = input_handler.prepare_pytorch_inputs()

    pt_outputs = transformed_model(**inputs)
    output_names = list(pt_outputs.keys())

    # Raise error if expected outputs are not present
    assert "logits" in output_names, "logits not found in output"
    assert "past_key_values" in output_names, "past_key_values not found in output"

    # Build inputs for next iteration from outputs
    # Build inputs for decode
    inputs = input_handler.update_pytorch_inputs(inputs, pt_outputs)

    # To avoid issues in onnx export
    inputs["position_ids"] = torch.full((1, 1), seq_len - 1)

    # Run PyTorch inference with past
    pt_outputs = transformed_model(**inputs)
    output_names = list(pt_outputs.keys())

    # Add pkv into output_names
    pkv = inputs["past_key_values"]
    pkv_idx = output_names.index("past_key_values")
    key_value_names = [f"past_{x}.{i}" for i in range(len(pkv)) for x in ["key", "value"]]
    output_names[pkv_idx : pkv_idx + 1] = [x + "_RetainedState" for x in key_value_names]

    # Replace nested past_key_values outputs with separate KV tensors
    pt_outputs = dict(pt_outputs)
    pkv_out = pt_outputs.pop("past_key_values")
    for i, (key, value) in enumerate(pkv_out):
        pt_outputs[f"past_key.{i}_RetainedState"] = key
        pt_outputs[f"past_value.{i}_RetainedState"] = value

    model_base_name = model_name.replace("/", "_") + "_kv"
    os.makedirs(onnx_dir_path, exist_ok=True)

    # Export and simplify ONNX model
    model_name = export_onnx(
        pt_model=transformed_model,
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
        onnx_path=os.path.join(onnx_dir_path, f"{model_name}.onnx"),
        inputs=inputs,
        output_names=output_names,
        pt_outputs=pt_outputs,
    )

    model_name = fix_onnx_fp16(
        inputs=inputs,
        output_names=output_names,
        ort_outputs=ort_outputs,
        gen_models_path=onnx_dir_path,
        model_base_name=model_name,
        pt_outputs=pt_outputs,
    )

    # Generate custom-IO files for fp16 and int8 kv
    with open(os.path.join(onnx_dir_path, "custom_io_fp16.yaml"), "w") as fp:
        fp.write("# Model Inputs\n\n")
        for input_name in key_value_names:
            fp.write(f" - IOName: {input_name}\n   Precision: float16\n\n")
            inputs[input_name] = inputs[input_name].to(torch.float16)
        fp.write("# Model Outputs\n\n")
        for output_name in key_value_names:
            fp.write(f" - IOName: {output_name}_RetainedState\n   Precision: float16\n\n")

    with open(os.path.join(onnx_dir_path, "custom_io_int8.yaml"), "w") as fp:
        fp.write("# Model Inputs\n\n")
        for input_name in key_value_names:
            fp.write(f" - IOName: {input_name}\n   Precision: mxint8\n\n")
        fp.write("# Model Outputs\n\n")
        for output_name in key_value_names:
            fp.write(f" - IOName: {output_name}_RetainedState\n   Precision: mxint8\n\n")

    # Generate inputFiles
    input_list_file = os.path.join(onnx_dir_path, "input_list.txt")
    generate_input_files(
        input_files_path=os.path.join(onnx_dir_path, "inputFiles"),
        input_names=input_names,
        inputs=inputs,
        input_list_file=input_list_file,
    )

    return model_name


def export_for_cloud(
    model_name: str,
    qeff_model: QEFFBaseModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_length: int = Constants.SEQ_LEN,
) -> str:
    # FIXME: move all this to class instead of here, and just call qeff_model.export here.
    if AUTO_MODEL_MAP_TO_MODEL_TYPE_MAP.get(qeff_model.__class__, None) == QEFF_MODEL_TYPE.CAUSALLM:  # type: ignore
        return export_lm_model_for_cloud(
            model_name=model_name,
            qeff_model=qeff_model,  # type: ignore
            tokenizer=tokenizer,
            onnx_dir_path=onnx_dir_path,
            seq_length=seq_length,
        )
    else:
        raise NotImplementedError(
            f"Only model type {QEFFAutoModelForCausalLM.__class__.__name__} is supported for export, got {type(qeff_model)}"
        )


def export_lm_model_for_cloud(
    model_name: str,
    qeff_model: QEFFAutoModelForCausalLM,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    onnx_dir_path: str,
    seq_length: int,
) -> str:
    if os.path.exists(onnx_dir_path):
        logger.warning(f"Overriding {onnx_dir_path}")
        shutil.rmtree(onnx_dir_path)

    if qeff_model.is_transformed:
        model_name = export_kvstyle_transformed_model_to_onnx(
            model_name=model_name,
            transformed_model=qeff_model.model,
            tokenizer=tokenizer,
            onnx_dir_path=onnx_dir_path,
            seq_len=seq_length,
        )  # type: ignore

    else:
        model_name = export_bertstyle_model_to_onnx(
            model_name=model_name,
            model=qeff_model.model,
            tokenizer=tokenizer,
            onnx_dir_path=onnx_dir_path,
            seq_len=seq_length,
        )  # type: ignore
    return os.path.join(onnx_dir_path, f"{model_name}.onnx")


def qualcomm_efficient_converter(
    model_name: str,
    model_kv: QEFFBaseModel = None,  # type: ignore
    local_model_dir: Optional[str] = None,
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
    cache_dir: Optional[str] = None,
    onnx_dir_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    seq_length: int = Constants.SEQ_LEN,
    kv: bool = True,
    form_factor: str = "cloud",
) -> Tuple[str, str]:
    """
    Function to convert the input string using the specified model and returns the result.

    Args:
        model_name (str): The name of the model to be used.
        model_kv (torch.nn.Module): Transformed KV torch model to be used
        local_model_dir(str): Path to custom model weights and config files
        tokenizer (HF AutoTokenizer): Tokenzier to prepare inputs.
        cache_dir (str): Path to cache dir if not specified, default HF cache_dir will be used.
        onnx_dir_path (str, optional): The path where the model is stored. If None, the model is loaded from the default location.
        hf_token (bool): If True, an authentication token will be used. Default is False.
        seq_len (int, optional): The length of the sequence. Default is 128.
        kv (bool): If True, key-value pairs will be used. Default is True.
        form_factor (str): form_factor of the hardware, currently only accepts "cloud".

    Returns:
        None, if automation is False, else path to exported Onnx file

    """
    warnings.warn(
        "\033[93mmodel_kv argument will be replaced by qeff_model of type QEFFBaseModel\033[0m",
        DeprecationWarning,
        stacklevel=2,
    )
    # Get model_kv first
    model_kv = (
        model_kv
        if model_kv
        else QEFFCommonLoader.from_pretrained(
            pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
            token=hf_token,
            cache_dir=cache_dir,
        )
    )

    # Transform if required
    if model_kv.is_transformed and not kv:
        raise AttributeError("Transformed model is passed while requsting to convert non-transformed model")

    model_kv = model_kv if model_kv.is_transformed else QEfficient.transform(model_kv) if kv else model_kv

    if onnx_dir_path is None:
        model_card_dir = os.path.join(QEFF_MODELS_DIR, str(model_name))
        onnx_dir_path = os.path.join(model_card_dir, "onnx")
        os.makedirs(onnx_dir_path, exist_ok=True)

    # Load tokenizer if not passed
    tokenizer = (
        tokenizer
        if tokenizer
        else load_hf_tokenizer(
            pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
            hf_token=hf_token,
            cache_dir=cache_dir,
        )
    )

    if form_factor == "cloud":
        generated_onnx_model_path = export_for_cloud(
            model_name=model_name,
            qeff_model=model_kv,
            tokenizer=tokenizer,
            onnx_dir_path=onnx_dir_path,
            seq_length=seq_length,
        )
        return onnx_dir_path, generated_onnx_model_path
    else:
        # [TODO]: Apply the class transformation to make changes for the KV models in edge use cases
        # model = QEfficient.transform(model_hf, type="Transformers", form_factor="edge")
        # model.eval()
        raise NotImplementedError("Oops! Reached too far!!")
