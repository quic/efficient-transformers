# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import shutil
import subprocess
import sys
from logging import error, info
from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import external_data_helper, numpy_helper


def export_onnx(
    pt_model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    gen_models_path: str,
    model_base_name: str,
) -> str:
    # Inspect the model's forward method arguments
    pt_model_code = pt_model.forward.__code__
    pt_input_names = pt_model_code.co_varnames[1 : pt_model_code.co_argcount]

    # Arrange the inputs in proper order to make tracing work properly
    pt_inputs = []
    input_names = []
    for input_name in pt_input_names:
        if input_name in inputs:
            if input_name == "past_key_values":
                for i in range(len(inputs[input_name])):
                    input_names.append(f"past_key.{i}")
                    input_names.append(f"past_value.{i}")
            else:
                input_names.append(input_name)
            pt_inputs.append(inputs[input_name])
        else:
            pt_inputs.append(None)

    # Create dynamic axes dict for inputs that need to have dynamic input shapes
    seq_len_inputs = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "encoder_outputs",
    }
    decoder_seq_inputs = {"decoder_input_ids", "decoder_attention_mask"}

    dynamic_axes = {}
    for iname in input_names:
        if iname in seq_len_inputs:
            dynamic_axes[iname] = {0: "batch_size", 1: "seq_len"}
        elif iname in decoder_seq_inputs:
            dynamic_axes[iname] = {0: "batch_size", 1: "decoder_seq_len"}
        elif iname.startswith("past_"):
            # KV-cache (batch_size, num_heads, past_len, embed_dim)
            dynamic_axes[iname] = {0: "batch_size", 2: "ctx_len"}
    if "past_key.0" in input_names and "attention_mask" in input_names:
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "ctx_len"}

    # return input_names, output_names, model_base_name
    os.makedirs(f"{gen_models_path}_tmp", exist_ok=True)
    try:
        info("Exporting to ONNX...")
        torch.onnx.export(
            pt_model,
            tuple(pt_inputs),
            f"{gen_models_path}_tmp/{model_base_name}.onnx",
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13,
            custom_opsets={"com.qti.aisw.onnx": 1},
        )
    except Exception as e:
        error("Exporting to ONNX failed. {}".format(e))
        return

    onnx.checker.check_model(f"{gen_models_path}_tmp/{model_base_name}.onnx")
    loaded_model = onnx.load(f"{gen_models_path}_tmp/{model_base_name}.onnx")
    shutil.rmtree(f"{gen_models_path}_tmp")
    os.makedirs(f"{gen_models_path}", exist_ok=True)
    info("Clearing files .. ")

    # Check if model uses external data format to save the weight tensors
    # model_uses_external_data = check_model_uses_external_data(loaded_model)
    # if model_uses_external_data:
    # Save model to single weight file
    info("ONNX model uses external data. Saving as external data.")
    onnx.save_model(
        loaded_model,
        os.path.join(gen_models_path, f"{model_base_name}.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{model_base_name}.onnxweights.data",
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(os.path.join(gen_models_path, f"{model_base_name}.onnx"))

    # Run shape inference in intial model itself
    onnx.shape_inference.infer_shapes_path(
        os.path.join(gen_models_path, f"{model_base_name}.onnx"),
        os.path.join(gen_models_path, f"{model_base_name}.onnx"),
        True,
        True,
        True,
    )

    info(f"input names {input_names}")
    info(f"output names {output_names}")
    info(f"Initial Model Export Completed...{model_base_name}")

    return model_base_name


def save_onnx(model: Union[onnx.ModelProto, str], gen_models_path: str, model_base_name: str) -> str:
    if isinstance(model, str):
        model = onnx.load(f"{gen_models_path}/{model}.onnx")

    # Load the external tensors into the ModelProto, so the right size is calculated
    # and re-exported into right external tensor file
    onnx.load_external_data_for_model(model, gen_models_path)
    GB = 2**30

    if model.ByteSize() <= 2 * GB:
        onnx.save(model, f=f"{gen_models_path}/{model_base_name}.onnx")
    else:
        file_num = 0
        current_file_size = 0
        for tensor in external_data_helper._get_all_tensors(model):
            if tensor.HasField("raw_data") and ((tsize := sys.getsizeof(tensor.raw_data)) >= 1024):
                current_file_size += tsize
                if current_file_size > 10 * GB:
                    file_num += 1
                    current_file_size = tsize
                external_data_helper.set_external_data(tensor, f"{model_base_name}_{file_num}.onnx.data")
        onnx.save(model, f=f"{gen_models_path}/{model_base_name}.onnx")

    return model_base_name


def remove_temp_file(file_path_model, file_path_weights):
    """
    Function to remove a temporary file

    :param str file_path: Path to the file to be deleted
    :file_path_weights: Path to the weights file
    """
    try:
        os.remove(file_path_model)
        os.remove(file_path_weights)
    except FileNotFoundError:
        print(f"File '{file_path_model}' does not exist.")
    except Exception as e:
        print(f"Error deleting file '{file_path_model}': {e}")


def fix_onnx_fp16(
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    ort_outputs: List[np.ndarray],
    gen_models_path: str,
    model_base_name: str,
    pt_outputs: Dict[str, torch.Tensor],
    save_fp32_onnx: bool = False,
) -> str:
    finfo = np.finfo(np.float16)
    fp16_max = finfo.max
    fp16_min = finfo.min
    abs_max_val = np.finfo(np.float16).max

    model = onnx.load(os.path.join(gen_models_path, f"{model_base_name}.onnx"))

    fp16_fix = False
    for tensor in external_data_helper._get_all_tensors(model):
        nptensor = numpy_helper.to_array(tensor, gen_models_path)
        if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
            nptensor = np.clip(nptensor, fp16_min, fp16_max)
            new_tensor = numpy_helper.from_array(nptensor, tensor.name)
            tensor.CopyFrom(new_tensor)
            fp16_fix = True

    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Constant":
            curr_data = numpy_helper.to_array(node.attribute[0].t)
            if np.any(np.abs(curr_data) > abs_max_val):
                updated_data = np.clip(curr_data, -abs_max_val, abs_max_val).astype(curr_data.dtype)
                updated_tensor = numpy_helper.from_array(updated_data)
                node.attribute[0].t.CopyFrom(updated_tensor)
                fp16_fix = True

    for node in model.graph.initializer:
        curr_data = numpy_helper.to_array(node)
        if np.any(np.abs(curr_data) > abs_max_val):
            updated_data = np.clip(curr_data, -abs_max_val, abs_max_val).astype(curr_data.dtype)
            updated_tensor = numpy_helper.from_array(updated_data)
            updated_tensor.name = node.name
            node.CopyFrom(updated_tensor)
            fp16_fix = True

    # TODO: Check this, variable "fp16_same_as_fp32" is not being used.
    # fp16_same_as_fp32 = True
    ort_outputs_fixed = []

    if fp16_fix:
        # Save FP16 model
        info("Found constants out of FP16 range, clipped to FP16 range")

        # remove the fp32 version of the model files to save space.
        if not save_fp32_onnx:
            remove_temp_file(
                os.path.join(gen_models_path, f"{model_base_name}.onnx"),
                os.path.join(gen_models_path, f"{model_base_name}.onnxweights.data"),
            )
        model_base_name += "_clipped_fp16"
        onnx.save_model(
            model,
            os.path.join(gen_models_path, f"{model_base_name}.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{model_base_name}.onnxweights.data",
            size_threshold=1024,
            convert_attribute=False,
        )

        # Check if the FP16-fixed model can be used for FP32
        close_outputs = []
        _, ort_outputs_fixed = run_model_on_ort(
            os.path.join(gen_models_path, f"{model_base_name}.onnx"),
            inputs,
            output_names,
            pt_outputs,
            False,
        )
        info("ONNXRT vs. ONNXRT fixed (MAD):")
        if ort_outputs is not None:
            for oname, orto, ortof in zip(output_names, ort_outputs, ort_outputs_fixed):
                fix_diff = np.abs(orto.astype(np.float32) - ortof.astype(np.float32)).max()
                # TODO: need to the debug this
                # info(oname, fix_diff)
                close_outputs.append(fix_diff < 1e-5)
        else:
            print(
                f"Failed to run the onnx {model_base_name} model in onnx runtime:%s",
                ort_outputs,
            )
        # Commenting the below line
        # fp16_same_as_fp32 = all(close_outputs)
    else:
        info("No constants out of FP16 range .. ")

    return model_base_name


def generate_input_files(
    input_files_path: str,
    input_names: List[str],
    inputs: Dict[str, torch.Tensor],
    input_list_file: str,
):
    # inputFiles
    os.makedirs(input_files_path, exist_ok=True)
    filenames = []
    for name in input_names:
        # We can't directly iterate with inputs.items() because
        # we have to maintain the order of input_names
        suffix = inputs[name].shape[-1] if len(inputs[name].shape) > 0 else 0
        filename = f"{input_files_path}/{name}_{suffix}.raw"
        inputs[name].detach().numpy().tofile(filename)
        filenames.append(filename.split("/", 1)[-1])

    # input_list.txt
    with open(input_list_file, "w") as fp:
        fp.write(",".join(filenames))
        fp.write("\n")


def run_model_on_ort(
    onnx_path: str,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    pt_outputs: Dict[str, torch.Tensor],
    dtype: bool = True,
) -> Tuple[List[str], List[np.ndarray]]:
    try:
        if dtype:
            info_string = "fp32"
        else:
            info_string = "fp16"
        ort_session = onnxruntime.InferenceSession(onnx_path)
        input_names = [x.name for x in ort_session.get_inputs()]
        ort_outputs = ort_session.run(
            output_names,
            {k: v.detach().numpy() for k, v in inputs.items() if k in input_names},
        )
        print(f"\n=============== PyTorch vs. {info_string} ONNXRT (MAD) ===============\n")
        past_key_sum = 0
        past_value_sum = 0
        num = 0
        for oname, orto in zip(output_names, ort_outputs):
            pto = pt_outputs[oname].detach().numpy()
            val = np.abs(pto.astype(np.float32) - orto.astype(np.float32)).max()
            if oname.startswith("past_key"):
                past_key_sum += val
                num = num + 1
            elif oname.startswith("past_value"):
                past_value_sum += val
            else:
                print(f"{oname} \t\t {val}")
        past_key_mean = past_value_sum / num
        past_value_mean = past_key_sum / num
        print(f"past_keys (mean) \t\t {past_key_mean}")
        print(f"past_value (mean) \t\t {past_value_mean}")
        print("\n=============================================================\n")

        return input_names, ort_outputs
    except Exception as e:
        model = onnx.load(onnx_path, load_external_data=False)
        input_names = [x.name for x in model.graph.input]
        print(f"Failed to run the onnx {onnx_path} model in onnx runtime:%s", e)
        print("\n=============================================================\n")
        return input_names, None


def run_model_on_cloud_ai_100(
    onnx_path: str,
    onnx_symbol_defs: Dict[str, int] = {},
    **kwargs,
) -> bool:
    args = [
        "/opt/qti-aic/exec/qaic-exec",
        f"-m={onnx_path}",
        "-aic-hw",
        "-aic-hw-version=2.0",
    ]
    for onnx_symbol, onnx_def in onnx_symbol_defs.items():
        args.append(f"-onnx-define-symbol={onnx_symbol},{onnx_def}")
    for k, v in kwargs.items():
        k = k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(f"-{k}")
            continue
        args.append(f"-{k}={v}")

    info("Running compiler:", " ".join(args))
    result = subprocess.run(args)
    return result.returncode == 0


def compile_kv_model_on_cloud_ai_100(
    onnx_path: str,
    specializations_json: str,
    num_cores: int,
    base_path: str,
    mxfp6: bool,
    custom_io_path: str,
    aic_enable_depth_first: bool,
    mos: int = -1,
    device_group: List[int] = [0],
    **kwargs,
) -> bool:
    import shutil

    aic_binary_dir = os.path.join(base_path, "qpcs")

    if os.path.isdir(aic_binary_dir):
        shutil.rmtree(aic_binary_dir)

    assert os.path.isfile(
        specializations_json
    ), f"Please use 'from QEfficient.cloud.compile import main as compile', as {specializations_json} file was not found"
    assert os.path.isfile(custom_io_path), f"{custom_io_path} file was not found!"
    command = [
        "/opt/qti-aic/exec/qaic-exec",
        f"-m={onnx_path}",
        "-aic-hw",
        "-aic-hw-version=2.0",
        f"-network-specialization-config={specializations_json}",
        "-convert-to-fp16",
        "-retained-state",
        f"-aic-num-cores={num_cores}",
        f"-custom-IO-list-file={custom_io_path}",
        "-compile-only",
        f"-aic-binary-dir={aic_binary_dir}",
    ]
    if mxfp6:
        command.append("-mxfp6-matmul")
    if mos > 0:
        command.append(f"-mos={mos}")
    if aic_enable_depth_first:
        command.append("-aic-enable-depth-first")
    if len(device_group) > 1:
        mdp_ts_config = {
            "connections": [{"devices": device_group, "type": "p2p"}],
            "partitions": [
                {
                    "name": "Partition0",
                    "devices": [{"deviceId": device, "numCores": num_cores} for device in device_group],
                }
            ],
        }
        mdp_ts_config_path = os.path.join(base_path, "mdp_ts_config.json")
        with open(mdp_ts_config_path, "w") as file:
            json.dump(mdp_ts_config, file, indent=4)
        command.append(f"-mdp-load-partition-config={mdp_ts_config_path}")
    print("Running AI 100 compiler:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compilation Failed!!\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}")

    print("\n===================== Compilation Done! =====================\n")
    return result.returncode == 0, aic_binary_dir
