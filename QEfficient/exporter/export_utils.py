# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
import shutil
import sys
from logging import info
from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import external_data_helper

from QEfficient.base.onnx_transforms import FP16ClipTransform, OnnxTransformPipeline
from QEfficient.utils import constants


def export_onnx(
    pt_model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    gen_models_path: str,
    model_base_name: str,
) -> str:
    """
    API for export PyTorch model to ONNX.

    Args:
        :pt_model (torch.nn.Module): PyTorch model that will be exported to ``ONNX`` format.
        :inputs (Dict[str, torch.Tensor]): Processed torch input for the model.
        :output_names (List[str]): Output of pytorch model inference.
        :gen_models_path (str): Path of generated ``ONNX`` model.
    :model_base_name (str): Base name for the exported ``ONNX`` file.

    Return:
        :str: Updated base name of exported ``ONNX`` model.
    """
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
    dynamic_axis_past_key = "full_batch_size" if "batch_index" in input_names else "batch_size"
    kv_cache_3d = len(inputs["past_key_values"][0][0].shape) == 3
    dynamic_axes = {}
    for iname in input_names:
        if iname in seq_len_inputs:
            dynamic_axes[iname] = {0: "batch_size", 1: "seq_len"}
        elif iname in decoder_seq_inputs:
            dynamic_axes[iname] = {0: "batch_size", 1: "decoder_seq_len"}
        elif kv_cache_3d and iname.startswith("past_"):
            # KV-cache (batch_size, ctx_len, d_head)
            dynamic_axes[iname] = {0: dynamic_axis_past_key, 1: "ctx_len"}
        elif iname.startswith("past_"):
            # KV-cache (batch_size, num_heads, past_len, embed_dim)
            dynamic_axes[iname] = {0: dynamic_axis_past_key, 2: "ctx_len"}
        elif iname == "batch_index":
            dynamic_axes[iname] = {0: "batch_size"}

    if "past_key.0" in input_names and "attention_mask" in input_names:
        dynamic_axes["attention_mask"] = {0: "batch_size", 1: "ctx_len"}

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
            opset_version=constants.ONNX_EXPORT_OPSET,
            custom_opsets={"com.qti.aisw.onnx": 1},
        )
    except Exception as e:
        raise RuntimeError("Exporting to ONNX failed. {}".format(e))

    onnx.checker.check_model(f"{gen_models_path}_tmp/{model_base_name}.onnx", full_check=True)
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
    onnx.checker.check_model(os.path.join(gen_models_path, f"{model_base_name}.onnx"), full_check=True)

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
    """
    API to save ONNX model and it's data separately if size of ``ONNX`` model is greater than 2GB.

    Args:
        :model (Union[onnx.ModelProto, str]): Pass ``ONNX`` model or path of the model.
        :gen_models_path (str): Path of generated ``ONNX`` model.
        :model_base_name (str): Base name of the HuggingFace model.

    Return:
        :str: Base name of ``ONNX`` exported model.
    """
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


def remove_temp_file(model_file_path: str, weights_file_path: str):
    """
    API to remove a temporary file

    Args:
        :model_file_path (str); Path to the file to be deleted
        :weights_file_path (str): Path to the weights file
    """
    try:
        os.remove(model_file_path)
        os.remove(weights_file_path)
    except FileNotFoundError:
        print(f"File '{model_file_path}' does not exist.")
    except Exception as e:
        print(f"Error deleting file '{model_file_path}': {e}")


def fix_onnx_fp16(
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    ort_outputs: List[np.ndarray],
    gen_models_path: str,
    model_base_name: str,
    pt_outputs: Dict[str, torch.Tensor],
) -> str:
    """
    API to clip model weights in fp16 range and save updated clipped ``ONNX`` model.

    Args:
        :inputs (Dict[str, torch.Tensor]): Processed torch input for the model.
        :output_names (List[str]): Output names of pytorch model inference.
        :ort_outputs (List[np.ndarray]): Output of onnxruntime.
        :gen_models_path (str): Path of generated ``ONNX`` model.
        :model_base_name (str): Base name for the exported ONNX model.
        :pt_outputs (Dict[str, torch.Tensor]): Output of PyTorch model inference.

    Return:
        :str: Updated base name of exported ONNX model.
    """
    model = onnx.load(os.path.join(gen_models_path, f"{model_base_name}.onnx"))
    onnx_transforms = OnnxTransformPipeline(transforms=[FP16ClipTransform])
    model, fp16_fix = onnx_transforms.apply(model, model_name="", onnx_base_dir=gen_models_path)

    if fp16_fix:
        # Save FP16 model
        info("Found constants out of FP16 range, clipped to FP16 range")

        # remove the fp32 version of the model files to save space.
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
                close_outputs.append(fix_diff < 1e-5)
    else:
        info("No constants out of FP16 range")

    return model_base_name


def generate_input_files(
    input_files_path: str,
    input_names: List[str],
    inputs: Dict[str, torch.Tensor],
    input_list_file: str,
):
    """
    API to generate input files, required for Cloud AI 100 execution.

    Args:
        :input_files_path (str): Path to save input files.
        :input_names (List[str]): Names of inputs to be saved.
        :inputs (dict[str, torch.tensor]): Input tensors to be saved in raw format.
        :input_list_file (str): File name to save the names of inputs in order. Example - "input_list.txt"
    """
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


# FIXME(ochougul/quic-mamta): Remove duplication with APIRunner
def run_model_on_ort(
    onnx_path: str,
    inputs: Dict[str, torch.Tensor],
    output_names: List[str],
    pt_outputs: Dict[str, torch.Tensor],
    dtype: bool = True,
) -> Tuple[List[str], List[np.ndarray]]:
    """
    API to run ONNX model on ONNX runtime

    Args:
        :onnx_path (str): Path of ONNX model.
        :inputs (Dict[str, torch.Tensor]): Processed torch input for the model.
        :output_names (List[str]): Output from pytorch inference.
        :pt_outputs (Dict[str, torch.Tensor]): Output of PyTorch model inference.
        :dtype (bool): If False it will consider you are passing clipped version of ``ONNX`` model.

    Return:
        :Tuple[List[str], List[np.ndarray]]: input_names
    """
    try:
        if dtype:
            info_string = "fp32"
        else:
            info_string = "fp32 clipped"
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
        print("\n=====================================================================\n")

        return input_names, ort_outputs
    except Exception as e:
        model = onnx.load(onnx_path, load_external_data=False)
        input_names = [x.name for x in model.graph.input]
        print(f"Failed to run the onnx {onnx_path} model in onnx runtime:%s", e)
        print("\n=====================================================================\n")
        return input_names, None


def compare_onnx_func(func1 : onnx.FunctionProto, func2 : onnx.FunctionProto):
    if len(func1.input) != len(func2.input) or len(func1.output) != len(func2.output) or len(func1.node) != len(func2.node):
        return False
    for i in range(len(func1.node)):
        node1 = func1.node[i]
        node2 = func2.node[i]
        # check inputs
        if len(node1.input) != len(node2.input):
            print(f"node {i}, input_length")
            return False
        for j in range(len(node1.input)):
            if node1.input[j] in func1.input: # input to the subfunction, not from another node
                idx = list(func1.input).index(node1.input[j])
                if node2.input[j] not in func2.input or list(func2.input).index(node2.input[j]) != idx:
                    print(f"node {i}, {j} input mismatch (function input)")
                    return False
            elif node1.input[j] != node2.input[j]:
                # some names might change to match output nodes
                if node1.input[j] in func1.output:
                    idx = list(func1.output).index(node1.input[j])
                    if node2.input[j] not in func2.output or  list(func2.output).index(node2.input[j]) != idx:
                        print(f"node {i}, propogated output mismatch (function output)")
                        return False
                else:
                    print(f"node {i}, input mismatch (node input)")
                    return False
        # check optype
        if node1.op_type != node2.op_type:
            print(f"node {i}, op type mismatch")
            return False
        # check attributes
        if len(node1.attribute) != len(node2.attribute):
            print(f"node {i}, attribute_length")
            return False
        for j in range(len(node1.attribute)):
            if node1.attribute[j] != node2.attribute[j]:
                print(f"node {i}, attribute mismatch")
                return False
        # check outputs
        if len(node1.output) != len(node2.output):
            print(f"node {i}, output length")
            return False
        for j in range(len(node1.output)):
            if node1.output[j] in func1.output:
                idx = list(func1.output).index(node1.output[j])
                if node2.output[j] not in func2.output or  list(func2.output).index(node2.output[j]) != idx:
                    print(f"node {i}, output mismatch (function output)")
                    return False
            else:
                if node1.output[j] != node2.output[j]:
                    print(f"node {i}, output mismatch (node output)")
                    return False
    return True


def merge_models(m1, m2, io_map):

    graph = onnx.compose.merge_graphs(
        m1.graph,
        m2.graph,
        io_map,
    )
    # import ipdb;ipdb.set_trace()
    model = onnx.helper.make_model_gen_version(
        graph,
        producer_name="QEfficient",
        producer_version="1.21",
        ir_version=10,
        opset_imports=m1.opset_import
    )

    # Merging model metadata props
    model_props = {}
    for meta_entry in m1.metadata_props:
        model_props[meta_entry.key] = meta_entry.value
    for meta_entry in m2.metadata_props:
        if meta_entry.key in model_props:
            value = model_props[meta_entry.key]
            if value != meta_entry.value:
                raise ValueError(
                    "Can't merge models with different values for the same model metadata property."
                    f" Found: property = {meta_entry.key}, with values {value} and {meta_entry.value}."
                )
        else:
            model_props[meta_entry.key] = meta_entry.value
    onnx.helper.set_model_props(model, model_props)

    # Merging functions
    # function_overlap = list(
    #     {f.name for f in m1.functions} & {f.name for f in m2.functions}
    # )
    # import ipdb; ipdb.set_trace()
    # if function_overlap:
    #     raise ValueError(
    #         "Can't merge models with overlapping local function names."
    #         " Found in both graphs: " + ", ".join(function_overlap)
    #     )
    for func1, func2, in zip(m1.functions, m2.functions):
        if compare_onnx_func(func1, func2):
            model.functions.MergeFrom([func1])
        else:
            model.functions.MergeFrom([func1])
            model.functions.MergeFrom([func2])
        
    # model.functions.MergeFrom(m2.functions)
    return model

def rewrite_tensors_with_prefix(model, prefix, func_attr_tens, size_threshold=1024, file_chunk_size=10* 2**30):
    size = 0
    file_num=0
    for tensor in external_data_helper._get_all_tensors(model):
        if tensor.HasField("raw_data") and tensor.name!="int64_2" and tensor.name not in func_attr_tens:
            tsize = len(tensor.raw_data)
            if tsize > size_threshold:
                if size + tsize > file_chunk_size:
                    file_num += 1
                    size = tsize
                else:
                    size += tsize
            print(f"setting location for {tensor.name}")
            location = prefix+"_"+str(file_num)+".onnx.data"
            external_data_helper.set_external_data(tensor, location)