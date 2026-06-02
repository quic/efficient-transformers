import os

import onnx
from onnx import external_data_helper

from QEfficient.base.onnx_transforms import CustomOpTransform


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
    import ipdb;ipdb.set_trace()
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
    function_overlap = list(
        {f.name for f in m1.functions} & {f.name for f in m2.functions}
    )
    import ipdb; ipdb.set_trace()
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

# ----------------------------- #
m1 = onnx.load("/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/layer_0/split_graph.onnx", load_external_data=False)
m2 = onnx.load("/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/layer_1/Glm4MoeForCausalLM_layer_1.onnx", load_external_data=False)

# add prefix
m1_pref = onnx.compose.add_prefix(m1, "layer_0/", rename_functions=False)
m1_base_path = "/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/layer_0/"
external_data_helper.load_external_data_for_model(m1_pref, m1_base_path)
func_attr_tens = {val.name for val in external_data_helper._get_attribute_tensors_from_graph(m1_pref.functions[0])}
rewrite_tensors_with_prefix(m1_pref, m1_base_path, rewrite_dir="/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp", prefix="layer_0", func_attr_tens=func_attr_tens)
onnx.save(m1_pref, "/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/Glm4MoeForCausalLM_layer_0_pref.onnx")
import ipdb; ipdb.set_trace()

m2_pref = onnx.compose.add_prefix(m2, "layer_1/", rename_functions=False)
m2_base_path = "/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/layer_1/"
rewrite_tensors_with_prefix(m2_pref, m2_base_path, rewrite_dir="/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp", prefix="layer_1")
onnx.save(m2_pref, "/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/Glm4MoeForCausalLM_layer_1_pref.onnx")
# --------------------------------#

# out = onnx.compose.merge_models(m1_pref.graph, m2_pref.graph, io_map = [("layer_0//model/layers.0/QEffGlm4MoeDecoderLayer_output_2", "layer_1//model/embed_tokens/Gather_output_0"), 
#                                                                         ("layer_0//model/Unsqueeze_1_output_0", "layer_1//model/Unsqueeze_1_output_0"),
#                                                                         ("layer_0/position_ids", "layer1/position_ids")])

# out = onnx.compose.merge_graphs(m1.graph, m2.graph, io_map = [("layer_0//model/layers.0/QEffGlm4MoeDecoderLayer_output_2", "layer_1//model/embed_tokens/Gather_output_0"), 
#                                                                         ("layer_0//model/Unsqueeze_1_output_0", "layer_1//model/Unsqueeze_1_output_0"),
#                                                                         ("layer_0/position_ids", "layer1/position_ids")])

m1_pref = onnx.load("/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/Glm4MoeForCausalLM_layer_0_pref.onnx", load_external_data=False)
m2_pref = onnx.load("/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/Glm4MoeForCausalLM_layer_1_pref.onnx", load_external_data=False)
merged_model = merge_models(m1_pref, m2_pref, io_map=[("layer_0//model/layers.0/QEffGlm4MoeDecoderLayer_output_2", "layer_1//model/embed_tokens/Gather_output_0"),
                                   ("layer_0//model/Unsqueeze_1_output_0", "layer_1//model/Unsqueeze_1_output_0"),
                                   ("layer_0/position_ids", "layer_1/position_ids")])

CustomOpTransform.apply(merged_model)
onnx.save(merged_model, "/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/merged_new.onnx")




def rename_just_subfunc(model, new_name="QEffGlm4MoeDecoderLayerNonMoe"):
    f_name_map = {}
    for f in model.functions:
        if f.name == "QEffGlm4MoeDecoderLayer":
            new_f_name = new_name
        else:
            new_f_name = f.name
        f_name_map[f.name] = new_f_name
        f.name = new_f_name
    # Adjust references to local functions in other local function
    # definitions
    for f in model.functions:
        for n in f.node:
            if n.op_type in f_name_map:
                n.op_type = f_name_map[n.op_type]
    # Adjust references to local functions in the graph
    for n in model.graph.node:
        if n.op_type in f_name_map:
            n.op_type = f_name_map[n.op_type]
    
    return model