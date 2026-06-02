import os

import onnx

from QEfficient.exporter.export_utils import merge_models, rewrite_tensors_with_prefix, compare_onnx_func

def merge_layerwise_onnx(base_path: str, num_layers):
    for i in range(num_layers):
        onnx_file_path = os.path.join(base_path, f"layer_{i}", f"Glm4MoeForCausalLM_layer_tmp_{i}.onnx")
        assert os.path.isfile(onnx_file_path), f"File not found: {onnx_file_path}"
        curr_model = onnx.load(onnx_file_path, load_external_data=False)
        pref_model = onnx.compose.add_prefix(curr_model, prefix=f"layer_{i}_", rename_functions=False)
            
        
        
        