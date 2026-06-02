import onnx
import onnxscript
import onnx_ir

from QEfficient.utils.onnx_extract_model import ExtractGraphPass

onnx_path = "/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/layer_0/Glm4MoeForCausalLM_layer_tmp_0.onnx"
input_names = ['input_ids', 'position_ids', 'past_key.0', 'past_value.0']
# output_names = ['/model/layers.0/QEffGlm4MoeDecoderLayer.2_output_0', '/model/layers.0/QEffGlm4MoeDecoderLayer.2_output_1', '4514', '4515', '/model/layers.0/QEffGlm4MoeDecoderLayer.2_output_4']
output_names = ['/model/layers.0/QEffGlm4MoeDecoderLayer_output_2', '2020', '2021', '/model/Unsqueeze_1_output_0', 'position_ids']

model = onnx_ir.load(onnx_path)
model.graph = onnx_ir.convenience.extract(model.graph, input_names, output_names)
onnx_ir.save(model, "/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/layer_0/split_graph.onnx")
# onnx.compose.merge_graphs()

# assert result.modified
# import ipdb; ipdb.set_trace()
# onnx.save(result, "/home/ochougul/.cache/qeff_models/Glm4MoeForCausalLM/Glm4MoeForCausalLM-b1311053a7588364/onnx_layerwise_tmp/layer_0/split_graph.onnx")