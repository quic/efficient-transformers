# Example script to test whisper on qefficient

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForSpeechSeq2Seq
from QEfficient.utils import constants, get_padding_shape_from_config
import torch

model_name = "openai/whisper-tiny"
qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(model_name)
print(f"{model_name} optimized for AI 100 \n", qeff_model)

onnx_path = qeff_model.export("../qeff_test_full")

print(onnx_path)

import onnxruntime
sess = onnxruntime.InferenceSession(onnx_path)
print([x.name for x in sess.get_inputs()])
print([x.name for x in sess.get_outputs()])

generated_qpc_path = qeff_model.compile(
    num_cores=14
)

print(generated_qpc_path)

# qeff_model.generate(prompts=["Once upon a time"], tokenizer=AutoTokenizer.from_pretrained(model_name))