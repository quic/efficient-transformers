import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from QEfficient import QEFFAutoModel

from inspect import signature


model_path = "Dream-org/Dream-v0-Instruct-7B"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
compile_length = 1000
# config.max_position_embeddings = compile_length

config.num_hidden_layers = 2

model = QEFFAutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True, config = config)
# model = QEFFAutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, config = config)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# breakpoint()

B,L = 1,128
model.export(export_dir = '/home/jsaisaga/qeff_llama/DreamModelRope2Layers/')
# breakpoint()
# model.compile(
#         # prefill_seq_len=128,
#         ct_len=4000,
#         # seq_len=4000,
#         num_cores=16,
#         # mxfp6_matmul=True,
#         num_devices=16,
#         mxint8_kv_cache=True,
#         aic_enable_depth_first=True,
#         # skip_vision=True,
#         mos=3,
#     )
# input_ids      = torch.randint(0, model.config.vocab_size, (B, L), dtype=torch.long)
# attention_mask = torch.ones(B, L, dtype=torch.float)
'''
position_ids = torch.arange(L, dtype=torch.float).unsqueeze(0).expand(B, L)

example_kwarg_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "use_cache": False,           
}

with torch.inference_mode():
    _ = model(**{k: v for k, v in example_kwarg_inputs.items()
                 if not isinstance(v, bool)} , use_cache=False)

# model = model.to("cuda").eval()
# print(model.config)

print(model)                          
print(signature(model.forward))
torch.onnx.export(
    model,
    args=(example_kwarg_inputs['input_ids'], example_kwarg_inputs['position_ids'], example_kwarg_inputs['attention_mask'], example_kwarg_inputs['use_cache']), 
    f="dream_no_cache.onnx",
    input_names=list(example_kwarg_inputs.keys()),
    output_names=["logits"],           # rename if your model returns more fields
    # example_kwarg_inputs=example_kwarg_inputs,
    opset_version=17,
    do_constant_folding=True,
    # dynamic_axes={
    #     "input_ids":      {0: "batch", 1: "seq"},
    #     "attention_mask": {0: "batch", 1: "seq"},
    #     "position_ids":   {0: "batch", 1: "seq"},
    #     "logits":         {0: "batch", 1: "seq_out"},  # typically same as seq
    # },
)
print("Exported dream_no_cache.onnx")
'''
'''
torch.onnx.export(
                model,
                (example_inputs,),
                str(tmp_onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=constants.ONNX_EXPORT_OPSET,
                # verbose=True,
                **export_kwargs,
            )
'''