import torch
from transformers import AutoConfig, AutoModelForCausalLM, GptOssForCausalLM, TextStreamer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner

Constants.INPUT_STR=["Make sure tokens don't repeat\n\nTo make a simple cup of coffee, start by boiling water. Add one to two teaspoons of instant coffee powder to a mug. Pour the hot water over the coffee and stir well. Add sugar and milk to taste, if desired. For brewed coffee, use a French press or drip filter. Add coarsely ground coffee to the device, pour hot water over it, and let it steep for four minutes. Press or filter the coffee, then serve"]

torch.manual_seed(42)
model_id = "openai/gpt-oss-20b"
config = AutoConfig.from_pretrained(model_id)
config.num_hidden_layers=2

# Remove the quantization_config attribute if it exists, to avoid MXFP4 Issues
if hasattr(config, "quantization_config"):
    delattr(config, "quantization_config")

model = GptOssForCausalLM.from_pretrained(
    "/home/vbaddi/transformers/src/transformers/models/gpt_oss/new_weights", torch_dtype=torch.float32, attn_implementation="eager", config=config
)
model.eval()
model.generation_config.sample=False
tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_id)
config = model.config
batch_size = len(Constants.INPUT_STR)

api_runner = ApiRunner(batch_size, tokenizer, config, Constants.INPUT_STR, 97, 256)
pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model)


qeff_model = QEFFAutoModelForCausalLM(model, continuous_batching=False)
# pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)

onnx_model_path = qeff_model.export()


qpc_path = qeff_model.compile(
    prefill_seq_len=128,
    ctx_len=256,
    num_cores=16,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
)
print(f"qpc path is {qpc_path}")
streamer = TextStreamer(tokenizer)
exec_info = qeff_model.generate(
    tokenizer,
    streamer=streamer,
    prompts=Constants.INPUT_STR[0],
    device_ids=[0],
)

print(pytorch_hf_tokens)
print(exec_info)
assert (exec_info.generated_ids[0][0,:159] == pytorch_hf_tokens).all()
