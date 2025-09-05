# # Initiate the Original Transformer model
# from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

# # Please uncomment and use appropriate Cache Directory for transformers, in case you don't want to use default ~/.cache dir.
# # os.environ["TRANSFORMERS_CACHE"] = "/local/mnt/workspace/hf_cache"

# # ROOT_DIR = os.path.dirname(os.path.abspath(""))
# # CACHE_DIR = os.path.join(ROOT_DIR, "tmp") #, you can use a different location for just one model by passing this param as cache_dir in below API.

# # Model-Card name to be onboarded (This is HF Model Card name) : https://huggingface.co/gpt2-xl
# model_name = "gpt2"  # Similar, we can change model name and generate corresponding models, if we have added the support in the lib.

# qeff_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="gpt2")
# print(f"{model_name} optimized for Cloud AI 100 \n", qeff_model)

# # We can now export the modified models to ONNX framework
# # This will generate single Onnx Model for both Prefill and Decode Variations which are optimized for
# # Cloud AI 100 Platform.

# # While generating the ONNX model, this will clip the overflow constants to fp16
# # Verify the model on Onnxruntime vs Pytorch

# # Then generate inputs and customio yaml file required for compilation.
# qeff_model.export()

# # Compile the model for provided compilation arguments
# # Please use platform SDK to Check num_cores for your card.

# qeff_model.compile(
#     num_cores=14,
#     mxfp6=True,
#     device_group=[0],
# )

# # post compilation, we can print the latency stats for the kv models, We provide API to print token and Latency stats on Cloud AI 100
# # We need the compiled prefill and decode qpc to compute the token generated, This is based on Greedy Sampling Approach

# qeff_model.generate(prompts=["My name is"])

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

print("done")
model_name = "gpt2"
# model_name = "google/gemma-3-1b-it"
# model_name = "meta-llama/Llama-3.1-8B"
# model_name = "meta-llama/Llama-3.2-1B"
# model_name = "meta-llama/Llama-3.1-70B"
# model_name = "meta-llama/Llama-3.1-8B"
model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
##########################################
model.export()
model.compile(prefill_seq_len=128, ctx_len=256, num_cores=16, num_devices=1)  # Qpc file

# model.compile(
#     num_cores=14,
#     mxfp6=True,
#     device_group=[0],
# )
print("done")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("done")
model.generate(prompts=["Hi there!!"], tokenizer=tokenizer, device_group=[0])
print("done")

# from qgenie import ChatMessage, QGenieClient


# client = QGenieClient()


# chat_response = client.chat(
#     messages=[
#         ChatMessage(role="user", content="Analyze this repository: https://github.com/quic/efficient-transformers")
#     ],
#     max_tokens=400,
# )

# print(chat_response.first_content)
