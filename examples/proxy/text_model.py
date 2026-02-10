from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

causal_models = [
    # "allenai/Molmo-7B-D-0924",
    # "allenai/OLMo-2-0425-1B",
    # "tiiuae/falcon-40b",
    # "Qwen/Qwen3-30B-A3B-Instruct-2507",
    # "google/codegemma-2b",
    # "google/codegemma-7b",
    # "google/gemma-2b",
    # "google/gemma-7b",
    # "google/gemma-2-2b",
    # "google/gemma-2-9b",
    # "google/gemma-2-27b",
    # "openai/gpt-oss-20b",
    # "bigcode/starcoder",
    # "bigcode/starcoder2-15b",
    # "EleutherAI/gpt-j-6b",
    "openai-community/gpt2",
    # "ibm-granite/granite-3.1-8b-instruct",
    # "ibm-granite/granite-guardian-3.1-8b",
    # "ibm-granite/granite-20b-code-base-8k",
    # "ibm-granite/granite-20b-code-instruct-8k",
    # "OpenGVLab/InternVL2_5-1B",
    # "OpenGVLab/InternVL3_5-1B",
    # "codellama/CodeLlama-7b-hf",
    # "codellama/CodeLlama-13b-hf",
    # "codellama/CodeLlama-34b-hf",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    # "inceptionai/jais-adapted-7b",
    # "inceptionai/jais-adapted-13b-chat",
    # "inceptionai/jais-adapted-70b",
    # "meta-llama/Llama-3.3-70B-Instruct",
    # "meta-llama/Llama-3.2-1B",
    # "meta-llama/Llama-3.2-3B",
    # "meta-llama/Llama-3.1-8B",
    # "meta-llama/Llama-3.1-70B",
    # "meta-llama/Meta-Llama-3-8B",
    # "meta-llama/Meta-Llama-3-70B",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "meta-llama/Llama-2-70b-chat-hf",
    # "lmsys/vicuna-13b-delta-v0",
    # "lmsys/vicuna-13b-v1.3",
    # "lmsys/vicuna-13b-v1.5",
    # "mistralai/Mistral-7B-Instruct-v0.1",
    # "mistralai/Codestral-22B-v0.1",
    # "mistralai/Mixtral-8x7B-v0.1",
    # "microsoft/Phi-3-mini-4k-instruct",
    # "DeepSeek-R1-Distill-Qwen-32B",
    # "Qwen/Qwen2-1.5B-Instruct",
    # "Snowflake/Llama-3.1-SwiftKV-8B-Instruct",
    # "hpcai-tech/grok-1",
]

for model_name in causal_models:
    print(f"\n\nTesting model: {model_name}")
    layers = 2
    export_dir = "qeff_models/" + model_name.split("/")[-1] + f"_{layers}layers"
    model = QEFFAutoModelForCausalLM.from_pretrained(model_name, num_hidden_layers=layers)  # Standard model load
    print("\n============Original Qeff Model===============\n")
    print(model)
    print("\n=====================================\n")

    model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name, num_hidden_layers=layers, enable_proxy=True
    )  # enable_proxy=True to use proxy model export i.e., export model disable the embedding and LM head layers
    print("\n============Proxy Qeff Model===============\n")
    print(model)
    print("\n=====================================\n")
    model.export(export_dir=export_dir)  # export the proxy model to disk
    model.compile(num_cores=16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.generate(prompts=["Hi there!!"], tokenizer=tokenizer, write_io=True)  # write_io = True to save io files


"""
Testing model: openai-community/gpt2

============Original Qeff Model===============

QEFFAutoModelForCausalLM
QEffGPT2LMHeadModel(
  (transformer): QEffGPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x QEffGPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): QEffGPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

=====================================


============Proxy Qeff Model===============

QEFFAutoModelForCausalLM
QEffGPT2LMHeadModel(
  (transformer): QEffGPT2Model(
    (wte): QeffProxyEmbedding()
    (wpe): QeffProxyEmbedding()
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x QEffGPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): QEffGPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): QeffProxyLinear()
)

=====================================
Prompt : Hi there!!
Completion :
========================= Performance Stats =========================
Average Prefill time a.k.a TTFT is= 0.01 sec        
Decode is= 707.13 tokens/sec        
Total is= 679.82 tokens/sec        
Total (E2E) inference time is= 0.18 sec
=====================================================================

"""
