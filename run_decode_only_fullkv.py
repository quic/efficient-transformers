
from transformers import AutoConfig, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_id = "openai/gpt-oss-20b"  # weights are not required to convert to fp32

prompt = """
Once upon a time, in a small town, there lived a young boy named Alex. Alex was a curious and adventurous child, always eager to explore the world around him. One day, while playing in the park, Alex stumbled upon a mysterious old book hidden beneath a pile of leaves. The book was filled with stories of distant lands, magical creatures, and extraordinary adventures.

As Alex flipped through the pages, he discovered a map that led to a hidden treasure. Excited by the prospect of a real-life treasure hunt, Alex decided to embark on a thrilling journey. He packed his backpack with snacks, a flashlight, and a compass, and set off into the unknown.

The path to the treasure was not an easy one. Alex had to navigate through dense forests, cross rickety bridges, and solve riddles that guarded the treasure's location.
"""
# Run prefill
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
PREFILL_SEQ_LEN = 128
CTX_LEN = 8192

qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_id, continuous_batching=True)

decode_qpc_path = qeff_model.compile(
    prefill_seq_len=1,
    ctx_len=CTX_LEN,
    num_cores=16,
    num_devices=1,
    mos=1,
    aic_enable_depth_first=True,
    num_speculative_tokens=None,
    offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation in the next step
    retain_full_kv=True,
    full_batch_siz=4
)


# decode_qpc_path = qeff_model.compile(
#     prefill_seq_len=1,
#     ctx_len=CTX_LEN,
#     num_cores=16,
#     mxfp6_matmul=True,
#     mxint8_kv_cache=True,
#     num_devices=4,
#     mos=1,
#     full_batch_size=4,
#     # kv_cache_batch_size=8,
#     aic_enable_depth_first=True,
#     num_speculative_tokens=None,
#     offload_pt_weights=False,  # Need the weights in memory for prefill-model export/compilation in the next step
# )
qeff_model.generate(
    prompts=["My name is"], tokenizer=tokenizer, device_id=[i for i in range(16, 32)], generation_len=120
)
