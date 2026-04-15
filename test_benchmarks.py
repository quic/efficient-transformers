from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.blocking.attention_blocking import AttentionBlockingConfig, BlockingMode

m = QEFFAutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM", enable_benchmark=True)
bc = AttentionBlockingConfig(mode=BlockingMode.HQKV, num_kv_blocks=4, num_q_blocks=2, head_block_size=1)
m.compile(prefill_seq_len=32, ctx_len=128, blocking_config=bc)
m.generate(tokenizer=None, prompts=[])
