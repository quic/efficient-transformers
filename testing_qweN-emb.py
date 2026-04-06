# Requires transformers>=4.51.0
import torch
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def build_prompts(pairs):
    return [prefix + pair + suffix for pair in pairs]


@torch.no_grad()
def compute_logits_qaic(prompts):
    # Compile dimensions are derived from actual prompt lengths to avoid over-compiling.
    tok = tokenizer(prompts, return_tensors="np", padding=True, truncation=True, max_length=max_length)
    prompt_lengths = tok["attention_mask"].sum(axis=1)
    max_prompt_len = int(prompt_lengths.max())
    prefill_seq_len = max_prompt_len
    ctx_len = max_prompt_len + 1

    model.export()
    model.compile(prefill_seq_len=prefill_seq_len, ctx_len=ctx_len, batch_size=len(prompts))

    # Use the causal wrapper helper to fetch QAIC prefill logits.
    logits = model.generate_logits(
        tokenizer=tokenizer, prompts=prompts, device_id=[0], ctx_len=ctx_len, generation_len=1
    )
    batch_scores = torch.from_numpy(logits[:, -1, :])

    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    pair_scores = torch.stack([false_vector, true_vector], dim=1)
    pair_scores = torch.nn.functional.log_softmax(pair_scores, dim=1)
    scores = pair_scores[:, 1].exp().tolist()
    return scores


model_name = "Qwen/Qwen3-Reranker-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
model = QEFFAutoModelForCausalLM.from_pretrained(model_name)

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

task = "Given a web search query, retrieve relevant passages that answer the query"

queries = [
    "What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
    "responsible for the movement of planets around the sun.",
]

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]
prompts = build_prompts(pairs)
scores = compute_logits_qaic(prompts)

print("scores:", scores)
