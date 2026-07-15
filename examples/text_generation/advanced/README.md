# Advanced text-generation examples

These scripts don't fit the canonical
``QEFFAutoModelForCausalLM.from_pretrained -> .compile -> .generate`` pipeline
that lives in ``../basic_inference.py``. Each one carries a workload-specific
workaround that would clutter the main entry point if promoted:

- **``kimik2_mla_absorption.py``** — Kimi-K2 Multi-head Latent Attention with
  a hand-rolled prefill + decode loop. ``qaic_config["mla_absorption"]``
  changes the KV cache shape (``compressed_kvs`` / ``k_pe`` low-rank buffers),
  so the standard ``.generate()`` runtime doesn't apply and the script drives
  ``qeff_model.model(**inputs)`` directly.

- **``glm4_kv_head_surgery.py``** — GLM-4-MoE with live ``torch.repeat_interleave``
  surgery on ``k_proj`` / ``v_proj`` weights and biases before compile. The
  goal is to make ``num_key_value_heads`` evenly shardable across
  ``num_devices`` at the cost of extra KV traffic. This is a deployment-shape
  fix, not a transform — it stays out of the main pipeline on purpose.

If your model works with the standard Auto API (which is the vast majority),
use ``../basic_inference.py``. Reach into this folder only when the workload
below matches.
