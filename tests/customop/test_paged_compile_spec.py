# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Verify compile(paged_kv=True) injects block-pool dims into every specialization.

The AIC compiler is box-only, so we monkeypatch _compile to capture the
`specializations` it would receive and assert each carries num_blocks / page_size
/ max_num_blocks (the dynamic axes declared by export(paged_kv=True)), and that the
paged_* kwargs are NOT leaked to the compiler.
"""

import pytest
import torch

pytest.importorskip("QEfficient")

from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM  # noqa: E402


def test_compile_paged_specializations():
    cfg = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rms_norm_eps=1e-6, tie_word_embeddings=False,
    )
    cfg.torch_dtype = torch.float32
    torch.manual_seed(0)
    qeff = QEFFAutoModelForCausalLM(Qwen2ForCausalLM(cfg), continuous_batching=False)

    captured = {}

    def _fake_compile(*args, **kwargs):
        captured["specializations"] = kwargs.get("specializations")
        captured["compiler_options"] = {
            k: kwargs.get(k) for k in ("paged_kv", "page_size", "num_blocks", "max_num_blocks")
        }
        return "/tmp/fake.qpc"

    qeff._compile = _fake_compile

    page_size, num_blocks = 8, 33
    ctx_len = 64
    qeff.compile(
        onnx_path="/tmp/does-not-matter.onnx",
        prefill_seq_len=16,
        ctx_len=ctx_len,
        paged_kv=True,
        page_size=page_size,
        num_blocks=num_blocks,
    )

    specs = captured["specializations"]
    assert specs and len(specs) >= 1, "no specializations built"
    expected_max_blocks = -(-ctx_len // page_size)  # 8
    for spec in specs:
        assert spec.get("num_blocks") == num_blocks, f"num_blocks missing/wrong: {spec}"
        assert spec.get("page_size") == page_size, f"page_size missing/wrong: {spec}"
        assert spec.get("max_num_blocks") == expected_max_blocks, f"max_num_blocks wrong: {spec}"
        assert "ctx_len" in spec, "ctx_len (attention_mask dim) must remain in spec"

    # paged_* must NOT leak to the AIC compiler as flags.
    for k, v in captured["compiler_options"].items():
        assert v is None, f"paged kwarg {k} leaked to _compile"

    print(f"[metrics] compile_paged_spec: {len(specs)} specs, each has "
          f"num_blocks={num_blocks} page_size={page_size} max_num_blocks={expected_max_blocks}")


if __name__ == "__main__":
    from _metrics import measure

    with measure("test_paged_compile_spec"):
        test_compile_paged_specializations()
    print("PAGED COMPILE SPEC: PASS")
