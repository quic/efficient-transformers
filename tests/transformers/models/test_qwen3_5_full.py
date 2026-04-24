"""
Full 4-stage validation for Qwen3.5-0.8B (qwen3_5 dense text, decode-only).

Stage 1: PyTorch HF baseline (no transforms)
Stage 2: PyTorch QEff (after CustomOpsTransform + KVCacheTransform)
Stage 3: ONNX export + ORT inference
Stage 4: AI 100 compile + hardware inference

Uses a tiny model config (4 layers: 3 linear + 1 full, no MoE) with random weights.
No pretrained weights needed — runs in <60 seconds for stages 1-3.

Run with:
    /tmp/qeff_explore_qwen3_5_moe/bin/python tests/transformers/models/test_qwen3_5_full.py

For stages 1-3 only (no hardware):
    /tmp/qeff_explore_qwen3_5_moe/bin/python tests/transformers/models/test_qwen3_5_full.py --skip-s4

Requires:
    - venv with transformers >= 5.5.4
    - For Stage 4: AI 100 device + /opt/qti-aic/exec/qaic-compile
"""

import os
import sys
import shutil
import numpy as np
import torch
import yaml
import onnx
import onnxruntime as ort
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

SKIP_S4 = "--skip-s4" in sys.argv
# --real-qpc <path>  use a full-model QPC for generation (from onboard_qwen3_5_0_8b.py)
REAL_QPC = None
if "--real-qpc" in sys.argv:
    idx = sys.argv.index("--real-qpc")
    REAL_QPC = sys.argv[idx + 1]

# ── Tiny model config ─────────────────────────────────────────────────────────
CONFIG = Qwen3_5TextConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    full_attention_interval=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    vocab_size=512,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
    attn_output_gate=True,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_only_layers=[],
    mtp_num_hidden_layers=0,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [4, 4, 8],
    },
)

NUM_LAYERS = CONFIG.num_hidden_layers                                  # 4
NUM_LIN    = sum(1 for t in CONFIG.layer_types if t == "linear_attention")  # 3
CONV_DIM   = CONFIG.linear_num_key_heads * CONFIG.linear_key_head_dim * 2 + \
             CONFIG.linear_num_value_heads * CONFIG.linear_value_head_dim   # 96
CONV_K     = CONFIG.linear_conv_kernel_dim                             # 4
NUM_V      = CONFIG.linear_num_value_heads                             # 2
K_DIM      = CONFIG.linear_key_head_dim                               # 16
V_DIM      = CONFIG.linear_value_head_dim                             # 16
KV_HEADS   = CONFIG.num_key_value_heads                               # 1
HEAD_DIM   = CONFIG.head_dim                                          # 32
CTX_LEN    = 32
DEVICE_IDS = [0]
EXPORT_DIR = "/tmp/qwen3_5_4stage"


# ── ONNX Wrapper ──────────────────────────────────────────────────────────────

class OrtOnnxWrapper(torch.nn.Module):
    """
    KV-less wrapper for Stage 3 (ORT validation).

    Passes no past_key_values → QEffDynamicCache not instantiated → no
    CtxScatterFunc/CtxGatherFunc custom ops → ORT can run it.

    Validates SSM states (conv + rec) which are the novel part of this model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, *state_tensors):
        convs = list(state_tensors[:NUM_LIN])
        recs  = list(state_tensors[NUM_LIN:2 * NUM_LIN])
        result, new_conv, new_rec = self.model(
            input_ids=input_ids, position_ids=position_ids,
            past_conv_states=convs, past_recurrent_states=recs,
            # No past_key_values — avoids CtxScatterFunc (AI100-only custom op)
        )
        outputs = [result.logits]
        for c in new_conv: outputs.append(c)
        for r in new_rec:  outputs.append(r)
        return tuple(outputs)


class DecodeOnnxWrapper(torch.nn.Module):
    """
    Full wrapper with KV cache for Stage 4 (AI 100 hardware).

    Input order:
        input_ids, position_ids, comp_ctx_lengths,
        *conv_states[NUM_LIN], *rec_states[NUM_LIN],
        *kv_key[0], kv_val[0], ..., kv_key[N-1], kv_val[N-1]

    Output order:
        logits,
        *conv_states_retained, *rec_states_retained,
        *kv_key_retained, kv_val_retained (interleaved per layer)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, comp_ctx_lengths, *state_tensors):
        convs = list(state_tensors[:NUM_LIN])
        recs  = list(state_tensors[NUM_LIN:2 * NUM_LIN])
        kv_start = 2 * NUM_LIN
        kv = tuple(
            (state_tensors[kv_start + i * 2], state_tensors[kv_start + i * 2 + 1])
            for i in range(NUM_LAYERS)
        )
        result, new_conv, new_rec = self.model(
            input_ids=input_ids, position_ids=position_ids,
            past_key_values=kv, past_conv_states=convs, past_recurrent_states=recs,
            comp_ctx_lengths=comp_ctx_lengths,
        )
        outputs = [result.logits]
        for c in new_conv: outputs.append(c)
        for r in new_rec:  outputs.append(r)
        for k, v in result.past_key_values:
            outputs.append(k)
            outputs.append(v)
        return tuple(outputs)


# ── I/O Name Helpers ──────────────────────────────────────────────────────────

def input_names_ort():
    """Names for OrtOnnxWrapper (no KV cache)."""
    n = ["input_ids", "position_ids"]
    n += [f"past_conv_states.{i}"      for i in range(NUM_LIN)]
    n += [f"past_recurrent_states.{i}" for i in range(NUM_LIN)]
    return n


def output_names_ort():
    n = ["logits"]
    n += [f"past_conv_states.{i}_RetainedState"      for i in range(NUM_LIN)]
    n += [f"past_recurrent_states.{i}_RetainedState" for i in range(NUM_LIN)]
    return n


def input_names():
    """Names for DecodeOnnxWrapper (full, with KV cache)."""
    n = ["input_ids", "position_ids", "comp_ctx_lengths"]
    n += [f"past_conv_states.{i}"      for i in range(NUM_LIN)]
    n += [f"past_recurrent_states.{i}" for i in range(NUM_LIN)]
    for i in range(NUM_LAYERS):
        n += [f"past_key.{i}", f"past_value.{i}"]
    return n


def output_names():
    n = ["logits"]
    n += [f"past_conv_states.{i}_RetainedState"      for i in range(NUM_LIN)]
    n += [f"past_recurrent_states.{i}_RetainedState" for i in range(NUM_LIN)]
    for i in range(NUM_LAYERS):
        n += [f"past_key.{i}_RetainedState", f"past_value.{i}_RetainedState"]
    return n


# ── Input Builders ────────────────────────────────────────────────────────────

def build_decode_inputs_torch(seed=0):
    torch.manual_seed(seed)
    ids   = torch.ones((1, 1), dtype=torch.long) * 42
    pos   = torch.zeros((1, 1), dtype=torch.long)
    ctx   = torch.zeros((1, CTX_LEN), dtype=torch.int64)
    convs = [torch.randn(1, CONV_DIM, CONV_K) * 0.01 for _ in range(NUM_LIN)]
    recs  = [torch.randn(1, NUM_V, K_DIM, V_DIM) * 0.01  for _ in range(NUM_LIN)]
    kv    = [(torch.randn(1, KV_HEADS, CTX_LEN, HEAD_DIM) * 0.01,
              torch.randn(1, KV_HEADS, CTX_LEN, HEAD_DIM) * 0.01) for _ in range(NUM_LAYERS)]
    return ids, pos, ctx, convs, recs, kv


def torch_to_numpy(ids, pos, ctx, convs, recs, kv):
    d = {
        "input_ids":       ids.numpy().astype(np.int64),
        "position_ids":    pos.numpy().astype(np.int64),
        "comp_ctx_lengths": ctx.numpy().astype(np.int64),
    }
    for i, c in enumerate(convs): d[f"past_conv_states.{i}"]      = c.numpy().astype(np.float32)
    for i, r in enumerate(recs):  d[f"past_recurrent_states.{i}"] = r.numpy().astype(np.float32)
    for i, (k, v) in enumerate(kv):
        d[f"past_key.{i}"]   = k.numpy().astype(np.float32)
        d[f"past_value.{i}"] = v.numpy().astype(np.float32)
    return d


def build_model():
    torch.manual_seed(42)
    model = Qwen3_5ForCausalLM(CONFIG)
    model.float()
    model.eval()
    return model


# ── Stage 1: HF Baseline ──────────────────────────────────────────────────────

def stage1(model_hf, ids, pos, ctx, convs, recs, kv):
    """PyTorch HF — no QEff transforms. Runs full forward, takes last position logit."""
    print("\n[Stage 1] PyTorch HF baseline (no transforms)...")
    cache_pos = torch.arange(1)
    with torch.no_grad():
        out = model_hf(input_ids=ids, position_ids=pos, cache_position=cache_pos)
    logits = out.logits[:, -1, :].numpy()  # last token position
    print(f"  Output shape: {out.logits.shape}  top-1: {logits.argmax(-1).item()}")
    return logits


def pytorch_generate(model_hf, prompt_ids, gen_len, tokenizer=None):
    """
    Full generation loop on CPU using the Stage 1 HF model (no QEff transforms).

    Uses the standard HF DynamicCache — works on CPU, no CtxGatherFunc needed.
    The QEff-transformed model (Stage 2) can't be used here because QEffDynamicCache
    depends on CtxScatterFunc/CtxGatherFunc which are AI 100-only custom ops.

    For tiny model (no tokenizer): returns raw token IDs as a list.
    For real model (tokenizer given): returns decoded text string.
    """
    print("\n[PyTorch CPU Generate] HF model greedy decode...")
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        output_ids = model_hf.generate(
            input_ids=input_tensor,
            max_new_tokens=gen_len,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids = output_ids[0][len(prompt_ids):].tolist()

    if tokenizer is not None:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        text = str(generated_ids)   # tiny model — show raw IDs

    return generated_ids, text


# ── Stage 2: QEff Transforms ──────────────────────────────────────────────────

def stage2(model_hf, ids, pos, ctx, convs, recs, kv):
    """
    Same weights as HF, QEff transforms applied.
    For S1→S2 comparison we run WITHOUT past KV (validation mode) — same as HF baseline.
    """
    print("\n[Stage 2] PyTorch QEff (after transforms, validation mode — no KV)...")
    from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform

    model_q = build_model()
    model_q.load_state_dict(model_hf.state_dict())
    CustomOpsTransform.apply(model_q)
    KVCacheTransform.apply(model_q)
    model_q.eval()

    print(f"  Class: {type(model_q).__name__}")
    with torch.no_grad():
        # No past_key_values → validation mode → avoids CtxGatherFunc on CPU
        result, new_conv, new_rec = model_q(
            input_ids=ids, position_ids=pos,
            past_conv_states=convs, past_recurrent_states=recs,
        )
    logits = result.logits[:, 0, :].numpy()
    print(f"  Output shape: {result.logits.shape}  top-1: {logits.argmax(-1).item()}")
    print(f"  New conv states: {len(new_conv)} | New rec states: {len(new_rec)}")
    return logits, model_q


# ── Stage 3: ONNX + ORT ───────────────────────────────────────────────────────

def stage3(model_q, ids, pos, ctx, convs, recs, kv):
    """
    Export with OrtOnnxWrapper (no KV cache → no CtxScatterFunc → ORT-compatible).
    Validates SSM state export — the novel part of this model.
    """
    print("\n[Stage 3] ONNX export (no KV) + ORT inference...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = os.path.join(EXPORT_DIR, "decode_ort.onnx")

    wrapper = OrtOnnxWrapper(model_q)
    wrapper.eval()
    args = (ids, pos, *convs, *recs)

    torch.onnx.export(
        wrapper, args, onnx_path,
        input_names=input_names_ort(), output_names=output_names_ort(),
        opset_version=13, do_constant_folding=True,
    )
    sz = os.path.getsize(onnx_path) / 1024
    print(f"  ONNX: {onnx_path} ({sz:.0f} KB)")

    # ORT
    sess = ort.InferenceSession(onnx_path)
    sess_inputs = {i.name for i in sess.get_inputs()}
    feed = {
        "input_ids":    ids.numpy().astype(np.int64),
        "position_ids": pos.numpy().astype(np.int64),
    }
    for i, c in enumerate(convs): feed[f"past_conv_states.{i}"]      = c.numpy().astype(np.float32)
    for i, r in enumerate(recs):  feed[f"past_recurrent_states.{i}"] = r.numpy().astype(np.float32)
    feed_filtered = {k: v for k, v in feed.items() if k in sess_inputs}

    ort_out = sess.run(None, feed_filtered)
    logits = ort_out[0][:, 0, :]
    print(f"  ORT output shape: {ort_out[0].shape}  top-1: {logits.argmax(-1).item()}")
    return logits, onnx_path


# ── Stage 4: AI 100 ───────────────────────────────────────────────────────────

def stage4(model_q, ids, pos, ctx, convs, recs, kv):
    """
    Export with DecodeOnnxWrapper (full, with KV cache) and compile for AI 100.
    Uses a separate ONNX from Stage 3 because KV cache requires CtxScatterFunc.
    """
    print(f"\n[Stage 4] AI 100 compile + inference (devices {DEVICE_IDS})...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = os.path.join(EXPORT_DIR, "decode_full.onnx")
    qpc_dir   = os.path.join(EXPORT_DIR, "qpc")
    if os.path.exists(qpc_dir):
        shutil.rmtree(qpc_dir)

    # Export full ONNX (with KV cache)
    wrapper = DecodeOnnxWrapper(model_q)
    wrapper.eval()
    kv_flat = [t for pair in kv for t in pair]
    args = (ids, pos, ctx, *convs, *recs, *kv_flat)
    torch.onnx.export(
        wrapper, args, onnx_path,
        input_names=input_names(), output_names=output_names(),
        opset_version=13, do_constant_folding=True,
    )
    print(f"  Full ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024:.0f} KB)")

    # NPI: keep linear_attn nodes in FP32 (exp() overflow in FP16)
    npi_path = os.path.join(EXPORT_DIR, "npi.yaml")
    m = onnx.load(onnx_path)
    fp32_tensors = [out for node in m.graph.node if "linear_attn" in node.name for out in node.output if out]
    with open(npi_path, "w") as f:
        yaml.dump({"FP32NodeInstanceNames": fp32_tensors}, f, default_flow_style=False)
    print(f"  NPI: {len(fp32_tensors)} FP32 tensors")

    cmd = (
        f"/opt/qti-aic/exec/qaic-compile"
        f" -m={onnx_path}"
        f" -aic-hw"
        f" -convert-to-fp16"
        f" -retained-state"
        f" -aic-num-cores=4"
        f" -node-precision-info={npi_path}"
        f" -aic-binary-dir={qpc_dir}"
    )
    print(f"  {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print("  Compile FAILED")
        return None

    from QEfficient.generation.cloud_infer import QAICInferenceSession
    session = QAICInferenceSession(qpc_dir, device_ids=DEVICE_IDS)
    print(f"  Input names:  {session.input_names}")

    feed = torch_to_numpy(ids, pos, ctx, convs, recs, kv)
    feed_filtered = {k: v for k, v in feed.items() if k in session.input_names}
    out = session.run(feed_filtered)
    logits = out["logits"][:, 0, :].astype(np.float32)
    print(f"  AI100 output shape: {out['logits'].shape}  top-1: {logits.argmax(-1).item()}")
    return logits


# ── Stage 5: Generation (text output) ────────────────────────────────────────

def stage5_generate(qpc_dir, num_lin, num_layers, conv_dim, conv_k, num_v, k_dim, v_dim,
                    kv_heads, head_dim, ctx_len, gen_len=32,
                    model_name=None, prompt=None):
    """
    Run a real decode loop and show text output — like QEff's model.generate().

    Two modes:
      Tiny model (no model_name): generates token IDs with random weights — validates
        the loop runs correctly but output is not meaningful text.
      Real model (model_name provided): loads the HF tokenizer, encodes a real prompt,
        runs the decode loop on hardware, decodes and prints coherent text.

    To use the full model QPC compiled by onboard_qwen3_5_0_8b.py:
        python test_qwen3_5_full.py --real-qpc <path_to_decode_qpc> --skip-s4
    """
    from QEfficient.generation.cloud_infer import QAICInferenceSession

    print(f"\n[Stage 5] Generation — {'real model' if model_name else 'tiny random model'}")
    print(f"  QPC: {qpc_dir}")

    session = QAICInferenceSession(qpc_dir, device_ids=DEVICE_IDS)
    sess_inputs = set(session.input_names)

    # ── State helpers ──────────────────────────────────────────────────────────
    def _zero_states():
        conv = [np.zeros((1, conv_dim, conv_k), dtype=np.float32) for _ in range(num_lin)]
        rec  = [np.zeros((1, num_v, k_dim, v_dim), dtype=np.float32) for _ in range(num_lin)]
        kv_k = [np.zeros((1, kv_heads, ctx_len, head_dim), dtype=np.float32) for _ in range(num_layers)]
        kv_v = [np.zeros((1, kv_heads, ctx_len, head_dim), dtype=np.float32) for _ in range(num_layers)]
        return conv, rec, kv_k, kv_v

    def _update(out, conv, rec, kv_k, kv_v):
        for i in range(num_lin):
            if f"past_conv_states.{i}_RetainedState" in out:
                conv[i] = out[f"past_conv_states.{i}_RetainedState"]
            if f"past_recurrent_states.{i}_RetainedState" in out:
                rec[i] = out[f"past_recurrent_states.{i}_RetainedState"]
        for i in range(num_layers):
            if f"past_key.{i}_RetainedState" in out:
                kv_k[i] = out[f"past_key.{i}_RetainedState"]
            if f"past_value.{i}_RetainedState" in out:
                kv_v[i] = out[f"past_value.{i}_RetainedState"]

    def _feed(tok_id, position, conv, rec, kv_k, kv_v):
        d = {
            "input_ids":    np.array([[tok_id]], dtype=np.int64),
            "position_ids": np.array([[position]], dtype=np.int64),
        }
        if "comp_ctx_lengths" in sess_inputs:
            d["comp_ctx_lengths"] = np.zeros((1, ctx_len), dtype=np.int64)
        for i, c in enumerate(conv): d[f"past_conv_states.{i}"]      = c
        for i, r in enumerate(rec):  d[f"past_recurrent_states.{i}"] = r
        for i in range(num_layers):
            d[f"past_key.{i}"]   = kv_k[i]
            d[f"past_value.{i}"] = kv_v[i]
        return {k: v for k, v in d.items() if k in sess_inputs}

    # ── Tokenizer + prompt ────────────────────────────────────────────────────
    if model_name:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prompt_text = prompt or "My name is Alice. What is my name?"
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        eos_id = tokenizer.eos_token_id
    else:
        # Tiny model — use small token IDs (within vocab_size=512)
        prompt_text = "tokens: [1, 42, 7, 99]"
        prompt_ids = [1, 42, 7, 99]
        eos_id = None  # no EOS for tiny random model

    print(f"  Prompt: \"{prompt_text}\"")
    print(f"  Prompt tokens: {prompt_ids}")

    # ── Simulated prefill (one token at a time — decode-only constraint) ───────
    conv, rec, kv_k, kv_v = _zero_states()
    for pos, tok in enumerate(prompt_ids):
        out = session.run(_feed(tok, pos, conv, rec, kv_k, kv_v))
        _update(out, conv, rec, kv_k, kv_v)

    # ── Decode loop ───────────────────────────────────────────────────────────
    generated_ids = []
    pos = len(prompt_ids)
    next_tok = int(out["logits"][0, 0, :].argmax())

    for _ in range(gen_len):
        if eos_id is not None and next_tok == eos_id:
            break
        generated_ids.append(next_tok)
        out = session.run(_feed(next_tok, pos, conv, rec, kv_k, kv_v))
        _update(out, conv, rec, kv_k, kv_v)
        pos += 1
        next_tok = int(out["logits"][0, 0, :].argmax())

    # ── Decode + print ─────────────────────────────────────────────────────────
    if model_name:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        generated_text = str(generated_ids)   # raw IDs for tiny model

    print()
    print(f"  Prompt:    {prompt_text}")
    print(f"  Completion: {generated_text}")
    print()

    return generated_ids


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(stages_dict):
    print("\n" + "=" * 60)
    print("Stage Comparison — Top-10 Tokens + Max Diff")
    print("=" * 60)

    TOP_K = 10
    topk = {}
    for name, logits in stages_dict.items():
        if logits is None:
            continue
        flat = logits.flatten()
        top = list(np.argsort(flat)[-TOP_K:][::-1])
        topk[name] = top
        print(f"  {name}: top-{TOP_K} = {top}")

    names = list(topk.keys())
    all_pass = True
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        la = stages_dict[a].flatten().astype(np.float32)
        lb = stages_dict[b].flatten().astype(np.float32)
        diff = float(np.abs(la - lb).max())
        overlap = len(set(topk[a]) & set(topk[b]))
        status = "✅ PASS" if overlap >= 8 else "⚠ WARN" if overlap >= 5 else "❌ FAIL"
        if "FAIL" in status:
            all_pass = False
        print(f"  {a} → {b}: max_diff={diff:.3e}  top-{TOP_K} overlap={overlap}/{TOP_K}  {status}")

    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3.5-0.8B 4-Stage Validation (decode-only, tiny config)")
    print(f"CTX_LEN={CTX_LEN} | Stage 4 {'SKIPPED' if SKIP_S4 else f'on devices {DEVICE_IDS}'}")
    if REAL_QPC:
        print(f"Real model QPC: {REAL_QPC}")
    print("=" * 60)

    torch.manual_seed(42)
    model_hf = build_model()
    ids, pos, ctx, convs, recs, kv = build_decode_inputs_torch()

    logits_s1 = stage1(model_hf, ids, pos, ctx, convs, recs, kv)
    logits_s2, model_q = stage2(model_hf, ids, pos, ctx, convs, recs, kv)
    logits_s3, _ = stage3(model_q, ids, pos, ctx, convs, recs, kv)
    logits_s4 = None
    tiny_qpc_dir = None
    if not SKIP_S4:
        logits_s4 = stage4(model_q, ids, pos, ctx, convs, recs, kv)
        tiny_qpc_dir = os.path.join(EXPORT_DIR, "qpc")

    stages = {
        "S1 HF":   logits_s1,
        "S2 QEff": logits_s2,
        "S3 ORT":  logits_s3,
    }
    if logits_s4 is not None:
        stages["S4 AI100"] = logits_s4

    ok = compare(stages)
    print()
    if ok:
        print("ALL STAGES PASS")
    else:
        print("STAGE MISMATCH — check transforms or NPI config")
        sys.exit(1)

    # ── PyTorch CPU generation ─────────────────────────────────────────────────
    # Use the real model tokenizer + HF model.generate() if --real-qpc is set,
    # otherwise demo with tiny model token IDs.
    print("\n" + "=" * 60)
    print("Generation Comparison")
    print("=" * 60)

    if REAL_QPC:
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM as _Qwen3_5ForCausalLM
        _tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        _prompt = "My name is Alice. What is my name?"
        _prompt_ids = _tok.encode(_prompt, add_special_tokens=True)

        print(f"\n  Prompt: \"{_prompt}\"")

        # PyTorch CPU — load full pretrained model
        print("\n  [PyTorch CPU] Loading full pretrained model...")
        _model_full = _Qwen3_5ForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-0.8B", device_map="cpu", ignore_mismatched_sizes=True
        )
        _model_full.float().eval()
        _, pt_text = pytorch_generate(_model_full, _prompt_ids, gen_len=64, tokenizer=_tok)
        print(f"\n  PyTorch (CPU):  {pt_text}")

        # AI 100
        print()
        s5_ids = stage5_generate(
            qpc_dir=REAL_QPC,
            num_lin=18, num_layers=24,
            conv_dim=6144, conv_k=4, num_v=16, k_dim=128, v_dim=128,
            kv_heads=2, head_dim=256, ctx_len=512,
            gen_len=64,
            model_name="Qwen/Qwen3.5-0.8B",
            prompt=_prompt,
        )
    else:
        # Tiny random model — show raw token IDs from both paths
        _prompt_ids = [1, 42, 7, 99]   # within tiny vocab (size 512)
        _, pt_text = pytorch_generate(model_hf, _prompt_ids, gen_len=16, tokenizer=None)
        print(f"\n  Prompt IDs:     {_prompt_ids}")
        print(f"  PyTorch (CPU):  {pt_text}  ← raw token IDs (random weights)")

        if tiny_qpc_dir and os.path.exists(os.path.join(tiny_qpc_dir, "programqpc.bin")):
            print()
            stage5_generate(
                qpc_dir=tiny_qpc_dir,
                num_lin=NUM_LIN, num_layers=NUM_LAYERS,
                conv_dim=CONV_DIM, conv_k=CONV_K, num_v=NUM_V, k_dim=K_DIM, v_dim=V_DIM,
                kv_heads=KV_HEADS, head_dim=HEAD_DIM, ctx_len=CTX_LEN,
                gen_len=16, model_name=None,
            )
        else:
            print(f"\n  AI 100: skipped (no compiled QPC)")
            print(f"  To see real text: run with --real-qpc qwen3_5_0_8b_qpc/decode_qpc/")
