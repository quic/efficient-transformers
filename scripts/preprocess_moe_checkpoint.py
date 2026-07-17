# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Convert a locally-stored HuggingFace MoE checkpoint for weight-free QAIC export.

Reads safetensors shards directly from local storage — no downloading, no
network access, no HuggingFace Hub authentication needed.

Stacks and pre-transposes per-expert weights into three tensors per layer:
  gate_proj_w : [E, H, I]  — gate weights transposed for x @ W bmm
  up_proj_w   : [E, H, I]  — up   weights transposed for x @ W bmm
  down_proj_w : [E, I, H]  — down weights transposed for intermediate @ W bmm

The weight-free forward() can then do a single Gather (gate_proj_w[flat_idx])
with no runtime torch.stack or .T — INT4 quantization propagates correctly.

Usage
-----
python -m QEfficient.utils.prepare_checkpoint_local \\
    --src  /home/huggingface_hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \\
    --out  /home/amarshar/qwen3-30b-prepared
"""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ── Constants ─────────────────────────────────────────────────────────────────

AUX_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
]

# Matches: model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.weight
EXPERT_RE = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


# ── LayerStacker ──────────────────────────────────────────────────────────────


class LayerStacker:
    """
    Collects all E experts for one MoE layer and produces HF-compatible batched tensors:

      gate_up_proj  [E, 2I, H]  = cat(gate_proj.weight, up_proj.weight, dim=0) per expert
      down_proj     [E, H,  I]  = down_proj.weight per expert stacked

    This matches Qwen3MoeExperts' batched parameter layout so QEfficient's
    weight-free exporter can find the keys directly in the checkpoint.
    """

    def __init__(self, prefix: str, num_experts: int):
        self.prefix = prefix
        self.num_experts = num_experts
        self._gate: torch.Tensor | None = None
        self._up: torch.Tensor | None = None
        self._down: torch.Tensor | None = None
        self.filled = 0

    def add(self, expert_idx: int, kind: str, t: torch.Tensor) -> None:
        t = t.to(torch.float16)
        if kind == "gate_proj":
            ffn_dim, hidden_dim = t.shape
            if self._gate is None:
                self._gate = torch.empty(self.num_experts, ffn_dim, hidden_dim, dtype=torch.float16)
            self._gate[expert_idx] = t
        elif kind == "up_proj":
            ffn_dim, hidden_dim = t.shape
            if self._up is None:
                self._up = torch.empty(self.num_experts, ffn_dim, hidden_dim, dtype=torch.float16)
            self._up[expert_idx] = t
        else:  # down_proj: [hidden_dim, ffn_dim]
            hidden_dim, ffn_dim = t.shape
            if self._down is None:
                self._down = torch.empty(self.num_experts, hidden_dim, ffn_dim, dtype=torch.float16)
            self._down[expert_idx] = t
        self.filled += 1

    @property
    def complete(self) -> bool:
        return self.filled == 3 * self.num_experts

    def tensors(self) -> dict[str, torch.Tensor]:
        # gate_up_proj: [E, 2I, H]  — matches HF Qwen3MoeExperts layout
        gate_up = torch.cat([self._gate, self._up], dim=1).contiguous()
        return {
            f"{self.prefix}.gate_up_proj": gate_up,  # [E, 2I, H]
            f"{self.prefix}.down_proj": self._down.contiguous(),  # [E, H, I]
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _atomic_save(tensors: dict, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
    tmp.replace(dst)


def _expert_shard_name(layer_idx: int) -> str:
    return f"experts-layer-{layer_idx:05d}.safetensors"


def _register_expert_keys(weight_map: dict, prefix: str, layer_idx: int) -> None:
    shard_name = _expert_shard_name(layer_idx)
    for key in ("gate_up_proj", "down_proj"):
        weight_map[f"{prefix}.{key}"] = shard_name


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Path to the local model snapshot directory (contains config.json and *.safetensors)",
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for the prepared checkpoint",
    )
    ap.add_argument(
        "--no-stack",
        action="store_true",
        help="Skip stacking; only convert BF16→FP32 (keeps per-expert layout)",
    )
    args = ap.parse_args()

    src: Path = args.src.resolve()
    out: Path = args.out.resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    if not (src / "config.json").exists():
        raise FileNotFoundError(f"No config.json found in {src} — is this a model snapshot directory?")

    out.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    # ── Step 1: Copy auxiliary files ──────────────────────────────────────────
    print(f"[src] {src}")
    print(f"[out] {out}")
    for name in AUX_FILES:
        if (src / name).exists() and not (out / name).exists():
            shutil.copy2(src / name, out / name)
            print(f"[aux] {name}")

    # ── Step 2: Read config ────────────────────────────────────────────────────
    config = json.loads((out / "config.json").read_text())
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config.get("num_experts") or config.get("n_routed_experts") or config.get("num_local_experts"))
    print(f"[config] num_hidden_layers={num_layers}  num_experts={num_experts}")

    # ── Step 3: Read shard index ───────────────────────────────────────────────
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        # Single-file model
        shard_files = sorted(src.glob("*.safetensors"))
        weight_map = {k: shard_files[0].name for k in ["*"]}
    else:
        index_data = json.loads(index_path.read_text())
        weight_map: dict[str, str] = index_data["weight_map"]

    shard_names = sorted(set(weight_map.values()))
    print(f"[index] {len(shard_names)} shards,  {len(weight_map)} keys")

    # ── Step 4: Process each shard ────────────────────────────────────────────
    new_weight_map: dict[str, str] = {}
    stackers: dict[int, LayerStacker] = {}
    total_layers_stacked = 0

    for si, shard_name in enumerate(shard_names):
        shard_path = src / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")

        base_out = out / (f"base-{shard_name}" if not args.no_stack else shard_name)
        sentinel = out / f".done-{Path(shard_name).stem}"

        # ── Resume: already processed ─────────────────────────────────────────
        if sentinel.exists():
            print(f"[{si + 1}/{len(shard_names)}] {shard_name}  (already done, skipping)")
            for key, v in weight_map.items():
                if v != shard_name:
                    continue
                m = EXPERT_RE.match(key)
                if m and not args.no_stack:
                    _register_expert_keys(new_weight_map, m.group(1), int(m.group(2)))
                else:
                    new_weight_map[key] = base_out.name
            continue

        # ── Process shard ─────────────────────────────────────────────────────
        proc_start = time.perf_counter()
        shard_size_mb = shard_path.stat().st_size / 1e6
        print(f"[{si + 1}/{len(shard_names)}] reading {shard_name}  ({shard_size_mb:.0f} MB)")

        base_tensors: dict[str, torch.Tensor] = {}
        layers_stacked_this_shard = 0

        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                # Skip MTP/speculative layers beyond num_layers
                layer_m = LAYER_RE.match(key)
                if layer_m and int(layer_m.group(1)) >= num_layers:
                    continue

                m = EXPERT_RE.match(key)
                if m and not args.no_stack:
                    # Expert weight — accumulate in stacker
                    li, ei, kind = int(m.group(2)), int(m.group(3)), m.group(4)
                    st = stackers.setdefault(li, LayerStacker(m.group(1), num_experts))
                    st.add(ei, kind, f.get_tensor(key))
                    _register_expert_keys(new_weight_map, m.group(1), li)

                    if st.complete:
                        stack_start = time.perf_counter()
                        out_path = out / _expert_shard_name(li)
                        stacked = st.tensors()
                        _atomic_save(stacked, out_path)
                        stack_elapsed = time.perf_counter() - stack_start

                        g_shape = stacked[f"{st.prefix}.gate_up_proj"].shape
                        d_shape = stacked[f"{st.prefix}.down_proj"].shape
                        print(
                            f"    [STACKED] layer {li}: "
                            f"gate_up_proj {tuple(g_shape)}, "
                            f"down_proj {tuple(d_shape)}  "
                            f"→ {out_path.stat().st_size / 1e6:.0f} MB  ({stack_elapsed:.1f}s)"
                        )
                        layers_stacked_this_shard += 1
                        total_layers_stacked += 1
                        del stackers[li]
                else:
                    # Non-expert weight — BF16→FP16 passthrough
                    t = f.get_tensor(key)
                    base_tensors[key] = t.to(torch.float16) if t.is_floating_point() else t
                    new_weight_map[key] = base_out.name

        if base_tensors:
            _atomic_save(base_tensors, base_out)

        sentinel.touch()
        proc_elapsed = time.perf_counter() - proc_start
        base_mb = base_out.stat().st_size / 1e6 if base_out.exists() else 0
        print(
            f"    → {base_out.name}  "
            f"({len(base_tensors)} base tensors, {base_mb:.0f} MB, "
            f"{layers_stacked_this_shard} layer(s) stacked)  {proc_elapsed:.1f}s"
        )

    # ── Step 5: Sanity check ──────────────────────────────────────────────────
    if stackers:
        raise RuntimeError(
            f"Incomplete expert layers after all shards: {sorted(stackers.keys())} — some expert weights are missing."
        )

    # ── Step 6: Update config dtype ───────────────────────────────────────────
    for k in ("dtype", "torch_dtype"):
        if k in config:
            config[k] = "float16"
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # ── Step 7: Write new index ───────────────────────────────────────────────
    output_files = sorted(set(new_weight_map.values()))
    total_size = sum((out / f).stat().st_size for f in output_files)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 60}")
    print(f"[SUMMARY] Source         : {src}")
    print(f"[SUMMARY] Output         : {out}")
    print(f"[SUMMARY] Layers stacked : {total_layers_stacked}")
    print(f"[SUMMARY] Output files   : {len(output_files)}")
    print(f"[SUMMARY] Total size     : {total_size / 1e9:.2f} GB")
    print(f"[SUMMARY] Total time     : {total_elapsed:.1f}s  ({total_elapsed / 60:.1f} min)")
    print(f'[SUMMARY] Use as         : model_name_or_path = "{out}"')
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
