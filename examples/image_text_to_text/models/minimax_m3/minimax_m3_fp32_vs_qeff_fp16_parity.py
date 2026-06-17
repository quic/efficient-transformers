# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""MiniMax-M3 parity gate: original fp32 vs QEff-modified fp16.

Workflow:
1) Run parity on reduced text-layer config (default: 8 layers)
2) If gate passes, run parity on full model layers
3) Write a markdown report with metrics and pass/fail decisions
"""

from __future__ import annotations

import argparse
import copy
import gc
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.transformers.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration, MiniMaxM3VLConfig

MODEL_ID = "MiniMaxAI/MiniMax-M3"
PROMPT = "tell me about yourself, Mini!"


@dataclass
class ParityResult:
    label: str
    layers: int
    run_ok: bool
    mad: float | None = None
    max_abs: float | None = None
    cos_sim: float | None = None
    top1_match: bool | None = None
    top1_orig_id: int | None = None
    top1_qeff_id: int | None = None
    top1_orig_token: str | None = None
    top1_qeff_token: str | None = None
    gate_pass: bool = False
    error: str = ""


def _setup_hf_cache() -> None:
    hf_home = os.environ.setdefault("HF_HOME", "/tmp/hf_minimax_cache")
    hub_cache = os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{hf_home}/hub")
    os.environ.setdefault("HF_HUB_CACHE", hub_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{hf_home}/transformers")
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    Path(hub_cache).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)


def _as_local_minimax_config(config) -> MiniMaxM3VLConfig:
    if isinstance(config, MiniMaxM3VLConfig):
        return config
    return MiniMaxM3VLConfig(**config.to_dict())


def _slice_or_pad(values: list[int], target_len: int, pad_value: int) -> list[int]:
    values = list(values)
    if len(values) < target_len:
        values.extend([pad_value] * (target_len - len(values)))
    return values[:target_len]


def _set_text_layers(config: MiniMaxM3VLConfig, num_layers: int) -> None:
    text = config.text_config
    text.num_hidden_layers = int(num_layers)

    if getattr(text, "layer_types", None) is not None:
        layer_types = list(text.layer_types)
        if len(layer_types) < text.num_hidden_layers:
            layer_types.extend(["full_attention"] * (text.num_hidden_layers - len(layer_types)))
        text.layer_types = layer_types[: text.num_hidden_layers]

    if getattr(text, "mlp_layer_types", None) is not None:
        mlp_types = list(text.mlp_layer_types)
        if len(mlp_types) < text.num_hidden_layers:
            mlp_types.extend(["sparse"] * (text.num_hidden_layers - len(mlp_types)))
        text.mlp_layer_types = mlp_types[: text.num_hidden_layers]


def _build_dummy_config_from_autoconfig(base_cfg: MiniMaxM3VLConfig, layers: int) -> MiniMaxM3VLConfig:
    """Create a fast, config-derived MiniMax shape while preserving model semantics."""
    config = MiniMaxM3VLConfig(**copy.deepcopy(base_cfg.to_dict()))
    _set_text_layers(config, layers)

    text = config.text_config
    vision = config.vision_config

    # Keep tokenizer-compatible vocab but shrink hidden dimensions aggressively.
    text.hidden_size = 64
    text.intermediate_size = 32
    text.dense_intermediate_size = 128
    text.shared_intermediate_size = 32
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.rotary_dim = min(getattr(text, "rotary_dim", 16), text.head_dim)
    text.max_position_embeddings = max(128, min(getattr(text, "max_position_embeddings", 128), 512))
    text.num_local_experts = 4
    text.num_experts_per_tok = min(2, text.num_local_experts)
    text.n_shared_experts = min(getattr(text, "n_shared_experts", 1), 1)
    text.routed_scaling_factor = float(getattr(text, "routed_scaling_factor", 1.0))

    if getattr(text, "moe_layer_freq", None) is not None:
        text.moe_layer_freq = _slice_or_pad(list(text.moe_layer_freq), layers, 1)

    sparse_cfg = dict(getattr(text, "sparse_attention_config", {}) or {})
    sparse_cfg["use_sparse_attention"] = bool(sparse_cfg.get("use_sparse_attention", True))
    sparse_cfg["sparse_index_dim"] = min(int(sparse_cfg.get("sparse_index_dim", 16)), text.hidden_size)
    sparse_cfg["sparse_num_index_heads"] = min(
        int(sparse_cfg.get("sparse_num_index_heads", 2)), text.num_attention_heads
    )
    sparse_cfg["sparse_topk_blocks"] = 2
    sparse_cfg["sparse_block_size"] = 4
    sparse_cfg["sparse_local_block"] = max(0, int(sparse_cfg.get("sparse_local_block", 1)))
    sparse_cfg["sparse_init_block"] = int(sparse_cfg.get("sparse_init_block", 0))
    sparse_cfg["sparse_score_type"] = sparse_cfg.get("sparse_score_type", "max")
    sparse_cfg["sparse_disable_index_value"] = _slice_or_pad(
        list(sparse_cfg.get("sparse_disable_index_value", [])),
        layers,
        1,
    )
    sparse_cfg["sparse_attention_freq"] = _slice_or_pad(
        list(sparse_cfg.get("sparse_attention_freq", [])),
        layers,
        1,
    )
    text.sparse_attention_config = sparse_cfg

    vision.hidden_size = 64
    vision.intermediate_size = 128
    vision.num_hidden_layers = min(int(getattr(vision, "num_hidden_layers", 2)), 2)
    vision.num_attention_heads = min(int(getattr(vision, "num_attention_heads", 4)), 4)
    vision.num_channels = int(getattr(vision, "num_channels", 3))
    vision.image_size = min(int(getattr(vision, "image_size", 56)), 56)
    vision.patch_size = min(int(getattr(vision, "patch_size", 14)), 14)
    if not hasattr(vision, "temporal_patch_size") or vision.temporal_patch_size is None:
        vision.temporal_patch_size = 2
    if hasattr(vision, "spatial_merge_size"):
        vision.spatial_merge_size = max(1, min(int(getattr(vision, "spatial_merge_size", 2)), 2))
    spatial_merge = int(getattr(vision, "spatial_merge_size", 1))

    # Keep projector dimensions coherent with resized text/vision hidden sizes.
    config.projector_hidden_size = max(128, text.hidden_size * 2)
    config.merged_hidden_size = text.hidden_size * (spatial_merge**2)

    vocab_size = int(getattr(text, "vocab_size", 0))
    if vocab_size > 1:
        config.image_token_index = min(int(getattr(config, "image_token_index", vocab_size - 2)), vocab_size - 2)
        config.video_token_index = min(int(getattr(config, "video_token_index", vocab_size - 1)), vocab_size - 1)

    return config


def _build_inputs(tokenizer, config: MiniMaxM3VLConfig, prompt: str):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(torch.int64)
    vocab_size = int(getattr(config.text_config, "vocab_size", 0))
    if vocab_size > 0 and int(input_ids.max().item()) >= vocab_size:
        input_ids = input_ids % vocab_size
    batch_size, seq_len = input_ids.shape

    image_token_id = int(getattr(config, "image_token_index", 0))
    image_token = torch.full((batch_size, 1), image_token_id, dtype=torch.int64)
    input_ids = torch.cat([image_token, input_ids], dim=1)
    seq_len = input_ids.shape[1]

    position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(batch_size, 1)
    temporal_patch_size = int(getattr(config.vision_config, "temporal_patch_size", 1) or 1)
    patch_dim = (
        config.vision_config.num_channels
        * temporal_patch_size
        * config.vision_config.patch_size
        * config.vision_config.patch_size
    )
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    num_image_patches = int(torch.prod(image_grid_thw).item())
    pixel_values_fp32 = torch.zeros((num_image_patches, patch_dim), dtype=torch.float32)
    pixel_values_fp16 = pixel_values_fp32.half()
    return input_ids, position_ids, image_grid_thw, pixel_values_fp32, pixel_values_fp16


def _select_qeff_logit_position(orig_logits: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    # QEff wrapper returns logits only at argmax(position_ids) per batch.
    logit_idx = position_ids.to(torch.int64).argmax(dim=1, keepdim=True)
    batch_idx = torch.arange(position_ids.shape[0], dtype=torch.int64).view(-1, 1)
    return orig_logits[batch_idx, logit_idx]


def _safe_token_decode(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def _fmt(v: float | None, ndigits: int = 6) -> str:
    if v is None:
        return "NA"
    return f"{v:.{ndigits}f}"


def run_parity(
    *,
    model_id: str,
    base_cfg: MiniMaxM3VLConfig,
    tokenizer,
    prompt: str,
    layers: int,
    mad_threshold: float,
    cos_threshold: float,
    seed: int,
    use_pretrained_weights: bool,
) -> ParityResult:
    label = f"{layers}-layer"
    try:
        if use_pretrained_weights:
            local_cfg = MiniMaxM3VLConfig(**copy.deepcopy(base_cfg.to_dict()))
            _set_text_layers(local_cfg, layers)
        else:
            local_cfg = _build_dummy_config_from_autoconfig(base_cfg, layers)

        input_ids, position_ids, image_grid_thw, pixel_values_fp32, pixel_values_fp16 = _build_inputs(
            tokenizer, local_cfg, prompt
        )

        # 1) Original model in fp32
        cfg_fp32 = MiniMaxM3VLConfig(**local_cfg.to_dict())
        cfg_fp32.torch_dtype = torch.float32
        cfg_fp32.text_config.torch_dtype = torch.float32
        run_seed = int(seed + layers)
        if use_pretrained_weights:
            model_fp32 = MiniMaxM3SparseForConditionalGeneration.from_pretrained(
                model_id,
                config=cfg_fp32,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            torch.manual_seed(run_seed)
            model_fp32 = MiniMaxM3SparseForConditionalGeneration(cfg_fp32)
        model_fp32.eval()

        with torch.no_grad():
            out_fp32 = model_fp32(
                input_ids=input_ids,
                position_ids=position_ids,
                pixel_values=pixel_values_fp32,
                image_grid_thw=image_grid_thw,
                use_cache=False,
            )
            ref_logits = _select_qeff_logit_position(out_fp32.logits.float(), position_ids)

        del out_fp32
        del model_fp32
        gc.collect()

        # 2) QEff-modified model in fp16
        cfg_fp16 = MiniMaxM3VLConfig(**local_cfg.to_dict())
        cfg_fp16.torch_dtype = torch.float16
        cfg_fp16.text_config.torch_dtype = torch.float16
        if use_pretrained_weights:
            model_fp16 = MiniMaxM3SparseForConditionalGeneration.from_pretrained(
                model_id,
                config=cfg_fp16,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            torch.manual_seed(run_seed)
            model_fp16 = MiniMaxM3SparseForConditionalGeneration(cfg_fp16)
            model_fp16.half()
        model_fp16.eval()

        qeff_wrapper = QEFFAutoModelForImageTextToText(model_fp16, kv_offload=False)
        qeff_wrapper.model.eval()

        with torch.no_grad():
            out_qeff = qeff_wrapper.model(
                input_ids=input_ids,
                position_ids=position_ids,
                pixel_values=pixel_values_fp16,
                image_grid_thw=image_grid_thw,
            )
            qeff_logits = out_qeff[0].float()

        diff = (ref_logits - qeff_logits).abs()
        mad = float(diff.mean().item())
        max_abs = float(diff.max().item())
        cos_sim = float(F.cosine_similarity(ref_logits.flatten(1), qeff_logits.flatten(1), dim=1).mean().item())

        top1_orig_id = int(ref_logits.argmax(dim=-1).item())
        top1_qeff_id = int(qeff_logits.argmax(dim=-1).item())
        top1_match = top1_orig_id == top1_qeff_id

        gate_pass = bool(top1_match and mad <= mad_threshold and cos_sim >= cos_threshold)

        del out_qeff
        del qeff_wrapper
        del model_fp16
        gc.collect()

        return ParityResult(
            label=label,
            layers=layers,
            run_ok=True,
            mad=mad,
            max_abs=max_abs,
            cos_sim=cos_sim,
            top1_match=top1_match,
            top1_orig_id=top1_orig_id,
            top1_qeff_id=top1_qeff_id,
            top1_orig_token=_safe_token_decode(tokenizer, top1_orig_id),
            top1_qeff_token=_safe_token_decode(tokenizer, top1_qeff_id),
            gate_pass=gate_pass,
        )
    except Exception:
        return ParityResult(
            label=label,
            layers=layers,
            run_ok=False,
            gate_pass=False,
            error=traceback.format_exc(),
        )


def write_report(
    *,
    report_path: Path,
    model_id: str,
    prompt: str,
    initial_layers: int,
    use_pretrained_weights: bool,
    seed: int,
    eight_layer_result: ParityResult,
    full_layer_result: ParityResult | None,
    mad_threshold: float,
    cos_threshold: float,
):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# MiniMax-M3 fp32 vs QEff fp16 Parity Report")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Timestamp: `{now_utc}`")
    lines.append(f"- Model: `{model_id}`")
    lines.append(f"- Prompt: `{prompt}`")
    lines.append(
        f"- Weights mode: `{'pretrained checkpoint' if use_pretrained_weights else 'autoconfig-derived dummy'}`"
    )
    lines.append(f"- Random seed: `{seed}`")
    lines.append(f"- Gate thresholds: `MAD <= {mad_threshold}`, `Cosine >= {cos_threshold}`, `Top1 match == True`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Run | Layers | Status | Top1 Match | MAD | MaxAbs | Cosine | Top1 Orig | Top1 QEff | Gate |")
    lines.append("|---|---:|---|---|---:|---:|---:|---|---|---|")

    def _row(name: str, r: ParityResult) -> str:
        return (
            f"| {name} | {r.layers} | {'PASS' if r.run_ok else 'FAIL'} | {r.top1_match} | "
            f"{_fmt(r.mad)} | {_fmt(r.max_abs)} | {_fmt(r.cos_sim)} | "
            f"`{r.top1_orig_id}` ({r.top1_orig_token}) | `{'None' if r.top1_qeff_id is None else r.top1_qeff_id}` ({r.top1_qeff_token}) | "
            f"{'PASS' if r.gate_pass else 'FAIL'} |"
        )

    lines.append(_row(f"{initial_layers}-layer parity", eight_layer_result))
    if full_layer_result is not None:
        lines.append(_row("Full-model parity", full_layer_result))

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if not eight_layer_result.run_ok:
        lines.append(f"- {initial_layers}-layer run failed to execute; full-model parity was skipped.")
    elif not eight_layer_result.gate_pass:
        lines.append(f"- {initial_layers}-layer gate did not pass; full-model parity was skipped by policy.")
    elif full_layer_result is None:
        lines.append(f"- {initial_layers}-layer gate passed; full-model run was not requested.")
    elif full_layer_result.run_ok and full_layer_result.gate_pass:
        lines.append(f"- {initial_layers}-layer and full-model gates both passed.")
    elif full_layer_result.run_ok:
        lines.append(f"- {initial_layers}-layer gate passed, but full-model gate did not pass.")
    else:
        lines.append(f"- {initial_layers}-layer gate passed, but full-model run failed to execute.")

    if eight_layer_result.error:
        lines.append("")
        lines.append(f"## {initial_layers}-layer Error")
        lines.append("")
        lines.append("```text")
        lines.append(eight_layer_result.error.strip())
        lines.append("```")

    if full_layer_result is not None and full_layer_result.error:
        lines.append("")
        lines.append("## Full-model Error")
        lines.append("")
        lines.append("```text")
        lines.append(full_layer_result.error.strip())
        lines.append("```")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniMax-M3 fp32 vs QEff fp16 parity gate")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--layers", type=int, default=8, help="Text layers for the first parity gate")
    parser.add_argument(
        "--use-pretrained-weights",
        action="store_true",
        help="Load full pretrained weights (slow/heavy); default uses autoconfig-derived dummy weights.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for dummy-config random initialization.")
    parser.add_argument("--mad-threshold", type=float, default=0.08)
    parser.add_argument("--cos-threshold", type=float, default=0.995)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/minimax_m3_parity/minimax_m3_fp32_vs_qeff_fp16.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_hf_cache()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    base_cfg = _as_local_minimax_config(AutoConfig.from_pretrained(args.model_id, trust_remote_code=True))
    full_layers = int(base_cfg.text_config.num_hidden_layers)
    mode = "pretrained checkpoint" if args.use_pretrained_weights else "autoconfig-derived dummy"

    print(f"[parity] mode: {mode}")
    print(f"[parity] running {args.layers}-layer gate")
    eight_layer_result = run_parity(
        model_id=args.model_id,
        base_cfg=base_cfg,
        tokenizer=tokenizer,
        prompt=args.prompt,
        layers=args.layers,
        mad_threshold=args.mad_threshold,
        cos_threshold=args.cos_threshold,
        seed=args.seed,
        use_pretrained_weights=args.use_pretrained_weights,
    )
    print(
        f"[parity] {args.layers}-layer: "
        f"ok={eight_layer_result.run_ok} gate={eight_layer_result.gate_pass} "
        f"mad={_fmt(eight_layer_result.mad)} cos={_fmt(eight_layer_result.cos_sim)} "
        f"top1_match={eight_layer_result.top1_match}"
    )

    full_result = None
    if not args.skip_full and eight_layer_result.run_ok and eight_layer_result.gate_pass:
        print(f"[parity] running full-model gate ({full_layers} layers)")
        full_result = run_parity(
            model_id=args.model_id,
            base_cfg=base_cfg,
            tokenizer=tokenizer,
            prompt=args.prompt,
            layers=full_layers,
            mad_threshold=args.mad_threshold,
            cos_threshold=args.cos_threshold,
            seed=args.seed,
            use_pretrained_weights=args.use_pretrained_weights,
        )
        print(
            "[parity] full-model: "
            f"ok={full_result.run_ok} gate={full_result.gate_pass} "
            f"mad={_fmt(full_result.mad)} cos={_fmt(full_result.cos_sim)} "
            f"top1_match={full_result.top1_match}"
        )
    elif args.skip_full:
        print("[parity] full-model run skipped by --skip-full")
    else:
        print(f"[parity] full-model run skipped because {args.layers}-layer gate did not pass")

    write_report(
        report_path=args.report_path,
        model_id=args.model_id,
        prompt=args.prompt,
        initial_layers=args.layers,
        use_pretrained_weights=args.use_pretrained_weights,
        seed=args.seed,
        eight_layer_result=eight_layer_result,
        full_layer_result=full_result,
        mad_threshold=args.mad_threshold,
        cos_threshold=args.cos_threshold,
    )
    print(f"[parity] report written: {args.report_path}")


if __name__ == "__main__":
    main()
