# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Shared argparse building blocks for QEfficient example scripts.

Every example script (``basic_inference.py`` today; VLM/embeddings/audio
tomorrow) builds its parser from the same group builders below. Each builder
returns a parent parser that ``argparse.ArgumentParser(parents=[...])`` can
compose, so the surface stays consistent without hiding the underlying
``QEFFAutoModel*.from_pretrained / .compile / .generate`` calls.

Advanced flags are hidden from ``--help`` by default and revealed by
``--help-advanced``; :func:`_advanced_active` scans ``sys.argv`` at
parser-build time to toggle their help text between the real string and
``argparse.SUPPRESS``. The value itself is always accepted on the
command line either way.

The collectors at the bottom (``build_qaic_config``, ``compiler_options``,
``resolve_prompts``) translate the parsed ``argparse.Namespace`` into the
kwargs those three methods actually accept.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---- CLI helpers -----------------------------------------------------------


def _device_group(spec: str) -> List[int]:
    """Parse ``[0,1,2,3]`` or ``0,1,2,3`` into a list of ints.

    Empty / degenerate inputs (``''``, ``'[]'``, ``',,'``) raise an
    ``ArgumentTypeError`` rather than silently returning ``[]`` and
    letting a ``num_devices=0`` propagate into ``.compile()``.
    """
    ids = [int(x) for x in spec.strip("[]").split(",") if x.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("--device-group must contain at least one integer (e.g. [0,1,2,3] or 0,1).")
    return ids


def _int_or_list(spec: str) -> Any:
    """Parse either ``512`` (int) or ``[512,1024]`` (list) for CCL-style flags.

    vLLM's disaggregated path stringifies these lists, so both forms are
    accepted and returned unchanged for downstream ``compile()``.
    """
    spec = spec.strip()
    if spec.startswith("["):
        return ast.literal_eval(spec)
    return int(spec)


def _advanced_active() -> bool:
    """Whether ``--help-advanced`` (or ``--help-advanced=VALUE``) is on argv.

    Evaluated at parser-build time — not import time — so a downstream
    caller that reconstructs a parser after munging ``sys.argv`` sees
    the current visibility state. The scan stops at the first bare
    ``--`` sentinel so post-``--`` positional payloads (e.g. a prompt
    that literally spells ``--help-advanced``) cannot flip visibility.
    """
    for tok in sys.argv[1:]:
        if tok == "--":
            break
        if tok == "--help-advanced" or tok.startswith("--help-advanced="):
            return True
    return False


def _adv(help_text: str) -> str:
    """Return ``help_text`` when ``--help-advanced`` was passed, else SUPPRESS."""
    return help_text if _advanced_active() else argparse.SUPPRESS


class _AdvancedHelpAction(argparse.Action):
    """Print the full help (with advanced flags visible) then exit."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.print_help()
        parser.exit()


# ---- Group builders --------------------------------------------------------


def _parent() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(add_help=False)


def model_group() -> argparse.ArgumentParser:
    """Model loading: what ``QEFFAutoModelForCausalLM.from_pretrained`` needs."""
    p = _parent()
    g = p.add_argument_group("model")
    g.add_argument(
        "--model-name",
        "--model-id",
        dest="model_name",
        default="Qwen/Qwen2-1.5B-Instruct",
        help="HuggingFace model ID or local path.",
    )
    g.add_argument(
        "--tokenizer-name",
        "--tokenizer-id",
        dest="tokenizer_name",
        default=None,
        help="Tokenizer ID if it differs from --model-name.",
    )
    g.add_argument("--gguf-file", default=None, help="GGUF file name inside the HF repo (for quantized GGUF models).")
    g.add_argument(
        "--continuous-batching",
        action="store_true",
        help="Load for continuous batching. Requires --full-batch-size at compile time.",
    )
    g.add_argument(
        "--max-seq-len-cached",
        type=int,
        default=None,
        help=_adv(
            "Cap the model's max_seq_len_cached at load time (needed for very large models "
            "whose upstream default would exceed device memory)."
        ),
    )
    return p


def compile_group() -> argparse.ArgumentParser:
    """Everything that flows into ``.compile()`` — basic and advanced."""
    p = _parent()
    basic = p.add_argument_group("compile-basic")
    basic.add_argument("--prefill-seq-len", type=int, default=32)
    basic.add_argument("--ctx-len", type=int, default=128)
    basic.add_argument("--batch-size", type=int, default=1)
    basic.add_argument(
        "--full-batch-size",
        type=int,
        default=None,
        help="Continuous-batching batch size. Required when --continuous-batching is set.",
    )
    basic.add_argument("--generation-len", type=int, default=None)
    basic.add_argument("--num-cores", type=int, default=16)
    basic.add_argument(
        "--num-devices", type=int, default=None, help="Number of AI 100 SoCs. Defaults to len(--device-group) or 1."
    )
    basic.add_argument("--device-group", type=_device_group, default=None, help="Device IDs, e.g. [0,1,2,3].")
    basic.add_argument("--mxfp6-matmul", action="store_true", help="Compress matmul weights to MXFP6.")
    basic.add_argument("--mxint8-kv-cache", action="store_true", help="Compress KV cache to MXINT8.")

    adv = p.add_argument_group("compile-advanced")
    adv.add_argument("--kv-cache-batch-size", type=int, default=None, help=_adv("KV cache batch size."))
    adv.add_argument(
        "--use-onnx-subfunctions",
        action="store_true",
        help=_adv("Emit one ONNX subgraph per transformer block; recommended for MoE/large models."),
    )
    adv.add_argument("--aic-enable-depth-first", action="store_true", help=_adv("Compiler DFS scheduling."))
    adv.add_argument(
        "--allow-mxint8-mdp-io", action="store_true", help=_adv("Allow MXINT8 compression on MDP IO boundaries.")
    )
    adv.add_argument("--mos", type=int, default=-1, help=_adv("On-chip memory optimization effort."))
    adv.add_argument("--user-tiled", action="store_true", help=_adv("Enable user-specified tiling."))
    adv.add_argument("--node-precision-info", default=None, help=_adv("YAML/JSON with per-node precision overrides."))
    adv.add_argument("--enable-qnn", action="store_true", help=_adv("Use QNN compiler backend."))
    adv.add_argument("--qnn-config", default=None, help=_adv("Path to QNN config file."))
    adv.add_argument("--compile-dir", default=None, help=_adv("Where to place the QPC package."))
    adv.add_argument("--onnx-path", default=None, help=_adv("Use a pre-exported ONNX; skip export."))
    adv.add_argument("--kv-cache-prefix", default=None, help=_adv("Prefix for exported KV cache tensor names."))
    adv.add_argument(
        "--offload-pt-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=_adv("Offload PyTorch weights during export."),
    )
    adv.add_argument(
        "--layerwise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=_adv("Export/compile one transformer layer window at a time (large-model layerwise driver)."),
    )
    adv.add_argument(
        "--layerwise-window-size",
        type=int,
        default=1,
        help=_adv("Number of layers per layerwise window when --layerwise is set."),
    )
    return p


def ccl_group() -> argparse.ArgumentParser:
    """Compute-Context-Length (chunked context) lists for prefill and decode."""
    p = _parent()
    g = p.add_argument_group("ccl (chunked context lengths)")
    g.add_argument(
        "--ccl-prefill",
        dest="comp_ctx_lengths_prefill",
        nargs="+",
        type=int,
        default=None,
        help="CCL list for prefill, e.g. --ccl-prefill 512 1024 2048.",
    )
    g.add_argument(
        "--ccl-decode", dest="comp_ctx_lengths_decode", nargs="+", type=int, default=None, help="CCL list for decode."
    )
    return p


def disagg_group() -> argparse.ArgumentParser:
    """Disaggregated serving: prefill and decode compiled as separate QPCs."""
    p = _parent()
    g = p.add_argument_group("disagg / chunked prefill")
    g.add_argument(
        "--stage",
        choices=["prefill", "decode", "both"],
        default="both",
        help="Compile only prefill, only decode, or both (default).",
    )
    g.add_argument(
        "--enable-chunking",
        action="store_true",
        help="Chunked prefill; pair with --stage prefill for MoE expert-blocked prefill.",
    )
    g.add_argument(
        "--moe-prefill-packed-chunk-size", type=int, default=256, help="Packed rows per expert-blocked MoE chunk."
    )
    g.add_argument("--retain-full-kv", action="store_true", help="Keep full KV in the decode QPC.")
    return p


def blocking_group() -> argparse.ArgumentParser:
    """``qaic_config`` fields for attention blocking (long-context feature)."""
    p = _parent()
    g = p.add_argument_group("blocking (attention tiling for long contexts)")
    g.add_argument("--enable-blocking", action="store_true")
    g.add_argument(
        "--blocking-mode",
        choices=["kv", "q", "h", "b", "qkv", "hqkv"],
        default="kv",
        help="Axes to tile along; combine as substring (k=key, v=value, q=query, h=head, b=batch).",
    )
    g.add_argument("--num-kv-blocks", type=int, default=None)
    g.add_argument("--num-q-blocks", type=int, default=None)
    g.add_argument("--num-batch-blocks", type=int, default=None)
    g.add_argument("--head-block-size", type=int, default=None)
    g.add_argument(
        "--num-kv-heads-repeat", type=int, default=None, help="KV-head replication factor (DeepSeek MLA workloads)."
    )
    g.add_argument("--skip-kv", action="store_true")
    return p


def speculative_group() -> argparse.ArgumentParser:
    """Speculative decoding (target-language-model side)."""
    p = _parent()
    g = p.add_argument_group("speculative decoding")
    g.add_argument(
        "--speculative-model-type", default=None, help="Set to 'target' to compile a TLM (routed through qaic_config)."
    )
    g.add_argument(
        "--num-speculative-tokens",
        nargs="+",
        type=int,
        default=None,
        help="Proposal length(s), e.g. --num-speculative-tokens 3 or --num-speculative-tokens 0 3.",
    )
    return p


def sampler_group() -> argparse.ArgumentParser:
    """On-device sampler + guided decoding."""
    p = _parent()
    g = p.add_argument_group("sampler")
    g.add_argument(
        "--include-sampler",
        action="store_true",
        help="Enable on-device sampler (see --help-advanced for tuning knobs).",
    )
    g.add_argument("--return-pdfs", action="store_true", help=_adv("Return probability distributions."))
    g.add_argument("--max-top-k-ids", type=int, default=512, help=_adv("Cap on top-k IDs sampled."))
    g.add_argument("--include-guided-decoding", action="store_true", help=_adv("Enable guided decoding."))
    return p


def runtime_group() -> argparse.ArgumentParser:
    """Runtime knobs consumed by ``.generate()``."""
    p = _parent()
    g = p.add_argument_group("runtime")
    g.add_argument("--prompt", nargs="+", default=None, help="One or more prompts (quote each if it contains spaces).")
    g.add_argument(
        "--prompts", default=None, help="Pipe-separated prompts (compat with older scripts). --prompt takes precedence."
    )
    g.add_argument("--prompts-file", default=None, help="Text file, one prompt per line.")
    g.add_argument("--iteration", type=int, default=1, help="Number of runtime iterations.")
    g.add_argument("--write-io", action="store_true", help="Persist IO tensors for debugging.")
    g.add_argument("--automation", action="store_true")
    return p


def meta_group() -> argparse.ArgumentParser:
    """Help/dry-run/CI toggles that don't affect compile or generate."""
    p = _parent()
    g = p.add_argument_group("meta")
    g.add_argument(
        "--help-advanced",
        action=_AdvancedHelpAction,
        help="Print the full flag list, including advanced/CI-only options.",
    )
    g.add_argument(
        "--print-resolved", action="store_true", help="Print the resolved argparse namespace before running."
    )
    g.add_argument("--dry-run", action="store_true", help="Parse and print but don't compile or generate.")
    g.add_argument(
        "--num-hidden-layers-override",
        type=int,
        default=None,
        help=_adv("CI/testing only: truncate the HF config's num_hidden_layers before load."),
    )
    return p


# ---- Post-parse resolution -------------------------------------------------


DEFAULT_PROMPT = "My name is"


def resolve_prompts(ns: argparse.Namespace) -> List[str]:
    """Merge the three prompt-input surfaces into one list.

    Precedence: ``--prompt`` (nargs=+) > ``--prompts-file`` > ``--prompts`` (pipe).
    Falls back to a single default prompt when nothing is provided.
    """
    if getattr(ns, "prompt", None):
        return list(ns.prompt)
    if getattr(ns, "prompts_file", None):
        return [line.strip() for line in Path(ns.prompts_file).read_text().splitlines() if line.strip()]
    if getattr(ns, "prompts", None):
        return [p for p in ns.prompts.split("|") if p]
    return [DEFAULT_PROMPT]


def resolve_num_devices(ns: argparse.Namespace) -> int:
    """``--num-devices`` wins; else len(--device-group); else 1.

    Use :func:`validate_args` right after ``parse_args`` to reject
    mismatches between the two before this ever runs.
    """
    if ns.num_devices is not None:
        return ns.num_devices
    if ns.device_group is not None:
        return len(ns.device_group)
    return 1


def validate_args(ns: argparse.Namespace, error_fn) -> None:
    """CLI-level cross-flag checks that argparse can't express on its own.

    ``error_fn`` should be ``parser.error`` (prints usage + message,
    exits 2). Kept as a callback so the args module stays parser-free
    and reusable across example scripts.
    """
    if ns.num_devices is not None and ns.device_group is not None:
        if ns.num_devices != len(ns.device_group):
            error_fn(f"--num-devices ({ns.num_devices}) does not match len(--device-group) ({len(ns.device_group)}).")
    if ns.continuous_batching and ns.full_batch_size is None:
        error_fn("--continuous-batching requires --full-batch-size.")
    if getattr(ns, "layerwise", False) and ns.layerwise_window_size < 1:
        error_fn("--layerwise-window-size must be >= 1.")


def resolve_prefill_only(ns: argparse.Namespace) -> Optional[bool]:
    """Map ``--stage`` to the tri-state ``prefill_only`` that ``.compile()`` expects."""
    return {"prefill": True, "decode": False, "both": None}[ns.stage]


def build_qaic_config(ns: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Collect the ``qaic_config`` dict for ``from_pretrained``.

    Returns ``None`` when no qaic_config knob is set — some transforms
    behave differently on ``None`` vs an empty dict.
    """
    cfg: Dict[str, Any] = {}

    # KV-head replication (DeepSeek-V3 / Kimi MLA) is driven by
    # ReplicateKVHeadTransform, which fires whenever
    # qaic_config["num_kv_heads_repeat"] > 1 regardless of blocking.
    # Emit it at the top level so --num-kv-heads-repeat is not silently
    # inert unless --enable-blocking is also set.
    if ns.num_kv_heads_repeat is not None:
        cfg["num_kv_heads_repeat"] = ns.num_kv_heads_repeat

    if ns.enable_blocking:
        cfg["enable_blocking"] = True
        cfg["blocking_mode"] = ns.blocking_mode
        for key, val in (
            ("num_kv_blocks", ns.num_kv_blocks),
            ("num_q_blocks", ns.num_q_blocks),
            ("num_batch_blocks", ns.num_batch_blocks),
            ("head_block_size", ns.head_block_size),
        ):
            if val is not None:
                cfg[key] = val
        if ns.skip_kv:
            cfg["skip_kv"] = True

    if ns.include_sampler:
        cfg["include_sampler"] = True
        cfg["max_top_k_ids"] = ns.max_top_k_ids
        if ns.return_pdfs:
            cfg["return_pdfs"] = True
        if ns.include_guided_decoding:
            cfg["include_guided_decoding"] = True

    if ns.speculative_model_type:
        cfg["speculative_model_type"] = ns.speculative_model_type

    return cfg or None


def compiler_options(ns: argparse.Namespace) -> Dict[str, Any]:
    """Kwargs that flow through ``.compile(**compiler_options)`` to qaic-compile."""
    opts: Dict[str, Any] = {}
    if ns.aic_enable_depth_first:
        opts["aic_enable_depth_first"] = True
    if ns.allow_mxint8_mdp_io:
        opts["allow_mxint8_mdp_io"] = True
    if ns.mos != -1:
        opts["mos"] = ns.mos
    if ns.user_tiled:
        opts["user_tiled"] = True
    if ns.node_precision_info:
        opts["node_precision_info"] = ns.node_precision_info
    if ns.enable_qnn:
        opts["enable_qnn"] = True
    if ns.qnn_config:
        opts["qnn_config"] = ns.qnn_config
    return opts


def num_speculative_tokens(ns: argparse.Namespace) -> Any:
    """Return the value ``.compile()`` expects (int, list, or None)."""
    val = getattr(ns, "num_speculative_tokens", None)
    if val is None:
        return None
    return val[0] if len(val) == 1 else val


def print_namespace(ns: argparse.Namespace) -> None:
    payload = {k: v for k, v in vars(ns).items() if not k.startswith("_")}
    print(json.dumps(payload, indent=2, default=str))


def apply_num_layers_override(from_pretrained_kwargs: Dict[str, Any], ns: argparse.Namespace) -> None:
    """CI-only: forward ``--num-hidden-layers-override`` into ``from_pretrained``."""
    if getattr(ns, "num_hidden_layers_override", None) is not None:
        from_pretrained_kwargs["num_hidden_layers"] = ns.num_hidden_layers_override


__all__: Sequence[str] = (
    "apply_num_layers_override",
    "blocking_group",
    "build_qaic_config",
    "ccl_group",
    "compile_group",
    "compiler_options",
    "disagg_group",
    "meta_group",
    "model_group",
    "num_speculative_tokens",
    "print_namespace",
    "resolve_num_devices",
    "resolve_prefill_only",
    "resolve_prompts",
    "runtime_group",
    "sampler_group",
    "speculative_group",
    "validate_args",
)
