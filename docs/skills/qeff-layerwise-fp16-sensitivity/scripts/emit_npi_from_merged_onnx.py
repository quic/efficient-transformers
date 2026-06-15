#!/usr/bin/env python3
"""Emit/validate NPI YAML from ONNX by resolving first outputs of selected nodes.

Supports:
1) templated layer+ops expansion (layerwise merged ONNX)
2) explicit node name list (works for diffusion/any graph)
3) validation of existing NPI YAML against ONNX output tensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import yaml


def _csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _read_node_names_file(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"node-names-file not found: {p}")
    names: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        names.append(s)
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate/validate FP32NodeInstanceNames from ONNX outputs.")
    p.add_argument("--onnx", default=None, help="Path to ONNX graph (merged or regular).")
    p.add_argument("--merged-onnx", default=None, help="Backward-compatible alias for --onnx.")

    p.add_argument("--layers", default=None, help="Comma-separated layer ids (templated mode).")
    p.add_argument("--ops", default=None, help="Comma-separated op suffixes under mlp/ (templated mode).")
    p.add_argument(
        "--node-name-template",
        default="layer_{layer}//language_model/layers.{layer}/mlp/{op}",
        help="Template used with --layers/--ops.",
    )

    p.add_argument("--node-names", default=None, help="Comma-separated exact node names.")
    p.add_argument("--node-names-file", default=None, help="Text file with one exact node name per line.")

    p.add_argument("--output-yaml", default=None, help="Output YAML path for FP32NodeInstanceNames.")
    p.add_argument("--validate-yaml", default=None, help="Existing NPI YAML to validate against ONNX outputs.")
    p.add_argument("--validate-only", action="store_true", help="Do not write new YAML.")
    return p


def resolve_onnx_path(args: argparse.Namespace) -> Path:
    onnx_path = args.onnx or args.merged_onnx
    if not onnx_path:
        raise ValueError("Provide --onnx (or --merged-onnx).")
    p = Path(onnx_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"ONNX not found: {p}")
    return p


def validate_yaml_against_outputs(validate_yaml: Path, output_names: set[str]) -> list[str]:
    payload = yaml.safe_load(validate_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "FP32NodeInstanceNames" not in payload:
        raise ValueError(f"Invalid NPI YAML structure: {validate_yaml}")
    names = payload["FP32NodeInstanceNames"]
    if not isinstance(names, list):
        raise ValueError(f"FP32NodeInstanceNames must be a list: {validate_yaml}")
    missing = [n for n in names if n not in output_names]
    print(f"[validate_yaml] {validate_yaml}")
    print(f"[entries] {len(names)}")
    print(f"[missing] {len(missing)}")
    for m in missing:
        print(f"MISSING {m}")
    return missing


def expand_node_names(args: argparse.Namespace) -> list[str]:
    explicit = _csv_list(args.node_names)
    explicit.extend(_read_node_names_file(args.node_names_file))
    if explicit:
        return list(dict.fromkeys(explicit))

    layers_raw = _csv_list(args.layers)
    ops = _csv_list(args.ops)
    if not layers_raw or not ops:
        raise ValueError(
            "Provide either explicit node names (--node-names/--node-names-file) "
            "or templated inputs (--layers and --ops)."
        )
    layers = [int(x) for x in layers_raw]
    out: list[str] = []
    for layer in layers:
        for op in ops:
            out.append(args.node_name_template.format(layer=layer, op=op))
    return out


def main() -> None:
    args = build_parser().parse_args()

    onnx_path = resolve_onnx_path(args)
    model = onnx.load(str(onnx_path), load_external_data=False)

    node_by_name = {n.name: n for n in model.graph.node}
    output_names = {o for n in model.graph.node for o in n.output if o}

    print(f"[onnx] {onnx_path}")
    print(f"[node_outputs] {len(output_names)}")

    if args.validate_yaml:
        validate_path = Path(args.validate_yaml).expanduser().resolve()
        if not validate_path.is_file():
            raise FileNotFoundError(f"validate-yaml not found: {validate_path}")
        missing = validate_yaml_against_outputs(validate_path, output_names)
        if missing:
            raise RuntimeError(f"Validation failed: {len(missing)} NPI entries missing in ONNX outputs.")

    target_node_names = expand_node_names(args)
    missing_nodes: list[str] = []
    resolved_outputs: list[str] = []

    for node_name in target_node_names:
        node = node_by_name.get(node_name)
        if node is None:
            missing_nodes.append(node_name)
            continue
        if not node.output or not node.output[0]:
            raise RuntimeError(f"Node has no first output: {node_name}")
        resolved_outputs.append(node.output[0])

    if missing_nodes:
        print(f"[missing_nodes] {len(missing_nodes)}")
        for n in missing_nodes:
            print(f"MISSING_NODE {n}")
        raise RuntimeError("Some node names were not found in ONNX.")

    resolved_outputs = list(dict.fromkeys(resolved_outputs))
    print(f"[resolved_outputs] {len(resolved_outputs)}")
    for name in resolved_outputs:
        print(name)

    if args.validate_only:
        return

    if not args.output_yaml:
        raise ValueError("--output-yaml is required unless --validate-only is set.")

    out = Path(args.output_yaml).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump({"FP32NodeInstanceNames": resolved_outputs}, f, sort_keys=False)
    print(f"[wrote_yaml] {out}")


if __name__ == "__main__":
    main()
