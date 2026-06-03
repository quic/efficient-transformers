"""ONNX graph analysis helpers for auto-perf skill.

This script supports two entry points:
1) Parse the ONNX path from `run_inference.py` output logs.
2) Load the ONNX graph, infer shapes when possible, and summarize hotspots
   that should guide PyTorch-side optimization hypotheses.

For any invocation that resolves an ONNX path, the script deletes that ONNX
file's parent directory after analysis/cleanup processing completes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import onnx
    from onnx import shape_inference
except ImportError as exc:  # pragma: no cover - runtime environment dependent
    raise SystemExit(
        "Missing dependency: `onnx`. Install it in the active environment to use ONNX graph analysis."
    ) from exc


Dim = Union[int, str]

EXPORT_PATH_PATTERN = re.compile(
    r"model exported to ONNX format at:\s*(?P<path>.+?)\s*$", re.IGNORECASE
)

COMPUTE_HEAVY_OPS = {"MatMul", "Gemm", "QLinearMatMul", "MatMulInteger", "Attention"}
LAYOUT_OPS = {"Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Concat", "Slice"}
ELEMENTWISE_FRAGMENT_OPS = {"Add", "Mul", "Div", "Sub", "Pow", "Softmax", "Gelu", "Relu"}


@dataclass
class NodeHotspot:
    name: str
    op_type: str
    output_name: str
    output_shape: Optional[List[Dim]]
    output_volume: Optional[int]
    reason: str


@dataclass
class DagNode:
    id: str
    name: str
    op_type: str
    topological_index: int
    scope: str
    inputs: List[str]
    outputs: List[str]
    upstream_node_ids: List[str]
    downstream_node_ids: List[str]
    output_shapes: Dict[str, List[Dim]]
    output_volumes: Dict[str, int]
    attributes: Dict[str, Any]


@dataclass
class DagEdge:
    source_node_id: str
    target_node_id: str
    tensor_name: str
    tensor_shape: Optional[List[Dim]]
    tensor_volume: Optional[int]


@dataclass
class DagBoundaryTensor:
    name: str
    kind: str
    shape: Optional[List[Dim]]
    consumers: List[str]
    producer: Optional[str] = None


@dataclass
class DagSummary:
    node_count: int
    edge_count: int
    topological_node_ids: List[str]
    graph_inputs: List[DagBoundaryTensor]
    graph_outputs: List[DagBoundaryTensor]
    initializers: List[DagBoundaryTensor]
    nodes: List[DagNode]
    edges: List[DagEdge]


@dataclass
class GraphSummary:
    onnx_path: str
    node_count: int
    initializer_count: int
    input_count: int
    output_count: int
    op_type_histogram: Dict[str, int]
    top_hotspots: List[NodeHotspot]
    optimization_hypotheses: List[str]
    dag: DagSummary


def extract_onnx_path_from_text(text: str) -> Path:
    """Extract ONNX path from run_inference output text."""
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        match = EXPORT_PATH_PATTERN.search(line)
        if match:
            raw_path = match.group("path").strip().strip("'\"")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            return path
    raise ValueError(
        "Unable to find ONNX path in run output. Expected line: "
        "'model exported to ONNX format at: <path-to-model.onnx>'."
    )


def _dim_to_value(dim_proto) -> Dim:
    if dim_proto.HasField("dim_value"):
        return int(dim_proto.dim_value)
    if dim_proto.HasField("dim_param") and dim_proto.dim_param:
        return str(dim_proto.dim_param)
    return "?"


def _shape_from_value_info(value_info) -> Optional[List[Dim]]:
    value_type = value_info.type
    if not value_type.HasField("tensor_type"):
        return None
    tensor_type = value_type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    return [_dim_to_value(dim) for dim in tensor_type.shape.dim]


def _collect_value_shapes(graph) -> Dict[str, List[Dim]]:
    shapes: Dict[str, List[Dim]] = {}
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        shape = _shape_from_value_info(value_info)
        if shape is not None:
            shapes[value_info.name] = shape
    for initializer in graph.initializer:
        if initializer.name not in shapes:
            shapes[initializer.name] = [int(dim) for dim in initializer.dims]
    return shapes


def _attribute_to_jsonable(attribute: onnx.AttributeProto) -> Any:
    attr_type = attribute.type
    if attr_type == onnx.AttributeProto.FLOAT:
        return attribute.f
    if attr_type == onnx.AttributeProto.INT:
        return attribute.i
    if attr_type == onnx.AttributeProto.STRING:
        return attribute.s.decode("utf-8", errors="replace")
    if attr_type == onnx.AttributeProto.FLOATS:
        return list(attribute.floats)
    if attr_type == onnx.AttributeProto.INTS:
        return list(attribute.ints)
    if attr_type == onnx.AttributeProto.STRINGS:
        return [item.decode("utf-8", errors="replace") for item in attribute.strings]
    if attr_type == onnx.AttributeProto.TENSOR:
        dims = list(attribute.t.dims)
        return {"tensor_data_type": int(attribute.t.data_type), "tensor_dims": dims}
    if attr_type == onnx.AttributeProto.TENSORS:
        return [
            {"tensor_data_type": int(tensor.data_type), "tensor_dims": list(tensor.dims)}
            for tensor in attribute.tensors
        ]
    if attr_type == onnx.AttributeProto.GRAPH:
        return {
            "graph_name": attribute.g.name,
            "graph_node_count": len(attribute.g.node),
        }
    if attr_type == onnx.AttributeProto.GRAPHS:
        return [
            {"graph_name": graph.name, "graph_node_count": len(graph.node)}
            for graph in attribute.graphs
        ]
    return str(attribute)


def _static_volume(shape: Optional[List[Dim]]) -> Optional[int]:
    if not shape:
        return None
    vol = 1
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            return None
        vol *= dim
    return vol


def _node_reason(op_type: str, volume: Optional[int]) -> str:
    if op_type in COMPUTE_HEAVY_OPS and volume is not None:
        return f"compute-heavy op with large static output volume={volume}"
    if op_type in COMPUTE_HEAVY_OPS:
        return "compute-heavy op"
    if op_type in LAYOUT_OPS:
        return "layout-conversion op; candidate for transpose/reshape chain cleanup"
    if op_type in ELEMENTWISE_FRAGMENT_OPS:
        return "frequent cheap op; candidate for kernel-fragmentation reduction/fusion"
    return "high-priority op by graph position/frequency"


def _score_node(op_type: str, volume: Optional[int]) -> Tuple[int, int]:
    if op_type in COMPUTE_HEAVY_OPS:
        op_rank = 3
    elif op_type in LAYOUT_OPS:
        op_rank = 2
    elif op_type in ELEMENTWISE_FRAGMENT_OPS:
        op_rank = 1
    else:
        op_rank = 0

    vol_rank = volume if volume is not None else -1
    return (op_rank, vol_rank)


def _derive_hypotheses(op_hist: Dict[str, int], hotspots: List[NodeHotspot]) -> List[str]:
    hypotheses: List[str] = []

    heavy_count = sum(op_hist.get(op, 0) for op in COMPUTE_HEAVY_OPS)
    layout_count = sum(op_hist.get(op, 0) for op in LAYOUT_OPS)
    elementwise_count = sum(op_hist.get(op, 0) for op in ELEMENTWISE_FRAGMENT_OPS)

    if heavy_count > 0:
        hypotheses.append(
            "Prioritize MatMul/Gemm/attention hotspots: remove pre/post layout churn and fuse cheap epilogues around these nodes."
        )
    if layout_count > 0:
        hypotheses.append(
            "Collapse repeated Transpose/Reshape/Concat/Slice chains around hotspot regions to reduce memory traffic and improve compiler fusion."
        )
    if elementwise_count > 0:
        hypotheses.append(
            "Reduce tiny-kernel fragmentation by combining adjacent elementwise ops in decode/prefill paths when numerically safe."
        )

    for hotspot in hotspots:
        if hotspot.op_type == "Attention":
            hypotheses.append(
                "For attention hotspots, evaluate Q/KV/H blocking with static loop bounds and streaming softmax accumulation."
            )
            break

    # Preserve order while removing duplicates.
    unique_hypotheses = list(dict.fromkeys(hypotheses))
    return unique_hypotheses[:5]


def _node_scope(node_name: str) -> str:
    if not node_name or "/" not in node_name:
        return ""
    scope, _, _ = node_name.rpartition("/")
    return scope


def _build_dag_summary(graph, value_shapes: Dict[str, List[Dim]]) -> DagSummary:
    producer_for_tensor: Dict[str, str] = {}
    consumers_for_tensor: Dict[str, List[str]] = {}
    node_ids: List[str] = []
    nodes_by_id: Dict[str, DagNode] = {}

    for index, node in enumerate(graph.node):
        node_name = node.name if node.name else f"{node.op_type}_{index}"
        node_id = f"n{index}"
        node_ids.append(node_id)

        for tensor_name in node.input:
            if tensor_name:
                consumers_for_tensor.setdefault(tensor_name, []).append(node_id)
        for tensor_name in node.output:
            if tensor_name:
                producer_for_tensor[tensor_name] = node_id

        attributes = {attribute.name: _attribute_to_jsonable(attribute) for attribute in node.attribute}
        output_shapes = {
            tensor_name: value_shapes[tensor_name]
            for tensor_name in node.output
            if tensor_name and tensor_name in value_shapes
        }
        output_volumes = {
            tensor_name: volume
            for tensor_name in node.output
            if tensor_name and (volume := _static_volume(value_shapes.get(tensor_name))) is not None
        }

        nodes_by_id[node_id] = DagNode(
            id=node_id,
            name=node_name,
            op_type=node.op_type,
            topological_index=index,
            scope=_node_scope(node_name),
            inputs=[tensor_name for tensor_name in node.input if tensor_name],
            outputs=[tensor_name for tensor_name in node.output if tensor_name],
            upstream_node_ids=[],
            downstream_node_ids=[],
            output_shapes=output_shapes,
            output_volumes=output_volumes,
            attributes=attributes,
        )

    edges: List[DagEdge] = []
    edge_keys = set()

    for tensor_name, target_node_ids in consumers_for_tensor.items():
        source_node_id = producer_for_tensor.get(tensor_name)
        if source_node_id is None:
            continue

        for target_node_id in target_node_ids:
            edge_key = (source_node_id, target_node_id, tensor_name)
            if edge_key in edge_keys:
                continue
            edge_keys.add(edge_key)

            source_node = nodes_by_id[source_node_id]
            target_node = nodes_by_id[target_node_id]
            if target_node_id not in source_node.downstream_node_ids:
                source_node.downstream_node_ids.append(target_node_id)
            if source_node_id not in target_node.upstream_node_ids:
                target_node.upstream_node_ids.append(source_node_id)

            tensor_shape = value_shapes.get(tensor_name)
            edges.append(
                DagEdge(
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    tensor_name=tensor_name,
                    tensor_shape=tensor_shape,
                    tensor_volume=_static_volume(tensor_shape),
                )
            )

    for node in nodes_by_id.values():
        node.upstream_node_ids.sort(key=lambda node_id: nodes_by_id[node_id].topological_index)
        node.downstream_node_ids.sort(key=lambda node_id: nodes_by_id[node_id].topological_index)

    initializer_names = {initializer.name for initializer in graph.initializer}
    graph_input_names = [value_info.name for value_info in graph.input]
    graph_output_names = [value_info.name for value_info in graph.output]

    graph_inputs = [
        DagBoundaryTensor(
            name=name,
            kind="graph_input",
            shape=value_shapes.get(name),
            consumers=consumers_for_tensor.get(name, []),
        )
        for name in graph_input_names
        if name not in initializer_names
    ]
    graph_outputs = [
        DagBoundaryTensor(
            name=name,
            kind="graph_output",
            shape=value_shapes.get(name),
            consumers=[],
            producer=producer_for_tensor.get(name),
        )
        for name in graph_output_names
    ]
    initializers = [
        DagBoundaryTensor(
            name=initializer.name,
            kind="initializer",
            shape=value_shapes.get(initializer.name),
            consumers=consumers_for_tensor.get(initializer.name, []),
        )
        for initializer in graph.initializer
    ]

    return DagSummary(
        node_count=len(nodes_by_id),
        edge_count=len(edges),
        topological_node_ids=node_ids,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        initializers=initializers,
        nodes=[nodes_by_id[node_id] for node_id in node_ids],
        edges=edges,
    )


def analyze_onnx_graph(onnx_path: Path) -> GraphSummary:
    """Load ONNX, infer shapes when possible, and summarize optimization signals."""
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    model = onnx.load(str(onnx_path), load_external_data=True)
    original_model = model
    try:
        inferred_model = shape_inference.infer_shapes(model)
        # Some models with custom domains/functions can be malformed by shape
        # inference (observed as an empty graph). Keep the original graph then.
        if len(inferred_model.graph.node) > 0 or len(original_model.graph.node) == 0:
            model = inferred_model
        else:
            model = original_model
    except Exception:
        # Shape inference can fail for some exported models; continue with known shapes.
        model = original_model

    graph = model.graph
    value_shapes = _collect_value_shapes(graph)
    dag = _build_dag_summary(graph, value_shapes)

    op_hist: Dict[str, int] = {}
    candidates: List[Tuple[Tuple[int, int], NodeHotspot]] = []

    for index, node in enumerate(graph.node):
        op_hist[node.op_type] = op_hist.get(node.op_type, 0) + 1

        first_output = node.output[0] if node.output else ""
        output_shape = value_shapes.get(first_output)
        output_volume = _static_volume(output_shape)

        node_name = node.name if node.name else f"{node.op_type}_{index}"
        hotspot = NodeHotspot(
            name=node_name,
            op_type=node.op_type,
            output_name=first_output,
            output_shape=output_shape,
            output_volume=output_volume,
            reason=_node_reason(node.op_type, output_volume),
        )
        candidates.append((_score_node(node.op_type, output_volume), hotspot))

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_hotspots = [item[1] for item in candidates[:10]]

    return GraphSummary(
        onnx_path=str(onnx_path),
        node_count=len(graph.node),
        initializer_count=len(graph.initializer),
        input_count=len(graph.input),
        output_count=len(graph.output),
        op_type_histogram=dict(sorted(op_hist.items(), key=lambda item: item[1], reverse=True)),
        top_hotspots=top_hotspots,
        optimization_hypotheses=_derive_hypotheses(op_hist, top_hotspots),
        dag=dag,
    )


def cleanup_onnx_parent_dir(onnx_path: Path) -> Optional[Path]:
    """Delete ONNX parent directory when ONNX exists; return removed dir."""
    path = onnx_path.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return None

    parent = path.parent
    # Guard rails against deleting broad/shared roots by mistake.
    protected_dirs = {
        Path("/"),
        Path("/tmp"),
        Path("/var"),
        Path("/home"),
        Path.home(),
        Path.cwd(),
    }
    if parent in protected_dirs:
        raise ValueError(f"Refusing to delete protected ONNX parent directory: {parent}")

    shutil.rmtree(parent)
    return parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exported ONNX graph for optimization hints.")
    parser.add_argument(
        "--run-output-log",
        type=Path,
        default=None,
        help="Path to run_inference output log. If provided, ONNX path is extracted from this log.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Direct ONNX path. If set, this is used instead of parsing run output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full summary as JSON.",
    )
    parser.add_argument(
        "--cleanup-parent-dir",
        action="store_true",
        help="Deprecated compatibility flag. ONNX parent-directory cleanup now always runs after main().",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Skip graph analysis output and perform cleanup only.",
    )
    return parser.parse_args()


def _resolve_onnx_path(args: argparse.Namespace) -> Path:
    if args.onnx_path is not None:
        path = args.onnx_path.expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    if args.run_output_log is None:
        raise ValueError("Provide either --onnx-path or --run-output-log.")

    log_path = args.run_output_log.expanduser()
    log_path = log_path if log_path.is_absolute() else (Path.cwd() / log_path).resolve()
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return extract_onnx_path_from_text(text)


def _render_text(summary: GraphSummary) -> str:
    lines = []
    lines.append(f"ONNX path: {summary.onnx_path}")
    lines.append(
        "Graph stats: "
        f"nodes={summary.node_count}, "
        f"initializers={summary.initializer_count}, "
        f"inputs={summary.input_count}, "
        f"outputs={summary.output_count}"
    )
    lines.append(
        "DAG stats: "
        f"nodes={summary.dag.node_count}, "
        f"edges={summary.dag.edge_count}, "
        f"graph_inputs={len(summary.dag.graph_inputs)}, "
        f"graph_outputs={len(summary.dag.graph_outputs)}"
    )
    lines.append("Top op counts:")
    for op_type, count in list(summary.op_type_histogram.items())[:10]:
        lines.append(f"  - {op_type}: {count}")
    lines.append("Top hotspots:")
    for hotspot in summary.top_hotspots[:5]:
        lines.append(
            f"  - {hotspot.name} ({hotspot.op_type}) "
            f"shape={hotspot.output_shape} volume={hotspot.output_volume} reason={hotspot.reason}"
        )
    lines.append("Optimization hypotheses:")
    for hypothesis in summary.optimization_hypotheses:
        lines.append(f"  - {hypothesis}")
    lines.append("Topological DAG preview:")
    for node in summary.dag.nodes[:5]:
        lines.append(
            f"  - {node.id} {node.name} ({node.op_type}) "
            f"upstream={len(node.upstream_node_ids)} downstream={len(node.downstream_node_ids)} "
            f"outputs={node.outputs[:2]}"
        )
    return "\n".join(lines)


def _emit_cleanup_result(onnx_path: Path) -> None:
    removed_parent = cleanup_onnx_parent_dir(onnx_path)
    if removed_parent is None:
        print(f"ONNX cleanup skipped: file does not exist at {onnx_path}")
    else:
        print(f"ONNX cleanup complete: removed parent directory {removed_parent}")


def main() -> None:
    args = _parse_args()
    onnx_path = _resolve_onnx_path(args)
    try:
        if not args.cleanup_only:
            summary = analyze_onnx_graph(onnx_path)

            if args.json:
                payload = asdict(summary)
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(_render_text(summary))
    finally:
        had_active_exception = sys.exc_info()[1] is not None
        try:
            _emit_cleanup_result(onnx_path)
        except Exception as exc:
            if had_active_exception:
                print(f"ONNX cleanup failed for {onnx_path}: {exc}", file=sys.stderr)
            else:
                raise


if __name__ == "__main__":
    main()
