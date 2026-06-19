# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""MDP generator for disaggregated prefill serving (PP-enabled, TS-enabled, stages>1)."""

import bisect
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import onnx

logger = logging.getLogger(__name__)


class MdpStrategy(str, Enum):
    """MDP partition-config generation strategy for disaggregated prefill.

    ONNX        : Enumerate every node from the exported ONNX graph and assign
                  to partitions by transformer layer index.  No prior compile
                  run needed; produces a ~19 MB JSON that is a superset of the
                  real Glow IR.  Default.

    INTERSECTION: Requires a prior ``qaic-compile -mdp-dump-partition-config``
                  run.  Generates the full ONNX-based MDP (superset), then
                  filters each partition's nodeList to only the exact Glow IR
                  node names present in the compiler dump.  Result is compact
                  (~1-2 MB) and contains only exact-match names for the
                  compiler's ``doManualPartitioning()``.
    """

    ONNX = "onnx"
    INTERSECTION = "intersection"


def _get_compiler_folded_nodes(graph) -> Set[str]:
    """Return node names the compiler will fold away during ONNX import.

    Mirrors computeIsConstantFoldable() in ONNXModelLoader.cpp: a node is
    foldable if every one of its inputs is a compile-time constant (initializer,
    Constant op output, or output of another foldable node). Folded nodes are
    absent from the compiler IR, so including them in nodeList is harmless but
    excluding them produces a cleaner MDP closer to the compiler dump.

    Op types that the compiler never folds (ProtobufLoader.cpp:68):
        Loop, Const, Identity, If, DequantizeLinear
    """
    # const_values: output tensor names whose value is known at compile time.
    # Seeded with all initializer names (model weights / constants).
    const_values: Set[str] = {init.name for init in graph.initializer}

    # Constant op outputs are trivially compile-time constants; collect them
    # upfront so the fixed-point loop below only needs one pass for everything else.
    for node in graph.node:
        if node.op_type == "Constant":
            const_values.update(out for out in node.output if out)

    # Never-folded op types (compiler explicitly skips these - ProtobufLoader.cpp:68).
    _NEVER_FOLD = frozenset({"Loop", "Const", "Identity", "If", "DequantizeLinear"})

    # Keep marking nodes foldable until no new ones are found.
    foldable_nodes: Set[str] = set()
    while True:
        changed = False
        for node in graph.node:
            if not node.name or node.name in foldable_nodes:
                continue
            if node.op_type in _NEVER_FOLD or not node.input:
                continue
            if all(inp in const_values for inp in node.input if inp):
                foldable_nodes.add(node.name)
                const_values.update(out for out in node.output if out)
                changed = True
        if not changed:
            break

    return foldable_nodes


def _get_layer_num(node_name: str) -> Optional[int]:
    """Return transformer layer index from node name, or None.

    Supports:
      - layers.N  (Llama/Mistral/Qwen/Gemma/Granite)
      - h.N       (GPT-2)
      - layer_N// prefix  (subfunctions/merged ONNX where each subgraph is
                           prefixed with the layer index it belongs to, e.g.
                           "layer_3//model/embed_tokens/Gather")
    """
    for part in node_name.split("/"):
        if part.startswith("layers."):
            suffix = part[len("layers.") :]
            if suffix.isdigit():
                return int(suffix)
        elif part.startswith("h."):
            suffix = part[len("h.") :]
            if suffix.isdigit():
                return int(suffix)
        elif part.startswith("layer_"):
            suffix = part[len("layer_") :]
            if suffix.isdigit():
                return int(suffix)
    return None


def _layer_partition_bounds(num_layers: int, num_partitions: int) -> List[int]:
    """Compute exclusive-upper-bound layer bounds for balanced pipeline partitioning.

    Distributes remainder layers so that the last partition (which carries postprocessing
    such as lm_head and final norm) always receives only the base allocation.  Remainder
    layers are spread to the middle partitions first (indices 1 .. num_partitions-2),
    filling left-to-right starting from the centre of the middle zone.  The first
    partition (which carries embedding preprocessing) absorbs any overflow only when
    the remainder exceeds the number of available middle stages.

    Worst case (remainder == num_partitions - 1): every stage except the last receives
    one extra layer.

    Special cases:
      num_partitions == 1  — single stage; returns an empty list.
      num_partitions == 2  — no middle zone; any remainder goes to the first stage.

    Returns:
        A list of length (num_partitions - 1).  bounds[i] is the exclusive upper
        bound of partition i.  Use ``bisect.bisect_right(bounds, layer_num)`` to
        map a layer index to its partition index.

    Examples (layer counts [partition-0, partition-1, ...]):
        12 layers / 4 stages  ->  [3, 3, 3, 3]   (no remainder)
        9  layers / 4 stages  ->  [2, 3, 2, 2]   (r=1  -> centre-middle gets +1)
        10 layers / 4 stages  ->  [2, 3, 3, 2]   (r=2  -> both middle get +1)
        11 layers / 4 stages  ->  [3, 3, 3, 2]   (r=3 = n-1 -> first+middle get +1)
        8  layers / 3 stages  ->  [3, 3, 2]       (r=2 = n-1 -> first+middle get +1)
        5  layers / 2 stages  ->  [3, 2]           (r=1 -> first gets +1)
    """
    base = num_layers // num_partitions
    remainder = num_layers % num_partitions
    sizes: List[int] = [base] * num_partitions

    if remainder > 0 and num_partitions >= 2:
        if num_partitions == 2:
            sizes[0] += remainder
        else:
            num_middle = num_partitions - 2
            middle_fill = min(remainder, num_middle)
            first_fill = remainder - middle_fill
            mid_center = (1 + num_partitions - 2) // 2
            left, right, filled = mid_center, mid_center + 1, 0
            while filled < middle_fill:
                if left >= 1:
                    sizes[left] += 1
                    left -= 1
                    filled += 1
                if filled < middle_fill and right <= num_partitions - 2:
                    sizes[right] += 1
                    right += 1
                    filled += 1
            if first_fill > 0:
                sizes[0] += 1

    cumsum = 0
    bounds: List[int] = []
    for sz in sizes[:-1]:
        cumsum += sz
        bounds.append(cumsum)
    return bounds


def _get_inlined_node_map(model) -> tuple:
    """Classify ONNX local functions and build inlined sub-node names.

    The compiler inlines a local function body into the parent graph during
    ONNX import if it has < 100 nodes AND is not a known custom op
    (ONNXModelLoaderSubFuns.cpp). Inlined call-sites do not appear in the
    compiler IR; their sub-nodes are named <call_site>/<func_node>.
    Known custom ops (registered via DEFINEKNOWNCUSTOMOP) keep their
    call-site name in the IR and must be included in nodeList as-is.

    Returns:
        inlined_node_map:   dict mapping call-site name -> list of inlined
                            sub-node names (<call_site>/<func_node>).
        non_inlined_funcs:  set of function names that are NOT inlined
                            (known custom ops or >= 100 nodes); their
                            call-site names are valid nodeList entries.
    """
    # Op types registered via DEFINEKNOWNCUSTOMOP in ONNXModelLoaderCustomOp.cpp.
    # The compiler loads these as a single named node (not expanded/inlined).
    _KNOWN_CUSTOM_OPS = frozenset(
        {
            "CustomRMSNorm",
            "CastToUInt4",
        }
    )

    local_functions = {f.name: f for f in model.functions}
    logger.info(f"Found {len(local_functions)} local function types: {set(local_functions.keys())}")

    inlined_funcs: Set[str] = set()
    non_inlined_funcs: Set[str] = set()
    for func_name, func in local_functions.items():
        if func_name in _KNOWN_CUSTOM_OPS or len(func.node) >= 100:
            non_inlined_funcs.add(func_name)
            logger.info(f"  {func_name}: not inlined")
        else:
            inlined_funcs.add(func_name)
            logger.info(f"  {func_name}: {len(func.node)} nodes, will inline")

    inlined_node_map: Dict[str, List[str]] = {}
    for node in model.graph.node:
        if node.op_type in inlined_funcs:
            func = local_functions[node.op_type]
            inlined_node_map[node.name] = [f"{node.name}/{fn.name}" for fn in func.node if fn.name]

    logger.info(f"Inlined sub-nodes mapped for {len(inlined_node_map)} call-sites")
    return inlined_node_map, non_inlined_funcs


def generate_disagg_mdp_partition_config(
    onnx_path: str,
    num_devices: int,
    num_partitions: int,
    num_layers: int,
    num_cores: int = 16,
) -> Dict[str, Any]:
    """Generate a pipeline-partitioned MDP config from an exported ONNX graph.

    Assigns nodes to partitions by transformer layer index using a balanced
    remainder distribution: the last partition (postprocessing — lm_head, final
    norm) always receives only the base layer count; any remainder layers are
    spread to the middle partitions first, then to the first partition (embedding
    preprocessing) only when the remainder exceeds the number of middle stages.
    Non-layer nodes (embeddings, lm_head) follow the nearest layer in topological
    order.  nodeList is a superset of the compiler dump; the compiler silently
    ignores optimized-away names.  Inlined local function call-sites
    (CtxScatterCB, CtxGatherCB) are excluded; their /nNN sub-nodes are assigned
    automatically.  Known custom ops (CustomRMSNorm) are included by call-site name.

    For PP+TS: num_devices // num_partitions devices per partition; the
    compiler applies tensor-slicing within each stage.

    Args:
        onnx_path:      Path to the exported ONNX file.
        num_devices:    Total devices (num_partitions * ts_per_stage).
        num_partitions: Number of pipeline stages.
        num_layers:     Number of transformer layers.
        num_cores:      NSP cores per device (default 16).

    Returns:
        dict with keys 'connections' and 'partitions'.
    """

    if num_partitions <= 0:
        raise ValueError(f"Invalid number of partitions: {num_partitions}")

    if num_partitions > num_devices:
        raise ValueError(
            f"Num of partitions should be <= number of devices. Found {num_partitions} partitions and {num_devices} devices"
        )

    partition_bounds = _layer_partition_bounds(num_layers, num_partitions)
    model = onnx.load(onnx_path, load_external_data=False)

    # Verify topological order (ONNX spec §3.3). Fails loudly on malformed exports.
    # Graph inputs and initializers are excluded — they are not produced by any node.
    graph_input_names: Set[str] = {inp.name for inp in model.graph.input}
    initializer_names: Set[str] = {init.name for init in model.graph.initializer}
    external_names: Set[str] = graph_input_names | initializer_names

    output_to_node: Dict[str, str] = {}
    for node in model.graph.node:
        for out in node.output:
            if out:  # "" marks optional unused outputs
                output_to_node[out] = node.name

    seen_outputs: Set[str] = set()
    for node in model.graph.node:
        for inp in node.input:
            if not inp:
                continue
            if inp in external_names:
                continue
            if inp in output_to_node and inp not in seen_outputs:
                raise ValueError(
                    f"ONNX graph has a cycle or violates topological order: "
                    f"node '{node.name}' consumes '{inp}' produced by "
                    f"'{output_to_node[inp]}', but that producer has not appeared yet."
                )
        for out in node.output:
            if out:
                seen_outputs.add(out)

    logger.info("Computing constant-foldable nodes...")
    folded_nodes = _get_compiler_folded_nodes(model.graph)
    logger.info(f"Found {len(folded_nodes)} compiler-folded nodes (excluded from nodeList)")

    _, non_inlined_functions = _get_inlined_node_map(model)
    inlined_functions = {f.name for f in model.functions} - non_inlined_functions
    local_functions = {f.name: f for f in model.functions}

    # Single pass: assign main graph nodes to partitions by layer index.
    # Inlined function call-sites are expanded inline (sub-nodes inserted at their
    # topological position) so the nodeList ordering matches the ONNX topsort.
    # The compiler's SplitPlanMerge reads nodeList as an ordered sequence; appending
    # inlined sub-nodes out-of-order (in a second pass) causes unresolvable tensor
    # splits in the split-plan merge loop.
    partitions: List[List[str]] = [[] for _ in range(num_partitions)]
    current_layer_partition = 0
    seen_first_layer = False
    max_layer_seen = -1

    for node in model.graph.node:
        if not node.name:
            continue
        if node.name in folded_nodes:
            continue

        layer_num = _get_layer_num(node.name)
        if layer_num is not None:
            max_layer_seen = max(max_layer_seen, layer_num)
            seen_first_layer = True
            partition_idx = bisect.bisect_right(partition_bounds, layer_num)
            current_layer_partition = partition_idx
        else:
            partition_idx = 0 if not seen_first_layer else current_layer_partition

        if node.op_type in inlined_functions:
            # Expand the call-site inline: emit sub-nodes at this topological position.
            func = local_functions[node.op_type]
            sub_nodes = [f"{node.name}/{fn.name}" for fn in func.node if fn.name]
            partitions[partition_idx].extend(sub_nodes)
        else:
            partitions[partition_idx].append(node.name)

    for i, partition in enumerate(partitions):
        logger.info(f"Partition {i}: {len(partition)} nodes")
    logger.info(f"Total nodes in MDP: {sum(len(p) for p in partitions)}")

    # PP-only: 1 device/partition; PP+TS: num_devices//num_partitions devices/partition.
    device_ids = list(range(num_devices))
    devices_per_partition = num_devices // num_partitions
    partition_objs = []
    for i, node_list in enumerate(partitions):
        assigned_devices = device_ids[i * devices_per_partition : (i + 1) * devices_per_partition]
        partition_objs.append(
            {
                "name": f"Partition{i}",
                "nodeList": node_list,
                "devices": [{"deviceId": dev_id, "numCores": num_cores} for dev_id in assigned_devices],
            }
        )

    return {
        "connections": [{"devices": device_ids, "type": "p2p"}],
        "partitions": partition_objs,
    }


def generate_disagg_mdp_intersection_config(
    onnx_path: str,
    compiler_dump_path: str,
    num_devices: int,
    num_partitions: int,
    num_layers: int,
    num_cores: int = 16,
) -> Dict[str, Any]:
    """Generate an MDP config by intersecting the QEff MDP with the compiler dump.

    The QEff MDP (derived from the ONNX graph) is always a superset of the
    compiler dump: every node the compiler actually processes appears in the
    QEff MDP, but the QEff MDP also contains nodes the compiler optimises away
    (constant-folded, etc.).  The intersection keeps only nodes that survive
    into the compiler IR, while preserving the **QEff MDP node order and
    partition assignment** as the authoritative source of truth.

    Algorithm
    ---------
    1. Generate the full QEff MDP via ``generate_disagg_mdp_partition_config``
       (ONNX topsort, layer-heuristic partition assignment).  This is the
       superset and determines both node order and partition boundaries.
    2. Read the compiler dump JSON and collect the set of real Glow IR node
       names the compiler will actually process.
    3. For each partition in the QEff MDP, retain only nodes whose names appear
       in the compiler dump set, in the original QEff order.

    Why this is better than re-assigning compiler-dump nodes by heuristic:
    - Node order follows the ONNX topsort (compiler's SplitPlanMerge requires
      this); the compiler dump order is the compiler's internal IR order which
      may differ after constant-folding and other transforms.
    - Partition boundaries are the same as the onnx strategy — no divergence.
    - Every retained node is an exact-match hit in ``doManualPartitioning()``.
    - Output JSON is compact (~1-2 MB) because constant-folded nodes are pruned.

    Parameters
    ----------
    onnx_path          : path to the exported ONNX file (same as used for
                         ``generate_disagg_mdp_partition_config``).
    compiler_dump_path : path to the JSON produced by
                         ``qaic-compile -mdp-dump-partition-config=<path>``.
    num_devices        : total devices (e.g. 2 for PP-only, 4 for 2x2 PP+TS).
    num_partitions     : pipeline stages (e.g. 2).
    num_layers         : transformer layers (e.g. 40, 94).
    num_cores          : NSP cores per device (default 16).

    Returns
    -------
    dict  with 'connections' and 'partitions' keys (same schema as all other
          generate_disagg_mdp_* functions).
    """
    import json as _json
    from pathlib import Path as _Path

    dump_path = _Path(compiler_dump_path)
    if not dump_path.exists():
        raise FileNotFoundError(
            f"Compiler dump not found: {dump_path}. Run qaic-compile with -mdp-dump-partition-config=<path> first."
        )

    logger.info(f"Generating QEff MDP (superset) from ONNX: {onnx_path}")
    qeff_mdp = generate_disagg_mdp_partition_config(
        onnx_path=onnx_path,
        num_devices=num_devices,
        num_partitions=num_partitions,
        num_layers=num_layers,
        num_cores=num_cores,
    )

    logger.info(f"Loading compiler dump for intersection: {dump_path}")
    with open(dump_path) as fh:
        dump = _json.load(fh)

    compiler_nodes: Set[str] = set()
    for part in dump.get("partitions", []):
        for name in part.get("nodeList", []):
            if name:
                compiler_nodes.add(name)

    logger.info(f"Compiler dump: {len(compiler_nodes)} unique node names")

    qeff_total = sum(len(p["nodeList"]) for p in qeff_mdp["partitions"])
    logger.info(f"QEff MDP (superset): {qeff_total} nodes across {num_partitions} partitions")

    partition_objs = []
    total_kept = 0
    total_dropped = 0
    for part in qeff_mdp["partitions"]:
        kept = [name for name in part["nodeList"] if name in compiler_nodes]
        dropped = len(part["nodeList"]) - len(kept)
        total_kept += len(kept)
        total_dropped += dropped
        logger.info(
            f"  {part['name']}: {len(part['nodeList'])} QEff nodes "
            f"-> {len(kept)} kept, {dropped} dropped (compiler-optimised away)"
        )
        partition_objs.append(
            {
                "name": part["name"],
                "nodeList": kept,
                "devices": part["devices"],
            }
        )

    logger.info(
        f"Intersection: {total_kept} nodes kept, {total_dropped} dropped "
        f"({total_dropped * 100 // max(qeff_total, 1)}% pruned by compiler)"
    )

    in_dump_not_qeff = compiler_nodes - {name for part in qeff_mdp["partitions"] for name in part["nodeList"]}
    if in_dump_not_qeff:
        logger.warning(
            f"{len(in_dump_not_qeff)} compiler-dump nodes not found in QEff MDP "
            f"(compiler may have renamed them). Examples: {list(in_dump_not_qeff)[:5]}"
        )

    return {
        "connections": qeff_mdp["connections"],
        "partitions": partition_objs,
    }
