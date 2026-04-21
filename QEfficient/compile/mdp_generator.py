# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""MDP generator for disaggregated prefill serving (PP-enabled, TS-enabled, stages>1)."""

import logging
from typing import Any, Dict, List, Optional, Set

import onnx

logger = logging.getLogger(__name__)


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

    Supports layers.N (Llama/Mistral/Qwen/Gemma/Granite) and h.N (GPT-2).
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
    return None


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
    # Registered with DEFINEKNOWNCUSTOMOP in ONNXModelLoader.cpp
    _KNOWN_CUSTOM_OPS = frozenset({"CustomRMSNorm"})

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

    Assigns nodes to partitions by transformer layer index. Non-layer nodes
    (embeddings, lm_head) follow the nearest layer in topological order.
    nodeList is a superset of the compiler dump; the compiler silently ignores
    optimized-away names. Inlined local function call-sites (CtxScatterCB,
    CtxGatherCB) are excluded; their /nNN sub-nodes are assigned automatically.
    Known custom ops (CustomRMSNorm) are included by call-site name.

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
    assert num_partitions <= num_devices, f"num_partitions ({num_partitions}) must be <= num_devices ({num_devices})"

    layers_per_partition = num_layers // num_partitions
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

    inlined_node_map, non_inlined_functions = _get_inlined_node_map(model)
    inlined_functions = {f.name for f in model.functions} - non_inlined_functions

    # First pass: assign main graph nodes to partitions by layer index.
    partitions: List[List[str]] = [[] for _ in range(num_partitions)]
    current_layer_partition = 0
    seen_first_layer = False
    max_layer_seen = -1

    for node in model.graph.node:
        if not node.name.startswith("/"):
            continue
        if node.name in folded_nodes:
            continue
        if node.op_type in inlined_functions:
            continue  # inlined; sub-nodes added in second pass

        layer_num = _get_layer_num(node.name)
        if layer_num is not None:
            max_layer_seen = max(max_layer_seen, layer_num)
            seen_first_layer = True
            partition_idx = min(layer_num // layers_per_partition, num_partitions - 1)
            current_layer_partition = partition_idx
            partitions[partition_idx].append(node.name)
        else:
            if not seen_first_layer:
                partitions[0].append(node.name)
            else:
                partitions[current_layer_partition].append(node.name)

    # Second pass: add inlined sub-nodes, inheriting their call-site's partition.
    for call_site_name, inlined_nodes in inlined_node_map.items():
        layer_num = _get_layer_num(call_site_name)
        if layer_num is not None:
            partition_idx = min(layer_num // layers_per_partition, num_partitions - 1)
        else:
            partition_idx = current_layer_partition
        partitions[partition_idx].extend(inlined_nodes)

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
