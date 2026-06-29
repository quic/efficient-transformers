# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import hashlib
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import onnx
import onnx_ir
from onnx import external_data_helper

from QEfficient.base.onnx_transforms import CustomOpTransform, RemovePrefix

# ============================================================
# PREFIX/DELETION CONFIG (defaults preserved)
# ============================================================
SAVE_WORKERS = 8
DELETE_WORKERS = 8
DELETE_SUFFIXES = ("all_down_proj", "all_gate_proj", "all_up_proj")
_delete_pool = ThreadPoolExecutor(max_workers=DELETE_WORKERS)

_NONDETERMINISTIC_ONNX_OPS = frozenset(
    {
        "Dropout",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)


def _discover_layer_windows(exported_path: str, start_layer: int = 0) -> List[Tuple[int, int]]:
    base_path = f"{exported_path}/onnx_layerwise_tmp"
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Missing layerwise directory: {base_path}")

    windows: List[Tuple[int, int]] = []
    pat = re.compile(r"^layer_(\d+)_(\d+)$")
    for entry in os.scandir(base_path):
        if not entry.is_dir():
            continue
        m = pat.match(entry.name)
        if not m:
            continue
        layer_start, layer_end = int(m.group(1)), int(m.group(2))
        if layer_end <= layer_start:
            continue
        if layer_start < start_layer:
            continue
        windows.append((layer_start, layer_end))

    windows.sort(key=lambda x: x[0])
    if not windows:
        raise RuntimeError(f"No layer windows found in {base_path}. Expected directories like layer_<start>_<end>.")
    return windows


def _window_paths(exported_path: str, layer_start: int, layer_end: int) -> Tuple[str, str, str]:
    base_dir = f"{exported_path}/onnx_layerwise_tmp/layer_{layer_start}_{layer_end}"
    suffix = f"layer_tmp_{layer_start}_{layer_end}.onnx"

    onnx_tmp = None
    for fname in os.listdir(base_dir):
        if fname.endswith(suffix):
            onnx_tmp = os.path.join(base_dir, fname)
            break

    if onnx_tmp is None:
        raise FileNotFoundError(f"No ONNX file found with suffix: {suffix}")

    split_graph = f"{base_dir}/split_graph.onnx"
    return base_dir, onnx_tmp, split_graph


# ============================================================
# STAGE 1: SPLITTING
# ============================================================
def split_layer_graph(
    shard_idx: int,
    total_shards: int,
    exported_path: str,
    layer_start: int,
    layer_end: int,
) -> bool:
    base_dir, onnx_path, out_path = _window_paths(exported_path, layer_start, layer_end)

    if not os.path.exists(onnx_path):
        return False

    model = onnx.load(onnx_path, load_external_data=False)

    model_ir = onnx_ir.load(onnx_path)

    graph_inputs = [v.name for v in model.graph.input]
    graph_outputs = [v.name for v in model.graph.output]

    if layer_start == 0:
        if "deepstack_features" in graph_inputs:
            preferred_inputs = ["input_ids", "position_ids", "deepstack_features"]
        else:
            preferred_inputs = ["input_ids", "position_ids"]
    else:
        preferred_inputs = ["inputs_embeds", "position_ids"]

    cache_inputs = sorted(
        [
            n
            for n in graph_inputs
            if n.startswith("past_key.")
            or n.startswith("past_value.")
            or n.startswith("conv_state.")
            or n.startswith("recurrent_state.")
            or n == "vision_embeds"
            or n == "image_idx"
        ]
    )
    input_names = [n for n in preferred_inputs if n in graph_inputs] + cache_inputs

    output_names = list(graph_outputs)
    if shard_idx != total_shards - 1 and "position_ids" in graph_inputs and "position_ids" not in output_names:
        output_names.append("position_ids")

    model_ir.graph = onnx_ir.convenience.extract(
        model_ir.graph,
        input_names,
        output_names,
    )

    onnx_ir.save(model_ir, out_path)
    onnx.load(out_path, load_external_data=False)

    return True


def run_split_pipeline(
    exported_path: str,
    num_layers: int = 61,
    start_layer: int = 0,
    windows: list[tuple[int, int]] = [],
    verbose: bool = False,
) -> None:
    windows = _discover_layer_windows(exported_path, start_layer=start_layer)
    for shard_idx, (layer_start, layer_end) in enumerate(windows):
        split_layer_graph(shard_idx, len(windows), exported_path, layer_start, layer_end)
    if verbose:
        print(f"[DONE] split pipeline complete ({len(windows)} windows)")


# ============================================================
# STAGE 2: PREFIX + DELETION
# ============================================================


def delete_layer_dirs(exported_path: str, layer_windows: List[Tuple[int, int]]) -> None:
    for layer_start, layer_end in layer_windows:
        layer_dir = f"{exported_path}/onnx_layerwise_tmp/layer_{layer_start}_{layer_end}"

        if os.path.isdir(layer_dir):
            shutil.rmtree(layer_dir)  # deletes entire directory


def rewrite_tensors_with_prefix(
    model: onnx.ModelProto,
    prefix: str,
    func_attr_tens,
    size_threshold: int = 1024,
    file_chunk_size: int = 10 * 2**30,
) -> None:
    size = 0
    file_num = 0

    for tensor in external_data_helper._get_all_tensors(model):
        if tensor.HasField("raw_data") and tensor.name != "int64_2" and tensor.name not in func_attr_tens:
            tsize = len(tensor.raw_data)
            if tsize > size_threshold:
                if size + tsize > file_chunk_size:
                    file_num += 1
                    size = tsize
                else:
                    size += tsize

                external_data_helper.set_external_data(tensor, f"{prefix}_{file_num}.onnx.data")


def saving_prefix_file(
    location: str, layer_start: int, layer_end: int, exported_path: str, final_data_dir: str
) -> None:
    model = onnx.load(location, load_external_data=False)

    model_pref = onnx.compose.add_prefix(model, f"layer_{layer_start}/", rename_functions=False)

    base_dir = f"{exported_path}/onnx_layerwise_tmp/layer_{layer_start}_{layer_end}"
    external_data_helper.load_external_data_for_model(model_pref, base_dir)

    func_attr_tens = set()
    if model_pref.functions:
        func_attr_tens = {
            v.name for v in external_data_helper._get_attribute_tensors_from_graph(model_pref.functions[0])
        }

    rewrite_tensors_with_prefix(
        model_pref,
        prefix=f"layer_{layer_start}",
        func_attr_tens=func_attr_tens,
    )

    out_dir = f"{exported_path}/{final_data_dir}"
    os.makedirs(out_dir, exist_ok=True)
    onnx.save(model_pref, f"{out_dir}/pref_{layer_start}.onnx")


def run_saving_prefix(layer_start: int, layer_end: int, exported_path: str, final_data_dir: str) -> int:
    _, _, loc = _window_paths(exported_path, layer_start, layer_end)
    saving_prefix_file(loc, layer_start, layer_end, exported_path, final_data_dir)
    return layer_start


def run_prefix_pipeline(
    exported_path: str,
    num_layers: int = 61,
    chunk_size: int = 8,
    final_data_dir: str = "final_data",
    windows: list[tuple[int, int]] = [],
    verbose: bool = False,
) -> None:
    windows = _discover_layer_windows(exported_path, start_layer=0)

    for chunk_start in range(0, len(windows), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(windows))
        chunk_windows = windows[chunk_start:chunk_end]
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=SAVE_WORKERS) as pool:
            futures = [
                pool.submit(run_saving_prefix, layer_start, layer_end, exported_path, final_data_dir)
                for (layer_start, layer_end) in chunk_windows
            ]
            for f in as_completed(futures):
                f.result()
        _ = time.time() - t0

        delete_layer_dirs(exported_path, chunk_windows)

    if verbose:
        print(f"[DONE] prefix+deletion pipeline complete ({len(windows)} windows)")


# ============================================================
# STAGE 3: MERGING
# ============================================================
def compare_onnx_func(func1: onnx.FunctionProto, func2: onnx.FunctionProto):
    if (
        len(func1.input) != len(func2.input)
        or len(func1.output) != len(func2.output)
        or len(func1.node) != len(func2.node)
    ):
        return False

    for i in range(len(func1.node)):
        node1 = func1.node[i]
        node2 = func2.node[i]

        if len(node1.input) != len(node2.input):
            return False
        for j in range(len(node1.input)):
            if node1.input[j] in func1.input:
                idx = list(func1.input).index(node1.input[j])
                if node2.input[j] not in func2.input or list(func2.input).index(node2.input[j]) != idx:
                    return False
            elif node1.input[j] != node2.input[j]:
                if node1.input[j] in func1.output:
                    idx = list(func1.output).index(node1.input[j])
                    if node2.input[j] not in func2.output or list(func2.output).index(node2.input[j]) != idx:
                        return False
                else:
                    return False

        if node1.op_type != node2.op_type:
            return False
        if len(node1.attribute) != len(node2.attribute):
            return False
        for j in range(len(node1.attribute)):
            if node1.attribute[j] != node2.attribute[j]:
                return False

        if len(node1.output) != len(node2.output):
            return False
        for j in range(len(node1.output)):
            if node1.output[j] in func1.output:
                idx = list(func1.output).index(node1.output[j])
                if node2.output[j] not in func2.output or list(func2.output).index(node2.output[j]) != idx:
                    return False
            else:
                if node1.output[j] != node2.output[j]:
                    return False

    return True


def merge_models(m1, m2, io_map):
    def is_decoder(name: str) -> bool:
        return "DecoderLayer" in name

    def copy_with_name(func: onnx.FunctionProto, new_name: str) -> onnx.FunctionProto:
        f = onnx.FunctionProto()
        f.CopyFrom(func)
        f.name = new_name
        return f

    def update_node_calls(graph: onnx.GraphProto, old_name: str, new_name: str):
        if old_name == new_name:
            return
        for node in graph.node:
            if node.op_type == old_name:
                node.op_type = new_name

    try:
        graph = onnx.compose.merge_graphs(m1.graph, m2.graph, io_map)
    except Exception:
        first, second = io_map[0]
        parts = first.rsplit("//", 1)
        layer = parts[0] if len(parts) == 2 else parts[1]
        io_map[0] = (f"{layer}/logits", second)
        graph = onnx.compose.merge_graphs(m1.graph, m2.graph, io_map)

    model = onnx.helper.make_model_gen_version(
        graph,
        producer_name="QEfficient",
        producer_version="1.21",
        ir_version=10,
        opset_imports=m1.opset_import,
    )

    props = {}
    for p in m1.metadata_props:
        props[p.key] = p.value
    for p in m2.metadata_props:
        if p.key in props and props[p.key] != p.value:
            raise ValueError(
                "Can't merge models with different values for the same model metadata property."
                f" Found: property = {p.key}, with values {props[p.key]} and {p.value}."
            )
        props[p.key] = p.value
    onnx.helper.set_model_props(model, props)

    m1_funcs = [f.name for f in m1.functions]
    m2_funcs = [f.name for f in m2.functions]
    decoder_variants = {}
    used_function_names = set(m1_funcs + m2_funcs)

    def assign_decoder_variant(base_name: str, func: onnx.FunctionProto, src_graph: onnx.GraphProto) -> str:
        """Assign a unique function name for decoder variants with different signatures."""
        variants = decoder_variants.setdefault(base_name, [])

        for existing_func, assigned_name in variants:
            if compare_onnx_func(func, existing_func):
                return assigned_name

        if not variants:
            assigned = base_name
        else:
            suffix = len(variants) + 1
            assigned = f"{base_name}__v{suffix}"
            # A previous merge can already contain __vN; keep probing until the
            # assigned name is globally unique across both source models.
            while assigned in used_function_names:
                suffix += 1
                assigned = f"{base_name}__v{suffix}"
        variants.append((func, assigned))
        used_function_names.add(assigned)
        if assigned != base_name:
            update_node_calls(src_graph, base_name, assigned)
        return assigned

    final_funcs = {}
    all_names = set(m1_funcs + m2_funcs)

    for name in all_names:
        in_m1 = name in m1_funcs
        in_m2 = name in m2_funcs

        if in_m1 and in_m2:
            func1 = m1.functions[m1_funcs.index(name)]
            func2 = m2.functions[m2_funcs.index(name)]

            if compare_onnx_func(func1, func2):
                final_funcs[(func1.domain, func1.name)] = func1
            else:
                if is_decoder(name):
                    name1 = assign_decoder_variant(name, func1, m1.graph)
                    name2 = assign_decoder_variant(name, func2, m2.graph)

                    f1 = func1 if func1.name == name1 else copy_with_name(func1, name1)
                    f2 = func2 if func2.name == name2 else copy_with_name(func2, name2)
                    final_funcs[(f1.domain, f1.name)] = f1
                    final_funcs[(f2.domain, f2.name)] = f2
                else:
                    raise ValueError(f"Function '{name}' differs between models and is not a DecoderLayer.")
        elif in_m1:
            f = m1.functions[m1_funcs.index(name)]
            final_funcs[(f.domain, f.name)] = f
        elif in_m2:
            f = m2.functions[m2_funcs.index(name)]
            final_funcs[(f.domain, f.name)] = f
        else:
            raise ValueError("Function not found")

    graph2 = onnx.compose.merge_graphs(m1.graph, m2.graph, io_map)
    model.graph.CopyFrom(graph2)

    for (domain, name), f in final_funcs.items():
        if f.name != name:
            f = copy_with_name(f, name)
        model.functions.MergeFrom([f])

    return model


def run_merge_pipeline(
    exported_path: str,
    num_layers: int = 61,
    final_data_dir: str = "final_data",
    windows: list[tuple[int, int]] = [],
    verbose: bool = False,
) -> str:
    if len(windows) < 1:
        raise ValueError("Need at least one discovered shard to merge")

    base_dir = f"{exported_path}/{final_data_dir}"
    start = time.time()

    shard_starts = [layer_start for (layer_start, _) in windows]
    first_start = windows[0][0]
    last_end = windows[-1][1]

    if len(shard_starts) == 1:
        only_model = f"{base_dir}/pref_{first_start}.onnx"
        if not os.path.exists(only_model):
            raise FileNotFoundError(f"Missing input model: {only_model}")
        return only_model

    for idx in range(len(shard_starts) - 1):
        left = shard_starts[len(shard_starts) - idx - 2]
        right = shard_starts[len(shard_starts) - idx - 1]

        m1_path = f"{base_dir}/pref_{left}.onnx"
        m2_path = f"{base_dir}/pref_{right}.onnx" if idx == 0 else f"{base_dir}/merged_{right}-{last_end}.onnx"

        if not os.path.exists(m1_path):
            raise FileNotFoundError(f"Missing input model: {m1_path}")
        if not os.path.exists(m2_path):
            raise FileNotFoundError(f"Missing input model: {m2_path}")

        m1_pref = onnx.load(m1_path, load_external_data=False)
        m2_pref = onnx.load(m2_path, load_external_data=False)

        graph_outputs = [output.name for output in m1_pref.graph.output]
        selected_output = next(
            (
                name
                for name in graph_outputs
                if "RetainedState" not in name and not name.endswith("position_ids") and "image_idx" not in name
            ),
            None,
        )
        if selected_output is None:
            raise RuntimeError(f"No mergeable decoder output found in {m1_path}. Outputs: {graph_outputs}")

        merged_model = merge_models(
            m1_pref,
            m2_pref,
            io_map=[
                (selected_output, f"layer_{right}/inputs_embeds"),
                (f"layer_{left}/position_ids", f"layer_{right}/position_ids"),
            ],
        )

        if idx == len(shard_starts) - 2:
            CustomOpTransform.apply(merged_model)

        out_path = f"{base_dir}/merged_{left}-{last_end}.onnx"
        onnx.save(merged_model, out_path)

    final_path = f"{base_dir}/merged_{first_start}-{last_end}.onnx"
    model = onnx.load(final_path, load_external_data=False)
    RemovePrefix.apply(model)
    _deduplicate_redundant_onnx_nodes(model, verbose=verbose)
    onnx.save(model, final_path)
    if verbose:
        print(f"[DONE] merge pipeline complete in {time.time() - start:.2f}s")
    return final_path


def _canonical_value_name(value_name: str, rename_map: dict[str, str]) -> str:
    """Resolve the canonical producer name after prior node dedup rewrites."""
    while value_name in rename_map:
        value_name = rename_map[value_name]
    return value_name


def _attribute_signature(attr: onnx.AttributeProto):
    """Build a stable hashable signature for ONNX node attributes."""
    attr_type = attr.type
    if attr_type == onnx.AttributeProto.FLOAT:
        return (attr.name, attr_type, attr.f)
    if attr_type == onnx.AttributeProto.INT:
        return (attr.name, attr_type, attr.i)
    if attr_type == onnx.AttributeProto.STRING:
        return (attr.name, attr_type, bytes(attr.s))
    if attr_type == onnx.AttributeProto.FLOATS:
        return (attr.name, attr_type, tuple(attr.floats))
    if attr_type == onnx.AttributeProto.INTS:
        return (attr.name, attr_type, tuple(attr.ints))
    if attr_type == onnx.AttributeProto.STRINGS:
        return (attr.name, attr_type, tuple(bytes(v) for v in attr.strings))
    if attr_type == onnx.AttributeProto.TENSOR:
        # Ignore TensorProto.name because exporters can emit different internal
        # names for identical constant values.
        tensor = onnx.TensorProto()
        tensor.CopyFrom(attr.t)
        tensor.name = ""
        tensor_data = tensor.SerializeToString()
        return (attr.name, attr_type, hashlib.sha256(tensor_data).hexdigest())
    return (attr.name, attr_type, attr.SerializeToString())


def _node_signature(node: onnx.NodeProto, normalized_inputs: list[str]):
    attrs = tuple(_attribute_signature(attr) for attr in sorted(node.attribute, key=lambda x: x.name))
    return (
        node.domain,
        node.op_type,
        tuple(normalized_inputs),
        attrs,
        len(node.output),
    )


def _deduplicate_redundant_onnx_nodes(model: onnx.ModelProto, verbose: bool = False) -> int:
    """Run a local CSE pass on merged layerwise ONNX graph nodes.

    Layerwise stitch can duplicate equivalent pure ONNX precompute chains
    (mask/rope prep) across windows. This pass canonicalizes duplicate nodes by
    reusing the first producer value.
    """
    graph = model.graph
    graph_outputs = {value.name for value in graph.output}
    rename_map: dict[str, str] = {}
    seen_signatures: dict[tuple, tuple[str, ...]] = {}
    deduped_nodes = []
    removed_nodes = 0

    for node in graph.node:
        normalized_inputs = [_canonical_value_name(name, rename_map) for name in node.input]
        if tuple(normalized_inputs) != tuple(node.input):
            del node.input[:]
            node.input.extend(normalized_inputs)

        has_subgraph_attr = any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attr in node.attribute
        )
        skip_dedup = (
            node.domain not in ("", "ai.onnx")
            or not node.output
            or node.op_type in _NONDETERMINISTIC_ONNX_OPS
            or has_subgraph_attr
            or any(output_name in graph_outputs for output_name in node.output)
        )
        if skip_dedup:
            deduped_nodes.append(node)
            continue

        signature = _node_signature(node, normalized_inputs)
        existing_outputs = seen_signatures.get(signature)
        if existing_outputs is None:
            seen_signatures[signature] = tuple(node.output)
            deduped_nodes.append(node)
            continue

        for duplicate_output, canonical_output in zip(node.output, existing_outputs):
            rename_map[duplicate_output] = canonical_output
        removed_nodes += 1

    if removed_nodes == 0:
        return 0

    for node in deduped_nodes:
        normalized_inputs = [_canonical_value_name(name, rename_map) for name in node.input]
        if tuple(normalized_inputs) != tuple(node.input):
            del node.input[:]
            node.input.extend(normalized_inputs)

    del graph.node[:]
    graph.node.extend(deduped_nodes)
    if verbose:
        print(f"[INFO] layerwise dedup removed {removed_nodes} duplicate ONNX nodes")
    return removed_nodes


# ============================================================
# ONE-SHOT ENTRY
# ============================================================
def run_sequential_pipeline(
    exported_path: str,
    num_layers: int = 61,
    start_layer: int = 0,
    chunk_size: int = 8,
    final_data_dir: str = "final_data",
    verbose: bool = False,
) -> str:
    windows = _discover_layer_windows(exported_path, start_layer=0)
    run_split_pipeline(
        exported_path=exported_path,
        num_layers=num_layers,
        start_layer=start_layer,
        windows=windows,
        verbose=verbose,
    )

    run_prefix_pipeline(
        exported_path=exported_path,
        num_layers=num_layers,
        chunk_size=chunk_size,
        final_data_dir=final_data_dir,
        windows=windows,
        verbose=verbose,
    )

    final_path = run_merge_pipeline(
        exported_path=exported_path,
        num_layers=num_layers,
        final_data_dir=final_data_dir,
        windows=windows,
        verbose=verbose,
    )
    return final_path


def layerwise_pipeline(
    exported_path: str,
    num_layers: int = 61,
    start_layer: int = 0,
    chunk_size: int = 8,
    final_data_dir: str = "final_data",
    verbose: bool = False,
) -> str:
    return run_sequential_pipeline(
        exported_path=exported_path,
        num_layers=num_layers,
        start_layer=start_layer,
        chunk_size=chunk_size,
        final_data_dir=final_data_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="All-in-one layer-wise ONNX split -> prefix/deletion -> merge pipeline."
    )
    parser.add_argument("--exported_path", required=True, help="Base export path")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--final-data-dir", default="final_data")
    parser.add_argument("--verbose", action="store_true", help="Enable progress logs")
    args = parser.parse_args()

    final_path = run_sequential_pipeline(
        exported_path=args.exported_path,
        num_layers=args.num_layers,
        start_layer=args.start_layer,
        chunk_size=args.chunk_size,
        final_data_dir=args.final_data_dir,
        verbose=args.verbose,
    )
    print(final_path)
