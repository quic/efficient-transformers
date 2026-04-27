#!/usr/bin/env python3
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import onnx
import onnx_ir
from onnx import external_data_helper

from QEfficient.base.onnx_transforms import CustomOpTransform

# ============================================================
# PREFIX/DELETION CONFIG (defaults preserved)
# ============================================================
SAVE_WORKERS = 8
DELETE_WORKERS = 8
DELETE_SUFFIXES = ("onnx.data",)
_delete_pool = ThreadPoolExecutor(max_workers=DELETE_WORKERS)


# ============================================================
# STAGE 1: SPLITTING
# ============================================================
def split_layer_graph(idx: int, num_layers: int, exported_path: str) -> bool:
    base_path = f"{exported_path}/onnx_layerwise_tmp"
    onnx_path = f"{base_path}/layer_{idx}_{idx + 1}/DeepseekV3ForCausalLM_layer_tmp_{idx}_{idx + 1}.onnx"

    if not os.path.exists(onnx_path):
        print(f"[SKIP] ONNX not found: {onnx_path}")
        return False

    model = onnx.load(onnx_path, load_external_data=False)

    decoder_input = None
    decoder_output = None
    for node in model.graph.node:
        if "DecoderLayer" in node.name:
            decoder_input = list(node.input)
            decoder_output = list(node.output)
            break

    if decoder_input is None or decoder_output is None:
        raise RuntimeError(f"DecoderLayer not found in layer {idx}")

    model_ir = onnx_ir.load(onnx_path)

    if idx == 0:
        input_names = [
            "input_ids",
            "position_ids",
            "compressed_kv.0",
            "k_pe.0",
        ]
        output_names = decoder_output + ["position_ids"]
    elif idx == num_layers - 1:
        input_names = decoder_input[:4]
        output_names = ["logits"] + decoder_output[:2]
    else:
        input_names = decoder_input[:4]
        output_names = decoder_output + ["position_ids"]

    model_ir.graph = onnx_ir.convenience.extract(
        model_ir.graph,
        input_names,
        output_names,
    )

    out_path = f"{base_path}/layer_{idx}_{idx + 1}/split_graph.onnx"
    onnx_ir.save(model_ir, out_path)
    onnx.load(out_path, load_external_data=False)

    print(f"[DONE] Layer {idx}: saved split graph -> {out_path}")
    return True


def run_split_pipeline(exported_path: str, num_layers: int = 61, start_layer: int = 0) -> None:
    print(f"[START] split pipeline | exported_path={exported_path}, start_layer={start_layer}, num_layers={num_layers}")
    for idx in range(start_layer, num_layers):
        print(f"[PROCESS] Layer {idx}")
        split_layer_graph(idx, num_layers, exported_path)
    print("[DONE] split pipeline complete")


# ============================================================
# STAGE 2: PREFIX + DELETION
# ============================================================
def async_delete_files(paths: List[str]) -> None:
    def _delete(p):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[delete] failed {p}: {e}")

    for p in paths:
        _delete_pool.submit(_delete, p)


def collect_chunk_deletable_files(exported_path: str, layer_indices: List[int]) -> List[str]:
    files = []
    for idx in layer_indices:
        layer_dir = f"{exported_path}/onnx_layerwise_tmp/layer_{idx}_{idx + 1}"
        if not os.path.isdir(layer_dir):
            continue
        for entry in os.scandir(layer_dir):
            if entry.is_file() and entry.name.endswith(DELETE_SUFFIXES):
                files.append(entry.path)
    return files


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


def saving_prefix_file(location: str, idx: int, exported_path: str, final_data_dir: str) -> None:
    model = onnx.load(location, load_external_data=False)

    model_pref = onnx.compose.add_prefix(model, f"layer_{idx}/", rename_functions=False)

    base_dir = f"{exported_path}/onnx_layerwise_tmp/layer_{idx}_{idx + 1}"
    external_data_helper.load_external_data_for_model(model_pref, base_dir)

    func_attr_tens = set()
    if model_pref.functions:
        func_attr_tens = {
            v.name for v in external_data_helper._get_attribute_tensors_from_graph(model_pref.functions[0])
        }

    rewrite_tensors_with_prefix(
        model_pref,
        prefix=f"layer_{idx}",
        func_attr_tens=func_attr_tens,
    )

    out_dir = f"{exported_path}/{final_data_dir}"
    os.makedirs(out_dir, exist_ok=True)
    onnx.save(model_pref, f"{out_dir}/pref_{idx}.onnx")


def run_saving_prefix(idx: int, exported_path: str, final_data_dir: str) -> int:
    loc = f"{exported_path}/onnx_layerwise_tmp/layer_{idx}_{idx + 1}/split_graph.onnx"
    saving_prefix_file(loc, idx, exported_path, final_data_dir)
    return idx


def run_prefix_pipeline(
    exported_path: str,
    num_layers: int = 61,
    chunk_size: int = 8,
    final_data_dir: str = "final_data",
) -> None:
    print(
        f"[START] prefix+deletion pipeline | exported_path={exported_path}, "
        f"num_layers={num_layers}, chunk_size={chunk_size}"
    )

    for chunk_start in range(0, num_layers, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_layers)
        layers = list(range(chunk_start, chunk_end))

        print(f"\\n[Chunk] {chunk_start} -> {chunk_end - 1}")
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=SAVE_WORKERS) as pool:
            futures = [pool.submit(run_saving_prefix, idx, exported_path, final_data_dir) for idx in layers]
            for f in as_completed(futures):
                f.result()

        print(f"[Chunk] saved in {time.time() - t0:.2f}s")

        # deletables = collect_chunk_deletable_files(exported_path, layers)
        # async_delete_files(deletables)
        # print(f"[Chunk] scheduled deletion of {len(deletables)} files")

    print("[DONE] prefix+deletion pipeline complete")


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

    def assign_decoder_variant(base_name: str, func: onnx.FunctionProto, src_graph: onnx.GraphProto) -> str:
        variants = decoder_variants.setdefault(base_name, [])

        for existing_func, assigned_name in variants:
            if compare_onnx_func(func, existing_func):
                return assigned_name

        assigned = base_name if not variants else f"{base_name}__v{len(variants) + 1}"
        variants.append((func, assigned))
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


def run_merge_pipeline(exported_path: str, num_layers: int = 61, final_data_dir: str = "final_data") -> str:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    base_dir = f"{exported_path}/{final_data_dir}"
    start = time.time()
    print(
        f"[START] merge pipeline | exported_path={exported_path}, "
        f"num_layers={num_layers}, final_data_dir={final_data_dir}"
    )

    if num_layers == 1:
        only_model = f"{base_dir}/pref_0.onnx"
        if not os.path.exists(only_model):
            raise FileNotFoundError(f"Missing input model: {only_model}")
        print(f"[DONE] merge pipeline skipped (single layer): {only_model}")
        return only_model

    for idx in range(num_layers - 1):
        left = num_layers - idx - 2
        right = num_layers - idx - 1

        m1_path = f"{base_dir}/pref_{left}.onnx"
        m2_path = f"{base_dir}/pref_{right}.onnx" if idx == 0 else f"{base_dir}/merged_{left + 1}-{num_layers - 1}.onnx"

        if not os.path.exists(m1_path):
            raise FileNotFoundError(f"Missing input model: {m1_path}")
        if not os.path.exists(m2_path):
            raise FileNotFoundError(f"Missing input model: {m2_path}")

        print(f"[MERGE] {left}-{num_layers - 1}")
        m1_pref = onnx.load(m1_path, load_external_data=False)
        m2_pref = onnx.load(m2_path, load_external_data=False)

        decoder_nodes = [n for n in m1_pref.graph.node if "DecoderLayer" in n.name]
        if not decoder_nodes:
            raise RuntimeError(f"DecoderLayer node not found in {m1_path}")

        decoder_output = list(decoder_nodes[0].output)
        merged_model = merge_models(
            m1_pref,
            m2_pref,
            io_map=[
                (f"{decoder_output[2]}", f"layer_{right}/inputs_embeds"),
                (f"layer_{left}/position_ids", f"layer_{right}/position_ids"),
            ],
        )

        if idx == num_layers - 2:
            CustomOpTransform.apply(merged_model)

        out_path = f"{base_dir}/merged_{left}-{num_layers - 1}.onnx"
        onnx.save(merged_model, out_path)
        print(f"[SAVED] {out_path}")

    final_path = f"{base_dir}/merged_0-{num_layers - 1}.onnx"
    print(f"[DONE] merge pipeline complete in {time.time() - start:.2f}s")
    return final_path


# ============================================================
# ONE-SHOT ENTRY
# ============================================================
def run_sequential_pipeline(
    exported_path: str,
    num_layers: int = 61,
    start_layer: int = 0,
    chunk_size: int = 8,
    final_data_dir: str = "final_data",
) -> str:
    print("\\n=== Stage 1/3: Splitting ===")
    run_split_pipeline(
        exported_path=exported_path,
        num_layers=num_layers,
        start_layer=start_layer,
    )

    print("\\n=== Stage 2/3: Prefix + Deletion ===")
    run_prefix_pipeline(
        exported_path=exported_path,
        num_layers=num_layers,
        chunk_size=chunk_size,
        final_data_dir=final_data_dir,
    )

    print("\\n=== Stage 3/3: Merging ===")
    final_path = run_merge_pipeline(
        exported_path=exported_path,
        num_layers=num_layers,
        final_data_dir=final_data_dir,
    )

    print(f"\\n[PIPELINE DONE] Final merged model: {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="All-in-one layer-wise ONNX split -> prefix/deletion -> merge pipeline."
    )
    parser.add_argument("--exported_path", help="Base export path")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--final-data-dir", default="final_data")
    args = parser.parse_args()

    run_sequential_pipeline(
        exported_path=args.exported_path,
        num_layers=args.num_layers,
        start_layer=args.start_layer,
        chunk_size=args.chunk_size,
        final_data_dir=args.final_data_dir,
    )
