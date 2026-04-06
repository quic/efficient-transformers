import argparse
import itertools
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.base.onnx_transforms import FP16ClipTransform
from QEfficient.generation.text_generation_inference import TextGeneration


def normalize_generated_ids(generated_ids):
    array = np.asarray(generated_ids)
    if array.dtype == object:
        array = np.asarray([np.asarray(row).reshape(-1) for row in generated_ids], dtype=np.int64)
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array.astype(np.int64, copy=False)


def load_reference_ids(model_id: str, rendered_prompt: str, generation_len: int):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    inputs = processor(text=rendered_prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    )
    model.eval()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=generation_len, do_sample=False)
    return processor, outputs[:, input_len:].cpu().numpy()


def build_rendered_prompt(processor, system_prompt: str, user_prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_trial(
    qeff_model,
    onnx_path: str,
    processor,
    rendered_prompt: str,
    generation_len: int,
    fp32_names: list[str],
    compile_root: Path,
):
    compile_root.mkdir(parents=True, exist_ok=True)
    npi_path = compile_root / "trial_npi.yaml"
    npi_path.write_text(yaml.safe_dump({"FP32NodeInstanceNames": fp32_names}, sort_keys=False))

    qpc_path = qeff_model.compile(
        onnx_path=onnx_path,
        compile_dir=compile_root,
        prefill_seq_len=8,
        ctx_len=32,
        num_devices=1,
        num_cores=16,
        use_onnx_subfunctions=True,
        mxfp6_matmul=False,
        mxint8_kv_cache=False,
        node_precision_info=str(npi_path),
        offload_pt_weights=False,
    )

    tg = TextGeneration(tokenizer=processor.tokenizer, qpc_path=qpc_path, device_id=[0], ctx_len=32)
    exec_info = tg.generate(prompt=[rendered_prompt], generation_len=generation_len, stream=False)
    qaic_ids = normalize_generated_ids(exec_info.generated_ids)[:, :generation_len]
    return qaic_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="tiny-random/gemma-4-dense")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--prompt", default="Hi, Who are you?")
    parser.add_argument("--generation-len", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=Path("scripts/debug/gemma4_matmul_reduction_results.json"))
    parser.add_argument(
        "--save-baseline-npi", type=Path, default=Path("scripts/debug/gemma4_working_baseline_npi.yaml")
    )
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    rendered_prompt = build_rendered_prompt(processor, args.system_prompt, args.prompt)
    processor, hf_ids = load_reference_ids(args.model_id, rendered_prompt, args.generation_len)

    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype="float32",
    )
    qeff_model._onnx_transforms = [t for t in qeff_model._onnx_transforms if t is not FP16ClipTransform]
    onnx_path = qeff_model.export(use_onnx_subfunctions=True, offload_pt_weights=False)

    baseline_npi_path = Path(qeff_model.model.generate_npi_file(onnx_path=onnx_path, model_name=args.model_id))
    baseline_names = yaml.safe_load(baseline_npi_path.read_text())["FP32NodeInstanceNames"]
    args.save_baseline_npi.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(baseline_npi_path, args.save_baseline_npi)

    baseline_matmuls = [name for name in baseline_names if "MatMul" in name]
    non_matmuls = [name for name in baseline_names if "MatMul" not in name]

    results = {
        "model_id": args.model_id,
        "rendered_prompt": rendered_prompt,
        "generation_len": args.generation_len,
        "baseline_npi_path": str(baseline_npi_path),
        "saved_baseline_npi_path": str(args.save_baseline_npi),
        "baseline_total_count": len(baseline_names),
        "baseline_matmul_entries": baseline_matmuls,
        "hf_ids": hf_ids.tolist(),
        "trials": [],
    }

    with tempfile.TemporaryDirectory(prefix="gemma4_matmul_reduce_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Verify baseline first.
        qaic_ids = run_trial(
            qeff_model=qeff_model,
            onnx_path=onnx_path,
            processor=processor,
            rendered_prompt=rendered_prompt,
            generation_len=args.generation_len,
            fp32_names=baseline_names,
            compile_root=tmpdir / "baseline",
        )
        baseline_match = bool(np.array_equal(hf_ids, qaic_ids))
        results["baseline_match"] = baseline_match
        results["baseline_qaic_ids"] = qaic_ids.tolist()
        if not baseline_match:
            raise RuntimeError("Baseline working NPI no longer matches; aborting reduction search.")

        current_matmuls = list(baseline_matmuls)
        greedy_steps = []

        # Greedy single-removal pass.
        changed = True
        while changed:
            changed = False
            for candidate in list(current_matmuls):
                trial_matmuls = [name for name in current_matmuls if name != candidate]
                trial_names = non_matmuls + trial_matmuls
                trial_dir = tmpdir / f"greedy_{len(greedy_steps)}"
                qaic_ids = run_trial(
                    qeff_model=qeff_model,
                    onnx_path=onnx_path,
                    processor=processor,
                    rendered_prompt=rendered_prompt,
                    generation_len=args.generation_len,
                    fp32_names=trial_names,
                    compile_root=trial_dir,
                )
                match = bool(np.array_equal(hf_ids, qaic_ids))
                step = {
                    "phase": "greedy_single_removal",
                    "candidate_removed": candidate,
                    "remaining_matmuls": trial_matmuls,
                    "final_npi_count": len(trial_names),
                    "qaic_ids": qaic_ids.tolist(),
                    "match": match,
                }
                greedy_steps.append(step)
                if match:
                    current_matmuls = trial_matmuls
                    changed = True
                    break

        results["greedy_steps"] = greedy_steps
        results["greedy_remaining_matmuls"] = current_matmuls

        # Exhaustive search across the remaining matmuls after greedy pruning.
        minimal_subset = None
        exhaustive_steps = []
        for subset_size in range(len(current_matmuls) + 1):
            for subset in itertools.combinations(current_matmuls, subset_size):
                trial_matmuls = list(subset)
                trial_names = non_matmuls + trial_matmuls
                trial_dir = tmpdir / f"subset_{subset_size}_{len(exhaustive_steps)}"
                qaic_ids = run_trial(
                    qeff_model=qeff_model,
                    onnx_path=onnx_path,
                    processor=processor,
                    rendered_prompt=rendered_prompt,
                    generation_len=args.generation_len,
                    fp32_names=trial_names,
                    compile_root=trial_dir,
                )
                match = bool(np.array_equal(hf_ids, qaic_ids))
                exhaustive_steps.append(
                    {
                        "phase": "exhaustive_subset_search",
                        "subset_size": subset_size,
                        "matmul_subset": trial_matmuls,
                        "final_npi_count": len(trial_names),
                        "qaic_ids": qaic_ids.tolist(),
                        "match": match,
                    }
                )
                if match:
                    minimal_subset = trial_matmuls
                    break
            if minimal_subset is not None:
                break

        results["exhaustive_steps"] = exhaustive_steps
        results["minimal_verified_matmuls"] = minimal_subset
        results["minimal_verified_npi_count"] = (
            None if minimal_subset is None else len(non_matmuls) + len(minimal_subset)
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
