import argparse
import contextlib
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_SPECS = {
    "codegen": {"id": "hf-internal-testing/tiny-random-CodeGenForCausalLM", "task": "causal_lm"},
    "falcon": {"id": "hf-internal-testing/tiny-random-FalconForCausalLM", "task": "causal_lm"},
    "gpt2": {"id": "hf-internal-testing/tiny-random-GPT2LMHeadModel", "task": "causal_lm"},
    "gpt_oss": {"id": "tiny-random/gpt-oss-bf16", "task": "causal_lm"},
    "gptj": {"id": "hf-internal-testing/tiny-random-GPTJForCausalLM", "task": "causal_lm"},
    "granite": {"id": "hf-internal-testing/tiny-random-GraniteForCausalLM", "task": "causal_lm"},
    "llama": {"id": "hf-internal-testing/tiny-random-LlamaForCausalLM", "task": "causal_lm"},
    "mistral": {"id": "hf-internal-testing/tiny-random-MistralForCausalLM", "task": "causal_lm"},
    "mixtral": {"id": "hf-internal-testing/tiny-random-MixtralForCausalLM", "task": "causal_lm"},
    "mpt": {"id": "hf-internal-testing/tiny-random-MptForCausalLM", "task": "causal_lm"},
    "olmo2": {"id": "hf-internal-testing/tiny-random-Olmo2ForCausalLM", "task": "causal_lm"},
    "phi": {"id": "hf-internal-testing/tiny-random-PhiForCausalLM", "task": "causal_lm"},
    "phi3": {"id": "tiny-random/phi-4", "task": "causal_lm"},
    "qwen2": {"id": "yujiepan/qwen2-tiny-random", "task": "causal_lm"},
    "qwen3_moe": {"id": "yujiepan/qwen3-moe-tiny-random", "task": "causal_lm"},
    "starcoder2": {"id": "hf-internal-testing/tiny-random-Starcoder2ForCausalLM", "task": "causal_lm"},
    "embedding_bert": {"id": "hf-internal-testing/tiny-random-BertModel", "task": "embedding"},
    "seqcls_bert": {"id": "ydshieh/tiny-random-BertForSequenceClassification", "task": "sequence_classification"},
    "ctc_wav2vec2": {"id": "hf-internal-testing/tiny-random-wav2vec2", "task": "ctc"},
    "whisper": {"id": "hf-internal-testing/tiny-random-WhisperForConditionalGeneration", "task": "speech_seq2seq"},
    "awq_mixtral": {"id": "optimum-intel-internal-testing/tiny-mixtral-AWQ-4bit", "task": "awq_causal_lm"},
    "vlm_gemma3": {"id": "tiny-random/gemma-3", "task": "vlm"},
    "vlm_qwen2_5": {"id": "optimum-intel-internal-testing/tiny-random-qwen2.5-vl", "task": "vlm"},
    "vlm_internvl2": {"id": "optimum-intel-internal-testing/tiny-random-internvl2", "task": "vlm"},
}

MEM_RE = re.compile(r"([+-]?[0-9.]+)\s+(B|KiB|MiB|GiB|TiB)|n/a")
UNITS = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}

EXPORT_SNIPPET = r"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    Qwen2Config,
)

from QEfficient.transformers.models.modeling_auto import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForCTC,
    QEFFAutoModelForImageTextToText,
    QEFFAutoModelForSequenceClassification,
    QEFFAutoModelForSpeechSeq2Seq,
)
from QEfficient.transformers.quantizers.auto import replace_transformers_quantizers

model_key, model_id, task, export_dir = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
kwargs = {"attn_implementation": "eager", "trust_remote_code": True, "low_cpu_mem_usage": False}

if task == "causal_lm":
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs, torch_dtype=torch.float16)
    model.eval()
    qeff_model = QEFFAutoModelForCausalLM(model, pretrained_model_name_or_path=model_id)
elif task == "awq_causal_lm":
    replace_transformers_quantizers()
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=False, torch_dtype=torch.float32)
    model.eval()
    qeff_model = QEFFAutoModelForCausalLM(model, pretrained_model_name_or_path=model_id)
elif task == "embedding":
    model = AutoModel.from_pretrained(model_id, **kwargs)
    model.eval()
    qeff_model = QEFFAutoModel(model, pretrained_model_name_or_path=model_id)
elif task == "sequence_classification":
    model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    qeff_model = QEFFAutoModelForSequenceClassification(model, pretrained_model_name_or_path=model_id)
elif task == "ctc":
    replace_transformers_quantizers()
    model = AutoModelForCTC.from_pretrained(model_id, **kwargs)
    model.eval()
    qeff_model = QEFFAutoModelForCTC(model, pretrained_model_name_or_path=model_id)
elif task == "speech_seq2seq":
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **kwargs)
    model.eval()
    qeff_model = QEFFAutoModelForSpeechSeq2Seq(model, pretrained_model_name_or_path=model_id)
elif task == "vlm":
    def text_fallback():
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        if model_type in {"qwen2_5_vl", "qwen2_5_vl_text"} and getattr(config, "text_config", None) is not None:
            cfg_dict = config.text_config.to_dict()
            cfg_dict["model_type"] = "qwen2"
            allowed = set(Qwen2Config().to_dict().keys())
            text_config = Qwen2Config(**{k: v for k, v in cfg_dict.items() if k in allowed})
        else:
            text_config = getattr(config, "text_config", None) or getattr(config, "llm_config", None)
        if text_config is None:
            raise RuntimeError(f"No text fallback config for {model_id}")
        text_model = AutoModelForCausalLM.from_config(text_config, trust_remote_code=True, attn_implementation="eager")
        text_model = text_model.to(torch.float32)
        text_model.eval()
        return QEFFAutoModelForCausalLM(text_model, pretrained_model_name_or_path=model_id + "#text-fallback")
    try:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        qeff_model = text_fallback()
else:
    raise ValueError(task)

t0 = time.perf_counter()
onnx_path = qeff_model.export(export_dir, offload_pt_weights=False)
elapsed = time.perf_counter() - t0
print(json.dumps({"onnx_path": str(onnx_path), "export_wall_sec": elapsed}))
"""


def parse_size(value):
    if value in (None, "n/a"):
        return None
    match = re.fullmatch(r"([+-]?[0-9.]+)\s+(B|KiB|MiB|GiB|TiB)", value)
    if not match:
        return None
    return int(float(match.group(1)) * UNITS[match.group(2)])


def mib(value):
    return None if value is None else value / 1024**2


def parse_mem_profile(stderr):
    result = {"stages": {}, "snapshots": []}
    for line in stderr.splitlines():
        if "[QEFF-MEM]" not in line:
            continue
        message = line.split("[QEFF-MEM]", 1)[1].strip()
        if message.startswith("snapshot "):
            label = message.split(":", 1)[0].removeprefix("snapshot ")
            fields = dict(re.findall(r"(rss|uss|vms|hwm|ru_maxrss|py)=([^=]+?)(?=\s+[a-zA-Z_]+=|$)", message))
            result["snapshots"].append(
                {
                    "label": label,
                    "rss_mib": mib(parse_size(fields.get("rss"))),
                    "hwm_mib": mib(parse_size(fields.get("hwm"))),
                    "ru_maxrss_mib": mib(parse_size(fields.get("ru_maxrss"))),
                }
            )
        elif message.startswith("stage "):
            match = re.match(
                r"stage ([^:]+): elapsed=([0-9.]+)s rss_delta=([^=]+?) hwm_delta=([^=]+?) ru_maxrss_delta=([^=]+?) stage_peak_rss=([^=]+)$",
                message,
            )
            if match:
                stage, elapsed, rss_delta, hwm_delta, ru_delta, stage_peak = match.groups()
                result["stages"][stage] = {
                    "elapsed_sec": float(elapsed),
                    "rss_delta_mib": mib(parse_size(rss_delta.strip())),
                    "hwm_delta_mib": mib(parse_size(hwm_delta.strip())),
                    "ru_maxrss_delta_mib": mib(parse_size(ru_delta.strip())),
                    "stage_peak_rss_mib": mib(parse_size(stage_peak.strip())),
                }
        elif message.startswith("report ") and "sampled_peak_rss" in message:
            sampled = re.search(r"sampled_peak_rss=([^=]+?) peak_hwm", message)
            hwm = re.search(r"peak_hwm=([^=]+?) peak_ru_maxrss", message)
            ru = re.search(r"peak_ru_maxrss=([^=]+)$", message)
            result["sampled_peak_rss_mib"] = mib(parse_size(sampled.group(1).strip())) if sampled else None
            result["peak_hwm_mib"] = mib(parse_size(hwm.group(1).strip())) if hwm else None
            result["peak_ru_maxrss_mib"] = mib(parse_size(ru.group(1).strip())) if ru else None
    return result


@contextlib.contextmanager
def temp_home(prefix):
    root = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def mode_env_value(mode):
    if mode == "legacy":
        return "0"
    if mode in {"after", "default"}:
        return None
    if mode == "forced_lowmem":
        return "1"
    raise ValueError(mode)


def run_one(model_key, spec, mode, python_exe, repo_root, keep_logs):
    with temp_home(f"qeff_{model_key}_{mode}_") as workdir:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        env["QEFF_HOME"] = str(workdir / "qeff_home")
        env["QEFF_PROFILE_EXPORT_MEMORY"] = "1"
        env["QEFF_PROFILE_MEMORY_INTERVAL_SEC"] = "0.02"
        env["QEFF_EXPORT_ONLY"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        lowmem_env = mode_env_value(mode)
        if lowmem_env is None:
            env.pop("QEFF_LOW_MEMORY_ONNX_EXPORT", None)
        else:
            env["QEFF_LOW_MEMORY_ONNX_EXPORT"] = lowmem_env
        export_dir = workdir / "export"
        cmd = [python_exe, "-c", EXPORT_SNIPPET, model_key, spec["id"], spec["task"], str(export_dir)]
        proc = subprocess.run(cmd, cwd=repo_root, env=env, text=True, capture_output=True)
        parsed = parse_mem_profile(proc.stderr)
        payload = None
        for line in proc.stdout.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{") and "onnx_path" in line:
                payload = json.loads(line)
                break
        onnx_file_size = None
        data_file_size = None
        if payload:
            onnx_path = Path(payload["onnx_path"])
            if onnx_path.exists():
                onnx_file_size = onnx_path.stat().st_size
            data_path = onnx_path.with_name(onnx_path.name + ".data")
            if data_path.exists():
                data_file_size = data_path.stat().st_size
        result = {
            "model_key": model_key,
            "model_id": spec["id"],
            "task": spec["task"],
            "mode": mode,
            "returncode": proc.returncode,
            "export_wall_sec": payload.get("export_wall_sec") if payload else None,
            "onnx_file_mib": mib(onnx_file_size),
            "data_file_mib": mib(data_file_size),
            **parsed,
        }
        if keep_logs:
            log_dir = repo_root / "docs" / "memory_profile" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{model_key}_{mode}.stderr.log").write_text(proc.stderr)
            (log_dir / f"{model_key}_{mode}.stdout.log").write_text(proc.stdout)
        if proc.returncode != 0:
            result["error_tail"] = proc.stderr[-4000:]
        del proc
        gc.collect()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--modes", nargs="+", default=["legacy", "after"])
    parser.add_argument("--out", default="docs/memory_profile/onnx_export_memory_results.json")
    parser.add_argument("--keep-logs", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results = []
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        for mode in args.modes:
            print(f"[bench] {model_key} {mode}", flush=True)
            result = run_one(model_key, spec, mode, sys.executable, repo_root, args.keep_logs)
            results.append(result)
            summary = {
                k: result.get(k)
                for k in [
                    "model_key",
                    "task",
                    "mode",
                    "returncode",
                    "export_wall_sec",
                    "sampled_peak_rss_mib",
                    "peak_hwm_mib",
                ]
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
            if result["returncode"] != 0 and not args.continue_on_error:
                print(result.get("error_tail", ""), file=sys.stderr)
                raise SystemExit(result["returncode"])
    out = repo_root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
