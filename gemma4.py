import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.base.onnx_transforms import FP16ClipTransform

torch.manual_seed(42)


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


def parse_device_group(device_ids):
    device_ids = device_ids.strip()
    if not device_ids:
        return None
    return [int(x) for x in device_ids.strip("[]").split(",") if x.strip()]


def build_messages(system_prompt: str, user_prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def prepare_inputs(processor, system_prompt: str, user_prompt: str):
    messages = build_messages(system_prompt, user_prompt)
    rendered_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=rendered_prompt, return_tensors="pt")
    return rendered_prompt, inputs


def run_hf_verification(model_name: str, inputs, generation_len: int, reference_dtype: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if reference_dtype == "fp16" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    ).to(device=device, dtype=dtype)
    model.eval()
    model_inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(**model_inputs, max_new_tokens=generation_len, do_sample=False)
    prompt_len = model_inputs["input_ids"].shape[1]
    return outputs[:, prompt_len:].cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Gemma4 processor/chat-template bring-up using QEFFAutoModelForImageTextToText with skip_vision=True."
    )
    parser.add_argument("--model-name", type=str, default="tiny-random/gemma-4-dense")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--prompt", type=str, default="Hi, Who are you?")
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--device-group", type=parse_device_group, default=None)
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    parser.add_argument("--mos", type=int, default=1)
    parser.add_argument("--enable-fp16clip", action="store_true")
    parser.add_argument("--enable-npi", action="store_true")
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--reference-dtype", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    rendered_prompt, inputs = prepare_inputs(processor, args.system_prompt, args.prompt)
    hf_inputs = {k: v.clone() if hasattr(v, "clone") else v for k, v in inputs.items()}

    model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        kv_offload=True,
        torch_dtype=torch.float32,
    )

    if not args.enable_fp16clip:
        model.lang_model._onnx_transforms = [t for t in model.lang_model._onnx_transforms if t is not FP16ClipTransform]
        if getattr(model, "vision_model", None) is not None:
            model.vision_model._onnx_transforms = [
                t for t in model.vision_model._onnx_transforms if t is not FP16ClipTransform
            ]

    compile_kwargs = dict(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        num_devices=(1 if args.device_group is None else len(args.device_group)),
        num_cores=args.num_cores,
        mxfp6_matmul=args.mxfp6_matmul,
        mxint8_kv_cache=args.mxint8_kv_cache,
        aic_enable_depth_first=args.aic_enable_depth_first,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        skip_vision=True,
    )
    if not args.enable_npi:
        compile_kwargs["node_precision_info"] = False
    if args.mos is not None:
        compile_kwargs["mos"] = args.mos

    qpc_path = model.compile(**compile_kwargs)

    print("\nRendered prompt:")
    print(rendered_prompt)
    print("\nCompile settings:")
    print(
        json.dumps(
            {
                "skip_vision": True,
                "kv_offload": True,
                "use_onnx_subfunctions": args.use_onnx_subfunctions,
                "mxfp6_matmul": args.mxfp6_matmul,
                "mxint8_kv_cache": args.mxint8_kv_cache,
                "npi_enabled": args.enable_npi,
                "fp16clip_enabled": args.enable_fp16clip,
                "qpc_path": str(qpc_path),
            },
            indent=2,
        )
    )

    exec_info = model.generate(
        inputs=inputs,
        device_ids=args.device_group,
        generation_len=args.generation_len,
    )
    qeff_ids = normalize_generated_ids(exec_info.generated_ids)[:, : args.generation_len]
    qeff_text = tokenizer.batch_decode(qeff_ids, skip_special_tokens=True)

    print("\nQEff generated ids:")
    print(qeff_ids.tolist())
    print("QEff generated text:")
    print(qeff_text)

    if args.verify_runtime:
        hf_ids = run_hf_verification(args.model_name, hf_inputs, args.generation_len, args.reference_dtype)
        hf_text = tokenizer.batch_decode(hf_ids, skip_special_tokens=True)
        qeff_prefix = qeff_ids[:, : hf_ids.shape[1]]
        payload = {
            "rendered_prompt": rendered_prompt,
            "reference_dtype": args.reference_dtype,
            "hf_ids": hf_ids.tolist(),
            "qeff_ids_prefix": qeff_prefix.tolist(),
            "hf_text": hf_text,
            "qeff_text_prefix": tokenizer.batch_decode(qeff_prefix, skip_special_tokens=True),
            "match": bool(np.array_equal(hf_ids, qeff_prefix)),
        }
        print("\nHF vs QEff parity:")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
