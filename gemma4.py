import argparse
import copy
import json

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from QEfficient.base.onnx_transforms import FP16ClipTransform
from QEfficient.utils.run_utils import ApiRunner

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


def resolve_effective_ctx_len(requested_ctx_len: int, prompt_len: int, generation_len: int) -> int:
    return max(requested_ctx_len, prompt_len + generation_len)


def resolve_effective_prefill_seq_len(requested_prefill_seq_len: int, prompt_len: int, use_image: bool) -> int:
    if not use_image:
        return requested_prefill_seq_len
    return max(requested_prefill_seq_len, prompt_len)


def build_messages(system_prompt: str, user_prompt: str, use_image: bool):
    messages = []
    if system_prompt and not use_image:
        messages.append({"role": "system", "content": system_prompt})
    if use_image:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": user_prompt})
    return messages


def prepare_inputs(processor, system_prompt: str, user_prompt: str, image_source: str | None):
    use_image = image_source is not None
    messages = build_messages(system_prompt, user_prompt, use_image)
    if use_image:
        messages[-1]["content"][0]["url"] = image_source
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        rendered_prompt = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
    else:
        rendered_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    return rendered_prompt, inputs


def load_gemma4_text_model(model_name: str):
    tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True).tokenizer
    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = ["input_ids", "attention_mask"]

    full_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
        torch_dtype=torch.float32,
    ).to(torch.float32)
    full_model.eval()

    text_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True).text_config
    text_model = AutoModelForCausalLM.from_config(
        text_config,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(torch.float32)
    text_model.model.load_state_dict(full_model.model.language_model.state_dict())
    text_model.lm_head.load_state_dict(full_model.lm_head.state_dict())
    text_model = text_model.to(torch.float32)
    text_model.eval()
    return tokenizer, text_model


def run_hf_verification(
    model_name: str, inputs, generation_len: int, reference_dtype: str, use_image: bool, hf_model=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if reference_dtype == "fp16" else torch.float32
    if hf_model is not None:
        model = hf_model.to(device=device, dtype=dtype)
    elif use_image:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
        ).to(device=device, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
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


def build_text_ort_tokens(tokenizer, config, rendered_prompt: str, onnx_path: str, generation_len: int):
    prompt_len = int(tokenizer(rendered_prompt, return_tensors="np")["input_ids"].shape[1])
    api_runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=config,
        prompt=[rendered_prompt],
        prompt_len=prompt_len,
        ctx_len=prompt_len + generation_len,
        full_batch_size=None,
    )
    return normalize_generated_ids(api_runner.run_kv_model_on_ort(str(onnx_path)))[:, :generation_len]


def run_ort_session(session: ort.InferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = {x.name for x in session.get_inputs()}
    feed = {k: v for k, v in inputs.items() if k in input_names}
    output_names = [x.name for x in session.get_outputs()]
    outputs = session.run(output_names, feed)
    return dict(zip(output_names, outputs))


def pad_prefill_inputs(inputs, prefill_seq_len: int, pad_token_id: int):
    input_ids = inputs["input_ids"]
    _, input_len = input_ids.shape
    num_chunks = -(input_len // -prefill_seq_len)
    padded_len = num_chunks * prefill_seq_len
    pad_len = padded_len - input_len

    if pad_len == 0:
        return inputs, num_chunks

    padded = dict(inputs)
    padded["input_ids"] = torch.nn.functional.pad(input_ids, (0, pad_len), "constant", pad_token_id)
    padded["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, pad_len), "constant", 0)
    padded["mm_token_type_ids"] = torch.nn.functional.pad(inputs["mm_token_type_ids"], (0, pad_len), "constant", 0)
    return padded, num_chunks


def build_initial_decoder_inputs(model, inputs, vision_embeds, ctx_len: int):
    lang_cfg = model.model.model.language_model.config
    seq_len = inputs["input_ids"].shape[1]
    past_key_values = model.model.get_dummy_pkv_cache(lang_cfg, batch_size=1, seq_len=ctx_len)
    position_ids = torch.where(
        inputs["attention_mask"].bool(),
        inputs["attention_mask"].cumsum(1) - 1,
        torch.full_like(inputs["attention_mask"], -1),
    )

    decoder_inputs = {
        "input_ids": inputs["input_ids"].cpu().numpy(),
        "vision_embeds": vision_embeds.astype(np.float32),
        "position_ids": position_ids.cpu().numpy(),
        "image_idx": np.zeros((1, 1), dtype=np.int64),
        "mm_token_type_ids": inputs["mm_token_type_ids"].cpu().numpy(),
    }
    for i, (key, value) in enumerate(past_key_values):
        decoder_inputs[f"past_key.{i}"] = key.cpu().numpy()
        decoder_inputs[f"past_value.{i}"] = value.cpu().numpy()
    return decoder_inputs


def update_decoder_inputs(inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray], num_layers: int):
    next_token = outputs["logits"].argmax(-1).astype(np.int64)
    next_inputs = {
        "input_ids": next_token,
        "position_ids": np.max(inputs["position_ids"], axis=1, keepdims=True) + 1,
        "image_idx": outputs["image_idx_output"],
        "vision_embeds": outputs.get("vision_embeds_RetainedState", inputs["vision_embeds"]),
        "mm_token_type_ids": np.zeros_like(next_token, dtype=inputs["mm_token_type_ids"].dtype),
    }
    for i in range(num_layers):
        next_inputs[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
        next_inputs[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]
    return next_inputs


def build_vlm_ort_tokens(
    model, onnx_paths, inputs, generation_len: int, ctx_len: int, prefill_seq_len: int, pad_token_id: int
):
    vision_session = ort.InferenceSession(str(onnx_paths[0]))
    decoder_session = ort.InferenceSession(str(onnx_paths[1]))

    vision_outputs = run_ort_session(
        vision_session,
        {
            "pixel_values": inputs["pixel_values"].cpu().numpy(),
            "image_position_ids": inputs["image_position_ids"].cpu().numpy(),
        },
    )
    vision_embeds = vision_outputs["vision_embeds"]

    padded_inputs, num_chunks = pad_prefill_inputs(inputs, prefill_seq_len, pad_token_id)
    decoder_state = build_initial_decoder_inputs(model, padded_inputs, vision_embeds, ctx_len=ctx_len)
    num_layers = model.model.model.language_model.config.num_hidden_layers
    chunk_image_idx = np.array([[0]], dtype=np.int64)
    outputs = None

    for chunk_idx in range(num_chunks):
        chunk_inputs = {
            "input_ids": decoder_state["input_ids"][:, chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len],
            "position_ids": decoder_state["position_ids"][
                :, chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
            ],
            "vision_embeds": decoder_state["vision_embeds"],
            "image_idx": chunk_image_idx,
            "mm_token_type_ids": decoder_state["mm_token_type_ids"][
                :, chunk_idx * prefill_seq_len : (chunk_idx + 1) * prefill_seq_len
            ],
        }
        for i in range(num_layers):
            chunk_inputs[f"past_key.{i}"] = decoder_state[f"past_key.{i}"]
            chunk_inputs[f"past_value.{i}"] = decoder_state[f"past_value.{i}"]

        outputs = run_ort_session(decoder_session, chunk_inputs)
        chunk_image_idx = outputs["image_idx_output"]
        decoder_state["vision_embeds"] = outputs.get("vision_embeds_RetainedState", decoder_state["vision_embeds"])
        for i in range(num_layers):
            decoder_state[f"past_key.{i}"] = outputs[f"past_key.{i}_RetainedState"]
            decoder_state[f"past_value.{i}"] = outputs[f"past_value.{i}_RetainedState"]

    if outputs is None:
        raise RuntimeError("No multimodal prefill chunk was executed.")

    decoder_inputs = {
        "input_ids": outputs["logits"].argmax(-1).astype(np.int64),
        "position_ids": np.max(decoder_state["position_ids"], axis=1, keepdims=True) + 1,
        "vision_embeds": decoder_state["vision_embeds"],
        "image_idx": chunk_image_idx,
        "mm_token_type_ids": np.zeros_like(
            outputs["logits"].argmax(-1), dtype=decoder_state["mm_token_type_ids"].dtype
        ),
    }
    for i in range(num_layers):
        decoder_inputs[f"past_key.{i}"] = decoder_state[f"past_key.{i}"]
        decoder_inputs[f"past_value.{i}"] = decoder_state[f"past_value.{i}"]

    generated = [decoder_inputs["input_ids"].reshape(-1, 1)]
    for _ in range(1, generation_len):
        outputs = run_ort_session(decoder_session, decoder_inputs)
        generated.append(outputs["logits"].argmax(-1).astype(np.int64).reshape(-1, 1))
        decoder_inputs = update_decoder_inputs(decoder_inputs, outputs, num_layers)
    return np.concatenate(generated, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Gemma4 text-only or image+text QAic example.")
    parser.add_argument("--model-name", type=str, default="tiny-random/gemma-4-dense")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--prompt", type=str, default="Hi, Who are you?")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--ctx-len", type=int, default=128)
    parser.add_argument("--generation-len", type=int, default=16)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--device-group", type=parse_device_group, default=None)
    parser.add_argument("--use-onnx-subfunctions", action="store_true")
    parser.add_argument("--vision-use-onnx-subfunctions", action="store_true")
    parser.add_argument("--lang-use-onnx-subfunctions", action="store_true")
    parser.add_argument("--mxfp6-matmul", action="store_true")
    parser.add_argument("--mxint8-kv-cache", action="store_true")
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    parser.add_argument("--mos", type=int, default=None)
    parser.add_argument("--enable-fp16clip", action="store_true")
    parser.add_argument("--enable-npi", action="store_true")
    parser.add_argument("--disable-npi", action="store_true")
    parser.add_argument("--verify-runtime", action="store_true")
    parser.add_argument("--verify-ort", action="store_true")
    parser.add_argument("--reference-dtype", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    use_image = args.image is not None
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer
    rendered_prompt, inputs = prepare_inputs(processor, args.system_prompt, args.prompt, args.image)
    prompt_len = int(inputs["input_ids"].shape[1])
    effective_ctx_len = resolve_effective_ctx_len(args.ctx_len, prompt_len, args.generation_len)
    effective_prefill_seq_len = resolve_effective_prefill_seq_len(args.prefill_seq_len, prompt_len, use_image)
    effective_fp16clip = args.enable_fp16clip or use_image
    effective_convert_to_fp16 = True
    npi_mode = "enabled" if args.enable_npi else "disabled" if args.disable_npi else "auto"
    hf_inputs = {k: v.clone() if hasattr(v, "clone") else v for k, v in inputs.items()}
    hf_ids = None
    ort_ids = None
    onnx_paths = None

    if use_image:
        model = QEFFAutoModelForImageTextToText.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            kv_offload=True,
            skip_vision=False,
            dtype="float32",
        )
        if not effective_fp16clip:
            model.lang_model._onnx_transforms = [
                t for t in model.lang_model._onnx_transforms if t is not FP16ClipTransform
            ]
            if getattr(model, "vision_model", None) is not None:
                model.vision_model._onnx_transforms = [
                    t for t in model.vision_model._onnx_transforms if t is not FP16ClipTransform
                ]
        if args.verify_runtime:
            hf_ids = run_hf_verification(args.model_name, hf_inputs, args.generation_len, args.reference_dtype, True)

        onnx_paths = model.export(
            use_onnx_subfunctions=True,
            vision_use_onnx_subfunctions=False,
            lang_use_onnx_subfunctions=True,
        )
        if args.verify_ort:
            ort_ids = build_vlm_ort_tokens(
                model,
                onnx_paths,
                inputs,
                args.generation_len,
                ctx_len=effective_ctx_len,
                prefill_seq_len=effective_prefill_seq_len,
                pad_token_id=tokenizer.pad_token_id,
            )
        compile_kwargs = dict(
            vision_onnx_path=str(onnx_paths[0]),
            lang_onnx_path=str(onnx_paths[1]),
            prefill_seq_len=effective_prefill_seq_len,
            ctx_len=effective_ctx_len,
            num_devices=(1 if args.device_group is None else len(args.device_group)),
            num_cores=args.num_cores,
            mxfp6_matmul=args.mxfp6_matmul,
            mxint8_kv_cache=args.mxint8_kv_cache,
            aic_enable_depth_first=args.aic_enable_depth_first,
            use_onnx_subfunctions=True,
            vision_use_onnx_subfunctions=False,
            lang_use_onnx_subfunctions=True,
        )
        if npi_mode == "enabled":
            compile_kwargs["node_precision_info"] = True
        elif npi_mode == "disabled":
            compile_kwargs["node_precision_info"] = False
            compile_kwargs["vision_node_precision_info"] = False
        if args.mos is not None:
            compile_kwargs["mos"] = args.mos

        qpc_path = model.compile(**compile_kwargs)
        exec_info = model.generate(inputs=inputs, device_ids=args.device_group, generation_len=args.generation_len)
        effective_use_onnx_subfunctions = True
        effective_vision_subfunctions = False
        effective_lang_subfunctions = True
    else:
        tokenizer, hf_text_model = load_gemma4_text_model(args.model_name)
        model = QEFFAutoModelForCausalLM(copy.deepcopy(hf_text_model), pretrained_model_name_or_path=args.model_name)
        model.model = model.model.to(torch.float32)
        if not effective_fp16clip:
            model._onnx_transforms = [t for t in model._onnx_transforms if t is not FP16ClipTransform]
        effective_text_subfunctions = True
        if args.verify_runtime:
            hf_ids = run_hf_verification(
                args.model_name,
                hf_inputs,
                args.generation_len,
                args.reference_dtype,
                False,
                hf_model=hf_text_model,
            )

        onnx_path = model.export(
            use_onnx_subfunctions=effective_text_subfunctions,
            offload_pt_weights=False,
        )
        if args.verify_ort:
            ort_ids = build_text_ort_tokens(
                tokenizer=tokenizer,
                config=hf_text_model.config,
                rendered_prompt=rendered_prompt,
                onnx_path=str(onnx_path),
                generation_len=args.generation_len,
            )
        compile_kwargs = dict(
            onnx_path=str(onnx_path),
            prefill_seq_len=effective_prefill_seq_len,
            ctx_len=effective_ctx_len,
            num_devices=(1 if args.device_group is None else len(args.device_group)),
            num_cores=args.num_cores,
            mxfp6_matmul=args.mxfp6_matmul,
            mxint8_kv_cache=args.mxint8_kv_cache,
            aic_enable_depth_first=args.aic_enable_depth_first,
            use_onnx_subfunctions=effective_text_subfunctions,
            convert_to_fp16=effective_convert_to_fp16,
        )
        if npi_mode == "enabled":
            compile_kwargs["node_precision_info"] = True
        elif npi_mode == "disabled":
            compile_kwargs["node_precision_info"] = False
        if args.mos is not None:
            compile_kwargs["mos"] = args.mos

        qpc_path = model.compile(**compile_kwargs)
        exec_info = model.generate(
            tokenizer=tokenizer,
            prompts=[rendered_prompt],
            device_id=args.device_group,
            generation_len=args.generation_len,
        )
        onnx_paths = [str(onnx_path)]
        effective_use_onnx_subfunctions = effective_text_subfunctions
        effective_vision_subfunctions = None
        effective_lang_subfunctions = effective_use_onnx_subfunctions

    qeff_ids = normalize_generated_ids(exec_info.generated_ids)[:, : args.generation_len]
    qeff_text = tokenizer.batch_decode(qeff_ids, skip_special_tokens=True)

    print("\nRendered prompt:")
    print(rendered_prompt)
    print("\nCompile settings:")
    print(
        json.dumps(
            {
                "image_mode": use_image,
                "skip_vision": not use_image,
                "kv_offload": use_image,
                "use_onnx_subfunctions": effective_use_onnx_subfunctions,
                "vision_use_onnx_subfunctions": effective_vision_subfunctions,
                "lang_use_onnx_subfunctions": effective_lang_subfunctions,
                "mxfp6_matmul": args.mxfp6_matmul,
                "mxint8_kv_cache": args.mxint8_kv_cache,
                "npi_mode": npi_mode,
                "fp16clip_enabled": effective_fp16clip,
                "convert_to_fp16": effective_convert_to_fp16,
                "prompt_len": prompt_len,
                "requested_prefill_seq_len": args.prefill_seq_len,
                "effective_prefill_seq_len": effective_prefill_seq_len,
                "requested_ctx_len": args.ctx_len,
                "effective_ctx_len": effective_ctx_len,
                "onnx_paths": [str(x) for x in onnx_paths] if onnx_paths is not None else None,
                "qpc_path": [str(x) for x in qpc_path] if isinstance(qpc_path, (list, tuple)) else str(qpc_path),
            },
            indent=2,
        )
    )

    print("\nQEff generated ids:")
    print(qeff_ids.tolist())
    print("QEff generated text:")
    print(qeff_text)

    if args.verify_ort:
        print("\nORT generated ids:")
        print(ort_ids.tolist())
        print("ORT generated text:")
        print(tokenizer.batch_decode(ort_ids, skip_special_tokens=True))

    if args.verify_runtime:
        hf_text = tokenizer.batch_decode(hf_ids, skip_special_tokens=True)
        qeff_prefix = qeff_ids[:, : hf_ids.shape[1]]
        print("\nHF vs QEff parity:")
        print(
            json.dumps(
                {
                    "image_mode": use_image,
                    "rendered_prompt": rendered_prompt,
                    "reference_dtype": args.reference_dtype,
                    "hf_ids": hf_ids.tolist(),
                    "qeff_ids_prefix": qeff_prefix.tolist(),
                    "hf_text": hf_text,
                    "qeff_text_prefix": tokenizer.batch_decode(qeff_prefix, skip_special_tokens=True),
                    "match": bool(np.array_equal(hf_ids, qeff_prefix)),
                },
                indent=2,
            )
        )
        if args.verify_ort:
            print("\nHF vs ORT parity:")
            print(
                json.dumps(
                    {
                        "image_mode": use_image,
                        "rendered_prompt": rendered_prompt,
                        "reference_dtype": args.reference_dtype,
                        "hf_ids": hf_ids.tolist(),
                        "ort_ids": ort_ids.tolist(),
                        "hf_text": hf_text,
                        "ort_text": tokenizer.batch_decode(ort_ids, skip_special_tokens=True),
                        "match": bool(np.array_equal(hf_ids, ort_ids)),
                    },
                    indent=2,
                )
            )
            print("\nORT vs QEff parity:")
            print(
                json.dumps(
                    {
                        "image_mode": use_image,
                        "ort_ids": ort_ids.tolist(),
                        "qeff_ids": qeff_ids.tolist(),
                        "ort_text": tokenizer.batch_decode(ort_ids, skip_special_tokens=True),
                        "qeff_text": qeff_text,
                        "match": bool(np.array_equal(ort_ids, qeff_ids)),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
