# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.finetune.experimental.inference import _build_parser, _to_string_list, main

if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        config_path=args.config_path,
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        prompt_template=args.prompt_template,
        system_prompt=args.system_prompt,
        use_chat_template=args.use_chat_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        stop_strings=_to_string_list(args.stop_strings),
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        auto_class_name=args.auto_class_name,
        attn_implementation=args.attn_implementation,
        use_cache=args.use_cache,
    )
