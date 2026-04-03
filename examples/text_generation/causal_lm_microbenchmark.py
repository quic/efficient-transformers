from QEfficient import QEFFAutoModelForCausalLM


def main():
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        enable_benchmark=True,
    )

    manifest_path = qeff_model.compile(prefill_seq_len=32, ctx_len=128)
    report_path = qeff_model.generate(tokenizer=None, prompts=[])

    print(f"benchmark manifest: {manifest_path}")
    print(f"benchmark report:   {report_path}")


if __name__ == "__main__":
    main()
