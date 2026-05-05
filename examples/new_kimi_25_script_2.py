import copy
import json
import tempfile
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModelForCausalLM


MODEL_PATH = Path(
    "/home/huggingface_hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"
)
NUM_HIDDEN_LAYERS = 2



def load_text_only_kimi(model_path: Path, num_hidden_layers: int):
    kimi_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

    # Kimi K2.5 is multimodal, so the text depth must be overridden on text_config.
    text_config = copy.deepcopy(kimi_config.text_config)
    text_config.num_hidden_layers = num_hidden_layers

    deepseek_cls = get_class_from_dynamic_module(
        "modeling_deepseek.DeepseekV3ForCausalLM", str(model_path)
    )

    checkpoint_index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = checkpoint_index["weight_map"]

    allowed_prefixes = [
        "language_model.model.embed_tokens.",
        "language_model.model.norm.",
        "language_model.lm_head.",
    ]
    allowed_prefixes.extend(
        f"language_model.model.layers.{layer_idx}." for layer_idx in range(num_hidden_layers)
    )

    required_shards = sorted(
        {
            shard_name
            for checkpoint_key, shard_name in weight_map.items()
            if any(checkpoint_key.startswith(prefix) for prefix in allowed_prefixes)
        }
    )
    filtered_weight_map = {
        checkpoint_key: shard_name
        for checkpoint_key, shard_name in weight_map.items()
        if any(checkpoint_key.startswith(prefix) for prefix in allowed_prefixes)
    }
    if not filtered_weight_map:
        raise RuntimeError("No text-only weights were selected from the Kimi K2.5 checkpoint.")


    with tempfile.TemporaryDirectory() as tmpdir:
        temp_model_path = Path(tmpdir)
        (temp_model_path / "config.json").write_text(text_config.to_json_string(use_diff=False))
        (temp_model_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum((model_path / shard_name).stat().st_size for shard_name in required_shards)
                    },
                    "weight_map": filtered_weight_map,
                }
            )
        )
        for shard_name in required_shards:
            (temp_model_path / shard_name).symlink_to(model_path / shard_name)

        # We are loading a task checkpoint into the base text model, so disable the
        # base/task prefix heuristic and let `key_mapping` strip `language_model.`.
        original_base_model_prefix = deepseek_cls.base_model_prefix
        deepseek_cls.base_model_prefix = ""
        try:
            model, loading_info = deepseek_cls.from_pretrained(
                str(temp_model_path),
                config=text_config,
                local_files_only=True,
                key_mapping={r"^language_model\.": ""},
                output_loading_info=True,
            )
        finally:
            deepseek_cls.base_model_prefix = original_base_model_prefix

    unexpected_keys = loading_info["unexpected_keys"]
    missing_keys = loading_info["missing_keys"]
    mismatched_keys = loading_info["mismatched_keys"]
    if unexpected_keys or missing_keys or mismatched_keys:
        raise RuntimeError(
            "Failed to load the text-only Kimi K2.5 checkpoint slice cleanly. "
            f"missing={missing_keys}, unexpected={unexpected_keys}, mismatched={mismatched_keys}"
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    return model, tokenizer


def main():
    model, tokenizer = load_text_only_kimi(MODEL_PATH, NUM_HIDDEN_LAYERS)
    print(f"Loaded {type(model).__name__} with {model.config.num_hidden_layers} text layers.")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    model, tokenizer = load_text_only_kimi(MODEL_PATH, NUM_HIDDEN_LAYERS)
    import ipdb; ipdb.set_trace()
    mla_absorption = {"cache_compressed": True, "absorption": False, "online": False}
    qaic_config = {"mla_absorption": mla_absorption, "enable_blocking": True, "blocking_mode": "kv",  "num_kv_heads_repeat":4}
    qeff_model = QEFFAutoModelForCausalLM(model, qaic_config=qaic_config)
    TS = 4
    enable_mla = True

    
    prefill_seq_len = 1
    ctx_len = 16 * 1024
    
    qpc_path = qeff_model.compile(
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        # enable_mla=enable_mla,
        # mla_absorption_config=mla_absorption_config,
        mxfp6_matmul=True,
        mxint8_kv_cache=False,
        num_devices=TS,
        num_cores=16,
        use_onnx_subfunctions=True,
        qaic_config=qaic_config,
    )
