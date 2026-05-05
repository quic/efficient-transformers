import copy
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from QEfficient import QEFFAutoModelForCausalLM

MODEL_PATH = Path(
    "/home/huggingface_hub/models--moonshotai--Kimi-K2.5/snapshots/54383e83fa343a1331754112fb9e3410c55efa2f"
)
NUM_HIDDEN_LAYERS = 2
LOADED_EXPERT_IDS = (0, 1, 2, 3)
NUM_EXPERTS_PER_TOKEN = 2

EXPERT_KEY_PATTERN = re.compile(
    r"^(language_model\.model\.layers\.\d+\.mlp\.experts\.)(\d+)(\..+)$"
)

def _validate_expert_subset(loaded_expert_ids, num_experts_per_tok, total_experts):
    expert_ids = tuple(loaded_expert_ids)
    if len(expert_ids) != 4:
        raise ValueError(f"Expected exactly 4 routed experts, got {expert_ids!r}.")
    if len(set(expert_ids)) != len(expert_ids):
        raise ValueError(f"Expert ids must be unique, got {expert_ids!r}.")
    invalid_ids = [expert_id for expert_id in expert_ids if expert_id < 0 or expert_id >= total_experts]
    if invalid_ids:
        raise ValueError(
            f"Expert ids {invalid_ids!r} are outside the valid range [0, {total_experts - 1}]."
        )
    if num_experts_per_tok > len(expert_ids):
        raise ValueError(
            f"num_experts_per_tok={num_experts_per_tok} cannot exceed {len(expert_ids)} loaded experts."
        )
    return expert_ids


def _remap_checkpoint_key(checkpoint_key, expert_index_map):
    match = EXPERT_KEY_PATTERN.match(checkpoint_key)
    if not match:
        return checkpoint_key

    original_expert_idx = int(match.group(2))
    remapped_expert_idx = expert_index_map.get(original_expert_idx)
    if remapped_expert_idx is None:
        return None
    return f"{match.group(1)}{remapped_expert_idx}{match.group(3)}"


def _is_routed_gate_weight(checkpoint_key):
    return checkpoint_key.endswith(".mlp.gate.weight")


def _is_routed_gate_bias(checkpoint_key):
    return checkpoint_key.endswith(".mlp.gate.e_score_correction_bias")


def _materialize_subset_checkpoint(
    model_path: Path,
    temp_model_path: Path,
    weight_map,
    allowed_prefixes,
    loaded_expert_ids,
):
    expert_index_map = {expert_id: remapped_idx for remapped_idx, expert_id in enumerate(loaded_expert_ids)}
    selected_by_shard = defaultdict(list)

    for checkpoint_key, shard_name in weight_map.items():
        if not any(checkpoint_key.startswith(prefix) for prefix in allowed_prefixes):
            continue

        remapped_key = _remap_checkpoint_key(checkpoint_key, expert_index_map)
        if remapped_key is None:
            continue
        selected_by_shard[shard_name].append((checkpoint_key, remapped_key))

    if not selected_by_shard:
        raise RuntimeError("No text-only weights were selected from the Kimi K2.5 checkpoint.")

    filtered_weight_map = {}
    subset_shards = []
    for shard_idx, (source_shard_name, shard_entries) in enumerate(sorted(selected_by_shard.items())):
        tensors = {}
        with safe_open(model_path / source_shard_name, framework="pt", device="cpu") as shard_reader:
            for checkpoint_key, remapped_key in shard_entries:
                tensor = shard_reader.get_tensor(checkpoint_key)
                if _is_routed_gate_weight(checkpoint_key):
                    tensor = tensor[list(loaded_expert_ids), :].contiguous()
                elif _is_routed_gate_bias(checkpoint_key):
                    tensor = tensor[list(loaded_expert_ids)].contiguous()
                tensors[remapped_key] = tensor

        subset_shard_name = f"model-subset-{shard_idx:05d}.safetensors"
        save_file(tensors, str(temp_model_path / subset_shard_name))
        subset_shards.append(subset_shard_name)
        filtered_weight_map.update(
            {remapped_key: subset_shard_name for _, remapped_key in shard_entries}
        )

    return filtered_weight_map, subset_shards


def load_text_only_kimi(
    model_path: Path,
    num_hidden_layers: int,
    loaded_expert_ids=LOADED_EXPERT_IDS,
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOKEN,
):
    kimi_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

    # Kimi K2.5 is multimodal, so the text depth must be overridden on text_config.
    text_config = copy.deepcopy(kimi_config.text_config)
    text_config.num_hidden_layers = num_hidden_layers
    loaded_expert_ids = _validate_expert_subset(
        loaded_expert_ids,
        num_experts_per_tok,
        text_config.n_routed_experts,
    )
    text_config.n_routed_experts = len(loaded_expert_ids)
    text_config.num_experts_per_tok = num_experts_per_tok
    text_config.n_group = 1
    text_config.topk_group = 1

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

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_model_path = Path(tmpdir)
        filtered_weight_map, subset_shards = _materialize_subset_checkpoint(
            model_path=model_path,
            temp_model_path=temp_model_path,
            weight_map=weight_map,
            allowed_prefixes=allowed_prefixes,
            loaded_expert_ids=loaded_expert_ids,
        )
        (temp_model_path / "config.json").write_text(text_config.to_json_string(use_diff=False))
        (temp_model_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum(
                            (temp_model_path / shard_name).stat().st_size for shard_name in subset_shards
                        )
                    },
                    "weight_map": filtered_weight_map,
                }
            )
        )

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
    model, tokenizer = load_text_only_kimi(
        MODEL_PATH,
        NUM_HIDDEN_LAYERS,
        loaded_expert_ids=LOADED_EXPERT_IDS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOKEN,
    )
    mla_absorption = {"cache_compressed": True, "absorption": False, "online": False}
    qaic_config = {"mla_absorption": mla_absorption, #"enable_blocking": True, "blocking_mode": "kv",  "num_kv_heads_repeat":4
                   }
    qeff_model = QEFFAutoModelForCausalLM(model, qaic_config=qaic_config)
    TS = 4

    prefill_seq_len = 128
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
        prefill_only=True
    )


if __name__ == "__main__":
    main()
