# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Optional

import onnx_ir as ir
from torch import nn

from QEfficient.exporter.weight_free.weight_spec import (
    ExternalDataFile,
    TiedWeightAlias,
    WeightSpec,
    WeightSpecInput,
    WeightSpecLocation,
)
from QEfficient.transformers.embeddings.embedding_utils import PooledModel
from QEfficient.utils.checkpoint_utils import checkpoint_root, load_checkpoint_index, resolve_checkpoint_files

_MOE_WEIGHT_LEGACY_SUFFIXES = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj_t",
    "gate_bias": "gate_proj_bias",
    "up_bias": "up_proj_bias",
    "down_bias": "down_proj_bias",
}


_COMPUTED_INITIALIZER_NAMES = {
    "cos_cached",
    "sin_cached",
    "inv_freq",
    "original_inv_freq",
    "embed_positions",
    "embed_scale",
}


def _collect_tied_weights(model: nn.Module) -> list[TiedWeightAlias]:
    """Return aliases for tied weights, keyed by the model's own tied-weights contract.

    Uses ``get_expanded_tied_weights_keys`` instead of comparing live module identity
    (``get_input_embeddings()``/``get_output_embeddings()`` against ``named_modules()``)
    so this stays correct even if a module was rebuilt/replaced since the tie was
    established — the mapping comes from ``model._tied_weights_keys``, not from
    whatever object graph happens to exist at export time.
    """
    get_expanded_tied_weights_keys = getattr(model, "get_expanded_tied_weights_keys", None)
    if get_expanded_tied_weights_keys is None:
        return []

    tied_mapping = get_expanded_tied_weights_keys(all_submodels=True)
    return [TiedWeightAlias(alias=alias, canonical=canonical) for alias, canonical in tied_mapping.items()]


def _moe_weight_aliases(name: str) -> List[str]:
    """Return equivalent checkpoint aliases for shared MoEWeights parameters."""
    aliases = []
    canonical = name
    if ".experts.moe_weights." in name:
        canonical = name.replace(".experts.moe_weights.", ".moe_weights.", 1)
        aliases.append(canonical)
    elif ".moe_weights." in name:
        aliases.append(name.replace(".moe_weights.", ".experts.moe_weights.", 1))

    prefix, separator, suffix = canonical.rpartition(".moe_weights.")
    if separator and suffix in _MOE_WEIGHT_LEGACY_SUFFIXES:
        aliases.append(f"{prefix}.experts.{_MOE_WEIGHT_LEGACY_SUFFIXES[suffix]}")

    return aliases


def _find_checkpoint_key(candidates: List[str], checkpoint_index: Dict[str, str], onnx_name: str) -> Optional[str]:
    """Return the unique matching checkpoint key, or fail on ambiguous matches."""
    seen = set()
    matches = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in checkpoint_index:
            matches.append(candidate)
        for alias in _moe_weight_aliases(candidate):
            if alias in seen:
                continue
            seen.add(alias)
            if alias in checkpoint_index:
                matches.append(alias)
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous checkpoint key for ONNX initializer '{onnx_name}': matched {matches}. "
            "Checkpoint transforms must produce unambiguous keys."
        )
    return matches[0] if matches else None


def _is_computed_initializer(name: str) -> bool:
    """Return True for generated constants that are not stored in HF checkpoints."""
    return name.rsplit(".", 1)[-1] in _COMPUTED_INITIALIZER_NAMES


def find_checkpoint_key(
    onnx_name: str,
    checkpoint_index: Dict[str, str],
    backbone: nn.Module,
) -> Optional[str]:
    """Resolve an ONNX initializer name to its safetensors checkpoint key.

    Most weights match directly. The fallback rules cover wrapper prefixes,
    task-head/base-model checkpoint differences, and known HF/QEff MoE naming
    differences without putting those details in the export orchestration path.
    TODO(wf): Make this explicit model/layout mapping we should always know what key to expect in what case.
    """
    candidates = [onnx_name]
    stripped = onnx_name.removeprefix("base_model.")
    candidates.append(stripped)

    prefix = getattr(backbone, "base_model_prefix", "")
    if prefix:
        candidates.append(f"{prefix}.{stripped}")

    if prefix and stripped.startswith(f"{prefix}."):
        candidates.append(stripped[len(f"{prefix}.") :])

    if ".mlp." in stripped:
        candidates.append(stripped.replace(".mlp.", ".block_sparse_moe."))

    if stripped.endswith(".mlp.gate.weight"):
        candidates.append(stripped[: -len(".gate.weight")] + ".router.weight")

    if stripped.endswith(".mlp.router.weight"):
        candidates.append(stripped[: -len(".router.weight")] + ".gate.weight")

    return _find_checkpoint_key(candidates, checkpoint_index, onnx_name)


def promote_initializers_and_build_spec(onnx_program, model_ref: str, model_name: str, qeff_model) -> WeightSpec:
    """Promote ONNX initializers to graph inputs and create the weight spec.

    Parameters
    ----------
    onnx_program
        Dynamo ONNX export program whose graph initializers should be promoted.
    model_ref : str
        Checkpoint directory or model reference used to resolve external weights.
    model_name : str
        Name stored in the emitted weight spec.
    qeff_model
        QEfficient model wrapper whose parameters and buffers define promotable weights.

    Returns
    -------
    WeightSpec
        Specification mapping promoted ONNX inputs to checkpoint tensor locations.
    """
    model_ir = onnx_program.model
    parameter_names = {name for name, _ in qeff_model.model.named_parameters()}
    buffer_names = {name for name, _ in qeff_model.model.named_buffers()}
    model_names = parameter_names | buffer_names
    tied_weight_map = {entry.alias: entry.canonical for entry in _collect_tied_weights(qeff_model.model)}
    # named_parameters()/named_buffers() dedup tied tensors by identity, so a tied alias
    # (e.g. lm_head.weight when tie_word_embeddings=True) is absent from model_names even
    # though torch.export still emits a distinct ONNX initializer for it. Add tied aliases
    # explicitly so they aren't skipped below and reach the tied_weight_map redirect.
    model_names.update(tied_weight_map.keys())
    checkpoint_files = resolve_checkpoint_files(model_ref)
    root = checkpoint_root(model_ref, checkpoint_files)
    checkpoint_index = load_checkpoint_index(checkpoint_files)
    relative_checkpoint_files = [
        ExternalDataFile(
            path=str(Path(checkpoint_file).relative_to(root)) if root is not None else Path(checkpoint_file).name,
            format="safetensors",
        )
        for checkpoint_file in checkpoint_files
    ]
    backbone = qeff_model.model.base_model if isinstance(qeff_model.model, PooledModel) else qeff_model.model
    promoted_inputs: List[WeightSpecInput] = []
    spec_input_names = set()

    for name, init_value in list(model_ir.graph.initializers.items()):
        if name not in model_names:
            continue

        onnx_name = tied_weight_map.get(name, name)
        checkpoint_key = find_checkpoint_key(onnx_name, checkpoint_index, backbone)
        if checkpoint_key is None:
            if _is_computed_initializer(onnx_name):
                continue
            raise ValueError(
                f"Could not resolve model initializer '{name}' to a safetensors checkpoint key "
                f"(resolved name: '{onnx_name}', model: '{model_ref}'). "
                "Only explicitly classified computed initializers may remain embedded in the ONNX model."
            )

        checkpoint_file = checkpoint_index[checkpoint_key]
        model_ir.graph.inputs.append(
            ir.Value(
                name=name,
                shape=init_value.shape,
                type=ir.TensorType(init_value.dtype),
            )
        )
        del model_ir.graph.initializers[name]
        promoted_inputs.append(
            WeightSpecInput(
                name=name,
                location=WeightSpecLocation(file=checkpoint_files.index(checkpoint_file), key=checkpoint_key),
            )
        )
        spec_input_names.add(name)

    for value in model_ir.graph.inputs:
        name = value.name
        if name in spec_input_names or name not in model_names:
            continue

        onnx_name = tied_weight_map.get(name, name)
        checkpoint_key = find_checkpoint_key(onnx_name, checkpoint_index, backbone)
        if checkpoint_key is None:
            if _is_computed_initializer(onnx_name):
                continue
            raise ValueError(
                f"Could not resolve model graph input '{name}' to a safetensors checkpoint key "
                f"(resolved name: '{onnx_name}', model: '{model_ref}')."
            )

        checkpoint_file = checkpoint_index[checkpoint_key]
        promoted_inputs.append(
            WeightSpecInput(
                name=name,
                location=WeightSpecLocation(file=checkpoint_files.index(checkpoint_file), key=checkpoint_key),
            )
        )
        spec_input_names.add(name)

    return WeightSpec(
        model_name=model_name,
        model_id=model_ref,
        files=relative_checkpoint_files,
        inputs=promoted_inputs,
    )
