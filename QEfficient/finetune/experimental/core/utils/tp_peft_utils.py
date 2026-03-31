import torch
from peft import get_peft_model
from transformers.integrations.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    distribute_model,
    replace_layer_number_by_wildcard,
)


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters, all params and percentage of trainablke params.
    Args:
        model: The PyTorch model.
    """
    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(
        f"Trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )


def is_rowwise_parallel(param: torch.distributed.tensor.DTensor) -> bool:
    """Check if a DTensor is row-wise parallel."""
    if not isinstance(param, torch.distributed.tensor.DTensor):
        return False
    placements = param.placements
    if len(placements) != 1:
        # Till now only TP is applied. If DP is also applied, then the placements might be of len 2.
        return False
    tp_placement = placements[0]
    return tp_placement.is_shard() and tp_placement.dim == 0  # Row-wise sharding


def is_colwise_parallel(param: torch.distributed.tensor.DTensor) -> bool:
    """Check if a DTensor is column-wise parallel."""
    if not isinstance(param, torch.distributed.tensor.DTensor):
        return False
    placements = param.placements
    if len(placements) != 1:
        # Till now only TP is applied. If DP is also applied, then the placements might be of len 2.
        return False
    tp_placement = placements[0]
    return tp_placement.is_shard() and tp_placement.dim == 1  # Column-wise sharding


def update_peft_tp_plan(model):
    # If original layer has colwise then Lora-A --> colwise and Lora-B --> rowwise
    # If original layer has rowwise then Lora-A --> rowwise and Lora-B --> colwise
    peft_tp_plan = {}
    for name, schema in model.tp_plan.items():
        lora_a_name = "base_model.model." + name + ".lora_A.default"
        lora_b_name = "base_model.model." + name + ".lora_B.default"
        if schema == "rowwise":
            peft_tp_plan[lora_a_name] = "rowwise"
            peft_tp_plan[lora_b_name] = "colwise"
        elif schema == "colwise":
            peft_tp_plan[lora_a_name] = "colwise"
            peft_tp_plan[lora_b_name] = "lora_rowwise"
    model.tp_plan.update(peft_tp_plan)


def apply_tp_modification_for_peft(model, tp_mesh=None):
    if tp_mesh is None:
        return

    state_dict = model.state_dict()
    for name, param in state_dict.items():
        if ("lora_A.default" in name) or ("lora_B.default" in name):
            name_for_tp = name.replace(".weight", "")
            name_for_tp = replace_layer_number_by_wildcard(name_for_tp)
            if name_for_tp not in model.tp_plan:
                raise RuntimeError(f"{name_for_tp} not found in model.tp_plan. Please include PEFT layers in tp_plan.")
            lora_plan = model.tp_plan[name_for_tp]

            empty_param = param.clone().to(device="meta")
            tp_layer_cls = ALL_PARALLEL_STYLES[lora_plan].__class__
            tp_layer = tp_layer_cls(
                device_mesh=tp_mesh,
                rank=tp_mesh.get_local_rank(),
                empty_param=empty_param.clone(),
            )
            module_path, _, param_name = name.rpartition(".")
            module_obj = model.get_submodule(module_path)

            # prepare_module_tp does same thing as distribute_model. Hence commented out.
            # Ideal order of opeartion would be prepare_module_tp followed by shard_tensor based on what HF's tensor parallel code.
            # tp_layer.prepare_module_tp(module_obj, tp_mesh)

            # Shard the param
            tp_layer.shard_tensor(param, tensor_idx=None, dtype=empty_param.dtype)
            setattr(getattr(module_obj, param_name), "data", param)


def apply_peft_to_model(model, tp_mesh=None, peft_config=None):
    peft_config = peft_config
    # Add PEFT adapters to the model
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    if tp_mesh is None:
        return

    # Include PEFT parameters in TP plan and update model.tp_plan inplace.
    update_peft_tp_plan(model)

    # Register pre-forward and post-forward hooks to convert input/output DTensor
    # to tensor and vice-versa.
    distribute_model(
        model,
        tp_plan=model.tp_plan,
        distributed_config=None,
        device_mesh=tp_mesh,
        tp_size=tp_mesh.size(),
    )

    # Convert PEFT weights from torch.Tensor to DTensor and apply TP modifications
    apply_tp_modification_for_peft(model, tp_mesh)

    return model
