# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Adapter registry for model-specific MoE variation points.

The shared MoE kernels need a small amount of model-specific knowledge: where the
expert weights live, how routing is computed, whether router logits are returned,
and whether blocked prefill is supported. Keeping that contract here lets the
transform pipeline own canonical weight conversion without growing one-off logic
inside each model file.
"""

from dataclasses import dataclass
from functools import partial
from types import MethodType
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from QEfficient.transformers.moe.flavours import MoEFlavour
from QEfficient.transformers.moe.profiles import MoEProfile, gptoss_clamped_glu_mlp, silu_glu_mlp
from QEfficient.transformers.moe.weights import MoEWeights, build_canonical_expert_weights, stack_expert_linears

WeightBuilder = Callable[[nn.Module], MoEWeights]
RouteFn = Callable[[nn.Module, torch.Tensor], tuple]
ProfileFn = Callable[[nn.Module], MoEProfile]
SharedExpertFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class MoEAdapterSpec:
    """Declarative adapter for a QEff MoE module class."""

    class_names: tuple[str, ...]
    build_weights: Optional[WeightBuilder] = None
    route: Optional[RouteFn] = None
    profile: Optional[ProfileFn] = None
    apply_shared_experts: Optional[SharedExpertFn] = None
    return_router_logits: Optional[bool] = None
    default_flavour: Optional[MoEFlavour] = None
    supports_prefill_blocking: Optional[bool] = None
    supports_static_prefill_chunks: Optional[bool] = None


_REGISTRY: dict[str, MoEAdapterSpec] = {}


def register_moe_adapter(spec: MoEAdapterSpec) -> MoEAdapterSpec:
    for class_name in spec.class_names:
        _REGISTRY[class_name] = spec
    return spec


def get_moe_adapter_spec(module: nn.Module) -> Optional[MoEAdapterSpec]:
    module_class = module.__class__
    return _REGISTRY.get(module_class.__name__) or _REGISTRY.get(
        f"{module_class.__module__}.{module_class.__qualname__}"
    )


def _adapter_build_moe_weights(self: nn.Module) -> MoEWeights:
    return build_moe_weights(self)


def _adapter_get_moe_weights(self: nn.Module) -> MoEWeights:
    if getattr(self, "moe_weights", None) is None:
        build_moe_weights(self)
    return self.moe_weights


def _adapter_route(self: nn.Module, x: torch.Tensor):
    spec = get_moe_adapter_spec(self)
    if spec is None or spec.route is None:
        raise NotImplementedError(f"No MoE route adapter registered for {self.__class__.__name__}")
    return spec.route(self, x)


def _adapter_moe_profile(self: nn.Module) -> MoEProfile:
    spec = get_moe_adapter_spec(self)
    if spec is None or spec.profile is None:
        raise NotImplementedError(f"No MoE profile adapter registered for {self.__class__.__name__}")
    return spec.profile(self)


def _adapter_apply_shared_experts(self: nn.Module, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    spec = get_moe_adapter_spec(self)
    if spec is None or spec.apply_shared_experts is None:
        return out
    return spec.apply_shared_experts(self, out, residual)


def bind_moe_adapter_methods(module: nn.Module) -> bool:
    """Bind registered adapter methods to ``module`` in-place."""

    spec = get_moe_adapter_spec(module)
    if spec is None:
        return False

    if spec.build_weights is not None:
        module.build_moe_weights = MethodType(_adapter_build_moe_weights, module)
        module.get_moe_weights = MethodType(_adapter_get_moe_weights, module)
    if spec.route is not None:
        module.route = MethodType(_adapter_route, module)
    if spec.profile is not None and not isinstance(getattr(type(module), "moe_profile", None), property):
        module.moe_profile = MethodType(_adapter_moe_profile, module)
    if spec.apply_shared_experts is not None:
        module.apply_shared_experts = MethodType(_adapter_apply_shared_experts, module)
    if spec.return_router_logits is not None:
        module._moe_return_router_logits = spec.return_router_logits
    if spec.default_flavour is not None:
        module._moe_flavour = spec.default_flavour
    if spec.supports_prefill_blocking is not None:
        module.supports_moe_prefill_blocking = spec.supports_prefill_blocking
    if spec.supports_static_prefill_chunks is not None:
        module.supports_static_moe_prefill_chunks = spec.supports_static_prefill_chunks
    return True


def build_moe_weights(module: nn.Module) -> MoEWeights:
    """Build canonical MoE weights through the registered adapter."""

    spec = get_moe_adapter_spec(module)
    if spec is None or spec.build_weights is None:
        raise NotImplementedError(f"No MoE weight adapter registered for {module.__class__.__name__}")
    return spec.build_weights(module)


def _alias_parameter(module: nn.Module, name: str, tensor: Optional[torch.Tensor]) -> None:
    if tensor is None:
        return
    # Temporary compatibility aliases for legacy forwards/tests that still read
    # the split expert parameters after the canonical MoEWeights are created.
    setattr(module, name, nn.Parameter(tensor, requires_grad=False))


def _qwen_style_expert_weights(module: nn.Module) -> MoEWeights:
    if getattr(module, "moe_weights", None) is not None:
        return module.moe_weights
    module.moe_weights = build_canonical_expert_weights(
        gate_up=module.gate_up_proj,
        down=module.down_proj,
        fused=True,
        fused_split_dim=1,
        transpose_gate_up=True,
        transpose_down=True,
    )
    _alias_parameter(module, "gate_proj", module.moe_weights.gate)
    _alias_parameter(module, "up_proj", module.moe_weights.up)
    _alias_parameter(module, "down_proj_t", module.moe_weights.down)
    return module.moe_weights


def _gpt_oss_expert_weights(module: nn.Module) -> MoEWeights:
    if getattr(module, "moe_weights", None) is not None:
        return module.moe_weights
    module.moe_weights = build_canonical_expert_weights(
        gate_up=module.gate_up_proj,
        down=module.down_proj,
        gate_up_bias=module.gate_up_proj_bias,
        down_bias=module.down_proj_bias,
        fused=True,
        fused_split_dim=2,
        interleaved=True,
        transpose_gate_up=False,
        transpose_down=False,
    )
    _alias_parameter(module, "gate_proj", module.moe_weights.gate)
    _alias_parameter(module, "up_proj", module.moe_weights.up)
    _alias_parameter(module, "gate_proj_bias", module.moe_weights.gate_bias)
    _alias_parameter(module, "up_proj_bias", module.moe_weights.up_bias)
    return module.moe_weights


def _llama4_expert_weights(module: nn.Module) -> MoEWeights:
    if getattr(module, "moe_weights", None) is not None:
        return module.moe_weights
    module.moe_weights = build_canonical_expert_weights(
        gate_up=module.gate_up_proj,
        down=module.down_proj,
        fused=True,
        fused_split_dim=2,
        transpose_gate_up=False,
        transpose_down=False,
    )
    _alias_parameter(module, "gate_proj", module.moe_weights.gate)
    _alias_parameter(module, "up_proj", module.moe_weights.up)
    return module.moe_weights


def _weights_from_experts(module: nn.Module) -> MoEWeights:
    if getattr(module, "moe_weights", None) is not None:
        return module.moe_weights
    if getattr(module.experts, "moe_weights", None) is None:
        spec = get_moe_adapter_spec(module.experts)
        if spec is not None and spec.build_weights is not None:
            build_moe_weights(module.experts)
        else:
            module.experts.build_moe_weights()
    module.moe_weights = module.experts.moe_weights
    return module.moe_weights


def _glm4_moe_weights(module: nn.Module) -> MoEWeights:
    if getattr(module, "moe_weights", None) is not None:
        return module.moe_weights
    if hasattr(module.experts, "gate_up_proj"):
        module.moe_weights = build_canonical_expert_weights(
            gate_up=module.experts.gate_up_proj,
            down=module.experts.down_proj,
            fused=True,
            fused_split_dim=1,
            transpose_gate_up=True,
            transpose_down=True,
        )
    else:
        module.moe_weights = MoEWeights(
            gate=stack_expert_linears(module.experts, lambda e: e.gate_proj.weight),
            up=stack_expert_linears(module.experts, lambda e: e.up_proj.weight),
            down=stack_expert_linears(module.experts, lambda e: e.down_proj.weight),
        )
    _alias_parameter(module, "all_gate_proj", module.moe_weights.gate)
    _alias_parameter(module, "all_up_proj", module.moe_weights.up)
    _alias_parameter(module, "all_down_proj", module.moe_weights.down)
    return module.moe_weights


def _mixtral_moe_weights(module: nn.Module) -> MoEWeights:
    if getattr(module, "moe_weights", None) is not None:
        return module.moe_weights
    if hasattr(module.experts, "gate_up_proj"):
        module.moe_weights = build_canonical_expert_weights(
            gate_up=module.experts.gate_up_proj,
            down=module.experts.down_proj,
            fused=True,
            fused_split_dim=1,
            transpose_gate_up=True,
            transpose_down=True,
        )
        module.act_fn = getattr(module.experts, "act_fn", F.silu)
    else:
        module.moe_weights = MoEWeights(
            gate=stack_expert_linears(module.experts, lambda e: e.w1.weight),
            up=stack_expert_linears(module.experts, lambda e: e.w3.weight),
            down=stack_expert_linears(module.experts, lambda e: e.w2.weight),
        )
        module.act_fn = getattr(module.experts[0], "act_fn", F.silu)
    return module.moe_weights


def _granite_moe_weights(module: nn.Module) -> MoEWeights:
    if getattr(module, "moe_weights", None) is not None:
        return module.moe_weights
    module.moe_weights = build_canonical_expert_weights(
        gate_up=module.input_linear.weight,
        down=module.output_linear.weight,
        fused=True,
        fused_split_dim=1,
        transpose_gate_up=True,
        transpose_down=True,
    )
    return module.moe_weights


def _qwen_profile(module: nn.Module) -> MoEProfile:
    return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=getattr(module.experts, "act_fn", F.silu)))


def _qwen3_route(module: nn.Module, x: torch.Tensor):
    router_logits, top_w, top_i = module.gate(x)
    if getattr(module, "norm_topk_prob", False):
        top_w = top_w / top_w.sum(-1, keepdim=True)
    return (top_i, top_w.to(x.dtype)), router_logits


def _qwen35_route(module: nn.Module, x: torch.Tensor):
    _, top_w, top_i = module.gate(x)
    return (top_i, top_w.to(x.dtype)), None


def _qwen35_shared_experts(module: nn.Module, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    shared = module.shared_expert(residual)
    shared = F.sigmoid(module.shared_expert_gate(residual)) * shared
    return out + shared


def _gpt_oss_profile(module: nn.Module) -> MoEProfile:
    return MoEProfile(
        expert_mlp=partial(gptoss_clamped_glu_mlp, limit=module.experts.limit, alpha=module.experts.alpha),
        has_bias=True,
    )


def _gpt_oss_route(module: nn.Module, x: torch.Tensor):
    router_logits = F.linear(x, module.router.weight, module.router.bias)
    top_w, top_i = torch.topk(router_logits, module.router.top_k, dim=-1)
    top_w = torch.nn.functional.softmax(top_w, dim=1, dtype=top_w.dtype)
    return (top_i, top_w), router_logits


def _glm4_profile(module: nn.Module) -> MoEProfile:
    return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=module.act_fn))


def _glm4_route(module: nn.Module, x: torch.Tensor):
    router_output = module.gate(x)
    if isinstance(router_output, tuple):
        topk_indices, topk_weights = router_output
    else:
        topk_indices, topk_weights = module.route_tokens_to_experts(router_output)
    return (topk_indices, topk_weights), None


def _glm4_shared_experts(module: nn.Module, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return out + module.shared_experts(residual.view(out.shape[0], -1)).view_as(out)


def _mixtral_profile(module: nn.Module) -> MoEProfile:
    act_fn = getattr(module, "act_fn", None)
    if act_fn is None:
        if hasattr(module.experts, "act_fn"):
            act_fn = module.experts.act_fn
        else:
            act_fn = getattr(module.experts[0], "act_fn", F.silu)
    return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=act_fn))


def _mixtral_route(module: nn.Module, x: torch.Tensor):
    gate_dtype = getattr(getattr(module.gate, "weight", None), "dtype", x.dtype)
    gate_out = module.gate(x.to(gate_dtype))
    if isinstance(gate_out, tuple) and len(gate_out) >= 3:
        router_logits, routing_weights, selected_experts = gate_out[0], gate_out[1], gate_out[2]
    else:
        router_logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, module.top_k, dim=-1)
        routing_weights /= torch.einsum("bi->b", routing_weights)[:, None]
        routing_weights = routing_weights.to(x.dtype)
    return (selected_experts, routing_weights), router_logits


def _granite_profile(module: nn.Module) -> MoEProfile:
    return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=module.activation))


def _granite_route(module: nn.Module, x: torch.Tensor):
    topk_gates, expert_mask, router_logits, _ = module.router(x)
    routing_weights = torch.einsum("bke,bk->be", expert_mask.permute(2, 1, 0).to(topk_gates.dtype), topk_gates)
    return routing_weights.to(x.dtype), router_logits


def _llama4_profile(module: nn.Module) -> MoEProfile:
    # Llama4's forward pre-scales expert inputs by sigmoid(top_w), so it remains bespoke.
    return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=module.experts.act_fn), scale_mode="pre")


_PREFILL_QWEN3_CLASSES = ("QEffPrefillChunkedQwen3MoeSparseMoeBlock",)
_PREFILL_QWEN3_VL_CLASSES = ("QEffPrefillChunkedQwen3VLMoeTextSparseMoeBlock",)
_PREFILL_QWEN35_CLASSES = ("QEffPrefillChunkedQwen3_5MoeSparseMoeBlock",)
_PREFILL_GPT_OSS_CLASSES = ("QEffPrefillOnlyChunkedGptOssMLP",)
_PREFILL_GLM4_CLASSES = ("QEffPrefillChunkedGlm4MoeMoE",)


register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffQwen3MoeExperts", "QEffQwen3VLMoeTextExperts", "QEffQwen3_5MoeExperts"),
        build_weights=_qwen_style_expert_weights,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffGptOssExperts",),
        build_weights=_gpt_oss_expert_weights,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffLlama4TextExperts",),
        build_weights=_llama4_expert_weights,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffQwen3MoeSparseMoeBlock", *_PREFILL_QWEN3_CLASSES),
        build_weights=_weights_from_experts,
        route=_qwen3_route,
        profile=_qwen_profile,
        return_router_logits=True,
        supports_prefill_blocking=False,
        supports_static_prefill_chunks=False,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=_PREFILL_QWEN3_CLASSES,
        build_weights=_weights_from_experts,
        route=_qwen3_route,
        profile=_qwen_profile,
        return_router_logits=True,
        supports_prefill_blocking=True,
        supports_static_prefill_chunks=True,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffQwen3VLMoeTextSparseMoeBlock", *_PREFILL_QWEN3_VL_CLASSES),
        build_weights=_weights_from_experts,
        route=_qwen3_route,
        profile=_qwen_profile,
        return_router_logits=True,
        supports_prefill_blocking=False,
        supports_static_prefill_chunks=False,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=_PREFILL_QWEN3_VL_CLASSES,
        build_weights=_weights_from_experts,
        route=_qwen3_route,
        profile=_qwen_profile,
        return_router_logits=True,
        supports_prefill_blocking=True,
        supports_static_prefill_chunks=True,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffQwen3_5MoeSparseMoeBlock", *_PREFILL_QWEN35_CLASSES),
        build_weights=_weights_from_experts,
        route=_qwen35_route,
        profile=_qwen_profile,
        apply_shared_experts=_qwen35_shared_experts,
        return_router_logits=False,
        supports_prefill_blocking=False,
        supports_static_prefill_chunks=False,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=_PREFILL_QWEN35_CLASSES,
        build_weights=_weights_from_experts,
        route=_qwen35_route,
        profile=_qwen_profile,
        apply_shared_experts=_qwen35_shared_experts,
        return_router_logits=False,
        supports_prefill_blocking=True,
        supports_static_prefill_chunks=True,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffPrefillOnlyGptOssMLP", *_PREFILL_GPT_OSS_CLASSES),
        build_weights=_weights_from_experts,
        route=_gpt_oss_route,
        profile=_gpt_oss_profile,
        return_router_logits=True,
        supports_prefill_blocking=False,
        supports_static_prefill_chunks=False,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=_PREFILL_GPT_OSS_CLASSES,
        build_weights=_weights_from_experts,
        route=_gpt_oss_route,
        profile=_gpt_oss_profile,
        return_router_logits=True,
        supports_prefill_blocking=True,
        supports_static_prefill_chunks=True,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffGlm4MoeMoE", *_PREFILL_GLM4_CLASSES),
        build_weights=_glm4_moe_weights,
        route=_glm4_route,
        profile=_glm4_profile,
        apply_shared_experts=_glm4_shared_experts,
        supports_prefill_blocking=False,
        supports_static_prefill_chunks=False,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=_PREFILL_GLM4_CLASSES,
        build_weights=_glm4_moe_weights,
        route=_glm4_route,
        profile=_glm4_profile,
        apply_shared_experts=_glm4_shared_experts,
        supports_prefill_blocking=True,
        supports_static_prefill_chunks=True,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffLlama4TextMoe",),
        build_weights=_weights_from_experts,
        profile=_llama4_profile,
        return_router_logits=True,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffMixtralSparseMoeBlock",),
        build_weights=_mixtral_moe_weights,
        route=_mixtral_route,
        profile=_mixtral_profile,
        return_router_logits=True,
    )
)
register_moe_adapter(
    MoEAdapterSpec(
        class_names=("QEffGraniteMoeMoE",),
        build_weights=_granite_moe_weights,
        route=_granite_route,
        profile=_granite_profile,
        return_router_logits=True,
        default_flavour=MoEFlavour.SIMPLE_LOOP,
    )
)
