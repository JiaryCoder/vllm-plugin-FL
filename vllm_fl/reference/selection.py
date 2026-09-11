# Copyright (c) 2026 BAAI. All rights reserved.
"""Logical names shared by dispatch, CustomOp, IR and function reference routes.

This is a selection inventory, not a purity certificate. Adding an alias does
not certify its implementation. Decisions apply to calls that reach a hook;
inlined math inside a larger reference cannot be selected independently.
"""
from functools import lru_cache
import re


_custom_groups = {}
_CLASS_PATH = re.compile(r"vllm(?:\.[A-Za-z_][A-Za-z_0-9]*)+\Z")
_IR_NAME = re.compile(r"[a-z_][a-z_0-9]*\Z")


def _module_groups(path):
    groups = set()
    for prefix, group in (
        ("vllm.model_executor.layers.activation.", "activation"),
        ("vllm.model_executor.layers.layernorm.", "normalization"),
        ("vllm.model_executor.layers.rotary_embedding.", "rope"),
        ("vllm.model_executor.layers.fused_moe.", "moe"),
        ("vllm.model_executor.layers.mamba.", "gdn"),
        ("vllm.model_executor.layers.fla.", "gdn"),
    ):
        if path.startswith(prefix):
            groups.add(group)
    return groups


def register_custom(cls):
    """Classify encountered CustomOps lazily; classification does not admit code."""
    identity = cls.__module__ + "." + cls.__qualname__
    groups = {"custom"}
    for base in cls.__mro__:
        groups.update(_module_groups(base.__module__ + "." + base.__qualname__))
    _custom_groups[identity] = frozenset(groups)


@lru_cache(maxsize=1)
def inventory():
    from .adapters import CUSTOM_ALIASES, CUSTOM_CLASSES, DISPATCH_OPS, BASE
    from .custom import CLASSES

    aliases = {name: name for name in DISPATCH_OPS}
    aliases.update(CUSTOM_CLASSES)
    aliases.update(CLASSES)
    aliases.update({name: aliases[target] for name, target in CUSTOM_ALIASES.items()})
    aliases.update({
        # Compatibility names are selection aliases, not native admission rules.
        BASE + "rotary_embedding.llama3_rope.Llama3RotaryEmbedding": "llama3_rope",
        "Llama3RotaryEmbedding": "llama3_rope",
        "attention": "attention_backend",
        "ir.rms_norm": "rms_norm",
        "ir.fused_add_rms_norm": "rms_norm",
        "fused_add_rms_norm": "rms_norm",
        BASE + "fused_moe.router.grouped_topk_router.GroupedTopk": "grouped_topk",
        BASE + "fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod": "fused_experts",
        "vllm_fl.ops.fused_moe.layer.UnquantizedFusedMoEMethodFL": "fused_experts",
        BASE + "mamba.gdn.qwen_gdn_linear_attn.ChunkGatedDeltaRule": "chunk_gated_delta_rule",
    })
    functions = {
        "fused_experts": [
            BASE + "fused_moe.oracle.unquantized:select_unquantized_moe_backend",
            "vllm_fl.ops.fused_moe.fused_moe_utils:select_unquantized_moe_backend_oot",
            BASE + "fused_moe.fused_moe:fused_experts",
            BASE + "fused_moe.experts.triton_moe:TritonExperts.apply",
            "vllm_fl.quantization.w8a8.moe_experts:FlagGemsW8A8Experts.apply",
            "vllm_fl.quantization.w8a8.moe_experts:VllmFunctionalW8A8Experts.apply",
            "vllm_fl.ops.fused_moe.fused_moe:fused_experts_impl",
        ],
        "grouped_topk": [
            BASE + "fused_moe.router.grouped_topk_router:grouped_topk",
            "vllm_fl.ops.fused_moe.router:_fl_grouped_topk",
            "vllm._custom_ops:grouped_topk",
        ],
        "topk_softmax": ["vllm._custom_ops:topk_softmax"],
        "moe_sum": ["vllm._custom_ops:moe_sum"],
        "apply_moe_activation": [
            BASE + "fused_moe.activation:apply_moe_activation",
            "vllm_fl.ops.fused_moe.activation:apply_moe_activation",
        ],
        "w8a8_linear": [
            "vllm.model_executor.kernels.linear:init_int8_linear_kernel",
            "vllm_fl.quantization.w8a8.linear:FLW8A8DynamicLinearKernel.apply_weights",
        ],
        "unpack_uint8b128_int32": [
            "vllm_fl.quantization.w8a8.packed:unpack_uint8b128_int32",
        ],
        "l2norm_fwd": [BASE + "fla.ops.l2norm:l2norm_fwd"],
        "chunk_gated_delta_rule": [BASE + "fla.ops.chunk:chunk_gated_delta_rule"],
        "fused_recurrent_gated_delta_rule": [
            BASE + "fla.ops.fused_recurrent:fused_recurrent_gated_delta_rule",
            BASE + "fla.ops.fused_recurrent:fused_recurrent_gated_delta_rule_fwd",
        ],
        "fused_recurrent_gated_delta_rule_packed_decode": [
            BASE + "fla.ops.fused_recurrent:fused_recurrent_gated_delta_rule_packed_decode",
        ],
        "fused_sigmoid_gating_delta_rule_update": [
            BASE + "fla.ops.fused_sigmoid_gating:fused_sigmoid_gating_delta_rule_update",
        ],
        "fused_post_conv_prep": [BASE + "fla.ops.fused_gdn_prefill_post_conv:fused_post_conv_prep"],
        "fused_gdn_gating": [BASE + "mamba.gdn.qwen_gdn_linear_attn:fused_gdn_gating"],
        "causal_conv1d_fn": [BASE + "mamba.ops.causal_conv1d:causal_conv1d_fn"],
        "causal_conv1d_update": [BASE + "mamba.ops.causal_conv1d:causal_conv1d_update"],
    }
    for name, paths in functions.items():
        aliases.update({path: name for path in paths})
    # Class short names are unambiguous within this explicit inventory.
    for path in (*CUSTOM_CLASSES, *CUSTOM_ALIASES, *CLASSES):
        aliases[path.rsplit(".", 1)[-1]] = aliases[path]
    names = set(aliases.values())
    aliases.update({name: name for name in names})
    groups = {
        "custom": set(),
        "activation": {"silu_and_mul", "gelu_and_mul", "swigluoai_and_mul", "swiglustep_and_mul"},
        "normalization": {"rms_norm", "gemma_rms_norm", "rms_norm_gated"},
        "rope": {"rotary_embedding", "apply_rotary_emb", "mrope", "mrope_interleaved", "ernie45_mrope", "llama3_rope"},
        "moe": {"topk_softmax", "grouped_topk", "moe_align_block_size", "moe_sum",
                "invoke_fused_moe_triton_kernel", "apply_moe_activation", "fused_experts"},
        "w8a8": {"w8a8_linear", "dynamic_per_token_quant_int8", "unpack_uint8b128_int32"},
        "gdn": {"causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                "fused_recurrent_gated_delta_rule", "l2norm_fwd", "fused_post_conv_prep",
                "fused_gdn_gating", "fused_sigmoid_gating_delta_rule_update",
                "fused_recurrent_gated_delta_rule_packed_decode"},
        "all": names,
    }
    return aliases, groups


def normalize_selectors(value, *, allow_none=False):
    if value is None:
        return None if allow_none else frozenset()
    if isinstance(value, str):
        value = [name.strip() for name in value.split(",") if name.strip()]
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("reference include/exclude must be a list or comma-separated string")
    if not value:
        return frozenset()
    aliases, groups = inventory()
    result = set()
    include_all = False
    for token in value:
        if not isinstance(token, str):
            raise ValueError("reference selectors must be strings")
        token = token.strip()
        include_all |= token == "all"
        if token.startswith("@group:") and token[7:] in groups:
            result.add(token)
        elif token in groups:
            result.update(groups[token])
            result.add("@group:" + token)
        elif token in aliases:
            result.add(aliases[token])
        elif token.startswith("custom:") or _CLASS_PATH.fullmatch(token):
            path = token.removeprefix("custom:")
            if not _CLASS_PATH.fullmatch(path):
                raise ValueError("custom selector requires a fully qualified vllm class path")
            result.add(aliases.get(path, path))
        elif token.startswith("ir:") and _IR_NAME.fullmatch(token[3:]):
            path = "ir." + token[3:]
            result.add(aliases.get(path, path))
        else:
            raise ValueError(f"Unknown reference selector {token!r}; supported names: "
                             + ", ".join(sorted(set(aliases.values()) | set(groups))))
    return None if allow_none and include_all else frozenset(result)


def canonical_name(op):
    return inventory()[0].get(op, op)


def selection_reason(op):
    """Return a user-selection reason when this entry must bypass reference."""
    from vllm_fl.dispatch.policy import get_policy
    policy = get_policy()
    name = canonical_name(op)
    groups = _custom_groups.get(op, _module_groups(op))
    def matches(selectors):
        return (name in selectors or op in selectors or "@group:all" in selectors
                or any("@group:" + group in selectors for group in groups))
    if matches(policy.reference_exclude):
        return f"excluded from reference: {name}"
    if policy.reference_include is not None and not matches(policy.reference_include):
        return f"outside reference include list: {name}"
    return None


def selection_active():
    from vllm_fl.dispatch.policy import get_policy
    policy = get_policy()
    return policy.reference_include is not None or bool(policy.reference_exclude)
