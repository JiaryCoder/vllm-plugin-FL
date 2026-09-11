# Copyright (c) 2026 BAAI. All rights reserved.
"""Audited candidates and thin calling-convention adapters for vLLM 0.24.

These entries retain established calling-convention and input checks. Other
in-tree CustomOps are discovered through native.py, without adding class names.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

from .engine import Candidate, ReferenceUnavailable, tensor_support

BASE = "vllm.model_executor.layers."
CUSTOM_CLASSES = {
    BASE + "activation.SiluAndMul": "silu_and_mul",
    BASE + "activation.GeluAndMul": "gelu_and_mul",
    BASE + "layernorm.RMSNorm": "rms_norm",
    BASE + "rotary_embedding.base.RotaryEmbedding": "rotary_embedding",
    BASE + "rotary_embedding.common.ApplyRotaryEmb": "apply_rotary_emb",
    BASE + "rotary_embedding.mrope.MRotaryEmbedding": "mrope",
}
CUSTOM_ALIASES = {
    "vllm_fl.ops.activation.SiluAndMulFL": BASE + "activation.SiluAndMul",
    "vllm_fl.ops.activation.GeluAndMulFL": BASE + "activation.GeluAndMul",
    "vllm_fl.ops.layernorm.RMSNormFL": BASE + "layernorm.RMSNorm",
    "vllm_fl.ops.rotary_embedding.RotaryEmbeddingFL":
        BASE + "rotary_embedding.base.RotaryEmbedding",
}
DISPATCH_OPS = {
    "silu_and_mul", "gelu_and_mul", "rms_norm",
    "rotary_embedding", "dynamic_per_token_quant_int8",
    "topk_softmax", "grouped_topk", "moe_align_block_size", "moe_sum",
    "invoke_fused_moe_triton_kernel", "attention_backend",
}
IR_OPS = {"rms_norm", "fused_add_rms_norm"}


def load(path):
    module, attrs = path.split(":")
    try:
        value = importlib.import_module(module)
        for attr in attrs.split("."):
            value = getattr(value, attr)
        return value
    except (ImportError, AttributeError) as exc:
        raise ReferenceUnavailable(f"upstream {path} unavailable: {exc}") from exc


def native_ir(name):
    if name not in IR_OPS:
        raise ReferenceUnavailable(f"IR op {name} is outside the audited IR inventory")
    op = load("vllm.ir.ops:" + name)
    native = op.impls.get("native")
    if native is None:
        raise ReferenceUnavailable(f"IR {name} has no native provider")
    fn = native.impl_fn
    if getattr(fn, "__module__", "") != "vllm.ir.ops.layernorm":
        raise ReferenceUnavailable(f"IR {name} native provider was replaced")
    return fn


def norm_parameters(obj, residual=None):
    pass_weight = getattr(
        obj, "pass_weight_add" if residual is not None else "pass_weight",
        getattr(obj, "has_weight", True),
    )
    weight = obj.weight if pass_weight else None
    if weight is not None:
        weight = weight.data
    return weight, obj.variance_epsilon, getattr(obj, "variance_size_override", None)


def _upstream_norm(obj, x, residual=None):
    name = "rms_norm" if residual is None else "fused_add_rms_norm"
    fn = native_ir(name)
    weight, eps, variance = norm_parameters(obj, residual)
    if residual is None:
        return fn(x, weight, eps, variance)
    return fn(x, residual, weight, eps, variance)


def _rotary_support(obj, query, key, cos, sin, position_ids, **kwargs):
    reason = tensor_support(query, key, cos, sin)
    if reason:
        return reason
    if query.ndim not in (3, 4):
        return "normalized rotary interface requires a rank-3 or rank-4 query"
    if cos.shape[-1] * 2 != query.shape[-1]:
        return "upstream ApplyRotaryEmb expects a compact half-width cache"
    return None


def _upstream_rotary(fn, obj, query, key, cos, sin, position_ids,
                     rotary_interleaved=False, inplace=True):
    # FL dispatch uses [B,H,S,D] for rank-4 tensors; vLLM's helper uses
    # [B,S,H,D]. A full RotaryEmbedding CustomOp bypasses this adapter.
    cos = cos[position_ids]
    sin = sin[position_ids]
    def apply(x):
        if x is None:
            return None
        if x.ndim == 4:
            x = x.transpose(1, 2)
            return fn(x, cos, sin, not rotary_interleaved).transpose(1, 2)
        return fn(x, cos, sin, not rotary_interleaved)
    return apply(query), apply(key)


def upstream_dispatch(op):
    if op == "silu_and_mul":
        fn = load(BASE + "activation:SiluAndMul.forward_native")
        return Candidate("vllm.native", fn.__qualname__,
                         lambda obj, x: fn(x), tensor_support)
    if op == "gelu_and_mul":
        fn = load(BASE + "activation:GeluAndMul.forward_native")
        def run(obj, x):
            return fn(obj if obj is not None else SimpleNamespace(approximate="none"), x)
        def supports(obj, x):
            from vllm.platforms import current_platform
            if (getattr(obj, "approximate", "none") == "tanh"
                    and current_platform.is_rocm()):
                return "upstream ROCm native changes tanh GELU to exact GELU"
            return tensor_support(x)
        return Candidate("vllm.native", fn.__qualname__, run, supports)
    if op == "rms_norm":
        # Check availability before executing any input-dependent operations.
        native_ir("rms_norm")
        native_ir("fused_add_rms_norm")
        return Candidate("vllm.native", "vllm.ir.ops.layernorm.native",
                         _upstream_norm, tensor_support)
    if op == "rotary_embedding":
        fn = load(BASE + "rotary_embedding.common:ApplyRotaryEmb.forward_static")
        def run(*args, **kwargs):
            return _upstream_rotary(fn, *args, **kwargs)
        def supports(obj, query, key, cos, sin, position_ids,
                     rotary_interleaved=False, inplace=True):
            return _rotary_support(obj, query, key, cos, sin, position_ids)
        return Candidate("vllm.native", fn.__qualname__, run, supports)
    # vLLM 0.24 per_token_quant_int8 launches Triton; it is not a torch candidate.
    return None


def plugin_dispatch(op):
    if op == "attention_backend":
        from .attention import backend_candidate
        return backend_candidate()
    from .moe import dispatch_candidate
    candidate = dispatch_candidate(op)
    if candidate is not None:
        return candidate
    if op not in DISPATCH_OPS:
        return None
    prefix = "vllm_fl.dispatch.backends.reference.impl."
    paths = {
        "silu_and_mul": prefix + "activation:silu_and_mul_torch",
        "gelu_and_mul": prefix + "activation:gelu_and_mul_torch",
        "rms_norm": prefix + "normalization:rms_norm_torch",
        "rotary_embedding": prefix + "rotary:rotary_embedding_torch",
        "dynamic_per_token_quant_int8":
            "vllm_fl.quantization.w8a8.reference:dynamic_per_token_quant_int8",
    }
    if op not in paths:
        return None
    fn = load(paths[op])
    return Candidate("plugin.torch", paths[op], fn, tensor_support)


def custom_identity(obj):
    cls = type(obj)
    return cls.__module__ + "." + cls.__qualname__


def custom_spec(obj):
    identity = custom_identity(obj)
    upstream = CUSTOM_ALIASES.get(identity, identity)
    if upstream not in CUSTOM_CLASSES:
        raise ReferenceUnavailable(
            f"{identity} has no audited class/variant reference"
        )
    return upstream, CUSTOM_CLASSES[upstream]


def has_custom_adapter(obj):
    """Existing adapter restrictions must not be bypassed by auto-discovery."""
    from .custom import CLASSES
    identity = custom_identity(obj)
    return identity in CUSTOM_CLASSES or identity in CUSTOM_ALIASES or identity in CLASSES or identity in {
        BASE + "fused_moe.router.grouped_topk_router.GroupedTopk",
        BASE + "fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod",
        "vllm_fl.ops.fused_moe.layer.UnquantizedFusedMoEMethodFL",
        BASE + "mamba.gdn.qwen_gdn_linear_attn.ChunkGatedDeltaRule",
    }


def upstream_custom(obj):
    if custom_identity(obj) == BASE + "fused_moe.router.grouped_topk_router.GroupedTopk":
        from .moe import grouped_custom
        return grouped_custom(obj)
    from .custom import upstream
    candidate = upstream(obj)
    if candidate is not None:
        return candidate
    path, op = custom_spec(obj)
    if op in {"silu_and_mul", "gelu_and_mul", "rms_norm"}:
        candidate = upstream_dispatch(op)
        def run(*args, **kwargs):
            return candidate.fn(obj, *args, **kwargs)
        def supports(*args, **kwargs):
            if candidate.supports is not None:
                return candidate.supports(obj, *args, **kwargs)
        return Candidate(candidate.source, candidate.implementation, run, supports)
    module, cls = path.rsplit(".", 1)
    klass = load(module + ":" + cls)
    fn = klass.forward_native
    if getattr(fn, "__module__", "") != module:
        raise ReferenceUnavailable(f"{path}.forward_native was replaced")
    def run(*args, **kwargs):
        return fn(obj, *args, **kwargs)
    def supports(*args, **kwargs):
        reason = tensor_support(*args, **kwargs)
        if reason:
            return reason
        if op == "mrope":
            if getattr(obj, "scaling_factor", None) is not None:
                return "YaRN-scaled MRoPE is outside the stage-one audit"
            key = args[2] if len(args) > 2 else kwargs.get("key")
            if key is None:
                return "vLLM MRotaryEmbedding native requires key"
            positions = args[0] if args else kwargs["positions"]
            if positions.ndim == 2 and (
                positions.shape[0] != 3 or not obj.mrope_section
            ):
                return "multimodal MRoPE requires three position axes and sections"
        return None
    return Candidate("vllm.native", path + ".forward_native", run, supports)


def _plugin_full_rotary(obj, positions, query, key=None):
    from vllm_fl.dispatch.backends.reference.impl.rotary import rotary_embedding_torch
    positions = positions.flatten()
    tokens = positions.numel()
    qshape = query.shape
    kshape = None if key is None else key.shape
    q = query.reshape(tokens, -1, obj.head_size)
    k = None if key is None else key.reshape(tokens, -1, obj.head_size)
    cache = obj.cos_sin_cache.to(device=query.device, dtype=query.dtype)
    cos, sin = cache.chunk(2, dim=-1)
    qr, kr = rotary_embedding_torch(
        obj, q[..., :obj.rotary_dim],
        None if k is None else k[..., :obj.rotary_dim],
        cos, sin, positions, not obj.is_neox_style, False,
    )
    import torch
    qo = torch.cat((qr, q[..., obj.rotary_dim:]), dim=-1).reshape(qshape)
    ko = None if k is None else torch.cat(
        (kr, k[..., obj.rotary_dim:]), dim=-1
    ).reshape(kshape)
    return qo, ko


def plugin_custom(obj):
    from .fla import custom
    candidate = custom(obj)
    if candidate is not None:
        return candidate
    if custom_identity(obj) in {
        BASE + "fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod",
        "vllm_fl.ops.fused_moe.layer.UnquantizedFusedMoEMethodFL",
    }:
        from .moe import unquantized_custom
        return unquantized_custom(obj)
    _, op = custom_spec(obj)
    if op == "rotary_embedding":
        return Candidate("plugin.torch", "plugin.rotary_embedding",
                         lambda *a, **k: _plugin_full_rotary(obj, *a, **k),
                         tensor_support)
    candidate = plugin_dispatch(op)
    if candidate is None:
        return None
    def run(*args, **kwargs):
        return candidate.fn(obj, *args, **kwargs)
    return Candidate(candidate.source, candidate.implementation, run, tensor_support)


def plugin_ir(name):
    if name not in IR_OPS:
        return None
    fn = load(
        "vllm_fl.dispatch.backends.reference.impl.normalization:rms_norm_torch"
    )
    def obj(weight, epsilon, variance_size):
        return SimpleNamespace(
            weight=weight, variance_epsilon=epsilon,
            variance_size_override=variance_size, has_weight=weight is not None,
        )
    if name == "rms_norm":
        def run(x, weight, epsilon, variance_size=None):
            return fn(obj(weight, epsilon, variance_size), x)
    else:
        def run(x, x_residual, weight, epsilon, variance_size=None):
            return fn(obj(weight, epsilon, variance_size), x, x_residual)
    return Candidate("plugin.torch", "plugin." + name, run, tensor_support)
