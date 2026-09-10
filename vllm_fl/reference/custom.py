# Copyright (c) 2026 BAAI. All rights reserved.
"""Audited vLLM 0.24 CustomOp additions for reference mode."""
from .engine import Candidate, tensor_support

BASE = "vllm.model_executor.layers."
CLASSES = {
    BASE + "layernorm.GemmaRMSNorm": "gemma_rms_norm",
    BASE + "layernorm.RMSNormGated": "rms_norm_gated",
    BASE + "activation.SwigluOAIAndMul": "swigluoai_and_mul",
    BASE + "activation.SwigluStepAndMul": "swiglustep_and_mul",
    BASE + "rotary_embedding.mrope_interleaved.MRotaryEmbeddingInterleaved":
        "mrope_interleaved",
    BASE + "rotary_embedding.ernie45_vl_rope.Ernie4_5_VLRotaryEmbedding":
        "ernie45_mrope",
    BASE + "rotary_embedding.llama3_rope.Llama3RotaryEmbedding": "llama3_rope",
}


def upstream(obj):
    from .adapters import custom_identity, load, native_ir
    path = custom_identity(obj)
    op = CLASSES.get(path)
    if op is None:
        return None
    module, cls = path.rsplit(".", 1)
    method = "forward" if op == "mrope_interleaved" else "forward_native"
    fn = load(module + ":" + cls + "." + method)
    implementation = path + "." + method
    expected_module = module
    if op == "llama3_rope":
        # This exact class only changes the torch inverse-frequency calculation.
        # Its forward/cache accessors are the already audited base implementation.
        # This does not certify arbitrary subclasses or other scaled RoPE types.
        base = load(BASE + "rotary_embedding.base:RotaryEmbedding")
        klass = load(module + ":" + cls)
        for name in ("forward_native", "forward_static", "_match_cos_sin_cache_dtype"):
            if getattr(klass, name) is not getattr(base, name):
                from .engine import ReferenceUnavailable
                raise ReferenceUnavailable(f"{path}.{name} no longer inherits the audited base")
        expected_module = BASE + "rotary_embedding.base"
        implementation = expected_module + ".RotaryEmbedding.forward_native"
    if fn.__module__ != expected_module:
        from .engine import ReferenceUnavailable
        raise ReferenceUnavailable(f"{path}.{method} was replaced")

    if op == "gemma_rms_norm":
        norm = native_ir("rms_norm")
        add_norm = native_ir("fused_add_rms_norm")
        def run(x, residual=None):
            weight = obj.weight.float() + 1.0
            if residual is None:
                return norm(x, weight, obj.variance_epsilon)
            return add_norm(x, residual, weight, obj.variance_epsilon)
    else:
        def run(*args, **kwargs):
            return fn(obj, *args, **kwargs)

    def supports(*args, **kwargs):
        reason = tensor_support(obj, *args, **kwargs)
        if reason:
            return reason
        if op in {"mrope_interleaved", "ernie45_mrope"}:
            positions = args[0] if args else kwargs["positions"]
            if op == "mrope_interleaved":
                if positions.ndim != 2 or positions.shape[0] != len(obj.mrope_section):
                    return "interleaved MRoPE requires one position axis per section"
            else:
                key = args[2] if len(args) > 2 else kwargs.get("key")
                if key is None:
                    return "Ernie MRoPE requires key"
                if positions.ndim == 2 and (
                    positions.shape[0] != 3 or len(obj.mrope_section) != 3
                    or obj.mrope_section[0] != obj.mrope_section[1]
                    or obj.mrope_section[2] <= 0
                ):
                    return "Ernie MRoPE requires three axes, equal H/W, nonempty T"
        return None
    return Candidate("vllm.native", implementation, run, supports)
