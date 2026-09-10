# Copyright (c) 2026 BAAI. All rights reserved.
"""Torch MoE references; optimized launch wrappers are never native candidates."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .engine import Candidate, ReferenceUnavailable, tensor_support
from .functions import candidate, check_signature, patch


def activation(x, name="silu", clamp_limit=None, alpha=1.0, beta=0.0):
    from .adapters import load
    name = getattr(name, "value", name)
    base = "vllm.model_executor.layers.activation:"
    if name == "silu" and clamp_limit is None:
        return load(base + "SiluAndMul.forward_native")(x)
    if name in {"silu", "swigluoai_uninterleave"}:
        if clamp_limit is None:
            raise ValueError("clamped SwiGLU requires clamp_limit")
        return load(base + "SiluAndMulWithClamp.forward_native")(
            SimpleNamespace(swiglu_limit=clamp_limit, alpha=alpha, beta=beta), x)
    if name in {"gelu", "gelu_tanh"}:
        # CPU helper is pure and does not change tanh behavior on ROCm.
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate="tanh" if name.endswith("tanh")
                      else "none") * x[..., d:]
    if name == "swigluoai":
        return load(base + "SwigluOAIAndMul.forward_native")(
            SimpleNamespace(alpha=1.702, limit=7.0), x)
    if name == "swiglustep":
        return load(base + "SwigluStepAndMul.forward_native")(
            SimpleNamespace(limit=7.0), x)
    funcs = {"silu_no_mul": F.silu, "gelu_no_mul": F.gelu,
             "gelu_tanh_no_mul": lambda t: F.gelu(t, approximate="tanh"),
             "relu2_no_mul": lambda t: F.relu(t).square()}
    return funcs[name](x)


ACTIVATIONS = {"silu", "gelu", "gelu_tanh", "swigluoai", "swiglustep",
               "swigluoai_uninterleave", "silu_no_mul", "gelu_no_mul",
               "gelu_tanh_no_mul", "relu2_no_mul"}


def apply_activation(activation, output, input, *, clamp_limit=None,
                     alpha=1.0, beta=0.0):
    output.copy_(globals()["activation"](input, activation, clamp_limit, alpha, beta))
    return output


def topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output,
                 renormalize=False):
    scores = gating_output.float().softmax(-1)
    # CUDA routing kernels break equal-score ties by smaller expert index.
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :topk_weights.shape[1]]
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    topk_weights.copy_(weights)
    topk_indices.copy_(ids)
    m, k = topk_weights.shape
    token_expert_indices.copy_(torch.arange(m, device=scores.device)[:, None]
                              + m * torch.arange(k, device=scores.device)[None, :])
    return topk_weights, topk_indices


def grouped_topk(scores, n_group, topk_group, topk, renormalize,
                 routed_scaling_factor, bias, scoring_func=0):
    scores = scores.float()
    if scoring_func == 1:
        scores = scores.sigmoid()
    elif scoring_func != 0:
        raise ValueError("grouped_topk scoring_func must be 0 (scores) or 1 (logits)")
    selection = scores if bias is None else scores + bias.float()
    grouped = selection.reshape(scores.shape[0], n_group, -1)
    group_scores = (grouped.amax(-1) if bias is None
                    else grouped.topk(2, dim=-1).values.sum(-1))
    group_ids = group_scores.topk(topk_group, dim=-1, sorted=False).indices
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, group_ids, True)
    mask = mask[..., None].expand_as(grouped).reshape_as(scores)
    ids = selection.masked_fill(~mask, -torch.inf).topk(topk, dim=-1, sorted=False).indices
    weights = scores.gather(-1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return (weights * routed_scaling_factor).float(), ids.to(torch.int32)


def moe_align_block_size(topk_ids, block_size, num_experts, expert_map=None,
                         pad_sorted_ids=False, ignore_invalid_experts=False):
    count = topk_ids.numel()
    capacity = count + num_experts * (block_size - 1)
    if pad_sorted_ids:
        capacity = (capacity + block_size - 1) // block_size * block_size
    if count < num_experts:
        capacity = min(count * block_size, capacity)
    sorted_ids = torch.full((capacity,), count, dtype=torch.int32, device=topk_ids.device)
    experts = torch.full(((capacity + block_size - 1) // block_size,), -1,
                         dtype=torch.int32, device=topk_ids.device)
    cursor = 0
    flat = topk_ids.flatten()
    for expert in range(num_experts):
        local = expert if expert_map is None else int(expert_map[expert])
        if ignore_invalid_experts and local < 0:
            continue
        indices = (flat == expert).nonzero().flatten()
        size = indices.numel()
        padded = (size + block_size - 1) // block_size * block_size
        sorted_ids[cursor:cursor + size] = indices.to(torch.int32)
        experts[cursor // block_size:(cursor + padded) // block_size] = local
        cursor += padded
    return sorted_ids, experts, torch.tensor([cursor], dtype=torch.int32,
                                             device=topk_ids.device)


def moe_sum(inp, out):
    out.copy_(inp.float().sum(dim=1).to(out.dtype))


def invoke_fused_moe_triton_kernel(
    A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids,
    num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type,
    use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
    use_int4_w4a16=False, per_channel_quant=False, block_shape=None, B_bias=None,
):
    from .w8a8 import scaled_mm
    flat = C.view(-1, C.shape[-1])
    flat.zero_()
    block = config["BLOCK_SIZE_M"]
    limit = int(num_tokens_post_padded.item())
    for start in range(0, limit, block):
        block_idx = start // block
        expert = int(expert_ids[block_idx])
        if expert < 0:
            continue
        if sorted_token_ids is None:
            rows = torch.tensor([block_idx], device=A.device)
        else:
            rows = sorted_token_ids[start:start + block].long()
            rows = rows[rows < flat.shape[0]]
        if rows.numel() == 0:
            continue
        inputs = rows // top_k
        bias = None if B_bias is None else B_bias[expert]
        if use_int8_w8a8:
            y = scaled_mm(A[inputs], B[expert].t(), A_scale[inputs],
                          B_scale[expert], out_dtype=torch.float32, bias=bias)
        else:
            y = F.linear(A[inputs].float(), B[expert].float(),
                         None if bias is None else bias.float())
        if mul_routed_weight:
            y *= topk_weights.reshape(-1)[rows, None]
        flat[rows] = y.to(C.dtype)


def _gemm_support(**k):
    if any(k[n] for n in ("use_fp8_w8a8", "use_int8_w8a16", "use_int4_w4a16")):
        return "only unquantized and dynamic per-channel W8A8 GEMM are audited"
    if k["block_shape"] is not None:
        return "block quantization is outside the stage-two contract"
    if k["use_int8_w8a8"] and (
        not k["per_channel_quant"] or k["A_scale"] is None or k["B_scale"] is None
    ):
        return "W8A8 requires per-token activation and per-channel weight scales"
    return None


def fused_experts_impl(
    hidden_states, w1, w2, topk_weights, topk_ids, inplace=False, activation="silu",
    apply_router_weight_on_input=False, use_fp8_w8a8=False, use_int8_w8a8=False,
    use_int8_w8a16=False, use_int4_w4a16=False, per_channel_quant=False,
    global_num_experts=-1, expert_map=None, w1_scale=None, w2_scale=None,
    w1_zp=None, w2_zp=None, a1_scale=None, a2_scale=None, block_shape=None,
    w1_bias=None, w2_bias=None,
):
    from vllm_fl.quantization.w8a8.reference import w8a8_linear_reference
    m, topk = topk_ids.shape
    output = torch.zeros((m, topk, w2.shape[1]), dtype=hidden_states.dtype,
                         device=hidden_states.device)
    ids = topk_ids.long()
    if expert_map is not None:
        valid = ids >= 0
        ids = torch.where(valid, expert_map[ids.clamp_min(0)].long(), -1)
    for expert in range(w1.shape[0]):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        x = hidden_states[token]
        if apply_router_weight_on_input:
            x = (x.float() * topk_weights[token, slot, None]).to(x.dtype)
        b1 = None if w1_bias is None else w1_bias[expert]
        b2 = None if w2_bias is None else w2_bias[expert]
        if use_int8_w8a8:
            up = w8a8_linear_reference(x, w1[expert], w1_scale[expert], b1)
        else:
            up = F.linear(x.float(), w1[expert].float(),
                          None if b1 is None else b1.float()).to(x.dtype)
        act = globals()["activation"](up, activation)
        if use_int8_w8a8:
            down = w8a8_linear_reference(act, w2[expert], w2_scale[expert], b2)
        else:
            down = F.linear(act.float(), w2[expert].float(),
                            None if b2 is None else b2.float())
        if not apply_router_weight_on_input:
            down = down.float() * topk_weights[token, slot, None]
        output[token, slot] = down.to(x.dtype)
    result = output.float().sum(1).to(hidden_states.dtype)
    if inplace:
        hidden_states.copy_(result)
        return hidden_states
    return result


def _experts_support(**k):
    if getattr(k["activation"], "value", k["activation"]) not in ACTIVATIONS - {"swigluoai_uninterleave"}:
        return "unaudited expert activation"
    if k["apply_router_weight_on_input"] and k["topk_ids"].shape[1] != 1:
        return "router weight on input requires top_k=1"
    if any(k[n] for n in ("use_fp8_w8a8", "use_int8_w8a16", "use_int4_w4a16")):
        return "only unquantized or dynamic symmetric per-channel INT8 experts"
    if any(k[n] is not None for n in ("w1_zp", "w2_zp", "a1_scale", "a2_scale", "block_shape")):
        return "zero points, static activation scales and block quantization unaudited"
    if k["use_int8_w8a8"]:
        if not k["per_channel_quant"] or k["w1_scale"] is None or k["w2_scale"] is None:
            return "W8A8 requires dynamic per-token and per-channel scales"
    elif not k["w1"].is_floating_point() or not k["w2"].is_floating_point():
        return "unquantized experts require floating-point weights"
    return None


def dispatch_candidate(op):
    functions = {
        "topk_softmax": topk_softmax, "grouped_topk": grouped_topk,
        "moe_align_block_size": moe_align_block_size, "moe_sum": moe_sum,
        "invoke_fused_moe_triton_kernel": invoke_fused_moe_triton_kernel,
    }
    fn = functions.get(op)
    if fn is None:
        return None
    supports = check_signature(fn, _gemm_support) if op == "invoke_fused_moe_triton_kernel" else None
    return candidate(fn, supports=supports)


def install():
    prefix = "vllm.model_executor.layers.fused_moe."
    for path in (
        prefix + "oracle.unquantized:select_unquantized_moe_backend",
        "vllm_fl.ops.fused_moe.fused_moe_utils:select_unquantized_moe_backend_oot",
    ):
        patch(path, (lambda: candidate(reference_unquantized_layout,
                                      source="reference.setup", supports=_layout_support),))
    patch(prefix + "fused_moe:fused_experts",
          (lambda: candidate(functional_experts, supports=check_signature(
              functional_experts, _functional_support)),))
    for target in (
        prefix + "experts.triton_moe:TritonExperts.apply",
        "vllm_fl.quantization.w8a8.moe_experts:FlagGemsW8A8Experts.apply",
        "vllm_fl.quantization.w8a8.moe_experts:VllmFunctionalW8A8Experts.apply",
    ):
        patch(target, (lambda: candidate(modular_experts, supports=check_signature(
            modular_experts, _modular_support)),))
    patch(prefix + "router.grouped_topk_router:grouped_topk",
          (grouped_high_candidate,))
    patch("vllm_fl.ops.fused_moe.router:_fl_grouped_topk", (grouped_high_candidate,))
    patch("vllm._custom_ops:topk_softmax", (lambda: candidate(topk_softmax),))
    patch("vllm._custom_ops:grouped_topk", (lambda: candidate(grouped_topk),))
    patch("vllm._custom_ops:moe_sum", (lambda: candidate(moe_sum),))
    patch(prefix + "activation:apply_moe_activation", (lambda: candidate(apply_activation),))
    patch("vllm_fl.ops.fused_moe.activation:apply_moe_activation",
          (lambda: candidate(apply_activation),))
    patch("vllm_fl.ops.fused_moe.fused_moe:fused_experts_impl",
          (upstream_experts_candidate, lambda: candidate(fused_experts_impl,
                             supports=check_signature(fused_experts_impl, _experts_support)),))

def upstream_experts_candidate():
    import inspect
    import weakref
    from .adapters import load
    from vllm.model_executor.layers.fused_moe import cpu_fused_moe as cpu
    fn = load("vllm.model_executor.layers.fused_moe.cpu_fused_moe:cpu_fused_moe_torch")
    from types import FunctionType
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    # Upstream's SILU table entry constructs CustomOp(compile_native=False).
    # An OOT replacement can have a narrower constructor. Bind the already
    # audited static torch method instead, without mutating upstream globals.
    activations = dict(cpu._CPU_MOE_ACT_FN)
    activations[MoEActivation.SILU] = load(
        "vllm.model_executor.layers.activation:SiluAndMul.forward_native")
    fn = FunctionType(fn.__code__, fn.__globals__ | {"_CPU_MOE_ACT_FN": activations},
                      fn.__name__, fn.__defaults__, fn.__closure__)
    signature = inspect.signature(fused_experts_impl)

    def supports(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        k = bound.arguments
        reason = tensor_support(*args, **kwargs) or _experts_support(**k)
        if reason:
            return reason
        if k["use_int8_w8a8"] or k["expert_map"] is not None:
            return "vLLM CPU torch MoE does not implement quantization or expert maps"
        if getattr(k["activation"], "value", k["activation"]) not in {
            "silu", "gelu", "gelu_tanh", "swigluoai",
        }:
            return "activation missing from vLLM CPU torch MoE"
        ids = k["topk_ids"]
        if ids.numel() == 0:
            return "vLLM CPU torch MoE has no empty-input contract"
        if bool(((ids < 0) | (ids >= k["w1"].shape[0])).any()):
            return "vLLM CPU torch MoE requires valid local expert ids"
        ordered = ids.sort(-1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            return "vLLM CPU torch MoE assumes unique expert assignments"

    def run(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        k = bound.arguments
        class Layer:
            pass
        layer = Layer()
        def linear(weight, bias):
            return lambda x: F.linear(x, weight, bias)
        layer.gate_up_linear = [
            linear(w, None if k["w1_bias"] is None else k["w1_bias"][i])
            for i, w in enumerate(k["w1"])
        ]
        layer.down_linear = [
            linear(w, None if k["w2_bias"] is None else k["w2_bias"][i])
            for i, w in enumerate(k["w2"])
        ]
        x = k["hidden_states"]
        if k["apply_router_weight_on_input"]:
            x = x * k["topk_weights"].to(x.dtype)
        out = torch.empty_like(x)
        cpu._CPU_MOE_LAYER_CACHE[id(layer)] = weakref.ref(layer)
        try:
            fn(id(layer), out, x, k["topk_weights"], k["topk_ids"],
               getattr(k["activation"], "value", k["activation"]), k["w1"].shape[0],
               k["apply_router_weight_on_input"])
        finally:
            cpu._CPU_MOE_LAYER_CACHE.pop(id(layer), None)
        if k["inplace"]:
            k["hidden_states"].copy_(out)
            return k["hidden_states"]
        return out
    return Candidate("vllm.native", fn.__module__ + "." + fn.__name__, run, supports)


def routed_experts(*args, **kwargs):
    from .engine import run_reference
    return run_reference("fused_experts", args, kwargs, (
        upstream_experts_candidate,
        lambda: candidate(fused_experts_impl,
                          supports=check_signature(fused_experts_impl, _experts_support)),
    ))


def _quant_kwargs(quant_config):
    if quant_config is None:
        return {}
    if getattr(quant_config, "ocp_mx_scheme", None) is not None:
        raise ReferenceUnavailable("OCP quantization is outside the stage-two audit")
    if getattr(quant_config, "swiglu_limit", None) is not None:
        raise ReferenceUnavailable("quant-config clamped MoE needs a separate adapter")
    return {name: getattr(quant_config, name) for name in (
        "use_fp8_w8a8", "use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16",
        "w1_scale", "w2_scale", "w1_zp", "w2_zp", "a1_scale", "a2_scale",
        "block_shape", "w1_bias", "w2_bias",
    )} | {"per_channel_quant": quant_config.per_act_token_quant}


def functional_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                       activation="silu", apply_router_weight_on_input=False,
                       global_num_experts=-1, expert_map=None, quant_config=None):
    return routed_experts(
        hidden_states, w1, w2, topk_weights, topk_ids, activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        global_num_experts=global_num_experts, expert_map=expert_map,
        **_quant_kwargs(quant_config),
    )


def _functional_support(**k):
    import inspect
    kwargs = _quant_kwargs(k.pop("quant_config"))
    bound = inspect.signature(fused_experts_impl).bind(**k, **kwargs)
    bound.apply_defaults()
    return _experts_support(**bound.arguments)


def modular_experts(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                    activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                    workspace13, workspace2, expert_tokens_meta,
                    apply_router_weight_on_input):
    # Modular prepare has already applied router weights on input.
    result = functional_experts(
        hidden_states, w1, w2, topk_weights, topk_ids, activation,
        False, global_num_experts, expert_map, self.quant_config,
    ) if not apply_router_weight_on_input else routed_experts(
        hidden_states, w1, w2, torch.ones_like(topk_weights), topk_ids,
        activation=activation, global_num_experts=global_num_experts,
        expert_map=expert_map, **_quant_kwargs(self.quant_config),
    )
    output.copy_(result)


def _modular_support(**k):
    if getattr(k["self"], "_lora_context", None) is not None:
        return "LoRA experts are outside the stage-two audit"
    if k["a1q_scale"] is not None or k["a2_scale"] is not None:
        return "reference experts require unquantized modular inputs"
    return _functional_support(**{key: k[key] for key in (
        "hidden_states", "w1", "w2", "topk_weights", "topk_ids", "activation",
        "global_num_experts", "expert_map", "apply_router_weight_on_input",
    )}, quant_config=k["self"].quant_config)


def grouped_high_candidate():
    from .adapters import load
    fn = load("vllm.model_executor.layers.fused_moe.cpu_fused_moe:grouped_topk")
    return candidate(fn, source="vllm.native")


def grouped_custom(obj):
    base = grouped_high_candidate()
    def run(hidden_states, gating_output, e_score_correction_bias=None):
        return base.fn(hidden_states, gating_output, obj.topk, obj.renormalize,
                       obj.num_expert_group, obj.topk_group, obj.scoring_func,
                       obj.routed_scaling_factor, e_score_correction_bias)
    return Candidate(base.source, base.implementation, run, tensor_support)


def unquantized_custom(obj):
    def supports(layer, x, topk_weights, topk_ids, shared_experts,
                 shared_experts_input):
        moe = obj.moe
        if any(getattr(moe, n) != 1 for n in ("dp_size", "pcp_size", "ep_size", "sp_size")):
            return "whole-layer reference currently requires no MoE dispatch collectives"
        if moe.is_lora_enabled or moe.swiglu_limit is not None:
            return "LoRA/clamped whole-layer MoE are outside this adapter"
        backend = getattr(obj.unquantized_backend, "name", "")
        if backend not in {"TRITON", "OOT"}:
            return f"whole-layer reference requires unshuffled weights, got {backend}"
        if shared_experts is not None:
            impl = getattr(obj.moe_kernel, "impl", None)
            prepare = getattr(impl, "prepare_finalize", None)
            if prepare is None or prepare.supports_async():
                return "asynchronous shared-expert scheduling is outside this adapter"
            # With synchronous prepare/finalize, MoERunner computes shared
            # experts independently; this method returns routed output only.
        import inspect
        bound = inspect.signature(fused_experts_impl).bind(
            x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts, expert_map=layer.expert_map,
            w1_bias=getattr(layer, "w13_bias", None), w2_bias=getattr(layer, "w2_bias", None),
        )
        bound.apply_defaults()
        return tensor_support(x, layer.w13_weight, layer.w2_weight) or _experts_support(**bound.arguments)
    def run(layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        return routed_experts(
            x, layer.w13_weight, layer.w2_weight, topk_weights, topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts, expert_map=layer.expert_map,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )
    return candidate(run, supports=supports)


def reference_unquantized_layout(moe_config):
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    # This backend's loader preserves the ordinary [E,N,K] weight layout.
    # Numerical execution is intercepted at CustomOp and experts.apply.
    return UnquantizedMoeBackend.TRITON, TritonExperts


def _layout_support(moe_config):
    if any(getattr(moe_config, name) != 1 for name in
           ("dp_size", "pcp_size", "ep_size", "sp_size")):
        return "reference whole-layer MoE does not implement dispatch collectives"
    if moe_config.is_lora_enabled or moe_config.swiglu_limit is not None:
        return "LoRA and clamped whole-layer MoE require a separate adapter"
    return None
