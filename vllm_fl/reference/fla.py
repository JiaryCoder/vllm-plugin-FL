# Copyright (c) 2026 BAAI. All rights reserved.
"""GDN references with explicit sequence boundaries and recurrent-cache writes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .engine import Candidate, ReferenceUnavailable, tensor_support
from .functions import candidate, check_signature, patch


def l2norm(x, eps=1e-6, output_dtype=None):
    y = x.float()
    return (y * (y.square().sum(-1, keepdim=True) + eps).rsqrt()).to(output_dtype or x.dtype)


def _sequences(q, cu_seqlens):
    if cu_seqlens is None:
        return [(b, 0, q.shape[1]) for b in range(q.shape[0])]
    if q.shape[0] != 1:
        raise ValueError("variable-length GDN requires batch dimension one")
    offsets = cu_seqlens.tolist()
    if offsets[0] != 0 or offsets[-1] != q.shape[1] or any(
        a > b for a, b in zip(offsets, offsets[1:])
    ):
        raise ValueError("invalid cumulative sequence lengths")
    return [(0, a, b) for a, b in zip(offsets, offsets[1:])]


def _step(q, k, v, g, beta, state, scale, normalize):
    repeat = v.shape[0] // k.shape[0]
    q, k = q.float(), k.float()
    if normalize:
        q, k = l2norm(q), l2norm(k)
    q, k = q.repeat_interleave(repeat, 0) * scale, k.repeat_interleave(repeat, 0)
    state = state * g.float().exp()[:, None, None]
    delta = v.float() - (state * k[:, None, :]).sum(-1)
    beta = beta.float()
    delta = delta * (beta[:, None] if beta.ndim == 1 else beta)
    state = state + delta[:, :, None] * k[:, None, :]
    return (state * q[:, None, :]).sum(-1), state


def chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None,
                           output_final_state=False, cu_seqlens=None,
                           chunk_indices=None, chunk_offsets=None,
                           use_qk_l2norm_in_kernel=False, core_attn_out=None):
    scale = k.shape[-1] ** -.5 if scale is None else scale
    if use_qk_l2norm_in_kernel:
        # Chunk path materializes normalized Q/K in the input dtype.
        q, k = l2norm(q), l2norm(k)
    sequences = _sequences(q, cu_seqlens)
    out = torch.zeros_like(v)
    final = torch.empty((len(sequences), v.shape[2], v.shape[-1], k.shape[-1]),
                        dtype=torch.float32, device=q.device)
    for seq, (batch, begin, end) in enumerate(sequences):
        state = (torch.zeros_like(final[seq]) if initial_state is None
                 else initial_state[seq].float().clone())
        for t in range(begin, end):
            y, state = _step(q[batch, t], k[batch, t], v[batch, t], g[batch, t],
                             beta[batch, t], state, scale, False)
            out[batch, t] = y.to(out.dtype)
        final[seq] = state
    if core_attn_out is not None:
        core_attn_out.view(-1)[:out.numel()].copy_(out.reshape(-1))
    return out, final if output_final_state else None


def fused_recurrent_gated_delta_rule(q, k, v, g, beta=None, scale=None,
                                     initial_state=None, inplace_final_state=True,
                                     cu_seqlens=None, ssm_state_indices=None,
                                     num_accepted_tokens=None,
                                     use_qk_l2norm_in_kernel=False):
    scale = k.shape[-1] ** -.5 if scale is None else scale
    sequences = _sequences(q, cu_seqlens)
    out = torch.zeros_like(v)
    final = initial_state if inplace_final_state else torch.zeros(
        (q.shape[0] * q.shape[1], *initial_state.shape[1:]),
        dtype=initial_state.dtype, device=initial_state.device)
    for seq, (batch, begin, end) in enumerate(sequences):
        if begin == end:
            continue
        def index(t):
            if ssm_state_indices is None:
                return batch * q.shape[1] + begin + t
            if ssm_state_indices.ndim == 1:
                return int(ssm_state_indices[seq])
            return int(ssm_state_indices[seq, t])
        first = 0 if num_accepted_tokens is None else int(num_accepted_tokens[seq]) - 1
        slot = index(first)
        if ssm_state_indices is not None and slot <= 0:
            continue
        state = initial_state[slot].float().clone()
        for t in range(begin, end):
            bt = (torch.ones(v.shape[2], device=v.device) if beta is None
                  else beta[batch, t])
            y, state = _step(q[batch, t], k[batch, t], v[batch, t], g[batch, t],
                             bt, state, scale, use_qk_l2norm_in_kernel)
            out[batch, t] = y.to(out.dtype)
            target = index(t - begin) if inplace_final_state else batch * q.shape[1] + t
            if ssm_state_indices is None or not inplace_final_state or target > 0:
                final[target] = state.to(final.dtype)
    return out, final


def _recurrent_support(**k):
    if k["initial_state"] is None:
        return "vLLM recurrent entry requires explicit state storage"
    q, v, state = k["q"], k["v"], k["initial_state"]
    if v.shape[2] % q.shape[2] or tuple(state.shape[1:]) != (
        v.shape[2], v.shape[-1], q.shape[-1]
    ):
        return "GDN expects value-major state [slots, HV, V, K] and grouped heads"
    indices = k["ssm_state_indices"]
    if indices is None and (k["inplace_final_state"] or q.shape[0] != 1):
        return "vLLM recurrent no-index mode is audited only for B=1 out-of-place"
    if indices is not None and indices.ndim == 1:
        lengths = ([q.shape[1]] * q.shape[0] if k["cu_seqlens"] is None
                   else k["cu_seqlens"].diff().tolist())
        if max(lengths, default=0) > 1:
            return "multi-token recurrent cache writes require a 2D slot table"
    return None


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float(),
                                        beta=beta, threshold=threshold)
    return g.unsqueeze(0), b.float().sigmoid().to(b.dtype).unsqueeze(0)


def fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, num_k_heads,
                         head_k_dim, head_v_dim, apply_l2norm=True, output_g_exp=False):
    length = conv_output.shape[0]
    h, hv = num_k_heads, A_log.numel()
    q, k, v = conv_output.split([h * head_k_dim, h * head_k_dim, hv * head_v_dim], -1)
    q, k = q.reshape(length, h, head_k_dim), k.reshape(length, h, head_k_dim)
    if apply_l2norm:
        q, k = l2norm(q), l2norm(k)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    if output_g_exp:
        g = g.exp()
    return q.contiguous(), k.contiguous(), v.reshape(length, hv, head_v_dim).contiguous(), g, b.float().sigmoid()


def fused_sigmoid_gating_delta_rule_update(
    A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0, scale=None,
    initial_state=None, inplace_final_state=True, cu_seqlens=None,
    ssm_state_indices=None, num_accepted_tokens=None,
    use_qk_l2norm_in_kernel=False, is_kda=False,
):
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float(),
                                        beta=beta, threshold=threshold)
    return fused_recurrent_gated_delta_rule(
        q, k, v, g.reshape(*v.shape[:3]), b.float().sigmoid().reshape(*v.shape[:3]),
        scale, initial_state, inplace_final_state, cu_seqlens, ssm_state_indices,
        num_accepted_tokens, use_qk_l2norm_in_kernel,
    )


def _sigmoid_support(**k):
    if k.pop("is_kda"):
        return "KDA vector gating requires a separate reference audit"
    for name in ("A_log", "a", "b", "dt_bias", "threshold"):
        k.pop(name)
    k["g"] = None
    return _recurrent_support(**k)


def packed_decode(mixed_qkv, a, b, A_log, dt_bias, scale, initial_state,
                  out, ssm_state_indices, use_qk_l2norm_in_kernel=False):
    batch = mixed_qkv.shape[0]
    hv, vd, kd = initial_state.shape[1:]
    qdim = (mixed_qkv.shape[-1] - hv * vd) // 2
    q, k, v = mixed_qkv.split([qdim, qdim, hv * vd], -1)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    y, state = fused_recurrent_gated_delta_rule(
        q.reshape(batch, 1, -1, kd), k.reshape(batch, 1, -1, kd),
        v.reshape(batch, 1, hv, vd), g[:, None], b.float().sigmoid()[:, None],
        scale, initial_state, True, None, ssm_state_indices, None,
        use_qk_l2norm_in_kernel,
    )
    out.copy_(y)
    return out, state


def causal_conv1d_fn(
    x, weight, bias, conv_states, query_start_loc, cache_indices=None,
    has_initial_state=None, activation="silu", pad_slot_id=-1, null_block_id=0,
    block_idx_first_scheduled_token=None, block_idx_last_scheduled_token=None,
    initial_state_idx=None, num_computed_tokens=None, block_size_to_align=0,
    metadata=None, validate_data=False,
):
    from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import causal_conv1d_torch
    offsets = query_start_loc.tolist()
    output = torch.zeros_like(x)
    activation = "silu" if activation is True else None if activation is False else activation
    for seq, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        slot = seq if cache_indices is None else int(cache_indices[seq])
        if slot in (pad_slot_id, null_block_id) or begin == end:
            continue
        initial = False if has_initial_state is None else bool(has_initial_state[seq])
        # The upstream CPU torch function is portable; adapt padding and dtype.
        local = causal_conv1d_torch(
            x[:, begin:end].to(conv_states.dtype), weight, bias, conv_states,
            torch.tensor([0, end-begin], device=x.device),
            torch.tensor([slot], device=x.device),
            torch.tensor([initial], device=x.device), activation,
        )
        output[:, begin:end] = local.to(x.dtype)
    return output


def _conv_support(**k):
    if any(k[name] is not None for name in (
        "block_idx_first_scheduled_token", "block_idx_last_scheduled_token",
        "initial_state_idx", "num_computed_tokens",
    )):
        return "block-aligned convolution prefix-cache snapshots are outside this adapter"
    if k["weight"].shape[-1] < 2:
        return "upstream CPU causal convolution requires width >= 2"
    return None


def causal_conv1d_update(
    x, conv_state, weight, bias=None, activation=None, conv_state_indices=None,
    num_accepted_tokens=None, query_start_loc=None, max_query_len=-1,
    null_block_id=0, block_idx_last_scheduled_token=None, initial_state_idx=None,
    validate_data=False,
):
    from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import causal_conv1d_update_torch
    original_dtype = x.dtype
    work = x.to(conv_state.dtype)
    activation = "silu" if activation is True else None if activation is False else activation
    length = weight.shape[-1] - 1
    if query_start_loc is None:
        offsets = None
        sequences = work.shape[0]
    else:
        offsets = query_start_loc.tolist()
        sequences = len(offsets) - 1
    for seq in range(sequences):
        slot = seq if conv_state_indices is None else int(conv_state_indices[seq])
        if conv_state_indices is not None and slot == null_block_id:
            continue
        if offsets is None:
            seq_x = work[seq:seq+1]
            if seq_x.ndim == 2:
                seq_x = seq_x[..., None]
        else:
            begin, end = offsets[seq:seq+2]
            seq_x = work[begin:end].t()[None]
        if seq_x.shape[-1] == 0:
            continue
        y = causal_conv1d_update_torch(
            seq_x, conv_state[slot:slot+1, :, :length], weight, bias, activation,
        )
        if offsets is None:
            work[seq:seq+1].copy_(y.squeeze(-1) if work.ndim == 2 else y)
        else:
            work[begin:end].copy_(y[0].t())
    return work.to(original_dtype)


def _update_support(**k):
    if any(k[name] is not None for name in (
        "num_accepted_tokens", "block_idx_last_scheduled_token", "initial_state_idx",
    )):
        return "speculative/APC convolution snapshots are outside this adapter"
    if k["weight"].shape[-1] < 2:
        return "upstream CPU causal convolution requires width >= 2"
    return None


def custom(obj):
    identity = type(obj).__module__ + "." + type(obj).__qualname__
    if identity == "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.ChunkGatedDeltaRule":
        def run(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens=None,
                chunk_indices=None, chunk_offsets=None, use_qk_l2norm_in_kernel=True,
                core_attn_out=None):
            return chunk_gated_delta_rule(
                q, k, v, g, beta, None, initial_state, output_final_state,
                cu_seqlens, chunk_indices, chunk_offsets, use_qk_l2norm_in_kernel,
                core_attn_out,
            )
        def supports(**kwargs):
            return _chunk_support(**kwargs)
        return candidate(run, supports=check_signature(run, supports))
    # Legacy FL FLA CustomOps use a different [K,V] state contract and
    # duplicate upstream registration names. They are deliberately not
    # certified by inheritance or by the operation's short name.
    return None


def install():
    prefix = "vllm.model_executor.layers."
    for path, fn, support, source in (
        ("fla.ops.l2norm:l2norm_fwd", l2norm, None, "plugin.torch"),
        ("fla.ops.chunk:chunk_gated_delta_rule", chunk_gated_delta_rule, _chunk_support, "plugin.torch"),
        ("fla.ops.fused_recurrent:fused_recurrent_gated_delta_rule",
         fused_recurrent_gated_delta_rule, _recurrent_support, "plugin.torch"),
        ("fla.ops.fused_recurrent:fused_recurrent_gated_delta_rule_fwd",
         fused_recurrent_gated_delta_rule, _recurrent_support, "plugin.torch"),
        ("fla.ops.fused_recurrent:fused_recurrent_gated_delta_rule_packed_decode",
         packed_decode, None, "plugin.torch"),
        ("fla.ops.fused_sigmoid_gating:fused_sigmoid_gating_delta_rule_update",
         fused_sigmoid_gating_delta_rule_update, _sigmoid_support, "plugin.torch"),
        ("fla.ops.fused_gdn_prefill_post_conv:fused_post_conv_prep",
         fused_post_conv_prep, None, "plugin.torch"),
        ("mamba.gdn.qwen_gdn_linear_attn:fused_gdn_gating",
         fused_gdn_gating, None, "plugin.torch"),
        ("mamba.ops.causal_conv1d:causal_conv1d_fn",
         causal_conv1d_fn, _conv_support, "vllm.native"),
        ("mamba.ops.causal_conv1d:causal_conv1d_update",
         causal_conv1d_update, _update_support, "vllm.native"),
    ):
        checker = None if support is None else check_signature(fn, support)
        patch(prefix + path, (lambda fn=fn, checker=checker, source=source:
                              candidate(fn, source=source, supports=checker),))

def _chunk_support(**k):
    q, key, value = k["q"], k["k"], k["v"]
    if q.ndim != 4 or q.shape != key.shape or value.shape[:2] != q.shape[:2]:
        return "GDN requires matching [B,T,H,K] Q/K and [B,T,HV,V] values"
    if value.shape[2] % q.shape[2]:
        return "value heads must be a multiple of query/key heads"
    n = q.shape[0] if k["cu_seqlens"] is None else k["cu_seqlens"].numel() - 1
    state = k["initial_state"]
    if state is not None and tuple(state.shape) != (n, value.shape[2], value.shape[-1], q.shape[-1]):
        return "chunk state must use vLLM 0.24 [N,HV,V,K] layout"
    if k["beta"].shape != value.shape[:3] or k["g"].shape != value.shape[:3]:
        return "chunk GDN requires scalar decay/beta per token and value head"
    return None
