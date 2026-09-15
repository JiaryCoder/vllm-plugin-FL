# Copyright (c) 2026 BAAI. All rights reserved.
"""Stage-two references: real entry points, independent numerical checks."""
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm_fl.dispatch import (
    SelectionPolicy, reset_global_policy, set_global_policy, reset_default_manager,
)
from vllm_fl.reference import clear_records, get_records, ReferenceUnavailable
from vllm_fl.reference.hooks import configure_reference

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture(autouse=True)
def setup_reference(monkeypatch):
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE", "1")
    monkeypatch.delenv("VLLM_FL_REFERENCE_REPORT_DIR", raising=False)
    reset_default_manager()
    reset_global_policy()
    set_global_policy(SelectionPolicy(strict=True))
    clear_records()
    cfg = VllmConfig()
    configure_reference(cfg)
    with set_current_vllm_config(cfg):
        yield
    clear_records()
    reset_default_manager()
    reset_global_policy()


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("residual", [False, True])
def test_gemma_norm(device, dtype, residual):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    obj = GemmaRMSNorm(8).to(device=device, dtype=dtype)
    obj.weight.data.copy_(torch.linspace(-.7, .7, 8, device=device, dtype=dtype))
    x = torch.full((3, 8), 300, device=device, dtype=dtype)
    r = torch.full_like(x, 100) if residual else None
    actual = obj(x, r)
    y = x.float() if r is None else x.float() + r.float()
    expected = (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6)
                * (1 + obj.weight.float())).to(dtype)
    if residual:
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[1], y.to(dtype))
    else:
        torch.testing.assert_close(actual, expected)
    assert get_records()[-1]["source"] == "vllm.native"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_size,before,activation", [
    (None, True, "silu"), (4, False, "swish"), (4, True, "sigmoid"),
])
def test_gated_norm(device, dtype, group_size, before, activation):
    from vllm.model_executor.layers.layernorm import RMSNormGated
    obj = RMSNormGated(8, group_size=group_size, norm_before_gate=before,
                      activation=activation, device=device, dtype=dtype)
    x = torch.linspace(-3, 3, 24, device=device, dtype=dtype).reshape(3, 8)
    z = torch.linspace(2, -2, 24, device=device, dtype=dtype).reshape(3, 8)
    gate = z.float().sigmoid()
    if activation != "sigmoid":
        gate = gate * z.float()
    y = x.float() if before else x.float() * gate
    grouped = y.reshape(3, -1, group_size or 8)
    expected = (grouped / (grouped.square().mean(-1, keepdim=True) + obj.eps).sqrt())
    expected = expected.reshape_as(x) * obj.weight.float()
    if before:
        expected *= gate
    torch.testing.assert_close(obj(x, z), expected.to(dtype))
    assert get_records()[-1]["source"] == "vllm.native"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["SwigluOAIAndMul", "SwigluStepAndMul"])
def test_extended_swiglu(device, dtype, name):
    from vllm.model_executor.layers import activation as module
    obj = getattr(module, name)(limit=4)
    x = torch.linspace(-10, 10, 64, device=device, dtype=dtype).reshape(4, 16)
    if name == "SwigluOAIAndMul":
        gate, up = x[..., ::2].clamp(max=4), x[..., 1::2].clamp(-4, 4)
        expected = (up + 1) * (gate * (gate * obj.alpha).sigmoid())
    else:
        gate, up = x.chunk(2, -1)
        expected = torch.nn.functional.silu(gate).clamp(max=4) * up.clamp(-4, 4)
    torch.testing.assert_close(obj(x), expected)
    assert get_records()[-1]["source"] == "vllm.native"


@pytest.mark.parametrize("variant", ["interleaved2", "interleaved3", "ernie"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_extra_mrope(device, dtype, variant):
    if variant.startswith("interleaved"):
        from vllm.model_executor.layers.rotary_embedding.mrope_interleaved import (
            MRotaryEmbeddingInterleaved,
        )
        section = [3, 3] if variant.endswith("2") else [2, 2, 2]
        obj = MRotaryEmbeddingInterleaved(16, 12, 32, 10000, True, dtype, section)
        axes = len(section)
        selection = obj.mrope_dim[:6]
    else:
        from vllm.model_executor.layers.rotary_embedding.ernie45_vl_rope import (
            Ernie4_5_VLRotaryEmbedding,
        )
        obj = Ernie4_5_VLRotaryEmbedding(
            16, 12, 32, 10000, True, dtype, mrope_section=[2, 2, 2],
        )
        axes = 3
        selection = [1, 2, 1, 2, 0, 0]
    obj = obj.to(device)
    pos = torch.arange(axes * 3, device=device).reshape(axes, 3)
    q = torch.randn(3, 2 * 16, device=device, dtype=dtype)
    k = torch.randn(3, 16, device=device, dtype=dtype)
    cache = obj.cos_sin_cache[pos]
    cos = torch.stack([cache[axis, :, i] for i, axis in enumerate(selection)], -1)
    sin = torch.stack([cache[axis, :, i + 6] for i, axis in enumerate(selection)], -1)
    def expected(x):
        v = x.reshape(3, -1, 16)
        first, second = v[..., :6], v[..., 6:12]
        return torch.cat((first * cos[:, None] - second * sin[:, None],
                          second * cos[:, None] + first * sin[:, None],
                          v[..., 12:]), -1).reshape_as(x)
    actual = obj(pos, q, k)
    torch.testing.assert_close(actual[0], expected(q), rtol=.02, atol=.02)
    torch.testing.assert_close(actual[1], expected(k), rtol=.02, atol=.02)
    assert get_records()[-1]["source"] == "vllm.native"


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_topk_route_matches_cuda(device, dtype, renormalize):
    from vllm_fl.dispatch import call_op
    from vllm_fl.reference.engine import optimized_fallback
    from vllm import _custom_ops
    logits = torch.tensor([[.1, 2, -1, 4], [0, 0, 0, 0]],
                          device=device, dtype=dtype)
    weights = torch.empty(2, 2, device=device)
    ids = torch.empty(2, 2, device=device, dtype=torch.int32)
    mapping = torch.empty_like(ids)
    result = call_op("topk_softmax", weights, ids, mapping, logits, renormalize)
    assert result[0] is weights and result[1] is ids
    assert ids.tolist() == [[3, 1], [0, 1]]
    assert mapping.tolist() == [[0, 2], [1, 3]]
    expected = logits.float().softmax(-1).gather(1, ids.long())
    if renormalize:
        expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected)
    if device == "cuda":
        ref_w, ref_i, ref_m = torch.empty_like(weights), torch.empty_like(ids), torch.empty_like(ids)
        with optimized_fallback():
            _custom_ops.topk_softmax(ref_w, ref_i, ref_m, logits, renormalize)
        torch.testing.assert_close(weights, ref_w)
        torch.testing.assert_close(ids, ref_i)
        torch.testing.assert_close(mapping, ref_m)


@pytest.mark.parametrize("invalid", [False, True])
def test_moe_alignment_expert_map_and_sum(device, invalid):
    from vllm_fl.dispatch import call_op
    ids = torch.tensor([[2, 0], [1, 2], [0, 1]], device=device, dtype=torch.int32)
    mapping = torch.tensor([1, -1, 0], device=device, dtype=torch.int32)
    sorted_ids, expert_ids, count = call_op(
        "moe_align_block_size", ids, 4, 3, mapping, False, invalid)
    n = int(count)
    assert n == (8 if invalid else 12)
    flat = ids.flatten()
    recovered = []
    for start in range(0, n, 4):
        e = int(expert_ids[start // 4])
        rows = sorted_ids[start:start+4]
        rows = rows[rows < ids.numel()].long()
        assert all(int(mapping[flat[r]]) == e for r in rows)
        recovered.extend(rows.tolist())
    assert sorted(recovered) == ([0, 1, 3, 4] if invalid else list(range(6)))
    x = torch.arange(24, device=device).reshape(2, 3, 4).float()
    out = torch.empty(2, 4, device=device)
    assert call_op("moe_sum", x, out) is None
    torch.testing.assert_close(out, x.sum(1))


@pytest.mark.parametrize("scoring", [0, 1])
def test_grouped_bias_uses_original_weights(device, scoring):
    from vllm_fl.dispatch import call_op
    x = torch.tensor([[.1, .2, .3, .4, .5, .6, .7, .8]], device=device)
    bias = torch.tensor([9., 10., 0., 0., 0., 0., 0., 0.], device=device)
    weights, ids = call_op("grouped_topk", x, 4, 1, 2, False, 2.0, bias, scoring)
    assert set(ids[0].tolist()) == {0, 1}
    scores = x if scoring == 0 else x.sigmoid()
    torch.testing.assert_close(weights, scores.gather(1, ids.long()) * 2.0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mapped", [False, True])
def test_fused_experts_real_function(device, dtype, mapped):
    from vllm_fl.ops.fused_moe.fused_moe import fused_experts_impl
    torch.manual_seed(5)
    x = torch.randn(3, 8, device=device, dtype=dtype)
    w1 = torch.randn(3, 12, 8, device=device, dtype=dtype) * .1
    w2 = torch.randn(3, 8, 6, device=device, dtype=dtype) * .1
    ids = torch.tensor([[0, 1], [2, 0], [1, 2]], device=device)
    weights = torch.tensor([[.2, .8], [.5, .5], [.7, .3]], device=device)
    mapping = torch.tensor([2, -1, 0], device=device) if mapped else None
    actual = fused_experts_impl(x, w1, w2, weights, ids, expert_map=mapping)
    expected = torch.zeros_like(x).float()
    for token in range(3):
        for slot in range(2):
            expert = int(ids[token, slot])
            if mapping is not None:
                expert = int(mapping[expert])
            if expert < 0:
                continue
            up = (x[token].float() @ w1[expert].float().t()).to(dtype)
            a = (torch.nn.functional.silu(up[:6]) * up[6:]).to(dtype)
            down = a.float() @ w2[expert].float().t()
            expected[token] += down * weights[token, slot]
    torch.testing.assert_close(actual, expected.to(dtype), atol=.003, rtol=.03)
    assert get_records()[-1]["source"] == ("plugin.torch" if mapped else "vllm.native")


@pytest.mark.parametrize("dtype", DTYPES)
def test_dispatched_expert_gemm_preserves_scatter_and_bias(device, dtype):
    from vllm_fl.dispatch import call_op
    x = torch.arange(12, device=device, dtype=dtype).reshape(3, 4) / 10
    w = torch.arange(40, device=device, dtype=dtype).reshape(2, 5, 4) / 30
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]], device=device, dtype=torch.int32)
    weights = torch.tensor([[.2, .8], [.5, .5], [.7, .3]], device=device)
    bias = torch.ones(2, 5, device=device, dtype=dtype) * .3
    sorted_ids, experts, count = call_op("moe_align_block_size", ids, 4, 2)
    out = torch.full((3, 2, 5), float("nan"), device=device, dtype=dtype)
    call_op("invoke_fused_moe_triton_kernel", x, w, out, None, None, weights,
            sorted_ids, experts, count, True, 2, {"BLOCK_SIZE_M": 4}, None,
            False, False, False, False, False, B_bias=bias)
    for t in range(3):
        for s in range(2):
            e = int(ids[t, s])
            expected = (x[t].float() @ w[e].float().t() + bias[e].float()) * weights[t, s]
            torch.testing.assert_close(out[t, s], expected.to(dtype))
    assert get_records()[-1]["source"] == "plugin.torch"


@pytest.mark.parametrize("dtype", DTYPES)
def test_dynamic_w8a8_linear_selected_kernel_and_integer_overflow(device, dtype):
    from vllm.model_executor.kernels.linear import init_int8_linear_kernel
    kernel = init_int8_linear_kernel(True, False, True, "reference_test")
    layer = torch.nn.Module()
    w = torch.arange(32, device=device).reshape(4, 8).to(torch.int8) - 10
    layer.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(torch.full((4, 1), .02, device=device),
                                           requires_grad=False)
    kernel.process_weights_after_loading(layer)
    x = torch.tensor([[0, 1, -1, 2, -2, 0, 0, 0], [0]*8], device=device, dtype=dtype)
    y = kernel.apply_weights(layer, x, torch.ones(4, device=device, dtype=dtype))
    q = torch.tensor([[0, 64, -64, 127, -127, 0, 0, 0], [0]*8], dtype=torch.int64)
    expected = (q @ w.cpu().long().t()).float() * (2/127) * .02 + 1
    torch.testing.assert_close(y, expected.to(device=device, dtype=dtype))
    assert get_records()[-1]["source"] == "plugin.torch"
    from vllm_fl.reference.w8a8 import integer_mm
    a = torch.full((1, 1101), 127, device=device, dtype=torch.int8)
    b = torch.full((1101, 1), 127, device=device, dtype=torch.int8)
    # Odd value beyond 2**24: float32 GEMM cannot represent it exactly.
    assert integer_mm(a, b).item() == 127 * 127 * 1101


@pytest.mark.parametrize("dtype", DTYPES)
def test_w8a8_moe_two_quantization_steps(device, dtype):
    from vllm_fl.ops.fused_moe.fused_moe import fused_experts_impl
    from vllm_fl.quantization.w8a8.reference import w8a8_linear_reference
    torch.manual_seed(8)
    x = torch.randn(3, 8, device=device, dtype=dtype)
    w1 = torch.randint(-50, 50, (2, 12, 8), device=device, dtype=torch.int8)
    w2 = torch.randint(-50, 50, (2, 8, 6), device=device, dtype=torch.int8)
    s1 = torch.full((2, 12, 1), .005, device=device)
    s2 = torch.full((2, 8, 1), .005, device=device)
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]], device=device)
    weights = torch.tensor([[.2, .8], [.5, .5], [.7, .3]], device=device)
    out = fused_experts_impl(x, w1, w2, weights, ids, use_int8_w8a8=True,
                             per_channel_quant=True, w1_scale=s1, w2_scale=s2)
    # CPU integer reference independently validates the GPU accumulator.
    expected = torch.zeros_like(x, device="cpu").float()
    for t in range(3):
        for slot in range(2):
            e = int(ids[t, slot])
            up = w8a8_linear_reference(x[t:t+1].cpu(), w1[e].cpu(), s1[e].cpu())
            act = torch.nn.functional.silu(up[:, :6]) * up[:, 6:]
            down = w8a8_linear_reference(act, w2[e].cpu(), s2[e].cpu())
            expected[t] += (down.float() * weights[t, slot].cpu()).to(dtype)[0].float()
    torch.testing.assert_close(out, expected.to(device=device, dtype=dtype),
                               atol=.001, rtol=.03)


def test_unsupported_quantization_fails_before_mutation(device):
    from vllm_fl.ops.fused_moe.fused_moe import fused_experts_impl
    x = torch.ones(1, 4, device=device)
    with pytest.raises(ReferenceUnavailable, match="only unquantized"):
        fused_experts_impl(x, torch.ones(1, 8, 4, device=device),
                          torch.ones(1, 4, 4, device=device),
                          torch.ones(1, 1, device=device),
                          torch.zeros(1, 1, device=device, dtype=torch.int32),
                          inplace=True, use_fp8_w8a8=True)
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("varlen", [False, True])
def test_gdn_chunk_matches_explicit_recurrence(device, dtype, varlen):
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
    torch.manual_seed(14)
    q = torch.randn(1 if varlen else 2, 5, 2, 4, device=device, dtype=dtype) * .2
    k = torch.randn_like(q) * .2
    v = torch.randn(*q.shape[:2], 2, 3, device=device, dtype=dtype)
    g = -torch.rand(*v.shape[:3], device=device)
    beta = torch.rand(*v.shape[:3], device=device)
    offsets = torch.tensor([0, 2, 5], device=device, dtype=torch.int32) if varlen else None
    h0 = torch.randn(2, 2, 3, 4, device=device) * .1
    hbefore = h0.clone()
    out_buffer = torch.full((v.numel() + 4,), 123.0, device=device, dtype=dtype)
    actual, state = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=h0, output_final_state=True,
        cu_seqlens=offsets, core_attn_out=out_buffer,
    )
    # Independent recurrence in key-major orientation using batched matrix products.
    expected = torch.zeros_like(v)
    final = []
    for seq, (batch, begin, end) in enumerate([(0, 0, 2), (0, 2, 5)] if varlen
                                             else [(0, 0, 5), (1, 0, 5)]):
        h = h0[seq].transpose(-1, -2).double()
        for t in range(begin, end):
            kt, qt = k[batch, t].double(), q[batch, t].double() * .5
            h = h * g[batch, t].double().exp()[:, None, None]
            prediction = torch.bmm(kt[:, None, :], h).squeeze(1)
            innovation = (v[batch, t].double() - prediction) * beta[batch, t, :, None]
            h += kt[:, :, None] * innovation[:, None, :]
            expected[batch, t] = torch.bmm(qt[:, None, :], h).squeeze(1).to(dtype)
        final.append(h.transpose(-1, -2).float())
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=.02)
    torch.testing.assert_close(state, torch.stack(final), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(h0, hbefore)
    torch.testing.assert_close(out_buffer[:v.numel()], actual.flatten())
    assert out_buffer[-4:].eq(123).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_gdn_packed_decode_gpu_kernel_and_state_slots(device, dtype):
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode
    from vllm_fl.reference.engine import optimized_fallback
    torch.manual_seed(42)
    m, h, hv, kd, vd = 3, 2, 4, 8, 6
    mixed = torch.randn(m, 2*h*kd + hv*vd, device=device, dtype=dtype) * .2
    a = torch.randn(m, hv, device=device, dtype=dtype)
    b = torch.randn_like(a)
    alog = torch.randn(hv, device=device)
    bias = torch.randn(hv, device=device)
    states = torch.randn(5, hv, vd, kd, device=device) * .1
    indices = torch.tensor([3, 1, 0], device=device, dtype=torch.int32)
    original = states.clone()
    out = torch.full((m, 1, hv, vd), 123., device=device, dtype=dtype)
    result, final = fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a, b, alog, bias, kd**-.5, states, out, indices, True,
    )
    assert result is out and final is states
    torch.testing.assert_close(states[[0, 2, 4]], original[[0, 2, 4]])
    assert out[2].eq(0).all()
    if device == "cuda":
        # Production FL registration already applies this existing numerical
        # compatibility patch. Compare the same FP32-beta contract here.
        from vllm_fl.patches.gdn_packed_decode import patch_vllm_packed_gdn_beta
        patch_vllm_packed_gdn_beta()
        kernel_state = original.clone()
        kernel_out = torch.full_like(out, 123.)
        with optimized_fallback():
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed, a, b, alog, bias, kd**-.5, kernel_state, kernel_out, indices, True,
            )
        torch.testing.assert_close(out, kernel_out, atol=.001, rtol=.02)
        torch.testing.assert_close(states, kernel_state, atol=1e-6, rtol=2e-5)


def test_gdn_speculative_slot_table_and_accepted_tokens(device):
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule
    # Only accepted old state 2 should seed the new sequence.
    states = torch.zeros(6, 1, 1, 1, device=device)
    states[1] = 99
    states[2] = 2
    slots = torch.tensor([[1, 2, 3]], device=device, dtype=torch.int32)
    q = torch.ones(1, 3, 1, 1, device=device)
    k = torch.full_like(q, .5)
    v = torch.ones_like(q)
    g = torch.zeros(1, 3, 1, device=device)
    beta = torch.full_like(g, .5)
    out, final = fused_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state=states,
        cu_seqlens=torch.tensor([0, 3], device=device, dtype=torch.int32),
        ssm_state_indices=slots, num_accepted_tokens=torch.tensor([2], device=device),
    )
    # h += .5 * (1 - .5*h) * .5; h=2 is a fixed point.
    torch.testing.assert_close(out, torch.full_like(out, 2))
    assert final is states
    assert states[1:4].eq(2).all()
    assert states[[0, 4, 5]].eq(0).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_causal_conv_prefill_decode_cache_continuity(device, dtype):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn, causal_conv1d_update,
    )
    weight = torch.tensor([[.1, .3, .6], [-.2, .4, .5]], device=device, dtype=dtype)
    x = torch.arange(10, device=device, dtype=dtype).reshape(2, 5) / 10
    states = torch.zeros(4, 2, 2, device=device, dtype=dtype)
    pre = causal_conv1d_fn(
        x[:, :3], weight, None, states,
        torch.tensor([0, 3], device=device, dtype=torch.int32),
        torch.tensor([2], device=device, dtype=torch.int32),
        torch.tensor([False], device=device), activation=None,
    )
    new = x[:, 3:].clone().unsqueeze(0)
    actual = causal_conv1d_update(
        new, states, weight, conv_state_indices=torch.tensor([2], device=device),
        activation=None,
    )
    expected = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x[None].float(), (2, 0)), weight[:, None].float(),
        groups=2,
    ).to(dtype)
    torch.testing.assert_close(torch.cat((pre[None], actual), -1), expected,
                               atol=.002, rtol=.02)
    torch.testing.assert_close(states[2], x[:, -2:])
    assert states[[0, 1, 3]].eq(0).all()
    assert actual.data_ptr() == new.data_ptr()


def test_gdn_constructor_dispatch_reassignment_cannot_bypass_hook(device):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import ChunkGatedDeltaRule
    from vllm.config import get_current_vllm_config
    get_current_vllm_config().model_config = SimpleNamespace(hf_text_config=SimpleNamespace(), dtype=torch.float32)
    obj = ChunkGatedDeltaRule()
    def forbidden(*args, **kwargs):
        raise AssertionError("optimized execution must not run")
    obj._forward_method = forbidden
    q = torch.ones(1, 1, 1, 2, device=device)
    v = torch.ones(1, 1, 1, 3, device=device)
    out, state = obj(q, q, v, torch.zeros(1, 1, 1, device=device),
                     torch.ones(1, 1, 1, device=device), None, True)
    assert out.shape == v.shape
    assert state.shape == (1, 1, 3, 2)
    assert get_records()[-1]["source"] == "plugin.torch"


def test_gdn_post_conv_gating_and_l2norm(device):
    from vllm.model_executor.layers.fla.ops import fused_post_conv_prep
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fused_gdn_gating
    x = torch.randn(4, 20, device=device)
    a, b = torch.randn(4, 2, device=device), torch.randn(4, 2, device=device)
    alog, dt = torch.randn(2, device=device), torch.randn(2, device=device)
    q, k, v, g, beta = fused_post_conv_prep(x, a, b, alog, dt, 2, 3, 4)
    assert (q.shape, k.shape, v.shape) == ((4, 2, 3), (4, 2, 3), (4, 2, 4))
    torch.testing.assert_close(q.square().sum(-1), torch.ones(4, 2, device=device),
                               atol=1e-4, rtol=1e-4)
    gg, bb = fused_gdn_gating(alog, a, b, dt)
    torch.testing.assert_close(g, gg[0])
    torch.testing.assert_close(beta, bb[0])


def test_convolution_unsupported_snapshot_fails_before_state_mutation(device):
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    x = torch.ones(1, 2, device=device)
    state = torch.ones(2, 2, 3, device=device)
    with pytest.raises(ReferenceUnavailable, match="speculative/APC"):
        causal_conv1d_update(x, state, torch.ones(2, 3, device=device),
                            num_accepted_tokens=torch.ones(1, device=device, dtype=torch.int32))
    assert state.eq(1).all() and x.eq(1).all()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("causal,window,softcap", [(True, None, None), (True, 3, 2.), (False, None, None)])
def test_paged_attention_prefill_decode_and_grouped_heads(device, dtype, kv_heads,
                                                         causal, window, softcap):
    from vllm_fl.reference.attention import TorchAttentionImpl, TorchAttentionMetadata
    torch.manual_seed(23)
    heads, dim, block = 4, 8, 2
    impl = TorchAttentionImpl(heads, dim, dim**-.5, kv_heads,
                              sliding_window=window, logits_soft_cap=softcap)
    # Logical blocks are deliberately out of order in physical cache storage.
    table = torch.tensor([[3, 1, 5], [4, 2, 6]], device=device, dtype=torch.int32)
    cache = torch.full((8, 2, block, kv_heads, dim), float("nan"), device=device, dtype=dtype)
    untouched = cache[0].clone()
    qs = [torch.randn(5, heads, dim, device=device, dtype=dtype) for _ in range(2)]
    ks = [torch.randn(5, kv_heads, dim, device=device, dtype=dtype) for _ in range(2)]
    vs = [torch.randn(5, kv_heads, dim, device=device, dtype=dtype) for _ in range(2)]
    def oracle(seq, begin, end, length):
        q = qs[seq][begin:end].double().transpose(0, 1)
        k = ks[seq][:length].double().repeat_interleave(heads // kv_heads, 1).transpose(0, 1)
        v = vs[seq][:length].double().repeat_interleave(heads // kv_heads, 1).transpose(0, 1)
        scores = q @ k.transpose(-1, -2) * dim**-.5
        if softcap:
            scores = (scores / softcap).tanh() * softcap
        qpos = torch.arange(begin, end, device=device)
        kpos = torch.arange(length, device=device)
        allowed = torch.ones(end-begin, length, dtype=torch.bool, device=device)
        if causal:
            allowed &= kpos[None] <= qpos[:, None]
        if window:
            allowed &= kpos[None] >= qpos[:, None] - window + 1
        return (scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v).transpose(0, 1).to(dtype)
    for begins, ends in [([0, 0], [3, 2]), ([3, 2], [4, 4]), ([4, 4], [5, 5])]:
        lens = [end-begin for begin, end in zip(begins, ends)]
        cumulative = torch.tensor([0, lens[0], sum(lens)], device=device, dtype=torch.int32)
        slots = []
        for seq, (begin, end) in enumerate(zip(begins, ends)):
            for pos in range(begin, end):
                slots.append(int(table[seq, pos // block]) * block + pos % block)
        metadata = TorchAttentionMetadata(
            sum(lens), cumulative, torch.tensor(ends, device=device), table,
            torch.tensor(slots, device=device), causal)
        query = torch.cat([qs[i][begins[i]:ends[i]] for i in range(2)])
        key = torch.cat([ks[i][begins[i]:ends[i]] for i in range(2)])
        value = torch.cat([vs[i][begins[i]:ends[i]] for i in range(2)])
        output = torch.empty_like(query)
        actual = impl.forward(None, query, key, value, cache, metadata, output)
        assert actual is output
        expected = torch.cat([oracle(i, begins[i], ends[i], ends[i]) for i in range(2)])
        torch.testing.assert_close(actual, expected, atol=.002, rtol=.02)
    for seq in range(2):
        for pos in range(5):
            torch.testing.assert_close(cache[int(table[seq, pos // block]), 0, pos % block], ks[seq][pos])
            torch.testing.assert_close(cache[int(table[seq, pos // block]), 1, pos % block], vs[seq][pos])
    torch.testing.assert_close(cache[0], untouched, equal_nan=True)
    assert get_records()[-1]["source"] == "plugin.torch"


def test_attention_backend_selection_and_metadata_builder(device):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm_fl.reference.attention import (
        PATH, TorchAttentionBackend, TorchAttentionMetadataBuilder,
    )
    from vllm.config import get_current_vllm_config
    config = AttentionSelectorConfig(8, torch.float32, "auto", None)
    assert PlatformFL.get_attn_backend_cls(None, config) == PATH
    assert TorchAttentionBackend.forward_includes_kv_cache_update
    qloc = torch.tensor([0, 2], device=device, dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=qloc, query_start_loc_cpu=qloc.cpu(),
        seq_lens=torch.tensor([2], device=device), num_reqs=1, num_actual_tokens=2,
        max_query_len=2, max_seq_len=2, block_table_tensor=torch.tensor([[1]], device=device),
        slot_mapping=torch.tensor([2, 3], device=device),
    )
    builder = TorchAttentionMetadataBuilder(None, ["layer"], get_current_vllm_config(), device)
    meta = builder.build(0, common)
    assert meta.block_table is common.block_table_tensor
    assert meta.query_start_loc is qloc
    with pytest.raises(ReferenceUnavailable, match="MLA"):
        PlatformFL.get_attn_backend_cls(None, config._replace(use_mla=True))


def test_attention_unsupported_selection_nonstrict_uses_policy(device, monkeypatch):
    # Other backend-detection tests can populate the class availability cache
    # while the platform is mocked. Re-detect actual hardware for this check.
    from vllm_fl.dispatch.backends.vendor.cuda.cuda import CudaBackend
    monkeypatch.setattr(CudaBackend, "_available", None)
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    set_global_policy(SelectionPolicy.from_dict(strict=False, per_op_order={"attention_backend": ["vendor:cuda"]}))
    result = PlatformFL.get_attn_backend_cls(
        None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None, use_mla=True))
    assert "flashmla" in result.lower()
    assert get_records()[-1]["source"] == "optimized_fallback"


def test_attention_alibi_profile_and_shared_cache(device):
    from vllm_fl.reference.attention import TorchAttentionImpl, TorchAttentionMetadata
    impl = TorchAttentionImpl(2, 4, .5, 1, alibi_slopes=[.5, 1.],
                              kv_sharing_target_layer_name="previous")
    cache = torch.zeros(2, 2, 4, 1, 4, device=device)
    cache[1, 1, :, 0] = torch.arange(4, device=device)[:, None]
    before = cache.clone()
    q = torch.zeros(1, 2, 4, device=device)
    meta = TorchAttentionMetadata(1, torch.tensor([0, 1], device=device),
                                   torch.tensor([4], device=device),
                                   torch.tensor([[1]], device=device),
                                   torch.tensor([7], device=device))
    out = impl.forward(None, q, None, None, cache, meta)
    expected = torch.stack([(torch.arange(-3, 1, device=device).float() * slope).softmax(-1)
                            @ torch.arange(4, device=device).float() for slope in [.5, 1.]])
    torch.testing.assert_close(out[0], expected[:, None].expand(2, 4))
    torch.testing.assert_close(cache, before)
    profile = impl.forward(None, q, None, None, cache, None)
    assert profile.eq(0).all()
    assert get_records()[-1]["source"] == "plugin.torch"


def test_free_function_aliases_and_nonstrict_execution_context(monkeypatch):
    import sys
    from types import ModuleType
    from vllm_fl.reference.functions import patch
    from vllm_fl.reference.engine import reference_enabled
    definition = ModuleType("vllm_fl._reference_test_definition")
    consumer = ModuleType("vllm_fl._reference_test_consumer")
    def optimized(x):
        assert not reference_enabled()
        return x + 7
    definition.compute = optimized
    consumer.cached_alias = optimized
    monkeypatch.setitem(sys.modules, definition.__name__, definition)
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    patch(definition.__name__ + ":compute", (lambda: None,))
    set_global_policy(SelectionPolicy(strict=False))
    assert consumer.cached_alias(3) == 10
    assert consumer.cached_alias is definition.compute
    assert get_records()[-1]["source"] == "optimized_fallback"
    set_global_policy(SelectionPolicy(strict=True))
    with pytest.raises(ReferenceUnavailable):
        consumer.cached_alias(3)
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE", "0")
    assert consumer.cached_alias(3) == 10


def test_upstream_moe_reuse_does_not_construct_oot_activation(monkeypatch, device):
    from vllm.model_executor.layers.fused_moe import cpu_fused_moe
    from vllm_fl.reference.moe import routed_experts
    def invalid_constructor(*args, **kwargs):
        raise AssertionError("OOT constructor is not a native math dependency")
    monkeypatch.setattr(cpu_fused_moe, "SiluAndMul", invalid_constructor)
    x = torch.ones(1, 2, device=device)
    result = routed_experts(
        x, torch.ones(1, 4, 2, device=device), torch.ones(1, 2, 2, device=device),
        torch.ones(1, 1, device=device), torch.zeros(1, 1, device=device, dtype=torch.int32),
    )
    expected = torch.full_like(x, 8 * torch.sigmoid(torch.tensor(2.)).item())
    torch.testing.assert_close(result, expected)
    assert get_records()[-1]["source"] == "vllm.native"


def test_gdn_rejects_legacy_key_major_state_before_writing(device):
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
    q = torch.ones(1, 2, 1, 3, device=device)
    v = torch.ones(1, 2, 1, 4, device=device)
    initial = torch.zeros(1, 1, 3, 4, device=device)
    out = torch.full((8,), 123., device=device)
    with pytest.raises(ReferenceUnavailable, match="V,K"):
        chunk_gated_delta_rule(q, q, v, torch.zeros(1, 2, 1, device=device),
                               torch.ones(1, 2, 1, device=device),
                               initial_state=initial, core_attn_out=out)
    assert initial.eq(0).all() and out.eq(123).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunk_gdn_agrees_with_upstream_cuda_kernel(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule
    from vllm_fl.reference.engine import optimized_fallback
    torch.manual_seed(31)
    q, k = [torch.randn(1, 7, 2, 8, device="cuda", dtype=dtype) for _ in range(2)]
    v = torch.randn(1, 7, 2, 6, device="cuda", dtype=dtype)
    g = -torch.rand(1, 7, 2, device="cuda")
    beta = torch.rand_like(g)
    state = torch.randn(2, 2, 6, 8, device="cuda") * .1
    offsets = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
    actual, actual_state = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
        cu_seqlens=offsets, use_qk_l2norm_in_kernel=True)
    with optimized_fallback():
        expected, expected_state = chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=state, output_final_state=True,
            cu_seqlens=offsets, use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(actual, expected, atol=.015, rtol=.03)
    torch.testing.assert_close(actual_state, expected_state, atol=.015, rtol=.03)


def test_packed_w8a8_load_route_and_signed_endpoints(device):
    from vllm_fl.quantization.w8a8.packed import unpack_uint8b128_int32
    codes = torch.tensor([[0, 1, 127, 128, 129, 254, 255, 0],
                          [255, 254, 129, 128, 127, 1, 0, 255]],
                         dtype=torch.uint8, device=device)
    packed = codes.view(torch.int32)
    actual = unpack_uint8b128_int32(packed, in_features=7)
    expected = (codes[:, :7].to(torch.int16) - 128).to(torch.int8)
    torch.testing.assert_close(actual, expected)
    assert get_records()[-1]["source"] == "plugin.torch"
    assert "unpack_uint8b128_int32" in get_records()[-1]["op"]
