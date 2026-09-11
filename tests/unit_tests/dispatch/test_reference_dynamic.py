# Copyright (c) 2026 BAAI. All rights reserved.
"""Native discovery on real vLLM classes plus routing/guard failure contracts."""
import math

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers import activation
from vllm_fl.dispatch import SelectionPolicy, reset_global_policy, set_global_policy
from vllm_fl.reference import clear_records, get_records, ReferenceUnavailable
from vllm_fl.reference.engine import (
    Candidate, ReferencePurityError, optimized_fallback, run_reference,
)
from vllm_fl.reference.hooks import configure_reference


@pytest.fixture(autouse=True)
def reference(monkeypatch):
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE", "1")
    monkeypatch.delenv("VLLM_FL_REFERENCE_REPORT_DIR", raising=False)
    set_global_policy(SelectionPolicy(strict=True))
    clear_records()
    config = VllmConfig()
    configure_reference(config)
    with set_current_vllm_config(config):
        yield
    reset_global_policy()
    clear_records()


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("name", ["FatreluAndMul", "NewGELU", "QuickGELU"])
def test_unlisted_activations_use_native(device, dtype, name, monkeypatch):
    cls = getattr(activation, name)
    obj = cls(.25) if name == "FatreluAndMul" else cls()
    def vendor_must_not_run(*args, **kwargs):
        raise AssertionError("platform implementation was called")
    monkeypatch.setattr(obj, "_forward_method", vendor_must_not_run)
    x = torch.linspace(-2, 2, 16, device=device, dtype=dtype).reshape(2, 8)
    if name == "FatreluAndMul":
        expected = torch.where(x[:, :4] > .25, x[:, :4], 0) * x[:, 4:]
    elif name == "NewGELU":
        expected = .5 * x * (1 + (math.sqrt(2 / math.pi) *
                                  (x + .044715 * x.pow(3))).tanh())
    else:
        expected = x * (1.702 * x).sigmoid()
    torch.testing.assert_close(obj(x=x), expected)
    row = get_records()[-1]
    assert row["source"] == "vllm.native.dynamic"
    assert row["implementation"].endswith(name + ".forward_native")


def _rope(kind, dtype):
    from vllm.model_executor.layers.rotary_embedding import (
        LinearScalingRotaryEmbedding, DynamicNTKScalingRotaryEmbedding,
        YaRNScalingRotaryEmbedding, Llama3RotaryEmbedding,
    )
    if kind == "linear":
        return LinearScalingRotaryEmbedding(16, 12, 64, 10000., True, 4., dtype)
    if kind == "ntk":
        return DynamicNTKScalingRotaryEmbedding(16, 12, 256, 64, 10000., True, 4., dtype)
    if kind == "yarn":
        return YaRNScalingRotaryEmbedding(16, 12, 64, 10000., True, 4., dtype)
    return Llama3RotaryEmbedding(16, 12, 256, 10000., True, dtype, 4., 1., 4., 64)


@pytest.mark.parametrize("kind", ["linear", "ntk", "yarn", "llama3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scaled_rope_keeps_instance_cache(device, dtype, kind):
    obj = _rope(kind, dtype).to(device)
    obj.use_flashinfer = False
    positions = torch.tensor([0, 63, 64, 127, 255], device=device)
    q = torch.randn(5, 32, device=device, dtype=dtype)
    k = torch.randn(5, 16, device=device, dtype=dtype)
    cache_before = obj.cos_sin_cache.clone()
    # An independent tensor formula uses the object's actual scaled cache.
    cos, sin = obj.cos_sin_cache[positions].chunk(2, -1)
    def expected(x):
        x3 = x.reshape(5, -1, 16)
        first, second = x3[..., :6], x3[..., 6:12]
        rotated = torch.cat((first * cos[:, None] - second * sin[:, None],
                             second * cos[:, None] + first * sin[:, None],
                             x3[..., 12:]), -1)
        return rotated.reshape_as(x)
    result = obj(positions, q, k)
    torch.testing.assert_close(result, (expected(q), expected(k)))
    torch.testing.assert_close(obj.cos_sin_cache, cache_before)
    assert get_records()[-1]["source"] == "vllm.native.dynamic"
    assert get_records()[-1]["implementation"].endswith("base.RotaryEmbedding.forward_native")
    if device == "cuda":
        with optimized_fallback():
            vendor = obj.forward_cuda(positions, q.clone(), k.clone())
        tolerance = 2e-6 if dtype == torch.float32 else .02
        torch.testing.assert_close(result, vendor, atol=tolerance, rtol=tolerance)


def test_discovered_class_group_selection_before_construction(monkeypatch):
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="rope"))
    obj = _rope("linear", torch.float32)
    monkeypatch.setattr(obj, "_forward_method", lambda *a: "original")
    assert obj(torch.zeros(1, dtype=torch.long), torch.ones(1, 16)) == "original"
    assert get_records()[-1]["source"] == "user_override"
    set_global_policy(SelectionPolicy(strict=True, reference_include="rope"))
    obj(torch.zeros(1, dtype=torch.long), torch.ones(1, 16))
    assert get_records()[-1]["source"] == "vllm.native.dynamic"


def test_new_activation_group_and_qualified_selectors(monkeypatch):
    path = "vllm.model_executor.layers.activation.NewGELU"
    set_global_policy(SelectionPolicy(strict=True, reference_include="activation"))
    obj = activation.NewGELU()
    x = torch.ones(2)
    obj(x)
    assert get_records()[-1]["source"] == "vllm.native.dynamic"
    set_global_policy(SelectionPolicy(strict=True, reference_include="custom:" + path))
    obj(x)
    assert get_records()[-1]["source"] == "vllm.native.dynamic"
    set_global_policy(SelectionPolicy(strict=True, reference_exclude=path))
    monkeypatch.setattr(obj, "_forward_method", lambda x: x + 9)
    torch.testing.assert_close(obj(x), x + 9)
    assert get_records()[-1]["source"] == "user_override"
    # 'all' must also cover classes absent from the old alias table.
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="all"))
    torch.testing.assert_close(obj(x), x + 9)


def test_foreign_subclass_cannot_claim_upstream_native():
    class External(activation.NewGELU):
        pass
    with pytest.raises(ReferenceUnavailable, match="not from vLLM"):
        External()(torch.ones(1))


def test_cached_discovery_does_not_hide_native_replacement(monkeypatch):
    obj = activation.NewGELU()
    obj(torch.ones(1))
    monkeypatch.setattr(activation.NewGELU, "forward_native", lambda self, x: x + 11)
    with pytest.raises(ReferenceUnavailable, match="not from vLLM"):
        obj(torch.ones(1))


def test_instance_override_is_not_silently_ignored(monkeypatch):
    obj = activation.NewGELU()
    monkeypatch.setattr(obj, "forward_native", lambda x: x + 11)
    with pytest.raises(ReferenceUnavailable, match="instance forward_native"):
        obj(torch.ones(1))


def test_native_stub_is_unavailable_before_execution(monkeypatch):
    from vllm.model_executor.custom_op import CustomOp
    monkeypatch.setattr(activation.NewGELU, "forward_native", CustomOp.forward_native)
    with pytest.raises(ReferenceUnavailable, match="no implemented forward_native"):
        activation.NewGELU()(torch.ones(1))


def test_forward_override_is_not_replaced_with_unrelated_inherited_native(monkeypatch):
    monkeypatch.setattr(activation.NewGELU, "forward", lambda self, x: x + 99)
    with pytest.raises(ReferenceUnavailable, match="overrides forward"):
        activation.NewGELU()(torch.ones(1))


def test_static_and_class_method_binding():
    from vllm_fl.reference.native import discover_method, NativeMethod
    x = torch.ones(1, 8)
    static = discover_method(activation.SiluAndMul()).bind(activation.SiluAndMul())
    torch.testing.assert_close(static(x), F.silu(x[:, :4]) * x[:, 4:])
    # The binding operation must supply cls, not an instance or two self args.
    method = NativeMethod(lambda cls, x: (cls, x), "class", "test")
    cls, value = method.bind(activation.NewGELU())(x)
    assert cls is activation.NewGELU and value is x


_DYNAMIC_IR = None


def dynamic_ir():
    global _DYNAMIC_IR
    if _DYNAMIC_IR is None:
        from vllm import ir
        _DYNAMIC_IR = ir.register_op(name="fl_dynamic_native_silu", allow_inplace=True)(
            activation.SiluAndMul.forward_native)
    return _DYNAMIC_IR


def test_dynamic_ir_and_nested_torch_wrapper(device):
    op = dynamic_ir()
    x = torch.randn(2, 8, device=device)
    before = x.clone()
    if "test_vendor" not in op.impls:
        @op.register_impl("test_vendor", inplace=True)
        def vendor(x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("optimized IR provider was selected")
    with op.set_priority(["test_vendor"]):
        for invoke in (op, op.maybe_inplace):
            # Public IR calls are opaque torch ops. The guard must unwrap only
            # registered overloads and re-enter our IR provider selection.
            result = run_reference("outer", (x,), {}, (
                lambda: Candidate("plugin.torch", "nested_ir", invoke),
            ))
            torch.testing.assert_close(result, F.silu(x[:, :4]) * x[:, 4:])
    torch.testing.assert_close(x, before)
    assert any(row["op"] == "ir.fl_dynamic_native_silu"
               and row["source"] == "vllm.native.dynamic" for row in get_records())


def test_unregistered_ir_namespace_stays_opaque():
    lib = torch.library.Library("vllm_ir", "FRAGMENT")
    lib.define("fl_dynamic_opaque(Tensor x) -> Tensor")
    lib.impl("fl_dynamic_opaque", lambda x: x + 1, "CPU")
    with pytest.raises(ReferencePurityError, match="opaque"):
        run_reference("outer", (torch.ones(1),), {}, (
            lambda: Candidate("plugin.torch", "unregistered", torch.ops.vllm_ir.fl_dynamic_opaque),
        ))


def test_ir_selector_applies_to_future_names():
    from vllm_fl.reference.selection import selection_reason
    set_global_policy(SelectionPolicy(reference_exclude="ir:fl_dynamic_native_silu"))
    assert selection_reason("ir.fl_dynamic_native_silu")


def test_sdpa_uses_math_under_reference(device):
    q, k, v = [torch.randn(1, 2, 7, 8, device=device) for _ in range(3)]
    mask = torch.ones(7, 7, device=device, dtype=torch.bool).tril()
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(8)
    expected = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v
    actual = run_reference("sdpa", (q, k, v), {}, (
        lambda: Candidate("plugin.torch", "sdpa_math",
                          lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
    ))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_triton_is_rejected_and_never_retried_after_mutation():
    from vllm.model_executor.layers.activation import swiglustep_and_mul_triton
    x = torch.zeros(1, 8)
    def bad(value):
        value.add_(1)
        swiglustep_and_mul_triton(torch.empty(1, 4), value)
    with pytest.raises(ReferencePurityError, match="direct Triton"):
        run_reference("bad_launch", (x,), {}, (
            lambda: Candidate("vllm.native.dynamic", "bad_launch", bad),
            lambda: Candidate("plugin.torch", "retry", lambda x: pytest.fail("retried")),
        ), lambda: pytest.fail("optimized retry"), strict=False)
    assert x.eq(1).all()
    assert get_records()[-1]["source"] == "error"


def test_guard_preserves_excluded_and_ordinary_triton():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from vllm.model_executor.layers.activation import swiglustep_and_mul_triton
    x = torch.randn(2, 8, device="cuda")
    out = torch.empty(2, 4, device="cuda")
    # Guard wrappers stay installed, but no reference computation is active.
    swiglustep_and_mul_triton(out, x)
    expected = F.silu(x[:, :4]).clamp(max=7) * x[:, 4:].clamp(-7, 7)
    torch.testing.assert_close(out, expected)
    set_global_policy(SelectionPolicy(reference_exclude="swiglustep_and_mul"))
    result = run_reference("swiglustep_and_mul", (), {}, (),
                           lambda: swiglustep_and_mul_triton(out, x))
    assert result is None
    torch.testing.assert_close(out, expected)
    assert get_records()[-1]["source"] == "user_override"
