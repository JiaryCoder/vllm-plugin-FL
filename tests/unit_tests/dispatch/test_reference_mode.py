# Copyright (c) 2026 BAAI. All rights reserved.
"""Stage-one routing, numerical contracts and unsupported-path checks."""
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm_fl.dispatch import (
    CachedOp, OpImpl, OpManager, BackendImplKind, SelectionPolicy,
    get_default_manager, reset_default_manager, reset_global_policy,
    set_global_policy, call_op, resolve_op,
)
from vllm_fl.reference import (
    Candidate, ReferenceUnavailable, ReferencePurityError,
    clear_records, get_records, run_reference,
)
from vllm_fl.reference.hooks import configure_reference
from vllm_fl.reference import adapters
from vllm_fl.reference.engine import optimized_fallback


@pytest.fixture(autouse=True)
def reference_mode(monkeypatch):
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE", "1")
    monkeypatch.setenv("VLLM_FL_STRICT", "1")
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_FL_REFERENCE_REPORT_DIR", raising=False)
    reset_default_manager()
    reset_global_policy()
    set_global_policy(SelectionPolicy(strict=True))
    clear_records()
    yield
    clear_records()
    reset_default_manager()
    reset_global_policy()


@pytest.fixture
def config():
    cfg = VllmConfig()
    configure_reference(cfg)
    with set_current_vllm_config(cfg):
        yield cfg


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device(request.param, 0) if request.param == "cuda" else torch.device("cpu")


DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
def test_existing_activation_dispatch_uses_upstream(dtype, device):
    x = torch.linspace(-3, 3, 32, device=device, dtype=dtype).reshape(2, 16)
    for name, activation, obj in [
        ("silu_and_mul", torch.nn.functional.silu, None),
        ("gelu_and_mul", torch.nn.functional.gelu, SimpleNamespace(approximate="none")),
    ]:
        # Test all public routes; CachedOp must not bypass the new resolver.
        expected = activation(x[..., :8]) * x[..., 8:]
        for caller in [lambda z: call_op(name, obj, z),
                       lambda z: resolve_op(name)(obj, z),
                       lambda z: CachedOp(name)(obj, z)]:
            torch.testing.assert_close(caller(x), expected)
        routes = [r for r in get_records() if r["op"] == name]
        assert routes and all(r["source"] == "vllm.native" for r in routes)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("residual", [False, True])
def test_rms_overflow_residual_and_weightless(dtype, device, residual):
    x = torch.full((2, 8), 300, device=device, dtype=dtype)
    r = torch.full_like(x, 100) if residual else None
    obj = SimpleNamespace(
        weight=torch.ones(8, device=device, dtype=dtype),
        variance_epsilon=1e-6, has_weight=False, variance_size_override=4,
    )
    actual = call_op("rms_norm", obj, x, r)
    if residual:
        output, new_residual = actual
        torch.testing.assert_close(new_residual, torch.full_like(x, 400))
    else:
        output = actual
    torch.testing.assert_close(output, torch.ones_like(x))
    torch.testing.assert_close(x, torch.full_like(x, 300))
    if r is not None:
        torch.testing.assert_close(r, torch.full_like(x, 100))


@pytest.mark.parametrize("dtype", DTYPES)
def test_plugin_rms_fallback_preserves_fp32_contract(monkeypatch, dtype, device):
    def unavailable(_):
        raise ReferenceUnavailable("simulated missing vLLM IR native")
    monkeypatch.setattr(adapters, "native_ir", unavailable)
    obj = SimpleNamespace(weight=torch.ones(4,device=device,dtype=dtype),
                          variance_epsilon=1e-6)
    x = torch.full((1, 4), 300, device=device, dtype=dtype)
    output, residual = call_op("rms_norm", obj, x, torch.zeros_like(x))
    torch.testing.assert_close(output, torch.ones_like(x))
    torch.testing.assert_close(residual, x)
    assert get_records()[-1]["source"] == "plugin.torch"


@pytest.mark.parametrize("dtype", DTYPES)
def test_dynamic_int8_zero_rows_and_rounding(dtype, device):
    x = torch.tensor([[0, 1, -1, 2, -2], [0, 0, 0, 0, 0]],
                     device=device, dtype=dtype)
    q, scales = call_op("dynamic_per_token_quant_int8", x)
    expected = torch.tensor([[0, 64, -64, 127, -127], [0,0,0,0,0]],
                            device=device, dtype=torch.int8)
    assert torch.equal(q, expected)
    torch.testing.assert_close(scales, torch.tensor([[2/127], [0]],
                                                       device=device,dtype=torch.float32))
    assert get_records()[-1]["source"] == "plugin.torch"


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("rank", [3, 4])
def test_dispatch_rotary_layout_and_plugin_fallback(monkeypatch, device, interleaved, rank):
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
    pos = torch.tensor([0, 1, 2], device=device)
    angles = torch.arange(24,device=device,dtype=torch.float32).reshape(4,6) / 10
    cos, sin = angles.cos(), angles.sin()
    q = torch.randn((3,2,12) if rank == 3 else (2,2,3,12), device=device)
    k = torch.randn((3,1,12) if rank == 3 else (2,1,3,12), device=device)
    q_before, k_before = q.clone(), k.clone()
    def expected(x):
        if rank == 4:
            return ApplyRotaryEmb.forward_static(
                x.transpose(1,2),cos[pos],sin[pos],not interleaved
            ).transpose(1,2)
        return ApplyRotaryEmb.forward_static(x,cos[pos],sin[pos],not interleaved)
    eq, ek = expected(q), expected(k)
    aq, ak = call_op("rotary_embedding", None,q,k,cos,sin,pos,interleaved,True)
    torch.testing.assert_close(aq,eq)
    torch.testing.assert_close(ak,ek)
    assert get_records()[-1]["source"] == "vllm.native"
    monkeypatch.setattr(adapters, "upstream_dispatch", lambda _: None)
    aq, ak = call_op("rotary_embedding", None,q,k,cos,sin,pos,interleaved,True)
    torch.testing.assert_close(aq,eq)
    torch.testing.assert_close(ak,ek)
    assert get_records()[-1]["source"] == "plugin.torch"
    assert torch.equal(q,q_before) and torch.equal(k,k_before)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("interleaved", [False, True])
def test_generic_mrope_without_plugin_kernel(config, dtype, device, interleaved):
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    rope = MRotaryEmbedding(16,12,16,10000,True,dtype,[2,2,2],interleaved).to(device)
    q = torch.randn(4,32,device=device,dtype=dtype)
    k = torch.randn(4,16,device=device,dtype=dtype)
    pos = torch.tensor([[0,1,2,3],[0,2,3,4],[0,3,4,5]],device=device)
    qo,ko = rope(pos,q,k)
    assert torch.isfinite(qo).all() and torch.isfinite(ko).all()
    assert torch.equal(qo[0],q[0]) and torch.equal(ko[0],k[0])
    assert torch.equal(qo.reshape(4,2,16)[...,12:],q.reshape(4,2,16)[...,12:])
    if device.type == "cuda":
        # Independent optimized upstream MRoPE kernel, on cloned inputs.
        with optimized_fallback():
            eq,ek = rope.forward_cuda(pos,q.clone(),k.clone())
        tol = 3e-2 if dtype == torch.bfloat16 else 3e-3
        torch.testing.assert_close(qo,eq,rtol=tol,atol=tol)
        torch.testing.assert_close(ko,ek,rtol=tol,atol=tol)
    routes = [r for r in get_records() if r["op"].endswith(".MRotaryEmbedding")]
    assert routes and all(r["source"] == "vllm.native" for r in routes)


@pytest.mark.parametrize("key_present", [False, True])
def test_generic_rope_partial_dimension_and_optional_key(config, device, key_present):
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
    rope = RotaryEmbedding(16,12,16,10000,False,torch.float32).to(device)
    q = torch.randn(4,32,device=device)
    k = torch.randn(4,16,device=device) if key_present else None
    pos = torch.arange(4,device=device)
    qo,ko = rope(pos,q,k)
    assert qo.shape == q.shape
    assert (ko is None) == (k is None)
    assert torch.equal(qo[0],q[0])
    assert torch.equal(qo.reshape(4,2,16)[...,12:],q.reshape(4,2,16)[...,12:])
    apply = ApplyRotaryEmb(is_neox_style=False)
    x = torch.randn(4,2,12,device=device)
    y = apply(x,torch.ones(4,6,device=device),torch.zeros(4,6,device=device))
    assert torch.equal(x,y)
    assert any(r["op"].endswith(".ApplyRotaryEmb") for r in get_records())


def test_customop_and_ir_norm_cannot_reenter_optimized(config, monkeypatch):
    from vllm import ir
    from vllm.model_executor.layers.layernorm import RMSNorm
    norm = RMSNorm(4,dtype=torch.float16)
    x = torch.full((1,4),300.,dtype=torch.float16)
    torch.testing.assert_close(norm(x),torch.ones_like(x))
    y,r = ir.ops.fused_add_rms_norm.maybe_inplace(
        x,torch.zeros_like(x),norm.weight,1e-6,None)
    torch.testing.assert_close(y,torch.ones_like(x))
    torch.testing.assert_close(r,x)
    assert any(row["op"] == "ir.fused_add_rms_norm" for row in get_records())
    assert ir.ops.rms_norm.dispatch(x,norm.weight,1e-6,None).provider == "native"


@pytest.mark.parametrize("op", ["stage_two_unknown", "attention_backend", "sparse_moe"])
def test_strict_rejects_missing_and_mislabeled_reference(op):
    # Standard attention and dense MoE now have audited stage-two candidates;
    # an unsupported MLA variant must still reject the legacy kernel wrapper.
    kwargs = {"use_mla": True} if op == "attention_backend" else {}
    with pytest.raises(ReferenceUnavailable, match="No audited torch reference"):
        call_op(op, **kwargs)
    assert get_records()[-1]["source"] == "unavailable"


def test_non_strict_uses_configured_second_backend():
    manager = OpManager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    manager.registry.register_many([
        OpImpl("missing","flagos.mock",BackendImplKind.DEFAULT,lambda x:x+100),
        OpImpl("missing","vendor.mock",BackendImplKind.VENDOR,lambda x:x+2,vendor="mock"),
        OpImpl("missing","reference.fake",BackendImplKind.REFERENCE,
               lambda x:pytest.fail("unreviewed reference must not execute")),
    ])
    set_global_policy(SelectionPolicy.from_dict(
        strict=False,per_op_order={"missing":["reference","vendor:mock","flagos"]}))
    assert manager.call("missing",3) == 5
    assert get_records()[-1]["implementation"] == "vendor.mock"


def test_reference_execution_error_does_not_retry():
    x = torch.zeros(1)
    def broken(x):
        x.add_(1)
        raise RuntimeError("reference bug")
    with pytest.raises(RuntimeError,match="reference bug"):
        run_reference("broken",(x,),{},(
            lambda:Candidate("vllm.native","broken",broken),
            lambda:Candidate("plugin.torch","next",lambda x:pytest.fail("retried")),
        ),lambda:pytest.fail("optimized retry"),strict=False)
    assert x.item() == 1


def test_opaque_custom_operation_is_rejected():
    lib = torch.library.Library("fl_reference_test","DEF")
    lib.define("opaque(Tensor x) -> Tensor")
    lib.impl("opaque",lambda x:x+1,"CPU")
    with pytest.raises(ReferencePurityError,match="opaque"):
        run_reference("bad",(torch.ones(1),),{},(
            lambda:Candidate("vllm.native","bad",torch.ops.fl_reference_test.opaque),
        ),strict=True)


def test_unknown_customop_and_forward_override_are_not_accepted(config):
    from vllm.model_executor.custom_op import CustomOp
    class Unknown(CustomOp):
        name = "unknown_stage1"
        def forward_native(self,x):
            return x+1
        def forward_cuda(self,x):
            return x+2
    class Direct(Unknown):
        def forward(self,x):
            return x+3
    for cls in (Unknown,Direct):
        with pytest.raises(ReferenceUnavailable,match="audited"):
            cls()(torch.ones(1))


def test_unknown_variant_does_not_inherit_rope_certificate(config):
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    class DifferentRope(RotaryEmbedding):
        pass
    rope=DifferentRope(8,8,8,10000,True,torch.float32)
    with pytest.raises(ReferenceUnavailable,match="DifferentRope"):
        rope(torch.zeros(1,dtype=torch.long),torch.ones(1,8),torch.ones(1,8))


def test_report_has_real_source_and_no_tensor_values(monkeypatch,tmp_path):
    monkeypatch.setenv("VLLM_FL_REFERENCE_REPORT_DIR",str(tmp_path))
    call_op("silu_and_mul",None,torch.ones(1,8))
    import json
    rows=[json.loads(s) for s in next(tmp_path.glob("reference-*.jsonl")).read_text().splitlines()]
    assert rows[0]["source"] == "vllm.native"
    assert "tensor" not in rows[0] and "values" not in rows[0]


def test_ir_uses_plugin_when_upstream_provider_is_unavailable(config,monkeypatch):
    from vllm import ir
    from vllm_fl.reference import hooks
    def unavailable(_):
        raise ReferenceUnavailable("upstream native missing")
    monkeypatch.setattr(hooks,"native_ir",unavailable)
    x=torch.full((1,4),300.,dtype=torch.float16)
    out=ir.ops.rms_norm(x,torch.ones(4,dtype=x.dtype),1e-6,None)
    torch.testing.assert_close(out,torch.ones_like(x))
    assert get_records()[-1]["source"] == "plugin.torch"


def test_active_flaggems_aten_overrides_are_rejected(monkeypatch):
    import flag_gems
    monkeypatch.setattr(flag_gems,"current_work_registrar",object(),raising=False)
    with pytest.raises(ReferencePurityError,match="fresh reference worker"):
        call_op("silu_and_mul",None,torch.ones(1,8))


def test_normal_mode_is_unchanged_after_hook_installation(config,monkeypatch):
    from vllm.model_executor.custom_op import CustomOp
    class Direct(CustomOp):
        name="ordinary_direct"
        def forward(self,x):
            return x+7
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE","0")
    layer=Direct()
    assert layer(torch.tensor(1)).item() == 8
    assert not get_records()


def test_non_strict_custom_forward_uses_original_route(config):
    from vllm.model_executor.custom_op import CustomOp
    class Direct(CustomOp):
        name="unreviewed_forward"
        def forward(self,x):
            return x+7
    set_global_policy(SelectionPolicy(strict=False))
    assert Direct()(torch.tensor(1)).item() == 8
    assert get_records()[-1]["source"] == "optimized_fallback"
    assert get_records()[-1]["implementation"].endswith("Direct.forward")


def test_generic_plugin_fallback_when_upstream_activation_missing(config,monkeypatch):
    from vllm.model_executor.layers.activation import SiluAndMul
    original=adapters.load
    def missing(path):
        if path.startswith("vllm.model_executor.layers.activation:"):
            raise ReferenceUnavailable("upstream absent")
        return original(path)
    monkeypatch.setattr(adapters,"load",missing)
    layer=SiluAndMul()
    x=torch.ones(1,8)
    torch.testing.assert_close(layer(x),torch.nn.functional.silu(x[:,:4]))
    assert get_records()[-1]["source"] == "plugin.torch"


def test_full_width_rotary_cache_uses_plugin_only(device):
    angles=torch.arange(12,device=device,dtype=torch.float32).reshape(2,6)/10
    cos=angles.cos().repeat_interleave(2,-1)
    sin=angles.sin().repeat_interleave(2,-1)
    pos=torch.arange(2,device=device)
    x=torch.randn(2,1,12,device=device)
    y,_=call_op("rotary_embedding",None,x,None,cos,sin,pos,True,False)
    xr=x.reshape(2,1,6,2)
    expected=torch.stack(
        (xr[...,0]*angles.cos()[:,None]-xr[...,1]*angles.sin()[:,None],
         xr[...,1]*angles.cos()[:,None]+xr[...,0]*angles.sin()[:,None]),dim=-1
    ).flatten(-2)
    torch.testing.assert_close(y,expected)
    assert get_records()[-1]["source"] == "plugin.torch"


def test_unsupported_dtype_strict_and_nonstrict():
    manager=OpManager()
    manager._state.initialized=True
    manager._state.init_pid=os.getpid()
    manager.registry.register_impl(OpImpl(
        "silu_and_mul","vendor.mock",BackendImplKind.VENDOR,
        lambda obj,x:x[:,:2]+10,vendor="mock"))
    x=torch.zeros(1,4,dtype=torch.float64)
    with pytest.raises(ReferenceUnavailable,match="unaudited floating dtype"):
        manager.call("silu_and_mul",None,x)
    set_global_policy(SelectionPolicy(strict=False,prefer="vendor"))
    assert torch.equal(manager.call("silu_and_mul",None,x),x[:,:2]+10)
    assert get_records()[-1]["implementation"] == "vendor.mock"


def _spawn_reference_worker(queue,report_directory):
    os.environ["VLLM_FL_REFERENCE_REPORT_DIR"]=report_directory
    reset_global_policy()
    reset_default_manager()
    clear_records()
    cfg=VllmConfig()
    configure_reference(cfg)
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    with set_current_vllm_config(cfg):
        layer=MRotaryEmbedding(12,12,8,10000,True,torch.float32,[2,2,2])
        x=torch.ones(1,12)
        y,_=layer(torch.zeros(3,1,dtype=torch.long),x,x)
        queue.put({"pid":os.getpid(),"value":y.tolist(),"records":get_records(),
                   "mode":str(cfg.compilation_config.mode)})


def test_spawn_workers_inherit_reference_policy_and_have_separate_reports(tmp_path):
    import multiprocessing as mp
    context=mp.get_context("spawn")
    queue=context.Queue()
    workers=[context.Process(target=_spawn_reference_worker,args=(queue,str(tmp_path)))
             for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        results=[queue.get(timeout=90) for _ in workers]
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
        assert len({r["pid"] for r in results}) == 2
        for result in results:
            assert result["value"] == [[1.0]*12]
            assert result["records"][-1]["source"] == "vllm.native"
            assert result["records"][-1]["pid"] == result["pid"]
        assert len(list(tmp_path.glob("reference-*.jsonl"))) == 2
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
        queue.close()


def test_unknown_ir_strict_and_original_execution_context(config):
    from vllm.ir.op import register_op
    from vllm_fl.reference.engine import reference_enabled
    @register_op
    def fl_stage1_unreviewed_ir(x: torch.Tensor) -> torch.Tensor:
        assert not reference_enabled(), "fallback execution re-entered reference"
        return x+5
    with pytest.raises(ReferenceUnavailable,match="outside the audited"):
        fl_stage1_unreviewed_ir(torch.ones(1))
    set_global_policy(SelectionPolicy(strict=False))
    assert fl_stage1_unreviewed_ir(torch.ones(1)).item() == 6
    assert get_records()[-1]["source"] == "optimized_fallback"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("residual_present", [False,True])
def test_plugin_norm_random_weight_and_variance_override(dtype,device,residual_present):
    from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch
    torch.manual_seed(13)
    x=torch.randn(3,16,device=device,dtype=dtype)*10
    residual=torch.randn_like(x) if residual_present else None
    obj=SimpleNamespace(weight=torch.randn(16,device=device,dtype=dtype),
                        variance_epsilon=1e-5,variance_size_override=8)
    original_x=x.clone()
    expected_x=x.float()
    if residual is not None:
        expected_x=expected_x+residual.float()
        expected_residual=expected_x.to(dtype)
    expected=expected_x*torch.rsqrt(expected_x[:,:8].square().mean(-1,keepdim=True)+1e-5)
    expected=expected.to(dtype)*obj.weight
    result=rms_norm_torch(obj,x,residual)
    if residual_present:
        result,new_residual=result
        torch.testing.assert_close(new_residual,expected_residual,rtol=0,atol=0)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
    assert torch.equal(x,original_x)


def test_tanh_gelu_and_text_mrope(config,device):
    from vllm.model_executor.layers.activation import GeluAndMul
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    x=torch.linspace(-4,4,32,device=device).reshape(2,16)
    layer=GeluAndMul(approximate="tanh")
    expected=torch.nn.functional.gelu(x[:,:8],approximate="tanh")*x[:,8:]
    torch.testing.assert_close(layer(x),expected)
    rope=MRotaryEmbedding(12,12,8,10000,False,torch.float32,[2,2,2]).to(device)
    pos=torch.arange(3,device=device)
    q=torch.randn(3,24,device=device)
    k=torch.randn(3,12,device=device)
    qo,ko=rope(pos,q,k)
    assert torch.equal(qo[0],q[0]) and torch.equal(ko[0],k[0])
    torch.testing.assert_close(qo.reshape(3,2,12).float().square().sum(-1),
                               q.reshape(3,2,12).float().square().sum(-1))


def test_mrope_unsupported_variant_is_reported_before_execution(config):
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    rope=MRotaryEmbedding(12,12,8,10000,True,torch.float32,[2,2,2])
    with pytest.raises(ReferenceUnavailable,match="requires key"):
        rope(torch.zeros(3,1,dtype=torch.long),torch.ones(1,12),None)
    rope.scaling_factor=2
    with pytest.raises(ReferenceUnavailable,match="YaRN"):
        rope(torch.zeros(3,1,dtype=torch.long),torch.ones(1,12),torch.ones(1,12))


def test_selected_implementation_tracks_return_to_previous_provider(monkeypatch):
    manager=get_default_manager()
    x=torch.ones(1,8)
    upstream=adapters.upstream_dispatch
    manager.call("silu_and_mul",None,x)
    first=manager.get_selected_impl_id("silu_and_mul")
    monkeypatch.setattr(adapters,"upstream_dispatch",lambda _:None)
    manager.call("silu_and_mul",None,x)
    assert manager.get_selected_impl_id("silu_and_mul") != first
    monkeypatch.setattr(adapters,"upstream_dispatch",upstream)
    manager.call("silu_and_mul",None,x)
    assert manager.get_selected_impl_id("silu_and_mul") == first
    assert get_records()[-1]["calls"] == 2


def test_unsupported_weight_dtype_is_not_certified():
    obj=SimpleNamespace(weight=torch.ones(4,dtype=torch.float64),
                        variance_epsilon=1e-6)
    with pytest.raises(ReferenceUnavailable,match="unaudited floating dtype"):
        call_op("rms_norm",obj,torch.ones(1,4))
