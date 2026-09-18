# Copyright (c) 2026 BAAI. All rights reserved.
"""Selective reference must preserve strictness and execute the requested path."""
import json
import os

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm_fl.dispatch import (
    BackendImplKind, CachedOp, OpImpl, OpManager, SelectionPolicy,
    call_op, get_policy, reset_global_policy, reset_default_manager, set_global_policy,
)
from vllm_fl.dispatch.policy import (
    policy_from_env, with_preference, with_strict_mode,
    with_allowed_vendors, with_denied_vendors,
)
from vllm_fl.reference import (
    Candidate, ReferenceUnavailable, ReferencePurityError,
    clear_records, get_records, run_reference,
)
from vllm_fl.reference.hooks import configure_reference
from vllm_fl.reference.selection import canonical_name, selection_reason


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for name in ("VLLM_FL_CONFIG", "VLLM_FL_REFERENCE_INCLUDE", "VLLM_FL_REFERENCE_EXCLUDE",
                 "VLLM_FL_REFERENCE_REPORT_DIR", "VLLM_FL_PER_OP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE", "1")
    monkeypatch.setenv("VLLM_FL_REFERENCE_ATTENTION_BACKEND", "TORCH")
    monkeypatch.setenv("VLLM_FL_PREFER", "vendor")
    monkeypatch.setenv("VLLM_FL_STRICT", "1")
    reset_global_policy()
    reset_default_manager()
    set_global_policy(SelectionPolicy(strict=True, prefer="vendor"))
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


@pytest.mark.parametrize("selector", ["attention", "attention_backend"])
def test_env_exclusion_and_strictness(monkeypatch, selector):
    monkeypatch.setenv("VLLM_FL_REFERENCE_EXCLUDE", selector)
    p = policy_from_env()
    set_global_policy(p)
    assert p.strict and p.reference_exclude == frozenset({"attention_backend"})
    assert selection_reason("attention")
    assert selection_reason("rms_norm") is None
    with pytest.raises(ReferenceUnavailable):
        run_reference("unreviewed", (), {}, (), lambda: pytest.fail("strict fallback"))


def test_yaml_completely_overrides_environment(monkeypatch, tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text("prefer: vendor\nstrict: true\nreference_include: [normalization, attention]\n"
                    "reference_exclude: [attention]\n")
    monkeypatch.setenv("VLLM_FL_CONFIG", str(path))
    monkeypatch.setenv("VLLM_FL_REFERENCE_INCLUDE", "gdn")
    monkeypatch.setenv("VLLM_FL_REFERENCE_EXCLUDE", "normalization")
    p = policy_from_env()
    set_global_policy(p)
    assert p.reference_include == frozenset({"rms_norm", "gemma_rms_norm", "rms_norm_gated", "attention_backend", "@group:normalization"})
    assert selection_reason("attention")
    assert selection_reason("rms_norm") is None
    assert selection_reason("chunk_gated_delta_rule")


@pytest.mark.parametrize("bad", ["attenton", "rms_nrom", ["attention", 1], {"attention": True}])
def test_invalid_selectors_fail_before_model_loading(bad):
    with pytest.raises(ValueError):
        SelectionPolicy(reference_exclude=bad)


def test_include_empty_all_and_exclude_precedence():
    set_global_policy(SelectionPolicy(reference_include=""))
    assert selection_reason("rms_norm")
    set_global_policy(SelectionPolicy(reference_include="all"))
    assert selection_reason("unknown_custom") is None
    set_global_policy(SelectionPolicy(reference_include="normalization", reference_exclude="rms_norm"))
    assert selection_reason("ir.fused_add_rms_norm")
    assert selection_reason("gemma_rms_norm") is None


@pytest.mark.parametrize("scope", [lambda: with_preference("vendor"), with_strict_mode,
                                  lambda: with_allowed_vendors("cuda"), lambda: with_denied_vendors("mock")])
def test_policy_scopes_preserve_selection(scope):
    original = SelectionPolicy(reference_include="normalization,attention", reference_exclude="attention")
    set_global_policy(original)
    with scope():
        assert get_policy().reference_include == original.reference_include
        assert get_policy().reference_exclude == original.reference_exclude
    assert get_policy() is original
    assert original.fingerprint() != SelectionPolicy().fingerprint()


@pytest.mark.parametrize("name", [
    "rms_norm", "ir.rms_norm", "ir.fused_add_rms_norm", "RMSNorm", "RMSNormFL",
    "vllm.model_executor.layers.layernorm.RMSNorm", "vllm_fl.ops.layernorm.RMSNormFL",
])
def test_rms_aliases_share_selection(name):
    set_global_policy(SelectionPolicy(reference_exclude="rms_norm"))
    assert canonical_name(name) == "rms_norm"
    assert selection_reason(name)


def mock_manager(vendor=True):
    manager = OpManager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    impls = [OpImpl("silu_and_mul", "flagos.mock", BackendImplKind.DEFAULT,
                    lambda *a: pytest.fail("FlagGems must not execute"), priority=1000)]
    if vendor:
        impls.append(OpImpl("silu_and_mul", "vendor.mock", BackendImplKind.VENDOR,
                            lambda obj, x: x + 42, vendor="mock"))
    manager.registry.register_many(impls)
    return manager


def test_cached_dispatch_explicit_override_beats_reference_and_default_preference(monkeypatch):
    from vllm_fl.dispatch import manager as module
    manager = mock_manager()
    monkeypatch.setattr(module, "_default_manager", manager)
    cached = CachedOp("silu_and_mul")
    x = torch.ones(1, 8)
    assert cached(None, x).shape == (1, 4)
    set_global_policy(SelectionPolicy(strict=True, prefer="flagos", reference_exclude="silu_and_mul"))
    torch.testing.assert_close(cached(None, x), x + 42)
    assert get_records()[-1]["source"] == "user_override"
    assert get_records()[-1]["implementation"] == "vendor.mock"
    set_global_policy(SelectionPolicy(strict=True))
    assert cached(None, x).shape == (1, 4)
    assert get_records()[-1]["source"] == "vllm.native"


@pytest.mark.parametrize("policy", [
    {"per_op_order": {"silu_and_mul": ["flagos"]}},
    {"deny_vendors": {"mock"}},
])
def test_explicit_override_never_uses_flaggems_or_denied_vendor(policy):
    set_global_policy(SelectionPolicy.from_dict(strict=False, reference_exclude="silu_and_mul", **policy))
    with pytest.raises(ReferenceUnavailable):
        mock_manager().call("silu_and_mul", None, torch.ones(1, 8))


def test_missing_vendor_does_not_silently_reenter_reference():
    set_global_policy(SelectionPolicy(strict=False, reference_exclude="silu_and_mul"))
    with pytest.raises(ReferenceUnavailable):
        mock_manager(False).call("silu_and_mul", None, torch.ones(1, 8))


def test_customop_calls_saved_forward_and_keeps_other_ops_reference(config, monkeypatch):
    from vllm.model_executor.layers.activation import SiluAndMul, GeluAndMul
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="silu_and_mul"))
    obj = SiluAndMul()
    monkeypatch.setattr(obj, "_forward_method", lambda x: x + 21)
    x = torch.ones(1, 8)
    torch.testing.assert_close(obj(x), x + 21)
    assert get_records()[-1]["source"] == "user_override"
    assert GeluAndMul()(x).shape == (1, 4)
    assert get_records()[-1]["source"] == "vllm.native"


def test_ir_preserves_selected_provider_and_functional_inplace_semantics():
    from vllm import ir
    from vllm_fl.dispatch.policy import policy_context
    with policy_context(SelectionPolicy(strict=True, reference_exclude="rms_norm")):
        cfg = VllmConfig()
        priorities = cfg.kernel_config.ir_op_priority
        before = list(priorities.rms_norm)
        configure_reference(cfg)
        assert list(priorities.rms_norm) == before
        with set_current_vllm_config(cfg):
            x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
            w = torch.ones(16, device="cuda", dtype=x.dtype)
            from vllm_fl.reference.engine import optimized_fallback
            with optimized_fallback():
                expected = ir.ops.rms_norm(x, w, 1e-6, None)
            torch.testing.assert_close(ir.ops.rms_norm(x, w, 1e-6, None), expected)
            residual = torch.randn_like(x)
            with optimized_fallback():
                expected = ir.ops.fused_add_rms_norm(x.clone(), residual.clone(), w, 1e-6, None)
            actual = ir.ops.fused_add_rms_norm.maybe_inplace(x.clone(), residual.clone(), w, 1e-6, None)
            torch.testing.assert_close(actual, expected)
            assert all(r["source"] == "user_override" for r in get_records())


def test_real_free_function_alias_is_excluded(config):
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="l2norm_fwd"))
    x = torch.randn(2, 16, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(l2norm_fwd(x), torch.nn.functional.normalize(x.float(), dim=-1).to(x.dtype),
                               atol=.002, rtol=.002)
    assert get_records()[-1]["source"] == "user_override"
    assert "l2norm" in get_records()[-1]["implementation"]


def test_selected_execution_error_never_retries_or_runs_reference():
    set_global_policy(SelectionPolicy(strict=False, reference_exclude="rms_norm"))
    x = torch.zeros(1)
    def original():
        x.add_(1)
        raise RuntimeError("optimized failed after mutation")
    with pytest.raises(RuntimeError, match="after mutation"):
        run_reference("rms_norm", (), {}, (lambda: pytest.fail("reference attempted"),), original)
    assert x.item() == 1
    assert get_records()[-1]["source"] == "user_override_error"


def test_nested_override_suspends_only_reference_guard_and_marks_parent_mixed():
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="rms_norm"))
    lib = torch.library.Library("fl_selection_test", "DEF")
    lib.define("opaque(Tensor x) -> Tensor")
    lib.impl("opaque", lambda x: x + 2, "CPU")
    def parent(x):
        return run_reference("rms_norm", (x,), {}, (), lambda: torch.ops.fl_selection_test.opaque(x))
    out = run_reference("silu_and_mul", (torch.ones(1),), {},
                        (lambda: Candidate("plugin.torch", "parent", parent),))
    assert out.item() == 3
    assert {r["source"] for r in get_records()} == {"user_override", "reference.mixed"}
    with pytest.raises(ReferencePurityError):
        run_reference("silu_and_mul", (torch.ones(1),), {},
                      (lambda: Candidate("plugin.torch", "opaque", torch.ops.fl_selection_test.opaque),))


def test_exclusions_disable_flaggems_registration_even_if_requested(monkeypatch):
    from vllm_fl.utils import is_oot_enabled, use_flaggems
    set_global_policy(SelectionPolicy(reference_exclude="attention"))
    monkeypatch.setenv("USE_FLAGGEMS", "1")
    monkeypatch.setenv("VLLM_FL_OOT_ENABLED", "1")
    assert not use_flaggems()
    assert not is_oot_enabled()


def test_active_flaggems_is_rejected_even_when_all_reference_is_excluded(monkeypatch):
    import flag_gems
    monkeypatch.setattr(flag_gems, "current_work_registrar", object(), raising=False)
    set_global_policy(SelectionPolicy(reference_include=""))
    with pytest.raises(ReferencePurityError, match="fresh reference worker"):
        run_reference("rms_norm", (), {}, (), lambda: pytest.fail("optimized execution"))


def test_attention_override_obeys_vendor_restrictions(config):
    from vllm_fl.platform import PlatformFL
    from vllm.platforms import current_platform
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    vendor = getattr(current_platform, "vendor_name", current_platform.device_name)
    vendor = {"nvidia": "cuda", "mthreads": "musa"}.get(vendor, vendor)
    set_global_policy(SelectionPolicy(reference_exclude="attention", deny_vendors={vendor}))
    with pytest.raises(ReferenceUnavailable, match="denied"):
        PlatformFL.get_attn_backend_cls(AttentionBackendEnum.TRITON_ATTN,
                                      AttentionSelectorConfig(128, torch.bfloat16, "auto", None))


def test_explicit_attention_overrides_dispatch_order(config):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    set_global_policy(SelectionPolicy.from_dict(
        reference_exclude="attention", per_op_order={"attention_backend": ["flagos"]},
    ))
    assert PlatformFL.get_attn_backend_cls(
        AttentionBackendEnum.TRITON_ATTN,
        AttentionSelectorConfig(128, torch.bfloat16, "auto", None),
    ) == "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"


def test_attention_uses_actual_vllm_triton_backend(config):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="attention"))
    path = PlatformFL.get_attn_backend_cls(
        AttentionBackendEnum.TRITON_ATTN,
        AttentionSelectorConfig(128, torch.bfloat16, "auto", None), 8,
    )
    assert path == "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    assert get_records()[-1]["source"] == "user_override"
    assert get_records()[-1]["implementation"] == path


def test_invalid_explicit_attention_fails_without_switching_backend(config):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    set_global_policy(SelectionPolicy(strict=False, reference_exclude="attention"))
    with pytest.raises(ValueError, match="not valid"):
        PlatformFL.get_attn_backend_cls(AttentionBackendEnum.TRITON_ATTN,
            AttentionSelectorConfig(128, torch.bfloat16, "auto", None, use_mla=True))


def test_explicit_backend_requires_exclusion(config):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    with pytest.raises(ValueError, match="REFERENCE_EXCLUDE"):
        PlatformFL.get_attn_backend_cls(AttentionBackendEnum.TRITON_ATTN,
                                      AttentionSelectorConfig(128, torch.bfloat16, "auto", None))


def attention_manager(monkeypatch, fn):
    from vllm_fl.dispatch import manager as module
    manager = OpManager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    manager.registry.register_many([
        OpImpl("attention_backend", "default.flagos", BackendImplKind.DEFAULT, fn),
        OpImpl("attention_backend", "vendor.mock", BackendImplKind.VENDOR,
               lambda **kw: pytest.fail("must not try another backend"), vendor="mock"),
    ])
    monkeypatch.setattr(module, "_default_manager", manager)
    return manager


@pytest.mark.parametrize("is_cuda", [True, False])
def test_explicit_attention_dispatch_works_on_each_platform(config, monkeypatch, is_cuda):
    from vllm.platforms import current_platform
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm_fl.reference.engine import reference_enabled
    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    path = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    attention_manager(monkeypatch, lambda **kw: path)
    set_global_policy(SelectionPolicy.from_dict(
        strict=True, reference_exclude="attention",
        per_op_order={"attention_backend": ["flagos"]},
    ))
    assert PlatformFL.get_attn_backend_cls(
        None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None),
    ) == path
    assert get_records()[-1]["implementation"] == path
    assert {r["source"] for r in get_records()} == {"user_override"}
    assert reference_enabled()
    with pytest.raises(ReferenceUnavailable):
        run_reference("unreviewed", (), {}, (), lambda: pytest.fail("strict fallback"))


def test_attention_dispatch_validates_configuration_without_retry(config, monkeypatch):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    attention_manager(monkeypatch, lambda **kw:
                      "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend")
    set_global_policy(SelectionPolicy.from_dict(
        strict=False, reference_exclude="attention",
        per_op_order={"attention_backend": ["flagos", "vendor"]},
    ))
    with pytest.raises(ValueError, match="MLA not supported"):
        PlatformFL.get_attn_backend_cls(
            None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None, use_mla=True),
        )
    assert get_records()[-1]["source"] == "user_override_error"


def test_attention_dispatch_runtime_error_is_not_retried(config, monkeypatch):
    calls = []
    def fail(**kwargs):
        calls.append(1)
        raise RuntimeError("chosen attention failed")
    manager = attention_manager(monkeypatch, fail)
    set_global_policy(SelectionPolicy.from_dict(
        strict=False, reference_exclude="attention",
        per_op_order={"attention_backend": ["flagos", "vendor"]},
    ))
    with pytest.raises(RuntimeError, match="chosen attention failed"):
        manager.call("attention_backend")
    assert calls == [1]
    assert get_records()[-1]["source"] == "user_override_error"


def test_attention_dispatch_cannot_reenter_torch_reference(config, monkeypatch):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm_fl.reference.attention import PATH
    attention_manager(monkeypatch, lambda **kw: PATH)
    set_global_policy(SelectionPolicy.from_dict(
        strict=True, reference_exclude="attention",
        per_op_order={"attention_backend": ["flagos"]},
    ))
    with pytest.raises(ReferenceUnavailable, match="cannot re-enter"):
        PlatformFL.get_attn_backend_cls(
            None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None),
        )


def test_attention_remains_torch_until_explicitly_excluded(config, monkeypatch):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm_fl.reference.attention import PATH
    attention_manager(monkeypatch, lambda **kw: pytest.fail("optimized attention selected"))
    set_global_policy(SelectionPolicy.from_dict(
        strict=True, per_op_order={"attention_backend": ["flagos"]},
    ))
    assert PlatformFL.get_attn_backend_cls(
        None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None),
    ) == PATH
    assert get_records()[-1]["source"] == "plugin.torch"


def test_reference_defaults_to_triton_without_exclusion_or_cli(config, monkeypatch):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    monkeypatch.delenv("VLLM_FL_REFERENCE_ATTENTION_BACKEND")
    assert selection_reason("attention")
    assert selection_reason("rms_norm") is None
    path = PlatformFL.get_attn_backend_cls(
        None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None),
    )
    assert path == "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    assert get_records()[-1]["source"] == "user_override"
    with pytest.raises(ReferenceUnavailable):
        run_reference("unreviewed", (), {}, (), lambda: pytest.fail("strict fallback"))


def test_default_triton_overrides_real_platform_per_op_defaults(config, monkeypatch):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    monkeypatch.delenv("VLLM_FL_REFERENCE_ATTENTION_BACKEND")
    monkeypatch.setenv("USE_FLAGGEMS", "0")
    # Use the real policy loader and registry, as spawned model workers do.
    set_global_policy(policy_from_env())
    reset_default_manager()
    path = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    assert PlatformFL.get_attn_backend_cls(
        None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None),
    ) == path
    assert call_op("attention_backend", use_mla=False, use_sparse=False) == path


def test_auto_attention_registers_only_the_real_flagos_selector(config, monkeypatch):
    from vllm_fl.platform import PlatformFL
    from vllm_fl.dispatch.manager import get_default_manager
    from vllm_fl.utils import use_flaggems
    from vllm.v1.attention.selector import AttentionSelectorConfig
    monkeypatch.setenv("VLLM_FL_REFERENCE_ATTENTION_BACKEND", "AUTO")
    monkeypatch.setenv("USE_FLAGGEMS", "0")
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_ATTN", "0")
    set_global_policy(SelectionPolicy.from_dict(
        strict=True, per_op_order={"attention_backend": ["flagos"]},
    ))
    reset_default_manager()
    assert PlatformFL.get_attn_backend_cls(
        None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None),
    ) == "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    snapshot = get_default_manager().registry.snapshot()
    defaults = {name for name, impls in snapshot.impls_by_op.items()
                if any(impl.kind == BackendImplKind.DEFAULT for impl in impls)}
    assert defaults == {"attention_backend"}
    assert not use_flaggems()


def test_default_attention_direct_dispatch_matches_platform(config, monkeypatch):
    monkeypatch.delenv("VLLM_FL_REFERENCE_ATTENTION_BACKEND")
    manager = attention_manager(monkeypatch, lambda **kw: pytest.fail("dispatch default leaked"))
    assert manager.call("attention_backend", use_mla=False, use_sparse=False) == (
        "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    )
    assert get_records()[-1]["source"] == "user_override"


def test_default_triton_never_falls_back_when_incompatible(config, monkeypatch):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    monkeypatch.delenv("VLLM_FL_REFERENCE_ATTENTION_BACKEND")
    set_global_policy(SelectionPolicy(strict=False))
    with pytest.raises(ValueError, match="MLA not supported"):
        PlatformFL.get_attn_backend_cls(
            None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None, use_mla=True),
        )
    assert get_records()[-1]["source"] == "user_override_error"


@pytest.mark.parametrize("value", ["tritno_attn", "", "reference"])
def test_invalid_attention_setting_fails_before_model_loading(monkeypatch, value):
    monkeypatch.setenv("VLLM_FL_REFERENCE_ATTENTION_BACKEND", value)
    with pytest.raises(ValueError, match="VLLM_FL_REFERENCE_ATTENTION_BACKEND"):
        configure_reference(VllmConfig())


def test_attention_backend_env_selects_registered_backend(config, monkeypatch):
    from vllm_fl.platform import PlatformFL
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.attention.backends import registry
    # A vendor/custom registration exercises selection without requiring that
    # this machine also install every other accelerator's attention kernels.
    path = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    monkeypatch.setitem(registry._ATTN_OVERRIDES, registry.AttentionBackendEnum.CUSTOM, path)
    monkeypatch.setenv("VLLM_FL_REFERENCE_ATTENTION_BACKEND", "CUSTOM")
    assert PlatformFL.get_attn_backend_cls(
        None, AttentionSelectorConfig(128, torch.bfloat16, "auto", None),
    ) == path


def test_route_report_distinguishes_user_selection_from_missing_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_FL_REFERENCE_REPORT_DIR", str(tmp_path))
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="silu_and_mul"))
    mock_manager().call("silu_and_mul", None, torch.ones(1, 8))
    rows = [json.loads(line) for line in next(tmp_path.glob("reference-*.jsonl")).read_text().splitlines()]
    assert rows[0]["source"] == "user_override"
    assert rows[0]["implementation"] == "vendor.mock"
    assert "excluded" in rows[0]["reason"]
