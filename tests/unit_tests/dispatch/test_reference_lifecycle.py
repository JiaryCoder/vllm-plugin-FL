"""Native configuration must cover initialization, execution and exclusions."""
import concurrent.futures
import time

import pytest
import torch

from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.activation import NewGELU, QuickGELU
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import DeepseekScalingRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding
from vllm_fl.dispatch import SelectionPolicy, reset_global_policy, set_global_policy
from vllm_fl.reference import ReferenceUnavailable, clear_records, get_records
from vllm_fl.reference.hooks import configure_reference
from vllm_fl.reference.lifecycle import invocation_scope, state_for


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE", "1")
    monkeypatch.delenv("VLLM_FL_REFERENCE_REPORT_DIR", raising=False)
    set_global_policy(SelectionPolicy(strict=True))
    cfg = VllmConfig(compilation_config=CompilationConfig(
        mode=CompilationMode.NONE, custom_ops=["all", "+rotary_embedding"],
    ))
    configure_reference(cfg)
    clear_records()
    with set_current_vllm_config(cfg):
        yield cfg
    clear_records()
    reset_global_policy()


def deepseek():
    return DeepseekScalingRotaryEmbedding(64, 64, 128, 10000., True, 2., torch.bfloat16)


def test_environment_only_controls_full_deepseek_initialization(config):
    before = list(config.compilation_config.custom_ops)
    obj = deepseek()
    assert not obj.enabled()
    assert not obj.use_flashinfer
    assert obj.cos_sin_cache.dtype == torch.bfloat16
    assert obj._forward_method.__func__ is DeepseekScalingRotaryEmbedding.forward_native
    q, k = torch.ones(2, 8, 64, dtype=torch.bfloat16), torch.ones(2, 1, 64, dtype=torch.bfloat16)
    result = obj(torch.arange(2), q, k)
    assert all(t.dtype == torch.bfloat16 for t in result)
    assert config.compilation_config.custom_ops == before
    assert get_current_vllm_config() is config
    assert any(row["source"] == "reference.setup" for row in get_records())
    assert get_records()[-1]["implementation"].endswith(".forward_native")


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_registration_does_not_change_excluded_class(config, reverse):
    assert DeepseekScalingRotaryEmbedding.name == Llama3RotaryEmbedding.name
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="deepseek_scaling_rope"))
    def llama():
        return Llama3RotaryEmbedding(64, 64, 256, 10000., True, torch.bfloat16, 4., 1., 4., 64)
    objs = [fn() for fn in ([llama, deepseek] if reverse else [deepseek, llama])]
    excluded, native = (objs[::-1] if reverse else objs)
    assert excluded.enabled() and excluded.use_flashinfer
    assert excluded.cos_sin_cache.dtype == torch.float32
    assert excluded._forward_method.__func__ is DeepseekScalingRotaryEmbedding.forward_cuda
    assert not native.enabled() and state_for(native).native
    assert state_for(excluded).config.compilation_config.custom_ops == ["all", "+rotary_embedding"]
    assert state_for(native).config.compilation_config.custom_ops == ["all", "-rotary_embedding"]
    assert config.compilation_config.custom_ops == ["all", "+rotary_embedding"]


def test_constructor_observes_config_before_super_and_forward_observes_same_view(config, monkeypatch):
    original = NewGELU.__init__
    observed = []
    def init(self, *args, **kwargs):
        observed.append((self.enabled(), list(get_current_vllm_config().compilation_config.custom_ops)))
        original(self, *args, **kwargs)
    monkeypatch.setattr(NewGELU, "__init__", init)
    obj = NewGELU()
    assert observed[0][0] is False
    assert "-" + NewGELU.name in observed[0][1]
    with invocation_scope(obj):
        assert get_current_vllm_config().compilation_config.custom_ops == observed[0][1]
    assert get_current_vllm_config() is config


def test_nested_excluded_child_uses_original_configuration(config, monkeypatch):
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="deepseek_scaling_rope"))
    original = NewGELU.__init__
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.child = deepseek()
    monkeypatch.setattr(NewGELU, "__init__", init)
    parent = NewGELU()
    assert state_for(parent).native
    assert parent.child.use_flashinfer
    assert parent.child.cos_sin_cache.dtype == torch.float32
    assert get_current_vllm_config() is config


def test_constructor_failure_restores_config(config, monkeypatch):
    def broken(self):
        assert not self.enabled()
        raise RuntimeError("constructor failed")
    monkeypatch.setattr(NewGELU, "__init__", broken)
    with pytest.raises(RuntimeError, match="constructor failed"):
        NewGELU()
    assert get_current_vllm_config() is config
    assert "-" + NewGELU.name not in config.compilation_config.custom_ops


def test_constructor_cannot_silently_override_native_dispatch(monkeypatch):
    original = NewGELU.__init__
    def bad(self):
        original(self)
        self._forward_method = self.forward_cuda
    monkeypatch.setattr(NewGELU, "__init__", bad)
    with pytest.raises(ReferenceUnavailable, match="constructor overrode"):
        NewGELU()


def test_forced_enable_is_rejected(monkeypatch):
    monkeypatch.setattr(NewGELU, "__init__", lambda self: CustomOp.__init__(self, enforce_enable=True))
    with pytest.raises(ReferenceUnavailable, match="enforce_enable"):
        NewGELU()


def test_overridden_enabled_does_not_claim_native_initialization(monkeypatch):
    monkeypatch.setattr(NewGELU, "enabled", classmethod(lambda cls: True))
    with pytest.raises(ReferenceUnavailable, match="overrides CustomOp.enabled"):
        NewGELU()(torch.ones(1))


def test_native_selection_bypasses_existing_oot_class_without_removing_registry(monkeypatch):
    class VendorNewGELU(NewGELU):
        def __init__(self):
            raise AssertionError("optimized constructor must not run")
    monkeypatch.setitem(op_registry_oot, "NewGELU", VendorNewGELU)
    obj = NewGELU()
    assert type(obj) is NewGELU
    assert state_for(obj).native
    assert op_registry_oot["NewGELU"] is VendorNewGELU


def test_non_strict_missing_native_keeps_optimized_initialization(monkeypatch):
    set_global_policy(SelectionPolicy(strict=False))
    monkeypatch.setattr(NewGELU, "forward_native", CustomOp.forward_native)
    monkeypatch.setattr(NewGELU, "forward_cuda", lambda self, x: x + 7)
    obj = NewGELU()
    assert not state_for(obj).native
    torch.testing.assert_close(obj(torch.ones(2)), torch.full((2,), 8.))
    assert get_records()[-1]["source"] == "optimized_fallback"


def test_late_unsupported_input_does_not_reuse_native_state_for_vendor(monkeypatch):
    set_global_policy(SelectionPolicy(strict=False))
    obj = NewGELU()
    obj._forward_method = lambda x: pytest.fail("unsafe optimized retry")
    with pytest.raises(ReferenceUnavailable, match="native-initialized state"):
        obj(torch.ones(2, dtype=torch.float64))


def test_new_worker_required_after_selection_changes():
    obj = deepseek()
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="rope"))
    with pytest.raises(ReferenceUnavailable, match="changed after construction"):
        obj(torch.arange(1), torch.ones(1, 1, 64), torch.ones(1, 1, 64))


def test_config_scopes_restore_across_threads(config):
    objects = [NewGELU(), QuickGELU()]
    def check(obj):
        for _ in range(5):
            with invocation_scope(obj):
                current = get_current_vllm_config()
                time.sleep(.001)
                assert get_current_vllm_config() is current
                assert not obj.enabled()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(check, objects):
            assert result is None
    assert get_current_vllm_config() is config


def test_hooks_do_not_change_ordinary_mode(config, monkeypatch):
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE", "0")
    obj = deepseek()
    assert obj.use_flashinfer
    assert obj.cos_sin_cache.dtype == torch.float32
    assert state_for(obj) is None
    assert config.compilation_config.custom_ops == ["all", "+rotary_embedding"]
