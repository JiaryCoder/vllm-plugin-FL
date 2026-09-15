"""Regression: configure native construction before the MLA cache writer's K."""
import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm_fl.dispatch import SelectionPolicy, reset_global_policy, set_global_policy
from vllm_fl.reference import clear_records, get_records, ReferenceUnavailable
from vllm_fl.reference.hooks import configure_reference


@pytest.fixture(autouse=True)
def reference(monkeypatch):
    monkeypatch.setenv("VLLM_FL_REFERENCE_MODE", "1")
    monkeypatch.delenv("VLLM_FL_REFERENCE_REPORT_DIR", raising=False)
    set_global_policy(SelectionPolicy(strict=True))
    clear_records()
    cfg = VllmConfig()
    configure_reference(cfg)
    with set_current_vllm_config(cfg):
        yield
    clear_records()
    reset_global_policy()


def rope(dtype, device, neox=True, rotary_dim=64):
    obj = DeepseekScalingRotaryEmbedding(
        64, rotary_dim, 4096, 10000., neox, 40., dtype,
        mscale=0.707, mscale_all_dim=0.707,
    ).to(device)
    assert not obj.use_flashinfer
    assert obj.cos_sin_cache.dtype == dtype
    return obj


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("rotary_dim", [32, 64])
def test_native_initialization_preserves_output_contract(dtype, neox, rotary_dim, monkeypatch):
    obj = rope(dtype, "cpu", neox, rotary_dim)
    positions = torch.tensor([0, 4095, 4096, 32766])
    offsets = torch.ones_like(positions)
    # Both are strided views, as in MultiHeadLatentAttentionWrapper.forward.
    q = torch.randn(4, 8, 192, dtype=dtype)[..., 128:]
    k = torch.randn(4, 1, 576, dtype=dtype)[..., 512:]
    before = (q.clone(), k.clone(), obj.cos_sin_cache.clone())
    def optimized_must_not_run(*a, **kw):
        raise AssertionError("reference called the optimized RoPE")
    monkeypatch.setattr(obj, "_forward_method", optimized_must_not_run)
    result = obj(positions=positions, query=q, key=k, offsets=offsets)
    cos, sin = obj.cos_sin_cache[positions + offsets].double().chunk(2, -1)
    def expected(x):
        rotated = x[..., :rotary_dim].double()
        if neox:
            a, b = rotated.chunk(2, -1)
            rotated = torch.cat((a*cos[:, None]-b*sin[:, None],
                                 b*cos[:, None]+a*sin[:, None]), -1)
        else:
            a, b = rotated[..., ::2], rotated[..., 1::2]
            rotated = torch.stack((a*cos[:, None]-b*sin[:, None],
                                   b*cos[:, None]+a*sin[:, None]), -1).flatten(-2)
        return torch.cat((rotated, x[..., rotary_dim:].double()), -1).to(dtype)
    for output, source in zip(result, (q, k)):
        assert output.dtype == source.dtype
        assert output.shape == source.shape
        assert output.isfinite().all()
        # The actual native path performs its intermediate products in the
        # input dtype; allow its rounding against the FP64 formula.
        tolerance = 2e-6 if dtype == torch.float32 else .02
        torch.testing.assert_close(output, expected(source), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close((q, k, obj.cos_sin_cache), before, rtol=0, atol=0)
    assert get_records()[-1]["implementation"].endswith("forward_native")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA cache writer regression")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mla_cache_writer_receives_correct_dtype(dtype):
    from vllm import _custom_ops as ops
    obj = rope(dtype, "cuda")
    positions = torch.arange(14, device="cuda")
    q = torch.randn(14, 8, 192, device="cuda", dtype=dtype)[..., 128:]
    k = torch.randn(14, 1, 576, device="cuda", dtype=dtype)[..., 512:]
    kv = torch.randn(14, 512, device="cuda", dtype=dtype)
    # Even a direct call of the unmodified upstream method now preserves dtype.
    assert obj.forward_native(positions, q, k)[1].dtype == dtype
    qr, kr = obj(positions, q, k)
    cache = torch.zeros(2, 64, 576, device="cuda", dtype=dtype)
    slots = positions + 64
    ops.concat_and_cache_mla(kv, kr.squeeze(1), cache, slots, "auto",
                            torch.ones(1, device="cuda"))
    expected = torch.cat((kv, kr.squeeze(1)), -1)
    assert cache[1, :14].isfinite().all()
    torch.testing.assert_close(cache[1, :14], expected, rtol=0, atol=0)
    from vllm_fl.reference.engine import optimized_fallback
    from vllm.config.compilation import CompilationMode
    # Build the vendor comparison under its own initialization configuration.
    cfg = VllmConfig()
    cfg.compilation_config.mode = CompilationMode.NONE
    cfg.compilation_config.custom_ops = ["all"]
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="rope"))
    with set_current_vllm_config(cfg):
        vendor_obj = DeepseekScalingRotaryEmbedding(
            64, 64, 4096, 10000., True, 40., dtype,
            mscale=0.707, mscale_all_dim=0.707,
        ).to("cuda")
    assert vendor_obj.use_flashinfer
    assert vendor_obj.cos_sin_cache.dtype == torch.float32
    with optimized_fallback():
        vendor = vendor_obj.forward_cuda(positions, q.clone(), k.clone())
    torch.testing.assert_close((qr, kr), vendor, rtol=.02, atol=.02)


def test_missing_key_and_selection():
    obj = rope(torch.bfloat16, "cpu")
    pos, q = torch.zeros(1, dtype=torch.long), torch.ones(1, 1, 64)
    with pytest.raises(ReferenceUnavailable, match="requires key"):
        obj(pos, q)
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="rope"))
    with pytest.raises(ReferenceUnavailable, match="changed after construction"):
        obj(pos, q, q)
    obj = DeepseekScalingRotaryEmbedding(64, 64, 16, 10000., True, 2., torch.float32)
    obj._forward_method = lambda *a: "vendor"
    assert obj(pos, q, q) == "vendor"
    assert get_records()[-1]["source"] == "user_override"
