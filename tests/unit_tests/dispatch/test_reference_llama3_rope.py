# Copyright (c) 2026 BAAI. All rights reserved.
"""Audit the exact Llama 3 scaled RoPE used by the Llama 3.1 benchmark."""
import math

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding
from vllm_fl.dispatch import SelectionPolicy, set_global_policy, reset_global_policy
from vllm_fl.reference import get_records, clear_records, ReferenceUnavailable
from vllm_fl.reference.hooks import configure_reference
from vllm_fl.reference.selection import selection_reason


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


def make_rope(dtype, neox=True, rotary_dim=128):
    return Llama3RotaryEmbedding(128, rotary_dim, 40960, 500000., neox, dtype,
                                8., 1., 4., 8192)


def test_scaled_frequencies_have_high_transition_and_low_bands():
    obj = make_rope(torch.float32)
    frequencies = 500000. ** (-torch.arange(0, 128, 2, dtype=torch.float64) / 128)
    cycles = frequencies * 8192 / (2 * math.pi)
    blend = ((cycles - 1) / 3).clamp(0, 1)
    expected = frequencies * (blend + (1 - blend) / 8)
    assert (cycles <= 1).any() and ((cycles > 1) & (cycles < 4)).any() and (cycles >= 4).any()
    torch.testing.assert_close(obj._compute_inv_freq(500000.).double(), expected, rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("rotary_dim", [64, 128])
@pytest.mark.parametrize("with_key", [True, False])
def test_long_position_forward_matches_vendor(dtype, neox, rotary_dim, with_key):
    obj = make_rope(dtype, neox, rotary_dim).to("cuda")
    obj.use_flashinfer = False
    positions = torch.tensor([0, 31, 8191, 8192, 32767, 40959], device="cuda")
    q = torch.randn(6, 4 * 128, device="cuda", dtype=dtype)
    k = torch.randn(6, 2 * 128, device="cuda", dtype=dtype) if with_key else None
    expected = obj.forward_cuda(positions, q.clone(), None if k is None else k.clone())
    actual = obj(positions, q, k)
    atol = 2e-6 if dtype == torch.float32 else .02
    torch.testing.assert_close(actual, expected, rtol=atol, atol=atol)
    row = get_records()[-1]
    assert row["source"] == "vllm.native.dynamic"
    assert row["op"].endswith("Llama3RotaryEmbedding")
    assert row["implementation"].endswith("base.RotaryEmbedding.forward_native")


def test_selection_and_unknown_subclasses():
    set_global_policy(SelectionPolicy(strict=True, reference_exclude="rope"))
    assert selection_reason("vllm.model_executor.layers.rotary_embedding.llama3_rope.Llama3RotaryEmbedding")
    set_global_policy(SelectionPolicy(strict=True))
    class OtherLlama3(Llama3RotaryEmbedding):
        pass
    obj = OtherLlama3(128, 128, 16, 500000., True, torch.float32, 8., 1., 4., 8192)
    with pytest.raises(ReferenceUnavailable, match="OtherLlama3"):
        obj(torch.tensor([0]), torch.ones(1, 128))
