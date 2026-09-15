# Copyright (c) 2026 BAAI. All rights reserved.
"""Reproducible stage-one acceptance without downloading model weights."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ["VLLM_FL_REFERENCE_MODE"] = "1"
    os.environ["VLLM_FL_STRICT"] = "1"
    # A pre-existing user YAML would override strictness; this smoke controls
    # only its own process and explicitly selects the strict policy below.
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm import ir
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    from vllm_fl.dispatch import call_op, SelectionPolicy, set_global_policy
    from vllm_fl.reference import clear_records, get_records
    from vllm_fl.reference.hooks import configure_reference
    from types import SimpleNamespace

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(42)
    cfg = VllmConfig()
    configure_reference(cfg)
    set_global_policy(SelectionPolicy(strict=True))
    clear_records()
    checks = {}
    with set_current_vllm_config(cfg), torch.no_grad():
        x = torch.randn(4, 24, device=device, dtype=dtype)
        for name, activation, obj in [
            ("silu_and_mul", torch.nn.functional.silu, None),
            ("gelu_and_mul", torch.nn.functional.gelu,
             SimpleNamespace(approximate="none")),
        ]:
            output = call_op(name, obj, x)
            torch.testing.assert_close(output, activation(x[:, :12]) * x[:, 12:])
            checks[name] = True
        norm = RMSNorm(24, dtype=dtype).to(device)
        big = torch.full_like(x, 300)
        result, residual = call_op("rms_norm", norm, big, torch.zeros_like(big))
        torch.testing.assert_close(result, torch.ones_like(big))
        torch.testing.assert_close(residual, big)
        torch.testing.assert_close(norm(big), torch.ones_like(big))
        checks["rms_norm"] = True
        pos = torch.zeros(4, device=device, dtype=torch.long)
        q = x.reshape(4, 2, 12)
        k = x[:, :12].reshape(4, 1, 12)
        cos = torch.ones(8, 6, device=device, dtype=dtype)
        sin = torch.zeros_like(cos)
        qr, kr = call_op("rotary_embedding", None, q, k, cos, sin, pos)
        torch.testing.assert_close(qr, q)
        torch.testing.assert_close(kr, k)
        checks["rotary_embedding"] = True
        q8, scales = call_op("dynamic_per_token_quant_int8", x)
        assert q8.dtype == torch.int8 and scales.dtype == torch.float32
        assert q8.shape == x.shape and scales.shape == (4, 1)
        checks["dynamic_per_token_quant_int8"] = True

        rope = RotaryEmbedding(12, 12, 8, 10000, True, dtype).to(device)
        r, _ = rope(pos, x, x[:, :12])
        torch.testing.assert_close(r, x)
        helper = ApplyRotaryEmb()
        torch.testing.assert_close(helper(q, cos[:4], sin[:4]), q)
        for interleaved in (False, True):
            mrope = MRotaryEmbedding(12, 12, 8, 10000, True, dtype,
                                    [2, 2, 2], interleaved).to(device)
            r, _ = mrope(pos.expand(3, -1), x, x[:, :12])
            torch.testing.assert_close(r, x)
        checks["generic_rope_apply_mrope"] = True
        for residual_present in (False, True):
            if residual_present:
                result, _ = ir.ops.fused_add_rms_norm.maybe_inplace(
                    big, torch.zeros_like(big), norm.weight, 1e-6, None)
            else:
                result = ir.ops.rms_norm(big, norm.weight, 1e-6, None)
            torch.testing.assert_close(result, torch.ones_like(big))
        checks["ir_rms_and_add_rms"] = True

    records = get_records()
    assert records and all(
        row["source"] in {"vllm.native", "plugin.torch"} for row in records
    )
    result = {
        "status": "passed", "device": str(device), "dtype": str(dtype),
        "torch": torch.__version__, "checks": checks, "routes": records,
        "compilation_mode": str(cfg.compilation_config.mode),
        "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
