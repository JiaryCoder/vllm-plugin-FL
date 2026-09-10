# Copyright (c) 2026 BAAI. All rights reserved.
"""Torch fallback matching vLLM's FP32 RMSNorm intermediate semantics."""
from typing import Optional, Union
import torch


def rms_norm_torch(obj, x: torch.Tensor, residual: Optional[torch.Tensor] = None
                   ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    orig_dtype = x.dtype
    has_residual = residual is not None
    x = x.to(torch.float32)
    if has_residual:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
    variance_size = getattr(obj, "variance_size_override", None)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + obj.variance_epsilon)
    pass_weight = getattr(
        obj, "pass_weight_add" if has_residual else "pass_weight",
        getattr(obj, "has_weight", True),
    )
    if pass_weight:
        x = x.to(obj.weight.dtype) * obj.weight
    output = x.to(orig_dtype)
    return (output, residual) if has_residual else output
