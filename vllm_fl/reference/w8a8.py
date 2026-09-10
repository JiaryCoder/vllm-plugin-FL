# Copyright (c) 2026 BAAI. All rights reserved.
"""Portable dynamic symmetric W8A8, including exact integer accumulation."""
import torch

from .engine import ReferenceUnavailable, run_reference, tensor_support
from .functions import candidate, check_signature, patch


def integer_mm(a, b):
    """Exact INT8 dot products using only ATen, with bounded temporary storage."""
    if a.device.type == "cpu":
        return a.to(torch.int32) @ b.to(torch.int32)
    # CUDA ATen addmm does not implement int32. Avoid an opaque int8 GEMM and
    # avoid float32 dot products, whose integer exactness ends at 2**24.
    output = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.int32)
    for row in range(0, a.shape[0], 32):
        for col in range(0, b.shape[1], 32):
            acc = torch.zeros((min(32, a.shape[0] - row), min(32, b.shape[1] - col)),
                              device=a.device, dtype=torch.int64)
            for k in range(0, a.shape[1], 256):
                left = a[row:row+32, k:k+256].to(torch.int32)
                right = b[k:k+256, col:col+32].to(torch.int32)
                acc += (left[:, :, None] * right[None, :, :]).sum(1, dtype=torch.int64)
            output[row:row+32, col:col+32] = acc.to(torch.int32)
    return output


def scaled_mm(a, b, scale_a, scale_b, bias=None, out_dtype=None):
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("symmetric W8A8 scaled_mm requires int8 operands")
    if scale_a.numel() != a.shape[0] or scale_b.numel() != b.shape[1]:
        raise ValueError("scaled_mm requires one scale per token/output channel")
    result = integer_mm(a, b).float()
    result *= scale_a.float().reshape(-1, 1)
    result *= scale_b.float().reshape(1, -1)
    if bias is not None:
        result += bias.float()
    return result.to(out_dtype or torch.float32)


def linear_apply(self, layer, x, bias=None):
    from vllm_fl.quantization.w8a8.reference import w8a8_linear_reference
    weight, weight_scale, input_scale, zp, azp = self._get_layer_params(layer)
    return w8a8_linear_reference(x, weight.t(), weight_scale, bias)


def _linear_support(self, layer, x, bias=None):
    reason = tensor_support(x, bias)
    if reason:
        return reason
    config = self.config
    if config.is_static_input_scheme or not config.input_symmetric or not config.is_channelwise:
        return "only dynamic symmetric per-token/per-channel W8A8 linear is audited"
    return None


def init_linear(is_channelwise, is_static_input_scheme, input_symmetric, module_name):
    from vllm.model_executor.kernels.linear import Int8ScaledMMLinearLayerConfig
    from vllm_fl.quantization.w8a8.linear import FLW8A8DynamicLinearKernel
    class TorchW8A8LinearKernel(FLW8A8DynamicLinearKernel):
        @classmethod
        def is_supported(cls, compute_capability=None):
            return True, None

        def apply_weights(self, layer, x, bias=None):
            original = super().apply_weights
            return run_reference(
                "w8a8_linear", (self, layer, x, bias), {},
                (lambda: candidate(linear_apply, supports=_linear_support),),
                lambda: original(layer, x, bias),
            )
    config = Int8ScaledMMLinearLayerConfig(
        is_channelwise=is_channelwise, is_static_input_scheme=is_static_input_scheme,
        input_symmetric=input_symmetric)
    return TorchW8A8LinearKernel(config, [
        "weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj",
    ])


def _init_support(is_channelwise, is_static_input_scheme, input_symmetric, module_name):
    if not is_channelwise or is_static_input_scheme or not input_symmetric:
        return "only dynamic symmetric per-channel W8A8 linear is audited"


def install():
    from vllm_fl.quantization.w8a8.reference import unpack_uint8b128_int32
    patch("vllm.model_executor.kernels.linear:init_int8_linear_kernel",
          (lambda: candidate(init_linear, supports=_init_support),))
    patch("vllm_fl.quantization.w8a8.linear:FLW8A8DynamicLinearKernel.apply_weights",
          (lambda: candidate(linear_apply, supports=_linear_support),))
    # The unpack implementation was already pure torch; record actual load-time use.
    patch("vllm_fl.quantization.w8a8.packed:unpack_uint8b128_int32",
          (lambda: candidate(unpack_uint8b128_int32),))
