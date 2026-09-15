# Copyright (c) 2026 BAAI. All rights reserved.
"""Use vLLM's MLA prefill flow with the installed MetaX FlashAttention API."""
from flash_attn import flash_attn_varlen_func

from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.flash_attn import FlashAttnPrefillBackend


class MacaFlashAttnPrefillBackend(FlashAttnPrefillBackend):
    @classmethod
    def is_available(cls):
        return callable(flash_attn_varlen_func)

    def __init__(self, *args, **kwargs):
        # The upstream constructor selects CUDA/ROCm FlashAttention globals.
        # Keep its common dimensions/config, then bind MetaX's upstream-style
        # API. All prefill methods are inherited unchanged from vLLM.
        MLAPrefillBackend.__init__(self, *args, **kwargs)
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = None  # MetaX does not accept fa_version.
        self.requires_v_padding = True
        self._is_vllm_fa = False  # return_attn_probs, not return_softmax_lse.

    def supports_quant_output(self, quant_key):
        return False
