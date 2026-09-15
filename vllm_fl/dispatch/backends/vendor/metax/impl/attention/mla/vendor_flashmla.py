# Copyright (c) 2026 BAAI. All rights reserved.
"""Bridge optional vLLM arguments to the installed vendor MLA implementation."""
import inspect

from vllm_metax.v1.attention.backends.mla.flashmla import (
    FlashMLAImpl as VendorFlashMLAImpl,
    MacaFlashMLABackend as VendorFlashMLABackend,
)


_mha_parameters = inspect.signature(VendorFlashMLAImpl.forward_mha).parameters
_accepts_output_scale = "output_scale" in _mha_parameters or any(
    p.kind == inspect.Parameter.VAR_KEYWORD for p in _mha_parameters.values()
)


class MacaFlashMLAImpl(VendorFlashMLAImpl):
    def forward_mha(self, *args, output_scale=None, **kwargs):
        if _accepts_output_scale:
            kwargs["output_scale"] = output_scale
        elif output_scale is not None:
            raise NotImplementedError(
                "The installed MetaX MLA backend does not support output_scale; "
                "fused prefill output quantization is unavailable."
            )
        # Older vendor methods omit this optional argument. None requests the
        # same unquantized output they already implement; tensor scales must
        # never be silently discarded.
        return super().forward_mha(*args, **kwargs)


class MacaFlashMLABackend(VendorFlashMLABackend):
    @staticmethod
    def get_impl_cls():
        return MacaFlashMLAImpl
