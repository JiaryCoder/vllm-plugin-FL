# Copyright (c) 2026 BAAI. All rights reserved.
"""Select a MetaX MLA implementation matching the installed vLLM interface."""
from importlib import import_module
import inspect


LEGACY_BACKEND = (
    "vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.flashmla."
    "MacaFlashMLABackend"
)
VENDOR_BACKEND = (
    "vllm_metax.v1.attention.backends.mla.flashmla.MacaFlashMLABackend"
)
COMPAT_BACKEND = (
    "vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.vendor_flashmla."
    "MacaFlashMLABackend"
)
PREFILL_BACKEND = (
    "vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.prefill."
    "MacaFlashAttnPrefillBackend"
)


def register_mla_prefill(backend_path):
    if backend_path not in {VENDOR_BACKEND, COMPAT_BACKEND}:
        return
    name = "vllm.v1.attention.backends.mla.prefill.registry"
    try:
        registry = import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
            return  # Earlier split-forward vLLM has no separate prefill registry.
        raise
    registry.register_mla_prefill_backend(
        registry.MLAPrefillBackendEnum.FLASH_ATTN, class_path=PREFILL_BACKEND,
    )


def dense_mla_backend_path():
    from vllm.v1.attention.backend import MLAAttentionImpl

    # The bundled implementation has the older single-forward contract.
    # vLLM now owns cache updates and projection and calls these two methods.
    # Renaming its old forward would duplicate that work and corrupt results.
    split_forward = {"forward_mha", "forward_mqa"}
    if not split_forward.intersection(MLAAttentionImpl.__abstractmethods__):
        return LEGACY_BACKEND

    module, name = VENDOR_BACKEND.rsplit(".", 1)
    try:
        backend = getattr(import_module(module), name)
        impl = backend.get_impl_cls()
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This vLLM requires the split MLA forward_mha/forward_mqa interface. "
            "Use a matching vendor vllm-metax installation providing "
            f"{VENDOR_BACKEND}; FL's bundled legacy MLA backend is incompatible."
        ) from exc

    if (not inspect.isclass(impl) or not issubclass(impl, MLAAttentionImpl)
            or inspect.isabstract(impl)):
        missing = sorted(getattr(impl, "__abstractmethods__", ()))
        raise RuntimeError(
            f"{VENDOR_BACKEND} is incompatible with the installed vLLM "
            f"MLAAttentionImpl (unimplemented methods: {missing}). "
            "Install matching vendor vLLM and vllm-metax packages."
        )
    return COMPAT_BACKEND
