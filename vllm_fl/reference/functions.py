# Copyright (c) 2026 BAAI. All rights reserved.
"""Route audited free functions and methods, including existing import aliases."""
from __future__ import annotations

import functools
import importlib
import inspect
import sys

from .engine import (
    Candidate, optimized_fallback, record_route, reference_enabled, run_reference,
    tensor_support,
)


def candidate(fn, source="plugin.torch", supports=None):
    return Candidate(source, fn.__module__ + "." + fn.__qualname__,
                     fn, supports or tensor_support)


def check_signature(fn, validate):
    signature = inspect.signature(fn)
    def supports(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return tensor_support(*args, **kwargs) or validate(**bound.arguments)
    return supports


def patch(path, factories):
    module_name, attributes = path.split(":")
    module = importlib.import_module(module_name)
    owner = module
    names = attributes.split(".")
    for name in names[:-1]:
        owner = getattr(owner, name)
    name = names[-1]
    original = getattr(owner, name)
    if getattr(original, "_fl_reference_function", False):
        return
    @functools.wraps(original)
    def routed(*args, **kwargs):
        if not reference_enabled():
            return original(*args, **kwargs)
        def fallback():
            with optimized_fallback():
                result = original(*args, **kwargs)
            record_route(path, "optimized_fallback",
                         original.__module__ + "." + original.__qualname__,
                         "no audited reference supports this call")
            return result
        return run_reference(path, args, kwargs, factories, fallback)
    routed._fl_reference_function = True
    routed._fl_reference_original = original
    setattr(owner, name, routed)
    if owner is module:
        # from module import func aliases created before worker initialization.
        # Identity matching prevents changing unrelated same-named functions.
        for imported_name, imported in tuple(sys.modules.items()):
            if imported_name.startswith(("vllm.", "vllm_fl.")) and imported is not None:
                for attr, value in tuple(vars(imported).items()):
                    if value is original:
                        setattr(imported, attr, routed)


def install_function_hooks():
    from . import moe, w8a8, fla, attention
    moe.install()
    w8a8.install()
    fla.install()
    attention.install()
