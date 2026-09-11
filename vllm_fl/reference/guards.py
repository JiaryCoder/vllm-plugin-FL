# Copyright (c) 2026 BAAI. All rights reserved.
"""Execution guards for native discovery; ordinary and excluded calls pass through."""
from __future__ import annotations

import functools
import importlib

from .engine import ReferencePurityError, reference_execution_active


def _guard_method(cls, name, *, returns_launcher=False):
    original = getattr(cls, name)
    if getattr(original, "_fl_reference_launch_guard", False):
        return

    def check():
        if reference_execution_active():
            raise ReferencePurityError(
                f"Reference attempted direct Triton launch: {cls.__name__}.{name}"
            )

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        check()
        result = original(*args, **kwargs)
        if not returns_launcher:
            return result
        # The launch closure can be obtained outside reference and invoked later.
        @functools.wraps(result)
        def launch(*a, **k):
            check()
            return result(*a, **k)
        return launch

    guarded._fl_reference_launch_guard = True
    setattr(cls, name, guarded)


def install_launch_guards():
    # Patch entry points, not model-specific kernel names. Context-local checks
    # keep other threads and intentional user exclusions on their normal route.
    for module_name, class_name, method, launcher in (
        ("triton.runtime.jit", "JITFunction", "run", False),
        ("triton.runtime.autotuner", "Autotuner", "run", False),
        ("triton.runtime.autotuner", "Heuristics", "run", False),
        ("triton.compiler.compiler", "CompiledKernel", "__getitem__", True),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue  # CPU-only builds may not install Triton.
        _guard_method(getattr(module, class_name), method, returns_launcher=launcher)


def math_function_mode():
    """Force the math implementation for PyTorch's public SDPA interface."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.overrides import TorchFunctionMode

    class NativeMath(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is F.scaled_dot_product_attention and reference_execution_active():
                with sdpa_kernel(SDPBackend.MATH):
                    return func(*args, **(kwargs or {}))
            return func(*args, **(kwargs or {}))

    return NativeMath()
