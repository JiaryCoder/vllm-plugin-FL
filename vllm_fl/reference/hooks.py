# Copyright (c) 2026 BAAI. All rights reserved.
"""Idempotent, process-local hooks installed before model construction."""
from __future__ import annotations

import copy
import functools

from .adapters import (
    IR_OPS, custom_identity, native_ir, plugin_custom, upstream_custom,
)
from .engine import (
    Candidate, ReferenceUnavailable, optimized_fallback, record_route,
    reference_enabled, reference_requested, run_reference, tensor_support,
    run_user_override,
)
from .selection import selection_reason


def configure_reference(vllm_config):
    """Establish eager execution in every process receiving the vLLM config."""
    if not reference_requested():
        return
    import torch
    # Reference FP32 matrix products must not silently use TF32 mantissas.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationMode

    cc = vllm_config.compilation_config
    cc.mode = CompilationMode.NONE
    cc.cudagraph_mode = CUDAGraphMode.NONE
    if vllm_config.model_config is not None:
        vllm_config.model_config.enforce_eager = True
    priorities = vllm_config.kernel_config.ir_op_priority
    for op in IR_OPS:
        if selection_reason("ir." + op) is None:
            setattr(priorities, op, ["native"])
    install_reference_hooks()
    from .guards import install_launch_guards
    install_launch_guards()
    from .functions import install_function_hooks
    install_function_hooks()


def install_reference_hooks():
    if not reference_requested():
        return
    from vllm.model_executor.custom_op import CustomOp
    from vllm.ir.op import IrOp

    if not getattr(CustomOp.__init__, "_fl_reference_hook", False):
        original_init = CustomOp.__init__

        @functools.wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not reference_requested():
                return
            original = self.forward
            identity = custom_identity(self)
            from .selection import register_custom
            register_custom(type(self))

            @functools.wraps(original)
            def routed(*a, **k):
                if not reference_enabled():
                    return original(*a, **k)
                def fallback():
                    with optimized_fallback():
                        result = original(*a, **k)
                    selected = (getattr(self, "_forward_method", original)
                                if getattr(original, "__qualname__", "") == "CustomOp.forward"
                                else original)
                    implementation = (getattr(selected, "__module__", "") + "." +
                                      getattr(selected, "__qualname__", type(selected).__name__))
                    record_route(identity, "optimized_fallback", implementation,
                                 "no audited class/variant reference")
                    return result
                from .native import custom_candidate
                return run_reference(
                    identity, a, k,
                    (lambda: upstream_custom(self), lambda: plugin_custom(self),
                     lambda: custom_candidate(self)),
                    fallback,
                )

            # Wrap the public entry, because subclasses may replace
            # _forward_method after super().__init__ returns (e.g. GDN).
            self.forward = routed

        init._fl_reference_hook = True
        CustomOp.__init__ = init

    if not getattr(IrOp.dispatch, "_fl_reference_hook", False):
        original_dispatch = IrOp.dispatch

        @functools.wraps(original_dispatch)
        def dispatch(self, *args, **kwargs):
            if not reference_enabled():
                return original_dispatch(self, *args, **kwargs)
            reason = selection_reason("ir." + self.name)
            if reason is not None:
                with optimized_fallback():
                    impl = original_dispatch(self, *args, **kwargs)
                selected = copy.copy(impl)
                def invoke_selected(*a, **k):
                    return run_user_override(
                        "ir." + self.name, lambda: impl.impl_fn(*a, **k),
                        impl.provider, reason,
                    )
                selected.impl_fn = invoke_selected
                return selected
            def factory():
                if self.name not in IR_OPS:
                    from .native import ir_candidate
                    return ir_candidate(self)
                fn = native_ir(self.name)
                return Candidate("vllm.native", f"vllm.ir.{self.name}.native",
                                 fn, tensor_support)
            # Copy instead of mutating the global provider shared with ordinary
            # inference. Functional/native also satisfies maybe_inplace's contract.
            template = self.impls.get("native")
            if template is None:
                raise ReferenceUnavailable(f"IR {self.name} has no native provider")
            routed_impl = copy.copy(template)
            def invoke(*a, **k):
                def fallback():
                    with optimized_fallback():
                        impl = original_dispatch(self, *a, **k)
                        result = impl.func_impl_fn(*a, **k)
                    record_route("ir." + self.name, "optimized_fallback",
                                 impl.provider, "no audited reference supports this call")
                    return result
                from .adapters import plugin_ir
                return run_reference(
                    "ir." + self.name, a, k,
                    (factory, lambda: plugin_ir(self.name)), fallback,
                )
            routed_impl.impl_fn = invoke
            return routed_impl

        dispatch._fl_reference_hook = True
        IrOp.dispatch = dispatch
