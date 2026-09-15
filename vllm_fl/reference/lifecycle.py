# Copyright (c) 2026 BAAI. All rights reserved.
"""Keep CustomOp construction and native execution on the same vLLM config.

vLLM's public custom_ops switch is keyed by registered name, which subclasses
can share. Use an isolated config view for each actual class, including during
the whole subclass constructor. Never edit a shared class name or op registry.
"""
from __future__ import annotations

import contextlib
import contextvars
import copy
import functools
import threading
from dataclasses import dataclass

from .engine import ReferenceUnavailable, record_route, reference_requested

_lock = threading.RLock()
_constructing = contextvars.ContextVar("fl_reference_constructing", default=frozenset())
_original_enabled = None
_original_new = None
_original_dispatch = None


@dataclass(frozen=True)
class NativeState:
    identity: str
    selected: bool
    eligible: bool
    native: bool
    problem: str
    config: object


def _identity(cls):
    return cls.__module__ + "." + cls.__qualname__


def _base_config(config):
    return getattr(config, "_fl_reference_base_config", config)


def _entry(cls):
    from vllm.model_executor.custom_op import CustomOp
    from .adapters import CUSTOM_ALIASES, SPECIAL_CUSTOM_ADAPTERS, load
    from .native import discover_class
    target = cls
    if _identity(cls) in SPECIAL_CUSTOM_ADAPTERS:
        return None, "uses a dedicated whole-op adapter rather than forward_native"
    path = CUSTOM_ALIASES.get(_identity(cls))
    if path is not None:
        module, name = path.rsplit(".", 1)
        target = load(module + ":" + name)
    try:
        entry = discover_class(target)
    except ReferenceUnavailable as exc:
        return None, str(exc)
    if getattr(cls.enabled, "__func__", None) not in {
        _original_enabled, getattr(CustomOp.enabled, "__func__", None),
    }:
        return entry, "class overrides CustomOp.enabled; native initialization needs an adapter"
    if cls.dispatch_forward is not CustomOp.dispatch_forward:
        return entry, "class overrides CustomOp.dispatch_forward; native initialization needs an adapter"
    if cls.__new__ not in {_original_new, CustomOp.__new__}:
        return entry, "class overrides CustomOp.__new__; native initialization needs an adapter"
    if not hasattr(cls, "name"):
        return entry, "class has no registered CustomOp name"
    return entry, ""


def _state(cls, config):
    from .selection import register_custom, selection_reason
    from vllm.config.compilation import CompilationMode
    from vllm.config import CUDAGraphMode
    register_custom(cls)
    identity = _identity(cls)
    selected = reference_requested() and selection_reason(identity) is None
    base = _base_config(config)
    entry, problem = _entry(cls)
    native = selected and entry is not None and not problem
    view = copy.copy(base)
    view.compilation_config = copy.copy(base.compilation_config)
    cc = view.compilation_config
    cc.custom_ops = list(base.compilation_config.custom_ops)
    if native:
        # Explicit +/- choices for this registration must not contradict each
        # other. Preserve all unrelated choices, including excluded classes.
        name = cls.name
        cc.custom_ops = [item for item in cc.custom_ops
                         if item not in {"+" + name, "-" + name}]
        cc.custom_ops.append("-" + name)
        cc.mode = CompilationMode.NONE
        cc.cudagraph_mode = CUDAGraphMode.NONE
    view._fl_reference_base_config = base
    view._fl_reference_owner = cls
    return NativeState(identity, selected, entry is not None, native, problem, view)


@contextlib.contextmanager
def _configuration(config):
    from vllm.config import get_current_vllm_config_or_none, set_current_vllm_config
    import vllm.config.vllm as config_module
    # vLLM 0.24's current-config context is process-global. Serialize our
    # overlays and preserve the surrounding prefix and exact config object.
    with _lock:
        if get_current_vllm_config_or_none() is config:
            yield
        else:
            with set_current_vllm_config(
                config, prefix=getattr(config_module, "_current_prefix", None),
            ):
                yield


def state_for(obj):
    return getattr(obj, "_fl_reference_native_state", None)


def require_native_state(obj):
    state = state_for(obj)
    if state is None or not state.native:
        problem = state.problem if state is not None else "object predates native lifecycle setup"
        raise ReferenceUnavailable(
            f"{_identity(type(obj))}: no matching native initialization ({problem}); "
            "create a fresh reference worker or provide an initialization adapter"
        )


def require_fallback_state(obj):
    state = state_for(obj)
    if state is not None and state.native:
        raise ReferenceUnavailable(
            f"{state.identity}: cannot run an optimized fallback on native-initialized "
            "state; exclude this operator before constructing a fresh worker"
        )


@contextlib.contextmanager
def invocation_scope(obj):
    state = state_for(obj)
    if state is None:
        yield
        return
    from .selection import selection_reason
    selected = reference_requested() and selection_reason(state.identity) is None
    if state.eligible and selected != state.selected:
        raise ReferenceUnavailable(
            f"{state.identity}: reference selection changed after construction; "
            "create a fresh worker so initialization and forward agree"
        )
    with _configuration(state.config):
        yield


def _wrap_constructor(cls):
    init = cls.__init__
    if getattr(init, "_fl_native_constructor", False):
        return

    @functools.wraps(init)
    def construct(self, *args, **kwargs):
        state = state_for(self)
        if state is None or id(self) in _constructing.get():
            return init(self, *args, **kwargs)
        token = _constructing.set(_constructing.get() | {id(self)})
        try:
            with _configuration(state.config):
                init(self, *args, **kwargs)
                if state.native:
                    from .native import discover_class
                    from .adapters import CUSTOM_ALIASES, load
                    target = type(self)
                    alias = CUSTOM_ALIASES.get(state.identity)
                    if alias is not None:
                        module, name = alias.rsplit(".", 1)
                        target = load(module + ":" + name)
                    expected = discover_class(target).function
                    actual = getattr(self, "_forward_method", None)
                    actual = getattr(actual, "__func__", actual)
                    if actual is not expected or self.enabled():
                        raise ReferenceUnavailable(
                            f"{state.identity}: constructor overrode native dispatch; "
                            "an initialization adapter is required"
                        )
                    record_route(state.identity, "reference.setup",
                                 expected.__module__ + "." + expected.__qualname__,
                                 "native initialization; custom_ops=" +
                                 repr(state.config.compilation_config.custom_ops))
        finally:
            _constructing.reset(token)

    construct._fl_native_constructor = True
    cls.__init__ = construct


def install_native_lifecycle():
    """Install before any model CustomOp is constructed; safe to call again."""
    global _original_enabled, _original_new, _original_dispatch
    from vllm.model_executor.custom_op import CustomOp
    from vllm.config import get_current_vllm_config_or_none
    if getattr(CustomOp.__new__, "_fl_native_lifecycle", False):
        return
    _original_new = CustomOp.__new__
    _original_enabled = CustomOp.enabled.__func__
    _original_dispatch = CustomOp.dispatch_forward

    @functools.wraps(_original_new)
    def new(cls, *args, **kwargs):
        if not reference_requested():
            return _original_new(cls, *args, **kwargs)
        config = get_current_vllm_config_or_none()
        if config is None:
            return _original_new(cls, *args, **kwargs)
        with _lock:
            state = _state(cls, config)
            if state.native:
                # Native methods must receive an upstream instance. OOT
                # registries may already exist when another plugin loaded.
                obj = object.__new__(cls)
            else:
                obj = _original_new(cls, *args, **kwargs)
                if type(obj) is not cls:
                    state = _state(type(obj), config)
            _wrap_constructor(type(obj))
            object.__setattr__(obj, "_fl_reference_native_state", state)
            return obj

    @classmethod
    @functools.wraps(_original_enabled)
    def enabled(cls):
        if not reference_requested():
            return _original_enabled(cls)
        config = get_current_vllm_config_or_none()
        if config is None or getattr(config, "_fl_reference_owner", None) is cls:
            return _original_enabled(cls)
        state = _state(cls, config)
        with _configuration(state.config):
            return _original_enabled(cls)

    @functools.wraps(_original_dispatch)
    def dispatch(self, compile_native):
        method = _original_dispatch(self, compile_native=compile_native)
        state = state_for(self)
        if state is not None and not state.native:
            from vllm.platforms import current_platform
            # PlatformFL is OOT even on NVIDIA. The base OOT method simply
            # calls native on an optimized-initialized object. For explicit
            # vendor paths, resolve that generic stub to the real CUDA entry.
            if (getattr(method, "__func__", method) is CustomOp.forward_oot
                    and current_platform.is_cuda()):
                return self.forward_cuda
        return method

    new._fl_native_lifecycle = True
    CustomOp.__new__ = staticmethod(new)
    CustomOp.enabled = enabled
    CustomOp.dispatch_forward = dispatch
