# Copyright (c) 2026 BAAI. All rights reserved.
"""Discover in-tree vLLM native interfaces without a per-class support list.

Discovery establishes the callable's origin and interface, not a proof of its
whole Python call graph. Execution remains guarded and records dynamic routes
separately from the previously audited adapters.
"""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .engine import Candidate, ReferenceUnavailable, tensor_support


def _upstream_origin(value):
    """Require a real Python definition in the installed vLLM package."""
    module_name = getattr(value, "__module__", "")
    if not module_name.startswith("vllm."):
        raise ReferenceUnavailable(f"{module_name}: native definition is not from vLLM")
    module = sys.modules.get(module_name)
    import vllm
    roots = [Path(path).resolve() for path in vllm.__path__]
    try:
        definition = Path(inspect.getfile(value)).resolve()
        module_file = Path(module.__file__).resolve()
    except (TypeError, AttributeError) as exc:
        raise ReferenceUnavailable("native definition has no inspectable Python origin") from exc
    if not any(definition.is_relative_to(root) and module_file.is_relative_to(root)
               for root in roots):
        raise ReferenceUnavailable(f"{module_name}: native definition is outside vLLM")
    return module


def _check_body(fn):
    if not inspect.isfunction(fn):
        raise ReferenceUnavailable("native interface is not a Python function")
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError, IndentationError) as exc:
        raise ReferenceUnavailable("native Python source is unavailable") from exc
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ReferenceUnavailable("native interface has no function definition")
    body = [item for item in node.body
            if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str))]
    if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Raise)):
        raise ReferenceUnavailable("native interface is a stub or unimplemented")
    # Such methods are platform wrappers, not native math implementations.
    # This is a conservative entry check, not a whole-program purity analysis.
    for item in ast.walk(node):
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute):
            if item.func.attr in {"forward_cuda", "forward_hip", "forward_xpu",
                                  "forward_oot", "_forward_method", "dispatch_forward"}:
                raise ReferenceUnavailable(
                    f"native interface delegates to platform entry {item.func.attr}"
                )


@dataclass(frozen=True)
class NativeMethod:
    function: object
    binding: str
    implementation: str

    def bind(self, obj):
        if self.binding == "static":
            return self.function
        if self.binding == "class":
            return self.function.__get__(type(obj), type(type(obj)))
        return self.function.__get__(obj, type(obj))


@lru_cache(maxsize=512)
def _discover_method(cls, descriptor, public_forward, code):
    from vllm.model_executor.custom_op import CustomOp

    module = _upstream_origin(cls)
    current = module
    for part in cls.__qualname__.split("."):
        current = getattr(current, part, None)
    if current is not cls:
        raise ReferenceUnavailable("CustomOp class is not the installed vLLM definition")
    if public_forward is not CustomOp.forward:
        raise ReferenceUnavailable(
            "CustomOp overrides forward; preserving its entry contract needs an adapter"
        )
    if isinstance(descriptor, staticmethod):
        fn, binding = descriptor.__func__, "static"
    elif isinstance(descriptor, classmethod):
        fn, binding = descriptor.__func__, "class"
    elif inspect.isfunction(descriptor):
        fn, binding = descriptor, "instance"
    else:
        raise ReferenceUnavailable("unsupported native method descriptor")
    if fn is CustomOp.forward_native:
        raise ReferenceUnavailable("CustomOp has no implemented forward_native")
    _upstream_origin(fn)
    _check_body(fn)
    return NativeMethod(fn, binding, fn.__module__ + "." + fn.__qualname__)


def discover_method(obj):
    cls = type(obj)
    # An instance replacement may capture a vendor kernel or use a different
    # calling convention. Do not mistake it for the class's native interface.
    if "forward_native" in vars(obj):
        raise ReferenceUnavailable("instance forward_native replacement needs an adapter")
    descriptor = inspect.getattr_static(cls, "forward_native", None)
    fn = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
    return _discover_method(cls, descriptor, inspect.getattr_static(cls, "forward", None),
                            getattr(fn, "__code__", None))


def custom_candidate(obj):
    from .adapters import has_custom_adapter
    if has_custom_adapter(obj):
        # Do not bypass a known adapter's shape/layout/semantic restrictions
        # after it has declared this particular call unsupported.
        return None
    entry = discover_method(obj)
    bound = entry.bind(obj)
    return Candidate("vllm.native.dynamic", entry.implementation, bound,
                     lambda *a, **k: tensor_support(obj, *a, **k))


@lru_cache(maxsize=128)
def _check_ir_function(fn, code):
    _upstream_origin(fn)
    _check_body(fn)


def ir_candidate(op):
    native = op.impls.get("native")
    if native is None:
        raise ReferenceUnavailable(f"IR {op.name} has no native provider")
    fn = native.impl_fn
    try:
        _check_ir_function(fn, getattr(fn, "__code__", None))
    except ReferenceUnavailable as exc:
        raise ReferenceUnavailable(
            f"IR {op.name} is outside the audited inventory; {exc}"
        ) from exc
    return Candidate("vllm.native.dynamic", fn.__module__ + "." + fn.__qualname__,
                     fn, tensor_support)


def ir_torch_entry(func):
    """Unwrap only the exact torch overload of a registered IR object."""
    from vllm.ir.op import IrOp
    name = func._schema.name.split("::", 1)[-1]
    op = IrOp.registry.get(name)
    if op is None:
        return None
    if func is op.torch_op:
        return op._inner_call
    inplace = getattr(op, "maybe_inplace", None)
    if inplace is not None and func is inplace.torch_op:
        return inplace._inner_call
    return None
