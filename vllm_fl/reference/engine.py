# Copyright (c) 2026 BAAI. All rights reserved.
"""Common selection and evidence for CustomOp, IR and FL dispatch.

Only candidate discovery/capability failures may advance the reference chain.
Errors after execution begins propagate: retrying an in-place op is unsafe.
"""
from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("vllm_fl.reference")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(getattr(logging, os.environ.get("VLLM_FL_LOG_LEVEL", "INFO").upper(),
                            logging.INFO))
    logger.propagate = False
_BYPASS = contextvars.ContextVar("fl_reference_optimized_fallback", default=False)
_USER_OVERRIDE = contextvars.ContextVar("fl_reference_user_override", default=None)
_REFERENCE_SCOPES = contextvars.ContextVar("fl_reference_scopes", default=())
_records: dict[tuple, dict[str, Any]] = {}
_record_pid = os.getpid()
_record_lock = threading.Lock()


class ReferenceUnavailable(RuntimeError):
    """No audited reference supports this operation/input."""


class ReferencePurityError(RuntimeError):
    """An audited reference attempted an opaque non-ATen operation."""


@dataclasses.dataclass(frozen=True)
class Candidate:
    source: str
    implementation: str
    fn: Callable
    supports: Callable | None = None


def reference_requested() -> bool:
    """Process-start setting; deliberately separate from backend preference."""
    value = os.environ.get("VLLM_FL_REFERENCE_MODE", "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError("VLLM_FL_REFERENCE_MODE must be 0/1 or false/true")
    return value in {"1", "true"}


def reference_enabled() -> bool:
    return reference_requested() and not _BYPASS.get()


def reference_execution_active() -> bool:
    """True only inside reference computation in this execution context."""
    return bool(_REFERENCE_SCOPES.get()) and not _BYPASS.get()


@contextlib.contextmanager
def optimized_fallback():
    token = _BYPASS.set(True)
    try:
        yield
    finally:
        _BYPASS.reset(token)


def user_override_reason():
    scope = _USER_OVERRIDE.get()
    return None if scope is None else scope["reason"]


def run_user_override(op, fn, implementation, reason):
    """An intentional optimized call is allowed even in strict reference mode."""
    ensure_vendor_aten()
    for parent in _REFERENCE_SCOPES.get():
        parent["mixed"] = True
    scope = {"reason": reason, "recorded": False}
    token = _USER_OVERRIDE.set(scope)
    try:
        with optimized_fallback():
            result = fn()
        if not scope["recorded"]:
            record_route(op, "user_override", implementation, reason)
        return result
    except Exception as exc:
        record_route(op, "user_override_error", implementation,
                     f"{reason}; {type(exc).__name__}")
        raise
    finally:
        _USER_OVERRIDE.reset(token)


def _reset_after_fork() -> None:
    global _records, _record_pid, _record_lock
    _records = {}
    _record_pid = os.getpid()
    _record_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def clear_records() -> None:
    with _record_lock:
        _records.clear()


def get_records() -> list[dict[str, Any]]:
    with _record_lock:
        return [dict(record) for record in _records.values()]


def record_route(op: str, source: str, implementation: str, reason: str = "") -> None:
    override = _USER_OVERRIDE.get()
    if override is not None and source in {"optimized_fallback", "user_override"}:
        source, reason = "user_override", override["reason"]
        override["recorded"] = True
    key = (op, source, implementation, reason)
    with _record_lock:
        first = key not in _records
        row = _records.setdefault(key, {
            "pid": os.getpid(), "op": op, "source": source,
            "implementation": implementation, "reason": reason, "calls": 0,
        })
        row["calls"] += 1
        # Keep aggregation and last-use ordering: selected-implementation
        # queries must follow a return to an earlier provider correctly.
        _records.pop(key)
        _records[key] = row
        event = dict(row)
    if first:
        logger.info(
            "[REFERENCE] %s -> %s (%s)%s", op, implementation, source,
            f"; {reason}" if reason else "",
        )
    directory = os.environ.get("VLLM_FL_REFERENCE_REPORT_DIR", "").strip()
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        # Each worker owns a file. No tensors or tensor values are serialized.
        with (path / f"reference-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")


def ensure_vendor_aten():
    gems = sys.modules.get("flag_gems")
    if gems is not None and getattr(gems, "current_work_registrar", None) is not None:
        raise ReferencePurityError(
            "FlagGems ATen overrides are already active; start a fresh reference worker"
        )


@contextlib.contextmanager
def aten_only():
    ensure_vendor_aten()
    from torch.utils._python_dispatch import TorchDispatchMode

    class AtenOnly(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            namespace = func._schema.name.split("::", 1)[0]
            if namespace not in {"aten", "prims"} and not _BYPASS.get():
                from .native import ir_torch_entry
                entry = ir_torch_entry(func)
                if entry is not None:
                    return entry(*args, **(kwargs or {}))
                raise ReferencePurityError(
                    f"Reference attempted opaque operation {func._schema.name}"
                )
            if not _BYPASS.get() and func._schema.name.startswith((
                "aten::_scaled_dot_product_flash_attention",
                "aten::_scaled_dot_product_efficient_attention",
                "aten::_scaled_dot_product_cudnn_attention",
                "aten::_flash_attention_forward",
            )):
                raise ReferencePurityError(
                    f"Reference attempted fused attention operation {func._schema.name}"
                )
            return func(*args, **(kwargs or {}))

    from .guards import math_function_mode
    with math_function_mode(), AtenOnly():
        yield


def run_reference(
    op: str,
    args: tuple,
    kwargs: dict,
    factories: tuple[Callable[[], Candidate | None], ...],
    fallback: Callable[[], Any] | None = None,
    *,
    strict: bool | None = None,
):
    """Select vLLM then plugin reference; never retry a failed execution."""
    from .selection import selection_reason
    reason = selection_reason(op)
    if reason is not None:
        if fallback is None:
            raise ReferenceUnavailable(f"{op}: {reason}; no original entry is available")
        implementation = getattr(fallback, "__module__", "") + "." + getattr(
            fallback, "__qualname__", type(fallback).__name__)
        return run_user_override(op, fallback, implementation, reason)
    if strict is None:
        from vllm_fl.dispatch.policy import get_policy
        strict = get_policy().strict
    reasons = []
    for factory in factories:
        try:
            candidate = factory()
            if candidate is None:
                reasons.append("no audited candidate")
                continue
            if candidate.supports is not None:
                reason = candidate.supports(*args, **kwargs)
                if reason:
                    reasons.append(f"{candidate.implementation}: {reason}")
                    continue
        except ReferenceUnavailable as exc:
            reasons.append(str(exc))
            continue
        # Do not put execution inside a fallback exception handler.
        scope = {"mixed": False}
        scope_token = _REFERENCE_SCOPES.set(_REFERENCE_SCOPES.get() + (scope,))
        try:
            with aten_only():
                result = candidate.fn(*args, **kwargs)
        except Exception as exc:
            record_route(op, "error", candidate.implementation, type(exc).__name__)
            raise
        finally:
            _REFERENCE_SCOPES.reset(scope_token)
        record_route(op, "reference.mixed" if scope["mixed"] else candidate.source,
                     candidate.implementation,
                     "contains explicit optimized child calls" if scope["mixed"] else "")
        return result
    reason = "; ".join(reasons) or "no audited reference"
    if strict or fallback is None:
        record_route(op, "unavailable", "", reason)
        raise ReferenceUnavailable(f"No audited torch reference for {op}: {reason}")
    # The fallback callback records the actual selected kernel, not merely a
    # preference. Its nested optimized calls are not labelled reference.
    record_route(op, "reference_unavailable", "", reason)
    with optimized_fallback():
        return fallback()


def tensor_support(*args, **kwargs):
    import torch
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        else:
            weight = getattr(value, "weight", None)
            if isinstance(weight, torch.Tensor):
                yield weight
    for t in tensors((args, kwargs)):
        if t.is_floating_point() and t.dtype not in {
            torch.float16, torch.bfloat16, torch.float32
        }:
            return f"unaudited floating dtype {t.dtype}"
    return None
