# Copyright (c) 2026 BAAI. All rights reserved.
"""Reference integration with the existing FL OpManager."""
from . import adapters
from .engine import (
    ReferenceUnavailable, optimized_fallback, record_route, run_reference,
    user_override_reason,
)


def call_optimized(manager, op_name, args, kwargs):
    from vllm_fl.dispatch.policy import get_policy
    from vllm_fl.dispatch.types import BackendImplKind, match_token

    manager.ensure_initialized()
    policy = get_policy()
    order = policy.get_per_op_order(op_name)
    if order is None:
        order = ["vendor"] if user_override_reason() else policy.get_default_order()
    candidates = [
        c for c in manager._compute_candidates(op_name, policy)
        if c.kind != BackendImplKind.REFERENCE
        and (not user_override_reason() or c.kind == BackendImplKind.VENDOR)
    ]
    for token in order:
        matches = sorted(
            (c for c in candidates if match_token(c, token)),
            key=lambda c: (c.priority, c.impl_id), reverse=True,
        )
        if not matches:
            continue
        impl = matches[0]
        # Availability is resolved before execution. Runtime errors propagate.
        with optimized_fallback():
            result = manager._call_with_hooks(op_name, impl.fn, args, kwargs)
        record_route(op_name, "optimized_fallback", impl.impl_id,
                     "no audited reference supports this call")
        return result
    raise ReferenceUnavailable(
        f"No permitted non-reference fallback for {op_name}; order={order}"
    )


def call_dispatch(manager, op_name, args, kwargs):
    return run_reference(
        op_name, args, kwargs,
        (lambda: adapters.upstream_dispatch(op_name), lambda: adapters.plugin_dispatch(op_name)),
        lambda: call_optimized(manager, op_name, args, kwargs),
    )
