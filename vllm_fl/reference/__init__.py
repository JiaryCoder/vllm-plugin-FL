# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in, audited PyTorch reference routing for precision debugging."""

from .engine import (
    Candidate,
    ReferenceUnavailable,
    ReferencePurityError,
    clear_records,
    get_records,
    reference_enabled,
    reference_requested,
    run_reference,
)

__all__ = [
    "Candidate", "ReferenceUnavailable", "ReferencePurityError",
    "clear_records", "get_records", "reference_enabled",
    "reference_requested", "run_reference",
]
