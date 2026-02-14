"""Compatibility wrapper for Excel event sheets.

Implementation lives in `replace_v3d.io.events_excel`.
"""

from __future__ import annotations

from .io.events_excel import (
    TrialEvents,
    load_subject_body_mass_kg,
    load_trial_events,
    parse_trial_from_filename,
)

__all__ = [
    "TrialEvents",
    "load_trial_events",
    "parse_trial_from_filename",
    "load_subject_body_mass_kg",
]

