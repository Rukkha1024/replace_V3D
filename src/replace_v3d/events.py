"""Compatibility wrapper for Excel event sheets.

Implementation lives in `replace_v3d.io.events_excel`.
"""

from __future__ import annotations

from .io.events_excel import (
    TrialEvents,
    load_subject_body_mass_kg,
    load_subject_leg_length_cm,
    load_trial_events,
    parse_subject_velocity_trial_from_filename,
    parse_trial_from_filename,
    resolve_subject_from_token,
)

__all__ = [
    "TrialEvents",
    "load_trial_events",
    "parse_trial_from_filename",
    "parse_subject_velocity_trial_from_filename",
    "resolve_subject_from_token",
    "load_subject_leg_length_cm",
    "load_subject_body_mass_kg",
]
