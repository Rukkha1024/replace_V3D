"""Anthropometrics helpers for inverse dynamics.

This module provides a small, consistent parameter source for segment mass
fractions and COM placement that matches the repo's existing COM model defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from replace_v3d.com.whole_body import COMModelParams


@dataclass(frozen=True)
class SegmentMassComInertiaParams:
    """Per-segment mass, COM, and inertia profile used in inverse dynamics."""

    mass_fraction: float
    com_fraction_from_prox: float
    rog_fraction_xyz: tuple[float, float, float]
    display_name: str | None = None


@dataclass(frozen=True)
class LowerBodyParams:
    thigh: SegmentMassComInertiaParams
    shank: SegmentMassComInertiaParams
    foot: SegmentMassComInertiaParams


@dataclass(frozen=True)
class UpperBodyParams:
    trunk: SegmentMassComInertiaParams
    head: SegmentMassComInertiaParams


@dataclass(frozen=True)
class BodySegmentParams:
    lower: LowerBodyParams
    upper: UpperBodyParams


def get_body_segment_params(*, params: COMModelParams | None = None) -> BodySegmentParams:
    p = COMModelParams() if params is None else params
    return BodySegmentParams(
        lower=LowerBodyParams(
            thigh=SegmentMassComInertiaParams(
                mass_fraction=float(p.mass_thigh),
                com_fraction_from_prox=float(p.frac_thigh),
                rog_fraction_xyz=(0.329, 0.329, 0.149),
                display_name="thigh",
            ),
            shank=SegmentMassComInertiaParams(
                mass_fraction=float(p.mass_shank),
                com_fraction_from_prox=float(p.frac_shank),
                rog_fraction_xyz=(0.251, 0.246, 0.102),
                display_name="shank",
            ),
            foot=SegmentMassComInertiaParams(
                mass_fraction=float(p.mass_foot),
                com_fraction_from_prox=float(p.frac_foot),
                rog_fraction_xyz=(0.257, 0.245, 0.124),
                display_name="foot",
            ),
        ),
        upper=UpperBodyParams(
            trunk=SegmentMassComInertiaParams(
                mass_fraction=float(p.mass_trunk),
                com_fraction_from_prox=float(p.trunk_alpha),
                rog_fraction_xyz=(0.328, 0.306, 0.169),
                display_name="trunk",
            ),
            head=SegmentMassComInertiaParams(
                mass_fraction=float(p.mass_head),
                com_fraction_from_prox=float(p.head_beta),
                rog_fraction_xyz=(0.303, 0.315, 0.261),
                display_name="head",
            ),
        ),
    )
