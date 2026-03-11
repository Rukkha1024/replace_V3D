"""Anthropometrics helpers for inverse dynamics.

This module provides a small, consistent parameter source for segment mass
fractions and COM placement that matches the repo's existing COM model defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from replace_v3d.com.whole_body import COMModelParams


@dataclass(frozen=True)
class SegmentMassComParams:
    mass_fraction: float
    com_fraction_from_prox: float


@dataclass(frozen=True)
class LowerBodyParams:
    thigh: SegmentMassComParams
    shank: SegmentMassComParams
    foot: SegmentMassComParams


@dataclass(frozen=True)
class UpperBodyParams:
    trunk: SegmentMassComParams
    head: SegmentMassComParams


@dataclass(frozen=True)
class BodySegmentParams:
    lower: LowerBodyParams
    upper: UpperBodyParams


def get_body_segment_params(*, params: COMModelParams | None = None) -> BodySegmentParams:
    p = COMModelParams() if params is None else params
    return BodySegmentParams(
        lower=LowerBodyParams(
            thigh=SegmentMassComParams(mass_fraction=float(p.mass_thigh), com_fraction_from_prox=float(p.frac_thigh)),
            shank=SegmentMassComParams(mass_fraction=float(p.mass_shank), com_fraction_from_prox=float(p.frac_shank)),
            foot=SegmentMassComParams(mass_fraction=float(p.mass_foot), com_fraction_from_prox=float(p.frac_foot)),
        ),
        upper=UpperBodyParams(
            trunk=SegmentMassComParams(mass_fraction=float(p.mass_trunk), com_fraction_from_prox=float(p.trunk_alpha)),
            head=SegmentMassComParams(mass_fraction=float(p.mass_head), com_fraction_from_prox=float(p.head_beta)),
        ),
    )

