"""Compatibility wrapper for C3D marker trajectories.

Implementation lives in `replace_v3d.io.c3d_reader`.
"""

from __future__ import annotations

from .io.c3d_reader import C3DPoints, _parse_parameters, read_c3d_points

__all__ = [
    "C3DPoints",
    "_parse_parameters",
    "read_c3d_points",
]

