"""Compatibility wrapper for sagittal-plane lower-limb angles.

Implementation lives in `replace_v3d.joint_angles.sagittal`.
"""

from __future__ import annotations

from .joint_angles.sagittal import LowerLimbAngles, compute_lower_limb_angles

__all__ = [
    "LowerLimbAngles",
    "compute_lower_limb_angles",
]

