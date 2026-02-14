from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    dot = np.sum(u * v, axis=-1)
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    cos = dot / (nu * nv + 1e-12)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _proj_sag(v3: np.ndarray) -> np.ndarray:
    """Sagittal projection for this dataset: keep X (A/P) and Z (Up)."""
    return v3[..., [0, 2]]


@dataclass(frozen=True)
class LowerLimbAngles:
    knee_flex_L_deg: np.ndarray
    knee_flex_R_deg: np.ndarray
    ankle_dorsi_L_deg: np.ndarray
    ankle_dorsi_R_deg: np.ndarray


def compute_lower_limb_angles(
    points: np.ndarray,
    labels: List[str],
    joint_centers: Dict[str, np.ndarray],
    end_frame: Optional[int] = None,
) -> LowerLimbAngles:
    """Compute simple sagittal-plane knee flexion and ankle dorsi/plantar angles.

    - Knee flexion = 180 - angle( (hip-knee), (ankle-knee) )
    - Ankle dorsi  = 90 - angle( (knee-ankle), (toe-ankle) )
    """
    T = points.shape[0]
    end = T if end_frame is None else max(1, min(int(end_frame), T))

    def midx(name: str) -> int:
        try:
            return labels.index(name)
        except ValueError as e:
            raise KeyError(f"Required marker not found: {name}") from e

    LTOE = points[:end, midx("LTOE"), :]
    RTOE = points[:end, midx("RTOE"), :]

    hip_L = joint_centers["hip_L"][:end]
    hip_R = joint_centers["hip_R"][:end]
    knee_L = joint_centers["knee_L"][:end]
    knee_R = joint_centers["knee_R"][:end]
    ankle_L = joint_centers["ankle_L"][:end]
    ankle_R = joint_centers["ankle_R"][:end]

    # Knee included angle at knee (sagittal)
    v1_L = hip_L - knee_L
    v2_L = ankle_L - knee_L
    v1_R = hip_R - knee_R
    v2_R = ankle_R - knee_R

    knee_included_L = _angle_between(_proj_sag(v1_L), _proj_sag(v2_L))
    knee_included_R = _angle_between(_proj_sag(v1_R), _proj_sag(v2_R))
    knee_flex_L = 180.0 - knee_included_L
    knee_flex_R = 180.0 - knee_included_R

    # Ankle included angle at ankle (sagittal)
    v1a_L = knee_L - ankle_L
    v2a_L = LTOE - ankle_L
    v1a_R = knee_R - ankle_R
    v2a_R = RTOE - ankle_R

    ankle_included_L = _angle_between(_proj_sag(v1a_L), _proj_sag(v2a_L))
    ankle_included_R = _angle_between(_proj_sag(v1a_R), _proj_sag(v2a_R))
    ankle_dorsi_L = 90.0 - ankle_included_L
    ankle_dorsi_R = 90.0 - ankle_included_R

    return LowerLimbAngles(
        knee_flex_L_deg=knee_flex_L,
        knee_flex_R_deg=knee_flex_R,
        ankle_dorsi_L_deg=ankle_dorsi_L,
        ankle_dorsi_R_deg=ankle_dorsi_R,
    )
