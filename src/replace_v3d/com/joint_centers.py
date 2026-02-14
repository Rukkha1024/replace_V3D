from __future__ import annotations

from typing import Dict

import numpy as np


def _idx(labels: list[str], name: str) -> int:
    try:
        return labels.index(name)
    except ValueError as e:
        raise KeyError(f"Required marker not found: {name}") from e


def compute_joint_centers(points: np.ndarray, labels: list[str]) -> Dict[str, np.ndarray]:
    """Compute key joint centers from the OptiTrack Conventional (39) markers.

    Returns dict of arrays shape (n_frames,3) with keys:
    - hip_L, hip_R
    - knee_L, knee_R
    - ankle_L, ankle_R
    - elbow_L, elbow_R
    - wrist_L, wrist_R
    - pelvis_origin, C7, head_center
    """
    pts = points  # (T, N, 3)

    def m(name: str) -> np.ndarray:
        return pts[:, _idx(labels, name), :]

    LASI, RASI, LPSI, RPSI = m("LASI"), m("RASI"), m("LPSI"), m("RPSI")
    C7 = m("C7")
    LFHD, RFHD, LBHD, RBHD = m("LFHD"), m("RFHD"), m("LBHD"), m("RBHD")

    # Medial/lateral marker pairs (from optitrack_marker_context.xml):
    # - Knee: LKNE (lateral) + LShin_3 (medial)
    # - Ankle: LANK (lateral) + LFoot_3 (medial)
    # - Elbow: LELB (lateral) + LUArm_3 (medial)
    LKNE, RKNE = m("LKNE"), m("RKNE")
    LShin, RShin = m("LShin_3"), m("RShin_3")
    LANK, RANK = m("LANK"), m("RANK")
    LFoot, RFoot = m("LFoot_3"), m("RFoot_3")
    LELB, RELB = m("LELB"), m("RELB")
    LUArm, RUArm = m("LUArm_3"), m("RUArm_3")

    LWRA, LWRB = m("LWRA"), m("LWRB")
    RWRA, RWRB = m("RWRA"), m("RWRB")

    knee_L = (LKNE + LShin) / 2.0
    knee_R = (RKNE + RShin) / 2.0
    ankle_L = (LANK + LFoot) / 2.0
    ankle_R = (RANK + RFoot) / 2.0
    elbow_L = (LELB + LUArm) / 2.0
    elbow_R = (RELB + RUArm) / 2.0
    wrist_L = (LWRA + LWRB) / 2.0
    wrist_R = (RWRA + RWRB) / 2.0

    pelvis_origin = (LASI + RASI) / 2.0
    pelvis_midPSIS = (LPSI + RPSI) / 2.0

    # Pelvis axes: Z=right, X=anterior, Y=up
    z_axis = RASI - LASI
    z_axis = z_axis / np.linalg.norm(z_axis, axis=1, keepdims=True)

    x_axis = pelvis_origin - pelvis_midPSIS
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=1, keepdims=True)

    # re-orthogonalize X
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)

    # Harrington (2007) hip joint center regression (mm -> m)
    PW = np.linalg.norm(RASI - LASI, axis=1) * 1000.0
    PD = np.linalg.norm(pelvis_origin - pelvis_midPSIS, axis=1) * 1000.0

    x_off = (-0.24 * PW - 9.9) / 1000.0
    y_off = (-0.30 * PD - 10.9) / 1000.0
    z_off = (0.33 * PW + 7.3) / 1000.0

    hip_R = pelvis_origin + x_off[:, None] * x_axis + y_off[:, None] * y_axis + z_off[:, None] * z_axis
    hip_L = pelvis_origin + x_off[:, None] * x_axis + y_off[:, None] * y_axis - z_off[:, None] * z_axis

    head_center = (LFHD + RFHD + LBHD + RBHD) / 4.0

    return {
        "hip_L": hip_L,
        "hip_R": hip_R,
        "knee_L": knee_L,
        "knee_R": knee_R,
        "ankle_L": ankle_L,
        "ankle_R": ankle_R,
        "elbow_L": elbow_L,
        "elbow_R": elbow_R,
        "wrist_L": wrist_L,
        "wrist_R": wrist_R,
        "pelvis_origin": pelvis_origin,
        "C7": C7,
        "head_center": head_center,
    }

