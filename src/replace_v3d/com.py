from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Anthropometrics:
    """Subject anthropometrics used for xCOM etc."""

    leg_length_m: float


@dataclass(frozen=True)
class COMModelParams:
    """Parameters chosen to mimic Visual3D's link-model COM reasonably well."""

    trunk_alpha: float = 0.45  # pelvis_origin -> C7
    head_beta: float = 0.80  # C7 -> head_center

    # De Leva (1996) mass fractions (male) – sum to 1 with bilateral segments
    mass_head: float = 0.0694
    mass_trunk: float = 0.4346
    mass_upperarm: float = 0.0271
    mass_forearm: float = 0.0162
    mass_hand: float = 0.0061
    mass_thigh: float = 0.1416
    mass_shank: float = 0.0433
    mass_foot: float = 0.0137

    # COM location fraction along segment from proximal
    frac_upperarm: float = 0.436
    frac_forearm: float = 0.430
    frac_hand: float = 0.506
    frac_thigh: float = 0.433
    frac_shank: float = 0.433
    frac_foot: float = 0.500


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


def compute_whole_body_com(
    points: np.ndarray,
    labels: list[str],
    params: COMModelParams = COMModelParams(),
) -> np.ndarray:
    """Compute whole-body COM time-series (meters) from marker trajectories.

    Notes
    -----
    - Uses De Leva mass fractions (male) and simple segment definitions.
    - Trunk COM is approximated on the line pelvis_origin -> C7 using `params.trunk_alpha`.
    - Head COM is approximated on the line C7 -> head_center using `params.head_beta`.
    """
    jc = compute_joint_centers(points, labels)

    def m(name: str) -> np.ndarray:
        return points[:, _idx(labels, name), :]

    LSHO, RSHO = m("LSHO"), m("RSHO")
    LFIN, RFIN = m("LFIN"), m("RFIN")
    LHEE, RHEE = m("LHEE"), m("RHEE")
    LTOE, RTOE = m("LTOE"), m("RTOE")

    hip_L, hip_R = jc["hip_L"], jc["hip_R"]
    knee_L, knee_R = jc["knee_L"], jc["knee_R"]
    ankle_L, ankle_R = jc["ankle_L"], jc["ankle_R"]
    elbow_L, elbow_R = jc["elbow_L"], jc["elbow_R"]
    wrist_L, wrist_R = jc["wrist_L"], jc["wrist_R"]

    pelvis_origin = jc["pelvis_origin"]
    C7 = jc["C7"]
    head_center = jc["head_center"]

    # segment COMs
    trunk_com = pelvis_origin + params.trunk_alpha * (C7 - pelvis_origin)
    head_com = C7 + params.head_beta * (head_center - C7)

    # upper limb COMs
    L_upper = LSHO + params.frac_upperarm * (elbow_L - LSHO)
    R_upper = RSHO + params.frac_upperarm * (elbow_R - RSHO)

    L_fore = elbow_L + params.frac_forearm * (wrist_L - elbow_L)
    R_fore = elbow_R + params.frac_forearm * (wrist_R - elbow_R)

    L_hand = wrist_L + params.frac_hand * (LFIN - wrist_L)
    R_hand = wrist_R + params.frac_hand * (RFIN - wrist_R)

    # lower limb COMs
    L_thigh = hip_L + params.frac_thigh * (knee_L - hip_L)
    R_thigh = hip_R + params.frac_thigh * (knee_R - hip_R)

    L_shank = knee_L + params.frac_shank * (ankle_L - knee_L)
    R_shank = knee_R + params.frac_shank * (ankle_R - knee_R)

    L_foot = LHEE + params.frac_foot * (LTOE - LHEE)
    R_foot = RHEE + params.frac_foot * (RTOE - RHEE)

    COM = (
        params.mass_head * head_com
        + params.mass_trunk * trunk_com
        + params.mass_upperarm * (L_upper + R_upper)
        + params.mass_forearm * (L_fore + R_fore)
        + params.mass_hand * (L_hand + R_hand)
        + params.mass_thigh * (L_thigh + R_thigh)
        + params.mass_shank * (L_shank + R_shank)
        + params.mass_foot * (L_foot + R_foot)
    )
    return COM


def derivative(signal: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference derivative (numpy.gradient)."""
    return np.gradient(signal, dt, axis=0)


def compute_xcom(COM: np.ndarray, vCOM: np.ndarray, leg_length_m: float, g: float = 9.81) -> np.ndarray:
    """Hof (XCoM) extrapolated COM."""
    w0 = np.sqrt(g / float(leg_length_m))
    return COM + vCOM / w0

