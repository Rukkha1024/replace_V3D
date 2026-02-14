from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .joint_centers import _idx, compute_joint_centers


@dataclass(frozen=True)
class Anthropometrics:
    """Subject anthropometrics used for xCOM etc."""

    leg_length_m: float


@dataclass(frozen=True)
class COMModelParams:
    """Parameters chosen to mimic Visual3D's link-model COM reasonably well."""

    # --- Trunk / head segment geometry ---
    #
    # Visual3D's thorax (and thus the trunk COM) is typically defined using multiple
    # thorax markers (e.g., C7/STRN/CLAV/T10). In this repo we use a lightweight proxy
    # point on the thorax to avoid a full thorax coordinate system:
    #
    #   thorax_ref = thorax_w*C7 + (1-thorax_w)*STRN
    #
    # and place trunk COM along the line:
    #
    #   trunk_com = pelvis_origin + trunk_alpha*(thorax_ref - pelvis_origin)
    #
    # The defaults below were chosen to match the provided Visual3D export
    # (LINK_MODEL_BASED::ORIGINAL::COM) closely for the OptiTrack Conventional 39-marker set.
    trunk_alpha: float = 0.797  # pelvis_origin -> thorax_ref (0..1)

    # Head COM is placed along C7 -> head_center.
    # Setting this to 1.0 uses head_center directly.
    head_beta: float = 1.00

    # Weight for C7 vs STRN when building `thorax_ref` (0..1).
    thorax_w: float = 0.56

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


def compute_whole_body_com(
    points: np.ndarray,
    labels: list[str],
    params: COMModelParams = COMModelParams(),
) -> np.ndarray:
    """Compute whole-body COM time-series (meters) from marker trajectories.

    Notes
    -----
    - Uses De Leva mass fractions (male) and simple segment definitions.
    - Trunk COM is approximated on the line pelvis_origin -> thorax_ref using `params.trunk_alpha`.
    - Head COM is approximated on the line C7 -> head_center using `params.head_beta`.
    """
    jc: Dict[str, np.ndarray] = compute_joint_centers(points, labels)

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

    # Thorax reference point (proxy for a full thorax segment definition).
    # Prefer STRN if present; fall back to C7 if not.
    try:
        STRN = m("STRN")
        thorax_ref = params.thorax_w * C7 + (1.0 - params.thorax_w) * STRN
    except KeyError:
        thorax_ref = C7

    # segment COMs
    trunk_com = pelvis_origin + params.trunk_alpha * (thorax_ref - pelvis_origin)
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
