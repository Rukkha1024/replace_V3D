from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AnkleTorqueResult:
    """Computed ankle torques from a single (net) force plate."""

    F_lab: np.ndarray  # (T,3) N
    M_lab_at_fp_origin: np.ndarray  # (T,3) Nm
    COP_lab: np.ndarray  # (T,3) m
    fp_origin_lab: np.ndarray  # (3,) m

    ankle_L: np.ndarray  # (T,3) m
    ankle_R: np.ndarray  # (T,3) m
    ankle_mid: np.ndarray  # (T,3) m

    torque_mid_ext: np.ndarray  # (T,3) Nm
    torque_mid_int: np.ndarray  # (T,3) Nm
    torque_L_ext: np.ndarray  # (T,3) Nm
    torque_L_int: np.ndarray  # (T,3) Nm
    torque_R_ext: np.ndarray  # (T,3) Nm
    torque_R_int: np.ndarray  # (T,3) Nm

    torque_mid_int_Y_Nm_per_kg: Optional[np.ndarray]  # (T,) Nm/kg (if body mass provided)


def compute_ankle_torque_from_net_wrench(
    *,
    F_lab: np.ndarray,
    M_lab_at_fp_origin: np.ndarray,
    COP_lab: Optional[np.ndarray] = None,
    fp_origin_lab: np.ndarray,
    ankle_L: np.ndarray,
    ankle_R: np.ndarray,
    body_mass_kg: Optional[float] = None,
) -> AnkleTorqueResult:
    """Compute external/internal ankle torques using a single (net) GRF + GRM.

    This follows the standard rigid-body moment transfer:

        M_A = M_O + (r_O - r_A) x F

    where:
      - O is the force plate origin
      - A is the ankle joint center (left/right or mid)
      - F is the ground reaction force (lab)
      - M_O is the ground reaction moment about O (lab)

    Notes
    -----
    - Result is an *external* moment applied by the ground about the ankle.
      Internal joint moment (muscle/ligament) is often reported as the negative.
    """

    F_lab = np.asarray(F_lab, dtype=float)
    M_lab_at_fp_origin = np.asarray(M_lab_at_fp_origin, dtype=float)
    fp_origin_lab = np.asarray(fp_origin_lab, dtype=float).reshape(3)
    ankle_L = np.asarray(ankle_L, dtype=float)
    ankle_R = np.asarray(ankle_R, dtype=float)
    if F_lab.shape != M_lab_at_fp_origin.shape:
        raise ValueError("F_lab and M_lab_at_fp_origin must have the same shape (T,3)")
    if ankle_L.shape != ankle_R.shape:
        raise ValueError("ankle_L and ankle_R must have the same shape (T,3)")

    ankle_mid = 0.5 * (ankle_L + ankle_R)

    rO_minus_mid = fp_origin_lab[None, :] - ankle_mid
    rO_minus_L = fp_origin_lab[None, :] - ankle_L
    rO_minus_R = fp_origin_lab[None, :] - ankle_R

    torque_mid_ext = M_lab_at_fp_origin + np.cross(rO_minus_mid, F_lab)
    torque_L_ext = M_lab_at_fp_origin + np.cross(rO_minus_L, F_lab)
    torque_R_ext = M_lab_at_fp_origin + np.cross(rO_minus_R, F_lab)

    torque_mid_int = -torque_mid_ext
    torque_L_int = -torque_L_ext
    torque_R_int = -torque_R_ext

    if COP_lab is None:
        COP_lab = np.full_like(F_lab, np.nan, dtype=float)
    else:
        COP_lab = np.asarray(COP_lab, dtype=float)
        if COP_lab.shape != F_lab.shape:
            raise ValueError("COP_lab must have the same shape as F_lab (T,3)")

    torque_mid_int_Y_Nm_per_kg = None
    if body_mass_kg is not None and float(body_mass_kg) > 0:
        torque_mid_int_Y_Nm_per_kg = torque_mid_int[:, 1] / float(body_mass_kg)

    return AnkleTorqueResult(
        F_lab=F_lab,
        M_lab_at_fp_origin=M_lab_at_fp_origin,
        COP_lab=COP_lab,
        fp_origin_lab=fp_origin_lab,
        ankle_L=ankle_L,
        ankle_R=ankle_R,
        ankle_mid=ankle_mid,
        torque_mid_ext=torque_mid_ext,
        torque_mid_int=torque_mid_int,
        torque_L_ext=torque_L_ext,
        torque_L_int=torque_L_int,
        torque_R_ext=torque_R_ext,
        torque_R_int=torque_R_int,
        torque_mid_int_Y_Nm_per_kg=torque_mid_int_Y_Nm_per_kg,
    )
