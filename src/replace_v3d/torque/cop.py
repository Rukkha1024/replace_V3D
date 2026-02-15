from __future__ import annotations

import numpy as np


def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den2 = np.where(np.abs(den) < eps, np.nan, den)
    return num / den2


def compute_cop_lab(
    *,
    F_plate: np.ndarray,
    M_plate: np.ndarray,
    fp_origin_lab: np.ndarray,
    R_pl2lab: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """COP in plate coordinates then rotate/translate into lab."""

    Fz = F_plate[:, 2]
    cop_x = safe_div(-M_plate[:, 1], Fz, eps=eps)
    cop_y = safe_div(M_plate[:, 0], Fz, eps=eps)
    cop_plate = np.column_stack([cop_x, cop_y, np.zeros_like(cop_x)])
    return fp_origin_lab[None, :] + cop_plate @ R_pl2lab.T

