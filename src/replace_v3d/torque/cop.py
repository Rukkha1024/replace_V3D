from __future__ import annotations

import numpy as np


def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den2 = np.where(np.abs(den) < eps, np.nan, den)
    return num / den2


def compute_cop_lab_xy(
    *,
    F_plate: np.ndarray,
    M_plate: np.ndarray,
    fp_origin_lab: np.ndarray,
    R_pl2lab: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute COP X/Y in lab coordinates.

    COP is defined on the force-plate plane, so only X/Y are returned.
    """

    F_plate = np.asarray(F_plate, dtype=float)
    M_plate = np.asarray(M_plate, dtype=float)
    fp_origin_lab = np.asarray(fp_origin_lab, dtype=float).reshape(3)
    R_pl2lab = np.asarray(R_pl2lab, dtype=float).reshape(3, 3)
    Fz = F_plate[:, 2]
    cop_x = safe_div(-M_plate[:, 1], Fz, eps=eps)
    cop_y = safe_div(M_plate[:, 0], Fz, eps=eps)

    # row-vector convention: v_lab = v_plate @ R^T
    # with v_plate = [cop_x, cop_y, 0], only lab X/Y are needed.
    lab_x = fp_origin_lab[0] + (cop_x * R_pl2lab[0, 0]) + (cop_y * R_pl2lab[0, 1])
    lab_y = fp_origin_lab[1] + (cop_x * R_pl2lab[1, 0]) + (cop_y * R_pl2lab[1, 1])
    return np.column_stack([lab_x, lab_y])
