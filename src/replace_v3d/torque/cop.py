from __future__ import annotations

import numpy as np


def safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den2 = np.where(np.abs(den) < eps, np.nan, den)
    return num / den2


def compute_cop_stage01_xy(
    *,
    F_stage01: np.ndarray,
    M_stage01: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute Stage01 COP columns (Cx/Cy) from transformed force/moment.

    After Stage01 axis transform:
      Cx = -My / Fz
      Cy =  Mx / Fz
    """

    F_arr = np.asarray(F_stage01, dtype=float)
    M_arr = np.asarray(M_stage01, dtype=float)
    if F_arr.ndim != 2 or F_arr.shape[1] != 3:
        raise ValueError(f"F_stage01 must be shape (T,3), got {F_arr.shape}")
    if M_arr.ndim != 2 or M_arr.shape[1] != 3:
        raise ValueError(f"M_stage01 must be shape (T,3), got {M_arr.shape}")
    if F_arr.shape[0] != M_arr.shape[0]:
        raise ValueError(f"F_stage01/M_stage01 length mismatch: {F_arr.shape[0]} vs {M_arr.shape[0]}")

    Fz = F_arr[:, 2]
    cop_x = safe_div(-M_arr[:, 1], Fz, eps=eps)
    cop_y = safe_div(M_arr[:, 0], Fz, eps=eps)
    return np.column_stack([cop_x, cop_y])
