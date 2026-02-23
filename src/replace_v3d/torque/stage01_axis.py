from __future__ import annotations

import numpy as np


def transform_force_moment_to_stage01(
    *,
    F_in: np.ndarray,
    M_in: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply shared_files Stage01 axis transform to force/moment channels.

    Stage01 mapping (shared_files/config.yaml):
      Fx <- +Fy
      Fy <- +Fx
      Fz <- -Fz
      Mx <- +My
      My <- +Mx
      Mz <- -Mz
    """

    F_arr = np.asarray(F_in, dtype=float)
    M_arr = np.asarray(M_in, dtype=float)
    if F_arr.ndim != 2 or F_arr.shape[1] != 3:
        raise ValueError(f"F_in must be shape (T,3), got {F_arr.shape}")
    if M_arr.ndim != 2 or M_arr.shape[1] != 3:
        raise ValueError(f"M_in must be shape (T,3), got {M_arr.shape}")
    if F_arr.shape[0] != M_arr.shape[0]:
        raise ValueError(f"F_in/M_in length mismatch: {F_arr.shape[0]} vs {M_arr.shape[0]}")

    Fx_raw = F_arr[:, 0]
    Fy_raw = F_arr[:, 1]
    Fz_raw = F_arr[:, 2]

    Mx_raw = M_arr[:, 0]
    My_raw = M_arr[:, 1]
    Mz_raw = M_arr[:, 2]

    F_out = np.column_stack(
        [
            Fy_raw,
            Fx_raw,
            -Fz_raw,
        ]
    )
    M_out = np.column_stack(
        [
            My_raw,
            Mx_raw,
            -Mz_raw,
        ]
    )
    return F_out, M_out

