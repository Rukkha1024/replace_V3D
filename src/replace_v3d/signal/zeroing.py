"""Baseline subtraction helpers for time-series exports.

This module provides a small, reusable helper for "onset-zeroing":
subtracting a baseline value at a designated frame index (typically platform onset)
so that the value at onset becomes 0 and subsequent values become Δ relative to onset.
"""

from __future__ import annotations

import numpy as np


def subtract_baseline_at_index(x: np.ndarray, idx0: int) -> np.ndarray:
    """Return `x - baseline`, where baseline is sampled near `idx0`.

    Parameters
    ----------
    x:
        1D array of values.
    idx0:
        Baseline index (0-based). Usually the platform onset frame in the file.

    Notes
    -----
    - If `x[idx0]` is not finite, the function searches outward from `idx0`
      (idx0-1, idx0+1, idx0-2, idx0+2, ...) for the nearest finite value
      and uses it as the baseline.
    - If no finite baseline value is found, returns a float copy of `x` unchanged.
    """

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError(f"subtract_baseline_at_index expects a 1D array. Got shape={x_arr.shape!r}")

    n = int(x_arr.shape[0])
    idx = int(idx0)
    if idx < 0 or idx >= n:
        raise ValueError(f"Baseline index out of range: idx0={idx} for array length={n}.")

    baseline = float(x_arr[idx]) if np.isfinite(x_arr[idx]) else float("nan")
    if not np.isfinite(baseline):
        left = idx - 1
        right = idx + 1
        while left >= 0 or right < n:
            if left >= 0 and np.isfinite(x_arr[left]):
                baseline = float(x_arr[left])
                break
            if right < n and np.isfinite(x_arr[right]):
                baseline = float(x_arr[right])
                break
            left -= 1
            right += 1

    if not np.isfinite(baseline):
        return x_arr.copy()

    return x_arr - baseline

