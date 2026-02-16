"""Low-pass Butterworth filter (zero-phase)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(
    signal: np.ndarray,
    cutoff_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        Input array. Shape (T,) or (T, D). Each column is filtered independently.
    cutoff_hz : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (applied twice via filtfilt, so effective order = 2*order).

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as *signal*.
    """
    nyq = 0.5 * fs
    normalized_cutoff = cutoff_hz / nyq
    b, a = butter(order, normalized_cutoff, btype="low")
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    out = np.empty_like(signal)
    for col in range(signal.shape[1]):
        out[:, col] = filtfilt(b, a, signal[:, col])
    return out
