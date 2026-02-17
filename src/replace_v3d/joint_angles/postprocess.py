"""Post-processing helpers for Visual3D-like joint angle outputs.

This module is intentionally post-process only: it does not change the underlying
segment definitions or joint-angle math. It exists to provide two analysis-friendly
conventions that are commonly needed when comparing left vs right:

1) Anatomical "presentation" convention:
   Visual3D-style right-hand-rule joint angles often produce opposite sign meanings
   for LEFT vs RIGHT in Y/Z (ab/adduction and internal/external rotation).
   A common practice is to negate LEFT Y/Z so that the sign meaning matches the
   RIGHT side.

2) Discontinuity resolution (Visual3D Resolve_Discontinuity-style; Ankle Z only):
   Euler/Cardan angles can show ±180° wrap discontinuities. Visual3D provides a
   convenience function that removes these by adding/subtracting a range (typically
   360°) at the point of discontinuity. We implement the same idea here, but keep
   the scope limited to the columns that matter for this project (Ankle Z).

3) Baseline-normalized convention (optional):
   Subtract a baseline window mean to remove static offsets (e.g., small SCS
   misalignments). This is useful when you want Δangles relative to a chosen
   baseline segment.

Current exports in this repo typically:

- apply LEFT Hip/Knee/Ankle Y/Z sign-unification
- apply Resolve_Discontinuity-style unwrapping for Ankle_*_Z_deg (range=360°; if needed)
- perform onset-zeroing at platform onset in the calling pipeline (so baseline subtraction here is disabled)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class JointAnglePostprocessMeta:
    """Metadata returned by :func:`postprocess_joint_angles`.

    Notes
    -----
    `baseline_values` stores the mean of the selected baseline window **after**
    applying sign-unification (if enabled). This is what is subtracted.
    """

    frame_col: str
    baseline_frame_start: int
    baseline_frame_end: int
    flipped_columns: tuple[str, ...]
    baseline_values: dict[str, float]
    resolved_columns: tuple[str, ...] = ()
    resolve_discontinuity_range_deg: float | None = None


def _default_flip_columns(columns: Sequence[str]) -> list[str]:
    """Default 'anatomical sign unification' columns.

    Convention used here:
    - keep X (flex/ext) as-is
    - flip LEFT Y/Z so that Y/Z have the same sign meaning as the RIGHT side

    Targets Hip/Knee/Ankle only; only flips columns that exist.
    """

    candidates: list[str] = []
    for joint in ("Hip", "Knee", "Ankle"):
        for axis in ("Y", "Z"):
            candidates.append(f"{joint}_L_{axis}_deg")
    return [c for c in candidates if c in columns]


def _default_discontinuity_columns(columns: Sequence[str]) -> list[str]:
    """Default columns for discontinuity resolution.

    Project policy: apply only to Ankle Z to fix baseline mean issues caused by Euler wrap
    discontinuities, without changing other angle channels.
    """
    candidates = ["Ankle_L_Z_deg", "Ankle_R_Z_deg"]
    return [c for c in candidates if c in columns]


def _needs_resolve_discontinuity(x_deg: np.ndarray, range_deg: float) -> bool:
    """Detect likely Euler-wrap jumps by checking consecutive finite diffs.

    If any finite consecutive jump exceeds range/2 (e.g., 180° for range=360°), we treat it
    as a discontinuity crossing and unwrap.
    """
    idx = np.flatnonzero(np.isfinite(x_deg))
    if idx.size < 2:
        return False
    diffs = np.diff(x_deg[idx])
    if diffs.size == 0:
        return False
    return bool(np.nanmax(np.abs(diffs)) > (range_deg / 2.0))


def _resolve_discontinuity_deg(x_deg: np.ndarray, *, range_deg: float = 360.0) -> tuple[np.ndarray, bool]:
    """Resolve Euler wrap discontinuities by adding/subtracting `range_deg`.

    Notes
    -----
    - Output may exceed [-180, 180]; this is expected for a "continuous" representation.
    - NaNs are preserved; processing starts from the first finite sample.
    """
    x = np.array(x_deg, dtype=float, copy=True)

    idx = np.flatnonzero(np.isfinite(x))
    if idx.size == 0:
        return x, False

    start = int(idx[0])
    offset = 0.0
    prev = float(x[start])
    changed = False

    for i in range(start + 1, len(x)):
        if not np.isfinite(x[i]):
            continue

        cur = float(x[i]) + offset
        diff = cur - prev

        if diff > range_deg / 2.0:
            offset -= range_deg
            cur -= range_deg
            changed = True
        elif diff < -range_deg / 2.0:
            offset += range_deg
            cur += range_deg
            changed = True

        x[i] = cur
        prev = cur

    return x, changed


def postprocess_joint_angles(
    df: pl.DataFrame,
    *,
    frame_col: str = "Frame",
    unify_lr_sign: bool = True,
    baseline_frames: tuple[int, int] | None = (1, 11),
    flip_columns: Iterable[str] | None = None,
    resolve_discontinuity_range_deg: float | None = 360.0,
    resolve_discontinuity_columns: Iterable[str] | None = None,
) -> tuple[pl.DataFrame, JointAnglePostprocessMeta]:
    """Apply analysis-friendly post-processing to a joint-angle time series.

    Parameters
    ----------
    df:
        Polars DataFrame containing a frame column and angle columns.
    frame_col:
        Name of the frame index column (e.g., ``"Frame"`` or ``"MocapFrame"``).
    unify_lr_sign:
        If True, flips selected LEFT Y/Z columns (multiplies by -1) so that
        left/right have consistent sign meaning.
    baseline_frames:
        If provided, subtract the mean of each angle column over this inclusive
        frame window ``(start, end)``.

        The default (1, 11) corresponds to python indices [0..10] often used for
        quiet standing selection.
    flip_columns:
        Optional explicit list of columns to flip. If None, uses the default set
        (Hip/Knee/Ankle LEFT Y/Z).
    resolve_discontinuity_range_deg:
        If not None, apply Visual3D Resolve_Discontinuity-style unwrapping to designated
        angle columns prior to baseline subtraction. Typical value is 360 degrees.
        Set to None to disable.
    resolve_discontinuity_columns:
        Optional explicit list of angle columns to apply discontinuity resolution to.
        If None, uses the project default: `Ankle_L_Z_deg` and `Ankle_R_Z_deg` (only if present).

    Returns
    -------
    df_out, meta
        Postprocessed DataFrame (same columns) and metadata.
    """

    if frame_col not in df.columns:
        raise KeyError(f"Frame column not found: {frame_col!r}")

    angle_cols = [c for c in df.columns if c.endswith("_deg")]

    flipped: list[str] = []
    resolved: list[str] = []
    out = df

    # 1) Sign unification (LEFT Y/Z)
    if unify_lr_sign:
        flip_cols = list(flip_columns) if flip_columns is not None else _default_flip_columns(df.columns)
        if flip_cols:
            out = out.with_columns([(-pl.col(c)).alias(c) for c in flip_cols])
            flipped = flip_cols

    # 2) Discontinuity resolution (Resolve_Discontinuity-style; Ankle Z only)
    if resolve_discontinuity_range_deg is not None:
        range_deg = float(resolve_discontinuity_range_deg)
        if range_deg <= 0:
            raise ValueError(
                f"resolve_discontinuity_range_deg must be > 0. Got {resolve_discontinuity_range_deg!r}"
            )

        cols = (
            [c for c in resolve_discontinuity_columns]  # type: ignore[arg-type]
            if resolve_discontinuity_columns is not None
            else _default_discontinuity_columns(out.columns)
        )
        cols = [c for c in cols if c in out.columns]

        repl: list[pl.Series] = []
        for c in cols:
            x = out.get_column(c).to_numpy()
            if not _needs_resolve_discontinuity(x, range_deg=range_deg):
                continue
            x2, changed = _resolve_discontinuity_deg(x, range_deg=range_deg)
            if changed:
                repl.append(pl.Series(c, x2))
                resolved.append(c)

        if repl:
            out = out.with_columns(repl)

    # 3) Baseline subtraction
    baseline_values: dict[str, float] = {}
    if baseline_frames is not None:
        b0, b1 = int(baseline_frames[0]), int(baseline_frames[1])
        if b1 < b0:
            raise ValueError(f"baseline_frames must satisfy end>=start. Got {baseline_frames!r}")

        base_df = out.filter(pl.col(frame_col).is_between(b0, b1, closed="both"))
        if base_df.height == 0:
            raise ValueError(
                f"No rows found for baseline window {baseline_frames!r} using frame_col={frame_col!r}."
            )

        base_row = base_df.select([pl.col(c).mean().alias(c) for c in angle_cols]).row(0)
        baseline_values = {c: float(v) for c, v in zip(angle_cols, base_row)}
        out = out.with_columns([(pl.col(c) - pl.lit(baseline_values[c])).alias(c) for c in angle_cols])

    meta = JointAnglePostprocessMeta(
        frame_col=str(frame_col),
        baseline_frame_start=int(baseline_frames[0]) if baseline_frames is not None else -1,
        baseline_frame_end=int(baseline_frames[1]) if baseline_frames is not None else -1,
        flipped_columns=tuple(flipped),
        baseline_values=baseline_values,
        resolved_columns=tuple(resolved),
        resolve_discontinuity_range_deg=resolve_discontinuity_range_deg,
    )

    return out, meta
