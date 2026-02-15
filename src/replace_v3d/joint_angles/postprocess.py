"""Post-processing helpers for Visual3D-like joint angle outputs.

This module is intentionally post-process only: it does not change the underlying
segment definitions or joint-angle math. It exists to provide two analysis-friendly
conventions that are commonly needed when comparing left vs right:

1) Anatomical "presentation" convention:
   Visual3D-style right-hand-rule joint angles often produce opposite sign meanings
   for LEFT vs RIGHT in Y/Z (ab/adduction and internal/external rotation).
   A common practice is to negate LEFT Y/Z so that the sign meaning matches the
   RIGHT side.

2) Baseline-normalized convention:
   Subtract the mean of a quiet-standing baseline window to remove static offsets
   (e.g., small SCS misalignments). This is useful for comparing Δangles.

The repo keeps the raw joint-angle CSV schema unchanged for reproducibility and MD5
validation. Use these helpers to generate additional outputs (`*_anat`, `*_ana0`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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


def postprocess_joint_angles(
    df: pl.DataFrame,
    *,
    frame_col: str = "Frame",
    unify_lr_sign: bool = True,
    baseline_frames: tuple[int, int] | None = (1, 11),
    flip_columns: Iterable[str] | None = None,
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

    Returns
    -------
    df_out, meta
        Postprocessed DataFrame (same columns) and metadata.
    """

    if frame_col not in df.columns:
        raise KeyError(f"Frame column not found: {frame_col!r}")

    angle_cols = [c for c in df.columns if c.endswith("_deg")]

    flipped: list[str] = []
    out = df

    # 1) Sign unification (LEFT Y/Z)
    if unify_lr_sign:
        flip_cols = list(flip_columns) if flip_columns is not None else _default_flip_columns(df.columns)
        if flip_cols:
            out = out.with_columns([(-pl.col(c)).alias(c) for c in flip_cols])
            flipped = flip_cols

    # 2) Baseline subtraction
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
    )

    return out, meta

