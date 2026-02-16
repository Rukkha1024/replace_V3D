"""Export schema finalization helpers.

Goal
----
Keep output schemas *clean* and stable by removing legacy / duplicate columns that
represent the same variable under an older name.

This repo previously kept backward-compatible alias columns (e.g. MOS_*_dir) to
avoid breaking downstream plots/scripts. Once the canonical columns are in place,
exports should not save both.
"""

from __future__ import annotations

from typing import Any, Sequence


_LEGACY_MOS_ALIAS_TO_CANONICAL: dict[str, str] = {
    # Backward-compatible aliases (same values) -> canonical Visual3D Closest_Bound outputs
    "MOS_AP_dir": "MOS_AP_v3d",
    "MOS_ML_dir": "MOS_ML_v3d",
}


def _legacy_mos_alias_drop_candidates(columns: Sequence[str]) -> list[str]:
    cols = set(columns)
    drop: list[str] = []
    for alias, canonical in _LEGACY_MOS_ALIAS_TO_CANONICAL.items():
        if alias in cols and canonical in cols:
            drop.append(alias)
    return drop


def drop_legacy_mos_alias_columns(columns: Sequence[str]) -> list[str]:
    """Return `columns` without legacy MOS alias columns when canonical columns exist."""

    drop = set(_legacy_mos_alias_drop_candidates(columns))
    if not drop:
        return list(columns)
    return [c for c in columns if c not in drop]


def finalize_export_df(df: Any, *, export_kind: str | None = None) -> Any:
    """Drop legacy/duplicate columns from an export DataFrame (polars or pandas).

    Parameters
    ----------
    df:
        Either a `polars.DataFrame` or `pandas.DataFrame`.
    export_kind:
        Optional label to help future debugging/logging. Currently unused.

    Returns
    -------
    df_out:
        Same type as input `df`, with legacy alias columns removed when safe.
    """

    _ = export_kind  # reserved for future use

    try:
        columns = list(df.columns)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Unsupported export DF type (missing .columns): {type(df)!r}") from exc

    drop_cols = _legacy_mos_alias_drop_candidates(columns)
    if not drop_cols:
        return df

    try:
        import polars as pl  # type: ignore
    except Exception:  # pragma: no cover
        pl = None  # type: ignore

    if pl is not None and isinstance(df, pl.DataFrame):
        return df.drop(drop_cols)

    try:
        import pandas as pd  # type: ignore
    except Exception:  # pragma: no cover
        pd = None  # type: ignore

    if pd is not None and isinstance(df, pd.DataFrame):
        return df.drop(columns=drop_cols)

    raise TypeError(f"Unsupported export DF type: {type(df)!r}")

