"""Export schema finalization helpers.

Goal
----
Keep output schemas *clean* and stable after per-trial payload assembly.
"""

from __future__ import annotations

from typing import Any, Sequence


def _legacy_mos_alias_drop_candidates(columns: Sequence[str]) -> list[str]:
    _ = columns
    return []


def drop_legacy_mos_alias_columns(columns: Sequence[str]) -> list[str]:
    """Return `columns` unchanged.

    Legacy MOS alias columns are no longer part of the export contract.
    """

    _ = _legacy_mos_alias_drop_candidates(columns)
    return list(columns)


def finalize_export_df(df: Any, *, export_kind: str | None = None) -> Any:
    """Finalize an export DataFrame.

    Parameters
    ----------
    df:
        Either a `polars.DataFrame` or `pandas.DataFrame`.
    export_kind:
        Optional label to help future debugging/logging. Currently unused.

    Returns
    -------
    df_out:
        Same type as input `df` (currently pass-through).
    """

    _ = export_kind  # reserved for future use

    try:
        columns = list(df.columns)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Unsupported export DF type (missing .columns): {type(df)!r}") from exc

    _ = _legacy_mos_alias_drop_candidates(columns)
    return df
