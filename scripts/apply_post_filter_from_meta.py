"""Apply post-batch filtering based on meta/platform sheets.

This script mirrors the filtering logic in `add_meta.ipynb` except the
`actual_velocity` join step (explicitly excluded by user request).

Filtering rule:
1) mixed == 1
2) age_group == "young"
3) keep rows where:
   - step_TF == "nonstep", or
   - step_TF == "step" and ipsilateral stepping:
       (dominant == "R" and state == "step_R") or
       (dominant == "L" and state == "step_L")
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply add_meta.ipynb filtering (excluding actual_velocity join) to "
            "a batch timeseries CSV."
        )
    )
    parser.add_argument(
        "--in_csv",
        default=str(_REPO_ROOT / "output" / "all_trials_timeseries.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--event_xlsm",
        default=str(_REPO_ROOT / "data" / "perturb_inform.xlsm"),
        help="Event/meta workbook path.",
    )
    parser.add_argument(
        "--out_csv",
        default=str(_REPO_ROOT / "output" / "all_trials_timeseries.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Output text encoding (default: utf-8-sig).",
    )
    return parser


def _load_meta_with_age_group(event_xlsm: Path) -> pl.DataFrame:
    meta_wide = pl.read_excel(str(event_xlsm), sheet_name="meta")
    if "subject" not in meta_wide.columns:
        raise ValueError("meta sheet is missing required column: subject")

    items = [name if name is not None else f"unknown_{i}" for i, name in enumerate(meta_wide["subject"].to_list())]
    subject_cols = [c for c in meta_wide.columns if c != "subject"]
    if not subject_cols:
        raise ValueError("meta sheet has no subject columns.")

    transposed = meta_wide.select(subject_cols).transpose(include_header=False)
    transposed.columns = items
    if "나이" not in transposed.columns or "주손 or 주발" not in transposed.columns:
        raise ValueError("meta transpose is missing one of required columns: 나이, 주손 or 주발")

    meta = (
        transposed.with_columns(pl.Series("subject", subject_cols))
        .with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("나이").cast(pl.Int32, strict=False).alias("나이"),
            pl.col("주손 or 주발").cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase().alias("주손 or 주발"),
        )
        .with_columns(
            pl.when(pl.col("나이") < 30)
            .then(pl.lit("young"))
            .otherwise(pl.lit("old"))
            .alias("age_group")
        )
        .select(["subject", "age_group", "주손 or 주발"])
    )
    return meta


def _load_platform_trial_meta(event_xlsm: Path) -> pl.DataFrame:
    platform = pl.read_excel(str(event_xlsm), sheet_name="platform")
    required = {"subject", "velocity", "trial", "step_TF", "state", "mixed"}
    missing = [c for c in required if c not in platform.columns]
    if missing:
        raise ValueError(f"platform sheet missing required columns: {missing}")

    return (
        platform.select(["subject", "velocity", "trial", "step_TF", "state", "mixed"])
        .with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial").cast(pl.Int64, strict=False),
            pl.col("step_TF").cast(pl.Utf8, strict=False).str.strip_chars(),
            pl.col("state").cast(pl.Utf8, strict=False).str.strip_chars(),
            pl.col("mixed").cast(pl.Float64, strict=False),
        )
    )


def _load_input_csv(in_csv: Path) -> pl.DataFrame:
    df = pl.read_csv(str(in_csv))
    required = {"subject", "velocity", "trial"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"input csv missing required key columns: {missing}")
    return (
        df.with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial").cast(pl.Int64, strict=False),
        )
    )


def _apply_filter(df: pl.DataFrame) -> pl.DataFrame:
    mask_base = (pl.col("mixed") == 1) & (pl.col("age_group") == "young")
    is_step = pl.col("step_TF") == "step"
    ipsilateral_step = is_step & (
        ((pl.col("주손 or 주발") == "R") & (pl.col("state") == "step_R"))
        | ((pl.col("주손 or 주발") == "L") & (pl.col("state") == "step_L"))
    )
    is_nonstep = pl.col("step_TF") == "nonstep"
    return df.filter(mask_base & (ipsilateral_step | is_nonstep))


def _write_csv(df: pl.DataFrame, out_csv: Path, encoding: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_name(f"{out_csv.name}.tmp")
    csv_text = df.write_csv()
    with tmp.open("w", encoding=encoding, newline="") as f:
        f.write(csv_text)
    tmp.replace(out_csv)


def main() -> None:
    args = _make_parser().parse_args()
    in_csv = Path(args.in_csv)
    event_xlsm = Path(args.event_xlsm)
    out_csv = Path(args.out_csv)

    if not in_csv.exists():
        raise FileNotFoundError(f"input csv not found: {in_csv}")
    if not event_xlsm.exists():
        raise FileNotFoundError(f"event_xlsm not found: {event_xlsm}")

    base = _load_input_csv(in_csv)
    meta = _load_meta_with_age_group(event_xlsm)
    platform = _load_platform_trial_meta(event_xlsm)

    # Replace existing columns to avoid stale values.
    base = base.drop(["age_group", "주손 or 주발", "step_TF", "state", "mixed"], strict=False)
    merged = base.join(meta, on="subject", how="left").join(
        platform,
        on=["subject", "velocity", "trial"],
        how="left",
    )

    required_after_join = ["age_group", "주손 or 주발", "step_TF", "state", "mixed"]
    missing_after_join = [c for c in required_after_join if c not in merged.columns]
    if missing_after_join:
        raise ValueError(f"failed to build filtering columns: {missing_after_join}")

    filtered = _apply_filter(merged)
    _write_csv(filtered, out_csv, args.encoding)

    print(
        "[OK] apply_post_filter_from_meta "
        f"input_rows={base.height}, output_rows={filtered.height}, output={out_csv}"
    )


if __name__ == "__main__":
    main()
