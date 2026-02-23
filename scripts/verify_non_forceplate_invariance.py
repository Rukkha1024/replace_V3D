"""Verify non-forceplate columns are unchanged before/after forceplate fixes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import polars as pl


def _is_forceplate_family_col(col: str) -> bool:
    prefixes = (
        "GRF_",
        "GRM_",
        "COP_",
        "AnkleTorque",
        "FP_origin_",
        "L_ankleJC_",
        "R_ankleJC_",
        "AnkleMid_",
        "time_from_platform_onset_s",
    )
    return col.startswith(prefixes)


def main() -> None:
    pl.Config.set_tbl_rows(999)
    pl.Config.set_tbl_cols(999)
    pl.Config.set_tbl_width_chars(120)

    ap = argparse.ArgumentParser()
    ap.add_argument("--before_csv", default="/tmp/all_trials_timeseries_before_stage01_forceplate.csv")
    ap.add_argument("--after_csv", default="output/all_trials_timeseries.csv")
    ap.add_argument("--out_summary_csv", default="output/non_forceplate_invariance_summary.csv")
    ap.add_argument("--out_examples_csv", default="output/non_forceplate_invariance_mismatch_examples.csv")
    ap.add_argument("--abs_tol", type=float, default=1e-12)
    ap.add_argument("--max_examples", type=int, default=300)
    ap.add_argument("--encoding", default="utf-8-sig")
    args = ap.parse_args()

    before_path = Path(args.before_csv)
    after_path = Path(args.after_csv)
    out_summary = Path(args.out_summary_csv)
    out_examples = Path(args.out_examples_csv)
    include_bom = str(args.encoding).lower().replace("_", "-") == "utf-8-sig"

    if not before_path.exists():
        raise FileNotFoundError(f"before_csv not found: {before_path}")
    if not after_path.exists():
        raise FileNotFoundError(f"after_csv not found: {after_path}")

    before = pl.read_csv(str(before_path))
    after = pl.read_csv(str(after_path))
    keys = ["subject", "velocity", "trial", "MocapFrame"]
    for k in keys:
        if k not in before.columns or k not in after.columns:
            raise ValueError(f"Key column missing in before/after: {k}")

    before_keys = before.select(keys)
    after_keys = after.select(keys)
    before_only_n = before_keys.join(after_keys, on=keys, how="anti").height
    after_only_n = after_keys.join(before_keys, on=keys, how="anti").height

    common_cols = [c for c in before.columns if c in set(after.columns)]
    target_cols = [c for c in common_cols if c not in keys and not _is_forceplate_family_col(c)]

    joined = before.join(after, on=keys, how="inner", suffix="_after")
    if joined.is_empty():
        raise RuntimeError("No overlapping rows between before/after CSVs.")

    summary_rows: List[Dict[str, object]] = [
        {"column": "__meta__", "kind": "before_only_rows", "mismatch_n": int(before_only_n), "compared_n": int(before.height)},
        {"column": "__meta__", "kind": "after_only_rows", "mismatch_n": int(after_only_n), "compared_n": int(after.height)},
    ]
    example_rows: List[Dict[str, object]] = []

    for col in target_cols:
        after_col = f"{col}_after"
        if after_col not in joined.columns:
            continue

        s_before = joined[col]
        s_after = joined[after_col]
        before_dtype = s_before.dtype
        is_numeric = before_dtype in (
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        )

        cmp = joined.select(keys + [pl.col(col).alias("__before"), pl.col(after_col).alias("__after")])
        if is_numeric:
            cmp = cmp.with_columns(
                [
                    pl.col("__before").cast(pl.Float64).alias("__before_f"),
                    pl.col("__after").cast(pl.Float64).alias("__after_f"),
                ]
            ).with_columns(
                (
                    ~(
                        ((pl.col("__after_f") - pl.col("__before_f")).abs() <= float(args.abs_tol))
                        | (pl.col("__before_f").is_nan() & pl.col("__after_f").is_nan())
                        | (pl.col("__before_f").is_null() & pl.col("__after_f").is_null())
                    )
                ).alias("__mismatch")
            )
        else:
            cmp = cmp.with_columns(
                (
                    ~(
                        (pl.col("__before") == pl.col("__after"))
                        | (pl.col("__before").is_null() & pl.col("__after").is_null())
                    )
                ).alias("__mismatch")
            )

        mismatch_n = int(cmp.filter(pl.col("__mismatch") == True).height)  # noqa: E712
        summary_rows.append(
            {
                "column": col,
                "kind": "non_forceplate",
                "mismatch_n": mismatch_n,
                "compared_n": int(cmp.height),
            }
        )

        if mismatch_n > 0 and len(example_rows) < int(args.max_examples):
            rem = int(args.max_examples) - len(example_rows)
            sample = cmp.filter(pl.col("__mismatch") == True).head(rem)
            for row in sample.iter_rows(named=True):
                example_rows.append(
                    {
                        "column": col,
                        "subject": row["subject"],
                        "velocity": row["velocity"],
                        "trial": row["trial"],
                        "MocapFrame": row["MocapFrame"],
                        "before": row["__before"],
                        "after": row["__after"],
                    }
                )

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_examples.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summary_rows).write_csv(out_summary, include_header=True, include_bom=include_bom)
    pl.DataFrame(example_rows).write_csv(out_examples, include_header=True, include_bom=include_bom)

    non_meta = [r for r in summary_rows if r["kind"] == "non_forceplate"]
    changed_cols_n = int(sum(1 for r in non_meta if int(r["mismatch_n"]) > 0))
    total_mismatch_n = int(sum(int(r["mismatch_n"]) for r in non_meta))

    print("[OK] non-forceplate invariance check complete")
    print(f"     compared_rows(inner): {joined.height}")
    print(f"     before_only_rows: {before_only_n}")
    print(f"     after_only_rows:  {after_only_n}")
    print(f"     changed_non_forceplate_columns: {changed_cols_n}")
    print(f"     total_non_forceplate_mismatches: {total_mismatch_n}")
    print(f"     summary_csv: {out_summary}")
    print(f"     examples_csv: {out_examples}")


if __name__ == "__main__":
    main()
