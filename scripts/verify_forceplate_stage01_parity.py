"""Verify forceplate parity against shared_files Stage01 outputs.

Compares shared Stage01 (parquet) and current repo batch CSV on overlapping
trial-frames after harmonizing to 100 Hz local index.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

import polars as pl

import _bootstrap

_bootstrap.ensure_src_on_path()


def _md5_of_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_shared_100hz(shared_parquet: Path) -> pl.DataFrame:
    lf = (
        pl.scan_parquet(str(shared_parquet))
        .select(
            [
                pl.col("subject").cast(pl.Utf8).str.strip_chars().alias("subject"),
                pl.col("velocity").cast(pl.Float64).alias("velocity"),
                pl.col("trial_num").cast(pl.Int64).alias("trial"),
                pl.col("DeviceFrame").cast(pl.Int64).alias("DeviceFrame"),
                pl.col("MocapFrame").cast(pl.Float64).alias("MocapFrame_abs"),
                pl.col("Cx").cast(pl.Float64).alias("Cx"),
                pl.col("Cy").cast(pl.Float64).alias("Cy"),
                pl.col("Fx").cast(pl.Float64).alias("Fx"),
                pl.col("Fy").cast(pl.Float64).alias("Fy"),
                pl.col("Fz").cast(pl.Float64).alias("Fz"),
                pl.col("Mx").cast(pl.Float64).alias("Mx"),
                pl.col("My").cast(pl.Float64).alias("My"),
                pl.col("Mz").cast(pl.Float64).alias("Mz"),
            ]
        )
        .with_columns((pl.col("DeviceFrame") // 10).cast(pl.Int64).alias("mocap_idx_local"))
        .group_by(["subject", "velocity", "trial", "mocap_idx_local"])
        .agg(
            [
                pl.col("MocapFrame_abs").mean().alias("MocapFrame_abs"),
                pl.col("Cx").mean().alias("Cx"),
                pl.col("Cy").mean().alias("Cy"),
                pl.col("Fx").mean().alias("Fx"),
                pl.col("Fy").mean().alias("Fy"),
                pl.col("Fz").mean().alias("Fz"),
                pl.col("Mx").mean().alias("Mx"),
                pl.col("My").mean().alias("My"),
                pl.col("Mz").mean().alias("Mz"),
            ]
        )
        .sort(["subject", "velocity", "trial", "mocap_idx_local"])
    )
    return lf.collect()


def _build_repo(repo_csv: Path) -> pl.DataFrame:
    lf = (
        pl.scan_csv(str(repo_csv), infer_schema_length=5000)
        .select(
            [
                pl.col("subject").cast(pl.Utf8).str.strip_chars().alias("subject"),
                pl.col("velocity").cast(pl.Float64).alias("velocity"),
                pl.col("trial").cast(pl.Int64).alias("trial"),
                (pl.col("MocapFrame").cast(pl.Int64) - 1).alias("mocap_idx_local"),
                pl.col("time_from_platform_onset_s").cast(pl.Float64).alias("time_from_platform_onset_s"),
                pl.col("COP_X_m").cast(pl.Float64).alias("COP_X_m"),
                pl.col("COP_Y_m").cast(pl.Float64).alias("COP_Y_m"),
                pl.col("GRF_X_N").cast(pl.Float64).alias("GRF_X_N"),
                pl.col("GRF_Y_N").cast(pl.Float64).alias("GRF_Y_N"),
                pl.col("GRF_Z_N").cast(pl.Float64).alias("GRF_Z_N"),
                pl.col("GRM_X_Nm_at_FPorigin").cast(pl.Float64).alias("GRM_X"),
                pl.col("GRM_Y_Nm_at_FPorigin").cast(pl.Float64).alias("GRM_Y"),
                pl.col("GRM_Z_Nm_at_FPorigin").cast(pl.Float64).alias("GRM_Z"),
            ]
        )
    )
    return lf.collect()


def _to_canonical(df: pl.DataFrame, *, value_cols: Iterable[str], round_decimals: int) -> pl.DataFrame:
    out = df.select(
        [
            pl.col("subject"),
            pl.col("velocity"),
            pl.col("trial"),
            pl.col("mocap_idx_local"),
            *[pl.col(c).cast(pl.Float64).round(round_decimals).alias(c) for c in value_cols],
        ]
    )
    return out.sort(["subject", "velocity", "trial", "mocap_idx_local"])


def main() -> None:
    pl.Config.set_tbl_rows(999)
    pl.Config.set_tbl_cols(999)
    pl.Config.set_tbl_width_chars(120)

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shared_parquet",
        default=str(
            Path("/mnt/c/Users/Alice/OneDrive - 청주대학교/근전도 분석 코드/shared_files/output/01_dataset")
            / "merged_data_comprehensive.parquet"
        ),
    )
    ap.add_argument("--repo_csv", default="output/all_trials_timeseries.csv")
    ap.add_argument("--out_summary_csv", default="output/forceplate_stage01_parity_summary.csv")
    ap.add_argument("--out_detail_csv", default="output/forceplate_stage01_parity_detail.csv")
    ap.add_argument("--out_mismatch_by_col_csv", default="output/forceplate_stage01_mismatch_by_col.csv")
    ap.add_argument("--out_shared_canonical_csv", default="output/forceplate_stage01_shared_canonical.csv")
    ap.add_argument("--out_repo_canonical_csv", default="output/forceplate_stage01_repo_canonical.csv")
    ap.add_argument("--out_md5_report_txt", default="output/forceplate_stage01_md5_report.txt")
    ap.add_argument("--round_decimals", type=int, default=9)
    ap.add_argument("--abs_tol", type=float, default=1e-9)
    ap.add_argument("--encoding", default="utf-8-sig")
    args = ap.parse_args()

    shared_path = Path(args.shared_parquet)
    repo_path = Path(args.repo_csv)
    out_summary = Path(args.out_summary_csv)
    out_detail = Path(args.out_detail_csv)
    out_mismatch = Path(args.out_mismatch_by_col_csv)
    out_shared_can = Path(args.out_shared_canonical_csv)
    out_repo_can = Path(args.out_repo_canonical_csv)
    out_md5_txt = Path(args.out_md5_report_txt)
    include_bom = str(args.encoding).lower().replace("_", "-") == "utf-8-sig"

    if not shared_path.exists():
        raise FileNotFoundError(f"shared parquet not found: {shared_path}")
    if not repo_path.exists():
        raise FileNotFoundError(f"repo csv not found: {repo_path}")

    shared = _build_shared_100hz(shared_path)
    repo = _build_repo(repo_path)
    joined = shared.join(repo, on=["subject", "velocity", "trial", "mocap_idx_local"], how="inner")
    if joined.is_empty():
        raise RuntimeError("No overlapping rows between shared and repo for parity check.")
    joined = (
        joined.with_columns(
            pl.col("mocap_idx_local")
            .filter(pl.col("time_from_platform_onset_s") == 0.0)
            .first()
            .over(["subject", "velocity", "trial"])
            .alias("__onset_idx_repo")
        )
        .with_columns(
            [
                (
                    pl.col("Fx")
                    - pl.col("Fx")
                    .filter(pl.col("mocap_idx_local") == pl.col("__onset_idx_repo"))
                    .first()
                    .over(["subject", "velocity", "trial"])
                ).alias("Fx_onset0"),
                (
                    pl.col("Fy")
                    - pl.col("Fy")
                    .filter(pl.col("mocap_idx_local") == pl.col("__onset_idx_repo"))
                    .first()
                    .over(["subject", "velocity", "trial"])
                ).alias("Fy_onset0"),
                (
                    pl.col("Fz")
                    - pl.col("Fz")
                    .filter(pl.col("mocap_idx_local") == pl.col("__onset_idx_repo"))
                    .first()
                    .over(["subject", "velocity", "trial"])
                ).alias("Fz_onset0"),
                (
                    pl.col("Mx")
                    - pl.col("Mx")
                    .filter(pl.col("mocap_idx_local") == pl.col("__onset_idx_repo"))
                    .first()
                    .over(["subject", "velocity", "trial"])
                ).alias("Mx_onset0"),
                (
                    pl.col("My")
                    - pl.col("My")
                    .filter(pl.col("mocap_idx_local") == pl.col("__onset_idx_repo"))
                    .first()
                    .over(["subject", "velocity", "trial"])
                ).alias("My_onset0"),
                (
                    pl.col("Mz")
                    - pl.col("Mz")
                    .filter(pl.col("mocap_idx_local") == pl.col("__onset_idx_repo"))
                    .first()
                    .over(["subject", "velocity", "trial"])
                ).alias("Mz_onset0"),
            ]
        )
        .drop(["__onset_idx_repo"])
    )

    detail = joined.select(
        [
            "subject",
            "velocity",
            "trial",
            "mocap_idx_local",
            "Cx",
            "Cy",
            "COP_X_m",
            "COP_Y_m",
            "Fx_onset0",
            "Fy_onset0",
            "Fz_onset0",
            "GRF_X_N",
            "GRF_Y_N",
            "GRF_Z_N",
            "Mx_onset0",
            "My_onset0",
            "Mz_onset0",
            "GRM_X",
            "GRM_Y",
            "GRM_Z",
            (pl.col("COP_X_m") - pl.col("Cx")).alias("diff_COP_X"),
            (pl.col("COP_Y_m") - pl.col("Cy")).alias("diff_COP_Y"),
            (pl.col("GRF_X_N") - pl.col("Fx_onset0")).alias("diff_GRF_X"),
            (pl.col("GRF_Y_N") - pl.col("Fy_onset0")).alias("diff_GRF_Y"),
            (pl.col("GRF_Z_N") - pl.col("Fz_onset0")).alias("diff_GRF_Z"),
            (pl.col("GRM_X") - pl.col("Mx_onset0")).alias("diff_GRM_X"),
            (pl.col("GRM_Y") - pl.col("My_onset0")).alias("diff_GRM_Y"),
            (pl.col("GRM_Z") - pl.col("Mz_onset0")).alias("diff_GRM_Z"),
        ]
    )

    summary = detail.select(
        [
            pl.len().alias("n_rows"),
            pl.col("subject").n_unique().alias("n_subjects"),
            pl.struct(["subject", "velocity", "trial"]).n_unique().alias("n_units"),
            pl.corr("COP_X_m", "Cx").alias("corr_COP_X"),
            pl.corr("COP_Y_m", "Cy").alias("corr_COP_Y"),
            pl.col("diff_COP_X").abs().mean().alias("mae_COP_X"),
            pl.col("diff_COP_Y").abs().mean().alias("mae_COP_Y"),
            pl.corr("GRF_X_N", "Fx_onset0").alias("corr_GRF_X"),
            pl.corr("GRF_Y_N", "Fy_onset0").alias("corr_GRF_Y"),
            pl.corr("GRF_Z_N", "Fz_onset0").alias("corr_GRF_Z"),
            pl.col("diff_GRF_X").abs().mean().alias("mae_GRF_X"),
            pl.col("diff_GRF_Y").abs().mean().alias("mae_GRF_Y"),
            pl.col("diff_GRF_Z").abs().mean().alias("mae_GRF_Z"),
            pl.corr("GRM_X", "Mx_onset0").alias("corr_GRM_X"),
            pl.corr("GRM_Y", "My_onset0").alias("corr_GRM_Y"),
            pl.corr("GRM_Z", "Mz_onset0").alias("corr_GRM_Z"),
            pl.col("diff_GRM_X").abs().mean().alias("mae_GRM_X"),
            pl.col("diff_GRM_Y").abs().mean().alias("mae_GRM_Y"),
            pl.col("diff_GRM_Z").abs().mean().alias("mae_GRM_Z"),
        ]
    )

    shared_cmp = detail.select(
        [
            "subject",
            "velocity",
            "trial",
            "mocap_idx_local",
            pl.col("Cx").alias("COP_X"),
            pl.col("Cy").alias("COP_Y"),
            pl.col("Fx_onset0").alias("GRF_X"),
            pl.col("Fy_onset0").alias("GRF_Y"),
            pl.col("Fz_onset0").alias("GRF_Z"),
            pl.col("Mx_onset0").alias("GRM_X"),
            pl.col("My_onset0").alias("GRM_Y"),
            pl.col("Mz_onset0").alias("GRM_Z"),
        ]
    )
    repo_cmp = detail.select(
        [
            "subject",
            "velocity",
            "trial",
            "mocap_idx_local",
            pl.col("COP_X_m").alias("COP_X"),
            pl.col("COP_Y_m").alias("COP_Y"),
            pl.col("GRF_X_N").alias("GRF_X"),
            pl.col("GRF_Y_N").alias("GRF_Y"),
            pl.col("GRF_Z_N").alias("GRF_Z"),
            pl.col("GRM_X").alias("GRM_X"),
            pl.col("GRM_Y").alias("GRM_Y"),
            pl.col("GRM_Z").alias("GRM_Z"),
        ]
    )

    shared_can = _to_canonical(
        shared_cmp,
        value_cols=["COP_X", "COP_Y", "GRF_X", "GRF_Y", "GRF_Z", "GRM_X", "GRM_Y", "GRM_Z"],
        round_decimals=int(args.round_decimals),
    )
    repo_can = _to_canonical(
        repo_cmp,
        value_cols=["COP_X", "COP_Y", "GRF_X", "GRF_Y", "GRF_Z", "GRM_X", "GRM_Y", "GRM_Z"],
        round_decimals=int(args.round_decimals),
    )

    mismatch_rows: List[Dict[str, object]] = []
    for col in ["COP_X", "COP_Y", "GRF_X", "GRF_Y", "GRF_Z", "GRM_X", "GRM_Y", "GRM_Z"]:
        d = (pl.col(f"{col}_repo") - pl.col(f"{col}_shared")).abs()
        cmp_df = repo_can.select(
            [
                pl.col("subject"),
                pl.col("velocity"),
                pl.col("trial"),
                pl.col("mocap_idx_local"),
                pl.col(col).alias(f"{col}_repo"),
            ]
        ).join(
            shared_can.select(
                [
                    pl.col("subject"),
                    pl.col("velocity"),
                    pl.col("trial"),
                    pl.col("mocap_idx_local"),
                    pl.col(col).alias(f"{col}_shared"),
                ]
            ),
            on=["subject", "velocity", "trial", "mocap_idx_local"],
            how="inner",
        )
        mism = (
            cmp_df.with_columns(d.alias("__abs_diff"))
            .filter(
                ~(
                    (pl.col("__abs_diff") <= float(args.abs_tol))
                    | (pl.col(f"{col}_repo").is_nan() & pl.col(f"{col}_shared").is_nan())
                    | (pl.col(f"{col}_repo").is_null() & pl.col(f"{col}_shared").is_null())
                )
            )
            .select(pl.len())
            .item()
        )
        mismatch_rows.append(
            {
                "column": col,
                "mismatch_n": int(mism),
                "compared_n": int(cmp_df.height),
            }
        )

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_detail.parent.mkdir(parents=True, exist_ok=True)
    out_mismatch.parent.mkdir(parents=True, exist_ok=True)
    out_shared_can.parent.mkdir(parents=True, exist_ok=True)
    out_repo_can.parent.mkdir(parents=True, exist_ok=True)
    out_md5_txt.parent.mkdir(parents=True, exist_ok=True)

    summary.write_csv(out_summary, include_header=True, include_bom=include_bom)
    detail.write_csv(out_detail, include_header=True, include_bom=include_bom)
    pl.DataFrame(mismatch_rows).write_csv(out_mismatch, include_header=True, include_bom=include_bom)
    shared_can.write_csv(out_shared_can, include_header=True, include_bom=include_bom)
    repo_can.write_csv(out_repo_can, include_header=True, include_bom=include_bom)

    md5_shared = _md5_of_file(out_shared_can)
    md5_repo = _md5_of_file(out_repo_can)
    md5_equal = bool(md5_shared == md5_repo)
    mismatch_total = int(sum(int(r["mismatch_n"]) for r in mismatch_rows))

    out_md5_txt.write_text(
        "\n".join(
            [
                "[MD5] canonical parity report",
                f"shared_canonical_csv: {out_shared_can}",
                f"repo_canonical_csv:   {out_repo_can}",
                f"md5_shared: {md5_shared}",
                f"md5_repo:   {md5_repo}",
                f"md5_equal:  {md5_equal}",
                f"mismatch_total: {mismatch_total}",
                f"mismatch_by_col_csv: {out_mismatch}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("[OK] forceplate Stage01 parity verification complete")
    print(f"     summary_csv: {out_summary}")
    print(f"     detail_csv:  {out_detail}")
    print(f"     md5_report:  {out_md5_txt}")
    print(f"     md5_equal:   {md5_equal}")
    print(f"     mismatch_total: {mismatch_total}")


if __name__ == "__main__":
    main()
