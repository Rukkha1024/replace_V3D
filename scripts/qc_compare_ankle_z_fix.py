from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl


def md5_of(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.md5(path.read_bytes()).hexdigest()


def ptp(series: pl.Series) -> float:
    vals = series.drop_nulls()
    if vals.len() == 0:
        return float("nan")
    return float(vals.max() - vals.min())


def write_tsv_bom(df: pl.DataFrame, path: Path) -> None:
    text = df.write_csv(separator="\t")
    path.write_text(text, encoding="utf-8-sig")


def load_reference_md5(reference_file: Path, target_path: str) -> str:
    if not reference_file.exists():
        return "MISSING_REFERENCE_FILE"
    ref_df = pl.read_csv(reference_file, separator="\t")
    hit = ref_df.filter(pl.col("path") == target_path)
    if hit.height == 0:
        return "MISSING_REFERENCE_ROW"
    return str(hit["md5"][0])


def main() -> int:
    root = Path("output/qc/ankle_z_fix")
    ref_csv = root / "ref/251128_강비은_perturb_30_004_JOINT_ANGLES_preStep.csv"
    new_csv = root / "new/251128_강비은_perturb_30_004_JOINT_ANGLES_preStep.csv"

    ref_fig = root / "ref_fig/joint_angles_lower__subject-강비은__velocity-30__sample.png"
    new_fig = root / "new_fig/joint_angles_lower__subject-강비은__velocity-30__sample.png"

    all_trials_csv = Path("output/all_trials_timeseries.csv")

    md5_dir = root / "md5"
    md5_dir.mkdir(parents=True, exist_ok=True)

    if not ref_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {ref_csv}")
    if not new_csv.exists():
        raise FileNotFoundError(f"New CSV not found: {new_csv}")

    ref_df = pl.read_csv(ref_csv)
    new_df = pl.read_csv(new_csv)

    range_rows = []
    for col in ("Ankle_L_Z_deg", "Ankle_R_Z_deg"):
        ref_ptp = ptp(ref_df[col])
        new_ptp = ptp(new_df[col])
        range_rows.append(
            {
                "metric": col,
                "ref_ptp_deg": ref_ptp,
                "new_ptp_deg": new_ptp,
                "delta_new_minus_ref_deg": new_ptp - ref_ptp,
                "is_micro_flat_new_ptp_lt_0p01": bool(new_ptp < 0.01),
            }
        )

    range_out = md5_dir / "ankle_z_range_comparison.tsv"
    write_tsv_bom(pl.DataFrame(range_rows), range_out)

    reference_md5_file = md5_dir / "reference_md5.tsv"
    all_trials_ref_md5 = load_reference_md5(reference_md5_file, "output/all_trials_timeseries.csv")

    md5_rows = [
        {
            "artifact": "single_trial_joint_angles_csv",
            "reference_path": str(ref_csv),
            "new_path": str(new_csv),
            "reference_md5": md5_of(ref_csv),
            "new_md5": md5_of(new_csv),
        },
        {
            "artifact": "grid_joint_angles_lower_sample_png",
            "reference_path": str(ref_fig),
            "new_path": str(new_fig),
            "reference_md5": md5_of(ref_fig),
            "new_md5": md5_of(new_fig),
        },
        {
            "artifact": "all_trials_timeseries_csv",
            "reference_path": str(reference_md5_file),
            "new_path": str(all_trials_csv),
            "reference_md5": all_trials_ref_md5,
            "new_md5": md5_of(all_trials_csv),
        },
    ]
    for row in md5_rows:
        row["same"] = row["reference_md5"] == row["new_md5"]

    md5_out = md5_dir / "md5_ankle_z_fix_ref_vs_new.tsv"
    write_tsv_bom(pl.DataFrame(md5_rows), md5_out)

    print(f"[OK] Wrote: {range_out}")
    print(f"[OK] Wrote: {md5_out}")
    print("[SUMMARY] ankle Z peak-to-peak comparison")
    print(pl.DataFrame(range_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
