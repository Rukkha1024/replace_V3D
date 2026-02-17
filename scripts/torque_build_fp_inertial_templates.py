"""Build per-velocity forceplate inertial-subtraction templates (100 Hz) and save to NPZ.

Reads unloaded trial CSVs, downsamples 1000→100 Hz, averages across trials,
interpolates gaps, and applies Lab→ISB/GRF axis transform.
Consumed at runtime by the ankle torque pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None

import pandas as pd


DEFAULT_AXIS_TRANSFORM: Dict[str, Dict[str, object]] = {
    "Fx": {"source": "Fy", "scale": 1.0},
    "Fy": {"source": "Fx", "scale": 1.0},
    "Fz": {"source": "Fz", "scale": -1.0},
    "Mx": {"source": "My", "scale": 1.0},
    "My": {"source": "Mx", "scale": 1.0},
    "Mz": {"source": "Mz", "scale": -1.0},
    # COP columns can exist in some exports; keep mapping for completeness.
    "Cx": {"source": "Cy", "scale": 1.0},
    "Cy": {"source": "Cx", "scale": 1.0},
    "Cz": {"source": "Cz", "scale": -1.0},
}


def _nearest_index(values: np.ndarray, target: int) -> int:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0
    return int(np.argmin(np.abs(values - float(target))))


def _apply_axis_transform(df: "pl.DataFrame", mapping: Dict[str, Dict[str, object]]) -> "pl.DataFrame":
    if pl is None:
        raise RuntimeError("polars is required for this script")
    # Important: mapping can include swaps like Fx<-Fy and Fy<-Fx.
    # We must compute all transformed columns from the *original* df, not in-place,
    # otherwise the second assignment will read the already-overwritten value.
    temp_exprs: List["pl.Expr"] = []
    temp_to_target: Dict[str, str] = {}
    for tgt, rule in mapping.items():
        src = str(rule.get("source"))
        scale = float(rule.get("scale", 1.0))
        if src in df.columns:
            tmp = f"__axis_{tgt}"
            temp_exprs.append((pl.col(src) * scale).alias(tmp))
            temp_to_target[tmp] = tgt

    if not temp_exprs:
        return df

    out = df.with_columns(temp_exprs)
    out = out.with_columns([pl.col(tmp).alias(tgt) for tmp, tgt in temp_to_target.items()])
    return out.drop(list(temp_to_target.keys()))


def _read_forceplate_csv_table(
    csv_path: Path,
    *,
    required_columns: List[str],
    axis_transform: Optional[Dict[str, Dict[str, object]]] = None,
) -> "pl.DataFrame":
    """Read forceplate csv with a variable number of header rows.

    Mirrors Stage01 `_read_forceplate_csv_table`.
    """

    if pl is None:
        raise RuntimeError("polars is required for this script")

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        header_lines = f.readlines()

    header_row_idx: Optional[int] = None
    for i, line in enumerate(header_lines):
        if line.startswith("MocapFrame"):
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError(f"Header row starting with 'MocapFrame' not found in: {csv_path}")

    # Read with polars
    df = pl.read_csv(
        str(csv_path),
        skip_rows=header_row_idx,
        has_header=True,
        infer_schema_length=1000,
    )

    # Some exports include spaces after commas in the header:
    # "MocapFrame, MocapTime, DeviceFrame, Fx, ..."
    # Normalize by stripping whitespace from column names.
    rename_map = {c: c.strip() for c in df.columns if c.strip() != c}
    if rename_map:
        df = df.rename(rename_map)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing}")

    df = df.select(required_columns)
    if axis_transform:
        df = _apply_axis_transform(df, axis_transform)
    return df


def _build_templates(
    *,
    unload_data_dir: Path,
    timing_df: "pl.DataFrame",
    timing_cols: Dict[str, str],
    file_pattern: str,
    frame_ratio: int,
    agg: str,
    axis_transform: Optional[Dict[str, Dict[str, object]]],
) -> Dict[int, Dict[str, object]]:
    if pl is None:
        raise RuntimeError("polars is required for this script")

    velocity_col = timing_cols["velocity"]
    trial_col = timing_cols["trial"]
    onset_col = timing_cols["onset"]
    offset_col = timing_cols["offset"]

    df0 = timing_df.select([velocity_col, trial_col, onset_col, offset_col]).drop_nulls()
    df0 = df0.with_columns(
        [
            pl.col(velocity_col).cast(pl.Float64).alias(velocity_col),
            pl.col(trial_col).cast(pl.Int64).alias(trial_col),
            pl.col(onset_col).cast(pl.Int64).alias(onset_col),
            pl.col(offset_col).cast(pl.Int64).alias(offset_col),
        ]
    )
    df0 = df0.with_columns(pl.col(velocity_col).round(0).cast(pl.Int64).alias("velocity_int"))
    df0 = df0.with_columns((pl.col(offset_col) - pl.col(onset_col)).alias("duration"))

    dur_by_v = df0.group_by("velocity_int").agg(pl.col("duration").median().alias("unload_range_frames"))
    df0 = df0.join(dur_by_v, on="velocity_int", how="left")

    required_cols = ["MocapFrame", "MocapTime", "DeviceFrame", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    templates: Dict[int, Dict[str, object]] = {}

    for v_int in sorted(df0["velocity_int"].unique().to_list()):
        v_int = int(v_int)
        v_rows = df0.filter(pl.col("velocity_int") == v_int)
        # Stage01 behavior: median duration can be X.5 when n_trials is even.
        # Use Python's bankers rounding (round half to even) to match diagnostics.
        unload_range_frames = int(round(float(v_rows["unload_range_frames"].drop_nulls().item(0))))
        if unload_range_frames <= 0:
            continue

        seg_tables: List["pl.DataFrame"] = []
        used_trials: List[int] = []
        used_files: List[str] = []

        for row in v_rows.select([trial_col, onset_col, "unload_range_frames"]).iter_rows(named=True):
            trial = int(row[trial_col])
            onset = int(row[onset_col])
            # In Stage01, offset for unloaded template uses onset + unload_range_frames
            offset = int(onset + unload_range_frames)

            patt = file_pattern.format(velocity=v_int, trial=trial)
            matches = list(unload_data_dir.glob(patt))
            if not matches:
                # fallback: allow velocity formatted as float in filenames
                patt2 = file_pattern.format(velocity=float(v_int), trial=trial)
                matches = list(unload_data_dir.glob(patt2))
            if not matches:
                continue
            csv_path = matches[0]

            sub_df = _read_forceplate_csv_table(
                csv_path,
                required_columns=required_cols,
                axis_transform=axis_transform,
            )

            # Downsample to 100 Hz using DeviceFrame // frame_ratio (mean)
            sub_df_100 = (
                sub_df.with_columns((pl.col("DeviceFrame") // int(frame_ratio)).alias("mocap_idx_local"))
                .group_by("mocap_idx_local")
                .agg(
                    [
                        pl.col("MocapFrame").mean().alias("MocapFrame"),
                        pl.col("MocapTime").mean().alias("MocapTime"),
                        pl.col("Fx").mean().alias("Fx"),
                        pl.col("Fy").mean().alias("Fy"),
                        pl.col("Fz").mean().alias("Fz"),
                        pl.col("Mx").mean().alias("Mx"),
                        pl.col("My").mean().alias("My"),
                        pl.col("Mz").mean().alias("Mz"),
                    ]
                )
                .sort("mocap_idx_local")
                .with_columns(pl.col("MocapFrame").cast(pl.Int64).alias("MocapFrame"))
            )

            mocap_frame = sub_df_100["MocapFrame"].to_numpy()
            onset_local = _nearest_index(mocap_frame, onset)
            offset_local = _nearest_index(mocap_frame, offset)
            if offset_local < onset_local:
                onset_local, offset_local = offset_local, onset_local
            nseg = int(offset_local - onset_local + 1)
            if nseg <= 0:
                continue

            rel = mocap_frame[onset_local : offset_local + 1] - mocap_frame[onset_local]
            seg = (
                sub_df_100.slice(onset_local, nseg)
                .select(["Fx", "Fy", "Fz", "Mx", "My", "Mz"])
                .with_columns(pl.Series("relative_frame", rel))
            )
            seg_tables.append(seg)
            used_trials.append(trial)
            used_files.append(str(csv_path))

        if not seg_tables:
            continue

        stacked = pl.concat(seg_tables, how="vertical")
        # group/aggregate across trials
        if agg == "median":
            gb = stacked.group_by("relative_frame").agg(
                [
                    pl.col("Fx").median().alias("Fx"),
                    pl.col("Fy").median().alias("Fy"),
                    pl.col("Fz").median().alias("Fz"),
                    pl.col("Mx").median().alias("Mx"),
                    pl.col("My").median().alias("My"),
                    pl.col("Mz").median().alias("Mz"),
                ]
            )
        else:
            gb = stacked.group_by("relative_frame").agg(
                [
                    pl.col("Fx").mean().alias("Fx"),
                    pl.col("Fy").mean().alias("Fy"),
                    pl.col("Fz").mean().alias("Fz"),
                    pl.col("Mx").mean().alias("Mx"),
                    pl.col("My").mean().alias("My"),
                    pl.col("Mz").mean().alias("Mz"),
                ]
            )

        full = pl.DataFrame({"relative_frame": np.arange(unload_range_frames + 1, dtype=int)})
        template_df = (
            full.join(gb, on="relative_frame", how="left")
            .sort("relative_frame")
            .with_columns(
                [
                    pl.col("Fx").interpolate(),
                    pl.col("Fy").interpolate(),
                    pl.col("Fz").interpolate(),
                    pl.col("Mx").interpolate(),
                    pl.col("My").interpolate(),
                    pl.col("Mz").interpolate(),
                ]
            )
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
        )

        fx = template_df["Fx"].to_numpy()
        fy = template_df["Fy"].to_numpy()
        fz = template_df["Fz"].to_numpy()
        mx = template_df["Mx"].to_numpy()
        my = template_df["My"].to_numpy()
        mz = template_df["Mz"].to_numpy()

        # baseline shift (Stage01): template[0] == 0
        fx = fx - float(fx[0])
        fy = fy - float(fy[0])
        fz = fz - float(fz[0])
        mx = mx - float(mx[0])
        my = my - float(my[0])
        mz = mz - float(mz[0])

        templates[v_int] = {
            "velocity_int": v_int,
            "unload_range_frames": unload_range_frames,
            "n_trials": len(used_trials),
            "fx": fx,
            "fy": fy,
            "fz": fz,
            "mx": mx,
            "my": my,
            "mz": mz,
            "meta": {
                "agg": agg,
                "frame_ratio": int(frame_ratio),
                "timing_cols": timing_cols,
                "file_pattern": file_pattern,
                "used_trials": used_trials,
                "used_files": used_files,
                "axis_transform": axis_transform,
            },
        }

    return templates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unload_data_dir", required=True, help="Folder containing *_forceplate_3.csv")
    ap.add_argument(
        "--timing_xlsx",
        required=True,
        help="FP_platform_on-offset.xlsx (columns: velocity, trial, onset, offset)",
    )
    ap.add_argument("--timing_sheet", default=0, help="Excel sheet name or index (default 0)")
    ap.add_argument(
        "--file_pattern",
        default="*_perturb_{velocity}_{trial:03d}_forceplate_3.csv",
        help="glob pattern under unload_data_dir. Use {velocity} and {trial:03d}",
    )
    ap.add_argument("--frame_ratio", type=int, default=10, help="Device(1000Hz)//Mocap(100Hz) ratio")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")
    ap.add_argument(
        "--axis_transform",
        choices=["shared_files_default", "none"],
        default="shared_files_default",
        help="Apply the same axis transform used in shared_files Stage01 (default) or none.",
    )
    ap.add_argument(
        "--out_npz",
        default="src/replace_v3d/torque/assets/fp_inertial_templates.npz",
        help="Output .npz path (small; can be committed)",
    )
    args = ap.parse_args()

    if pl is None:
        raise RuntimeError("polars is required (AGENTS.md rule). Install polars in env 'module'.")

    unload_dir = Path(args.unload_data_dir)
    timing_xlsx = Path(args.timing_xlsx)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    axis_transform = None if args.axis_transform == "none" else DEFAULT_AXIS_TRANSFORM

    # Read timing excel
    timing_cols = {"velocity": "velocity", "trial": "trial", "onset": "onset", "offset": "offset"}

    # Prefer polars.read_excel if available; fallback to pandas.
    timing_df: "pl.DataFrame"
    try:
        timing_out = pl.read_excel(str(timing_xlsx), sheet_id=args.timing_sheet, engine="openpyxl")
        # Polars may return either a DataFrame or a dict[sheet_name, DataFrame].
        if isinstance(timing_out, dict):
            if not timing_out:
                raise RuntimeError(f"pl.read_excel returned empty dict for: {timing_xlsx}")
            # If a specific sheet name was requested, prefer it.
            if isinstance(args.timing_sheet, str) and args.timing_sheet in timing_out:
                timing_df = timing_out[str(args.timing_sheet)]
            else:
                timing_df = next(iter(timing_out.values()))
        else:
            timing_df = timing_out
    except Exception:
        df_pd = pd.read_excel(timing_xlsx, sheet_name=args.timing_sheet)
        timing_df = pl.from_pandas(df_pd)

    templates = _build_templates(
        unload_data_dir=unload_dir,
        timing_df=timing_df,
        timing_cols=timing_cols,
        file_pattern=str(args.file_pattern),
        frame_ratio=int(args.frame_ratio),
        agg=str(args.agg),
        axis_transform=axis_transform,
    )

    if not templates:
        raise RuntimeError("No templates built. Check timing file columns and unload_data_dir patterns.")

    velocity_ints = np.array(sorted(templates.keys()), dtype=np.int64)
    npz_dict: Dict[str, object] = {"velocity_ints": velocity_ints}

    for v_int in velocity_ints.tolist():
        t = templates[int(v_int)]
        prefix = f"v{int(v_int)}_"
        npz_dict[f"{prefix}unload_range_frames"] = np.array(int(t["unload_range_frames"]), dtype=np.int64)
        npz_dict[f"{prefix}n_trials"] = np.array(int(t["n_trials"]), dtype=np.int64)
        npz_dict[f"{prefix}fx"] = np.asarray(t["fx"], dtype=float)
        npz_dict[f"{prefix}fy"] = np.asarray(t["fy"], dtype=float)
        npz_dict[f"{prefix}fz"] = np.asarray(t["fz"], dtype=float)
        npz_dict[f"{prefix}mx"] = np.asarray(t["mx"], dtype=float)
        npz_dict[f"{prefix}my"] = np.asarray(t["my"], dtype=float)
        npz_dict[f"{prefix}mz"] = np.asarray(t["mz"], dtype=float)
        npz_dict[f"{prefix}meta_json"] = np.array(json.dumps(t["meta"], ensure_ascii=False), dtype=object)

    np.savez_compressed(out_npz, **npz_dict)
    print(f"[OK] Saved templates: {out_npz}")
    print(f"      velocities: {velocity_ints.tolist()}")


if __name__ == "__main__":
    main()
