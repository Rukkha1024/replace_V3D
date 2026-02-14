from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running without installing the package
sys.path.insert(0, str((_Path(__file__).resolve().parent).resolve()))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from replace_v3d.c3d_reader import read_c3d_points
from replace_v3d.com import COMModelParams, compute_whole_body_com, derivative, compute_xcom
from replace_v3d.events import load_trial_events, parse_trial_from_filename
from replace_v3d.mos import compute_mos_timeseries
from replace_v3d.angles import compute_lower_limb_angles


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Correlation inputs must have same length.")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", required=True, help="Input C3D file")
    ap.add_argument("--event_xlsm", required=True, help="perturb_inform.xlsm")
    ap.add_argument("--subject", required=True, help="Subject name used in event sheet (e.g., 김우연)")
    ap.add_argument("--leg_length_cm", type=float, required=True, help="Leg length (cm) for xCOM ω0")
    ap.add_argument("--velocity", type=float, default=None, help="Velocity (if not parsed from filename)")
    ap.add_argument("--trial", type=int, default=None, help="Trial (if not parsed from filename)")
    ap.add_argument("--v3d_com_xlsx", default=None, help="Optional Visual3D COM xlsx for validation")
    ap.add_argument("--out_dir", default="output", help="Output directory")
    args = ap.parse_args()

    c3d_path = Path(args.c3d)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse velocity/trial from filename unless provided
    if args.velocity is None or args.trial is None:
        vel, tr = parse_trial_from_filename(c3d_path.name)
        velocity = vel if args.velocity is None else float(args.velocity)
        trial = tr if args.trial is None else int(args.trial)
    else:
        velocity = float(args.velocity)
        trial = int(args.trial)

    c3d = read_c3d_points(c3d_path)
    rate = c3d.rate_hz
    dt = 1.0 / rate

    events = load_trial_events(
        event_xlsm=args.event_xlsm,
        subject=args.subject,
        velocity=velocity,
        trial=trial,
        pre_frames=100,
        sheet_name="platform",
    )

    if events.step_onset_local is not None:
        end_frame = events.step_onset_local - 1  # IMPORTANT: analyze only until just before step onset
    else:
        end_frame = c3d.points.shape[0]

    # COM / xCOM
    params = COMModelParams()
    COM = compute_whole_body_com(c3d.points, c3d.labels, params=params)
    vCOM = derivative(COM, dt=dt)
    leg_length_m = float(args.leg_length_cm) / 100.0
    xCOM = compute_xcom(COM, vCOM, leg_length_m=leg_length_m, g=9.81)

    # MOS (BoS polygon + min distance)
    mos = compute_mos_timeseries(
        points=c3d.points,
        labels=c3d.labels,
        xcom=xCOM,
        vcom=vCOM,
        end_frame=end_frame,
    )

    # Angles (optional)
    jc = None
    try:
        from replace_v3d.com import compute_joint_centers
        jc = compute_joint_centers(c3d.points, c3d.labels)
        angles = compute_lower_limb_angles(c3d.points, c3d.labels, jc, end_frame=end_frame)
    except Exception:
        angles = None

    frames = np.arange(1, end_frame + 1)
    times = (frames - 1) / rate

    is_platform_onset = frames == events.platform_onset_local
    if events.step_onset_local is not None:
        is_step_onset = frames == events.step_onset_local
    else:
        is_step_onset = np.zeros_like(is_platform_onset, dtype=bool)

    df_pl = pl.DataFrame(
        {
            "Frame": frames,
            "Time_s": times,
            "COM_X": COM[:end_frame, 0],
            "COM_Y": COM[:end_frame, 1],
            "COM_Z": COM[:end_frame, 2],
            "vCOM_X": vCOM[:end_frame, 0],
            "vCOM_Y": vCOM[:end_frame, 1],
            "vCOM_Z": vCOM[:end_frame, 2],
            "xCOM_X": xCOM[:end_frame, 0],
            "xCOM_Y": xCOM[:end_frame, 1],
            "xCOM_Z": xCOM[:end_frame, 2],
            "BOS_area": mos.BOS_area,
            "BOS_minX": mos.BOS_minX,
            "BOS_maxX": mos.BOS_maxX,
            "BOS_minY": mos.BOS_minY,
            "BOS_maxY": mos.BOS_maxY,
            "MOS_minDist_signed": mos.MOS_signed,
            "MOS_AP_dir": mos.MOS_AP_dir,
            "MOS_ML_dir": mos.MOS_ML_dir,
            "Is_platform_onset_frame": is_platform_onset,
            "Is_step_onset_frame": is_step_onset,
        }
    )

    if angles is not None:
        df_pl = df_pl.with_columns(
            pl.Series("KneeFlex_L_deg", angles.knee_flex_L_deg),
            pl.Series("KneeFlex_R_deg", angles.knee_flex_R_deg),
            pl.Series("AnkleDorsi_L_deg", angles.ankle_dorsi_L_deg),
            pl.Series("AnkleDorsi_R_deg", angles.ankle_dorsi_R_deg),
        )

    # Summary metrics
    baseline_expr = pl.col("Frame") < int(events.platform_onset_local)
    pert_expr = pl.col("Frame") >= int(events.platform_onset_local)
    if events.step_onset_local is not None:
        pert_expr &= pl.col("Frame") < int(events.step_onset_local)

    baseline_df = df_pl.filter(baseline_expr)
    pert_df = df_pl.filter(pert_expr)

    summary = {
        "Trial": c3d_path.stem,
        "subject": events.subject,
        "velocity": events.velocity,
        "trial": events.trial,
        "platform_onset_local": events.platform_onset_local,
        "step_onset_local": events.step_onset_local,
        "analysis_end_local": end_frame,
        "baseline_MOS_mean": float(baseline_df.select(pl.col("MOS_minDist_signed").mean()).item()),
        "baseline_MOS_min": float(baseline_df.select(pl.col("MOS_minDist_signed").min()).item()),
        "perturb_MOS_min": float(pert_df.select(pl.col("MOS_minDist_signed").min()).item()),
    }

    if pert_df.height > 0:
        min_row = pert_df.sort("MOS_minDist_signed").head(1)
        summary["perturb_MOS_min_frame"] = int(min_row.select(pl.col("Frame")).item())

        onset_time = float(df_pl.filter(pl.col("Is_platform_onset_frame")).select(pl.col("Time_s")).item())
        min_time = float(min_row.select(pl.col("Time_s")).item())
        summary["perturb_MOS_min_time_from_onset_s"] = float(min_time - onset_time)

    # Optional V3D COM validation
    validation = None
    if args.v3d_com_xlsx:
        v3d_df = pd.read_excel(args.v3d_com_xlsx, skiprows=1)
        v3d = v3d_df[["X ", "Y ", "Z "]].to_numpy(dtype=float)

        corr_all = {
            "corr_all_X": _corr(COM[:, 0], v3d[:, 0]),
            "corr_all_Y": _corr(COM[:, 1], v3d[:, 1]),
            "corr_all_Z": _corr(COM[:, 2], v3d[:, 2]),
        }
        corr_pre = {
            "corr_preStep_X": _corr(COM[:end_frame, 0], v3d[:end_frame, 0]),
            "corr_preStep_Y": _corr(COM[:end_frame, 1], v3d[:end_frame, 1]),
            "corr_preStep_Z": _corr(COM[:end_frame, 2], v3d[:end_frame, 2]),
        }
        validation = {**corr_all, **corr_pre}

    # Write Excel
    df = df_pl.to_pandas()
    out_xlsx = out_dir / f"{c3d_path.stem}_MOS_preStep.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="timeseries_preStep", index=False)
        pl.DataFrame([summary]).to_pandas().to_excel(writer, sheet_name="summary", index=False)
        pl.DataFrame([vars(events)]).to_pandas().to_excel(writer, sheet_name="events", index=False)
        if validation is not None:
            pl.DataFrame([validation]).to_pandas().to_excel(writer, sheet_name="validation_COM", index=False)

    print(f"[OK] Saved: {out_xlsx}")
    if validation is not None:
        print("[COM validation] ", validation)


if __name__ == "__main__":
    main()
