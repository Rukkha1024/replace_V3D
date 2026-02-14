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

from c3d_reader import read_c3d_points
from events import load_trial_events, parse_trial_from_filename
from joint_angles.v3d_joint_angles import compute_v3d_joint_angles_3d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", required=True, help="Input C3D file")
    ap.add_argument("--event_xlsm", required=True, help="perturb_inform.xlsm")
    ap.add_argument("--subject", required=True, help="Subject name used in event sheet (e.g., 김우연)")
    ap.add_argument("--velocity", type=float, default=None, help="Velocity (if not parsed from filename)")
    ap.add_argument("--trial", type=int, default=None, help="Trial (if not parsed from filename)")
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
    rate = float(c3d.rate_hz)

    events = load_trial_events(
        event_xlsm=args.event_xlsm,
        subject=args.subject,
        velocity=velocity,
        trial=trial,
        pre_frames=100,
        sheet_name="platform",
    )

    if events.step_onset_local is not None:
        end_frame = int(events.step_onset_local) - 1  # analyze only until just before step onset
    else:
        end_frame = int(c3d.points.shape[0])

    angles = compute_v3d_joint_angles_3d(c3d.points, c3d.labels, end_frame=end_frame)

    frames = np.arange(1, end_frame + 1)
    times = (frames - 1) / rate

    is_platform_onset = frames == int(events.platform_onset_local)
    if events.step_onset_local is not None:
        is_step_onset = frames == int(events.step_onset_local)
    else:
        is_step_onset = np.zeros_like(is_platform_onset, dtype=bool)

    df_pl = pl.DataFrame(
        {
            "Frame": frames,
            "Time_s": times,
            "Is_platform_onset_frame": is_platform_onset,
            "Is_step_onset_frame": is_step_onset,
            # Hip
            "Hip_L_X_deg": angles.hip_L_X,
            "Hip_L_Y_deg": angles.hip_L_Y,
            "Hip_L_Z_deg": angles.hip_L_Z,
            "Hip_R_X_deg": angles.hip_R_X,
            "Hip_R_Y_deg": angles.hip_R_Y,
            "Hip_R_Z_deg": angles.hip_R_Z,
            # Knee
            "Knee_L_X_deg": angles.knee_L_X,
            "Knee_L_Y_deg": angles.knee_L_Y,
            "Knee_L_Z_deg": angles.knee_L_Z,
            "Knee_R_X_deg": angles.knee_R_X,
            "Knee_R_Y_deg": angles.knee_R_Y,
            "Knee_R_Z_deg": angles.knee_R_Z,
            # Ankle
            "Ankle_L_X_deg": angles.ankle_L_X,
            "Ankle_L_Y_deg": angles.ankle_L_Y,
            "Ankle_L_Z_deg": angles.ankle_L_Z,
            "Ankle_R_X_deg": angles.ankle_R_X,
            "Ankle_R_Y_deg": angles.ankle_R_Y,
            "Ankle_R_Z_deg": angles.ankle_R_Z,
            # Trunk / Neck (global, no side)
            "Trunk_X_deg": angles.trunk_X,
            "Trunk_Y_deg": angles.trunk_Y,
            "Trunk_Z_deg": angles.trunk_Z,
            "Neck_X_deg": angles.neck_X,
            "Neck_Y_deg": angles.neck_Y,
            "Neck_Z_deg": angles.neck_Z,
        }
    )

    # Outputs
    out_xlsx = out_dir / f"{c3d_path.stem}_JOINT_ANGLES_preStep.xlsx"
    out_csv = out_dir / f"{c3d_path.stem}_JOINT_ANGLES_preStep.csv"

    # CSV first (stable for MD5)
    df_pl.write_csv(out_csv)

    # Excel (for inspection)
    df = df_pl.to_pandas()
    meta = {
        "Trial": c3d_path.stem,
        "subject": events.subject,
        "velocity": events.velocity,
        "trial": events.trial,
        "rate_hz": rate,
        "platform_onset_local": events.platform_onset_local,
        "step_onset_local": events.step_onset_local,
        "analysis_end_local": end_frame,
        "angle_sequence": "Intrinsic XYZ (Visual3D-like: reference X, floating Y, non-reference Z)",
        "segment_axes": "X=+Right, Y=+Anterior, Z=+Up/Proximal",
        "knee_medial_marker": "LShin_3 / RShin_3",
        "ankle_medial_marker": "LFoot_3 / RFoot_3",
        "hip_center_method": "Harrington (via com.compute_joint_centers)",
    }

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="timeseries_preStep", index=False)
        pl.DataFrame([meta]).to_pandas().to_excel(writer, sheet_name="meta", index=False)
        pl.DataFrame([vars(events)]).to_pandas().to_excel(writer, sheet_name="events", index=False)

    print(f"[OK] Saved: {out_csv}")
    print(f"[OK] Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
