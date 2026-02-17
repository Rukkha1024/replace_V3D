"""Single-trial joint angle pipeline.

Computes 3D hip, knee, and ankle joint angles from C3D point data using the V3D method.
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap

_bootstrap.ensure_src_on_path()

import argparse

import numpy as np
import polars as pl

from replace_v3d.io.c3d_reader import read_c3d_points
from replace_v3d.cli.trial_resolve import resolve_velocity_trial
from replace_v3d.io.events_excel import load_trial_events, parse_trial_from_filename
from replace_v3d.joint_angles.v3d_joint_angles import compute_v3d_joint_angles_3d
from replace_v3d.joint_angles.postprocess import postprocess_joint_angles


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

    velocity, trial = resolve_velocity_trial(
        c3d_name=c3d_path.name,
        velocity_arg=args.velocity,
        trial_arg=args.trial,
        parse_fn=parse_trial_from_filename,
    )

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

    df_raw = pl.DataFrame(
        {
            "Frame": frames,
            "Time_s": times,
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

    # Standard output = ana0 (analysis-friendly):
    # - unify L/R sign meaning (LEFT Hip/Knee/Ankle Y/Z negated)
    # - subtract quiet-standing baseline mean (frames 1..11, inclusive)
    df_ana0, _meta_pp = postprocess_joint_angles(
        df_raw,
        frame_col="Frame",
        unify_lr_sign=True,
        baseline_frames=(1, 11),
    )

    out_csv = out_dir / f"{c3d_path.stem}_JOINT_ANGLES_preStep.csv"
    df_ana0.write_csv(out_csv)
    print(f"[OK] Saved: {out_csv}")


if __name__ == "__main__":
    main()
