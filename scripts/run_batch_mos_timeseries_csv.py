from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

import _bootstrap

_bootstrap.ensure_src_on_path()
_REPO_ROOT = _bootstrap.REPO_ROOT

from replace_v3d.c3d_reader import read_c3d_points
from replace_v3d.cli.batch_utils import append_rows_to_csv, build_trial_key, iter_c3d_files
from replace_v3d.com import COMModelParams, compute_whole_body_com, compute_xcom, derivative
from replace_v3d.events import (
    load_subject_leg_length_cm,
    load_trial_events,
    parse_subject_velocity_trial_from_filename,
    resolve_subject_from_token,
)
from replace_v3d.mos import compute_mos_timeseries


def _make_timeseries_dataframe(
    *,
    c3d_file: Path,
    subject_token: str,
    subject: str,
    velocity: float,
    trial: int,
    leg_length_cm: float,
    rate_hz: float,
    end_frame: int,
    platform_onset_local: int,
    platform_offset_local: int,
    step_onset_local: int | None,
    COM: np.ndarray,
    vCOM: np.ndarray,
    xCOM: np.ndarray,
    mos: Any,
) -> pd.DataFrame:
    mocap_frames = np.arange(1, end_frame + 1, dtype=int)
    time_s = (mocap_frames - 1) / rate_hz
    subject_key = build_trial_key(subject, velocity, trial)

    is_platform_onset = mocap_frames == int(platform_onset_local)
    if step_onset_local is None:
        is_step_onset = np.zeros_like(is_platform_onset, dtype=bool)
    else:
        is_step_onset = mocap_frames == int(step_onset_local)

    frame_count = len(mocap_frames)
    payload = {
        "subject-velocity-trial": [subject_key] * frame_count,
        "subject": [subject] * frame_count,
        "velocity": [float(velocity)] * frame_count,
        "trial": [int(trial)] * frame_count,
        "c3d_file": [c3d_file.name] * frame_count,
        "subject_token": [subject_token] * frame_count,
        "rate_hz": [float(rate_hz)] * frame_count,
        "leg_length_cm": [float(leg_length_cm)] * frame_count,
        "platform_onset_local": [int(platform_onset_local)] * frame_count,
        "platform_offset_local": [int(platform_offset_local)] * frame_count,
        "step_onset_local": [None if step_onset_local is None else int(step_onset_local)] * frame_count,
        "analysis_end_local": [int(end_frame)] * frame_count,
        "MocapFrame": mocap_frames,
        "Time_s": time_s,
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
        "MOS_AP_v3d": mos.MOS_AP_v3d,
        "MOS_ML_v3d": mos.MOS_ML_v3d,
        "MOS_v3d": mos.MOS_v3d,
        "MOS_AP_dir": mos.MOS_AP_dir,
        "MOS_ML_dir": mos.MOS_ML_dir,
        "Is_platform_onset_frame": is_platform_onset,
        "Is_step_onset_frame": is_step_onset,
    }
    return pl.DataFrame(payload).to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch export COM/vCOM/xCOM/BOS/MOS time series from C3D files into one long-format CSV."
        )
    )
    parser.add_argument(
        "--c3d_dir",
        default=str(_REPO_ROOT / "data" / "all_data"),
        help="Directory that contains .c3d files (default: data/all_data)",
    )
    parser.add_argument(
        "--event_xlsm",
        default=str(_REPO_ROOT / "data" / "perturb_inform.xlsm"),
        help="Path to perturb_inform.xlsm",
    )
    parser.add_argument(
        "--out_csv",
        default=str(_REPO_ROOT / "output" / "all_trials_mos_timeseries.csv"),
        help="Output CSV path (long format)",
    )
    parser.add_argument(
        "--pre_frames",
        type=int,
        default=100,
        help="Trim assumption for local event conversion (default: 100)",
    )
    parser.add_argument(
        "--skip_unmatched",
        action="store_true",
        help="Skip subject/event matching failures and continue batch processing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV text encoding (default: utf-8-sig; recommended for Korean text in Excel).",
    )
    args = parser.parse_args()

    c3d_dir = Path(args.c3d_dir)
    event_xlsm = Path(args.event_xlsm)
    out_csv = Path(args.out_csv)
    pre_frames = int(args.pre_frames)

    if not c3d_dir.exists():
        raise FileNotFoundError(f"C3D directory not found: {c3d_dir}")
    if not event_xlsm.exists():
        raise FileNotFoundError(f"Event workbook not found: {event_xlsm}")

    if out_csv.exists():
        if args.overwrite:
            out_csv.unlink()
        else:
            raise FileExistsError(f"Output already exists: {out_csv}. Use --overwrite to replace it.")

    c3d_files = iter_c3d_files(c3d_dir)
    if not c3d_files:
        raise FileNotFoundError(f"No .c3d files found under {c3d_dir}")

    header_written = False
    processed = 0
    skipped = 0

    for c3d_file in c3d_files:
        try:
            subject_token, velocity, trial = parse_subject_velocity_trial_from_filename(c3d_file.name)
            subject = resolve_subject_from_token(event_xlsm, subject_token)
            leg_length_cm = load_subject_leg_length_cm(event_xlsm, subject)
            if leg_length_cm is None:
                raise ValueError(f"Leg length not found for subject='{subject}'.")

            events = load_trial_events(
                event_xlsm=event_xlsm,
                subject=subject,
                velocity=velocity,
                trial=trial,
                pre_frames=pre_frames,
                sheet_name="platform",
            )

            c3d = read_c3d_points(c3d_file)
            rate_hz = float(c3d.rate_hz)
            dt = 1.0 / rate_hz
            total_frames = int(c3d.points.shape[0])

            if events.step_onset_local is not None:
                end_frame = int(events.step_onset_local) - 1
            else:
                end_frame = total_frames
            end_frame = max(1, min(end_frame, total_frames))

            params = COMModelParams()
            COM = compute_whole_body_com(c3d.points, c3d.labels, params=params)
            vCOM = derivative(COM, dt=dt)
            xCOM = compute_xcom(COM, vCOM, leg_length_m=float(leg_length_cm) / 100.0, g=9.81)

            mos = compute_mos_timeseries(
                points=c3d.points,
                labels=c3d.labels,
                xcom=xCOM,
                vcom=vCOM,
                end_frame=end_frame,
            )

            df_ts = _make_timeseries_dataframe(
                c3d_file=c3d_file,
                subject_token=subject_token,
                subject=subject,
                velocity=velocity,
                trial=trial,
                leg_length_cm=float(leg_length_cm),
                rate_hz=rate_hz,
                end_frame=end_frame,
                platform_onset_local=int(events.platform_onset_local),
                platform_offset_local=int(events.platform_offset_local),
                step_onset_local=None if events.step_onset_local is None else int(events.step_onset_local),
                COM=COM,
                vCOM=vCOM,
                xCOM=xCOM,
                mos=mos,
            )

            header_written = append_rows_to_csv(
                out_csv,
                df_ts,
                header_written=header_written,
                encoding=str(args.encoding),
            )
            processed += 1
        except Exception as exc:
            if args.skip_unmatched:
                skipped += 1
                print(f"[SKIP] {c3d_file.name}: {exc}")
                continue
            raise RuntimeError(f"Failed on file '{c3d_file}': {exc}") from exc

    print(f"[OK] Saved: {out_csv}")
    print(f"Processed files: {processed}")
    print(f"Skipped files: {skipped}")


if __name__ == "__main__":
    main()
