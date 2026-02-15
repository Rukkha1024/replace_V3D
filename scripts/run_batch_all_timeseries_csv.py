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

from replace_v3d.angles import compute_lower_limb_angles
from replace_v3d.c3d_reader import read_c3d_points
from replace_v3d.cli.batch_utils import append_rows_to_csv, iter_c3d_files
from replace_v3d.com import (
    COMModelParams,
    compute_joint_centers,
    compute_whole_body_com,
    compute_xcom,
    derivative,
)
from replace_v3d.events import (
    load_subject_body_mass_kg,
    load_subject_leg_length_cm,
    load_trial_events,
    parse_subject_velocity_trial_from_filename,
    resolve_subject_from_token,
)
from replace_v3d.joint_angles.v3d_joint_angles import compute_v3d_joint_angles_3d
from replace_v3d.joint_angles.postprocess import postprocess_joint_angles
from replace_v3d.mos import compute_mos_timeseries
from replace_v3d.torque.ankle_torque import compute_ankle_torque_from_net_wrench
from replace_v3d.torque.cop import compute_cop_lab
from replace_v3d.torque.forceplate import (
    choose_active_force_platform,
    extract_platform_wrenches_lab,
    read_force_platforms,
)
from replace_v3d.torque.forceplate_inertial import (
    apply_forceplate_inertial_subtract,
    load_forceplate_inertial_templates,
)


def _make_timeseries_dataframe(
    *,
    subject: str,
    velocity: float,
    trial: int,
    end_frame: int,
    platform_onset_local: int,
    platform_offset_local: int,
    step_onset_local: int | None,
    COM: np.ndarray,
    vCOM: np.ndarray,
    xCOM: np.ndarray,
    mos: Any,
    angles: Any,
    lower_limb_angles: Any | None,
    angles_ana0: dict[str, np.ndarray] | None = None,
    angles_anat: dict[str, np.ndarray] | None = None,
    torque_payload: dict[str, np.ndarray],
) -> pd.DataFrame:
    mocap_frames = np.arange(1, end_frame + 1, dtype=int)

    is_platform_onset = mocap_frames == int(platform_onset_local)
    if step_onset_local is None:
        is_step_onset = np.zeros_like(is_platform_onset, dtype=bool)
    else:
        is_step_onset = mocap_frames == int(step_onset_local)

    frame_count = len(mocap_frames)
    if lower_limb_angles is None:
        knee_flex_L_deg = np.full(end_frame, np.nan, dtype=float)
        knee_flex_R_deg = np.full(end_frame, np.nan, dtype=float)
        ankle_dorsi_L_deg = np.full(end_frame, np.nan, dtype=float)
        ankle_dorsi_R_deg = np.full(end_frame, np.nan, dtype=float)
    else:
        knee_flex_L_deg = lower_limb_angles.knee_flex_L_deg
        knee_flex_R_deg = lower_limb_angles.knee_flex_R_deg
        ankle_dorsi_L_deg = lower_limb_angles.ankle_dorsi_L_deg
        ankle_dorsi_R_deg = lower_limb_angles.ankle_dorsi_R_deg

    payload: dict[str, Any] = {
        "subject": [subject] * frame_count,
        "velocity": [float(velocity)] * frame_count,
        "trial": [int(trial)] * frame_count,
        "platform_onset_local": [int(platform_onset_local)] * frame_count,
        "platform_offset_local": [int(platform_offset_local)] * frame_count,
        "step_onset_local": [None if step_onset_local is None else int(step_onset_local)] * frame_count,
        "analysis_end_local": [int(end_frame)] * frame_count,
        "MocapFrame": mocap_frames,
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
        "MOS_AP_velDir": mos.MOS_AP_velDir,
        "MOS_ML_velDir": mos.MOS_ML_velDir,
        "Is_platform_onset_frame": is_platform_onset,
        "Is_step_onset_frame": is_step_onset,
        # Joint angles (Visual3D-like)
        "Hip_L_X_deg": angles.hip_L_X,
        "Hip_L_Y_deg": angles.hip_L_Y,
        "Hip_L_Z_deg": angles.hip_L_Z,
        "Hip_R_X_deg": angles.hip_R_X,
        "Hip_R_Y_deg": angles.hip_R_Y,
        "Hip_R_Z_deg": angles.hip_R_Z,
        "Knee_L_X_deg": angles.knee_L_X,
        "Knee_L_Y_deg": angles.knee_L_Y,
        "Knee_L_Z_deg": angles.knee_L_Z,
        "Knee_R_X_deg": angles.knee_R_X,
        "Knee_R_Y_deg": angles.knee_R_Y,
        "Knee_R_Z_deg": angles.knee_R_Z,
        "Ankle_L_X_deg": angles.ankle_L_X,
        "Ankle_L_Y_deg": angles.ankle_L_Y,
        "Ankle_L_Z_deg": angles.ankle_L_Z,
        "Ankle_R_X_deg": angles.ankle_R_X,
        "Ankle_R_Y_deg": angles.ankle_R_Y,
        "Ankle_R_Z_deg": angles.ankle_R_Z,
        "Trunk_X_deg": angles.trunk_X,
        "Trunk_Y_deg": angles.trunk_Y,
        "Trunk_Z_deg": angles.trunk_Z,
        "Neck_X_deg": angles.neck_X,
        "Neck_Y_deg": angles.neck_Y,
        "Neck_Z_deg": angles.neck_Z,
        # Simple sagittal summary angles (match single-trial MOS workbook schema)
        "KneeFlex_L_deg": knee_flex_L_deg,
        "KneeFlex_R_deg": knee_flex_R_deg,
        "AnkleDorsi_L_deg": ankle_dorsi_L_deg,
        "AnkleDorsi_R_deg": ankle_dorsi_R_deg,
    }

    # Optional: analysis-friendly joint angles (sign-unified + baseline-subtracted).
    # These columns get suffix `_ana0` to avoid breaking existing schemas.
    if angles_ana0 is not None:
        for k, v in angles_ana0.items():
            payload[k] = v

    # Optional: anatomical-convention joint angles (sign-unified only).
    # These columns get suffix `_anat` to avoid breaking existing schemas.
    if angles_anat is not None:
        for k, v in angles_anat.items():
            payload[k] = v

    for key, values in torque_payload.items():
        payload[key] = values

    return pl.DataFrame(payload).to_pandas()


def _compute_ankle_torque_payload(
    *,
    c3d_file: Path,
    velocity: float,
    points: np.ndarray,
    labels: list[str],
    rate_hz: float,
    end_frame: int,
    platform_onset_local: int,
    platform_offset_local: int,
    force_plate_index_1based: int | None,
    body_mass_kg: float | None,
    fp_inertial_templates: dict[int, Any],
    fp_inertial_policy: str,
    fp_inertial_qc_fz_threshold_n: float,
    fp_inertial_qc_margin_m: float,
    fp_inertial_qc_strict: bool,
) -> tuple[int, dict[str, np.ndarray]]:
    fp_coll = read_force_platforms(c3d_file)
    analog_avg = fp_coll.analog.values
    n_frames = int(points.shape[0])
    if analog_avg.shape[0] != n_frames:
        raise ValueError(
            f"Analog frames ({analog_avg.shape[0]}) != point frames ({n_frames}). "
            "Check that C3D is trimmed consistently."
        )

    if force_plate_index_1based is not None:
        fp = next((p for p in fp_coll.platforms if p.index_1based == int(force_plate_index_1based)), None)
        if fp is None:
            raise ValueError(f"Requested force plate index not found: {force_plate_index_1based}")
    else:
        fp = choose_active_force_platform(analog_avg, fp_coll.platforms)

    analog_used = analog_avg
    onset0 = int(platform_onset_local) - 1
    offset0 = int(platform_offset_local) - 1
    analog_used, inertial_info = apply_forceplate_inertial_subtract(
        analog_avg,
        fp,
        velocity=float(velocity),
        onset0=int(onset0),
        offset0=int(offset0),
        templates=fp_inertial_templates,
        missing_policy=str(fp_inertial_policy),
        qc_fz_threshold_n=float(fp_inertial_qc_fz_threshold_n),
        qc_margin_m=float(fp_inertial_qc_margin_m),
    )
    if not inertial_info.get("applied"):
        raise ValueError(
            "Forceplate inertial subtract did not apply "
            f"for {c3d_file.name} (reason={inertial_info.get('reason')}, policy={inertial_info.get('missing_policy')})."
        )
    if inertial_info.get("qc_failed"):
        msg = (
            "[WARN] Forceplate inertial subtract QC failed "
            f"for {c3d_file.name} (COP in-bounds after={inertial_info.get('after_qc_cop_in_bounds_frac')}). "
            "Check axis transform / templates."
        )
        if fp_inertial_qc_strict:
            raise ValueError(msg)
        print(msg)

    F_lab, M_lab = extract_platform_wrenches_lab(analog_used, fp)
    idx = fp.channel_indices_0based.astype(int)
    F_plate = analog_used[:, idx[0:3]]
    M_plate = analog_used[:, idx[3:6]]
    COP_lab = compute_cop_lab(
        F_plate=F_plate,
        M_plate=M_plate,
        fp_origin_lab=fp.origin_lab,
        R_pl2lab=fp.R_pl2lab,
    )

    jc = compute_joint_centers(points, labels)
    ankle_L = jc["ankle_L"]
    ankle_R = jc["ankle_R"]

    res = compute_ankle_torque_from_net_wrench(
        F_lab=F_lab,
        M_lab_at_fp_origin=M_lab,
        COP_lab=COP_lab,
        fp_origin_lab=fp.origin_lab,
        ankle_L=ankle_L,
        ankle_R=ankle_R,
        body_mass_kg=body_mass_kg,
    )

    frames0 = np.arange(n_frames, dtype=int)
    onset0 = int(platform_onset_local) - 1
    time_from_onset = (frames0 - onset0) / float(rate_hz)

    end = int(end_frame)
    payload = {
        "time_from_platform_onset_s": time_from_onset[:end],
        "GRF_X_N": res.F_lab[:end, 0],
        "GRF_Y_N": res.F_lab[:end, 1],
        "GRF_Z_N": res.F_lab[:end, 2],
        "GRM_X_Nm_at_FPorigin": res.M_lab_at_fp_origin[:end, 0],
        "GRM_Y_Nm_at_FPorigin": res.M_lab_at_fp_origin[:end, 1],
        "GRM_Z_Nm_at_FPorigin": res.M_lab_at_fp_origin[:end, 2],
        "COP_X_m": res.COP_lab[:end, 0],
        "COP_Y_m": res.COP_lab[:end, 1],
        "COP_Z_m": res.COP_lab[:end, 2],
        "FP_origin_X_m": np.full(end, float(res.fp_origin_lab[0])),
        "FP_origin_Y_m": np.full(end, float(res.fp_origin_lab[1])),
        "FP_origin_Z_m": np.full(end, float(res.fp_origin_lab[2])),
        "L_ankleJC_X_m": res.ankle_L[:end, 0],
        "L_ankleJC_Y_m": res.ankle_L[:end, 1],
        "L_ankleJC_Z_m": res.ankle_L[:end, 2],
        "R_ankleJC_X_m": res.ankle_R[:end, 0],
        "R_ankleJC_Y_m": res.ankle_R[:end, 1],
        "R_ankleJC_Z_m": res.ankle_R[:end, 2],
        "AnkleMid_X_m": res.ankle_mid[:end, 0],
        "AnkleMid_Y_m": res.ankle_mid[:end, 1],
        "AnkleMid_Z_m": res.ankle_mid[:end, 2],
        "AnkleTorqueMid_ext_X_Nm": res.torque_mid_ext[:end, 0],
        "AnkleTorqueMid_ext_Y_Nm": res.torque_mid_ext[:end, 1],
        "AnkleTorqueMid_ext_Z_Nm": res.torque_mid_ext[:end, 2],
        "AnkleTorqueMid_int_X_Nm": res.torque_mid_int[:end, 0],
        "AnkleTorqueMid_int_Y_Nm": res.torque_mid_int[:end, 1],
        "AnkleTorqueMid_int_Z_Nm": res.torque_mid_int[:end, 2],
        "AnkleTorqueMid_int_Y_Nm_per_kg": (
            np.full(end, np.nan)
            if res.torque_mid_int_Y_Nm_per_kg is None
            else res.torque_mid_int_Y_Nm_per_kg[:end]
        ),
        "AnkleTorqueL_ext_X_Nm": res.torque_L_ext[:end, 0],
        "AnkleTorqueL_ext_Y_Nm": res.torque_L_ext[:end, 1],
        "AnkleTorqueL_ext_Z_Nm": res.torque_L_ext[:end, 2],
        "AnkleTorqueL_int_X_Nm": res.torque_L_int[:end, 0],
        "AnkleTorqueL_int_Y_Nm": res.torque_L_int[:end, 1],
        "AnkleTorqueL_int_Z_Nm": res.torque_L_int[:end, 2],
        "AnkleTorqueR_ext_X_Nm": res.torque_R_ext[:end, 0],
        "AnkleTorqueR_ext_Y_Nm": res.torque_R_ext[:end, 1],
        "AnkleTorqueR_ext_Z_Nm": res.torque_R_ext[:end, 2],
        "AnkleTorqueR_int_X_Nm": res.torque_R_int[:end, 0],
        "AnkleTorqueR_int_Y_Nm": res.torque_R_int[:end, 1],
        "AnkleTorqueR_int_Z_Nm": res.torque_R_int[:end, 2],
    }
    return int(fp.index_1based), payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch export unified time series (MOS/COM/xCOM/BOS + joint angles + ankle torque) from C3D files.\n"
            "\n"
            "Output is a long-format CSV: one row per subject-velocity-trial x MocapFrame.\n"
            "\n"
            "Notes:\n"
            "- Analysis is preStep: up to just before step onset (end_frame = step_onset_local - 1)\n"
            "- FORCE_PLATFORM/ANALOG is required (torque); missing forceplate aborts the run.\n"
        )
    )
    parser.add_argument("--c3d_dir", required=True, help="Directory containing C3D files (recursive).")
    parser.add_argument("--event_xlsm", required=True, help="Event workbook (perturb_inform.xlsm).")
    parser.add_argument(
        "--out_csv",
        default=str(_REPO_ROOT / "output" / "all_trials_timeseries.csv"),
        help="Output CSV path (default: output/all_trials_timeseries.csv).",
    )
    parser.add_argument(
        "--pre_frames",
        type=int,
        default=100,
        help="Assumed pre-frames used when trimming mocap around platform onset (default: 100).",
    )
    parser.add_argument(
        "--force_plate",
        type=int,
        default=None,
        help="Optional force plate index (1-based). If omitted, auto-select by |Fz|.",
    )
    parser.add_argument(
        "--fp_inertial_templates",
        default="src/replace_v3d/torque/assets/fp_inertial_templates.npz",
        help="NPZ created by scripts/torque_build_fp_inertial_templates.py (repo-relative OK)",
    )
    parser.add_argument(
        "--fp_inertial_policy",
        choices=["skip", "nearest", "interpolate"],
        default="skip",
        help="If template for this velocity is missing: skip | nearest | interpolate",
    )
    parser.add_argument(
        "--fp_inertial_qc_fz_threshold",
        type=float,
        default=20.0,
        help="QC threshold (N) for COP-in-bounds check",
    )
    parser.add_argument(
        "--fp_inertial_qc_margin_m",
        type=float,
        default=0.0,
        help="QC margin (m) added to plate bounds when checking COP",
    )
    parser.add_argument(
        "--fp_inertial_qc_strict",
        action="store_true",
        help="If QC fails after subtraction, raise instead of warning.",
    )
    parser.add_argument(
        "--skip_unmatched",
        action="store_true",
        help="Skip subject/event matching failures and continue batch processing (torque forceplate failures still abort).",
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
    parser.add_argument(
        "--angles_ana0",
        action="store_true",
        help=(
            "Also export analysis-friendly joint angles with suffix `_ana0`: "
            "LEFT Y/Z sign-unified + quiet-standing baseline removed (frames 1..11)."
        ),
    )
    parser.add_argument(
        "--angles_anat",
        action="store_true",
        help=(
            "Also export anatomical-convention joint angles with suffix `_anat`: "
            "flip LEFT Y/Z only (no baseline subtraction)."
        ),
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of C3D files (for quick checks).",
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
    if args.max_files is not None:
        c3d_files = c3d_files[: int(args.max_files)]
    if not c3d_files:
        raise FileNotFoundError(f"No .c3d files found under {c3d_dir}")

    header_written = False
    processed = 0
    skipped = 0

    tmpl_path = Path(args.fp_inertial_templates)
    if not tmpl_path.is_absolute():
        tmpl_path = _REPO_ROOT / tmpl_path
    if not tmpl_path.exists():
        raise FileNotFoundError(f"Inertial templates not found: {tmpl_path}")
    fp_inertial_templates = load_forceplate_inertial_templates(tmpl_path)

    for c3d_file in c3d_files:
        # Subject/token + events matching can be skipped. Torque (forceplate) must abort.
        try:
            subject_token, velocity, trial = parse_subject_velocity_trial_from_filename(c3d_file.name)
            subject = resolve_subject_from_token(event_xlsm, subject_token)
            leg_length_cm = load_subject_leg_length_cm(event_xlsm, subject)
            if leg_length_cm is None:
                raise ValueError(f"Leg length not found for subject='{subject}'.")
            body_mass_kg = load_subject_body_mass_kg(event_xlsm, subject)

            events = load_trial_events(
                event_xlsm=event_xlsm,
                subject=subject,
                velocity=velocity,
                trial=trial,
                pre_frames=pre_frames,
                sheet_name="platform",
            )
        except Exception as exc:
            message = f"[SKIP] {c3d_file.name}: {exc}"
            if args.skip_unmatched:
                skipped += 1
                print(message)
                continue
            raise RuntimeError(f"Failed on file '{c3d_file}': {exc}") from exc

        # Read points
        try:
            c3d = read_c3d_points(c3d_file)
        except Exception as exc:
            message = f"[SKIP] {c3d_file.name}: cannot read C3D points ({exc})"
            if args.skip_unmatched:
                skipped += 1
                print(message)
                continue
            raise RuntimeError(f"Failed on file '{c3d_file}': {exc}") from exc

        rate_hz = float(c3d.rate_hz)
        dt = 1.0 / rate_hz
        total_frames = int(c3d.points.shape[0])

        if events.step_onset_local is not None:
            end_frame = int(events.step_onset_local) - 1
        else:
            end_frame = total_frames
        end_frame = max(1, min(end_frame, total_frames))

        # Torque requires FORCE_PLATFORM/ANALOG; always abort on failure.
        try:
            force_plate_used, torque_payload = _compute_ankle_torque_payload(
                c3d_file=c3d_file,
                velocity=float(velocity),
                points=c3d.points,
                labels=c3d.labels,
                rate_hz=rate_hz,
                end_frame=end_frame,
                platform_onset_local=int(events.platform_onset_local),
                platform_offset_local=int(events.platform_offset_local),
                force_plate_index_1based=None if args.force_plate is None else int(args.force_plate),
                body_mass_kg=None if body_mass_kg is None else float(body_mass_kg),
                fp_inertial_templates=fp_inertial_templates,
                fp_inertial_policy=str(args.fp_inertial_policy),
                fp_inertial_qc_fz_threshold_n=float(args.fp_inertial_qc_fz_threshold),
                fp_inertial_qc_margin_m=float(args.fp_inertial_qc_margin_m),
                fp_inertial_qc_strict=bool(args.fp_inertial_qc_strict),
            )
        except Exception as exc:
            raise RuntimeError(f"Forceplate/torque extraction failed for '{c3d_file.name}': {exc}") from exc

        # Remaining computations can be skipped if requested.
        try:
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

            angles = compute_v3d_joint_angles_3d(c3d.points, c3d.labels, end_frame=end_frame)

            lower_limb_angles = None
            try:
                jc = compute_joint_centers(c3d.points, c3d.labels)
                lower_limb_angles = compute_lower_limb_angles(
                    c3d.points,
                    c3d.labels,
                    jc,
                    end_frame=end_frame,
                )
            except Exception:
                lower_limb_angles = None

            angles_ana0_payload: dict[str, np.ndarray] | None = None
            if args.angles_ana0:
                mocap_frames = np.arange(1, end_frame + 1, dtype=int)
                df_angles = pl.DataFrame(
                    {
                        "MocapFrame": mocap_frames,
                        "Hip_L_X_deg": angles.hip_L_X,
                        "Hip_L_Y_deg": angles.hip_L_Y,
                        "Hip_L_Z_deg": angles.hip_L_Z,
                        "Hip_R_X_deg": angles.hip_R_X,
                        "Hip_R_Y_deg": angles.hip_R_Y,
                        "Hip_R_Z_deg": angles.hip_R_Z,
                        "Knee_L_X_deg": angles.knee_L_X,
                        "Knee_L_Y_deg": angles.knee_L_Y,
                        "Knee_L_Z_deg": angles.knee_L_Z,
                        "Knee_R_X_deg": angles.knee_R_X,
                        "Knee_R_Y_deg": angles.knee_R_Y,
                        "Knee_R_Z_deg": angles.knee_R_Z,
                        "Ankle_L_X_deg": angles.ankle_L_X,
                        "Ankle_L_Y_deg": angles.ankle_L_Y,
                        "Ankle_L_Z_deg": angles.ankle_L_Z,
                        "Ankle_R_X_deg": angles.ankle_R_X,
                        "Ankle_R_Y_deg": angles.ankle_R_Y,
                        "Ankle_R_Z_deg": angles.ankle_R_Z,
                        "Trunk_X_deg": angles.trunk_X,
                        "Trunk_Y_deg": angles.trunk_Y,
                        "Trunk_Z_deg": angles.trunk_Z,
                        "Neck_X_deg": angles.neck_X,
                        "Neck_Y_deg": angles.neck_Y,
                        "Neck_Z_deg": angles.neck_Z,
                    }
                )

                df_pp, _meta_pp = postprocess_joint_angles(
                    df_angles,
                    frame_col="MocapFrame",
                    unify_lr_sign=True,
                    baseline_frames=(1, 11),
                )
                angle_cols = [c for c in df_pp.columns if c.endswith("_deg")]
                angles_ana0_payload = {f"{c}_ana0": df_pp[c].to_numpy() for c in angle_cols}

            angles_anat_payload: dict[str, np.ndarray] | None = None
            if args.angles_anat:
                mocap_frames = np.arange(1, end_frame + 1, dtype=int)
                df_angles = pl.DataFrame(
                    {
                        "MocapFrame": mocap_frames,
                        "Hip_L_X_deg": angles.hip_L_X,
                        "Hip_L_Y_deg": angles.hip_L_Y,
                        "Hip_L_Z_deg": angles.hip_L_Z,
                        "Hip_R_X_deg": angles.hip_R_X,
                        "Hip_R_Y_deg": angles.hip_R_Y,
                        "Hip_R_Z_deg": angles.hip_R_Z,
                        "Knee_L_X_deg": angles.knee_L_X,
                        "Knee_L_Y_deg": angles.knee_L_Y,
                        "Knee_L_Z_deg": angles.knee_L_Z,
                        "Knee_R_X_deg": angles.knee_R_X,
                        "Knee_R_Y_deg": angles.knee_R_Y,
                        "Knee_R_Z_deg": angles.knee_R_Z,
                        "Ankle_L_X_deg": angles.ankle_L_X,
                        "Ankle_L_Y_deg": angles.ankle_L_Y,
                        "Ankle_L_Z_deg": angles.ankle_L_Z,
                        "Ankle_R_X_deg": angles.ankle_R_X,
                        "Ankle_R_Y_deg": angles.ankle_R_Y,
                        "Ankle_R_Z_deg": angles.ankle_R_Z,
                        "Trunk_X_deg": angles.trunk_X,
                        "Trunk_Y_deg": angles.trunk_Y,
                        "Trunk_Z_deg": angles.trunk_Z,
                        "Neck_X_deg": angles.neck_X,
                        "Neck_Y_deg": angles.neck_Y,
                        "Neck_Z_deg": angles.neck_Z,
                    }
                )

                df_pp_anat, _meta_anat = postprocess_joint_angles(
                    df_angles,
                    frame_col="MocapFrame",
                    unify_lr_sign=True,
                    baseline_frames=None,
                )
                angle_cols = [c for c in df_pp_anat.columns if c.endswith("_deg")]
                angles_anat_payload = {f"{c}_anat": df_pp_anat[c].to_numpy() for c in angle_cols}

            df_ts = _make_timeseries_dataframe(
                subject=subject,
                velocity=velocity,
                trial=trial,
                end_frame=end_frame,
                platform_onset_local=int(events.platform_onset_local),
                platform_offset_local=int(events.platform_offset_local),
                step_onset_local=None if events.step_onset_local is None else int(events.step_onset_local),
                COM=COM,
                vCOM=vCOM,
                xCOM=xCOM,
                mos=mos,
                angles=angles,
                lower_limb_angles=lower_limb_angles,
                angles_ana0=angles_ana0_payload,
                angles_anat=angles_anat_payload,
                torque_payload=torque_payload,
            )

            header_written = append_rows_to_csv(
                out_csv,
                df_ts,
                header_written=header_written,
                encoding=str(args.encoding),
            )
            processed += 1
        except Exception as exc:
            message = f"[SKIP] {c3d_file.name}: {exc}"
            if args.skip_unmatched:
                skipped += 1
                print(message)
                continue
            raise RuntimeError(f"Failed on file '{c3d_file}': {exc}") from exc

    print(f"[OK] Saved: {out_csv}")
    print(f"Processed files: {processed}")
    print(f"Skipped files: {skipped}")


if __name__ == "__main__":
    main()
