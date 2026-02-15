from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import argparse

import numpy as np
import pandas as pd

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None

from replace_v3d.c3d_reader import read_c3d_points
from replace_v3d.com import compute_joint_centers
from replace_v3d.events import load_subject_body_mass_kg, load_trial_events, parse_trial_from_filename
from replace_v3d.torque.ankle_torque import compute_ankle_torque_from_net_wrench
from replace_v3d.torque.forceplate import (
    choose_active_force_platform,
    extract_platform_wrenches_lab,
    read_force_platforms,
)
from replace_v3d.torque.forceplate_inertial import (
    apply_forceplate_inertial_subtract,
    load_forceplate_inertial_templates,
)


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den2 = np.where(np.abs(den) < eps, np.nan, den)
    return num / den2


def _compute_cop_lab(
    *,
    F_plate: np.ndarray,
    M_plate: np.ndarray,
    fp_origin_lab: np.ndarray,
    R_pl2lab: np.ndarray,
) -> np.ndarray:
    """COP in plate coordinates then rotate/translate into lab."""

    Fz = F_plate[:, 2]
    cop_x = _safe_div(-M_plate[:, 1], Fz)
    cop_y = _safe_div(M_plate[:, 0], Fz)
    cop_plate = np.column_stack([cop_x, cop_y, np.zeros_like(cop_x)])
    return fp_origin_lab[None, :] + cop_plate @ R_pl2lab.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d", required=True, help="Input C3D file")
    ap.add_argument("--event_xlsm", required=True, help="perturb_inform.xlsm")
    ap.add_argument("--subject", required=True, help="Subject name used in event sheet (e.g., 김우연)")
    ap.add_argument("--velocity", type=float, default=None, help="Velocity (if not parsed from filename)")
    ap.add_argument("--trial", type=int, default=None, help="Trial (if not parsed from filename)")
    ap.add_argument(
        "--mass_kg",
        type=float,
        default=None,
        help="Optional body mass (kg). If omitted, reads from xlsm meta sheet.",
    )
    ap.add_argument(
        "--force_plate",
        type=int,
        default=None,
        help="Force plate index (1-based). Default: auto-select by |Fz|.",
    )

    # Forceplate inertial subtract (Stage01-compatible)
    ap.add_argument(
        "--fp_inertial_templates",
        default="assets/fp_inertial_templates.npz",
        help="NPZ created by scripts/torque_build_fp_inertial_templates.py (repo-relative OK)",
    )
    ap.add_argument(
        "--fp_inertial_policy",
        choices=["skip", "nearest", "interpolate"],
        default="skip",
        help="If template for this velocity is missing: skip | nearest | interpolate",
    )
    ap.add_argument(
        "--fp_inertial_qc_fz_threshold",
        type=float,
        default=20.0,
        help="QC threshold (N) for COP-in-bounds check",
    )
    ap.add_argument(
        "--fp_inertial_qc_margin_m",
        type=float,
        default=0.0,
        help="QC margin (m) added to plate bounds when checking COP",
    )
    ap.add_argument(
        "--fp_inertial_qc_strict",
        action="store_true",
        help="If QC fails after subtraction, raise instead of warning.",
    )
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

    # Read kinematics
    c3d = read_c3d_points(c3d_path)
    rate = float(c3d.rate_hz)
    n_frames = int(c3d.points.shape[0])

    # Events (platform onset, etc.)
    events = load_trial_events(
        event_xlsm=args.event_xlsm,
        subject=args.subject,
        velocity=velocity,
        trial=trial,
        pre_frames=100,
        sheet_name="platform",
    )

    # File-local onset/offset (0-based) used by inertial subtraction and time axis
    onset0 = int(events.platform_onset_local) - 1
    offset0 = int(events.platform_offset_local) - 1

    # Body mass
    mass_kg = float(args.mass_kg) if args.mass_kg is not None else load_subject_body_mass_kg(args.event_xlsm, args.subject)

    # Force plate
    fp_coll = read_force_platforms(c3d_path)
    analog_avg = fp_coll.analog.values
    if analog_avg.shape[0] != n_frames:
        raise ValueError(
            f"Analog frames ({analog_avg.shape[0]}) != point frames ({n_frames}). "
            "Check that C3D is trimmed consistently."
        )

    if args.force_plate is not None:
        fp = next((p for p in fp_coll.platforms if p.index_1based == int(args.force_plate)), None)
        if fp is None:
            raise ValueError(f"Requested force plate index not found: {args.force_plate}")
    else:
        fp = choose_active_force_platform(analog_avg, fp_coll.platforms)

    # Apply Stage01-style inertial subtraction (template) before wrench/COP.
    # Note: C3D forceplate axes are already transformed in the provided dataset.
    inertial_info = {
        "enabled": True,
        "applied": False,
        "reason": "not_run",
    }
    analog_used = analog_avg
    tmpl_path = Path(args.fp_inertial_templates)
    if not tmpl_path.is_absolute():
        tmpl_path = _REPO_ROOT / tmpl_path
    if not tmpl_path.exists():
        raise FileNotFoundError(f"Inertial templates not found: {tmpl_path}")

    templates = load_forceplate_inertial_templates(tmpl_path)
    analog_used, inertial_info = apply_forceplate_inertial_subtract(
        analog_avg,
        fp,
        velocity=float(velocity),
        onset0=int(onset0),
        offset0=int(offset0),
        templates=templates,
        missing_policy=str(args.fp_inertial_policy),
        qc_fz_threshold_n=float(args.fp_inertial_qc_fz_threshold),
        qc_margin_m=float(args.fp_inertial_qc_margin_m),
    )
    inertial_info["enabled"] = True
    inertial_info["templates_path"] = str(tmpl_path)

    if not inertial_info.get("applied"):
        raise ValueError(
            "Forceplate inertial subtract did not apply "
            f"(reason={inertial_info.get('reason')}, policy={inertial_info.get('missing_policy')})."
        )
    if inertial_info.get("qc_failed"):
        msg = (
            "[WARN] Forceplate inertial subtract QC failed "
            f"(COP in-bounds after={inertial_info.get('after_qc_cop_in_bounds_frac')}). "
            "Check axis transform / template file."
        )
        if args.fp_inertial_qc_strict:
            raise ValueError(msg)
        print(msg)

    # Extract plate wrenches
    F_lab, M_lab = extract_platform_wrenches_lab(analog_used, fp)
    idx = fp.channel_indices_0based.astype(int)
    F_plate = analog_used[:, idx[0:3]]
    M_plate = analog_used[:, idx[3:6]]
    COP_lab = _compute_cop_lab(F_plate=F_plate, M_plate=M_plate, fp_origin_lab=fp.origin_lab, R_pl2lab=fp.R_pl2lab)

    # Joint centers (medial markers already baked into com.compute_joint_centers)
    jc = compute_joint_centers(c3d.points, c3d.labels)
    ankle_L = jc["ankle_L"]
    ankle_R = jc["ankle_R"]

    res = compute_ankle_torque_from_net_wrench(
        F_lab=F_lab,
        M_lab_at_fp_origin=M_lab,
        COP_lab=COP_lab,
        fp_origin_lab=fp.origin_lab,
        ankle_L=ankle_L,
        ankle_R=ankle_R,
        body_mass_kg=mass_kg,
    )

    # Time axis (0-based frames like the V3D export)
    frames0 = np.arange(n_frames, dtype=int)
    time_s = frames0 / rate
    time_from_onset = (frames0 - onset0) / rate

    df_dict = {
        "frame": frames0,
        "time_s": time_s,
        "time_from_platform_onset_s": time_from_onset,
        "GRF_X_N": res.F_lab[:, 0],
        "GRF_Y_N": res.F_lab[:, 1],
        "GRF_Z_N": res.F_lab[:, 2],
        "GRM_X_Nm_at_FPorigin": res.M_lab_at_fp_origin[:, 0],
        "GRM_Y_Nm_at_FPorigin": res.M_lab_at_fp_origin[:, 1],
        "GRM_Z_Nm_at_FPorigin": res.M_lab_at_fp_origin[:, 2],
        "COP_X_m": res.COP_lab[:, 0],
        "COP_Y_m": res.COP_lab[:, 1],
        "COP_Z_m": res.COP_lab[:, 2],
        "FP_origin_X_m": float(res.fp_origin_lab[0]),
        "FP_origin_Y_m": float(res.fp_origin_lab[1]),
        "FP_origin_Z_m": float(res.fp_origin_lab[2]),
        "L_ankleJC_X_m": res.ankle_L[:, 0],
        "L_ankleJC_Y_m": res.ankle_L[:, 1],
        "L_ankleJC_Z_m": res.ankle_L[:, 2],
        "R_ankleJC_X_m": res.ankle_R[:, 0],
        "R_ankleJC_Y_m": res.ankle_R[:, 1],
        "R_ankleJC_Z_m": res.ankle_R[:, 2],
        "AnkleMid_X_m": res.ankle_mid[:, 0],
        "AnkleMid_Y_m": res.ankle_mid[:, 1],
        "AnkleMid_Z_m": res.ankle_mid[:, 2],
        "AnkleTorqueMid_ext_X_Nm": res.torque_mid_ext[:, 0],
        "AnkleTorqueMid_ext_Y_Nm": res.torque_mid_ext[:, 1],
        "AnkleTorqueMid_ext_Z_Nm": res.torque_mid_ext[:, 2],
        "AnkleTorqueMid_int_X_Nm": res.torque_mid_int[:, 0],
        "AnkleTorqueMid_int_Y_Nm": res.torque_mid_int[:, 1],
        "AnkleTorqueMid_int_Z_Nm": res.torque_mid_int[:, 2],
        "AnkleTorqueMid_int_Y_Nm_per_kg": (
            np.full(n_frames, np.nan)
            if res.torque_mid_int_Y_Nm_per_kg is None
            else res.torque_mid_int_Y_Nm_per_kg
        ),
        "AnkleTorqueL_ext_X_Nm": res.torque_L_ext[:, 0],
        "AnkleTorqueL_ext_Y_Nm": res.torque_L_ext[:, 1],
        "AnkleTorqueL_ext_Z_Nm": res.torque_L_ext[:, 2],
        "AnkleTorqueL_int_X_Nm": res.torque_L_int[:, 0],
        "AnkleTorqueL_int_Y_Nm": res.torque_L_int[:, 1],
        "AnkleTorqueL_int_Z_Nm": res.torque_L_int[:, 2],
        "AnkleTorqueR_ext_X_Nm": res.torque_R_ext[:, 0],
        "AnkleTorqueR_ext_Y_Nm": res.torque_R_ext[:, 1],
        "AnkleTorqueR_ext_Z_Nm": res.torque_R_ext[:, 2],
        "AnkleTorqueR_int_X_Nm": res.torque_R_int[:, 0],
        "AnkleTorqueR_int_Y_Nm": res.torque_R_int[:, 1],
        "AnkleTorqueR_int_Z_Nm": res.torque_R_int[:, 2],
    }

    if pl is not None:
        df_ts_pd = pl.DataFrame(df_dict).to_pandas()
    else:
        df_ts_pd = pd.DataFrame(df_dict)

    # Meta sheet (key/value)
    # filename tokens: {date}_{initial}_perturb_{velocity}_{trial}.c3d
    stem = c3d_path.stem
    parts = stem.split("_")
    name_initial = parts[1] if len(parts) >= 2 else ""
    trial_str = parts[4] if len(parts) >= 5 else str(trial)
    trim_end_raw = int(events.platform_offset_original + 100)
    meta_rows = [
        ("c3d_file", c3d_path.name),
        ("subject", events.subject),
        ("name_initial", name_initial),
        ("velocity", int(events.velocity) if float(events.velocity).is_integer() else events.velocity),
        ("trial", trial_str),
        ("point_rate_Hz", int(rate) if float(rate).is_integer() else rate),
        (
            "analog_rate_Hz",
            int(fp_coll.analog.rate_hz)
            if float(fp_coll.analog.rate_hz).is_integer()
            else fp_coll.analog.rate_hz,
        ),
        ("analog_samples_per_frame", fp_coll.analog.samples_per_frame),
        ("platform_onset_rawframe", events.platform_onset_original),
        ("platform_offset_rawframe", events.platform_offset_original),
        ("step_onset_rawframe", events.step_onset_original),
        ("trim_start_rawframe", events.trim_start_original),
        ("trim_end_rawframe", trim_end_raw),
        ("platform_onset_frame_in_file", onset0),
        ("platform_offset_frame_in_file", int(events.platform_offset_local) - 1),
        ("step_onset_frame_in_file", None if events.step_onset_local is None else int(events.step_onset_local) - 1),
        ("body_mass_kg", mass_kg),
        ("active_force_plate_index_1based", fp.index_1based),
        ("force_plate_type", fp.fp_type),
        ("fp_inertial_templates_path", inertial_info.get("templates_path", args.fp_inertial_templates)),
        ("fp_inertial_subtract_enabled", bool(inertial_info.get("enabled", False))),
        ("fp_inertial_subtract_applied", bool(inertial_info.get("applied", False))),
        ("fp_inertial_subtract_reason", inertial_info.get("reason")),
        ("fp_inertial_template_policy", inertial_info.get("template_policy")),
        ("fp_inertial_template_velocity_int_used", inertial_info.get("template_velocity_int_used")),
        ("fp_inertial_template_velocity_int_lo", inertial_info.get("template_velocity_int_lo")),
        ("fp_inertial_template_velocity_int_hi", inertial_info.get("template_velocity_int_hi")),
        ("fp_inertial_template_interp_weight", inertial_info.get("template_interp_weight")),
        ("fp_inertial_template_len", inertial_info.get("template_len")),
        ("fp_inertial_unload_range_frames", inertial_info.get("unload_range_frames")),
        ("fp_inertial_template_n_trials", inertial_info.get("template_n_trials")),
        ("fp_inertial_qc_fz_threshold_n", inertial_info.get("qc_fz_threshold_n")),
        ("fp_inertial_qc_margin_m", inertial_info.get("qc_margin_m")),
        ("fp_inertial_qc_valid_n_before", inertial_info.get("before_qc_valid_n")),
        ("fp_inertial_qc_valid_n_after", inertial_info.get("after_qc_valid_n")),
        ("fp_inertial_qc_cop_in_bounds_frac_before", inertial_info.get("before_qc_cop_in_bounds_frac")),
        ("fp_inertial_qc_cop_in_bounds_frac_after", inertial_info.get("after_qc_cop_in_bounds_frac")),
        ("fp_inertial_qc_fz_positive_frac_before", inertial_info.get("before_qc_fz_positive_frac")),
        ("fp_inertial_qc_fz_positive_frac_after", inertial_info.get("after_qc_fz_positive_frac")),
        ("fp_inertial_qc_failed", inertial_info.get("qc_failed")),
    ]
    meta_df = pd.DataFrame(meta_rows, columns=["key", "value"])

    # Corners sheet
    corners_df = pd.DataFrame(
        {
            "corner": [1, 2, 3, 4],
            "X": fp.corners_lab[:, 0],
            "Y": fp.corners_lab[:, 1],
            "Z": fp.corners_lab[:, 2],
        }
    )

    out_xlsx = out_dir / f"{c3d_path.stem}_ankle_torque.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_ts_pd.to_excel(writer, sheet_name="ankle_torque", index=False)
        meta_df.to_excel(writer, sheet_name="meta", index=False)
        corners_df.to_excel(writer, sheet_name=f"force_plate{fp.index_1based}_corners", index=False)

    print(f"[OK] Saved: {out_xlsx}")


if __name__ == "__main__":
    main()
