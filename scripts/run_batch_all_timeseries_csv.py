"""Batch pipeline for all biomechanical variables.

Processes multiple C3D files to compute COM, xCOM, joint angles
(hip/knee/ankle/trunk/neck), ankle torque, and MOS, then exports
a single CSV (all_trials_timeseries.csv).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml

import _bootstrap

_bootstrap.ensure_src_on_path()
_REPO_ROOT = _bootstrap.REPO_ROOT

from replace_v3d.joint_angles.sagittal import compute_lower_limb_angles
from replace_v3d.io.c3d_reader import read_c3d_points
from replace_v3d.cli.batch_utils import append_rows_to_csv, build_trial_key, iter_c3d_files
from replace_v3d.com import (
    COMModelParams,
    compute_joint_centers,
    compute_whole_body_com,
    compute_xcom,
    derivative,
)
from replace_v3d.io.events_excel import (
    load_subject_body_mass_kg,
    load_subject_leg_length_cm,
    load_trial_events,
    parse_subject_velocity_trial_from_filename,
    resolve_subject_from_token,
)
from replace_v3d.joint_angles.v3d_joint_angles import compute_v3d_joint_angles_3d
from replace_v3d.joint_angles.postprocess import postprocess_joint_angles
from replace_v3d.mos import compute_mos_timeseries
from replace_v3d.signal.zeroing import subtract_baseline_at_index
from replace_v3d.torque.ankle_torque import compute_ankle_torque_from_net_wrench
from replace_v3d.torque.cop import compute_cop_stage01_xy
from replace_v3d.torque.forceplate import (
    apply_force_platform_corner_overrides,
    choose_active_force_platform,
    read_force_platforms,
)
from replace_v3d.torque.forceplate_inertial import (
    apply_forceplate_inertial_subtract,
    load_forceplate_inertial_templates,
)
from replace_v3d.torque.stage01_axis import transform_force_moment_to_stage01


def _load_meta_with_age_group(event_xlsm: Path) -> pl.DataFrame:
    meta_wide = pl.read_excel(str(event_xlsm), sheet_name="meta")
    if "subject" not in meta_wide.columns:
        raise ValueError("meta sheet is missing required column: subject")

    items = [name if name is not None else f"unknown_{i}" for i, name in enumerate(meta_wide["subject"].to_list())]
    subject_cols = [c for c in meta_wide.columns if c != "subject"]
    if not subject_cols:
        raise ValueError("meta sheet has no subject columns.")

    transposed = meta_wide.select(subject_cols).transpose(include_header=False)
    transposed.columns = items
    if "나이" not in transposed.columns or "주손 or 주발" not in transposed.columns:
        raise ValueError("meta transpose is missing one of required columns: 나이, 주손 or 주발")

    return (
        transposed.with_columns(pl.Series("subject", subject_cols))
        .with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("나이").cast(pl.Int32, strict=False).alias("나이"),
            pl.col("주손 or 주발").cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase().alias("주손 or 주발"),
        )
        .with_columns(
            pl.when(pl.col("나이") < 30)
            .then(pl.lit("young"))
            .otherwise(pl.lit("old"))
            .alias("age_group")
        )
        .select(["subject", "age_group", "주손 or 주발"])
    )


def _load_platform_trial_meta(event_xlsm: Path) -> pl.DataFrame:
    platform = pl.read_excel(str(event_xlsm), sheet_name="platform")
    required = {"subject", "velocity", "trial", "step_TF", "state", "mixed"}
    missing = [c for c in required if c not in platform.columns]
    if missing:
        raise ValueError(f"platform sheet missing required columns: {missing}")

    return (
        platform.select(["subject", "velocity", "trial", "step_TF", "state", "mixed"])
        .with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial").cast(pl.Int64, strict=False),
            pl.col("step_TF").cast(pl.Utf8, strict=False).str.strip_chars(),
            pl.col("state").cast(pl.Utf8, strict=False).str.strip_chars(),
            pl.col("mixed").cast(pl.Float64, strict=False),
        )
    )


def _build_meta_prefilter_trials(event_xlsm: Path) -> pl.DataFrame:
    platform = _load_platform_trial_meta(event_xlsm)
    meta = _load_meta_with_age_group(event_xlsm)
    merged = platform.join(meta, on="subject", how="left")

    mask_base = (pl.col("mixed") == 1) & (pl.col("age_group") == "young")
    is_nonstep = pl.col("step_TF") == "nonstep"
    is_step = pl.col("step_TF") == "step"
    return (
        # NOTE:
        # Previously, step trials were limited to ipsilateral stepping (dominant side),
        # which can drop valid mixed-velocity step trials for some subjects and break
        # within-subject step vs nonstep comparisons. For "주제 2" analyses, include
        # all mixed==1 young step/nonstep trials at the selected mixed velocity.
        merged.filter(mask_base & (is_step | is_nonstep))
        .select(["subject", "velocity", "trial", "age_group", "주손 or 주발", "step_TF", "state", "mixed"])
        .drop_nulls(["subject", "velocity", "trial"])
        .unique()
        .sort(["subject", "velocity", "trial"])
    )


def _parse_corner_triplet(raw: Any, *, fp_key: str, corner_key: str) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{fp_key}.{corner_key} must be a 3-item list [x,y,z].")
    try:
        x, y, z = float(raw[0]), float(raw[1]), float(raw[2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{fp_key}.{corner_key} must contain numeric values.") from exc
    return (x, y, z)


def _parse_axis_map_token(raw: Any, *, fp_key: str) -> tuple[float, int]:
    text = str(raw).strip().upper()
    sign = -1.0 if text.startswith("-") else 1.0
    axis = text[1:] if text.startswith("-") else text
    axis_to_idx = {"X": 0, "Y": 1, "Z": 2}
    if axis not in axis_to_idx:
        raise ValueError(
            f"{fp_key}.axis_map tokens must be one of: X, Y, Z, -X, -Y, -Z. Got: {raw!r}"
        )
    return (sign, axis_to_idx[axis])


def _apply_axis_map(corners: np.ndarray, axis_map: Any, *, fp_key: str) -> np.ndarray:
    """Apply an axis swap/sign flip mapping to a (N,3) corner array."""
    if not isinstance(axis_map, (list, tuple)) or len(axis_map) != 3:
        raise ValueError(
            f"{fp_key}.axis_map must be a 3-item list like ['X','Y','Z'] or ['-X','Z','Y']. Got: {axis_map!r}"
        )

    mapped = np.empty_like(corners, dtype=float)
    used: set[int] = set()
    for out_axis_idx, tok in enumerate(axis_map):
        sign, in_axis_idx = _parse_axis_map_token(tok, fp_key=fp_key)
        if in_axis_idx in used:
            raise ValueError(f"{fp_key}.axis_map must not repeat axes. Got: {axis_map!r}")
        used.add(in_axis_idx)
        mapped[:, out_axis_idx] = float(sign) * corners[:, in_axis_idx]

    return mapped


def _load_forceplate_corner_overrides(config_path: Path) -> dict[int, np.ndarray]:
    """Load forceplate corner overrides from config.yaml.

    Supports per-plate `units` conversion (mm -> m) so corner values can be
    specified in the same units as Visual3D exports while keeping the pipeline
    in meters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        return {}

    forceplate_cfg = raw.get("forceplate")
    if not isinstance(forceplate_cfg, dict):
        return {}

    coordination_cfg = forceplate_cfg.get("coordination")
    if not isinstance(coordination_cfg, dict):
        return {}

    overrides: dict[int, np.ndarray] = {}
    for fp_key, fp_cfg in coordination_cfg.items():
        if not isinstance(fp_cfg, dict):
            continue
        if not bool(fp_cfg.get("enabled", False)):
            continue

        key_text = str(fp_key).strip().lower()
        if not key_text.startswith("fp") or (not key_text[2:].isdigit()):
            raise ValueError(
                f"forceplate.coordination key must look like fp1/fp2/fp3. Got: {fp_key!r}"
            )
        fp_idx = int(key_text[2:])

        units_text = str(fp_cfg.get("units", "m")).strip().lower()
        if units_text in ("m", "meter", "meters"):
            unit_scale = 1.0
        elif units_text in ("mm", "millimeter", "millimeters"):
            unit_scale = 0.001
        else:
            raise ValueError(f"{fp_key}.units must be one of: m, mm. Got: {fp_cfg.get('units')!r}")

        corners_cfg = fp_cfg.get("corners")
        if not isinstance(corners_cfg, dict):
            raise ValueError(f"{fp_key}.corners must be a mapping with corner0..corner3")

        corners = np.asarray(
            [
                _parse_corner_triplet(corners_cfg.get("corner0"), fp_key=str(fp_key), corner_key="corner0"),
                _parse_corner_triplet(corners_cfg.get("corner1"), fp_key=str(fp_key), corner_key="corner1"),
                _parse_corner_triplet(corners_cfg.get("corner2"), fp_key=str(fp_key), corner_key="corner2"),
                _parse_corner_triplet(corners_cfg.get("corner3"), fp_key=str(fp_key), corner_key="corner3"),
            ],
            dtype=float,
        )
        corners = corners * float(unit_scale)
        if "axis_map" in fp_cfg:
            corners = _apply_axis_map(corners, fp_cfg.get("axis_map"), fp_key=str(fp_key))
        overrides[int(fp_idx)] = corners

    return overrides


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
    angles: Any | None,
    lower_limb_angles: Any | None,
    torque_payload: dict[str, np.ndarray],
) -> pd.DataFrame:
    mocap_frames = np.arange(1, end_frame + 1, dtype=int)
    onset_idx0 = int(platform_onset_local) - 1
    if onset_idx0 < 0 or onset_idx0 >= int(end_frame):
        raise ValueError(
            "platform_onset_local out of range for this trial export window: "
            f"platform_onset_local={platform_onset_local}, end_frame={end_frame}."
        )

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

    # Onset-zero (platform onset) for analysis-friendly comparisons.
    knee_flex_L_deg = subtract_baseline_at_index(knee_flex_L_deg, onset_idx0)
    knee_flex_R_deg = subtract_baseline_at_index(knee_flex_R_deg, onset_idx0)
    ankle_dorsi_L_deg = subtract_baseline_at_index(ankle_dorsi_L_deg, onset_idx0)
    ankle_dorsi_R_deg = subtract_baseline_at_index(ankle_dorsi_R_deg, onset_idx0)

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
        "Is_platform_onset_frame": is_platform_onset,
        "Is_step_onset_frame": is_step_onset,
        # Simple sagittal summary angles (match single-trial MOS workbook schema)
        "KneeFlex_L_deg": knee_flex_L_deg,
        "KneeFlex_R_deg": knee_flex_R_deg,
        "AnkleDorsi_L_deg": ankle_dorsi_L_deg,
        "AnkleDorsi_R_deg": ankle_dorsi_R_deg,
    }

    if angles is None:
        # Some C3D files may lack required trunk/neck markers (e.g., T10).
        # Keep the trial and export NaNs for joint angles rather than failing
        # the whole batch export.
        angle_cols = [
            "Hip_L_X_deg", "Hip_L_Y_deg", "Hip_L_Z_deg",
            "Hip_R_X_deg", "Hip_R_Y_deg", "Hip_R_Z_deg",
            "Knee_L_X_deg", "Knee_L_Y_deg", "Knee_L_Z_deg",
            "Knee_R_X_deg", "Knee_R_Y_deg", "Knee_R_Z_deg",
            "Ankle_L_X_deg", "Ankle_L_Y_deg", "Ankle_L_Z_deg",
            "Ankle_R_X_deg", "Ankle_R_Y_deg", "Ankle_R_Z_deg",
            "Trunk_X_deg", "Trunk_Y_deg", "Trunk_Z_deg",
            "Neck_X_deg", "Neck_Y_deg", "Neck_Z_deg",
        ]
        for c in angle_cols:
            payload[c] = np.full(end_frame, np.nan, dtype=float)
    else:
        # Joint angles (standard = ana0):
        # - unify L/R sign meaning (LEFT Hip/Knee/Ankle Y/Z negated)
        # - no quiet-standing baseline subtraction; outputs are onset-zeroed (platform onset)
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
            baseline_frames=None,
        )
        angle_cols = [c for c in df_pp.columns if c.endswith("_deg")]
        for c in angle_cols:
            payload[c] = subtract_baseline_at_index(df_pp[c].to_numpy(), onset_idx0)

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
    fp_corner_overrides: dict[int, np.ndarray] | None,
) -> tuple[int, dict[str, np.ndarray]]:
    fp_coll = read_force_platforms(c3d_file)
    fp_coll = apply_force_platform_corner_overrides(fp_coll, fp_corner_overrides)
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

    idx = fp.channel_indices_0based.astype(int)
    F_raw = analog_avg[:, idx[0:3]]
    M_raw = analog_avg[:, idx[3:6]]
    F_stage01_raw, M_stage01_raw = transform_force_moment_to_stage01(
        F_in=F_raw,
        M_in=M_raw,
    )

    analog_stage01 = np.asarray(analog_avg, dtype=float).copy()
    analog_stage01[:, idx[0:3]] = F_stage01_raw
    analog_stage01[:, idx[3:6]] = M_stage01_raw

    # Align channel sign to the repository Stage01 template convention before subtract.
    # Stage01 templates use the opposite sign of the direct C3D-mapped Stage01 raw.
    analog_shared_sign = analog_stage01.copy()
    analog_shared_sign[:, idx[0:3]] *= -1.0
    analog_shared_sign[:, idx[3:6]] *= -1.0

    analog_used = analog_shared_sign
    onset0 = int(platform_onset_local) - 1
    offset0 = int(platform_offset_local) - 1
    analog_used, inertial_info = apply_forceplate_inertial_subtract(
        analog_shared_sign,
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

    # `analog_used` is already aligned to the Stage01 template sign convention.
    F_stage01 = analog_used[:, idx[0:3]]
    M_stage01 = analog_used[:, idx[3:6]]
    COP_stage01_xy = compute_cop_stage01_xy(
        F_stage01=F_stage01,
        M_stage01=M_stage01,
    )

    jc = compute_joint_centers(points, labels)
    ankle_L = jc["ankle_L"]
    ankle_R = jc["ankle_R"]

    res = compute_ankle_torque_from_net_wrench(
        F_lab=F_stage01,
        M_lab_at_fp_origin=M_stage01,
        fp_origin_lab=fp.origin_lab,
        ankle_L=ankle_L,
        ankle_R=ankle_R,
        body_mass_kg=body_mass_kg,
    )

    frames0 = np.arange(n_frames, dtype=int)
    onset0 = int(platform_onset_local) - 1
    time_from_onset = (frames0 - onset0) / float(rate_hz)

    end = int(end_frame)
    if onset0 < 0 or onset0 >= end:
        raise ValueError(
            "platform_onset_local out of range for torque payload window: "
            f"platform_onset_local={platform_onset_local}, onset0={onset0}, end_frame={end_frame}."
        )
    payload = {
        "time_from_platform_onset_s": time_from_onset[:end],
        "GRF_X_N": res.F_lab[:end, 0],
        "GRF_Y_N": res.F_lab[:end, 1],
        "GRF_Z_N": res.F_lab[:end, 2],
        "GRM_X_Nm_at_FPorigin": res.M_lab_at_fp_origin[:end, 0],
        "GRM_Y_Nm_at_FPorigin": res.M_lab_at_fp_origin[:end, 1],
        "GRM_Z_Nm_at_FPorigin": res.M_lab_at_fp_origin[:end, 2],
        # NOTE:
        # COP is exported in absolute (lab) coordinates so it is directly
        # comparable to COM_X/COM_Y. Stage01 COP is origin-relative (Cx/Cy),
        # so we add the selected forceplate origin (lab) translation.
        "COP_X_m": COP_stage01_xy[:end, 0] + float(res.fp_origin_lab[0]),
        "COP_Y_m": COP_stage01_xy[:end, 1] + float(res.fp_origin_lab[1]),
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

    # Onset-zero force / moment / torque outputs (replace existing values).
    for key in list(payload.keys()):
        if key.startswith(("GRF_", "GRM_", "AnkleTorque")):
            payload[key] = subtract_baseline_at_index(payload[key], onset0)

    return int(fp.index_1based), payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch export unified time series (MOS/COM/xCOM/BOS + joint angles + ankle torque) from C3D files.\n"
            "\n"
            "Output is a long-format CSV: one row per subject-velocity-trial x MocapFrame.\n"
            "\n"
            "Notes:\n"
            "- Default exports the full trimmed C3D range (use --analysis_mode prestep for legacy preStep export).\n"
            "- FORCE_PLATFORM/ANALOG is required (torque); missing forceplate aborts the run.\n"
        )
    )
    parser.add_argument("--c3d_dir", required=True, help="Directory containing C3D files (recursive).")
    parser.add_argument("--event_xlsm", required=True, help="Event workbook (perturb_inform.xlsm).")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config.yaml"),
        help="Config YAML path (forceplate.coordination overrides).",
    )
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
        "--analysis_mode",
        choices=["full", "prestep"],
        default="full",
        help=(
            "Export range mode. "
            "'full' exports all frames in the trimmed C3D (default). "
            "'prestep' exports up to just before step onset (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists.",
    )
    parser.add_argument(
        "--backup_on_overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --overwrite and output exists, move it to a .bak_TIMESTAMP file first (default: enabled).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV text encoding (default: utf-8-sig; recommended for Korean text in Excel).",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on number of C3D files (for quick checks).",
    )
    parser.add_argument(
        "--meta_prefilter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply add_meta.ipynb-equivalent trial filter before per-file computation "
            "(mixed==1, age_group==young, ipsilateral-step/nonstep)."
        ),
    )
    args = parser.parse_args()

    c3d_dir = Path(args.c3d_dir)
    event_xlsm = Path(args.event_xlsm)
    config_path = Path(args.config)
    out_csv = Path(args.out_csv)
    pre_frames = int(args.pre_frames)

    if not c3d_dir.exists():
        raise FileNotFoundError(f"C3D directory not found: {c3d_dir}")
    if not event_xlsm.exists():
        raise FileNotFoundError(f"Event workbook not found: {event_xlsm}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if out_csv.exists():
        if args.overwrite:
            if args.backup_on_overwrite:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = out_csv.with_name(f"{out_csv.name}.bak_{timestamp}")
                out_csv.replace(backup_path)
                print(f"[INFO] Existing output backed up: {backup_path}")
            else:
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
    fp_corner_overrides = _load_forceplate_corner_overrides(config_path)
    if fp_corner_overrides:
        enabled = ", ".join([f"FP{idx}" for idx in sorted(fp_corner_overrides.keys())])
        print(f"[INFO] forceplate corner override enabled: {enabled}")

    allowed_trial_keys: set[str] | None = None
    trial_meta_lookup: dict[str, dict[str, Any]] | None = None
    if bool(args.meta_prefilter):
        prefilter_trials = _build_meta_prefilter_trials(event_xlsm)
        allowed_trial_keys = set()
        trial_meta_lookup = {}
        for row in prefilter_trials.iter_rows(named=True):
            trial_key = build_trial_key(
                str(row["subject"]).strip(),
                float(row["velocity"]),
                int(row["trial"]),
            )
            allowed_trial_keys.add(trial_key)
            trial_meta_lookup[trial_key] = {
                "age_group": row["age_group"],
                "주손 or 주발": row["주손 or 주발"],
                "step_TF": row["step_TF"],
                "state": row["state"],
                "mixed": row["mixed"],
            }
        print(
            "[INFO] meta_prefilter enabled: "
            f"subjects={prefilter_trials.select('subject').unique().height}, "
            f"trials={prefilter_trials.height}"
        )

    for c3d_file in c3d_files:
        # Subject/token + events matching can be skipped. Torque (forceplate) must abort.
        try:
            subject_token, velocity, trial = parse_subject_velocity_trial_from_filename(c3d_file.name)
            subject = resolve_subject_from_token(event_xlsm, subject_token)
        except Exception as exc:
            message = f"[SKIP] {c3d_file.name}: {exc}"
            if args.skip_unmatched:
                skipped += 1
                print(message)
                continue
            raise RuntimeError(f"Failed on file '{c3d_file}': {exc}") from exc

        trial_key = build_trial_key(str(subject).strip(), float(velocity), int(trial))
        if (allowed_trial_keys is not None) and (trial_key not in allowed_trial_keys):
            skipped += 1
            print(f"[SKIP][meta_prefilter] {c3d_file.name}: trial={trial_key} not selected by meta filter")
            continue
        trial_meta = None if trial_meta_lookup is None else trial_meta_lookup.get(trial_key)

        try:
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

        if str(args.analysis_mode) == "prestep" and events.step_onset_local is not None:
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
                fp_corner_overrides=fp_corner_overrides,
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

            angles = None
            try:
                angles = compute_v3d_joint_angles_3d(c3d.points, c3d.labels, end_frame=end_frame)
            except Exception as exc:
                # Keep exporting COM/COP/GRF/MoS for trials that lack required
                # joint-angle markers (e.g., T10). Export NaNs for joint angles.
                print(f"[WARN] Joint angle computation skipped for {c3d_file.name}: {exc}")

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
                torque_payload=torque_payload,
            )
            if trial_meta is not None:
                for meta_col, meta_val in trial_meta.items():
                    df_ts[meta_col] = meta_val

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
