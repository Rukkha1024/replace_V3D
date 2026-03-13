"""Reproduce platform-onset initial posture LMM and refresh markdown reports.
Load trial data with polars, join metadata, and recompute absolute onset joint/force
features from C3D files. Fit DV ~ step_TF + (1|subject) with R lmerTest and apply
BH-FDR across all testable onset variables. Print console summary and overwrite
`report.md` plus `결과) 주제2-Segement Angle.md` from the current run results.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# path bootstrap (replaces _bootstrap dependency)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests

from replace_v3d.io.c3d_reader import read_c3d_points
from replace_v3d.io.events_excel import (
    load_subject_body_mass_kg,
    load_trial_events,
    parse_subject_velocity_trial_from_filename,
    resolve_subject_from_token,
)
from replace_v3d.com import compute_joint_centers
from replace_v3d.joint_angles.v3d_joint_angles import compute_v3d_joint_angles_3d
from replace_v3d.torque.ankle_torque import compute_ankle_torque_from_net_wrench
from replace_v3d.torque.cop import compute_cop_stage01_xy
from replace_v3d.torque.forceplate import choose_active_force_platform, read_force_platforms
from replace_v3d.torque.forceplate_inertial import (
    apply_forceplate_inertial_subtract,
    load_forceplate_inertial_templates,
)
from replace_v3d.torque.stage01_axis import transform_force_moment_to_stage01

# Required display config by repo rule
pl.Config.set_tbl_rows(999)
pl.Config.set_tbl_cols(999)
pl.Config.set_tbl_width_chars(120)

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_C3D_DIR = REPO_ROOT / "data" / "all_data"
DEFAULT_FP_INERTIAL_TEMPLATES = REPO_ROOT / "src" / "replace_v3d" / "torque" / "assets" / "fp_inertial_templates.npz"
DEFAULT_OUT_DIR = SCRIPT_DIR
DEFAULT_REPORT_MD = SCRIPT_DIR / "report.md"
DEFAULT_SEGMENT_ANGLE_MD = SCRIPT_DIR / "결과) 주제2-Segement Angle.md"

TRIAL_KEYS = ["subject", "velocity", "trial"]
ANGLE_AXES = ("X", "Y", "Z")
STANCE_SEGMENTS = ("Hip", "Knee", "Ankle")
MIDLINE_SEGMENTS = ("Trunk", "Neck")
ANGULAR_VELOCITY_RESOLUTIONS = ("ref", "mov")
STANCE_DYNAMIC_SOURCES: list[tuple[str, str, str]] = []
for _seg in STANCE_SEGMENTS:
    for _res in ANGULAR_VELOCITY_RESOLUTIONS:
        for _axis in ANGLE_AXES:
            STANCE_DYNAMIC_SOURCES.append(
                (
                    f"{_seg}_stance_{_res}_{_axis}_deg_s",
                    f"{_seg}_L_{_res}_{_axis}_deg_s",
                    f"{_seg}_R_{_res}_{_axis}_deg_s",
                )
            )
    for _axis in ANGLE_AXES:
        STANCE_DYNAMIC_SOURCES.append(
            (
                f"{_seg}_stance_ref_{_axis}_Nm",
                f"{_seg}_L_ref_{_axis}_Nm",
                f"{_seg}_R_ref_{_axis}_Nm",
            )
        )

WINDOWS_R_HOME = Path(r"C:\Users\Alice\miniconda3\envs\module\lib\R")
WINDOWS_RSCRIPT = WINDOWS_R_HOME / "bin" / "x64" / "Rscript.exe"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--c3d_dir", type=Path, default=DEFAULT_C3D_DIR)
    ap.add_argument("--fp_inertial_templates", type=Path, default=DEFAULT_FP_INERTIAL_TEMPLATES)
    ap.add_argument(
        "--fp_inertial_policy",
        choices=["skip", "nearest", "interpolate"],
        default="skip",
        help="Missing template policy for force inertial subtraction.",
    )
    ap.add_argument("--fp_inertial_qc_fz_threshold", type=float, default=20.0)
    ap.add_argument("--fp_inertial_qc_margin_m", type=float, default=0.0)
    ap.add_argument("--fp_inertial_qc_strict", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--report_md", type=Path, default=DEFAULT_REPORT_MD)
    ap.add_argument("--segment_angle_md", type=Path, default=DEFAULT_SEGMENT_ANGLE_MD)
    ap.add_argument("--dry-run", action="store_true", help="Only run loading/matching/audit; skip LMM.")
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="Accepted for compatibility; this analysis does not generate figures.",
    )
    return ap.parse_args()


def _candidate_rscripts() -> list[Path]:
    candidates: list[Path] = []

    which_r = shutil.which("Rscript")
    if which_r:
        candidates.append(Path(which_r))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cp = Path(conda_prefix)
        candidates.extend([
            cp / "bin" / "Rscript",
            cp / "Scripts" / "Rscript.exe",
            cp / "lib" / "R" / "bin" / "Rscript",
            cp / "lib" / "R" / "bin" / "x64" / "Rscript.exe",
        ])

    if os.name == "nt":
        candidates.append(WINDOWS_RSCRIPT)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _infer_r_home(rscript_path: Path) -> Path | None:
    env_r_home = os.environ.get("R_HOME")
    if env_r_home:
        candidate = Path(env_r_home)
        if candidate.exists():
            return candidate

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "lib" / "R"
        if candidate.exists():
            return candidate

    if rscript_path.parent.name.lower() == "x64":
        candidate = rscript_path.parent.parent.parent
        if (candidate / "bin").exists():
            return candidate

    candidate = rscript_path.parent.parent
    if (candidate / "bin").exists():
        return candidate

    if os.name == "nt" and rscript_path == WINDOWS_RSCRIPT:
        return WINDOWS_R_HOME
    return None


def resolve_r_runtime() -> tuple[str, dict[str, str]]:
    candidates = _candidate_rscripts()
    resolved: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate
            break

    if resolved is None:
        tried = ", ".join(str(p) for p in candidates) if candidates else "(no candidates)"
        raise FileNotFoundError(
            "Rscript executable not found. "
            f"Tried: {tried}. Install R in env 'module' or ensure Rscript is on PATH."
        )

    env = os.environ.copy()
    env["PATH"] = str(resolved.parent) + os.pathsep + env.get("PATH", "")
    r_home = _infer_r_home(resolved)
    if r_home is not None:
        env["R_HOME"] = str(r_home)
    return str(resolved), env


def load_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(str(path), encoding="utf8-lossy", infer_schema_length=10000)
    if "trial" not in df.columns and "trial_num" in df.columns:
        df = df.rename({"trial_num": "trial"})
    return df


def load_platform_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(str(path), sheet_name="platform")
    required = {
        "subject",
        "velocity",
        "trial",
        "step_TF",
        "state",
        "mixed",
        "platform_onset",
        "step_onset",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"platform sheet missing required columns: {missing}")

    out = df[
        [
            "subject",
            "velocity",
            "trial",
            "step_TF",
            "state",
            "mixed",
            "platform_onset",
            "step_onset",
        ]
    ].copy()
    out["subject"] = out["subject"].astype(str).str.strip()
    out["velocity"] = pd.to_numeric(out["velocity"], errors="coerce")
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce").astype("Int64")
    out["step_TF"] = out["step_TF"].astype(str).str.strip().str.lower()
    out["state"] = out["state"].astype(str).str.strip().str.lower()
    out["mixed"] = pd.to_numeric(out["mixed"], errors="coerce")
    out["platform_onset"] = pd.to_numeric(out["platform_onset"], errors="coerce")
    out["step_onset"] = pd.to_numeric(out["step_onset"], errors="coerce")
    return out


def build_subject_major_step_side(platform: pd.DataFrame) -> pd.DataFrame:
    """Determine subject-level major stepping side from mixed step trials.

    step_r means left leg is stance; step_l means right leg is stance.
    Nonstep uses this major side. Tie keeps both sides and is handled as L/R mean.
    """
    trial_unique = platform[TRIAL_KEYS + ["step_TF", "state", "mixed"]].drop_duplicates().copy()
    mask = (
        trial_unique["mixed"].eq(1)
        & trial_unique["step_TF"].eq("step")
        & trial_unique["state"].isin(["step_r", "step_l"])
    )
    valid = trial_unique.loc[mask, ["subject", "state"]].copy()
    if valid.empty:
        raise ValueError("No mixed==1 step trials with state in {step_r, step_l} were found.")

    counts = (
        valid.assign(n=1)
        .pivot_table(index="subject", columns="state", values="n", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    counts.columns.name = None
    if "step_r" not in counts.columns:
        counts["step_r"] = 0
    if "step_l" not in counts.columns:
        counts["step_l"] = 0

    counts["step_r_count"] = pd.to_numeric(counts["step_r"], errors="coerce").fillna(0).astype(int)
    counts["step_l_count"] = pd.to_numeric(counts["step_l"], errors="coerce").fillna(0).astype(int)
    counts["major_step_side"] = np.where(
        counts["step_r_count"] > counts["step_l_count"],
        "step_r",
        np.where(counts["step_l_count"] > counts["step_r_count"], "step_l", "tie"),
    )
    return counts[["subject", "step_r_count", "step_l_count", "major_step_side"]]


def build_trial_meta(df: pl.DataFrame, platform: pd.DataFrame) -> pd.DataFrame:
    trials = df.select(TRIAL_KEYS).unique().sort(TRIAL_KEYS).to_pandas()
    trials["subject"] = trials["subject"].astype(str).str.strip()
    trials["velocity"] = pd.to_numeric(trials["velocity"], errors="coerce")
    trials["trial"] = pd.to_numeric(trials["trial"], errors="coerce").astype("Int64")

    plat_sub = platform[TRIAL_KEYS + ["step_TF", "state", "mixed"]].drop_duplicates().copy()
    out = trials.merge(plat_sub, on=TRIAL_KEYS, how="left")
    if out["step_TF"].isna().any():
        missing = out.loc[out["step_TF"].isna(), TRIAL_KEYS].head(10)
        raise ValueError(
            "Missing step_TF after joining platform metadata. Sample missing keys: "
            f"{missing.to_dict(orient='records')}"
        )

    out = out[out["step_TF"].isin(["step", "nonstep"])].reset_index(drop=True)
    return out


def _joint_angle_abs_cols() -> list[str]:
    cols: list[str] = []
    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            cols.append(f"{seg}_stance_{axis}_abs_onset")
    for seg in MIDLINE_SEGMENTS:
        for axis in ANGLE_AXES:
            cols.append(f"{seg}_{axis}_abs_onset")
    return cols


def _joint_angle_step_cols() -> list[str]:
    cols: list[str] = []
    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            cols.append(f"{seg}_stance_{axis}_step_onset")
    for seg in MIDLINE_SEGMENTS:
        for axis in ANGLE_AXES:
            cols.append(f"{seg}_{axis}_step_onset")
    return cols


def _platform_onset_dynamic_cols() -> list[str]:
    cols: list[str] = []
    for seg in STANCE_SEGMENTS:
        for res in ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in ANGLE_AXES:
                cols.append(f"{seg}_stance_{res}_{axis}_deg_s")
        for axis in ANGLE_AXES:
            cols.append(f"{seg}_stance_ref_{axis}_Nm")
    for seg in MIDLINE_SEGMENTS:
        for res in ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in ANGLE_AXES:
                cols.append(f"{seg}_{res}_{axis}_deg_s")
        for axis in ANGLE_AXES:
            cols.append(f"{seg}_ref_{axis}_Nm")
    return cols


def _step_onset_variable_catalog() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = [
        {"dv": "COM_X_step_onset", "family": "Balance_step_onset"},
        {"dv": "COM_Y_step_onset", "family": "Balance_step_onset"},
        {"dv": "vCOM_X_step_onset", "family": "Balance_step_onset"},
        {"dv": "vCOM_Y_step_onset", "family": "Balance_step_onset"},
        {"dv": "MOS_minDist_signed_step_onset", "family": "Balance_step_onset"},
        {"dv": "MOS_AP_v3d_step_onset", "family": "Balance_step_onset"},
        {"dv": "MOS_ML_v3d_step_onset", "family": "Balance_step_onset"},
        {"dv": "xCOM_BOS_norm_step_onset", "family": "Balance_step_onset"},
    ]
    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_stance_{axis}_step_onset", "family": "Joint_step_onset"})
    for seg in MIDLINE_SEGMENTS:
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_{axis}_step_onset", "family": "Joint_step_onset"})
    for seg in STANCE_SEGMENTS:
        for res in ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in ANGLE_AXES:
                specs.append(
                    {
                        "dv": f"{seg}_stance_{res}_{axis}_deg_s_step_onset",
                        "family": "Velocity_step_onset",
                    }
                )
        for axis in ANGLE_AXES:
            specs.append(
                {
                    "dv": f"{seg}_stance_ref_{axis}_Nm_step_onset",
                    "family": "Moment_step_onset",
                }
            )
    for seg in MIDLINE_SEGMENTS:
        for res in ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in ANGLE_AXES:
                specs.append(
                    {
                        "dv": f"{seg}_{res}_{axis}_deg_s_step_onset",
                        "family": "Velocity_step_onset",
                    }
                )
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_ref_{axis}_Nm_step_onset", "family": "Moment_step_onset"})
    specs.extend([
        {"dv": "COP_X_step_onset", "family": "Force_step_onset"},
        {"dv": "COP_Y_step_onset", "family": "Force_step_onset"},
        {"dv": "GRF_X_step_onset", "family": "Force_step_onset"},
        {"dv": "GRF_Y_step_onset", "family": "Force_step_onset"},
        {"dv": "GRF_Z_step_onset", "family": "Force_step_onset"},
        {"dv": "AnkleTorqueMid_Y_perkg_step_onset", "family": "Force_step_onset"},
    ])
    return specs


def add_stance_cols_pl(df: pl.DataFrame) -> pl.DataFrame:
    exprs: list[pl.Expr] = []
    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            left_col = f"{seg}_L_{axis}_deg"
            right_col = f"{seg}_R_{axis}_deg"
            out_col = f"{seg}_stance_{axis}_deg"
            exprs.append(
                pl.when(pl.col("state") == "step_r")
                .then(pl.col(left_col))
                .when(pl.col("state") == "step_l")
                .then(pl.col(right_col))
                .when(pl.col("major_step_side") == "step_r")
                .then(pl.col(left_col))
                .when(pl.col("major_step_side") == "step_l")
                .then(pl.col(right_col))
                .otherwise((pl.col(left_col) + pl.col(right_col)) / 2.0)
                .alias(out_col)
            )
    for out_col, left_col, right_col in STANCE_DYNAMIC_SOURCES:
        exprs.append(
            pl.when(pl.col("state") == "step_r")
            .then(pl.col(left_col))
            .when(pl.col("state") == "step_l")
            .then(pl.col(right_col))
            .when(pl.col("major_step_side") == "step_r")
            .then(pl.col(left_col))
            .when(pl.col("major_step_side") == "step_l")
            .then(pl.col(right_col))
            .otherwise((pl.col(left_col) + pl.col(right_col)) / 2.0)
            .alias(out_col)
        )
    return df.with_columns(exprs)


def build_onset_snapshot(
    df: pl.DataFrame,
    trial_meta: pd.DataFrame,
    major_side: pd.DataFrame,
) -> pd.DataFrame:
    onset = (
        df.group_by(TRIAL_KEYS)
        .agg(pl.col("platform_onset_local").drop_nulls().first().cast(pl.Int64).alias("platform_onset_local"))
        .sort(TRIAL_KEYS)
    )

    snap = (
        df.join(onset, on=TRIAL_KEYS, how="inner")
        .filter(pl.col("MocapFrame") == pl.col("platform_onset_local"))
        .join(pl.from_pandas(trial_meta), on=TRIAL_KEYS, how="left")
        .join(pl.from_pandas(major_side), on="subject", how="left")
    )

    dup = snap.group_by(TRIAL_KEYS).len().filter(pl.col("len") != 1)
    if dup.height > 0:
        raise ValueError(f"Onset snapshot has non-unique rows for {dup.height} trials.")

    snap = add_stance_cols_pl(snap)

    snap = snap.with_columns(
        pl.when((pl.col("BOS_maxX") - pl.col("BOS_minX")) > 0)
        .then((pl.col("xCOM_X") - pl.col("BOS_minX")) / (pl.col("BOS_maxX") - pl.col("BOS_minX")))
        .otherwise(None)
        .alias("xCOM_BOS_norm_onset")
    )

    return snap.to_pandas()


def build_step_onset_snapshot(
    df: pl.DataFrame,
    trial_meta: pd.DataFrame,
    major_side: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df_base = df.drop([c for c in ["step_TF", "state", "mixed"] if c in df.columns])
    trial_meta_pl = pl.from_pandas(trial_meta[TRIAL_KEYS + ["step_TF", "state", "mixed"]].copy())
    major_side_pl = pl.from_pandas(major_side[["subject", "major_step_side"]].copy())

    trial_step = (
        df.group_by(TRIAL_KEYS)
        .agg(pl.col("step_onset_local").drop_nulls().first().cast(pl.Float64).alias("step_onset_local"))
        .sort(TRIAL_KEYS)
    )
    trial_info = trial_meta_pl.join(trial_step, on=TRIAL_KEYS, how="left")

    subject_step_ref = (
        trial_info
        .filter((pl.col("step_TF") == "step") & pl.col("step_onset_local").is_not_null())
        .group_by("subject")
        .agg(pl.col("step_onset_local").mean().alias("step_onset_subject_mean_local"))
    )

    target = (
        trial_info.join(subject_step_ref, on="subject", how="left")
        .with_columns(
            pl.when(pl.col("step_TF") == "step")
            .then(pl.col("step_onset_local"))
            .otherwise(pl.col("step_onset_subject_mean_local"))
            .alias("step_onset_target_local_float")
        )
        .with_columns(
            pl.when(pl.col("step_onset_target_local_float").is_not_null())
            .then(pl.col("step_onset_target_local_float").round(0).cast(pl.Int64))
            .otherwise(None)
            .alias("step_onset_target_local")
        )
    )

    snap = (
        df_base.join(
            target.select(
                TRIAL_KEYS
                + [
                    "step_onset_local",
                    "step_onset_subject_mean_local",
                    "step_onset_target_local",
                ]
            ),
            on=TRIAL_KEYS,
            how="inner",
        )
        .filter(pl.col("step_onset_target_local").is_not_null())
        .filter(pl.col("MocapFrame").cast(pl.Int64) == pl.col("step_onset_target_local"))
        .join(trial_meta_pl, on=TRIAL_KEYS, how="left")
        .join(major_side_pl, on="subject", how="left")
    )

    dup = snap.group_by(TRIAL_KEYS).len().filter(pl.col("len") != 1)
    if dup.height > 0:
        raise ValueError(f"Step-onset snapshot has non-unique rows for {dup.height} trials.")

    snap = add_stance_cols_pl(snap)
    snap = snap.with_columns(
        pl.when((pl.col("BOS_maxX") - pl.col("BOS_minX")) > 0)
        .then((pl.col("xCOM_X") - pl.col("BOS_minX")) / (pl.col("BOS_maxX") - pl.col("BOS_minX")))
        .otherwise(None)
        .alias("xCOM_BOS_norm_step_onset")
    )

    src_to_out: dict[str, str] = {}
    src_to_out.update(
        {
            "COM_X": "COM_X_step_onset",
            "COM_Y": "COM_Y_step_onset",
            "vCOM_X": "vCOM_X_step_onset",
            "vCOM_Y": "vCOM_Y_step_onset",
            "MOS_minDist_signed": "MOS_minDist_signed_step_onset",
            "MOS_AP_v3d": "MOS_AP_v3d_step_onset",
            "MOS_ML_v3d": "MOS_ML_v3d_step_onset",
            "xCOM_BOS_norm_step_onset": "xCOM_BOS_norm_step_onset",
            "COP_X_m": "COP_X_step_onset",
            "COP_Y_m": "COP_Y_step_onset",
            "GRF_X_N": "GRF_X_step_onset",
            "GRF_Y_N": "GRF_Y_step_onset",
            "GRF_Z_N": "GRF_Z_step_onset",
            "AnkleTorqueMid_int_Y_Nm_per_kg": "AnkleTorqueMid_Y_perkg_step_onset",
        }
    )
    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            src_to_out[f"{seg}_stance_{axis}_deg"] = f"{seg}_stance_{axis}_step_onset"
    for seg in MIDLINE_SEGMENTS:
        for axis in ANGLE_AXES:
            src_to_out[f"{seg}_{axis}_deg"] = f"{seg}_{axis}_step_onset"
    for seg in STANCE_SEGMENTS:
        for res in ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in ANGLE_AXES:
                src_to_out[f"{seg}_stance_{res}_{axis}_deg_s"] = f"{seg}_stance_{res}_{axis}_deg_s_step_onset"
        for axis in ANGLE_AXES:
            src_to_out[f"{seg}_stance_ref_{axis}_Nm"] = f"{seg}_stance_ref_{axis}_Nm_step_onset"
    for seg in MIDLINE_SEGMENTS:
        for res in ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in ANGLE_AXES:
                src_to_out[f"{seg}_{res}_{axis}_deg_s"] = f"{seg}_{res}_{axis}_deg_s_step_onset"
        for axis in ANGLE_AXES:
            src_to_out[f"{seg}_ref_{axis}_Nm"] = f"{seg}_ref_{axis}_Nm_step_onset"

    missing_src = sorted(set(src_to_out.keys()) - set(snap.columns))
    if missing_src:
        raise ValueError(f"Missing step-onset snapshot columns from CSV snapshot: {missing_src}")

    step_df = (
        snap.select(TRIAL_KEYS + ["step_TF"] + list(src_to_out.keys()))
        .rename(src_to_out)
        .to_pandas()
    )

    target_pd = target.to_pandas()
    n_trials_total = int(len(target_pd))
    n_step_total = int((target_pd["step_TF"] == "step").sum())
    n_nonstep_total = int((target_pd["step_TF"] == "nonstep").sum())
    n_step_missing_direct = int(
        ((target_pd["step_TF"] == "step") & target_pd["step_onset_local"].isna()).sum()
    )
    n_nonstep_missing_ref = int(
        ((target_pd["step_TF"] == "nonstep") & target_pd["step_onset_subject_mean_local"].isna()).sum()
    )
    missing_ref_subjects = sorted(
        target_pd.loc[
            (target_pd["step_TF"] == "nonstep") & target_pd["step_onset_subject_mean_local"].isna(),
            "subject",
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    n_target_available = int(target_pd["step_onset_target_local"].notna().sum())
    n_used = int(len(step_df))
    n_step_used = int((step_df["step_TF"] == "step").sum())
    n_nonstep_used = int((step_df["step_TF"] == "nonstep").sum())
    n_target_no_frame = int(max(0, n_target_available - n_used))
    n_excluded = int(max(0, n_trials_total - n_used))
    n_subject_ref = int(target_pd["step_onset_subject_mean_local"].notna().groupby(target_pd["subject"]).any().sum())

    stats = {
        "n_trials_total": n_trials_total,
        "n_step_total": n_step_total,
        "n_nonstep_total": n_nonstep_total,
        "n_subject_ref": n_subject_ref,
        "n_step_missing_direct": n_step_missing_direct,
        "n_nonstep_missing_ref": n_nonstep_missing_ref,
        "missing_ref_subjects": missing_ref_subjects,
        "n_target_available": n_target_available,
        "n_used": n_used,
        "n_step_used": n_step_used,
        "n_nonstep_used": n_nonstep_used,
        "n_target_no_frame": n_target_no_frame,
        "n_excluded": n_excluded,
    }
    return step_df, stats


def _pick_stance_value(row: pd.Series, left_col: str, right_col: str) -> float:
    """Select stance-side value for one trial.

    Priority:
    1) step_r -> left side, step_l -> right side.
    2) nonstep -> subject major_step_side.
    3) tie -> average of left and right.
    """
    state = str(row.get("state", "")).lower().strip()
    major = str(row.get("major_step_side", "")).lower().strip()
    if state == "step_r":
        return float(row[left_col])
    if state == "step_l":
        return float(row[right_col])
    if major == "step_r":
        return float(row[left_col])
    if major == "step_l":
        return float(row[right_col])
    return float(np.nanmean([row[left_col], row[right_col]]))


def summarize_major_step_side(major_side: pd.DataFrame) -> dict[str, Any]:
    """Summarize subject-level major step side and tie subjects."""
    counts = major_side["major_step_side"].value_counts(dropna=False).to_dict()
    tie_subjects = (
        major_side.loc[major_side["major_step_side"] == "tie", "subject"]
        .astype(str)
        .tolist()
    )
    return {
        "subjects": int(len(major_side)),
        "step_r_major": int(counts.get("step_r", 0)),
        "step_l_major": int(counts.get("step_l", 0)),
        "tie_major": int(counts.get("tie", 0)),
        "tie_subjects": tie_subjects,
    }


def _build_c3d_key_map(c3d_dir: Path, platform_xlsm: Path, trial_keys: set[tuple[str, float, int]]) -> dict[tuple[str, float, int], Path]:
    key_to_file: dict[tuple[str, float, int], Path] = {}
    duplicates: list[tuple[tuple[str, float, int], Path, Path]] = []

    for c3d_file in sorted(c3d_dir.rglob("*.c3d")):
        try:
            token, velocity, trial = parse_subject_velocity_trial_from_filename(c3d_file.name)
            subject = resolve_subject_from_token(platform_xlsm, token)
        except Exception:
            continue

        key = (str(subject).strip(), float(velocity), int(trial))
        if key not in trial_keys:
            continue
        if key in key_to_file:
            duplicates.append((key, key_to_file[key], c3d_file))
            continue
        key_to_file[key] = c3d_file

    if duplicates:
        sample = [
            {
                "key": x[0],
                "first": str(x[1]),
                "second": str(x[2]),
            }
            for x in duplicates[:5]
        ]
        raise ValueError(f"Duplicate C3D match for trial keys. Sample: {sample}")

    return key_to_file


def compute_absolute_onset_features(
    trial_meta: pd.DataFrame,
    platform_xlsm: Path,
    c3d_dir: Path,
    major_side: pd.DataFrame,
    fp_inertial_templates_path: Path,
    fp_inertial_policy: str,
    fp_inertial_qc_fz_threshold: float,
    fp_inertial_qc_margin_m: float,
    fp_inertial_qc_strict: bool,
) -> pd.DataFrame:
    templates = load_forceplate_inertial_templates(fp_inertial_templates_path)

    keys = set(
        (str(r.subject).strip(), float(r.velocity), int(r.trial))
        for r in trial_meta.itertuples(index=False)
    )

    key_to_file = _build_c3d_key_map(c3d_dir=c3d_dir, platform_xlsm=platform_xlsm, trial_keys=keys)
    missing_keys = sorted(keys - set(key_to_file.keys()))
    if missing_keys:
        raise FileNotFoundError(
            "Missing matched C3D files for trial keys. "
            f"count={len(missing_keys)}, sample={missing_keys[:10]}"
        )

    major_lookup = major_side.set_index("subject")["major_step_side"].to_dict()
    trial_lookup = trial_meta.set_index(TRIAL_KEYS)[["step_TF", "state"]].to_dict("index")

    rows: list[dict[str, Any]] = []
    skipped_trials: list[dict[str, str]] = []
    qc_warn_count = 0

    for key in sorted(keys):
        subject, velocity, trial = key
        c3d_file = key_to_file[key]
        try:
            c3d = read_c3d_points(c3d_file)
            events = load_trial_events(
                event_xlsm=platform_xlsm,
                subject=subject,
                velocity=float(velocity),
                trial=int(trial),
                pre_frames=100,
                sheet_name="platform",
            )
            onset_local = int(events.platform_onset_local)
            idx0 = onset_local - 1

            if idx0 < 0 or idx0 >= int(c3d.points.shape[0]):
                raise IndexError(
                    "platform_onset_local out of range for raw C3D frames: "
                    f"key={key}, onset={onset_local}, n_frames={c3d.points.shape[0]}"
                )

            angles = compute_v3d_joint_angles_3d(c3d.points, c3d.labels, end_frame=c3d.points.shape[0])

            bilateral: dict[str, float] = {}
            for seg in STANCE_SEGMENTS:
                seg_key = seg.lower()
                for side in ("L", "R"):
                    for axis in ANGLE_AXES:
                        attr = f"{seg_key}_{side}_{axis}"
                        col = f"{seg}_{side}_{axis}_abs_onset"
                        v = float(getattr(angles, attr)[idx0])
                        # Match repository joint-angle sign convention used by the CSV pipeline:
                        # LEFT Hip/Knee/Ankle Y/Z are negated so that Y/Z sign meaning matches RIGHT.
                        # This matters when we later mix stance across L/R.
                        if side == "L" and axis in ("Y", "Z"):
                            v *= -1.0
                        bilateral[col] = v
            for seg in MIDLINE_SEGMENTS:
                seg_key = seg.lower()
                for axis in ANGLE_AXES:
                    attr = f"{seg_key}_{axis}"
                    col = f"{seg}_{axis}_abs_onset"
                    bilateral[col] = float(getattr(angles, attr)[idx0])

            step_state = trial_lookup[(subject, velocity, trial)]
            row = {
                "subject": subject,
                "velocity": float(velocity),
                "trial": int(trial),
                "step_TF": str(step_state["step_TF"]).lower().strip(),
                "state": str(step_state["state"]).lower().strip(),
                "major_step_side": str(major_lookup.get(subject, "tie")).lower().strip(),
                **bilateral,
            }

            s = pd.Series(row)
            for seg in STANCE_SEGMENTS:
                for axis in ANGLE_AXES:
                    left_col = f"{seg}_L_{axis}_abs_onset"
                    right_col = f"{seg}_R_{axis}_abs_onset"
                    out_col = f"{seg}_stance_{axis}_abs_onset"
                    row[out_col] = _pick_stance_value(s, left_col, right_col)

            fp_coll = read_force_platforms(c3d_file)
            analog_avg = fp_coll.analog.values
            n_frames = int(c3d.points.shape[0])
            if analog_avg.shape[0] != n_frames:
                raise ValueError(
                    f"Analog frames ({analog_avg.shape[0]}) != point frames ({n_frames}) for {c3d_file.name}"
                )
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

            # Align sign to repository Stage01 inertial-template convention.
            analog_shared_sign = analog_stage01.copy()
            analog_shared_sign[:, idx[0:3]] *= -1.0
            analog_shared_sign[:, idx[3:6]] *= -1.0

            offset0 = int(events.platform_offset_local) - 1
            analog_used, inertial_info = apply_forceplate_inertial_subtract(
                analog_shared_sign,
                fp,
                velocity=float(velocity),
                onset0=int(idx0),
                offset0=int(offset0),
                templates=templates,
                missing_policy=str(fp_inertial_policy),
                qc_fz_threshold_n=float(fp_inertial_qc_fz_threshold),
                qc_margin_m=float(fp_inertial_qc_margin_m),
            )
            if not inertial_info.get("applied"):
                raise ValueError(
                    "Forceplate inertial subtract did not apply for "
                    f"{c3d_file.name} (reason={inertial_info.get('reason')}, "
                    f"policy={inertial_info.get('missing_policy')})."
                )
            if inertial_info.get("qc_failed"):
                msg = (
                    "Forceplate inertial subtract QC failed "
                    f"for {c3d_file.name} (COP in-bounds after="
                    f"{inertial_info.get('after_qc_cop_in_bounds_frac')})."
                )
                if fp_inertial_qc_strict:
                    raise ValueError(msg)
                qc_warn_count += 1

            F_stage01 = analog_used[:, idx[0:3]]
            M_stage01 = analog_used[:, idx[3:6]]
            COP_stage01_xy = compute_cop_stage01_xy(
                F_stage01=F_stage01,
                M_stage01=M_stage01,
            )
            jc = compute_joint_centers(c3d.points, c3d.labels)
            body_mass_kg = load_subject_body_mass_kg(platform_xlsm, subject)
            torque_res = compute_ankle_torque_from_net_wrench(
                F_lab=F_stage01,
                M_lab_at_fp_origin=M_stage01,
                fp_origin_lab=fp.origin_lab,
                ankle_L=jc["ankle_L"],
                ankle_R=jc["ankle_R"],
                body_mass_kg=body_mass_kg,
            )

            row["GRF_X_abs_onset"] = float(F_stage01[idx0, 0])
            row["GRF_Y_abs_onset"] = float(F_stage01[idx0, 1])
            row["GRF_Z_abs_onset"] = float(F_stage01[idx0, 2])
            row["COP_X_abs_onset"] = float(COP_stage01_xy[idx0, 0])
            row["COP_Y_abs_onset"] = float(COP_stage01_xy[idx0, 1])
            row["AnkleTorqueMid_Y_perkg_abs_onset"] = (
                float(np.nan)
                if torque_res.torque_mid_int_Y_Nm_per_kg is None
                else float(torque_res.torque_mid_int_Y_Nm_per_kg[idx0])
            )

            rows.append(row)
        except Exception as exc:
            skipped_trials.append(
                {
                    "subject": str(subject),
                    "velocity": str(velocity),
                    "trial": str(trial),
                    "file": c3d_file.name,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

    if not rows:
        raise ValueError("No valid trials remained after absolute onset feature computation.")

    out = pd.DataFrame(rows)
    if qc_warn_count > 0:
        print(f"  Warning: force inertial subtract QC failed in {qc_warn_count} trials (non-strict mode).")
    if skipped_trials:
        print(
            "  Warning: skipped "
            f"{len(skipped_trials)} trials during absolute onset feature computation."
        )
        for skipped in skipped_trials[:5]:
            print(
                "    - "
                f"{skipped['subject']}, vel={skipped['velocity']}, trial={skipped['trial']} "
                f"[{skipped['file']}] -> {skipped['reason']}"
            )
        if len(skipped_trials) > 5:
            print(f"    ... {len(skipped_trials) - 5} more skipped trials")

    return out[
        TRIAL_KEYS
        + _joint_angle_abs_cols()
        + [
            "GRF_X_abs_onset",
            "GRF_Y_abs_onset",
            "GRF_Z_abs_onset",
            "COP_X_abs_onset",
            "COP_Y_abs_onset",
            "AnkleTorqueMid_Y_perkg_abs_onset",
        ]
    ]


def build_analysis_dataframe(
    csv_path: Path,
    platform_xlsm: Path,
    c3d_dir: Path,
    fp_inertial_templates_path: Path,
    fp_inertial_policy: str,
    fp_inertial_qc_fz_threshold: float,
    fp_inertial_qc_margin_m: float,
    fp_inertial_qc_strict: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print("  Loading CSV...")
    df = load_csv(csv_path)
    print(f"  Frames: {df.height}, Columns: {df.width}")

    print("  Loading platform metadata...")
    platform = load_platform_sheet(platform_xlsm)

    trial_meta = build_trial_meta(df=df, platform=platform)
    major_side = build_subject_major_step_side(platform=platform)
    trial_subjects = set(trial_meta["subject"].astype(str).tolist())
    major_side = major_side[major_side["subject"].astype(str).isin(trial_subjects)].copy()

    print(
        "  Trial set: "
        f"{len(trial_meta)} (step={(trial_meta['step_TF'] == 'step').sum()}, "
        f"nonstep={(trial_meta['step_TF'] == 'nonstep').sum()})"
    )

    print("  Building onset snapshot variables from exported CSV...")
    onset_df = build_onset_snapshot(df=df, trial_meta=trial_meta, major_side=major_side)
    print("  Building step-onset snapshot (step mean imputation for nonstep)...")
    step_onset_df, step_joint_stats = build_step_onset_snapshot(
        df=df,
        trial_meta=trial_meta,
        major_side=major_side,
    )

    keep_cols = TRIAL_KEYS + [
        "step_TF",
        "state",
        "major_step_side",
        "COM_X",
        "COM_Y",
        "vCOM_X",
        "vCOM_Y",
        "MOS_minDist_signed",
        "MOS_AP_v3d",
        "MOS_ML_v3d",
        "xCOM_BOS_norm_onset",
    ] + _platform_onset_dynamic_cols()
    missing_cols = sorted(set(keep_cols) - set(onset_df.columns))
    if missing_cols:
        raise ValueError(f"Missing onset columns from CSV snapshot: {missing_cols}")

    onset_df = onset_df[keep_cols].copy()

    print("  Recomputing absolute onset angles/forces from C3D...")
    abs_features = compute_absolute_onset_features(
        trial_meta=trial_meta,
        platform_xlsm=platform_xlsm,
        c3d_dir=c3d_dir,
        major_side=major_side,
        fp_inertial_templates_path=fp_inertial_templates_path,
        fp_inertial_policy=fp_inertial_policy,
        fp_inertial_qc_fz_threshold=fp_inertial_qc_fz_threshold,
        fp_inertial_qc_margin_m=fp_inertial_qc_margin_m,
        fp_inertial_qc_strict=fp_inertial_qc_strict,
    )

    available_keys = abs_features[TRIAL_KEYS].drop_duplicates()
    n_abs_dropped = int(len(trial_meta) - len(trial_meta.merge(available_keys, on=TRIAL_KEYS, how="inner")))
    if n_abs_dropped > 0:
        print(f"  Warning: dropping {n_abs_dropped} trials without valid absolute onset features.")
    trial_meta = trial_meta.merge(available_keys, on=TRIAL_KEYS, how="inner").reset_index(drop=True)
    onset_df = onset_df.merge(available_keys, on=TRIAL_KEYS, how="inner")
    analysis_df = onset_df.merge(abs_features, on=TRIAL_KEYS, how="inner")
    required_abs_cols = _joint_angle_abs_cols() + [
        "GRF_X_abs_onset",
        "GRF_Y_abs_onset",
        "GRF_Z_abs_onset",
        "COP_X_abs_onset",
        "COP_Y_abs_onset",
        "AnkleTorqueMid_Y_perkg_abs_onset",
    ]
    missing_abs_mask = analysis_df[required_abs_cols].isna().any(axis=1)
    if bool(missing_abs_mask.any()):
        n_missing_abs = int(missing_abs_mask.sum())
        print(f"  Warning: dropping {n_missing_abs} trials with incomplete absolute onset feature values.")
        analysis_df = analysis_df.loc[~missing_abs_mask].reset_index(drop=True)
        valid_keys = analysis_df[TRIAL_KEYS].drop_duplicates()
        trial_meta = trial_meta.merge(valid_keys, on=TRIAL_KEYS, how="inner").reset_index(drop=True)
    if analysis_df.empty:
        raise ValueError("No valid analysis trials remained after absolute onset feature filtering.")

    return analysis_df, trial_meta, abs_features, major_side, step_onset_df, step_joint_stats


def variable_catalog() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = [
        {"dv": "COM_X", "family": "Balance"},
        {"dv": "COM_Y", "family": "Balance"},
        {"dv": "vCOM_X", "family": "Balance"},
        {"dv": "vCOM_Y", "family": "Balance"},
        {"dv": "MOS_minDist_signed", "family": "Balance"},
        {"dv": "MOS_AP_v3d", "family": "Balance"},
        {"dv": "MOS_ML_v3d", "family": "Balance"},
        {"dv": "xCOM_BOS_norm_onset", "family": "Balance"},
    ]
    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_stance_{axis}_abs_onset", "family": "Joint_absolute"})
    for seg in MIDLINE_SEGMENTS:
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_{axis}_abs_onset", "family": "Joint_absolute"})
    for seg in STANCE_SEGMENTS:
        for res in ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in ANGLE_AXES:
                specs.append(
                    {
                        "dv": f"{seg}_stance_{res}_{axis}_deg_s",
                        "family": "Velocity_platform_onset",
                    }
                )
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_stance_ref_{axis}_Nm", "family": "Moment_platform_onset"})
    for seg in MIDLINE_SEGMENTS:
        for res in ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in ANGLE_AXES:
                specs.append(
                    {
                        "dv": f"{seg}_{res}_{axis}_deg_s",
                        "family": "Velocity_platform_onset",
                    }
                )
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_ref_{axis}_Nm", "family": "Moment_platform_onset"})
    specs.extend([
        {"dv": "COP_X_abs_onset", "family": "Force_absolute"},
        {"dv": "COP_Y_abs_onset", "family": "Force_absolute"},
        {"dv": "GRF_X_abs_onset", "family": "Force_absolute"},
        {"dv": "GRF_Y_abs_onset", "family": "Force_absolute"},
        {"dv": "GRF_Z_abs_onset", "family": "Force_absolute"},
        {"dv": "AnkleTorqueMid_Y_perkg_abs_onset", "family": "Force_absolute"},
    ])
    return specs


def audit_variables(analysis_df: pd.DataFrame, specs: list[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        dv = spec["dv"]
        family = spec["family"]
        s = pd.to_numeric(analysis_df[dv], errors="coerce")
        nonnull = int(s.notna().sum())
        uniq = int(s.dropna().nunique()) if nonnull > 0 else 0

        if nonnull == 0:
            status = "untestable"
            reason = "all_missing"
        elif uniq <= 1:
            status = "untestable"
            val = float(s.dropna().iloc[0])
            reason = "constant_zero" if abs(val) < 1e-12 else "constant_nonzero"
        else:
            status = "testable"
            reason = "ok"

        rows.append(
            {
                "dv": dv,
                "family": family,
                "n_nonnull": nonnull,
                "n_unique": uniq,
                "status": status,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


def _group_outlier_mask(series: pd.Series) -> pd.Series:
    """Mark non-outlier values using the 1.5*IQR rule within one group."""
    values = pd.to_numeric(series, errors="coerce")
    keep = pd.Series(False, index=values.index, dtype=bool)
    finite = values.dropna()
    if finite.empty:
        return keep

    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    keep.loc[finite.index] = (finite >= lower) & (finite <= upper)
    return keep


def compute_outlier_summary(analysis_df: pd.DataFrame, specs: list[dict[str, str]]) -> pd.DataFrame:
    """Summarize per-variable outlier counts within step/nonstep groups."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        dv = spec["dv"]
        sub = analysis_df[TRIAL_KEYS + ["step_TF", dv]].copy()
        sub[dv] = pd.to_numeric(sub[dv], errors="coerce")
        sub = sub[sub["step_TF"].isin(["step", "nonstep"]) & sub[dv].notna()].copy()

        if sub.empty:
            rows.append(
                {
                    "dv": dv,
                    "n_step_raw": 0,
                    "n_nonstep_raw": 0,
                    "n_step_outliers": 0,
                    "n_nonstep_outliers": 0,
                    "n_step_kept": 0,
                    "n_nonstep_kept": 0,
                }
            )
            continue

        keep = pd.Series(False, index=sub.index, dtype=bool)
        for group in ("step", "nonstep"):
            group_idx = sub.index[sub["step_TF"] == group]
            keep.loc[group_idx] = _group_outlier_mask(sub.loc[group_idx, dv])

        n_step_raw = int((sub["step_TF"] == "step").sum())
        n_nonstep_raw = int((sub["step_TF"] == "nonstep").sum())
        n_step_kept = int(((sub["step_TF"] == "step") & keep).sum())
        n_nonstep_kept = int(((sub["step_TF"] == "nonstep") & keep).sum())
        rows.append(
            {
                "dv": dv,
                "n_step_raw": n_step_raw,
                "n_nonstep_raw": n_nonstep_raw,
                "n_step_outliers": n_step_raw - n_step_kept,
                "n_nonstep_outliers": n_nonstep_raw - n_nonstep_kept,
                "n_step_kept": n_step_kept,
                "n_nonstep_kept": n_nonstep_kept,
            }
        )
    return pd.DataFrame(rows)


def _build_r_lmm_script(csv_path: str, output_path: str) -> str:
    return f"""
library(lmerTest)

data <- read.csv("{csv_path}", stringsAsFactors = FALSE)
data$step_TF <- tolower(trimws(as.character(data$step_TF)))
data$step_TF <- factor(data$step_TF, levels = c("nonstep", "step"))
dv_cols <- colnames(data)[!colnames(data) %in% c("subject", "velocity", "trial", "step_TF")]

iqr_keep <- function(x) {{
  keep <- rep(FALSE, length(x))
  finite_idx <- which(is.finite(x))
  if (length(finite_idx) == 0) {{
    return(keep)
  }}
  vals <- x[finite_idx]
  q1 <- unname(quantile(vals, 0.25, na.rm = TRUE, type = 7))
  q3 <- unname(quantile(vals, 0.75, na.rm = TRUE, type = 7))
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  keep_vals <- vals >= lower & vals <= upper
  keep[finite_idx] <- keep_vals
  keep
}}

results <- data.frame(
  dv = character(),
  analysis_status = character(),
  estimate = numeric(),
  SE = numeric(),
  df = numeric(),
  t_value = numeric(),
  p_value = numeric(),
  mean_step = numeric(),
  sd_step = numeric(),
  mean_nonstep = numeric(),
  sd_nonstep = numeric(),
  n_step_raw = integer(),
  n_nonstep_raw = integer(),
  n_step_outliers = integer(),
  n_nonstep_outliers = integer(),
  n_step = integer(),
  n_nonstep = integer(),
  converged = logical(),
  stringsAsFactors = FALSE
)

for (dv in dv_cols) {{
  sub <- data[!is.na(data[[dv]]) & !is.na(data$step_TF), ]
  step_mask <- sub$step_TF == "step"
  nonstep_mask <- sub$step_TF == "nonstep"
  n_step_raw <- sum(step_mask)
  n_nonstep_raw <- sum(nonstep_mask)

  keep_mask <- rep(FALSE, nrow(sub))
  keep_mask[step_mask] <- iqr_keep(sub[[dv]][step_mask])
  keep_mask[nonstep_mask] <- iqr_keep(sub[[dv]][nonstep_mask])

  n_step_outliers <- n_step_raw - sum(keep_mask[step_mask])
  n_nonstep_outliers <- n_nonstep_raw - sum(keep_mask[nonstep_mask])
  sub <- sub[keep_mask, ]

  n_s <- sum(sub$step_TF == "step")
  n_ns <- sum(sub$step_TF == "nonstep")

  m_s <- mean(sub[[dv]][sub$step_TF == "step"], na.rm = TRUE)
  sd_s <- sd(sub[[dv]][sub$step_TF == "step"], na.rm = TRUE)
  m_ns <- mean(sub[[dv]][sub$step_TF == "nonstep"], na.rm = TRUE)
  sd_ns <- sd(sub[[dv]][sub$step_TF == "nonstep"], na.rm = TRUE)

  if (n_s == 0 || n_ns == 0) {{
    results <- rbind(results, data.frame(
      dv = dv, analysis_status = "group_empty_after_outlier", estimate = NA, SE = NA, df = NA,
      t_value = NA, p_value = NA,
      mean_step = m_s, sd_step = sd_s,
      mean_nonstep = m_ns, sd_nonstep = sd_ns,
      n_step_raw = n_step_raw, n_nonstep_raw = n_nonstep_raw,
      n_step_outliers = n_step_outliers, n_nonstep_outliers = n_nonstep_outliers,
      n_step = n_s, n_nonstep = n_ns, converged = FALSE,
      stringsAsFactors = FALSE
    ))
    next
  }}

  if (length(unique(sub[[dv]])) <= 1) {{
    results <- rbind(results, data.frame(
      dv = dv, analysis_status = "constant_after_outlier", estimate = NA, SE = NA, df = NA,
      t_value = NA, p_value = NA,
      mean_step = m_s, sd_step = sd_s,
      mean_nonstep = m_ns, sd_nonstep = sd_ns,
      n_step_raw = n_step_raw, n_nonstep_raw = n_nonstep_raw,
      n_step_outliers = n_step_outliers, n_nonstep_outliers = n_nonstep_outliers,
      n_step = n_s, n_nonstep = n_ns, converged = FALSE,
      stringsAsFactors = FALSE
    ))
    next
  }}

  formula_str <- paste0("`", dv, "` ~ step_TF + (1|subject)")
  tryCatch({{
    m <- lmer(as.formula(formula_str), data = sub, REML = TRUE)
    s <- summary(m)
    co <- coef(s)

    row_name <- grep("^step_TFstep$", rownames(co), value = TRUE)[1]
    if (!is.na(row_name)) {{
      est <- co[row_name, "Estimate"]
      se <- co[row_name, "Std. Error"]
      df_val <- co[row_name, "df"]
      t_val <- co[row_name, "t value"]
      p_val <- co[row_name, "Pr(>|t|)"]
    }} else {{
      est <- NA; se <- NA; df_val <- NA; t_val <- NA; p_val <- NA
    }}

    results <- rbind(results, data.frame(
      dv = dv, analysis_status = "ok", estimate = est, SE = se, df = df_val,
      t_value = t_val, p_value = p_val,
      mean_step = m_s, sd_step = sd_s,
      mean_nonstep = m_ns, sd_nonstep = sd_ns,
      n_step_raw = n_step_raw, n_nonstep_raw = n_nonstep_raw,
      n_step_outliers = n_step_outliers, n_nonstep_outliers = n_nonstep_outliers,
      n_step = n_s, n_nonstep = n_ns, converged = TRUE,
      stringsAsFactors = FALSE
    ))
  }}, error = function(e) {{
    results <<- rbind(results, data.frame(
      dv = dv, analysis_status = "model_error", estimate = NA, SE = NA, df = NA,
      t_value = NA, p_value = NA,
      mean_step = m_s, sd_step = sd_s,
      mean_nonstep = m_ns, sd_nonstep = sd_ns,
      n_step_raw = n_step_raw, n_nonstep_raw = n_nonstep_raw,
      n_step_outliers = n_step_outliers, n_nonstep_outliers = n_nonstep_outliers,
      n_step = n_s, n_nonstep = n_ns, converged = FALSE,
      stringsAsFactors = FALSE
    ))
  }})
}}

write.csv(results, "{output_path}", row.names = FALSE)
cat("LMM fitting complete:", nrow(results), "models\\n")
"""


def fit_lmm_all(analysis_df: pd.DataFrame, testable_vars: list[str], dv_to_family: dict[str, str]) -> pd.DataFrame:
    export_cols = TRIAL_KEYS + ["step_TF"] + testable_vars
    export_df = analysis_df[export_cols].copy()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, dir=tempfile.gettempdir(), newline="") as f:
        export_df.to_csv(f, index=False)
        data_csv = f.name

    result_csv = tempfile.mktemp(suffix="_lmm_results.csv", dir=tempfile.gettempdir())

    r_code = _build_r_lmm_script(
        data_csv.replace("\\", "/"),
        result_csv.replace("\\", "/"),
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(r_code)
        r_script = f.name

    try:
        rscript_cmd, r_env = resolve_r_runtime()
        print(f"  Using Rscript: {rscript_cmd}")
        proc = subprocess.run([rscript_cmd, r_script], capture_output=True, text=True, timeout=900, env=r_env)
        if proc.returncode != 0:
            print(f"  R stderr: {proc.stderr[:1500]}")
            raise RuntimeError(f"Rscript failed with return code {proc.returncode}")
        if proc.stdout:
            print(f"  R: {proc.stdout.strip()}")

        results = pd.read_csv(result_csv)
    finally:
        for p in [data_csv, result_csv, r_script]:
            try:
                os.unlink(p)
            except OSError:
                pass

    results["family"] = results["dv"].map(dv_to_family)

    results["p_fdr"] = np.nan
    p_mask = results["p_value"].notna()
    if p_mask.sum() > 0:
        _, p_fdr, _, _ = multipletests(results.loc[p_mask, "p_value"].values, method="fdr_bh")
        results.loc[p_mask, "p_fdr"] = p_fdr

    def _sig(p: float | Any) -> str:
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    results["sig"] = results["p_fdr"].apply(_sig)
    return results


def print_significant_only(results: pd.DataFrame) -> None:
    sig_df = results[results["sig"] != ""].copy().sort_values("p_fdr")

    print("\n" + "=" * 96)
    print("Significant Variables Only (BH-FDR < 0.05)")
    print("=" * 96)

    if sig_df.empty:
        print("No significant variables under BH-FDR < 0.05.")
        return

    fmt = "{:<28s} {:<20s} {:>10s} {:>5s}"
    print(fmt.format("DV", "Family", "Estimate", "Sig"))
    print("-" * 96)
    for _, row in sig_df.iterrows():
        print(
            fmt.format(
                str(row["dv"])[:28],
                str(row["family"])[:20],
                f"{row['estimate']:.2f}" if pd.notna(row["estimate"]) else "NA",
                str(row["sig"]),
            )
        )
    print("-" * 96)


def _fmt_num(v: Any, digits: int = 2) -> str:
    if pd.isna(v):
        return "NA"
    return f"{float(v):.{digits}f}"


def _fmt_mean_sd(mean_v: Any, sd_v: Any, digits: int = 2) -> str:
    return f"{_fmt_num(mean_v, digits)}±{_fmt_num(sd_v, digits)}"


def _result_status_map(results: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in results.itertuples(index=False):
        status = str(getattr(row, "analysis_status", "ok"))
        if status != "ok":
            out[str(row.dv)] = status
            continue
        sig = str(row.sig) if not pd.isna(row.sig) else ""
        out[str(row.dv)] = sig if sig else "n.s."
    return out


def _build_analyzed_variables_table(
    specs: list[dict[str, str]],
    audit_df: pd.DataFrame,
    result_status: dict[str, str],
) -> str:
    audit_lookup = audit_df.set_index("dv").to_dict("index")
    lines = [
        "| Variable | Family | Testability at onset | Result status |",
        "|---|---|---|---|",
    ]
    for spec in specs:
        dv = spec["dv"]
        family = spec["family"]
        status = str(audit_lookup[dv]["status"])
        result = result_status.get(dv, "untestable")
        lines.append(f"| `{dv}` | {family} | {status} | {result} |")
    return "\n".join(lines)


def _build_significant_table(results: pd.DataFrame) -> str:
    sig_df = results[results["sig"] != ""].copy().sort_values("p_fdr")
    lines = [
        "| Variable | Family | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in sig_df.itertuples(index=False):
        lines.append(
            "| "
            f"`{row.dv}` | {row.family} | "
            f"{_fmt_mean_sd(row.mean_step, row.sd_step, 2)} | "
            f"{_fmt_mean_sd(row.mean_nonstep, row.sd_nonstep, 2)} | "
            f"{_fmt_num(row.estimate, 2)} | {row.sig} |"
        )
    if sig_df.empty:
        lines.append("| (none) | - | - | - | - | - |")
    return "\n".join(lines)


def _build_joint_angle_table(results: pd.DataFrame, dvs: list[str]) -> str:
    lookup = {str(row.dv): row for row in results.itertuples(index=False)}
    lines = [
        "| Variable | Step (M±SD) | Nonstep (M±SD) | Estimate (step−nonstep) | Sig |",
        "|---|---:|---:|---:|---|",
    ]
    for dv in dvs:
        row = lookup.get(dv)
        if row is None:
            lines.append(f"| `{dv}` | NA±NA | NA±NA | NA | untestable |")
            continue
        sig = row.sig if isinstance(row.sig, str) and row.sig else "n.s."
        if str(getattr(row, "analysis_status", "ok")) != "ok":
            sig = str(getattr(row, "analysis_status"))
        lines.append(
            "| "
            f"`{dv}` | "
            f"{_fmt_mean_sd(row.mean_step, row.sd_step, 2)} | "
            f"{_fmt_mean_sd(row.mean_nonstep, row.sd_nonstep, 2)} | "
            f"{_fmt_num(row.estimate, 2)} | {sig} |"
        )
    return "\n".join(lines)


def _build_outlier_table(results: pd.DataFrame) -> str:
    outlier_df = results[
        ["dv", "n_step_raw", "n_step_outliers", "n_step", "n_nonstep_raw", "n_nonstep_outliers", "n_nonstep"]
    ].copy()
    outlier_df = outlier_df.sort_values("dv")
    lines = [
        "| Variable | Step raw | Step outliers | Step kept | Nonstep raw | Nonstep outliers | Nonstep kept |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in outlier_df.itertuples(index=False):
        lines.append(
            f"| `{row.dv}` | {int(row.n_step_raw)} | {int(row.n_step_outliers)} | {int(row.n_step)} | "
            f"{int(row.n_nonstep_raw)} | {int(row.n_nonstep_outliers)} | {int(row.n_nonstep)} |"
        )
    return "\n".join(lines)


def _joint_angle_significance(results: pd.DataFrame, joint_dvs: list[str]) -> tuple[int, int, list[str]]:
    sub = results[results["dv"].isin(joint_dvs)].copy()
    sub["sig"] = sub["sig"].fillna("")
    sig_sub = sub[sub["sig"] != ""].copy()
    sig_names = sig_sub.sort_values("p_fdr")["dv"].tolist()
    return len(joint_dvs), len(sig_sub), sig_names


def write_report_markdown(
    report_md: Path,
    specs: list[dict[str, str]],
    audit_df: pd.DataFrame,
    results: pd.DataFrame,
    step_specs: list[dict[str, str]],
    step_audit_df: pd.DataFrame,
    step_results: pd.DataFrame,
    step_onset_stats: dict[str, Any],
    major_side: pd.DataFrame,
    n_trials: int,
    n_step: int,
    n_nonstep: int,
    n_subjects: int,
    qc_strict: bool,
) -> None:
    n_testable = int((audit_df["status"] == "testable").sum())
    n_untestable = int((audit_df["status"] == "untestable").sum())
    n_sig = int((results["sig"] != "").sum())
    ratio = f"{n_sig}/{len(results)}"
    verdict = "PASS" if (len(results) > 0 and n_sig == len(results) and n_untestable == 0) else "FAIL"
    qc_mode = "strict" if qc_strict else "non-strict"
    major_summary = summarize_major_step_side(major_side)
    tie_subjects = major_summary["tie_subjects"]
    tie_subjects_str = ", ".join(tie_subjects) if tie_subjects else "(none)"
    joint_total, joint_sig_count, joint_sig_names = _joint_angle_significance(
        results,
        _joint_angle_abs_cols(),
    )
    if joint_sig_count == 0:
        joint_line = (
            f"관절 각도 변수(`Hip/Knee/Ankle`의 stance `X/Y/Z` + `Trunk/Neck`의 `X/Y/Z`, 총 {joint_total}개)는 "
            "`n.s.`였고"
        )
    else:
        joint_line = (
            f"관절 각도 변수는 총 {joint_total}개 중 {joint_sig_count}개가 유의했고 "
            f"(`{', '.join(joint_sig_names)}`), 나머지는 `n.s.`였다."
        )
    result_status = _result_status_map(results)
    analyzed_table = _build_analyzed_variables_table(specs, audit_df, result_status)
    significant_table = _build_significant_table(results)
    outlier_table = _build_outlier_table(results)
    step_result_status = _result_status_map(step_results)
    step_analyzed_table = _build_analyzed_variables_table(step_specs, step_audit_df, step_result_status)
    step_significant_table = _build_significant_table(step_results)
    step_outlier_table = _build_outlier_table(step_results)
    step_joint_total, step_joint_sig_count, step_joint_sig_names = _joint_angle_significance(
        step_results,
        _joint_angle_step_cols(),
    )
    step_sig_total = int((step_results["sig"] != "").sum())
    step_sig_ratio = f"{step_sig_total}/{len(step_results)}"
    step_missing_ref_subjects = step_onset_stats.get("missing_ref_subjects", [])
    step_missing_ref_subjects_str = ", ".join(step_missing_ref_subjects) if step_missing_ref_subjects else "(none)"

    report_text = f"""# Initial Posture Strategy LMM (Single-Frame Comparison)

## Research Question

**"Van Wouwe et al. (2021) 관점에서, single-frame posture 지표가 step/nonstep 전략 차이를 설명한다면 platform onset과 step onset 중 어떤 시점에서 분화가 더 뚜렷한가?"**

이번 보고서는 `platform_onset_local`과 `step_onset_target_local`의 **단일 프레임 LMM** 결과를 다룬다. baseline range mean 결과는 별도 문서인 `report_baseline.md`에서 다루며, 본 문서에는 `95% CI`를 포함하지 않는다.

## Prior Studies

### Van Wouwe et al. (2021) — Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance

- **Methodology**: 예측 시뮬레이션 + 실험 데이터 결합. 초기 자세(COM 위치)와 task-level goal(노력-안정성 우선순위)의 상호작용으로 전략 variability를 설명.
- **Experimental design**: 10명의 젊은 성인, 예측 불가능한 backward support-surface translation, stepping/nonstepping 반응 기록.
- **Key results**:
  - 최대 trunk lean 변동성: within-subject mean range 약 `28.3°`, across-subject mean range 약 `39.9°`
  - initial COM position과 maximal trunk lean 관계는 subject-specific (`R^2 = 0.29–0.82`)
  - `xCOM/BOS_onset`, `xCOM/BOS_300ms`를 안정성 지표로 사용
- **Conclusions**: 초기 자세는 intra-subject variability에, task-level goal은 inter-subject 차이에 기여하며, 두 요인 상호작용이 전략 차이를 설명.

## Data Summary

- Trials: **{n_trials}** (`step={n_step}`, `nonstep={n_nonstep}`), subjects={n_subjects}
- Input:
  - `output/all_trials_timeseries.csv`
  - `data/perturb_inform.xlsm`
  - `data/all_data/*.c3d`
  - `src/replace_v3d/torque/assets/fp_inertial_templates.npz`
- 분석 변수:
  - onset 후보 총 **{len(specs)}개**
  - 검정 가능(testable) **{n_testable}개**
  - 검정 불가(untestable) **{n_untestable}개**
- Force inertial QC mode: **{qc_mode}**

## Analysis Methodology

이 보고서의 질문은 onset 전 평균 posture가 아니라 **특정 단일 프레임에서 step/nonstep 차이가 얼마나 드러나는지**를 보는 것이다. 따라서 baseline 평균 분석과는 목적이 다르며, `report_baseline.md`와 직접 같은 질문으로 읽으면 안 된다.

- **Analysis point**: `platform_onset_local` 단일 프레임
- **Statistical model**: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`)
- **Outlier rule**: 각 변수별 `step/nonstep` 그룹 내부에서 `1.5×IQR` 밖 trial 제거
- **Confidence interval policy**: 본 단일 프레임 보고서에서는 `Estimate`와 `Sig`만 보고하고, `95% CI`는 baseline range mean 보고서에서만 제시
- **Multiple comparison correction**: BH-FDR ({len(specs)}개 onset 변수 전체 1회)
- **Significance reporting**: `Sig` only (`***`, `**`, `*`, `n.s.`), `alpha=0.05`
- **Displayed result policy**: Results 표에는 **FDR 유의 변수만** 표시

### Coordinate Definition (Joint Angle)

- Joint angle는 `compute_v3d_joint_angles_3d` 기준의 **intrinsic XYZ Euler sequence**를 사용한다.
- Segment 좌표계는 전역 기준으로 `X=+Right`, `Y=+Anterior`, `Z=+Up/+Proximal`로 구성된다.
- 따라서 `*_X/*_Y/*_Z`는 각각 해당 축 회전 성분이며, 단순히 sagittal/frontal/transverse와 1:1로 고정 해석하면 안 된다.

### Stance-Leg Selection Rule

- `step_r` trial: left leg angle을 stance로 사용
- `step_l` trial: right leg angle을 stance로 사용
- `nonstep` trial: 해당 subject의 step trial 분포(`major_step_side`)로 stance를 선택
- `tie` (`step_r_count == step_l_count`): left/right 평균값 사용
- Subject summary: `step_r_major={major_summary["step_r_major"]}`, `step_l_major={major_summary["step_l_major"]}`, `tie={major_summary["tie_major"]}`
- Tie subjects: `{tie_subjects_str}`

### Analyzed Variables (Full Set, n={len(specs)})

{analyzed_table}

## Results

### Hypothesis Verdict (strict)

- **Rule**: testable onset 변수 전부가 FDR 유의여야 PASS
- **Observed**: testable significant ratio = `{ratio}`, untestable=`{n_untestable}`
- **Verdict**: **{verdict}**

### Significant Variables Only (BH-FDR < 0.05)

{significant_table}

### Outlier Exclusion Summary (Platform Onset)

{outlier_table}

## Step-Onset Single-Frame Analysis

- **Analysis point**: `step_onset_target_local` 단일 프레임
- **Target rule**:
  - step trial: 해당 trial의 `step_onset_local` 사용
  - nonstep trial: 동일 subject의 step trial `step_onset_local` 평균값을 대입한 후 frame으로 반올림
- **Valid trials**: `{step_onset_stats["n_used"]}/{step_onset_stats["n_trials_total"]}` (step=`{step_onset_stats["n_step_used"]}`, nonstep=`{step_onset_stats["n_nonstep_used"]}`)
- **Excluded trials**: `{step_onset_stats["n_excluded"]}` (step_onset 결측 step=`{step_onset_stats["n_step_missing_direct"]}`, step 참조 부재 nonstep=`{step_onset_stats["n_nonstep_missing_ref"]}`, frame 불일치=`{step_onset_stats["n_target_no_frame"]}`)
- **nonstep step_onset 참조 부재 subject**: `{step_missing_ref_subjects_str}`
- **Observed**: testable significant ratio = `{step_sig_ratio}`

### Step-Onset Variables (Full Set, n={len(step_specs)})

{step_analyzed_table}

### Significant Step-Onset Variables Only (BH-FDR < 0.05)

{step_significant_table}

### Outlier Exclusion Summary (Step Onset)

{step_outlier_table}

## Interpretation & Conclusion

1. platform onset 단일 프레임에서는 총 `{len(specs)}`개 변수 중 `{n_sig}`개만 유의해 strict 기준 가설은 **{verdict}**였다. 즉, 섭동 직후 posture snapshot만으로 전략 분화를 광범위하게 설명하기는 어려웠다.
2. platform onset에서는 {joint_line} 나머지 유의 변수도 balance, joint velocity/moment, force/torque 영역의 일부 변수에 제한적으로 나타났다.
3. step onset 단일 프레임에서는 총 `{step_sig_total}`개가 FDR 유의였고, joint-angle 15개 중 `{step_joint_sig_count}`개가 유의했다 (`{', '.join(step_joint_sig_names) if step_joint_sig_names else '(none)'}`). 본 데이터에서는 전략 분화가 섭동 직후보다 발 들기 직전 프레임에서 더 강하게 관찰됐다.
4. 따라서 single-frame 비교만 놓고 보면, step/nonstep 전략 차이는 `platform onset`의 초기 snapshot보다 `step onset` 직전의 준비 자세에서 더 뚜렷하다. 다만 `platform onset` {len(specs)}개 변수와 `step onset` {len(step_specs)}개 변수 전체가 일관되게 유의하지는 않으므로, 단일 프레임 변수만으로 전략 차이를 완전히 설명한다고 단정할 수는 없다.

## Limitations

1. 원문의 task-level goal 파라미터를 직접 모델링하지 않았다.
2. 본 분석은 Van Wouwe 2021의 simulation 기반 인과 프레임을 1:1 재현한 결과가 아니다.
3. step onset은 nonstep trial에서 subject 평균 `step_onset_local`을 참조하므로, step trial의 실제 발 들기 순간과 완전히 같은 관측점은 아니다.
4. 본 문서는 single-frame 분석만 다루며, onset 전 구간 평균과 `95% CI` 해석은 `report_baseline.md`를 따로 봐야 한다.

## Reproduction

```bash
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py --dry-run
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py
```

- Output: 콘솔 통계 결과 + 자동 갱신 `report.md`, `결과) 주제2-Segement Angle.md`

---
Auto-generated by analyze_initial_posture_strategy_lmm.py.
"""

    report_md.write_text(report_text, encoding="utf-8-sig")


def write_segment_angle_markdown(
    segment_md: Path,
    results: pd.DataFrame,
    step_results: pd.DataFrame,
    major_side: pd.DataFrame,
    step_onset_stats: dict[str, Any],
    verdict_pass: bool,
) -> None:
    """Write segment-angle results markdown while preserving the manual interpretation body.

    The top tables + metadata are refreshed on each run, but the detailed narrative under
    '# 결과 해석' is user-edited and should not be destroyed by re-runs.
    """
    platform_dvs = _joint_angle_abs_cols()
    step_dvs = _joint_angle_step_cols()
    table = _build_joint_angle_table(results, platform_dvs)
    step_table = _build_joint_angle_table(step_results, step_dvs)
    verdict = "PASS" if verdict_pass else "FAIL"
    major_summary = summarize_major_step_side(major_side)
    tie_subjects = major_summary["tie_subjects"]
    tie_subjects_str = ", ".join(tie_subjects) if tie_subjects else "(none)"
    joint_total, joint_sig_count, joint_sig_names = _joint_angle_significance(results, platform_dvs)
    step_joint_total, step_joint_sig_count, step_joint_sig_names = _joint_angle_significance(step_results, step_dvs)

    if joint_sig_count == 0:
        joint_note = f"{joint_total}개 segment angle 변수(X/Y/Z) 모두 FDR 보정 후 `n.s.`였다."
    else:
        joint_note = (
            f"{joint_total}개 segment angle 변수(X/Y/Z) 중 {joint_sig_count}개가 FDR 유의였다: "
            f"`{', '.join(joint_sig_names)}`."
        )
    if step_joint_sig_count == 0:
        step_joint_note = f"{step_joint_total}개 step_onset segment angle 변수(X/Y/Z) 모두 FDR 보정 후 `n.s.`였다."
    else:
        step_joint_note = (
            f"{step_joint_total}개 step_onset segment angle 변수(X/Y/Z) 중 {step_joint_sig_count}개가 FDR 유의였다: "
            f"`{', '.join(step_joint_sig_names)}`."
        )

    missing_ref_subjects = step_onset_stats.get("missing_ref_subjects", [])
    missing_ref_subjects_str = ", ".join(missing_ref_subjects) if missing_ref_subjects else "(none)"

    interpretation_body = f"""# 결과 해석

## platform_onset 해석

- platform onset joint-angle 15개 변수 중 `{joint_sig_count}`개가 FDR 유의였다: `{', '.join(joint_sig_names) if joint_sig_names else '(none)'}`.
- 이 시점은 baseline 평균이 아니라 섭동 직후의 posture snapshot에 가깝다.
- 따라서 platform onset에서는 지지다리 및 체간 정렬 차이가 일부 축에서만 관찰되며, 전략 분화의 출발점이라기보다 제한적인 초기 반응 차이로 해석하는 편이 안전하다.

## step_onset 해석

- step onset joint-angle 15개 변수 중 `{step_joint_sig_count}`개가 FDR 유의였다: `{', '.join(step_joint_sig_names) if step_joint_sig_names else '(none)'}`.
- step onset은 `step` trial의 실제 `step_onset_local`과, `nonstep` trial의 subject 평균 step onset 참조 frame을 사용한다.
- 따라서 이 결과는 평균적인 초기 자세라기보다 실제 발 들기 직전의 준비 자세 또는 전략 실행 직전 posture 차이에 더 가깝다.

## 종합 해석

- 같은 단일 프레임 이상치 제외 규칙에서 보면, platform onset보다 step onset에서 유의한 joint-angle 차이가 더 많이 관찰된다.
- 즉, 전략 차이는 섭동 직후 정적 snapshot보다 실제 발 들기 직전 single-frame에서 더 뚜렷하게 나타나는 경향이 있다.
- 다만 두 시점 모두 전축이 일관되게 유의하지는 않으므로, 관절각만으로 step/nonstep 전략 차이를 완전히 설명한다고 단정하기는 어렵다.
- onset 전 구간 평균과 `95% CI` 해석은 baseline 보고서와 분리해서 읽어야 한다.
"""

    text = f"""---
---
# 가설

1. initial phase에서 nonstep과 step의 관절 각도는 차이가 있을 것이다.

# results

## platform_onset 단일시점 LMM

{table}

## step_onset 단일시점 LMM

{step_table}

## coordinate 해석 기준

- 관절각 계산은 Visual3D-like intrinsic `XYZ` 순서를 사용한다.
- Segment 좌표계 기준은 `X=+Right`, `Y=+Anterior`, `Z=+Up/+Proximal`이다.
- Hip/Knee/Ankle의 `Y/Z`는 좌우(L/R) 해석 일관성을 위해 **LEFT side 값을 부호 반전**하여 RIGHT 의미와 통일한다.
- 따라서 `X/Y/Z`는 각 축 회전 성분이며, 임상적 평면(sagittal/frontal/transverse)과 완전한 1:1 대응으로 단정하지 않는다.

## stance 기준

- step trial은 `step_r -> 좌측 stance`, `step_l -> 우측 stance`로 계산한다.
- nonstep trial은 subject별 step trial의 `major_step_side`를 stance 기준으로 사용한다.
- `step_r_count == step_l_count`인 tie subject는 좌/우 평균으로 계산한다.
- 이번 실행 요약: `step_r_major={major_summary["step_r_major"]}`, `step_l_major={major_summary["step_l_major"]}`, `tie={major_summary["tie_major"]}` (tie subjects: `{tie_subjects_str}`)

- step_onset 비교 규칙:
  - step trial: 해당 trial의 `step_onset_local` 사용
  - nonstep trial: 동일 subject의 step trial `step_onset_local` 평균값을 대입한 후 frame으로 반올림
  - step_onset 기준 유효 trial: `{step_onset_stats["n_used"]}/{step_onset_stats["n_trials_total"]}` (step=`{step_onset_stats["n_step_used"]}`, nonstep=`{step_onset_stats["n_nonstep_used"]}`)
  - 제외 trial: `{step_onset_stats["n_excluded"]}` (step_onset 결측 step=`{step_onset_stats["n_step_missing_direct"]}`, step 참조 부재 nonstep=`{step_onset_stats["n_nonstep_missing_ref"]}`, frame 불일치=`{step_onset_stats["n_target_no_frame"]}`)
  - nonstep step_onset 참조 부재 subject: `{missing_ref_subjects_str}`

- 해석 노트:
  - platform_onset: {joint_note}
  - step_onset: {step_joint_note}
  - 두 시점 모두에서 `Estimate`와 `Sig`는 변수별 `step/nonstep` 그룹 내부 `1.5×IQR` 이상치 제외 후 계산했다.
  - `95% CI`는 single-frame 보고서에 포함하지 않으며, baseline range mean 보고서에서만 제시한다.
  - 두 시점 모두에서 전축이 일관되게 유의하지 않다면, 관절각만으로 전략 차이를 설명하는 근거는 제한적이다.

{interpretation_body}

# 결론

- 가설 1 결과: **{verdict}**
- single-frame 비교에서는 step onset이 platform onset보다 더 강한 분화를 보였지만, 관절각만으로 전략 차이를 단정하기에는 근거가 제한적이다.

# keypapers

1. Van Wouwe et al. (2021): 초기 자세와 전략 variability의 상호작용을 제시했으며, 본 문서는 그 질문을 두 개의 single-frame 시점으로 나누어 비교한다.

---
Auto-generated by analyze_initial_posture_strategy_lmm.py.
"""
    segment_md.write_text(text, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    _ = args.no_figures  # compatibility arg; intentionally unused

    print("=" * 72)
    print("Initial Posture Strategy LMM")
    print("=" * 72)

    print("\n[M1] Load and prepare data...")
    analysis_df, trial_meta, _abs_angles, major_side, step_onset_df, step_onset_stats = build_analysis_dataframe(
        csv_path=args.csv,
        platform_xlsm=args.platform_xlsm,
        c3d_dir=args.c3d_dir,
        fp_inertial_templates_path=args.fp_inertial_templates,
        fp_inertial_policy=args.fp_inertial_policy,
        fp_inertial_qc_fz_threshold=args.fp_inertial_qc_fz_threshold,
        fp_inertial_qc_margin_m=args.fp_inertial_qc_margin_m,
        fp_inertial_qc_strict=args.fp_inertial_qc_strict,
    )

    specs = variable_catalog()
    dv_to_family = {s["dv"]: s["family"] for s in specs}
    step_specs = _step_onset_variable_catalog()
    step_dv_to_family = {s["dv"]: s["family"] for s in step_specs}

    print("\n[M2] Variable audit at platform onset...")
    audit_df = audit_variables(analysis_df=analysis_df, specs=specs)
    n_total = len(audit_df)
    n_testable = int((audit_df["status"] == "testable").sum())
    n_untestable = int((audit_df["status"] == "untestable").sum())
    print(f"  Variables total={n_total}, testable={n_testable}, untestable={n_untestable}")

    untestable = audit_df[audit_df["status"] == "untestable"].copy()
    if not untestable.empty:
        by_reason = untestable.groupby("reason").size().to_dict()
        print(f"  Untestable reasons: {by_reason}")

    n_trials = len(trial_meta)
    n_step = int((trial_meta["step_TF"] == "step").sum())
    n_nonstep = int((trial_meta["step_TF"] == "nonstep").sum())
    n_subjects = int(trial_meta["subject"].nunique())
    print(f"  Trial set fixed: {n_trials} (step={n_step}, nonstep={n_nonstep})")
    major_summary = summarize_major_step_side(major_side)
    print(
        "  major_step_side summary: "
        f"subjects={major_summary['subjects']}, "
        f"step_r_major={major_summary['step_r_major']}, "
        f"step_l_major={major_summary['step_l_major']}, "
        f"tie={major_summary['tie_major']}"
    )
    if major_summary["tie_subjects"]:
        print(f"  tie subjects (L/R mean): {', '.join(major_summary['tie_subjects'])}")
    print(
        "  step_onset trial usage: "
        f"used={step_onset_stats['n_used']}/{step_onset_stats['n_trials_total']} "
        f"(step={step_onset_stats['n_step_used']}, nonstep={step_onset_stats['n_nonstep_used']})"
    )
    print(
        "  step_onset exclusions: "
        f"step_missing={step_onset_stats['n_step_missing_direct']}, "
        f"nonstep_missing_ref={step_onset_stats['n_nonstep_missing_ref']}, "
        f"frame_mismatch={step_onset_stats['n_target_no_frame']}"
    )
    if step_onset_stats["missing_ref_subjects"]:
        print(
            "  nonstep subjects without step_onset reference: "
            + ", ".join(step_onset_stats["missing_ref_subjects"])
        )
    onset_outlier_summary = compute_outlier_summary(analysis_df=analysis_df, specs=specs)
    step_outlier_summary = compute_outlier_summary(analysis_df=step_onset_df, specs=step_specs)
    onset_total_outliers = int(onset_outlier_summary["n_step_outliers"].sum() + onset_outlier_summary["n_nonstep_outliers"].sum())
    step_total_outliers = int(step_outlier_summary["n_step_outliers"].sum() + step_outlier_summary["n_nonstep_outliers"].sum())
    print(f"  platform_onset outlier candidates across variables: {onset_total_outliers}")
    print(f"  step_onset outlier candidates across variables: {step_total_outliers}")

    if args.dry_run:
        print("\nDry run complete.")
        return

    print("\n[M3] Fit LMM (testable variables only)...")
    testable_vars = audit_df.loc[audit_df["status"] == "testable", "dv"].tolist()
    results = fit_lmm_all(analysis_df=analysis_df, testable_vars=testable_vars, dv_to_family=dv_to_family)

    print_significant_only(results)

    print("\n[M3b] Fit step_onset single-frame LMM...")
    step_audit_df = audit_variables(analysis_df=step_onset_df, specs=step_specs)
    step_testable_vars = step_audit_df.loc[step_audit_df["status"] == "testable", "dv"].tolist()
    step_results = fit_lmm_all(
        analysis_df=step_onset_df,
        testable_vars=step_testable_vars,
        dv_to_family=step_dv_to_family,
    )

    n_sig = int((results["sig"] != "").sum())
    n_testable_modeled = len(results)
    has_untestable = n_untestable > 0
    all_testable_sig = (n_testable_modeled > 0) and (n_sig == n_testable_modeled)
    verdict_pass = bool(all_testable_sig and (not has_untestable))

    print("\n" + "=" * 72)
    print("Hypothesis Verdict (strict rule)")
    print("=" * 72)
    print(f"Testable significant ratio: {n_sig}/{n_testable_modeled}")
    print(f"Untestable variable count: {n_untestable}")
    print(f"VERDICT: {'PASS' if verdict_pass else 'FAIL'}")

    if n_sig > 0:
        sig_names = results.loc[results["sig"] != "", "dv"].tolist()
        print(f"Significant variables ({len(sig_names)}): {', '.join(sig_names)}")
    else:
        print("Significant variables (0): none")

    print("\n[M4] Refresh markdown outputs...")
    write_report_markdown(
        report_md=args.report_md,
        specs=specs,
        audit_df=audit_df,
        results=results,
        step_specs=step_specs,
        step_audit_df=step_audit_df,
        step_results=step_results,
        step_onset_stats=step_onset_stats,
        major_side=major_side,
        n_trials=n_trials,
        n_step=n_step,
        n_nonstep=n_nonstep,
        n_subjects=n_subjects,
        qc_strict=bool(args.fp_inertial_qc_strict),
    )
    write_segment_angle_markdown(
        segment_md=args.segment_angle_md,
        results=results,
        step_results=step_results,
        major_side=major_side,
        step_onset_stats=step_onset_stats,
        verdict_pass=verdict_pass,
    )
    print(f"  Updated: {args.report_md}")
    print(f"  Updated: {args.segment_angle_md}")


if __name__ == "__main__":
    main()
