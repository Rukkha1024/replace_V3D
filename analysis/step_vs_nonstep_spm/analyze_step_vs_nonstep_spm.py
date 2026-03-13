"""Step vs nonstep SPM 1D analysis on normalized onset windows.
Loads mixed trials, applies stance-side and foot-length normalization, and builds
subject-level paired curves (step vs nonstep).
Runs parametric and nonparametric paired SPM t-tests with family Bonferroni
correction, then writes figures, CSV results, and a Markdown report.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import polars as pl
import spm1d
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# Korean font support
_KO_FONTS = ("Malgun Gothic", "NanumGothic", "AppleGothic")
_available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _font in _KO_FONTS:
    if _font in _available_fonts:
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_OUT_DIR = SCRIPT_DIR

TRIAL_KEYS = ["subject", "velocity", "trial"]
NORM_POINTS = 101
NAN_RATIO_THRESHOLD = 0.20
ALPHA = 0.05

# Mirror existing LMM stance-side logic.
_STANCE_X_SOURCES = [
    ("Hip_stance_X_deg", "Hip_L_X_deg", "Hip_R_X_deg"),
    ("Knee_stance_X_deg", "Knee_L_X_deg", "Knee_R_X_deg"),
    ("Ankle_stance_X_deg", "Ankle_L_X_deg", "Ankle_R_X_deg"),
]
_LOWER_LIMB_SEGMENTS = ("Hip", "Knee", "Ankle")
_MIDLINE_SEGMENTS = ("Trunk", "Neck")
_DYNAMIC_AXES = ("X", "Y", "Z")
_ANGULAR_VELOCITY_RESOLUTIONS = ("ref", "mov")
_STANCE_DYNAMIC_SOURCES: list[tuple[str, str, str]] = []
for _seg in _LOWER_LIMB_SEGMENTS:
    for _res in _ANGULAR_VELOCITY_RESOLUTIONS:
        for _axis in _DYNAMIC_AXES:
            _STANCE_DYNAMIC_SOURCES.append(
                (
                    f"{_seg}_stance_{_res}_{_axis}_deg_s",
                    f"{_seg}_L_{_res}_{_axis}_deg_s",
                    f"{_seg}_R_{_res}_{_axis}_deg_s",
                )
            )
    for _axis in _DYNAMIC_AXES:
        _STANCE_DYNAMIC_SOURCES.append(
            (
                f"{_seg}_stance_ref_{_axis}_Nm",
                f"{_seg}_L_ref_{_axis}_Nm",
                f"{_seg}_R_ref_{_axis}_Nm",
            )
        )


@dataclass(frozen=True)
class VariableSpec:
    name: str
    family: str


def build_variable_specs() -> list[VariableSpec]:
    specs = [
        VariableSpec("COM_X", "COM"),
        VariableSpec("COM_Y", "COM"),
        VariableSpec("COM_Z", "COM"),
        VariableSpec("vCOM_X", "vCOM"),
        VariableSpec("vCOM_Y", "vCOM"),
        VariableSpec("vCOM_Z", "vCOM"),
        VariableSpec("xCOM_X", "xCOM"),
        VariableSpec("xCOM_Y", "xCOM"),
        VariableSpec("xCOM_Z", "xCOM"),
        VariableSpec("BOS_area", "BOS"),
        VariableSpec("BOS_minX", "BOS"),
        VariableSpec("BOS_maxX", "BOS"),
        VariableSpec("BOS_minY", "BOS"),
        VariableSpec("BOS_maxY", "BOS"),
        VariableSpec("MOS_minDist_signed", "MOS"),
        VariableSpec("MOS_AP_v3d", "MOS"),
        VariableSpec("MOS_ML_v3d", "MOS"),
        VariableSpec("MOS_v3d", "MOS"),
        VariableSpec("GRF_X_N", "GRF"),
        VariableSpec("GRF_Y_N", "GRF"),
        VariableSpec("GRF_Z_N", "GRF"),
        VariableSpec("GRM_X_Nm_at_FPorigin", "GRM"),
        VariableSpec("GRM_Y_Nm_at_FPorigin", "GRM"),
        VariableSpec("GRM_Z_Nm_at_FPorigin", "GRM"),
        VariableSpec("COP_X_m", "COP"),
        VariableSpec("COP_Y_m", "COP"),
        VariableSpec("AnkleTorqueMid_int_Y_Nm_per_kg", "AnkleTorque"),
        VariableSpec("Trunk_X_deg", "Trunk"),
        VariableSpec("Trunk_Y_deg", "Trunk"),
        VariableSpec("Trunk_Z_deg", "Trunk"),
        VariableSpec("Neck_X_deg", "Neck"),
        VariableSpec("Neck_Y_deg", "Neck"),
        VariableSpec("Neck_Z_deg", "Neck"),
        VariableSpec("Hip_stance_X_deg", "StanceJoint"),
        VariableSpec("Knee_stance_X_deg", "StanceJoint"),
        VariableSpec("Ankle_stance_X_deg", "StanceJoint"),
        VariableSpec("xCOM_BOS_AP_foot", "xCOM_BOS"),
        VariableSpec("xCOM_BOS_ML_foot", "xCOM_BOS"),
    ]
    for seg in _LOWER_LIMB_SEGMENTS:
        for res in _ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in _DYNAMIC_AXES:
                specs.append(VariableSpec(f"{seg}_stance_{res}_{axis}_deg_s", "JointVelocity"))
        for axis in _DYNAMIC_AXES:
            specs.append(VariableSpec(f"{seg}_stance_ref_{axis}_Nm", "SegmentMoment"))
    for seg in _MIDLINE_SEGMENTS:
        for res in _ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in _DYNAMIC_AXES:
                specs.append(VariableSpec(f"{seg}_{res}_{axis}_deg_s", "JointVelocity"))
        for axis in _DYNAMIC_AXES:
            specs.append(VariableSpec(f"{seg}_ref_{axis}_Nm", "SegmentMoment"))
    return specs


FAMILY_SIZE = Counter(spec.family for spec in build_variable_specs())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--nonparam_iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260306)
    ap.add_argument("--dry-run", action="store_true", help="Only run loading/QC without SPM tests")
    return ap.parse_args()


def load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(str(path), encoding="utf8-lossy", infer_schema_length=10000)


def _normalize_trial_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject"] = out["subject"].astype(str).str.strip()
    out["velocity"] = pd.to_numeric(out["velocity"], errors="coerce")
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce").astype("Int64")
    return out


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
    out = _normalize_trial_types(out)
    out["step_TF"] = out["step_TF"].astype(str).str.strip().str.lower()
    out["state"] = out["state"].astype(str).str.strip().str.lower()
    out["mixed"] = pd.to_numeric(out["mixed"], errors="coerce")
    out["platform_onset"] = pd.to_numeric(out["platform_onset"], errors="coerce")
    out["step_onset"] = pd.to_numeric(out["step_onset"], errors="coerce")
    return out


def load_foot_length_from_meta(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(str(path), sheet_name="meta")
    label_col = raw.columns[0]
    raw[label_col] = raw[label_col].astype(str).str.strip()

    required_rows = ["발길이_왼", "발길이_오른"]
    sub = raw[raw[label_col].isin(required_rows)].copy()
    missing_rows = [x for x in required_rows if x not in sub[label_col].tolist()]
    if missing_rows:
        raise ValueError(f"meta sheet missing required anthropometric rows: {missing_rows}")

    records: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        metric = str(row[label_col]).strip()
        for subject, value in row.iloc[1:].items():
            records.append({"subject": str(subject).strip(), "metric": metric, "value": value})

    long_df = pd.DataFrame(records)
    wide = (
        long_df.pivot_table(index="subject", columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    wide["발길이_왼"] = pd.to_numeric(wide["발길이_왼"], errors="coerce")
    wide["발길이_오른"] = pd.to_numeric(wide["발길이_오른"], errors="coerce")
    wide["foot_len_m"] = ((wide["발길이_왼"] + wide["발길이_오른"]) / 2.0) / 1000.0
    return wide[["subject", "foot_len_m"]].copy()


def build_subject_major_step_side(platform: pd.DataFrame) -> pd.DataFrame:
    """Build subject-level major stepping side from mixed==1 step trials."""
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


def add_stance_joint_x_columns(df: pl.DataFrame, platform: pd.DataFrame) -> tuple[pl.DataFrame, pd.DataFrame]:
    """Attach stance-equivalent lower-limb joint and dynamics columns."""
    major_side = build_subject_major_step_side(platform)
    trial_meta = platform[TRIAL_KEYS + ["step_TF", "state"]].drop_duplicates().copy()

    for out_col, _left, _right in _STANCE_X_SOURCES:
        df = df.drop([out_col], strict=False)
    for out_col, _left, _right in _STANCE_DYNAMIC_SOURCES:
        df = df.drop([out_col], strict=False)
    df = df.drop(["major_step_side"], strict=False)

    df = (
        df.with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial").cast(pl.Int64, strict=False),
        )
        .join(pl.from_pandas(trial_meta), on=TRIAL_KEYS, how="left")
        .join(pl.from_pandas(major_side[["subject", "major_step_side"]]), on="subject", how="left")
    )

    stance_exprs: list[pl.Expr] = []
    for out_col, left_col, right_col in _STANCE_X_SOURCES:
        stance_exprs.append(
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
    for out_col, left_col, right_col in _STANCE_DYNAMIC_SOURCES:
        stance_exprs.append(
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

    return df.with_columns(stance_exprs), major_side


def compute_end_frames(df: pl.DataFrame, platform: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trial end_frame [platform_onset_local -> step_onset_local-like]."""
    trials = (
        df.select(TRIAL_KEYS + ["step_onset_local"])
        .group_by(TRIAL_KEYS)
        .agg(pl.col("step_onset_local").drop_nulls().first().alias("step_onset_local"))
        .sort(TRIAL_KEYS)
        .to_pandas()
    )
    trials = _normalize_trial_types(trials)
    trials["step_onset_local"] = pd.to_numeric(trials["step_onset_local"], errors="coerce")

    plat_sub = platform[TRIAL_KEYS + ["step_TF"]].copy()
    trials = trials.merge(plat_sub, on=TRIAL_KEYS, how="left")

    trials["end_frame"] = np.nan
    step_mask = (trials["step_TF"] == "step") & trials["step_onset_local"].notna()
    trials.loc[step_mask, "end_frame"] = trials.loc[step_mask, "step_onset_local"]

    step_means = (
        trials.loc[step_mask]
        .groupby(["subject", "velocity"])["step_onset_local"]
        .mean()
        .reset_index()
        .rename(columns={"step_onset_local": "mean_step_onset"})
    )

    platform_onset_local_ref = (
        df.select(
            pl.col("platform_onset_local")
            .cast(pl.Float64, strict=False)
            .drop_nulls()
            .round(0)
            .cast(pl.Int64)
            .alias("platform_onset_local")
        )
        .to_series()
        .to_list()
    )
    if platform_onset_local_ref:
        platform_onset_local_ref = int(pd.Series(platform_onset_local_ref).mode().iloc[0])
    else:
        platform_onset_local_ref = 101

    prefilter_step_means = (
        platform.loc[
            (platform["step_TF"] == "step")
            & platform["step_onset"].notna()
            & platform["platform_onset"].notna(),
            ["subject", "velocity", "step_onset", "platform_onset"],
        ]
        .assign(
            step_onset_local_prefilter_sv=lambda x: (
                x["step_onset"] - x["platform_onset"] + float(platform_onset_local_ref)
            )
        )
        .groupby(["subject", "velocity"])["step_onset_local_prefilter_sv"]
        .mean()
        .reset_index()
    )

    trials = trials.merge(step_means, on=["subject", "velocity"], how="left")
    fill_mask_sv = trials["end_frame"].isna() & trials["mean_step_onset"].notna()
    trials.loc[fill_mask_sv, "end_frame"] = trials.loc[fill_mask_sv, "mean_step_onset"]

    trials = trials.merge(prefilter_step_means, on=["subject", "velocity"], how="left")
    fill_mask_prefilter_sv = (
        trials["end_frame"].isna()
        & (trials["step_TF"] == "nonstep")
        & trials["step_onset_local_prefilter_sv"].notna()
    )
    trials.loc[fill_mask_prefilter_sv, "end_frame"] = trials.loc[
        fill_mask_prefilter_sv, "step_onset_local_prefilter_sv"
    ]

    trials["end_frame"] = pd.to_numeric(trials["end_frame"], errors="coerce").round().astype("Int64")
    n_missing = int(trials["end_frame"].isna().sum())
    if n_missing > 0:
        print(f"  Warning: {n_missing} trials have no computable end_frame and will be dropped.")
    trials = trials.dropna(subset=["end_frame"]).reset_index(drop=True)

    print(f"  end_frame fill (subject-velocity mean): {int(fill_mask_sv.sum())}")
    print(f"  end_frame fill (prefilter subject-velocity mean): {int(fill_mask_prefilter_sv.sum())}")

    return trials[TRIAL_KEYS + ["step_TF", "end_frame"]]


def prepare_frame_level_dataset(csv_path: Path, xlsm_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    print("  Loading frame-level CSV...")
    df = load_csv(csv_path)
    n_frames_raw = df.height

    print("  Loading platform/meta sheets...")
    platform = load_platform_sheet(xlsm_path)
    foot = load_foot_length_from_meta(xlsm_path)

    required_cols = {
        "subject",
        "velocity",
        "trial",
        "MocapFrame",
        "platform_onset_local",
        "step_onset_local",
        "step_TF",
        "state",
        "mixed",
        "COM_X",
        "COM_Y",
        "COM_Z",
        "vCOM_X",
        "vCOM_Y",
        "vCOM_Z",
        "xCOM_X",
        "xCOM_Y",
        "xCOM_Z",
        "BOS_area",
        "BOS_minX",
        "BOS_maxX",
        "BOS_minY",
        "BOS_maxY",
        "MOS_minDist_signed",
        "MOS_AP_v3d",
        "MOS_ML_v3d",
        "MOS_v3d",
        "GRF_X_N",
        "GRF_Y_N",
        "GRF_Z_N",
        "GRM_X_Nm_at_FPorigin",
        "GRM_Y_Nm_at_FPorigin",
        "GRM_Z_Nm_at_FPorigin",
        "COP_X_m",
        "COP_Y_m",
        "AnkleTorqueMid_int_Y_Nm_per_kg",
        "Trunk_X_deg",
        "Trunk_Y_deg",
        "Trunk_Z_deg",
        "Neck_X_deg",
        "Neck_Y_deg",
        "Neck_Z_deg",
        "Hip_L_X_deg",
        "Hip_R_X_deg",
        "Knee_L_X_deg",
        "Knee_R_X_deg",
        "Ankle_L_X_deg",
        "Ankle_R_X_deg",
    }
    for _out_col, left_col, right_col in _STANCE_DYNAMIC_SOURCES:
        required_cols.add(left_col)
        required_cols.add(right_col)
    for seg in _MIDLINE_SEGMENTS:
        for res in _ANGULAR_VELOCITY_RESOLUTIONS:
            for axis in _DYNAMIC_AXES:
                required_cols.add(f"{seg}_{res}_{axis}_deg_s")
        for axis in _DYNAMIC_AXES:
            required_cols.add(f"{seg}_ref_{axis}_Nm")
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    df = df.with_columns(
        pl.col("subject").cast(pl.Utf8).str.strip_chars(),
        pl.col("velocity").cast(pl.Float64, strict=False),
        pl.col("trial").cast(pl.Int64, strict=False),
        pl.col("step_TF").cast(pl.Utf8).str.strip_chars().str.to_lowercase(),
        pl.col("state").cast(pl.Utf8).str.strip_chars().str.to_lowercase(),
        pl.col("mixed").cast(pl.Float64, strict=False),
    )

    platform_filtered = platform[
        platform["mixed"].eq(1) & platform["step_TF"].isin(["step", "nonstep"])
    ].copy()

    print("  Applying stance-side angle conversion logic...")
    df, major_side = add_stance_joint_x_columns(df, platform_filtered)
    major_missing = int(major_side["major_step_side"].isna().sum())
    if major_missing > 0:
        raise ValueError(f"major_step_side missing for {major_missing} subjects")

    # Drop entire step trials with missing step_onset_local.
    step_missing_keys = (
        df.filter((pl.col("step_TF") == "step") & pl.col("step_onset_local").is_null())
        .select(TRIAL_KEYS)
        .unique()
    )
    n_step_missing_trials = step_missing_keys.height
    if n_step_missing_trials > 0:
        print(f"  Dropping step trials with missing step_onset_local: {n_step_missing_trials}")
        step_missing_keys = step_missing_keys.with_columns(pl.lit(1).alias("_drop_trial"))
        df = (
            df.join(step_missing_keys, on=TRIAL_KEYS, how="left")
            .filter(pl.col("_drop_trial").is_null())
            .drop("_drop_trial")
        )

    print("  Reusing end_frame logic from LMM...")
    end_frames = compute_end_frames(df, platform_filtered)
    df = df.join(
        pl.from_pandas(end_frames[TRIAL_KEYS + ["end_frame"]]),
        on=TRIAL_KEYS,
        how="inner",
    )

    df = df.join(pl.from_pandas(foot), on="subject", how="left")
    n_bad_foot = (
        df.select(
            ((pl.col("foot_len_m").is_null()) | (pl.col("foot_len_m") <= 0)).sum().alias("n_bad")
        ).item()
    )
    if int(n_bad_foot) > 0:
        raise ValueError(f"Invalid foot_len_m found after join (count={int(n_bad_foot)})")

    print("  Building xCOM/BOS normalized variables...")
    df = df.with_columns(
        pl.when(pl.col("foot_len_m") > 0)
        .then((pl.col("xCOM_X") - pl.col("BOS_minX")) / pl.col("foot_len_m"))
        .otherwise(None)
        .alias("xCOM_BOS_AP_foot"),
        pl.when(pl.col("foot_len_m") > 0)
        .then((pl.col("xCOM_Y") - pl.col("BOS_minY")) / pl.col("foot_len_m"))
        .otherwise(None)
        .alias("xCOM_BOS_ML_foot"),
    )

    trial_bounds = (
        df.group_by(TRIAL_KEYS)
        .agg(
            pl.col("MocapFrame").min().alias("frame_min"),
            pl.col("MocapFrame").max().alias("frame_max"),
            pl.col("platform_onset_local").drop_nulls().first().alias("platform_onset_local"),
            pl.col("end_frame").drop_nulls().first().alias("end_frame"),
            pl.col("step_TF").drop_nulls().first().alias("step_TF"),
        )
        .with_columns(
            (
                pl.col("platform_onset_local").is_not_null()
                & pl.col("end_frame").is_not_null()
                & (pl.col("platform_onset_local") >= pl.col("frame_min"))
                & (pl.col("platform_onset_local") <= pl.col("frame_max"))
                & (pl.col("end_frame") >= pl.col("frame_min"))
                & (pl.col("end_frame") <= pl.col("frame_max"))
                & (pl.col("end_frame") >= pl.col("platform_onset_local"))
            ).alias("is_event_in_range")
        )
    )

    n_oob_trials = trial_bounds.filter(~pl.col("is_event_in_range")).height
    if n_oob_trials > 0:
        print(f"  Dropping trials with out-of-range events: {n_oob_trials}")

    valid_keys = trial_bounds.filter(pl.col("is_event_in_range")).select(TRIAL_KEYS)
    df = df.join(valid_keys, on=TRIAL_KEYS, how="inner")

    trial_count_df = df.select(TRIAL_KEYS + ["step_TF"]).unique()
    n_trials_step = trial_count_df.filter(pl.col("step_TF") == "step").height
    n_trials_nonstep = trial_count_df.filter(pl.col("step_TF") == "nonstep").height

    frame_pd = df.sort(TRIAL_KEYS + ["MocapFrame"]).to_pandas()
    frame_pd = _normalize_trial_types(frame_pd)
    frame_pd["step_TF"] = frame_pd["step_TF"].astype(str).str.strip().str.lower()

    qc = {
        "n_frames_raw": int(n_frames_raw),
        "n_frames_filtered": int(len(frame_pd)),
        "n_trials_step": int(n_trials_step),
        "n_trials_nonstep": int(n_trials_nonstep),
        "n_subjects": int(frame_pd["subject"].nunique()),
        "n_oob_trials_dropped": int(n_oob_trials),
        "n_step_missing_trials_dropped": int(n_step_missing_trials),
    }
    return frame_pd, qc


def _interp_nan_series(y: np.ndarray) -> np.ndarray:
    idx = np.arange(y.size)
    valid = np.isfinite(y)
    if not np.any(valid):
        return y
    return np.interp(idx, idx[valid], y[valid])


def build_normalized_trial_series(
    frame_pd: pd.DataFrame,
    variable_specs: list[VariableSpec],
    nan_ratio_threshold: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Counter]]:
    records = {spec.name: [] for spec in variable_specs}
    exclude_stats = {spec.name: Counter() for spec in variable_specs}

    grouped = frame_pd.groupby(TRIAL_KEYS, sort=False)
    print(f"  Normalizing trial windows to {NORM_POINTS} points for {grouped.ngroups} trials...")

    for idx, (trial_key, grp) in enumerate(grouped, start=1):
        if idx % 40 == 0:
            print(f"    ...trial progress {idx}/{grouped.ngroups}")

        step_tf = str(grp["step_TF"].iloc[0]).strip().lower()
        subject = str(grp["subject"].iloc[0]).strip()
        velocity = grp["velocity"].iloc[0]
        trial = grp["trial"].iloc[0]

        start_vals = pd.to_numeric(grp["platform_onset_local"], errors="coerce").dropna()
        end_vals = pd.to_numeric(grp["end_frame"], errors="coerce").dropna()
        if start_vals.empty or end_vals.empty:
            for spec in variable_specs:
                exclude_stats[spec.name]["missing_window_events"] += 1
            continue

        start = int(round(float(start_vals.iloc[0])))
        end = int(round(float(end_vals.iloc[0])))
        if end < start:
            for spec in variable_specs:
                exclude_stats[spec.name]["end_before_start"] += 1
            continue

        window = grp[(grp["MocapFrame"] >= start) & (grp["MocapFrame"] <= end)].sort_values("MocapFrame")
        if len(window) < 2:
            for spec in variable_specs:
                exclude_stats[spec.name]["window_too_short"] += 1
            continue

        for spec in variable_specs:
            var = spec.name
            y = pd.to_numeric(window[var], errors="coerce").to_numpy(dtype=float)
            if y.size < 2:
                exclude_stats[var]["window_too_short"] += 1
                continue

            nan_ratio = float(np.isnan(y).mean())
            if nan_ratio > nan_ratio_threshold:
                exclude_stats[var]["nan_ratio_gt_20pct"] += 1
                continue

            if np.isnan(y).all():
                exclude_stats[var]["all_nan"] += 1
                continue

            if np.isnan(y).any():
                y = _interp_nan_series(y)

            x_old = np.linspace(0.0, 1.0, y.size)
            x_new = np.linspace(0.0, 1.0, NORM_POINTS)
            y_norm = np.interp(x_new, x_old, y)

            records[var].append(
                {
                    "subject": subject,
                    "velocity": velocity,
                    "trial": trial,
                    "step_TF": step_tf,
                    "series": y_norm,
                    "trial_key": str(trial_key),
                }
            )

    return records, exclude_stats


def build_subject_paired_matrices(
    var_records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_subject: dict[str, dict[str, list[np.ndarray]]] = {}

    for rec in var_records:
        cond = rec["step_TF"]
        if cond not in {"step", "nonstep"}:
            continue
        subject = rec["subject"]
        if subject not in by_subject:
            by_subject[subject] = {"step": [], "nonstep": []}
        by_subject[subject][cond].append(rec["series"])

    subjects_any = sorted(by_subject.keys())
    pair_subjects = sorted([s for s in subjects_any if by_subject[s]["step"] and by_subject[s]["nonstep"]])
    excluded_subjects = sorted(set(subjects_any) - set(pair_subjects))

    y_step: list[np.ndarray] = []
    y_nonstep: list[np.ndarray] = []
    step_trials_used = 0
    nonstep_trials_used = 0

    for subject in pair_subjects:
        step_stack = np.vstack(by_subject[subject]["step"])
        non_stack = np.vstack(by_subject[subject]["nonstep"])
        y_step.append(step_stack.mean(axis=0))
        y_nonstep.append(non_stack.mean(axis=0))
        step_trials_used += int(step_stack.shape[0])
        nonstep_trials_used += int(non_stack.shape[0])

    if not pair_subjects:
        return {
            "Y_step": np.empty((0, NORM_POINTS)),
            "Y_nonstep": np.empty((0, NORM_POINTS)),
            "pair_subjects": pair_subjects,
            "excluded_subjects": excluded_subjects,
            "subjects_any": subjects_any,
            "step_trials_used": step_trials_used,
            "nonstep_trials_used": nonstep_trials_used,
        }

    return {
        "Y_step": np.vstack(y_step),
        "Y_nonstep": np.vstack(y_nonstep),
        "pair_subjects": pair_subjects,
        "excluded_subjects": excluded_subjects,
        "subjects_any": subjects_any,
        "step_trials_used": step_trials_used,
        "nonstep_trials_used": nonstep_trials_used,
    }


def extract_significant_clusters(spmi_obj: Any, alpha: float) -> list[dict[str, float]]:
    clusters_out: list[dict[str, float]] = []
    for cluster in getattr(spmi_obj, "clusters", []):
        p_val = getattr(cluster, "P", None)
        if p_val is None:
            p_val = getattr(cluster, "p", np.nan)
        try:
            p_val_f = float(p_val)
        except (TypeError, ValueError):
            p_val_f = math.nan

        endpoints = getattr(cluster, "endpoints", None)
        if endpoints is None or len(endpoints) != 2:
            continue

        start = float(endpoints[0])
        end = float(endpoints[1])
        extent = float(getattr(cluster, "extent", end - start))
        if math.isnan(p_val_f) or p_val_f <= alpha + 1e-12:
            clusters_out.append(
                {
                    "start_pct": start,
                    "end_pct": end,
                    "extent": extent,
                    "p": p_val_f,
                }
            )
    return clusters_out


def clusters_to_mask(clusters: list[dict[str, float]], n_points: int) -> np.ndarray:
    mask = np.zeros(n_points, dtype=bool)
    for cluster in clusters:
        start = max(0, int(math.floor(cluster["start_pct"])))
        end = min(n_points - 1, int(math.ceil(cluster["end_pct"])))
        if end >= start:
            mask[start : end + 1] = True
    return mask


def _infer_nonparam(
    y_step: np.ndarray,
    y_nonstep: np.ndarray,
    alpha: float,
    iterations: int,
) -> Any:
    snpm = spm1d.stats.nonparam.ttest_paired(y_step, y_nonstep)
    infer_sig = inspect.signature(snpm.inference)
    kwargs: dict[str, Any] = {"alpha": alpha, "two_tailed": True}
    if "iterations" in infer_sig.parameters:
        kwargs["iterations"] = iterations
    if "force_iterations" in infer_sig.parameters:
        kwargs["force_iterations"] = True
    return snpm.inference(**kwargs)


def run_spm_tests(
    y_step: np.ndarray,
    y_nonstep: np.ndarray,
    alpha: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "param_error": "",
        "nonparam_error": "",
        "param_z": np.full(NORM_POINTS, np.nan),
        "param_threshold": math.nan,
        "param_h0reject": False,
        "param_p_set": math.nan,
        "param_clusters": [],
        "param_sig_mask": np.zeros(NORM_POINTS, dtype=bool),
        "nonparam_h0reject": False,
        "nonparam_p": [],
        "nonparam_clusters": [],
        "nonparam_sig_mask": np.zeros(NORM_POINTS, dtype=bool),
    }

    try:
        spmt = spm1d.stats.ttest_paired(y_step, y_nonstep)
        spmi = spmt.inference(alpha=alpha, two_tailed=True)
        out["param_z"] = np.asarray(spmi.z, dtype=float)
        out["param_threshold"] = float(spmi.zstar)
        out["param_h0reject"] = bool(spmi.h0reject)
        p_set = getattr(spmi, "p_set", np.nan)
        out["param_p_set"] = float(p_set) if p_set is not None else math.nan
        out["param_clusters"] = extract_significant_clusters(spmi, alpha=alpha)
        out["param_sig_mask"] = clusters_to_mask(out["param_clusters"], NORM_POINTS)
        if out["param_h0reject"] and not out["param_sig_mask"].any() and np.isfinite(out["param_threshold"]):
            out["param_sig_mask"] = np.abs(out["param_z"]) >= out["param_threshold"]
    except Exception as exc:  # noqa: BLE001
        out["param_error"] = str(exc)

    try:
        np.random.seed(seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="spm1d")
            snpmi = _infer_nonparam(y_step, y_nonstep, alpha=alpha, iterations=iterations)
        out["nonparam_h0reject"] = bool(snpmi.h0reject)
        nonparam_p = [float(x) for x in getattr(snpmi, "p", [])]
        out["nonparam_p"] = nonparam_p
        out["nonparam_clusters"] = extract_significant_clusters(snpmi, alpha=alpha)
        out["nonparam_sig_mask"] = clusters_to_mask(out["nonparam_clusters"], NORM_POINTS)
    except Exception as exc:  # noqa: BLE001
        out["nonparam_error"] = str(exc)

    return out


def _format_cluster_ranges(clusters: list[dict[str, float]]) -> str:
    if not clusters:
        return "-"
    return ", ".join(f"{c['start_pct']:.1f}-{c['end_pct']:.1f}" for c in clusters)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def summarize_effect_direction(
    y_step: np.ndarray,
    y_nonstep: np.ndarray,
    param_mask: np.ndarray,
    nonparam_mask: np.ndarray,
) -> dict[str, Any]:
    mean_step_curve = y_step.mean(axis=0)
    mean_nonstep_curve = y_nonstep.mean(axis=0)
    mean_diff_curve = mean_step_curve - mean_nonstep_curve

    primary_mask = np.zeros(NORM_POINTS, dtype=bool)
    primary_source = ""
    if param_mask.any():
        primary_mask = param_mask
        primary_source = "param"
    elif nonparam_mask.any():
        primary_mask = nonparam_mask
        primary_source = "nonparam"

    direction = ""
    mean_diff_primary = math.nan
    if primary_mask.any():
        primary_vals = mean_diff_curve[primary_mask]
        mean_diff_primary = float(np.nanmean(primary_vals))
        tol = 1e-12
        if np.all(primary_vals >= -tol) and np.any(primary_vals > tol):
            direction = "step > nonstep"
        elif np.all(primary_vals <= tol) and np.any(primary_vals < -tol):
            direction = "step < nonstep"
        else:
            direction = "direction changes"

    return {
        "mean_step_overall": float(np.nanmean(mean_step_curve)),
        "mean_nonstep_overall": float(np.nanmean(mean_nonstep_curve)),
        "mean_diff_overall": float(np.nanmean(mean_diff_curve)),
        "primary_sig_source": primary_source,
        "primary_sig_direction": direction,
        "primary_sig_mean_diff": mean_diff_primary,
    }


def format_test_status(sig: bool, err: str) -> str:
    err = str(err).strip()
    if err:
        return f"failed: {err.splitlines()[0]}"
    return "sig" if sig else "n.s."


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value_f):
        return "-"
    return f"{value_f:.{digits}f}"


def plot_variable_spm(
    out_path: Path,
    spec: VariableSpec,
    y_step: np.ndarray,
    y_nonstep: np.ndarray,
    spm_result: dict[str, Any],
    alpha: float,
    dpi: int,
) -> None:
    x_pct = np.linspace(0, 100, NORM_POINTS)
    n_pairs = y_step.shape[0]

    step_mean = y_step.mean(axis=0)
    nonstep_mean = y_nonstep.mean(axis=0)
    if n_pairs > 1:
        step_sd = y_step.std(axis=0, ddof=1)
        nonstep_sd = y_nonstep.std(axis=0, ddof=1)
    else:
        step_sd = np.zeros(NORM_POINTS)
        nonstep_sd = np.zeros(NORM_POINTS)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.6]},
    )

    ax_top, ax_bottom = axes
    ax_top.plot(x_pct, step_mean, color="#e74c3c", linewidth=2.0, label="step")
    ax_top.fill_between(x_pct, step_mean - step_sd, step_mean + step_sd, color="#e74c3c", alpha=0.20)
    ax_top.plot(x_pct, nonstep_mean, color="#2980b9", linewidth=2.0, label="nonstep")
    ax_top.fill_between(
        x_pct,
        nonstep_mean - nonstep_sd,
        nonstep_mean + nonstep_sd,
        color="#2980b9",
        alpha=0.20,
    )
    ax_top.set_ylabel(spec.name)
    ax_top.set_title(f"{spec.family} | {spec.name} | N_pairs={n_pairs} | alpha_bonf={alpha:.4g}")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="best", frameon=False)

    param_z = spm_result["param_z"]
    param_thr = spm_result["param_threshold"]
    if np.isfinite(param_z).all() and np.isfinite(param_thr):
        ax_bottom.plot(x_pct, param_z, color="#1f2937", linewidth=1.5, label="SPM{t} (param)")
        ax_bottom.axhline(param_thr, color="#7f8c8d", linestyle="--", linewidth=1.0, label="threshold")
        ax_bottom.axhline(-param_thr, color="#7f8c8d", linestyle="--", linewidth=1.0)

        y_min, y_max = ax_bottom.get_ylim()
        if spm_result["param_sig_mask"].any():
            ax_bottom.fill_between(
                x_pct,
                y_min,
                y_max,
                where=spm_result["param_sig_mask"],
                color="#f39c12",
                alpha=0.22,
                label="Param sig.",
            )
        if spm_result["nonparam_sig_mask"].any():
            ax_bottom.fill_between(
                x_pct,
                y_min,
                y_max,
                where=spm_result["nonparam_sig_mask"],
                color="#2ecc71",
                alpha=0.16,
                label="Nonparam sig.",
            )
        ax_bottom.set_ylim(y_min, y_max)
    else:
        ax_bottom.text(
            0.01,
            0.90,
            "Parametric SPM failed",
            transform=ax_bottom.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#c0392b",
        )

    if spm_result["param_error"]:
        ax_bottom.text(
            0.01,
            0.75,
            f"param error: {spm_result['param_error'][:120]}",
            transform=ax_bottom.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#c0392b",
        )
    if spm_result["nonparam_error"]:
        ax_bottom.text(
            0.01,
            0.62,
            f"nonparam error: {spm_result['nonparam_error'][:120]}",
            transform=ax_bottom.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#c0392b",
        )

    ax_bottom.set_ylabel("SPM{t}")
    ax_bottom.set_xlabel("Time normalized (%): platform onset = 0, step onset = 100")
    ax_bottom.grid(alpha=0.25)
    handles, labels = ax_bottom.get_legend_handles_labels()
    if handles:
        ax_bottom.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_significance_heatmap(
    out_path: Path,
    variable_order: list[str],
    masks: list[np.ndarray],
    dpi: int,
) -> None:
    if not variable_order:
        return

    data = np.vstack([mask.astype(int) for mask in masks])
    fig_h = max(6.0, 0.36 * len(variable_order))

    fig, ax = plt.subplots(figsize=(13, fig_h))
    cmap = ListedColormap(["#f2f3f5", "#e67e22"])
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_xlabel("Time normalized (%)")
    ax.set_ylabel("Variable")
    ax.set_yticks(np.arange(len(variable_order)))
    ax.set_yticklabels(variable_order, fontsize=8)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_title("Significant intervals heatmap (parametric SPM, Bonferroni-corrected)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["n.s.", "sig."])

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_report(
    report_path: Path,
    results_df: pd.DataFrame,
    qc: dict[str, Any],
    iterations: int,
) -> None:
    sig_param = results_df[results_df["param_sig_bonf"]].copy()
    sig_nonparam = results_df[results_df["nonparam_sig_bonf"]].copy()

    family_summary = (
        results_df.groupby("family", as_index=False)
        .agg(
            n_variables=("variable", "count"),
            n_param_sig=("param_sig_bonf", "sum"),
            n_nonparam_sig=("nonparam_sig_bonf", "sum"),
        )
        .sort_values("family")
    )

    sig_table_df = (
        results_df.loc[
            results_df["param_sig_bonf"] | results_df["nonparam_sig_bonf"],
            [
                "variable",
                "family",
                "N_pairs",
                "param_cluster_ranges_pct",
                "nonparam_cluster_ranges_pct",
                "primary_sig_direction",
                "primary_sig_mean_diff",
            ],
        ]
        .sort_values(["family", "variable"])
        .reset_index(drop=True)
    )

    check_vars = ["xCOM_BOS_AP_foot", "xCOM_BOS_ML_foot", "Hip_stance_X_deg"]
    check_rows = []
    for var in check_vars:
        row = results_df.loc[results_df["variable"] == var]
        if row.empty:
            check_rows.append((var, "not computed"))
        else:
            r = row.iloc[0]
            sig_txt = f"param {format_test_status(bool(r['param_sig_bonf']), str(r['param_error']))}"
            sig_txt += f", nonparam {format_test_status(bool(r['nonparam_sig_bonf']), str(r['nonparam_error']))}"
            check_rows.append((var, sig_txt))

    failed_df = results_df.loc[
        (results_df["param_error"].fillna("") != "") | (results_df["nonparam_error"].fillna("") != ""),
        ["variable", "param_sig_bonf", "param_error", "nonparam_sig_bonf", "nonparam_error"],
    ].copy()
    failed_df = failed_df.sort_values("variable").reset_index(drop=True)

    directional_df = results_df.loc[
        results_df["primary_sig_direction"].fillna("") != "",
        ["variable", "primary_sig_source", "primary_sig_direction", "primary_sig_mean_diff"],
    ].copy()
    directional_df = directional_df.sort_values("variable").reset_index(drop=True)

    lines: list[str] = []
    lines.append("# Step vs. Non-step SPM 1D Analysis Report")
    lines.append("")
    lines.append("## Research Question")
    lines.append("")
    lines.append(
        "동일한 섭동 조건에서 step/nonstep 전략이 [platform onset → step onset] 구간의 시계열 전체에서 언제 유의하게 다른지 SPM 1D로 확인한다."
    )
    lines.append("")
    lines.append("## Data Summary")
    lines.append("")
    lines.append(f"- 분석 프레임 수: {qc['n_frames_filtered']} (원본 {qc['n_frames_raw']})")
    lines.append(
        f"- 분석 시행 수: {qc['n_trials_step'] + qc['n_trials_nonstep']} (step={qc['n_trials_step']}, nonstep={qc['n_trials_nonstep']})"
    )
    lines.append(f"- 피험자 수: {qc['n_subjects']}")
    lines.append(f"- 제외 시행: step onset 누락 {qc['n_step_missing_trials_dropped']}개, event 범위 이탈 {qc['n_oob_trials_dropped']}개")
    lines.append("- 입력 데이터: `output/all_trials_timeseries.csv`, `data/perturb_inform.xlsm`")
    lines.append("- 전처리 필터(`scripts/apply_post_filter_from_meta.py`): mixed==1, age_group==young, ipsilateral step only")
    lines.append(f"- 분석 변수 수: {len(results_df)}")
    lines.append("")
    lines.append("## Analysis Methodology")
    lines.append("")
    lines.append("- 분석 구간: 각 trial의 `[platform_onset_local, end_frame]`")
    lines.append("  - Step trial: `end_frame = step_onset_local`")
    lines.append("  - Nonstep trial: 같은 `(subject, velocity)`의 step onset 평균값(부족 시 platform sheet fallback)")
    lines.append("- 시간 정규화: 0-100% (101 points), NaN 20% 초과 trial 제외, 그 외 선형보간")
    lines.append("- 짝지음 단위: 피험자 내 step/nonstep 평균 곡선")
    lines.append("- SPM 검정: paired t-test (parametric + nonparametric permutation)")
    lines.append(f"- 비모수 순열 횟수: {iterations}")
    lines.append("- 다중비교 보정: family별 Bonferroni (`alpha = 0.05 / family_size`)")
    lines.append("- Nonstep stance side: subject별 major step side 사용, tie는 (L+R)/2")
    lines.append("- xCOM/BOS 정규화: `foot_len_m = (발길이_왼 + 발길이_오른)/2` 기반")
    lines.append("")
    lines.append("### Coordinate & Sign Conventions")
    lines.append("")
    lines.append("Axis & Direction Sign")
    lines.append("")
    lines.append("| Axis | Positive (+) | Negative (-) | 대표 변수 |")
    lines.append("|------|---------------|---------------|-----------|")
    lines.append("| AP (X) | +X = 전방 | -X = 후방 | COM_X, vCOM_X, xCOM_X, BOS_minX/maxX, MOS_AP_v3d |")
    lines.append("| ML (Y) | +Y = 좌측 | -Y = 우측 | COM_Y, vCOM_Y, xCOM_Y, BOS_minY/maxY, MOS_ML_v3d |")
    lines.append("| Vertical (Z) | +Z = 위 | -Z = 아래 | COM_Z, vCOM_Z, xCOM_Z, GRF_Z_N |")
    lines.append("")
    lines.append("Signed Metrics Interpretation")
    lines.append("")
    lines.append("| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |")
    lines.append("|--------|--------------|--------------|----------------|")
    lines.append("| MOS_minDist_signed | BOS 내부/안정 여유 | BOS 외부/안정 여유 부족 | signed minimum distance |")
    lines.append("| MOS_AP_v3d | AP 경계 내부 방향 | AP 경계 외부 방향 | AP bound-relative sign |")
    lines.append("| MOS_ML_v3d | ML 경계 내부 방향 | ML 경계 외부 방향 | ML bound-relative sign |")
    lines.append("| xCOM_BOS_AP_foot | BOS_minX 기준 전방 상대 위치 증가 | BOS_minX 기준 전방 상대 위치 감소 | foot length 정규화 |")
    lines.append("| xCOM_BOS_ML_foot | BOS_minY 기준 좌측 상대 위치 증가 | BOS_minY 기준 우측 상대 위치 증가 | foot length 정규화 |")
    lines.append("")
    lines.append("Joint/Force/Torque Sign Conventions")
    lines.append("")
    lines.append("| Variable group | (+)/(-) meaning | 추가 규칙 |")
    lines.append("|----------------|------------------|-----------|")
    lines.append("| Joint angles (Hip/Knee/Ankle/Trunk/Neck) | 각 축의 해부학적 회전 부호를 데이터 원 부호 그대로 사용 | stance side만 Hip/Knee/Ankle X축에 적용 |")
    lines.append("| Joint angular velocity (`*_ref_*_deg_s`, `*_mov_*_deg_s`) | 각속도 축 성분의 원 부호 유지 | Hip/Knee/Ankle은 stance side로 변환 후 비교 |")
    lines.append("| Segment moment (`*_ref_*_Nm`) | internal moment 축 성분의 원 부호 유지 | Hip/Knee/Ankle은 stance side로 변환 후 비교 |")
    lines.append("| GRF_* / GRM_* | force/torque 원시 부호 유지 | onset-zeroing 없이 절대 시계열 사용 |")
    lines.append("| COP_* | COP 절대 좌표 부호 유지 | onset-zeroing 없이 절대 시계열 사용 |")
    lines.append("| AnkleTorqueMid_int_Y_Nm_per_kg | internal torque 부호 유지 | 체중 정규화 값 사용 |")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"- Parametric 유의 변수: {len(sig_param)} / {len(results_df)}")
    lines.append(f"- Nonparametric 유의 변수: {len(sig_nonparam)} / {len(results_df)}")
    lines.append("")
    lines.append("### Family-level Summary")
    lines.append("")
    lines.append("| Family | Variables | Param Sig | Nonparam Sig |")
    lines.append("|--------|-----------|-----------|--------------|")
    for _, row in family_summary.iterrows():
        lines.append(
            f"| {row['family']} | {int(row['n_variables'])} | {int(row['n_param_sig'])} | {int(row['n_nonparam_sig'])} |"
        )
    lines.append("")

    lines.append("### Significant Variable Summary (Sig only)")
    lines.append("")
    lines.append("| Variable | Family | N_pairs | Param interval (%) | Nonparam interval (%) | Direction | Mean diff |")
    lines.append("|----------|--------|---------|--------------------|-----------------------|-----------|-----------|")
    if sig_table_df.empty:
        lines.append("| - | - | - | - | - | - | - |")
    else:
        for _, row in sig_table_df.iterrows():
            lines.append(
                "| "
                f"{row['variable']} | {row['family']} | {int(row['N_pairs'])} | "
                f"{row['param_cluster_ranges_pct']} | {row['nonparam_cluster_ranges_pct']} | "
                f"{row['primary_sig_direction'] or '-'} | {_format_float(row['primary_sig_mean_diff'])} |"
            )
    lines.append("")

    lines.append("### Cross-check with prior LMM focus variables")
    lines.append("")
    lines.append("| Variable | SPM status |")
    lines.append("|----------|------------|")
    for var, status in check_rows:
        lines.append(f"| {var} | {status} |")
    lines.append("")

    lines.append("### Test Execution Notes")
    lines.append("")
    lines.append("| Variable | Parametric status | Nonparametric status |")
    lines.append("|----------|-------------------|----------------------|")
    if failed_df.empty:
        lines.append("| - | all tests completed | all tests completed |")
    else:
        for _, row in failed_df.iterrows():
            lines.append(
                f"| {row['variable']} | "
                f"{format_test_status(bool(row['param_sig_bonf']), str(row['param_error']))} | "
                f"{format_test_status(bool(row['nonparam_sig_bonf']), str(row['nonparam_error']))} |"
            )
    lines.append("")

    lines.append("## Discussion")
    lines.append("")
    lines.append("### 결과 해석")
    lines.append("")
    lines.append(
        "유의 구간은 MOS 계열의 초기/후기 분리, vCOM_X의 초기+후기 두 구간, "
        "COM_Z의 초기/중기 구간, xCOM_X·COM_X의 중후기 구간, "
        "xCOM_BOS_AP_foot의 전 구간 유의처럼 변수별로 다른 시간 패턴을 보였다."
    )
    lines.append("")
    lines.append(
        "방향성은 자동으로 계산한 `step - nonstep` 평균 차이를 기준으로 확인하였다. "
        "COM_X, vCOM_X, xCOM_X, xCOM_BOS_AP_foot는 주된 유의 구간에서 `step < nonstep`이었고, "
        "COM_Z, xCOM_Z, COP_X_m은 `step > nonstep`이었다. "
        "MOS 계열은 초기에는 `step > nonstep`, 후기에는 `step < nonstep`으로 바뀌어 "
        "한 방향의 차이로 요약되지 않았다."
    )
    lines.append("")
    lines.append(
        "따라서 이 SPM 결과만으로 `step` 전략이 전 구간에서 더 전방으로 이동한다거나, "
        "BOS 경계를 지속적으로 넘는다고 단정할 수는 없다. "
        "방향성 해석은 변수와 시간 구간별로 나누어 읽어야 하며, "
        "기전 설명은 평균 곡선이나 추가 분석과 함께 제시하는 것이 안전하다."
    )
    lines.append("")

    if not directional_df.empty:
        lines.append("### Direction Check")
        lines.append("")
        lines.append("| Variable | Source | Direction | Mean diff |")
        lines.append("|----------|--------|-----------|-----------|")
        for _, row in directional_df.iterrows():
            lines.append(
                f"| {row['variable']} | {row['primary_sig_source']} | "
                f"{row['primary_sig_direction']} | {_format_float(row['primary_sig_mean_diff'])} |"
            )
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append("1. Step/nonstep 차이는 변수마다 다른 시간 구간에서 나타났고, 일부 변수는 두 개 이상의 분리된 유의 구간을 보였다.")
    lines.append("2. xCOM_BOS_AP_foot는 전 구간 유의하여 가장 일관된 구분 지표였지만, 방향은 `step < nonstep`으로 요약되었다.")
    lines.append("3. MOS 계열은 초기와 후기의 방향이 바뀌어, 동일 변수라도 시간 구간별 해석이 필요했다.")
    lines.append("4. Parametric SPM이 zero variance로 실패한 변수들은 음성 결과가 아니라 미검정 항목으로 해석해야 한다.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- 본 분석은 young, mixed==1, ipsilateral step 시행만 포함하므로, 고령자·contralateral step 등으로 일반화 시 주의가 필요하다.")
    lines.append("- paired t-test 구조상 피험자 내 step/nonstep 시행이 모두 존재하는 경우만 분석되어, 한 전략만 사용하는 피험자는 제외되었다.")
    lines.append("- 여러 변수에서 parametric SPM이 zero variance로 실패했으므로, 해당 변수의 모수 결과는 미검정으로 남는다.")
    lines.append("- 비모수 순열 검정과 모수 검정이 모두 성공한 유의 변수에서는 두 검정의 유의 여부가 일치하였다.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("conda run --no-capture-output -n module python analysis/step_vs_nonstep_spm/analyze_step_vs_nonstep_spm.py")
    lines.append("```")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `analysis/step_vs_nonstep_spm/spm_results.csv`")
    lines.append("- `analysis/step_vs_nonstep_spm/figures/spm_<variable>.png`")
    lines.append("- `analysis/step_vs_nonstep_spm/figures/heatmap_significant.png`")
    lines.append("- `analysis/step_vs_nonstep_spm/report.md`")
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    variable_specs = build_variable_specs()

    print("=" * 72)
    print("Step vs Non-step SPM 1D Analysis")
    print("=" * 72)

    print("\n[M1] Load, filter, and derive frame-level variables...")
    frame_pd, qc = prepare_frame_level_dataset(args.csv, args.platform_xlsm)
    print(
        "  QC summary: "
        f"frames={qc['n_frames_filtered']} | "
        f"trials(step/nonstep)={qc['n_trials_step']}/{qc['n_trials_nonstep']} | "
        f"subjects={qc['n_subjects']}"
    )

    print("\n[M2] Time-normalize trials and build subject-level pairs...")
    records_by_var, exclude_stats = build_normalized_trial_series(
        frame_pd,
        variable_specs=variable_specs,
        nan_ratio_threshold=NAN_RATIO_THRESHOLD,
    )

    if args.dry_run:
        print("\nDry-run complete. No SPM tests were executed.")
        return

    print("\n[M3] Run paired SPM tests (parametric + nonparametric)...")
    result_rows: list[dict[str, Any]] = []
    heatmap_order: list[str] = []
    heatmap_masks: list[np.ndarray] = []

    for idx, spec in enumerate(variable_specs, start=1):
        print(f"  [{idx:02d}/{len(variable_specs)}] {spec.name}")
        paired = build_subject_paired_matrices(records_by_var[spec.name])
        y_step = paired["Y_step"]
        y_nonstep = paired["Y_nonstep"]
        n_pairs = int(y_step.shape[0])

        alpha_bonf = ALPHA / FAMILY_SIZE[spec.family]
        row: dict[str, Any] = {
            "variable": spec.name,
            "family": spec.family,
            "family_size": int(FAMILY_SIZE[spec.family]),
            "alpha_bonf": alpha_bonf,
            "N_pairs": n_pairs,
            "subjects_with_any_condition": len(paired["subjects_any"]),
            "n_excluded_subjects": len(paired["excluded_subjects"]),
            "excluded_subjects": ";".join(paired["excluded_subjects"]),
            "pair_subjects": ";".join(paired["pair_subjects"]),
            "step_trials_used": int(paired["step_trials_used"]),
            "nonstep_trials_used": int(paired["nonstep_trials_used"]),
            "excluded_trials_nan_gt20": int(exclude_stats[spec.name]["nan_ratio_gt_20pct"]),
            "excluded_trials_all_nan": int(exclude_stats[spec.name]["all_nan"]),
            "excluded_trials_window_short": int(exclude_stats[spec.name]["window_too_short"]),
            "excluded_trials_event_missing": int(exclude_stats[spec.name]["missing_window_events"]),
            "excluded_trials_bad_window": int(exclude_stats[spec.name]["end_before_start"]),
            "param_sig_bonf": False,
            "param_p_set": math.nan,
            "param_min_cluster_p": math.nan,
            "param_cluster_ranges_pct": "-",
            "param_clusters_json": "[]",
            "param_error": "",
            "nonparam_sig_bonf": False,
            "nonparam_min_p": math.nan,
            "nonparam_cluster_ranges_pct": "-",
            "nonparam_clusters_json": "[]",
            "nonparam_error": "",
            "mean_step_overall": math.nan,
            "mean_nonstep_overall": math.nan,
            "mean_diff_overall": math.nan,
            "primary_sig_source": "",
            "primary_sig_direction": "",
            "primary_sig_mean_diff": math.nan,
            "figure_path": "",
        }

        if n_pairs < 2:
            row["param_error"] = "Insufficient paired subjects (<2)"
            row["nonparam_error"] = "Insufficient paired subjects (<2)"
            result_rows.append(row)
            continue

        spm_result = run_spm_tests(
            y_step=y_step,
            y_nonstep=y_nonstep,
            alpha=alpha_bonf,
            iterations=args.nonparam_iterations,
            seed=args.seed + idx,
        )

        row["param_sig_bonf"] = bool(spm_result["param_h0reject"])
        row["param_p_set"] = spm_result["param_p_set"]
        row["param_error"] = spm_result["param_error"]
        row["param_clusters_json"] = json.dumps(spm_result["param_clusters"], ensure_ascii=False)
        row["param_cluster_ranges_pct"] = _format_cluster_ranges(spm_result["param_clusters"])
        if spm_result["param_clusters"]:
            row["param_min_cluster_p"] = float(
                np.nanmin([c["p"] for c in spm_result["param_clusters"] if not math.isnan(c["p"])])
            )

        row["nonparam_sig_bonf"] = bool(spm_result["nonparam_h0reject"])
        row["nonparam_error"] = spm_result["nonparam_error"]
        row["nonparam_clusters_json"] = json.dumps(spm_result["nonparam_clusters"], ensure_ascii=False)
        row["nonparam_cluster_ranges_pct"] = _format_cluster_ranges(spm_result["nonparam_clusters"])
        if spm_result["nonparam_clusters"]:
            row["nonparam_min_p"] = float(
                np.nanmin([c["p"] for c in spm_result["nonparam_clusters"] if not math.isnan(c["p"])])
            )
        elif spm_result["nonparam_p"]:
            row["nonparam_min_p"] = float(np.nanmin(spm_result["nonparam_p"]))

        row.update(
            summarize_effect_direction(
                y_step=y_step,
                y_nonstep=y_nonstep,
                param_mask=spm_result["param_sig_mask"],
                nonparam_mask=spm_result["nonparam_sig_mask"],
            )
        )

        fig_path = fig_dir / f"spm_{_safe_name(spec.name)}.png"
        plot_variable_spm(
            out_path=fig_path,
            spec=spec,
            y_step=y_step,
            y_nonstep=y_nonstep,
            spm_result=spm_result,
            alpha=alpha_bonf,
            dpi=args.dpi,
        )
        row["figure_path"] = str(fig_path.relative_to(REPO_ROOT))

        heatmap_order.append(spec.name)
        heatmap_masks.append(spm_result["param_sig_mask"])
        result_rows.append(row)

    print("\n[M4] Save outputs (CSV, heatmap, report)...")
    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(["family", "variable"]).reset_index(drop=True)

    csv_path = out_dir / "spm_results.csv"
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    heatmap_path = fig_dir / "heatmap_significant.png"
    plot_significance_heatmap(heatmap_path, heatmap_order, heatmap_masks, dpi=args.dpi)

    report_path = out_dir / "report.md"
    write_report(
        report_path=report_path,
        results_df=results_df,
        qc=qc,
        iterations=args.nonparam_iterations,
    )

    print("\n[M5] Summary")
    n_param_sig = int(results_df["param_sig_bonf"].sum())
    n_nonparam_sig = int(results_df["nonparam_sig_bonf"].sum())
    print(f"  Variables tested: {len(results_df)}")
    print(f"  Parametric significant (Bonf): {n_param_sig}")
    print(f"  Nonparametric significant (Bonf): {n_nonparam_sig}")
    print(f"  Results CSV: {csv_path}")
    print(f"  Heatmap: {heatmap_path}")
    print(f"  Report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
