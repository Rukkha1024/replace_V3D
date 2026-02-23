"""Initial Posture Strategy LMM Analysis.

Answers:
"Can initial posture at platform onset explain step vs nonstep strategy differences?"

Statistical method: Linear Mixed Model (LMM) with lmerTest inference
  Model: DV ~ step_TF + (1|subject)
  Multiple comparison: Benjamini-Hochberg FDR (all testable onset variables)
  Analysis point: platform_onset_local (single-frame snapshot)

Notes:
- Existing exported angles/GRF/torque are onset-zeroed in the batch CSV. Therefore,
  many onset variables become structural constants (untestable).
- To evaluate initial posture for angle variables, this script recomputes raw
  Visual3D-like joint angles from C3D and extracts absolute values at onset.

Produces:
- stdout summary and significant-variable-only result table
- no figure/csv/xlsx outputs (temporary files for R subprocess are deleted)

Usage:
    conda run --no-capture-output -n module python \
      analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py

    conda run --no-capture-output -n module python \
      analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_lmm.py --dry-run
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
    load_trial_events,
    parse_subject_velocity_trial_from_filename,
    resolve_subject_from_token,
)
from replace_v3d.joint_angles.v3d_joint_angles import compute_v3d_joint_angles_3d

# Required display config by repo rule
pl.Config.set_tbl_rows(999)
pl.Config.set_tbl_cols(999)
pl.Config.set_tbl_width_chars(120)

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_C3D_DIR = REPO_ROOT / "data" / "all_data"
DEFAULT_OUT_DIR = SCRIPT_DIR

TRIAL_KEYS = ["subject", "velocity", "trial"]

WINDOWS_R_HOME = Path(r"C:\Users\Alice\miniconda3\envs\module\lib\R")
WINDOWS_RSCRIPT = WINDOWS_R_HOME / "bin" / "x64" / "Rscript.exe"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--c3d_dir", type=Path, default=DEFAULT_C3D_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
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
    return counts[["subject", "major_step_side"]]


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


def add_stance_cols_pl(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("state") == "step_r")
        .then(pl.col("Hip_L_X_deg"))
        .when(pl.col("state") == "step_l")
        .then(pl.col("Hip_R_X_deg"))
        .when(pl.col("major_step_side") == "step_r")
        .then(pl.col("Hip_L_X_deg"))
        .when(pl.col("major_step_side") == "step_l")
        .then(pl.col("Hip_R_X_deg"))
        .otherwise((pl.col("Hip_L_X_deg") + pl.col("Hip_R_X_deg")) / 2.0)
        .alias("Hip_stance_X_deg"),
        pl.when(pl.col("state") == "step_r")
        .then(pl.col("Knee_L_X_deg"))
        .when(pl.col("state") == "step_l")
        .then(pl.col("Knee_R_X_deg"))
        .when(pl.col("major_step_side") == "step_r")
        .then(pl.col("Knee_L_X_deg"))
        .when(pl.col("major_step_side") == "step_l")
        .then(pl.col("Knee_R_X_deg"))
        .otherwise((pl.col("Knee_L_X_deg") + pl.col("Knee_R_X_deg")) / 2.0)
        .alias("Knee_stance_X_deg"),
        pl.when(pl.col("state") == "step_r")
        .then(pl.col("Ankle_L_X_deg"))
        .when(pl.col("state") == "step_l")
        .then(pl.col("Ankle_R_X_deg"))
        .when(pl.col("major_step_side") == "step_r")
        .then(pl.col("Ankle_L_X_deg"))
        .when(pl.col("major_step_side") == "step_l")
        .then(pl.col("Ankle_R_X_deg"))
        .otherwise((pl.col("Ankle_L_X_deg") + pl.col("Ankle_R_X_deg")) / 2.0)
        .alias("Ankle_stance_X_deg"),
    )


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


def _pick_stance_value(row: pd.Series, left_col: str, right_col: str) -> float:
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


def compute_absolute_onset_angles(
    trial_meta: pd.DataFrame,
    platform_xlsm: Path,
    c3d_dir: Path,
    major_side: pd.DataFrame,
) -> pd.DataFrame:
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

    for key in sorted(keys):
        subject, velocity, trial = key
        c3d = read_c3d_points(key_to_file[key])
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

        bilateral = {
            "Hip_L_X_abs_onset": float(angles.hip_L_X[idx0]),
            "Hip_R_X_abs_onset": float(angles.hip_R_X[idx0]),
            "Knee_L_X_abs_onset": float(angles.knee_L_X[idx0]),
            "Knee_R_X_abs_onset": float(angles.knee_R_X[idx0]),
            "Ankle_L_X_abs_onset": float(angles.ankle_L_X[idx0]),
            "Ankle_R_X_abs_onset": float(angles.ankle_R_X[idx0]),
            "Trunk_X_abs_onset": float(angles.trunk_X[idx0]),
            "Neck_X_abs_onset": float(angles.neck_X[idx0]),
        }

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
        row["Hip_stance_X_abs_onset"] = _pick_stance_value(s, "Hip_L_X_abs_onset", "Hip_R_X_abs_onset")
        row["Knee_stance_X_abs_onset"] = _pick_stance_value(s, "Knee_L_X_abs_onset", "Knee_R_X_abs_onset")
        row["Ankle_stance_X_abs_onset"] = _pick_stance_value(s, "Ankle_L_X_abs_onset", "Ankle_R_X_abs_onset")

        rows.append(row)

    out = pd.DataFrame(rows)
    return out[TRIAL_KEYS + [
        "Hip_stance_X_abs_onset",
        "Knee_stance_X_abs_onset",
        "Ankle_stance_X_abs_onset",
        "Trunk_X_abs_onset",
        "Neck_X_abs_onset",
    ]]


def build_analysis_dataframe(csv_path: Path, platform_xlsm: Path, c3d_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("  Loading CSV...")
    df = load_csv(csv_path)
    print(f"  Frames: {df.height}, Columns: {df.width}")

    print("  Loading platform metadata...")
    platform = load_platform_sheet(platform_xlsm)

    trial_meta = build_trial_meta(df=df, platform=platform)
    major_side = build_subject_major_step_side(platform=platform)

    print(
        "  Trial set: "
        f"{len(trial_meta)} (step={(trial_meta['step_TF'] == 'step').sum()}, "
        f"nonstep={(trial_meta['step_TF'] == 'nonstep').sum()})"
    )

    print("  Building onset snapshot variables from exported CSV...")
    onset_df = build_onset_snapshot(df=df, trial_meta=trial_meta, major_side=major_side)

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
        "Hip_stance_X_deg",
        "Knee_stance_X_deg",
        "Ankle_stance_X_deg",
        "Trunk_X_deg",
        "Neck_X_deg",
        "COP_X_m_onset0",
        "COP_Y_m_onset0",
        "GRF_X_N",
        "GRF_Y_N",
        "GRF_Z_N",
        "AnkleTorqueMid_int_Y_Nm_per_kg",
    ]
    missing_cols = sorted(set(keep_cols) - set(onset_df.columns))
    if missing_cols:
        raise ValueError(f"Missing onset columns from CSV snapshot: {missing_cols}")

    onset_df = onset_df[keep_cols].copy()

    print("  Recomputing absolute onset angles from C3D...")
    abs_angles = compute_absolute_onset_angles(
        trial_meta=trial_meta,
        platform_xlsm=platform_xlsm,
        c3d_dir=c3d_dir,
        major_side=major_side,
    )

    analysis_df = onset_df.merge(abs_angles, on=TRIAL_KEYS, how="left")
    if analysis_df[[
        "Hip_stance_X_abs_onset",
        "Knee_stance_X_abs_onset",
        "Ankle_stance_X_abs_onset",
        "Trunk_X_abs_onset",
        "Neck_X_abs_onset",
    ]].isna().any().any():
        raise ValueError("Missing absolute onset angle values after merge.")

    return analysis_df, trial_meta, abs_angles


def variable_catalog() -> list[dict[str, str]]:
    return [
        {"dv": "COM_X", "family": "Balance"},
        {"dv": "COM_Y", "family": "Balance"},
        {"dv": "vCOM_X", "family": "Balance"},
        {"dv": "vCOM_Y", "family": "Balance"},
        {"dv": "MOS_minDist_signed", "family": "Balance"},
        {"dv": "MOS_AP_v3d", "family": "Balance"},
        {"dv": "MOS_ML_v3d", "family": "Balance"},
        {"dv": "xCOM_BOS_norm_onset", "family": "Balance"},
        {"dv": "Hip_stance_X_deg", "family": "Joint_onset_zeroed"},
        {"dv": "Knee_stance_X_deg", "family": "Joint_onset_zeroed"},
        {"dv": "Ankle_stance_X_deg", "family": "Joint_onset_zeroed"},
        {"dv": "Trunk_X_deg", "family": "Joint_onset_zeroed"},
        {"dv": "Neck_X_deg", "family": "Joint_onset_zeroed"},
        {"dv": "COP_X_m_onset0", "family": "Force_onset_zeroed"},
        {"dv": "COP_Y_m_onset0", "family": "Force_onset_zeroed"},
        {"dv": "GRF_X_N", "family": "Force_onset_zeroed"},
        {"dv": "GRF_Y_N", "family": "Force_onset_zeroed"},
        {"dv": "GRF_Z_N", "family": "Force_onset_zeroed"},
        {"dv": "AnkleTorqueMid_int_Y_Nm_per_kg", "family": "Force_onset_zeroed"},
        {"dv": "Hip_stance_X_abs_onset", "family": "Joint_absolute"},
        {"dv": "Knee_stance_X_abs_onset", "family": "Joint_absolute"},
        {"dv": "Ankle_stance_X_abs_onset", "family": "Joint_absolute"},
        {"dv": "Trunk_X_abs_onset", "family": "Joint_absolute"},
        {"dv": "Neck_X_abs_onset", "family": "Joint_absolute"},
    ]


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


def _build_r_lmm_script(csv_path: str, output_path: str) -> str:
    return f"""
library(lmerTest)

data <- read.csv("{csv_path}", stringsAsFactors = FALSE)
data$step_TF <- tolower(trimws(as.character(data$step_TF)))
data$step_TF <- factor(data$step_TF, levels = c("nonstep", "step"))
dv_cols <- colnames(data)[!colnames(data) %in% c("subject", "velocity", "trial", "step_TF")]

results <- data.frame(
  dv = character(),
  estimate = numeric(),
  SE = numeric(),
  df = numeric(),
  t_value = numeric(),
  p_value = numeric(),
  mean_step = numeric(),
  sd_step = numeric(),
  mean_nonstep = numeric(),
  sd_nonstep = numeric(),
  n_step = integer(),
  n_nonstep = integer(),
  converged = logical(),
  stringsAsFactors = FALSE
)

for (dv in dv_cols) {{
  sub <- data[!is.na(data[[dv]]) & !is.na(data$step_TF), ]
  n_s <- sum(sub$step_TF == "step")
  n_ns <- sum(sub$step_TF == "nonstep")

  m_s <- mean(sub[[dv]][sub$step_TF == "step"], na.rm = TRUE)
  sd_s <- sd(sub[[dv]][sub$step_TF == "step"], na.rm = TRUE)
  m_ns <- mean(sub[[dv]][sub$step_TF == "nonstep"], na.rm = TRUE)
  sd_ns <- sd(sub[[dv]][sub$step_TF == "nonstep"], na.rm = TRUE)

  if (n_s == 0 || n_ns == 0) {{
    results <- rbind(results, data.frame(
      dv = dv, estimate = NA, SE = NA, df = NA,
      t_value = NA, p_value = NA,
      mean_step = m_s, sd_step = sd_s,
      mean_nonstep = m_ns, sd_nonstep = sd_ns,
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
      dv = dv, estimate = est, SE = se, df = df_val,
      t_value = t_val, p_value = p_val,
      mean_step = m_s, sd_step = sd_s,
      mean_nonstep = m_ns, sd_nonstep = sd_ns,
      n_step = n_s, n_nonstep = n_ns, converged = TRUE,
      stringsAsFactors = FALSE
    ))
  }}, error = function(e) {{
    results <<- rbind(results, data.frame(
      dv = dv, estimate = NA, SE = NA, df = NA,
      t_value = NA, p_value = NA,
      mean_step = m_s, sd_step = sd_s,
      mean_nonstep = m_ns, sd_nonstep = sd_ns,
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

    fmt = "{:<28s} {:<20s} {:>10s} {:>10s} {:>8s} {:>5s}"
    print(fmt.format("DV", "Family", "Estimate", "SE", "t", "Sig"))
    print("-" * 96)
    for _, row in sig_df.iterrows():
        print(
            fmt.format(
                str(row["dv"])[:28],
                str(row["family"])[:20],
                f"{row['estimate']:.4f}" if pd.notna(row["estimate"]) else "NA",
                f"{row['SE']:.4f}" if pd.notna(row["SE"]) else "NA",
                f"{row['t_value']:.3f}" if pd.notna(row["t_value"]) else "NA",
                str(row["sig"]),
            )
        )
    print("-" * 96)


def main() -> None:
    args = parse_args()
    _ = args.no_figures  # compatibility arg; intentionally unused

    print("=" * 72)
    print("Initial Posture Strategy LMM")
    print("=" * 72)

    print("\n[M1] Load and prepare data...")
    analysis_df, trial_meta, _abs_angles = build_analysis_dataframe(
        csv_path=args.csv,
        platform_xlsm=args.platform_xlsm,
        c3d_dir=args.c3d_dir,
    )

    specs = variable_catalog()
    dv_to_family = {s["dv"]: s["family"] for s in specs}

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
    print(f"  Trial set fixed: {n_trials} (step={n_step}, nonstep={n_nonstep})")

    if args.dry_run:
        print("\nDry run complete.")
        return

    print("\n[M3] Fit LMM (testable variables only)...")
    testable_vars = audit_df.loc[audit_df["status"] == "testable", "dv"].tolist()
    results = fit_lmm_all(analysis_df=analysis_df, testable_vars=testable_vars, dv_to_family=dv_to_family)

    print_significant_only(results)

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


if __name__ == "__main__":
    main()
