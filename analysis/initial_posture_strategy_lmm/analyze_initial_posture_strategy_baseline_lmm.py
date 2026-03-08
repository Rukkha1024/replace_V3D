"""Run baseline-window initial posture LMM for step vs nonstep trials.
Load trial-level baseline means from the exported timeseries CSV, join platform
metadata, fit DV ~ step_TF + (1|subject) with R lmerTest, and refresh baseline
markdown reports. The baseline window is [-0.30 s, 0.00 s] relative to platform onset.
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

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Required display config by repo rule
pl.Config.set_tbl_rows(999)
pl.Config.set_tbl_cols(999)
pl.Config.set_tbl_width_chars(120)

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_REPORT_MD = SCRIPT_DIR / "report_baseline.md"
DEFAULT_SEGMENT_ANGLE_MD = SCRIPT_DIR / "결과) 주제2-Segement Angle_baseline.md"

TRIAL_KEYS = ["subject", "velocity", "trial"]
ANGLE_AXES = ("X", "Y", "Z")
STANCE_SEGMENTS = ("Hip", "Knee", "Ankle")
MIDLINE_SEGMENTS = ("Trunk", "Neck")
WINDOW_START_S = -0.30
WINDOW_END_S = 0.00

WINDOWS_R_HOME = Path(r"C:\Users\Alice\miniconda3\envs\module\lib\R")
WINDOWS_RSCRIPT = WINDOWS_R_HOME / "bin" / "x64" / "Rscript.exe"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the baseline analysis workflow."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--report_md", type=Path, default=DEFAULT_REPORT_MD)
    ap.add_argument("--segment_angle_md", type=Path, default=DEFAULT_SEGMENT_ANGLE_MD)
    ap.add_argument("--dry-run", action="store_true", help="Only load data and audit variables.")
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="Accepted for compatibility; this analysis does not generate figures.",
    )
    return ap.parse_args()


def load_csv(path: Path) -> pl.DataFrame:
    """Load the exported trial timeseries CSV with a stable schema."""
    df = pl.read_csv(str(path), encoding="utf8-lossy", infer_schema_length=10000)
    if "trial" not in df.columns and "trial_num" in df.columns:
        df = df.rename({"trial_num": "trial"})
    return df


def load_platform_sheet(path: Path) -> pd.DataFrame:
    """Load step/nonstep platform metadata required for trial labeling."""
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
    """Determine each subject's major stepping side from mixed step trials."""
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
    """Assemble one-row-per-trial metadata for the exported timeseries dataset."""
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
    """Create frame-wise stance-side joint-angle columns from left/right exports."""
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
    return df.with_columns(exprs)


def _joint_angle_baseline_cols() -> list[str]:
    cols: list[str] = []
    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            cols.append(f"{seg}_stance_{axis}_baseline")
    for seg in MIDLINE_SEGMENTS:
        for axis in ANGLE_AXES:
            cols.append(f"{seg}_{axis}_baseline")
    return cols


def summarize_major_step_side(major_side: pd.DataFrame) -> dict[str, Any]:
    """Summarize subject-level major step side distribution."""
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


def variable_catalog() -> list[dict[str, str]]:
    """Return the fixed 29-variable baseline analysis specification."""
    specs: list[dict[str, str]] = [
        {"dv": "COM_X_baseline", "family": "Balance"},
        {"dv": "COM_Y_baseline", "family": "Balance"},
        {"dv": "vCOM_X_baseline", "family": "Balance"},
        {"dv": "vCOM_Y_baseline", "family": "Balance"},
        {"dv": "MOS_minDist_signed_baseline", "family": "Balance"},
        {"dv": "MOS_AP_v3d_baseline", "family": "Balance"},
        {"dv": "MOS_ML_v3d_baseline", "family": "Balance"},
        {"dv": "xCOM_BOS_norm_baseline", "family": "Balance"},
    ]
    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_stance_{axis}_baseline", "family": "Joint_baseline"})
    for seg in MIDLINE_SEGMENTS:
        for axis in ANGLE_AXES:
            specs.append({"dv": f"{seg}_{axis}_baseline", "family": "Joint_baseline"})
    specs.extend([
        {"dv": "COP_X_baseline", "family": "Force_baseline"},
        {"dv": "COP_Y_baseline", "family": "Force_baseline"},
        {"dv": "GRF_X_baseline", "family": "Force_baseline"},
        {"dv": "GRF_Y_baseline", "family": "Force_baseline"},
        {"dv": "GRF_Z_baseline", "family": "Force_baseline"},
        {"dv": "AnkleTorqueMid_Y_perkg_baseline", "family": "Force_baseline"},
    ])
    return specs


def build_baseline_dataframe(
    df: pl.DataFrame,
    trial_meta: pd.DataFrame,
    major_side: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate frame-wise baseline means per trial from the exported CSV."""
    df_base = df.drop([c for c in ["step_TF", "state", "mixed"] if c in df.columns])
    trial_meta_pl = pl.from_pandas(trial_meta[TRIAL_KEYS + ["step_TF", "state", "mixed"]].copy())
    major_side_pl = pl.from_pandas(major_side.copy())

    baseline_frames = (
        df_base.join(trial_meta_pl, on=TRIAL_KEYS, how="inner")
        .join(major_side_pl, on="subject", how="left")
        .filter(pl.col("time_from_platform_onset_s").is_between(WINDOW_START_S, WINDOW_END_S, closed="both"))
    )
    if baseline_frames.is_empty():
        raise ValueError("No frames were found in the requested baseline window.")

    baseline_frames = add_stance_cols_pl(baseline_frames).with_columns(
        pl.when((pl.col("BOS_maxX") - pl.col("BOS_minX")) > 0)
        .then((pl.col("xCOM_X") - pl.col("BOS_minX")) / (pl.col("BOS_maxX") - pl.col("BOS_minX")))
        .otherwise(None)
        .alias("xCOM_BOS_norm_frame")
    )

    agg_exprs: list[pl.Expr] = [
        pl.len().alias("baseline_n_frames"),
        pl.col("step_TF").drop_nulls().first().alias("step_TF"),
        pl.col("state").drop_nulls().first().alias("state"),
        pl.col("major_step_side").drop_nulls().first().alias("major_step_side"),
        pl.col("COM_X").mean().alias("COM_X_baseline"),
        pl.col("COM_Y").mean().alias("COM_Y_baseline"),
        pl.col("vCOM_X").mean().alias("vCOM_X_baseline"),
        pl.col("vCOM_Y").mean().alias("vCOM_Y_baseline"),
        pl.col("MOS_minDist_signed").mean().alias("MOS_minDist_signed_baseline"),
        pl.col("MOS_AP_v3d").mean().alias("MOS_AP_v3d_baseline"),
        pl.col("MOS_ML_v3d").mean().alias("MOS_ML_v3d_baseline"),
        pl.col("xCOM_BOS_norm_frame").mean().alias("xCOM_BOS_norm_baseline"),
        pl.col("Trunk_X_deg").mean().alias("Trunk_X_baseline"),
        pl.col("Trunk_Y_deg").mean().alias("Trunk_Y_baseline"),
        pl.col("Trunk_Z_deg").mean().alias("Trunk_Z_baseline"),
        pl.col("Neck_X_deg").mean().alias("Neck_X_baseline"),
        pl.col("Neck_Y_deg").mean().alias("Neck_Y_baseline"),
        pl.col("Neck_Z_deg").mean().alias("Neck_Z_baseline"),
        pl.col("COP_X_m").mean().alias("COP_X_baseline"),
        pl.col("COP_Y_m").mean().alias("COP_Y_baseline"),
        pl.col("GRF_X_N").mean().alias("GRF_X_baseline"),
        pl.col("GRF_Y_N").mean().alias("GRF_Y_baseline"),
        pl.col("GRF_Z_N").mean().alias("GRF_Z_baseline"),
        pl.col("AnkleTorqueMid_int_Y_Nm_per_kg").mean().alias("AnkleTorqueMid_Y_perkg_baseline"),
    ]

    for seg in STANCE_SEGMENTS:
        for axis in ANGLE_AXES:
            agg_exprs.append(
                pl.col(f"{seg}_stance_{axis}_deg").mean().alias(f"{seg}_stance_{axis}_baseline")
            )

    baseline_df = baseline_frames.group_by(TRIAL_KEYS).agg(agg_exprs).sort(TRIAL_KEYS)

    dup = baseline_df.group_by(TRIAL_KEYS).len().filter(pl.col("len") != 1)
    if dup.height > 0:
        raise ValueError(f"Baseline aggregation produced duplicate trial rows for {dup.height} trials.")

    baseline_pd = baseline_df.to_pandas()
    available_keys = baseline_pd[TRIAL_KEYS].drop_duplicates()
    used_trial_meta = trial_meta.merge(available_keys, on=TRIAL_KEYS, how="inner").reset_index(drop=True)

    frame_counts = pd.to_numeric(baseline_pd["baseline_n_frames"], errors="coerce")
    stats = {
        "window_start_s": WINDOW_START_S,
        "window_end_s": WINDOW_END_S,
        "n_trials_total": int(len(trial_meta)),
        "n_trials_used": int(len(used_trial_meta)),
        "n_trials_excluded": int(len(trial_meta) - len(used_trial_meta)),
        "n_step_used": int((used_trial_meta["step_TF"] == "step").sum()),
        "n_nonstep_used": int((used_trial_meta["step_TF"] == "nonstep").sum()),
        "baseline_frames_min": int(frame_counts.min()) if not frame_counts.empty else 0,
        "baseline_frames_max": int(frame_counts.max()) if not frame_counts.empty else 0,
        "baseline_frames_mean": float(frame_counts.mean()) if not frame_counts.empty else 0.0,
    }
    return baseline_pd, used_trial_meta, stats


def audit_variables(analysis_df: pd.DataFrame, specs: list[dict[str, str]]) -> pd.DataFrame:
    """Classify each dependent variable as testable or untestable."""
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
    """Locate a usable Rscript runtime inside the module environment."""
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


def _build_r_lmm_script(csv_path: str, output_path: str) -> str:
    """Generate the Rscript body that fits one LMM per dependent variable."""
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
    """Fit one subject-random-intercept LMM per dependent variable."""
    export_cols = TRIAL_KEYS + ["step_TF"] + testable_vars
    export_df = analysis_df[export_cols].copy()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, dir=tempfile.gettempdir(), newline="") as f:
        export_df.to_csv(f, index=False)
        data_csv = f.name

    result_csv = tempfile.mktemp(suffix="_baseline_lmm_results.csv", dir=tempfile.gettempdir())

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
    """Print the BH-FDR significant variables in a compact console table."""
    sig_df = results[results["sig"] != ""].copy().sort_values("p_fdr")

    print("\n" + "=" * 96)
    print("Significant Baseline Variables Only (BH-FDR < 0.05)")
    print("=" * 96)

    if sig_df.empty:
        print("No significant variables under BH-FDR < 0.05.")
        return

    fmt = "{:<32s} {:<18s} {:>10s} {:>10s} {:>8s} {:>5s}"
    print(fmt.format("DV", "Family", "Estimate", "SE", "t", "Sig"))
    print("-" * 96)
    for _, row in sig_df.iterrows():
        print(
            fmt.format(
                str(row["dv"])[:32],
                str(row["family"])[:18],
                f"{row['estimate']:.2f}" if pd.notna(row["estimate"]) else "NA",
                f"{row['SE']:.2f}" if pd.notna(row["SE"]) else "NA",
                f"{row['t_value']:.2f}" if pd.notna(row["t_value"]) else "NA",
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
        "| Variable | Family | Testability at baseline | Result status |",
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
        lines.append(
            "| "
            f"`{dv}` | "
            f"{_fmt_mean_sd(row.mean_step, row.sd_step, 2)} | "
            f"{_fmt_mean_sd(row.mean_nonstep, row.sd_nonstep, 2)} | "
            f"{_fmt_num(row.estimate, 2)} | {sig} |"
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
    major_side: pd.DataFrame,
    trial_stats: dict[str, Any],
) -> None:
    """Write the baseline comparison report aligned to the paper-replication format."""
    n_testable = int((audit_df["status"] == "testable").sum())
    n_untestable = int((audit_df["status"] == "untestable").sum())
    n_sig = int((results["sig"] != "").sum())
    ratio = f"{n_sig}/{len(results)}"
    verdict = "PASS" if (len(results) > 0 and n_sig == len(results) and n_untestable == 0) else "FAIL"
    major_summary = summarize_major_step_side(major_side)
    tie_subjects = major_summary["tie_subjects"]
    tie_subjects_str = ", ".join(tie_subjects) if tie_subjects else "(none)"
    result_status = _result_status_map(results)
    analyzed_table = _build_analyzed_variables_table(specs, audit_df, result_status)
    significant_table = _build_significant_table(results)
    joint_total, joint_sig_count, joint_sig_names = _joint_angle_significance(results, _joint_angle_baseline_cols())
    sig_names = results.loc[results["sig"] != "", "dv"].tolist()
    sig_names_str = ", ".join(sig_names) if sig_names else "(none)"
    joint_sig_names_str = ", ".join(joint_sig_names) if joint_sig_names else "(none)"

    comparison_verdict = "Consistent" if joint_sig_count > 0 else "Partially consistent"
    stability_verdict = "Consistent" if any("MOS" in name or "xCOM" in name for name in sig_names) else "Not tested"

    report_text = f"""# Initial Posture Strategy LMM (Baseline Mean Before Platform Onset)

## Research Question

**"Van Wouwe et al. (2021) 관점에서, platform onset 직전 300 ms baseline posture 평균이 step/nonstep 전략 차이를 설명한다면 baseline 변수에서 step/nonstep 차이가 광범위하게 유의한가?"**

이번 버전은 `platform_onset_local` 단일 프레임 대신 onset 전 `[-0.30, 0.00] s` 구간 평균을 사용해 초기 자세를 단일 샘플이 아니라 baseline posture로 요약한다.

## Prior Studies

### Van Wouwe et al. (2021) — Interactions between initial posture and task-level goal explain experimental variability in postural responses to perturbations of standing balance

- **Methodology**: 예측 시뮬레이션 + 실험 데이터 결합. 초기 자세(COM 위치)와 task-level goal(노력-안정성 우선순위)의 상호작용으로 전략 variability를 설명.
- **Experimental design**: 10명의 젊은 성인, 예측 불가능한 backward support-surface translation, stepping/nonstepping 반응 기록.
- **Key results**:
  - 최대 trunk lean 변동성: within-subject mean range 약 `28.3°`, across-subject mean range 약 `39.9°`
  - initial COM position과 maximal trunk lean 관계는 subject-specific (`R^2 = 0.29–0.82`)
  - `xCOM/BOS_onset`, `xCOM/BOS_300ms`를 안정성 지표로 사용
- **Conclusions**: 초기 자세는 intra-subject variability에, task-level goal은 inter-subject 차이에 기여하며, 두 요인 상호작용이 전략 차이를 설명한다.

## Methodological Adaptation

| Prior Method | Current Implementation | Deviation Rationale |
|---|---|---|
| 단일 시점 또는 특정 초기 posture 지표로 전략 variability를 해석 | `time_from_platform_onset_s ∈ [{trial_stats["window_start_s"]:.2f}, {trial_stats["window_end_s"]:.2f}]` 구간 평균을 사용 | 단일 프레임 노이즈를 줄이고 onset 직전 자세를 baseline posture로 요약하기 위해 |
| 초기 posture와 안정성 지표를 함께 해석 | COM/MOS/xCOM-BOS + joint angle + force/torque를 같은 LMM 틀에서 비교 | 동일 조건 step/nonstep 차이를 여러 biomechanical domain에서 동시에 비교하기 위해 |
| 실험/시뮬레이션 기반 posture 정의 | `output/all_trials_timeseries.csv`의 export 값을 frame-wise 평균 | 현재 저장소의 재현 가능한 분석 입력을 유지하고 새 파일 export를 만들지 않기 위해 |

This analysis adopts the prior study's focus on initial posture and stability metrics, but modifies the operational definition from a single onset frame to a 300 ms pre-onset baseline mean because the current repository stores posture signals as onset-aligned timeseries exports.

## Data Summary

- Trials used: **{trial_stats["n_trials_used"]}** (`step={trial_stats["n_step_used"]}`, `nonstep={trial_stats["n_nonstep_used"]}`), subjects={trial_stats["n_subjects_used"]}
- Baseline window: **`[{trial_stats["window_start_s"]:.2f}, {trial_stats["window_end_s"]:.2f}] s`**
- Baseline frames per trial: min=`{trial_stats["baseline_frames_min"]}`, max=`{trial_stats["baseline_frames_max"]}`, mean=`{trial_stats["baseline_frames_mean"]:.2f}`
- Excluded trials without baseline frames: **{trial_stats["n_trials_excluded"]}**
- Input:
  - `output/all_trials_timeseries.csv`
  - `data/perturb_inform.xlsm`
- 분석 변수:
  - baseline 후보 총 **{len(specs)}개**
  - 검정 가능(testable) **{n_testable}개**
  - 검정 불가(untestable) **{n_untestable}개**

## Analysis Methodology

- **Analysis window**: `time_from_platform_onset_s ∈ [{trial_stats["window_start_s"]:.2f}, {trial_stats["window_end_s"]:.2f}]`
- **Statistical model**: `DV ~ step_TF + (1|subject)` (REML, `lmerTest`)
- **Multiple comparison correction**: BH-FDR ({len(specs)}개 baseline 변수 전체 1회)
- **Significance reporting**: `Sig` only (`***`, `**`, `*`, `n.s.`), `alpha=0.05`
- **Displayed result policy**: Results 표에는 **FDR 유의 변수만** 표시

### Axis & Direction Sign

| Axis | Positive (+) | Negative (-) | 대표 변수 |
|---|---|---|---|
| X | exported X-axis positive direction | exported X-axis opposite direction | `COM_X_baseline`, `GRF_X_baseline`, `Hip_stance_X_baseline` |
| Y | exported Y-axis positive direction | exported Y-axis opposite direction | `COM_Y_baseline`, `GRF_Y_baseline`, `Hip_stance_Y_baseline` |
| Z | exported Z-axis positive direction | exported Z-axis opposite direction | `GRF_Z_baseline`, `Trunk_Z_baseline` |

### Signed Metrics Interpretation

| Metric | (+) meaning | (-) meaning | 판정 기준/참조 |
|---|---|---|---|
| `MOS_minDist_signed_baseline` | BOS 안쪽 여유가 큰 방향 | BOS 경계 밖 또는 여유 감소 방향 | exported `MOS_minDist_signed` baseline 평균 |
| `MOS_AP_v3d_baseline` | AP margin이 더 큰 상태 | AP margin이 더 작은 상태 | exported `MOS_AP_v3d` baseline 평균 |
| `MOS_ML_v3d_baseline` | ML margin이 더 큰 상태 | ML margin이 더 작은 상태 | exported `MOS_ML_v3d` baseline 평균 |
| `xCOM_BOS_norm_baseline` | `BOS_minX`에서 `BOS_maxX` 쪽으로 더 큰 상대 위치 | `BOS_minX` 쪽으로 더 작은 상대 위치 | `(xCOM_X - BOS_minX) / (BOS_maxX - BOS_minX)` frame 평균 |

### Joint/Force/Torque Sign Conventions

| Variable group | (+)/(-) meaning | 추가 규칙 |
|---|---|---|
| Joint angle (`*_baseline`) | exported joint-angle axis의 양/음 방향 | frame-wise stance 선택 후 baseline 평균, 추가 C3D 재계산 없음 |
| COP / COM / GRF | exported CSV 축 방향의 양/음 값 | onset-aligned timeseries CSV 값을 직접 평균 |
| `AnkleTorqueMid_Y_perkg_baseline` | exported internal ankle torque Y축 양/음 방향 | `AnkleTorqueMid_int_Y_Nm_per_kg` baseline 평균 |

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

- **Rule**: testable baseline 변수 전부가 FDR 유의여야 PASS
- **Observed**: testable significant ratio = `{ratio}`, untestable=`{n_untestable}`
- **Verdict**: **{verdict}**

### Significant Variables Only (BH-FDR < 0.05)

{significant_table}

## Comparison with Prior Studies

| Comparison Item | Prior Study Result | Current Result | Verdict |
|---|---|---|---|
| Initial posture operationalization | onset/posture-related 초기 상태가 전략 variability를 설명 | onset 직전 300 ms baseline 평균으로 posture를 정의 | Partially consistent |
| Broad posture separation by kinematic variables | initial posture effect가 존재하지만 subject/task interaction이 중요 | baseline joint-angle 유의 변수 `{joint_sig_count}/{joint_total}` (`{joint_sig_names_str}`) | {comparison_verdict} |
| Stability metric relevance | `xCOM/BOS`와 onset stability interpretation 제시 | baseline 유의 변수 `{sig_names_str}` | {stability_verdict} |

## Interpretation & Conclusion

1. baseline 평균으로 초기 자세를 요약해도 strict 기준 가설은 **{verdict}**였다.
2. baseline joint-angle 변수는 총 `{joint_total}`개 중 `{joint_sig_count}`개가 FDR 유의였다.
3. 따라서 본 데이터에서는 onset 직전 baseline posture 차이가 step/nonstep 전략 차이를 완전히 설명한다고 단정하기 어렵고, prior study가 제시한 posture-goal interaction의 일부만 현재 집단 비교에서 포착된 것으로 해석하는 것이 안전하다.

## Limitations

1. 본 분석은 `output/all_trials_timeseries.csv`의 export 값을 평균한 결과로, C3D 재계산 기반 onset absolute 변수와 직접 동일하지 않다.
2. task-level goal 파라미터를 직접 모델링하지 않았다.
3. baseline 평균은 초기 자세를 안정적으로 요약하지만, onset 직전 순간적인 준비 동작은 희석할 수 있다.

## Reproduction

```bash
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_baseline_lmm.py --dry-run
conda run -n module python analysis/initial_posture_strategy_lmm/analyze_initial_posture_strategy_baseline_lmm.py
```

- Output: 콘솔 통계 결과 + 자동 갱신 `report_baseline.md`, `결과) 주제2-Segement Angle_baseline.md`

## Figures

| File | Description |
|---|---|
| (none) | This baseline analysis does not generate figures. |

---
Auto-generated by analyze_initial_posture_strategy_baseline_lmm.py.
"""

    report_md.write_text(report_text, encoding="utf-8-sig")


def write_segment_angle_markdown(
    segment_md: Path,
    results: pd.DataFrame,
    major_side: pd.DataFrame,
    trial_stats: dict[str, Any],
) -> None:
    """Write a baseline-focused segment-angle summary markdown file."""
    baseline_dvs = _joint_angle_baseline_cols()
    table = _build_joint_angle_table(results, baseline_dvs)
    major_summary = summarize_major_step_side(major_side)
    tie_subjects = major_summary["tie_subjects"]
    tie_subjects_str = ", ".join(tie_subjects) if tie_subjects else "(none)"
    joint_total, joint_sig_count, joint_sig_names = _joint_angle_significance(results, baseline_dvs)
    if joint_sig_count == 0:
        joint_note = f"{joint_total}개 baseline segment angle 변수(X/Y/Z) 모두 FDR 보정 후 `n.s.`였다."
    else:
        joint_note = (
            f"{joint_total}개 baseline segment angle 변수(X/Y/Z) 중 {joint_sig_count}개가 FDR 유의였다: "
            f"`{', '.join(joint_sig_names)}`."
        )

    text = f"""---
---
# 가설

1. platform onset 이전 baseline posture에서 nonstep과 step의 관절 각도는 차이가 있을 것이다.

# results

## baseline `[-300 ms, onset]` 평균 LMM

{table}

## coordinate 해석 기준

- 관절각 계산은 `output/all_trials_timeseries.csv`에 export된 `*_deg` 값을 사용한다.
- baseline window는 `time_from_platform_onset_s ∈ [{trial_stats["window_start_s"]:.2f}, {trial_stats["window_end_s"]:.2f}]` 이다.
- `X/Y/Z`는 export된 joint-angle 축 회전 성분이며, 이번 baseline 분석에서는 추가 C3D 재계산 없이 frame-wise 평균만 수행한다.

## stance 기준

- step trial은 `step_r -> 좌측 stance`, `step_l -> 우측 stance`로 계산한다.
- nonstep trial은 subject별 step trial의 `major_step_side`를 stance 기준으로 사용한다.
- `step_r_count == step_l_count`인 tie subject는 좌/우 평균으로 계산한다.
- 이번 실행 요약: `step_r_major={major_summary["step_r_major"]}`, `step_l_major={major_summary["step_l_major"]}`, `tie={major_summary["tie_major"]}` (tie subjects: `{tie_subjects_str}`)

## baseline window 요약

- baseline window: `[{trial_stats["window_start_s"]:.2f}, {trial_stats["window_end_s"]:.2f}] s`
- baseline 기준 유효 trial: `{trial_stats["n_trials_used"]}/{trial_stats["n_trials_total"]}` (step=`{trial_stats["n_step_used"]}`, nonstep=`{trial_stats["n_nonstep_used"]}`)
- 제외 trial: `{trial_stats["n_trials_excluded"]}`
- baseline frame 수: min=`{trial_stats["baseline_frames_min"]}`, max=`{trial_stats["baseline_frames_max"]}`, mean=`{trial_stats["baseline_frames_mean"]:.2f}`

## 결과 해석

- baseline 관절각 요약: {joint_note}
- 모든 축이 일관되게 유의하지 않다면, baseline posture만으로 전략 차이를 설명하는 근거는 제한적이다.
- 본 결과는 onset 직전 single-frame 차이와는 다른 질문을 다루며, "초기 자세를 평균 posture로 보았을 때도 차이가 남는가"를 확인하는 데 의미가 있다.

# 결론

- baseline 평균 관절각 비교 기준으로, 관절각만으로 전략 차이를 단정하기에는 근거가 제한적일 수 있다.
- 최종 해석은 `report_baseline.md`의 balance/force 변수와 함께 읽어야 한다.

# keypapers

1. Van Wouwe et al. (2021): 초기 자세와 전략 variability의 상호작용을 제시했으며, 본 분석은 onset 단일 프레임 대신 onset 전 300 ms baseline 평균으로 그 질문을 다시 검토한다.

---
Auto-generated by analyze_initial_posture_strategy_baseline_lmm.py.
"""
    segment_md.write_text(text, encoding="utf-8-sig")


def build_analysis_dataframe(
    csv_path: Path,
    platform_xlsm: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load inputs and build the per-trial baseline analysis dataframe."""
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

    print("  Building baseline window means from exported CSV...")
    analysis_df, used_trial_meta, baseline_stats = build_baseline_dataframe(
        df=df,
        trial_meta=trial_meta,
        major_side=major_side,
    )
    if analysis_df.empty:
        raise ValueError("No valid trials remained after baseline window aggregation.")

    baseline_stats["n_subjects_used"] = int(used_trial_meta["subject"].nunique())
    return analysis_df, used_trial_meta, major_side, baseline_stats


def main() -> None:
    args = parse_args()
    _ = args.no_figures  # compatibility arg; intentionally unused

    print("=" * 72)
    print("Initial Posture Strategy LMM - Baseline Mean")
    print("=" * 72)

    print("\n[M1] Load and prepare data...")
    analysis_df, trial_meta, major_side, baseline_stats = build_analysis_dataframe(
        csv_path=args.csv,
        platform_xlsm=args.platform_xlsm,
    )

    specs = variable_catalog()
    dv_to_family = {s["dv"]: s["family"] for s in specs}

    print("\n[M2] Variable audit at baseline...")
    audit_df = audit_variables(analysis_df=analysis_df, specs=specs)
    n_total = len(audit_df)
    n_testable = int((audit_df["status"] == "testable").sum())
    n_untestable = int((audit_df["status"] == "untestable").sum())
    print(f"  Variables total={n_total}, testable={n_testable}, untestable={n_untestable}")

    untestable = audit_df[audit_df["status"] == "untestable"].copy()
    if not untestable.empty:
        by_reason = untestable.groupby("reason").size().to_dict()
        print(f"  Untestable reasons: {by_reason}")

    print(
        "  Baseline window: "
        f"[{baseline_stats['window_start_s']:.2f}, {baseline_stats['window_end_s']:.2f}] s"
    )
    print(
        "  Baseline frames per trial: "
        f"min={baseline_stats['baseline_frames_min']}, "
        f"max={baseline_stats['baseline_frames_max']}, "
        f"mean={baseline_stats['baseline_frames_mean']:.2f}"
    )
    print(
        "  Trial usage: "
        f"used={baseline_stats['n_trials_used']}/{baseline_stats['n_trials_total']} "
        f"(step={baseline_stats['n_step_used']}, nonstep={baseline_stats['n_nonstep_used']})"
    )

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

    print("\n[M4] Refresh markdown outputs...")
    write_report_markdown(
        report_md=args.report_md,
        specs=specs,
        audit_df=audit_df,
        results=results,
        major_side=major_side,
        trial_stats=baseline_stats,
    )
    write_segment_angle_markdown(
        segment_md=args.segment_angle_md,
        results=results,
        major_side=major_side,
        trial_stats=baseline_stats,
    )
    print(f"  Updated: {args.report_md}")
    print(f"  Updated: {args.segment_angle_md}")


if __name__ == "__main__":
    main()
