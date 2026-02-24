"""xCOM/BOS Normalization Step-vs-Nonstep LMM Analysis.

Answers:
"Can xCOM/BOS-family normalized metrics from prior-study methods
(statistically consolidated) distinguish step vs nonstep strategies?"

Statistical method: Linear Mixed Model (LMM) with lmerTest inference
  Model: DV ~ step_TF + (1|subject)
  Multiple comparison: Benjamini-Hochberg FDR across DVs
  Analysis events: platform onset and step onset

Produces:
  - 3 publication-quality figures (saved alongside this script)
  - stdout summary statistics

Usage:
    conda run --no-capture-output -n module python \
      analysis/xCOM&BOS_normalization/analyze_xcom_bos_normalization_lmm.py

    conda run --no-capture-output -n module python \
      analysis/xCOM&BOS_normalization/analyze_xcom_bos_normalization_lmm.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# path bootstrap (replaces _bootstrap dependency)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests

# Required display config by repo rule
pl.Config.set_tbl_rows(999)
pl.Config.set_tbl_cols(999)
pl.Config.set_tbl_width_chars(120)

# Korean font support
_KO_FONTS = ("Malgun Gothic", "NanumGothic", "AppleGothic")
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _fname in _KO_FONTS:
    if _fname in _available:
        plt.rcParams["font.family"] = _fname
        break
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_OUT_DIR = SCRIPT_DIR

TRIAL_KEYS = ["subject", "velocity", "trial"]

G = 9.81
FRAME_RATE_HZ = 100.0
DT = 1.0 / FRAME_RATE_HZ

COLORS = {"step": "#E74C3C", "nonstep": "#3498DB"}
EVENT_COLORS = {"platform_onset": "#2ecc71", "step_onset": "#3498db"}

# ---------------------------------------------------------------------------
# R configuration (subprocess-based, bypasses broken rpy2 on Windows)
# ---------------------------------------------------------------------------
WINDOWS_R_HOME = Path(r"C:\Users\Alice\miniconda3\envs\module\lib\R")
WINDOWS_RSCRIPT = WINDOWS_R_HOME / "bin" / "x64" / "Rscript.exe"


def _candidate_rscripts() -> list[Path]:
    candidates: list[Path] = []

    which_r = shutil.which("Rscript")
    if which_r:
        candidates.append(Path(which_r))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cp = Path(conda_prefix)
        candidates.extend(
            [
                cp / "bin" / "Rscript",
                cp / "Scripts" / "Rscript.exe",
                cp / "lib" / "R" / "bin" / "Rscript",
                cp / "lib" / "R" / "bin" / "x64" / "Rscript.exe",
            ]
        )

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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="Only load data; skip analysis")
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation and keep existing figure files untouched.",
    )
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
        "platform_onset",
        "step_onset",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"platform sheet missing required columns: {missing}")

    out = df[["subject", "velocity", "trial", "step_TF", "platform_onset", "step_onset"]].copy()
    out = _normalize_trial_types(out)
    out["step_TF"] = out["step_TF"].astype(str).str.strip().str.lower()
    out["platform_onset"] = pd.to_numeric(out["platform_onset"], errors="coerce")
    out["step_onset"] = pd.to_numeric(out["step_onset"], errors="coerce")
    return out


def load_anthropometrics_from_meta(path: Path) -> pd.DataFrame:
    """Extract anthropometrics from row-based `meta` sheet.

    The `meta` sheet keeps variables as rows and subjects as columns.
    We reconstruct subject-wise values for 키, 다리길이, 발길이_왼, 발길이_오른.
    """
    raw = pd.read_excel(str(path), sheet_name="meta")
    label_col = raw.columns[0]
    raw[label_col] = raw[label_col].astype(str).str.strip()

    required_rows = ["키", "다리길이", "발길이_왼", "발길이_오른"]
    sub = raw[raw[label_col].isin(required_rows)].copy()
    missing_rows = [x for x in required_rows if x not in sub[label_col].tolist()]
    if missing_rows:
        raise ValueError(f"meta sheet missing required anthropometric rows: {missing_rows}")

    records: list[dict] = []
    for _, row in sub.iterrows():
        metric = str(row[label_col]).strip()
        for subject, value in row.iloc[1:].items():
            records.append(
                {
                    "subject": str(subject).strip(),
                    "metric": metric,
                    "value": value,
                }
            )

    long_df = pd.DataFrame(records)
    wide = (
        long_df.pivot_table(index="subject", columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for col in required_rows:
        wide[col] = pd.to_numeric(wide[col], errors="coerce")

    wide = wide.rename(
        columns={
            "키": "height_cm",
            "다리길이": "leg_len_cm",
            "발길이_왼": "foot_len_left_mm",
            "발길이_오른": "foot_len_right_mm",
        }
    )

    wide["foot_len_m"] = ((wide["foot_len_left_mm"] + wide["foot_len_right_mm"]) / 2.0) / 1000.0
    wide["height_m"] = wide["height_cm"] / 100.0
    wide["leg_len_m"] = wide["leg_len_cm"] / 100.0

    return wide[
        [
            "subject",
            "height_cm",
            "leg_len_cm",
            "foot_len_left_mm",
            "foot_len_right_mm",
            "height_m",
            "leg_len_m",
            "foot_len_m",
        ]
    ].copy()


def _compute_trial_event_frames(
    frame_bounds: pd.DataFrame,
    platform: pd.DataFrame,
) -> pd.DataFrame:
    """Compute platform/step event frames in local MocapFrame domain.

    step onset policy:
      - step trial: actual step_onset_local
      - nonstep trial: subject-velocity mean step_onset_local from step trials
      - fallback: prefilter platform subject-velocity step mean (raw events converted to local)
    """
    trials = frame_bounds.copy()

    # Join event/step label from platform metadata
    plat = (
        platform[TRIAL_KEYS + ["step_TF", "platform_onset", "step_onset"]]
        .drop_duplicates()
        .copy()
    )
    trials = trials.merge(plat, on=TRIAL_KEYS, how="left")
    trials = trials[trials["step_TF"].isin(["step", "nonstep"])].reset_index(drop=True)

    trials["step_onset_eval"] = np.nan
    step_mask = (trials["step_TF"] == "step") & trials["step_onset_local"].notna()
    trials.loc[step_mask, "step_onset_eval"] = trials.loc[step_mask, "step_onset_local"]

    step_means = (
        trials.loc[step_mask]
        .groupby(["subject", "velocity"], as_index=False)["step_onset_local"]
        .mean()
        .rename(columns={"step_onset_local": "mean_step_onset_sv"})
    )

    platform_onset_local_ref_values = (
        trials["platform_onset_local"].dropna().round().astype("Int64").dropna().astype(int)
    )
    if len(platform_onset_local_ref_values) == 0:
        platform_onset_local_ref = 101
    else:
        platform_onset_local_ref = int(pd.Series(platform_onset_local_ref_values).mode().iloc[0])

    prefilter_step_means = (
        platform.loc[
            (platform["step_TF"] == "step")
            & platform["step_onset"].notna()
            & platform["platform_onset"].notna(),
            ["subject", "velocity", "step_onset", "platform_onset"],
        ]
        .assign(
            step_onset_local_prefilter=lambda x: (
                x["step_onset"] - x["platform_onset"] + float(platform_onset_local_ref)
            )
        )
        .groupby(["subject", "velocity"], as_index=False)["step_onset_local_prefilter"]
        .mean()
    )

    # Trial-level prefilter fallback reconstructed from raw platform events.
    trials["step_onset_local_prefilter_trial"] = np.where(
        trials["step_onset"].notna() & trials["platform_onset"].notna(),
        trials["step_onset"] - trials["platform_onset"] + float(platform_onset_local_ref),
        np.nan,
    )
    fill_mask_trial_pref = trials["step_onset_eval"].isna() & trials["step_onset_local_prefilter_trial"].notna()
    trials.loc[fill_mask_trial_pref, "step_onset_eval"] = trials.loc[
        fill_mask_trial_pref, "step_onset_local_prefilter_trial"
    ]
    filled_trial_pref = int(fill_mask_trial_pref.sum())

    trials = trials.merge(step_means, on=["subject", "velocity"], how="left")
    needs_fill = trials["step_onset_eval"].isna()
    fill_mask_sv = needs_fill & trials["mean_step_onset_sv"].notna()
    trials.loc[fill_mask_sv, "step_onset_eval"] = trials.loc[fill_mask_sv, "mean_step_onset_sv"]
    filled_sv = int(fill_mask_sv.sum())

    trials = trials.merge(prefilter_step_means, on=["subject", "velocity"], how="left")
    needs_fill2 = trials["step_onset_eval"].isna()
    fill_mask_pref = needs_fill2 & trials["step_onset_local_prefilter"].notna()
    trials.loc[fill_mask_pref, "step_onset_eval"] = trials.loc[fill_mask_pref, "step_onset_local_prefilter"]
    filled_pref = int(fill_mask_pref.sum())

    trials["platform_eval_frame"] = trials["platform_onset_local"].round()
    trials["step_onset_eval"] = trials["step_onset_eval"].round()

    # Clean nullable ints for stable joins
    trials["platform_eval_frame"] = pd.to_numeric(trials["platform_eval_frame"], errors="coerce").astype("Int64")
    trials["step_onset_eval"] = pd.to_numeric(trials["step_onset_eval"], errors="coerce").astype("Int64")

    missing_after_fill = int(trials["step_onset_eval"].isna().sum())
    if missing_after_fill > 0:
        print(f"  Warning: {missing_after_fill} trials still missing step_onset_eval (dropped)")
        trials = trials.dropna(subset=["step_onset_eval"]).reset_index(drop=True)

    oob_platform = (
        (trials["platform_eval_frame"] < trials["frame_min"])
        | (trials["platform_eval_frame"] > trials["frame_max"])
    ).sum()
    oob_step = (
        (trials["step_onset_eval"] < trials["frame_min"])
        | (trials["step_onset_eval"] > trials["frame_max"])
    ).sum()
    if int(oob_platform) > 0 or int(oob_step) > 0:
        raise ValueError(
            "Event frame out of range: "
            f"platform={int(oob_platform)}, step_onset={int(oob_step)}"
        )

    print(f"  step_onset fill (trial-level prefilter): {filled_trial_pref}")
    print(f"  step_onset fill (subject-velocity mean): {filled_sv}")
    print(f"  step_onset fill (prefilter platform subject-velocity mean): {filled_pref}")
    print(f"  Event range validation passed: platform_oob={int(oob_platform)}, step_oob={int(oob_step)}")

    return trials[
        TRIAL_KEYS
        + [
            "step_TF",
            "frame_min",
            "frame_max",
            "platform_eval_frame",
            "step_onset_eval",
            "platform_onset_local",
            "step_onset_local",
        ]
    ].copy()


def build_dv_specs() -> list[dict]:
    return [
        {
            "dv": "DV1_xcom_hof_rear_over_foot_platformonset",
            "frame_col": "dv1_xcom_hof_rear_over_foot",
            "event_col": "platform_eval_frame",
            "metric": "DV1_xcom_hof_rear_over_foot",
            "event": "platform_onset",
            "paper_group": "VanWouwe+Salot+Patel+Bhatt",
        },
        {
            "dv": "DV1_xcom_hof_rear_over_foot_steponset",
            "frame_col": "dv1_xcom_hof_rear_over_foot",
            "event_col": "step_onset_eval",
            "metric": "DV1_xcom_hof_rear_over_foot",
            "event": "step_onset",
            "paper_group": "VanWouwe+Salot+Patel+Bhatt",
        },
        {
            "dv": "DV2_com_rear_over_foot_platformonset",
            "frame_col": "dv2_com_rear_over_foot",
            "event_col": "platform_eval_frame",
            "metric": "DV2_com_rear_over_foot",
            "event": "platform_onset",
            "paper_group": "Joshi_position",
        },
        {
            "dv": "DV2_com_rear_over_foot_steponset",
            "frame_col": "dv2_com_rear_over_foot",
            "event_col": "step_onset_eval",
            "metric": "DV2_com_rear_over_foot",
            "event": "step_onset",
            "paper_group": "Joshi_position",
        },
        {
            "dv": "DV3_vcom_rel_over_sqrtgh_platformonset",
            "frame_col": "dv3_vcom_rel_over_sqrtgh",
            "event_col": "platform_eval_frame",
            "metric": "DV3_vcom_rel_over_sqrtgh",
            "event": "platform_onset",
            "paper_group": "Joshi_velocity",
        },
        {
            "dv": "DV3_vcom_rel_over_sqrtgh_steponset",
            "frame_col": "dv3_vcom_rel_over_sqrtgh",
            "event_col": "step_onset_eval",
            "metric": "DV3_vcom_rel_over_sqrtgh",
            "event": "step_onset",
            "paper_group": "Joshi_velocity",
        },
    ]


def load_and_prepare(csv_path: Path, xlsm_path: Path, dv_specs: list[dict]) -> tuple[pd.DataFrame, dict]:
    print("  Loading CSV...")
    df = load_csv(csv_path)
    n_frames = len(df)

    required_csv_cols = {
        "subject",
        "velocity",
        "trial",
        "MocapFrame",
        "platform_onset_local",
        "step_onset_local",
        "COM_X",
        "vCOM_X",
        "BOS_minX",
    }
    missing_csv = sorted(required_csv_cols - set(df.columns))
    if missing_csv:
        raise ValueError(f"CSV missing required columns: {missing_csv}")

    print("  Loading platform metadata...")
    platform = load_platform_sheet(xlsm_path)

    print("  Loading anthropometrics from meta sheet...")
    anthro = load_anthropometrics_from_meta(xlsm_path)

    print("  Building trial frame bounds...")
    frame_bounds = (
        df.select(TRIAL_KEYS + ["MocapFrame", "platform_onset_local", "step_onset_local"])
        .group_by(TRIAL_KEYS)
        .agg(
            pl.col("MocapFrame").min().alias("frame_min"),
            pl.col("MocapFrame").max().alias("frame_max"),
            pl.col("platform_onset_local").drop_nulls().first().alias("platform_onset_local"),
            pl.col("step_onset_local").drop_nulls().first().alias("step_onset_local"),
        )
        .sort(TRIAL_KEYS)
        .to_pandas()
    )
    frame_bounds = _normalize_trial_types(frame_bounds)
    frame_bounds["frame_min"] = pd.to_numeric(frame_bounds["frame_min"], errors="coerce").astype("Int64")
    frame_bounds["frame_max"] = pd.to_numeric(frame_bounds["frame_max"], errors="coerce").astype("Int64")
    frame_bounds["platform_onset_local"] = pd.to_numeric(
        frame_bounds["platform_onset_local"], errors="coerce"
    )
    frame_bounds["step_onset_local"] = pd.to_numeric(frame_bounds["step_onset_local"], errors="coerce")

    print("  Computing trial event frames (platform/step onset)...")
    trial_meta = _compute_trial_event_frames(frame_bounds, platform)

    print("  Joining anthropometrics...")
    trial_meta = trial_meta.merge(anthro, on="subject", how="left")
    for col in ["height_m", "leg_len_m", "foot_len_m"]:
        trial_meta[col] = pd.to_numeric(trial_meta[col], errors="coerce")

    bad_den = trial_meta[
        trial_meta[["height_m", "leg_len_m", "foot_len_m"]].isna().any(axis=1)
        | (trial_meta["height_m"] <= 0)
        | (trial_meta["leg_len_m"] <= 0)
        | (trial_meta["foot_len_m"] <= 0)
    ]
    if not bad_den.empty:
        bad_subj = ", ".join(sorted(bad_den["subject"].astype(str).unique().tolist()))
        raise ValueError(
            "Invalid anthropometrics for analysis subjects. "
            f"subjects={bad_subj}; ensure meta sheet contains valid 키/다리길이/발길이 values."
        )

    print("  Building frame-wise normalized variables...")
    trial_meta_pl = pl.from_pandas(trial_meta)
    frame_df = (
        df.with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial").cast(pl.Int64, strict=False),
        )
        .join(trial_meta_pl, on=TRIAL_KEYS, how="inner")
        .with_columns(
            (pl.lit(G) / pl.col("leg_len_m")).sqrt().alias("omega0"),
            # rear-of-BOS velocity in AP direction (framewise finite difference)
            (pl.col("BOS_minX").cast(pl.Float64).diff().over(TRIAL_KEYS) / DT)
            .fill_null(0.0)
            .alias("v_bos_rear"),
        )
        .with_columns(
            (pl.col("COM_X") + (pl.col("vCOM_X") / pl.col("omega0"))).alias("xcom_hof"),
            pl.col("BOS_minX").alias("bos_rear"),
        )
        .with_columns(
            pl.when(pl.col("foot_len_m") > 0)
            .then((pl.col("xcom_hof") - pl.col("bos_rear")) / pl.col("foot_len_m"))
            .otherwise(None)
            .alias("dv1_xcom_hof_rear_over_foot"),
            pl.when(pl.col("foot_len_m") > 0)
            .then((pl.col("COM_X") - pl.col("bos_rear")) / pl.col("foot_len_m"))
            .otherwise(None)
            .alias("dv2_com_rear_over_foot"),
            pl.when(pl.col("height_m") > 0)
            .then((pl.col("vCOM_X") - pl.col("v_bos_rear")) / (pl.lit(G) * pl.col("height_m")).sqrt())
            .otherwise(None)
            .alias("dv3_vcom_rel_over_sqrtgh"),
        )
    )

    # Aggregate event-point values
    agg_exprs: list[pl.Expr] = []
    for spec in dv_specs:
        agg_exprs.append(
            pl.col(spec["frame_col"])
            .filter(pl.col("MocapFrame") == pl.col(spec["event_col"]))
            .drop_nulls()
            .first()
            .alias(spec["dv"])
        )

    trial_values = frame_df.group_by(TRIAL_KEYS).agg(agg_exprs).sort(TRIAL_KEYS).to_pandas()
    trial_values = _normalize_trial_types(trial_values)

    trial_df = trial_values.merge(
        trial_meta[TRIAL_KEYS + ["step_TF"]],
        on=TRIAL_KEYS,
        how="left",
    )

    trial_df = trial_df[trial_df["step_TF"].isin(["step", "nonstep"])].reset_index(drop=True)
    if trial_df[TRIAL_KEYS].isna().any().any():
        raise ValueError("Null key values detected in trial-level table.")

    n_step = int((trial_df["step_TF"] == "step").sum())
    n_nonstep = int((trial_df["step_TF"] == "nonstep").sum())
    print(f"  Frames: {n_frames}")
    print(f"  Trials: {len(trial_df)} (step={n_step}, nonstep={n_nonstep})")

    for spec in dv_specs:
        miss = int(trial_df[spec["dv"]].isna().sum())
        print(f"  {spec['dv']} missing: {miss}")

    summary = {
        "n_frames": n_frames,
        "n_trials": int(len(trial_df)),
        "n_step": n_step,
        "n_nonstep": n_nonstep,
        "velocity_mean": float(trial_df["velocity"].mean()),
        "velocity_sd": float(trial_df["velocity"].std(ddof=1)),
        "n_subjects": int(trial_df["subject"].nunique()),
    }
    return trial_df, summary


def _build_r_lmm_script(csv_path: str, output_path: str) -> str:
    return f"""
library(lmerTest)

data <- read.csv("{csv_path}", stringsAsFactors = FALSE)
data$subject <- as.factor(data$subject)
data$step_TF <- factor(data$step_TF, levels = c("nonstep", "step"))

dv_cols <- colnames(data)[!colnames(data) %in% c("subject", "velocity", "trial", "step_TF")]

results <- data.frame(
  dv = character(),
  term = character(),
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

append_row <- function(dv, term, est, se, df_val, t_val, p_val, m_s, sd_s, m_ns, sd_ns, n_s, n_ns, conv) {{
  results <<- rbind(results, data.frame(
    dv = dv,
    term = term,
    estimate = est,
    SE = se,
    df = df_val,
    t_value = t_val,
    p_value = p_val,
    mean_step = m_s,
    sd_step = sd_s,
    mean_nonstep = m_ns,
    sd_nonstep = sd_ns,
    n_step = n_s,
    n_nonstep = n_ns,
    converged = conv,
    stringsAsFactors = FALSE
  ))
}}

for (dv in dv_cols) {{
  formula_str <- paste0("`", dv, "` ~ step_TF + (1|subject)")
  sub <- data[!is.na(data[[dv]]) & !is.na(data$step_TF), ]

  n_s <- sum(sub$step_TF == "step")
  n_ns <- sum(sub$step_TF == "nonstep")
  m_s <- mean(sub[[dv]][sub$step_TF == "step"], na.rm = TRUE)
  sd_s <- sd(sub[[dv]][sub$step_TF == "step"], na.rm = TRUE)
  m_ns <- mean(sub[[dv]][sub$step_TF == "nonstep"], na.rm = TRUE)
  sd_ns <- sd(sub[[dv]][sub$step_TF == "nonstep"], na.rm = TRUE)

  tryCatch({{
    m <- lmer(as.formula(formula_str), data = sub, REML = TRUE)
    co <- coef(summary(m))
    rn <- rownames(co)

    row_main <- if ("step_TFstep" %in% rn) "step_TFstep" else NA

    if (!is.na(row_main)) {{
      append_row(
        dv, "main_step_effect",
        co[row_main, "Estimate"],
        co[row_main, "Std. Error"],
        co[row_main, "df"],
        co[row_main, "t value"],
        co[row_main, "Pr(>|t|)"],
        m_s, sd_s, m_ns, sd_ns, n_s, n_ns, TRUE
      )
    }} else {{
      append_row(dv, "main_step_effect", NA, NA, NA, NA, NA, m_s, sd_s, m_ns, sd_ns, n_s, n_ns, FALSE)
    }}
  }}, error = function(e) {{
    append_row(dv, "main_step_effect", NA, NA, NA, NA, NA, m_s, sd_s, m_ns, sd_ns, n_s, n_ns, FALSE)
  }})
}}

write.csv(results, "{output_path}", row.names = FALSE)
cat("LMM fitting complete:", nrow(results), "rows\\n")
"""


def fit_lmm_all(trial_df: pd.DataFrame, dv_specs: list[dict]) -> pd.DataFrame:
    dv_names = [x["dv"] for x in dv_specs]
    export_cols = ["subject", "velocity", "trial", "step_TF"] + dv_names
    export_df = trial_df[export_cols].copy()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir=tempfile.gettempdir(), newline=""
    ) as f:
        export_df.to_csv(f, index=False)
        data_csv = f.name

    result_csv = tempfile.mktemp(suffix="_xcom_bos_lmm_results.csv", dir=tempfile.gettempdir())

    r_code = _build_r_lmm_script(
        data_csv.replace("\\", "/"),
        result_csv.replace("\\", "/"),
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".R", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(r_code)
        r_script = f.name

    try:
        rscript_cmd, r_env = resolve_r_runtime()
        print("  Running Rscript for step-effect LMM fitting...")
        print(f"  Using Rscript: {rscript_cmd}")
        proc = subprocess.run(
            [rscript_cmd, r_script],
            capture_output=True,
            text=True,
            timeout=600,
            env=r_env,
        )
        if proc.returncode != 0:
            print(f"  R stderr: {proc.stderr[:1200]}")
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

    spec_meta = {
        s["dv"]: {
            "metric": s["metric"],
            "event": s["event"],
            "paper_group": s["paper_group"],
        }
        for s in dv_specs
    }
    results["metric"] = results["dv"].map(lambda x: spec_meta[x]["metric"])
    results["event"] = results["dv"].map(lambda x: spec_meta[x]["event"])
    results["paper_group"] = results["dv"].map(lambda x: spec_meta[x]["paper_group"])

    results["p_fdr"] = np.nan
    for term in results["term"].unique():
        mask = (results["term"] == term) & results["p_value"].notna()
        if int(mask.sum()) == 0:
            continue
        _, p_adj, _, _ = multipletests(results.loc[mask, "p_value"].values, method="fdr_bh")
        results.loc[mask, "p_fdr"] = p_adj

    def _sig(p: float) -> str:
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


def _event_label(v: str) -> str:
    return "platform_onset" if v == "platform_onset" else "step_onset"


def _metric_label(metric: str) -> str:
    mapping = {
        "DV1_xcom_hof_rear_over_foot": "DV1 xCOM_hof-rear / foot_len",
        "DV2_com_rear_over_foot": "DV2 COM-rear / foot_len",
        "DV3_vcom_rel_over_sqrtgh": "DV3 (vCOM-vBOSrear) / sqrt(g*h)",
    }
    return mapping.get(metric, metric)


def print_results_table(results: pd.DataFrame) -> None:
    print("\n" + "=" * 96)
    print("LMM Results: DV ~ step_TF + (1|subject), REML")
    print("FDR: BH across all DVs")
    print("=" * 96)
    fmt = "{:<44s} {:<28s} {:>10s} {:>9s} {:>8s} {:>6s}"
    print(fmt.format("DV", "Term", "Estimate", "SE", "t", "Sig"))
    print("-" * 96)

    for term in ["main_step_effect"]:
        sub = results[results["term"] == term].copy()
        sub = sub.sort_values(["metric", "event", "dv"]).reset_index(drop=True)
        for _, row in sub.iterrows():
            est = f"{row['estimate']:.4f}" if pd.notna(row["estimate"]) else "FAIL"
            se = f"{row['SE']:.4f}" if pd.notna(row["SE"]) else ""
            tv = f"{row['t_value']:.3f}" if pd.notna(row["t_value"]) else ""
            sig = row["sig"] if row["sig"] else "n.s."
            print(fmt.format(row["dv"][:44], term, est, se, tv, sig))
        print("-" * 96)

    n_sig = int((results["sig"] != "").sum())
    n_total = int(len(results))
    print(f"\nFDR significant coefficients: {n_sig}/{n_total}")


def _forest_plot(results: pd.DataFrame, term: str, out_path: Path, title: str) -> None:
    sub = results[(results["term"] == term) & (results["converged"] == True)].copy()  # noqa: E712
    if sub.empty:
        return
    sub = sub.sort_values(["event", "metric", "dv"]).reset_index(drop=True)
    sub["ci_lo"] = sub["estimate"] - 1.96 * sub["SE"]
    sub["ci_hi"] = sub["estimate"] + 1.96 * sub["SE"]
    sub["label"] = sub.apply(
        lambda r: f"{_metric_label(r['metric'])} @ {_event_label(r['event'])}",
        axis=1,
    )

    n = len(sub)
    fig_h = max(5, n * 0.55 + 1)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    for i, (_, row) in enumerate(sub.iterrows()):
        color = EVENT_COLORS.get(row["event"], "gray")
        marker = "o" if row["sig"] else "d"
        alpha = 1.0 if row["sig"] else 0.45
        lw = 2.2 if row["sig"] else 1.1

        ax.errorbar(
            row["estimate"],
            i,
            xerr=[[row["estimate"] - row["ci_lo"]], [row["ci_hi"] - row["estimate"]]],
            fmt=marker,
            color=color,
            alpha=alpha,
            markersize=6,
            capsize=3,
            linewidth=lw,
        )
        weight = "bold" if row["sig"] else "normal"
        ax.text(
            -0.01,
            i,
            row["label"],
            ha="right",
            va="center",
            fontsize=8,
            fontweight=weight,
            transform=ax.get_yaxis_transform(),
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([""] * n)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient estimate", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=EVENT_COLORS["platform_onset"], label="platform_onset"),
        Patch(facecolor=EVENT_COLORS["step_onset"], label="step_onset"),
        Patch(facecolor="white", edgecolor="black", label="● = FDR sig, ◆ = n.s."),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.subplots_adjust(left=0.46)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_violin_significant(
    trial_df: pd.DataFrame,
    results: pd.DataFrame,
    out_path: Path,
) -> None:
    import seaborn as sns

    sig_dvs = results[results["sig"] != ""]["dv"].drop_duplicates().tolist()
    if not sig_dvs:
        sig_dvs = (
            results.dropna(subset=["p_value"])
            .sort_values("p_value")
            ["dv"]
            .drop_duplicates()
            .head(4)
            .tolist()
        )

    if not sig_dvs:
        return

    np.random.seed(0)

    n_vars = min(len(sig_dvs), 6)
    sig_dvs = sig_dvs[:n_vars]
    ncols = min(3, n_vars)
    nrows = int(np.ceil(n_vars / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows), squeeze=False)

    for idx, dv in enumerate(sig_dvs):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        valid = trial_df[trial_df["step_TF"].isin(["step", "nonstep"])].dropna(subset=[dv])
        sns.violinplot(
            data=valid,
            x="step_TF",
            y=dv,
            hue="step_TF",
            order=["step", "nonstep"],
            hue_order=["step", "nonstep"],
            palette=[COLORS["step"], COLORS["nonstep"]],
            inner=None,
            alpha=0.28,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=valid,
            x="step_TF",
            y=dv,
            hue="step_TF",
            order=["step", "nonstep"],
            hue_order=["step", "nonstep"],
            palette=[COLORS["step"], COLORS["nonstep"]],
            size=3,
            alpha=0.6,
            jitter=0.2,
            legend=False,
            ax=ax,
        )

        rows = results[results["dv"] == dv]
        sig_main = rows.loc[rows["term"] == "main_step_effect", "sig"]
        main_txt = sig_main.iloc[0] if len(sig_main) > 0 and sig_main.iloc[0] else "n.s."
        ax.set_title(f"{dv}\nmain={main_txt}", fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("value", fontsize=8)

    for idx in range(n_vars, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Significant (or top-p) DVs: step vs nonstep",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dv_specs = build_dv_specs()

    print("=" * 72)
    print("xCOM/BOS Normalization Step-vs-Nonstep LMM")
    print("=" * 72)

    print("\n[M1] Load and prepare data...")
    trial_df, data_summary = load_and_prepare(args.csv, args.platform_xlsm, dv_specs)

    if args.dry_run:
        print(
            "\nDry run complete. "
            f"Trials={data_summary['n_trials']} (step={data_summary['n_step']}, nonstep={data_summary['n_nonstep']}), "
            f"subjects={data_summary['n_subjects']}."
        )
        return

    print("\n[M2] Fit LMMs...")
    results = fit_lmm_all(trial_df, dv_specs)
    print_results_table(results)

    if args.no_figures:
        print("\n[M3] Skipping figure generation (--no-figures).")
    else:
        print("\n[M3] Generate figures...")
        _forest_plot(
            results,
            term="main_step_effect",
            out_path=out_dir / "fig1_main_effect_forest.png",
            title="Main effect: step_TFstep",
        )
        print("  fig1_main_effect_forest.png")

        legacy_interaction_fig = out_dir / "fig2_interaction_forest.png"
        if legacy_interaction_fig.exists():
            legacy_interaction_fig.unlink()
            print("  removed legacy fig2_interaction_forest.png")

        fig3_violin_significant(
            trial_df,
            results,
            out_path=out_dir / "fig3_violin_significant.png",
        )
        print("  fig3_violin_significant.png")

    print("\n" + "=" * 72)
    print("Analysis complete.")
    print(f"Output directory: {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
