"""Step vs. Non-step Biomechanical LMM Analysis.

Answers: "Do biomechanical variables differ between step and non-step
balance recovery strategies under identical perturbation intensity?"

Statistical method: Linear Mixed Model (LMM) with Satterthwaite df
  Model: DV ~ step_TF + (1|subject)
  Multiple comparison: Benjamini-Hochberg FDR per variable family
  Analysis window: [platform_onset, step_onset] per trial
    - step trials: actual step_onset_local
    - nonstep trials: mean step_onset of same (subject, velocity) step trials

Produces:
  - 3 publication-quality figures (saved alongside this script)
  - stdout summary statistics

Usage:
    conda run -n module python analysis/step_vs_nonstep_lmm/analyze_step_vs_nonstep_lmm.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# path bootstrap (replaces _bootstrap dependency)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")

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

# ---------------------------------------------------------------------------
# R configuration (subprocess-based, bypasses broken rpy2 on Windows)
# ---------------------------------------------------------------------------
R_HOME = Path(r"C:\Users\Alice\miniconda3\envs\module\lib\R")
R_BIN = R_HOME / "bin" / "x64"
RSCRIPT = str(R_BIN / "Rscript.exe")


def _r_env() -> dict:
    """Build environment dict with R paths for subprocess calls."""
    env = os.environ.copy()
    extra = [
        str(R_BIN),
        r"C:\Users\Alice\miniconda3\envs\module\Library\bin",
        r"C:\Users\Alice\miniconda3\envs\module\Library\mingw-w64\bin",
    ]
    env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
    env["R_HOME"] = str(R_HOME)
    return env


# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

# (dv_name, source_col, aggregation_type)
# aggregation types: range, path_length, abs_peak, min_val, abs_peak_velocity
_COM_AXES = ["X", "Y"]
_COP_AXES = ["X", "Y"]
_GRF_AXES = ["X", "Y", "Z"]
_JOINT_NAMES = [
    ("Hip_R", "Hip_R_X_deg"),
    ("Knee_R", "Knee_R_X_deg"),
    ("Ankle_R", "Ankle_R_X_deg"),
    ("Trunk", "Trunk_X_deg"),
    ("Neck", "Neck_X_deg"),
]

FAMILY_BALANCE = "Balance/Stability"
FAMILY_JOINT = "Joint Angles"
FAMILY_FORCE = "Force/Torque"


def build_dv_specs() -> list[dict]:
    """Define all dependent variables with their aggregation rules."""
    specs = []

    # COM
    for ax in _COM_AXES:
        specs.append({"dv": f"COM_{ax}_range", "col": f"COM_{ax}", "agg": "range", "family": FAMILY_BALANCE})
        specs.append({"dv": f"COM_{ax}_path_length", "col": f"COM_{ax}", "agg": "path_length", "family": FAMILY_BALANCE})
        specs.append({"dv": f"vCOM_{ax}_peak", "col": f"vCOM_{ax}", "agg": "abs_peak", "family": FAMILY_BALANCE})

    # COP (onset-zeroed)
    for ax in _COP_AXES:
        col = f"COP_{ax}_m_onset0"
        specs.append({"dv": f"COP_{ax}_range", "col": col, "agg": "range", "family": FAMILY_BALANCE})
        specs.append({"dv": f"COP_{ax}_path_length", "col": col, "agg": "path_length", "family": FAMILY_BALANCE})
        specs.append({"dv": f"COP_{ax}_peak_velocity", "col": col, "agg": "abs_peak_velocity", "family": FAMILY_BALANCE})

    # MoS
    specs.append({"dv": "MOS_minDist_signed_min", "col": "MOS_minDist_signed", "agg": "min_val", "family": FAMILY_BALANCE})
    specs.append({"dv": "MOS_AP_v3d_min", "col": "MOS_AP_v3d", "agg": "min_val", "family": FAMILY_BALANCE})
    specs.append({"dv": "MOS_ML_v3d_min", "col": "MOS_ML_v3d", "agg": "min_val", "family": FAMILY_BALANCE})

    # Joint angles (sagittal plane, right side)
    for name, col in _JOINT_NAMES:
        specs.append({"dv": f"{name}_ROM", "col": col, "agg": "range", "family": FAMILY_JOINT})
        specs.append({"dv": f"{name}_peak", "col": col, "agg": "abs_peak", "family": FAMILY_JOINT})

    # GRF
    for ax in _GRF_AXES:
        col = f"GRF_{ax}_N"
        specs.append({"dv": f"GRF_{ax}_peak", "col": col, "agg": "abs_peak", "family": FAMILY_FORCE})
        specs.append({"dv": f"GRF_{ax}_range", "col": col, "agg": "range", "family": FAMILY_FORCE})

    # Ankle torque
    specs.append({"dv": "AnkleTorqueMid_Y_peak", "col": "AnkleTorqueMid_int_Y_Nm_per_kg", "agg": "abs_peak", "family": FAMILY_FORCE})

    return specs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true", help="Only load data; skip analysis")
    return ap.parse_args()


def load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(str(path), encoding="utf8-lossy", infer_schema_length=10000)


def load_platform_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(str(path), sheet_name="platform")
    df["subject"] = df["subject"].astype(str).str.strip()
    df["velocity"] = pd.to_numeric(df["velocity"], errors="coerce")
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce").astype("Int64")
    df["step_TF"] = df["step_TF"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Milestone 1: Data loading & trial-level aggregation
# ---------------------------------------------------------------------------

def aggregate_trial_features(df: pl.DataFrame, specs: list[dict]) -> pl.DataFrame:
    """Compute per-trial summary statistics for all DVs via polars group_by.

    Time window: [platform_onset_local, end_frame] per trial (MocapFrame units).
    ``end_frame`` must already be joined to *df* before calling this function.
    """

    # Filter time window: [platform_onset, end_frame] per trial
    df_win = df.filter(
        pl.col("end_frame").is_not_null()
    ).filter(
        (pl.col("MocapFrame") >= pl.col("platform_onset_local"))
        & (pl.col("MocapFrame") <= pl.col("end_frame"))
    )

    # Build aggregation expressions
    agg_exprs = []
    for spec in specs:
        col = spec["col"]
        dv = spec["dv"]
        agg = spec["agg"]

        if agg == "range":
            agg_exprs.append((pl.col(col).max() - pl.col(col).min()).alias(dv))
        elif agg == "path_length":
            agg_exprs.append(pl.col(col).diff().abs().sum().alias(dv))
        elif agg == "abs_peak":
            agg_exprs.append(pl.col(col).abs().max().alias(dv))
        elif agg == "min_val":
            agg_exprs.append(pl.col(col).min().alias(dv))
        elif agg == "abs_peak_velocity":
            # peak velocity: max of |diff/dt|, dt=0.01s (100Hz)
            agg_exprs.append((pl.col(col).diff().abs() / 0.01).max().alias(dv))

    result = df_win.group_by(TRIAL_KEYS).agg(agg_exprs).sort(TRIAL_KEYS)
    return result


def _compute_end_frames(df: pl.DataFrame, platform: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trial end_frame for the analysis window.

    - step trials: end_frame = step_onset_local (actual step onset)
    - nonstep trials: end_frame = mean(step_onset_local) of step trials
      in the same (subject, velocity) group

    Returns a pandas DataFrame with [subject, velocity, trial, end_frame].
    """
    # Extract per-trial step_onset_local
    trials = (
        df.select(TRIAL_KEYS + ["step_onset_local"])
        .group_by(TRIAL_KEYS)
        .agg(pl.col("step_onset_local").drop_nulls().first().alias("step_onset_local"))
        .sort(TRIAL_KEYS)
        .to_pandas()
    )
    trials["subject"] = trials["subject"].astype(str).str.strip()
    trials["trial"] = pd.to_numeric(trials["trial"], errors="coerce").astype("Int64")
    trials["velocity"] = pd.to_numeric(trials["velocity"], errors="coerce")

    # Join step_TF
    plat_sub = platform[["subject", "velocity", "trial", "step_TF"]].copy()
    trials = trials.merge(plat_sub, on=TRIAL_KEYS, how="left")

    # end_frame: step → step_onset_local, nonstep → mean
    trials["end_frame"] = np.nan
    step_mask = (trials["step_TF"] == "step") & trials["step_onset_local"].notna()
    trials.loc[step_mask, "end_frame"] = trials.loc[step_mask, "step_onset_local"]

    # Mean step_onset per (subject, velocity)
    step_means = (
        trials.loc[step_mask]
        .groupby(["subject", "velocity"])["step_onset_local"]
        .mean()
        .reset_index()
        .rename(columns={"step_onset_local": "mean_step_onset"})
    )

    needs_end = trials["end_frame"].isna()
    if needs_end.any():
        trials = trials.merge(step_means, on=["subject", "velocity"], how="left")
        fill_mask = needs_end & trials["mean_step_onset"].notna()
        trials.loc[fill_mask, "end_frame"] = trials.loc[fill_mask, "mean_step_onset"]
        trials.drop(columns=["mean_step_onset"], inplace=True)

    trials["end_frame"] = trials["end_frame"].round().astype("Int64")

    n_missing = trials["end_frame"].isna().sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} trials with no computable end_frame (dropped)")
    trials = trials.dropna(subset=["end_frame"]).reset_index(drop=True)

    return trials[TRIAL_KEYS + ["step_TF", "end_frame"]]


def load_and_prepare(csv_path: Path, xlsm_path: Path, specs: list[dict]) -> pd.DataFrame:
    """Load, filter, aggregate, join step_TF. Returns 1-row-per-trial DataFrame."""

    print("  Loading CSV...")
    df = load_csv(csv_path)
    n_frames = len(df)

    print("  Loading platform sheet...")
    platform = load_platform_sheet(xlsm_path)

    # Compute per-trial end_frame (step_onset or subject-velocity mean)
    print("  Computing per-trial end_frame [platform_onset → step_onset]...")
    end_frames = _compute_end_frames(df, platform)

    # Join end_frame to main timeseries for per-trial windowing
    end_pl = pl.from_pandas(end_frames[TRIAL_KEYS + ["end_frame"]])
    df = df.join(end_pl, on=TRIAL_KEYS, how="left")

    print("  Aggregating to trial-level features [platform_onset, step_onset] window...")
    trial_pl = aggregate_trial_features(df, specs)
    trial_df = trial_pl.to_pandas()

    # Normalize types for join
    trial_df["subject"] = trial_df["subject"].astype(str).str.strip()
    trial_df["velocity"] = pd.to_numeric(trial_df["velocity"], errors="coerce")
    trial_df["trial"] = pd.to_numeric(trial_df["trial"], errors="coerce").astype("Int64")

    # Join step_TF from end_frames (already computed)
    trial_df = trial_df.merge(
        end_frames[TRIAL_KEYS + ["step_TF"]], on=TRIAL_KEYS, how="left"
    )

    # Keep only step/nonstep
    trial_df = trial_df[trial_df["step_TF"].isin(["step", "nonstep"])].reset_index(drop=True)

    n_trials = len(trial_df)
    n_step = (trial_df["step_TF"] == "step").sum()
    n_nonstep = (trial_df["step_TF"] == "nonstep").sum()

    print(f"  Frames: {n_frames} → filtered & aggregated")
    print(f"  Trials: {n_trials} (step={n_step}, nonstep={n_nonstep})")

    return trial_df


# ---------------------------------------------------------------------------
# Milestone 2: LMM fitting via Rscript subprocess
# ---------------------------------------------------------------------------

def _build_r_lmm_script(csv_path: str, output_path: str) -> str:
    """Generate R script that fits LMM for all DVs and writes results CSV."""
    return f"""
library(lmerTest)

data <- read.csv("{csv_path}", stringsAsFactors = FALSE)
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
  formula_str <- paste0("`", dv, "` ~ step_TF + (1|subject)")

  # Subset to non-NA rows for this DV
  sub <- data[!is.na(data[[dv]]), ]
  n_s <- sum(sub$step_TF == "step")
  n_ns <- sum(sub$step_TF == "nonstep")
  m_s <- mean(sub[[dv]][sub$step_TF == "step"], na.rm = TRUE)
  sd_s <- sd(sub[[dv]][sub$step_TF == "step"], na.rm = TRUE)
  m_ns <- mean(sub[[dv]][sub$step_TF == "nonstep"], na.rm = TRUE)
  sd_ns <- sd(sub[[dv]][sub$step_TF == "nonstep"], na.rm = TRUE)

  tryCatch({{
    m <- lmer(as.formula(formula_str), data = sub, REML = TRUE)
    s <- summary(m)
    co <- coef(s)

    # Find the step_TF row (step_TFstep)
    row_name <- grep("step_TF", rownames(co), value = TRUE)[1]
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


def fit_lmm_all(trial_df: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    """Fit LMM for each DV via Rscript, return results with FDR correction."""

    # Prepare data for R: subject, velocity, trial, step_TF + all DVs
    dv_names = [s["dv"] for s in specs]
    cols_to_export = TRIAL_KEYS + ["step_TF"] + dv_names
    export_df = trial_df[cols_to_export].copy()

    # Write to temp CSV
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir=tempfile.gettempdir(), newline=""
    ) as f:
        export_df.to_csv(f, index=False)
        data_csv = f.name

    result_csv = tempfile.mktemp(suffix="_lmm_results.csv", dir=tempfile.gettempdir())

    # Build and write R script
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
        print("  Running Rscript for LMM fitting...")
        proc = subprocess.run(
            [RSCRIPT, r_script],
            capture_output=True, text=True, timeout=600, env=_r_env(),
        )
        if proc.returncode != 0:
            print(f"  R stderr: {proc.stderr[:1000]}")
            raise RuntimeError(f"Rscript failed with return code {proc.returncode}")
        if proc.stdout:
            print(f"  R: {proc.stdout.strip()}")

        # Read results
        results = pd.read_csv(result_csv)
    finally:
        for p in [data_csv, result_csv, r_script]:
            try:
                os.unlink(p)
            except OSError:
                pass

    # Add family info
    dv_to_family = {s["dv"]: s["family"] for s in specs}
    results["family"] = results["dv"].map(dv_to_family)

    # Apply BH-FDR per family
    results["p_fdr"] = np.nan
    for fam in results["family"].unique():
        mask = (results["family"] == fam) & results["p_value"].notna()
        if mask.sum() == 0:
            continue
        pvals = results.loc[mask, "p_value"].values
        _, fdr_pvals, _, _ = multipletests(pvals, method="fdr_bh")
        results.loc[mask, "p_fdr"] = fdr_pvals

    # Significance markers
    def _sig(p):
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


def print_results_table(results: pd.DataFrame) -> None:
    """Print formatted LMM results to stdout."""
    print("\n" + "=" * 110)
    print("LMM Results: DV ~ step_TF + (1|subject), REML, Satterthwaite df")
    print("=" * 110)
    fmt = "{:<30s} {:<20s} {:>9s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s} {:>4s}"
    print(fmt.format("DV", "Family", "Estimate", "SE", "df", "t", "p", "p_FDR", "Sig"))
    print("-" * 110)

    for fam in [FAMILY_BALANCE, FAMILY_JOINT, FAMILY_FORCE]:
        sub = results[results["family"] == fam]
        for _, row in sub.iterrows():
            est = f"{row['estimate']:.4f}" if pd.notna(row["estimate"]) else "FAIL"
            se = f"{row['SE']:.4f}" if pd.notna(row["SE"]) else ""
            df = f"{row['df']:.1f}" if pd.notna(row["df"]) else ""
            t = f"{row['t_value']:.3f}" if pd.notna(row["t_value"]) else ""
            p = f"{row['p_value']:.4f}" if pd.notna(row["p_value"]) else ""
            pfdr = f"{row['p_fdr']:.4f}" if pd.notna(row["p_fdr"]) else ""
            sig = row["sig"]
            fam_short = fam[:18]
            print(fmt.format(row["dv"][:30], fam_short, est, se, df, t, p, pfdr, sig))
        print("-" * 110)

    n_sig = (results["sig"] != "").sum()
    n_total = len(results)
    print(f"\nFDR significant: {n_sig}/{n_total} variables")

    # Descriptive stats for significant variables
    sig_rows = results[results["sig"] != ""].sort_values("p_fdr")
    if not sig_rows.empty:
        print("\n--- Significant Variables (FDR < 0.05) ---")
        for _, row in sig_rows.iterrows():
            print(
                f"  {row['dv']}: step={row['mean_step']:.4f}±{row['sd_step']:.4f}, "
                f"nonstep={row['mean_nonstep']:.4f}±{row['sd_nonstep']:.4f}, "
                f"p_FDR={row['p_fdr']:.4f} {row['sig']}"
            )


# ---------------------------------------------------------------------------
# Milestone 3: Figures
# ---------------------------------------------------------------------------

COLORS = {"step": "#E74C3C", "nonstep": "#3498DB"}
FAMILY_COLORS = {
    FAMILY_BALANCE: "#2ECC71",
    FAMILY_JOINT: "#9B59B6",
    FAMILY_FORCE: "#E67E22",
}


def fig1_forest(results: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """Forest plot of LMM estimates ± 95% CI."""
    df = results[results["converged"] == True].copy()  # noqa: E712
    df = df.sort_values(["family", "dv"]).reset_index(drop=True)

    # CI = estimate ± 1.96 * SE
    df["ci_lo"] = df["estimate"] - 1.96 * df["SE"]
    df["ci_hi"] = df["estimate"] + 1.96 * df["SE"]

    n = len(df)
    fig_height = max(6, n * 0.35 + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(n)
    for i, (_, row) in enumerate(df.iterrows()):
        color = FAMILY_COLORS.get(row["family"], "gray")
        marker = "o" if row["sig"] else "d"
        alpha = 1.0 if row["sig"] else 0.5
        lw = 2.0 if row["sig"] else 1.0

        ax.errorbar(
            row["estimate"], i,
            xerr=[[row["estimate"] - row["ci_lo"]], [row["ci_hi"] - row["estimate"]]],
            fmt=marker, color=color, alpha=alpha, markersize=6, capsize=3, linewidth=lw,
        )
        # Bold label if significant
        weight = "bold" if row["sig"] else "normal"
        ax.text(-0.01, i, row["dv"], ha="right", va="center", fontsize=7,
                fontweight=weight, transform=ax.get_yaxis_transform())

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * n)  # labels drawn manually
    ax.set_xlabel("LMM Estimate (step − nonstep)", fontsize=10)
    ax.set_title("Forest Plot: LMM Fixed Effect Estimates ± 95% CI", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Legend for families
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    legend_elements.append(Patch(facecolor="white", edgecolor="black", label="● = FDR sig, ◆ = n.s."))
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.subplots_adjust(left=0.30)
    fig.savefig(out_dir / "fig1_lmm_forest_plot.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2_violin(trial_df: pd.DataFrame, results: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """Violin + strip plots for FDR-significant variables."""
    import seaborn as sns

    sig_dvs = results[results["sig"] != ""].sort_values("p_fdr")["dv"].tolist()
    if not sig_dvs:
        print("  No FDR-significant variables for fig2. Generating placeholder.")
        # Show top 6 by p-value instead
        sig_dvs = results.dropna(subset=["p_value"]).nsmallest(6, "p_value")["dv"].tolist()
        if not sig_dvs:
            return

    n_vars = min(len(sig_dvs), 9)
    sig_dvs = sig_dvs[:n_vars]

    ncols = min(3, n_vars)
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for idx, dv in enumerate(sig_dvs):
        row_idx, col_idx = divmod(idx, ncols)
        ax = axes[row_idx][col_idx]

        valid = trial_df[trial_df["step_TF"].isin(["step", "nonstep"])].dropna(subset=[dv])
        palette = [COLORS["step"], COLORS["nonstep"]]

        sns.violinplot(
            data=valid, x="step_TF", y=dv, hue="step_TF",
            order=["step", "nonstep"], hue_order=["step", "nonstep"],
            palette=palette, inner=None, alpha=0.3, legend=False, ax=ax,
        )
        sns.stripplot(
            data=valid, x="step_TF", y=dv, hue="step_TF",
            order=["step", "nonstep"], hue_order=["step", "nonstep"],
            palette=palette, size=3, alpha=0.6, jitter=0.2, legend=False, ax=ax,
        )

        r = results[results["dv"] == dv].iloc[0]
        p_text = f"p_FDR={r['p_fdr']:.4f}" if pd.notna(r["p_fdr"]) else ""
        ax.set_title(f"{dv}\n{p_text} {r['sig']}", fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(dv, fontsize=8)

    # Hide unused axes
    for idx in range(n_vars, nrows * ncols):
        row_idx, col_idx = divmod(idx, ncols)
        axes[row_idx][col_idx].set_visible(False)

    fig.suptitle("Step vs. Non-step: FDR-Significant Variables", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_violin_significant.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_heatmap(results: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """Heatmap of z-scored group means with significance markers."""
    df = results[results["converged"] == True].copy()  # noqa: E712
    df = df.sort_values(["family", "dv"]).reset_index(drop=True)

    # Z-score across groups: (mean - grand_mean) / grand_sd
    grand_mean = (df["mean_step"] + df["mean_nonstep"]) / 2
    grand_sd = np.sqrt((df["sd_step"] ** 2 + df["sd_nonstep"] ** 2) / 2)
    grand_sd = grand_sd.replace(0, np.nan)

    z_step = (df["mean_step"] - grand_mean) / grand_sd
    z_nonstep = (df["mean_nonstep"] - grand_mean) / grand_sd

    heat_data = pd.DataFrame({"step": z_step.values, "nonstep": z_nonstep.values}, index=df["dv"].values)

    n = len(heat_data)
    fig_height = max(6, n * 0.3 + 2)
    fig, ax = plt.subplots(figsize=(5, fig_height))

    import seaborn as sns
    sns.heatmap(
        heat_data, annot=False, cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax, cbar_kws={"label": "z-score"},
    )

    # Add significance markers
    for i, (_, row) in enumerate(df.iterrows()):
        if row["sig"]:
            ax.text(0.5, i + 0.5, row["sig"], ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black")
            ax.text(1.5, i + 0.5, row["sig"], ha="center", va="center",
                    fontsize=8, fontweight="bold", color="black")

    # Add family color bands on left
    for i, (_, row) in enumerate(df.iterrows()):
        color = FAMILY_COLORS.get(row["family"], "gray")
        ax.add_patch(plt.Rectangle((-0.3, i), 0.25, 1, color=color, clip_on=False, transform=ax.transData))

    ax.set_ylabel("")
    ax.set_title("Group Mean Comparison (z-scored)\n* p_FDR<.05, ** <.01, *** <.001",
                 fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_descriptive_heatmap.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    specs = build_dv_specs()
    dv_names = [s["dv"] for s in specs]

    print("=" * 60)
    print("Step vs. Non-step Biomechanical LMM Analysis")
    print("=" * 60)

    # --- Milestone 1 ---
    print("\n[M1] Loading and aggregating data...")
    trial_df = load_and_prepare(args.csv, args.platform_xlsm, specs)
    print(f"  DVs: {len(dv_names)} variables")

    if args.dry_run:
        print(f"\nDry run complete. {len(trial_df)} trials, {len(dv_names)} DVs.")
        return

    # --- Milestone 2 ---
    print("\n[M2] Fitting LMMs...")
    results = fit_lmm_all(trial_df, specs)
    print_results_table(results)

    # --- Milestone 3 ---
    print("\n[M3] Generating figures...")
    fig1_forest(results, out_dir, dpi)
    print("  fig1_lmm_forest_plot.png")

    fig2_violin(trial_df, results, out_dir, dpi)
    print("  fig2_violin_significant.png")

    fig3_heatmap(results, out_dir, dpi)
    print("  fig3_descriptive_heatmap.png")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Output directory: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
