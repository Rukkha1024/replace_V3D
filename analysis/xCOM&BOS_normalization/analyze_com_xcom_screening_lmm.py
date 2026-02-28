"""COM/xCOM/xCOM_BOS 1차 스크리닝 LMM.

Window: [platform_onset, step_onset_eval] per trial.
Model: DV ~ step_TF + (1|subject), REML.
Purpose: step/nonstep 유의 변수 탐색(전체 DV BH-FDR).
Usage: conda run -n module python analysis/xCOM&BOS_normalization/analyze_com_xcom_screening_lmm.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_OUT_CSV = SCRIPT_DIR / "com_xcom_screening_lmm_results.csv"

TRIAL_KEYS = ["subject", "velocity", "trial"]
AXES = ("X", "Y", "Z")
G = 9.81
DT = 0.01

WINDOWS_R_HOME = Path(r"C:\Users\Alice\miniconda3\envs\module\lib\R")
WINDOWS_RSCRIPT = WINDOWS_R_HOME / "bin" / "x64" / "Rscript.exe"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


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


def _normalize_trial_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject"] = out["subject"].astype(str).str.strip()
    out["velocity"] = pd.to_numeric(out["velocity"], errors="coerce")
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce").astype("Int64")
    return out


def load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(str(path), encoding="utf8-lossy", infer_schema_length=10000)


def load_platform_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(str(path), sheet_name="platform")
    required = {"subject", "velocity", "trial", "step_TF", "platform_onset", "step_onset"}
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
            records.append({"subject": str(subject).strip(), "metric": metric, "value": value})

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
    wide["height_m"] = wide["height_cm"] / 100.0
    wide["leg_len_m"] = wide["leg_len_cm"] / 100.0
    wide["foot_len_m"] = ((wide["foot_len_left_mm"] + wide["foot_len_right_mm"]) / 2.0) / 1000.0
    return wide[["subject", "height_m", "leg_len_m", "foot_len_m"]].copy()


def _compute_end_frames(df: pl.DataFrame, platform: pd.DataFrame) -> pd.DataFrame:
    frame_bounds = (
        df.select(TRIAL_KEYS + ["MocapFrame", "platform_onset_local", "step_onset_local"])
        .group_by(TRIAL_KEYS)
        .agg(
            pl.col("MocapFrame").min().alias("frame_min"),
            pl.col("MocapFrame").max().alias("frame_max"),
            pl.col("platform_onset_local").drop_nulls().first().alias("platform_onset_local"),
            pl.col("step_onset_local").drop_nulls().first().alias("step_onset_local"),
        )
        .to_pandas()
    )
    frame_bounds = _normalize_trial_types(frame_bounds)
    frame_bounds["frame_min"] = pd.to_numeric(frame_bounds["frame_min"], errors="coerce").astype("Int64")
    frame_bounds["frame_max"] = pd.to_numeric(frame_bounds["frame_max"], errors="coerce").astype("Int64")
    frame_bounds["platform_onset_local"] = pd.to_numeric(frame_bounds["platform_onset_local"], errors="coerce")
    frame_bounds["step_onset_local"] = pd.to_numeric(frame_bounds["step_onset_local"], errors="coerce")

    plat = platform[TRIAL_KEYS + ["step_TF", "platform_onset", "step_onset"]].drop_duplicates().copy()
    trials = frame_bounds.merge(plat, on=TRIAL_KEYS, how="left")
    trials = trials[trials["step_TF"].isin(["step", "nonstep"])].reset_index(drop=True)

    trials["end_frame"] = np.nan
    step_mask = (trials["step_TF"] == "step") & trials["step_onset_local"].notna()
    trials.loc[step_mask, "end_frame"] = trials.loc[step_mask, "step_onset_local"]

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

    trials = trials.merge(step_means, on=["subject", "velocity"], how="left")
    fill_mask_sv = trials["end_frame"].isna() & trials["mean_step_onset_sv"].notna()
    trials.loc[fill_mask_sv, "end_frame"] = trials.loc[fill_mask_sv, "mean_step_onset_sv"]
    filled_sv = int(fill_mask_sv.sum())

    trials = trials.merge(prefilter_step_means, on=["subject", "velocity"], how="left")
    fill_mask_pref = (
        trials["end_frame"].isna()
        & (trials["step_TF"] == "nonstep")
        & trials["step_onset_local_prefilter"].notna()
    )
    trials.loc[fill_mask_pref, "end_frame"] = trials.loc[fill_mask_pref, "step_onset_local_prefilter"]
    filled_pref = int(fill_mask_pref.sum())

    trials["end_frame"] = pd.to_numeric(trials["end_frame"], errors="coerce").round().astype("Int64")
    missing_after_fill = int(trials["end_frame"].isna().sum())
    if missing_after_fill > 0:
        print(f"  Warning: {missing_after_fill} trials missing end_frame (dropped)")
        trials = trials.dropna(subset=["end_frame"]).reset_index(drop=True)

    oob = (
        (trials["end_frame"] < trials["frame_min"])
        | (trials["end_frame"] > trials["frame_max"])
    ).sum()
    if int(oob) > 0:
        raise ValueError(f"end_frame out of range count={int(oob)}")

    print(f"  end_frame fill (subject-velocity mean): {filled_sv}")
    print(f"  end_frame fill (prefilter platform subject-velocity mean): {filled_pref}")
    print(f"  end_frame range validation passed: oob={int(oob)}")

    return trials[
        TRIAL_KEYS
        + [
            "step_TF",
            "frame_min",
            "frame_max",
            "platform_onset_local",
            "step_onset_local",
            "end_frame",
        ]
    ].copy()


def build_dv_specs() -> list[dict]:
    specs: list[dict] = []
    for ax in AXES:
        specs.append(
            {
                "dv": f"COM_{ax}_max_min",
                "col": f"COM_{ax}",
                "agg": "range",
                "family": "COM",
                "axis": ax,
                "window_type": "window",
            }
        )
        specs.append(
            {
                "dv": f"COM_{ax}_mean_velocity",
                "col": f"COM_{ax}",
                "agg": "mean_velocity",
                "family": "COM",
                "axis": ax,
                "window_type": "window",
            }
        )
        specs.append(
            {
                "dv": f"COM_{ax}_peak_velocity",
                "col": f"vCOM_{ax}",
                "agg": "abs_peak",
                "family": "COM",
                "axis": ax,
                "window_type": "window",
            }
        )

    for ax in AXES:
        specs.append(
            {
                "dv": f"xCOM_{ax}_max_min",
                "col": f"xCOM_{ax}_resolved",
                "agg": "range",
                "family": "xCOM",
                "axis": ax,
                "window_type": "window",
            }
        )
        specs.append(
            {
                "dv": f"xCOM_{ax}_mean_velocity",
                "col": f"xCOM_{ax}_resolved",
                "agg": "mean_velocity",
                "family": "xCOM",
                "axis": ax,
                "window_type": "window",
            }
        )
        specs.append(
            {
                "dv": f"xCOM_{ax}_peak_velocity",
                "col": f"xCOM_{ax}_resolved",
                "agg": "abs_peak_velocity",
                "family": "xCOM",
                "axis": ax,
                "window_type": "window",
            }
        )

    for axis_name, col_name in [("AP", "xCOM_BOS_AP_foot"), ("ML", "xCOM_BOS_ML_foot")]:
        specs.append(
            {
                "dv": f"xCOM_BOS_{axis_name}_foot_platformonset",
                "col": col_name,
                "agg": "value_at_event",
                "event_col": "platform_eval_frame",
                "family": "xCOM_BOS",
                "axis": axis_name,
                "window_type": "platform_onset",
            }
        )
        specs.append(
            {
                "dv": f"xCOM_BOS_{axis_name}_foot_steponset",
                "col": col_name,
                "agg": "value_at_event",
                "event_col": "step_eval_frame",
                "family": "xCOM_BOS",
                "axis": axis_name,
                "window_type": "step_onset",
            }
        )
        specs.append(
            {
                "dv": f"xCOM_BOS_{axis_name}_foot_mean_window",
                "col": col_name,
                "agg": "mean_in_window",
                "family": "xCOM_BOS",
                "axis": axis_name,
                "window_type": "window_mean",
            }
        )
    return specs


def _mean_velocity_expr(col: str, in_window: pl.Expr) -> pl.Expr:
    num = pl.col(col).filter(in_window).diff().abs().sum()
    den = (pl.col(col).filter(in_window).count() - 1).cast(pl.Float64) * DT
    return pl.when(den > 0).then(num / den).otherwise(None)


def aggregate_trial_features(df: pl.DataFrame, specs: list[dict]) -> pl.DataFrame:
    in_window = (
        (pl.col("MocapFrame") >= pl.col("platform_onset_local"))
        & (pl.col("MocapFrame") <= pl.col("end_frame"))
    )

    agg_exprs: list[pl.Expr] = []
    for spec in specs:
        col = spec["col"]
        dv = spec["dv"]
        agg = spec["agg"]
        if agg == "range":
            agg_exprs.append(
                (pl.col(col).filter(in_window).max() - pl.col(col).filter(in_window).min()).alias(dv)
            )
        elif agg == "mean_velocity":
            agg_exprs.append(_mean_velocity_expr(col, in_window).alias(dv))
        elif agg == "abs_peak":
            agg_exprs.append(pl.col(col).filter(in_window).abs().max().alias(dv))
        elif agg == "abs_peak_velocity":
            agg_exprs.append((pl.col(col).filter(in_window).diff().abs() / DT).max().alias(dv))
        elif agg == "value_at_event":
            event_col = spec["event_col"]
            agg_exprs.append(
                pl.col(col)
                .filter(pl.col("MocapFrame") == pl.col(event_col))
                .drop_nulls()
                .first()
                .alias(dv)
            )
        elif agg == "mean_in_window":
            agg_exprs.append(pl.col(col).filter(in_window).mean().alias(dv))
        else:
            raise ValueError(f"Unsupported agg: {agg}")

    return df.group_by(TRIAL_KEYS).agg(agg_exprs).sort(TRIAL_KEYS)


def load_and_prepare(csv_path: Path, xlsm_path: Path, dv_specs: list[dict]) -> tuple[pd.DataFrame, dict]:
    print("  Loading CSV...")
    df = load_csv(csv_path)
    n_frames = df.height

    required_cols = {
        "subject",
        "velocity",
        "trial",
        "MocapFrame",
        "platform_onset_local",
        "step_onset_local",
        "COM_X",
        "COM_Y",
        "COM_Z",
        "vCOM_X",
        "vCOM_Y",
        "vCOM_Z",
        "BOS_minX",
        "BOS_minY",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    print("  Loading platform/meta...")
    platform = load_platform_sheet(xlsm_path)
    anthro = load_anthropometrics_from_meta(xlsm_path)

    print("  Computing end_frame [platform_onset -> step_onset_eval]...")
    end_frames = _compute_end_frames(df, platform)
    end_pl = pl.from_pandas(end_frames[TRIAL_KEYS + ["step_TF", "platform_onset_local", "end_frame"]])
    anthro_pl = pl.from_pandas(anthro)

    print("  Joining trial metadata and anthropometrics...")
    frame_df = (
        df.with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial").cast(pl.Int64, strict=False),
        )
        .join(end_pl, on=TRIAL_KEYS, how="inner")
        .join(anthro_pl, on="subject", how="left")
    )

    bad = frame_df.select(
        (
            (pl.col("leg_len_m").is_null() | (pl.col("leg_len_m") <= 0))
            | (pl.col("foot_len_m").is_null() | (pl.col("foot_len_m") <= 0))
        ).sum().alias("n_bad")
    ).item()
    if int(bad) > 0:
        raise ValueError(f"Invalid anthropometrics in joined frames count={int(bad)}")

    print("  Building xCOM/xCOM_BOS framewise variables...")
    with_kin = frame_df.with_columns(
        (pl.lit(G) / pl.col("leg_len_m")).sqrt().alias("omega0"),
        pl.col("platform_onset_local").cast(pl.Int64, strict=False).alias("platform_eval_frame"),
        pl.col("end_frame").cast(pl.Int64, strict=False).alias("step_eval_frame"),
    )

    xcom_exprs: list[pl.Expr] = []
    for ax in AXES:
        computed = pl.col(f"COM_{ax}") + pl.col(f"vCOM_{ax}") / pl.col("omega0")
        raw_col = f"xCOM_{ax}"
        if raw_col in with_kin.columns:
            xcom_exprs.append(pl.coalesce([pl.col(raw_col).cast(pl.Float64, strict=False), computed]).alias(f"xCOM_{ax}_resolved"))
        else:
            xcom_exprs.append(computed.alias(f"xCOM_{ax}_resolved"))

    with_kin = with_kin.with_columns(xcom_exprs).with_columns(
        pl.when(pl.col("foot_len_m") > 0)
        .then((pl.col("xCOM_X_resolved") - pl.col("BOS_minX")) / pl.col("foot_len_m"))
        .otherwise(None)
        .alias("xCOM_BOS_AP_foot"),
        pl.when(pl.col("foot_len_m") > 0)
        .then((pl.col("xCOM_Y_resolved") - pl.col("BOS_minY")) / pl.col("foot_len_m"))
        .otherwise(None)
        .alias("xCOM_BOS_ML_foot"),
    )

    print("  Aggregating trial-level DVs...")
    trial_pl = aggregate_trial_features(with_kin, dv_specs)
    trial_df = trial_pl.to_pandas()
    trial_df = _normalize_trial_types(trial_df)
    trial_df = trial_df.merge(
        end_frames[TRIAL_KEYS + ["step_TF"]],
        on=TRIAL_KEYS,
        how="left",
    )
    trial_df = trial_df[trial_df["step_TF"].isin(["step", "nonstep"])].reset_index(drop=True)

    n_step = int((trial_df["step_TF"] == "step").sum())
    n_nonstep = int((trial_df["step_TF"] == "nonstep").sum())
    print(f"  Frames: {n_frames}")
    print(f"  Trials: {len(trial_df)} (step={n_step}, nonstep={n_nonstep})")
    for spec in dv_specs:
        miss = int(trial_df[spec["dv"]].isna().sum())
        print(f"  {spec['dv']} missing: {miss}")

    summary = {
        "n_frames": int(n_frames),
        "n_trials": int(len(trial_df)),
        "n_step": n_step,
        "n_nonstep": n_nonstep,
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
    export_df = trial_df[["subject", "velocity", "trial", "step_TF"] + dv_names].copy()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir=tempfile.gettempdir(), newline=""
    ) as f:
        export_df.to_csv(f, index=False)
        data_csv = f.name
    result_csv = tempfile.mktemp(suffix="_com_xcom_screen_lmm.csv", dir=tempfile.gettempdir())

    r_code = _build_r_lmm_script(data_csv.replace("\\", "/"), result_csv.replace("\\", "/"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(r_code)
        r_script = f.name

    try:
        rscript_cmd, r_env = resolve_r_runtime()
        print(f"  Running Rscript: {rscript_cmd}")
        proc = subprocess.run(
            [rscript_cmd, r_script],
            capture_output=True,
            text=True,
            timeout=600,
            env=r_env,
        )
        if proc.returncode != 0:
            print(proc.stderr[:1200])
            raise RuntimeError(f"Rscript failed with code {proc.returncode}")
        if proc.stdout.strip():
            print(f"  R: {proc.stdout.strip()}")
        results = pd.read_csv(result_csv)
    finally:
        for p in (data_csv, result_csv, r_script):
            try:
                os.unlink(p)
            except OSError:
                pass

    spec_meta = {
        s["dv"]: {"family": s["family"], "axis": s["axis"], "window_type": s["window_type"]}
        for s in dv_specs
    }
    results["family"] = results["dv"].map(lambda x: spec_meta[x]["family"])
    results["axis"] = results["dv"].map(lambda x: spec_meta[x]["axis"])
    results["window_type"] = results["dv"].map(lambda x: spec_meta[x]["window_type"])

    mask = (results["term"] == "main_step_effect") & results["p_value"].notna()
    results["p_fdr"] = np.nan
    if int(mask.sum()) > 0:
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
        return "n.s."

    results["sig"] = results["p_fdr"].apply(_sig)
    return results


def print_summary(results: pd.DataFrame) -> None:
    sub = results[results["term"] == "main_step_effect"].copy()
    n_sig = int(sub["sig"].isin(["*", "**", "***"]).sum())
    n_total = int(len(sub))
    print("=" * 88)
    print("1차 스크리닝 LMM: DV ~ step_TF + (1|subject), REML")
    print(f"FDR significant: {n_sig}/{n_total} (BH across all DVs)")
    print("=" * 88)

    view = sub.sort_values(["p_fdr", "p_value", "dv"]).reset_index(drop=True)
    print(view[["dv", "estimate", "SE", "t_value", "p_value", "p_fdr", "sig"]].to_string(index=False))


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("COM + xCOM + xCOM_BOS Screening LMM")
    print("=" * 72)

    dv_specs = build_dv_specs()
    print("\n[M1] Load and aggregate...")
    trial_df, summary = load_and_prepare(args.csv, args.platform_xlsm, dv_specs)
    print(f"  DVs: {len(dv_specs)}")

    if args.dry_run:
        print(
            "\nDry run complete. "
            f"Trials={summary['n_trials']} (step={summary['n_step']}, nonstep={summary['n_nonstep']}), "
            f"subjects={summary['n_subjects']}."
        )
        return

    print("\n[M2] Fit LMMs...")
    results = fit_lmm_all(trial_df, dv_specs)
    print_summary(results)

    print("\n[M3] Save results CSV...")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ordered_cols = [
        "dv",
        "family",
        "axis",
        "window_type",
        "estimate",
        "SE",
        "df",
        "t_value",
        "p_value",
        "p_fdr",
        "sig",
        "mean_step",
        "sd_step",
        "mean_nonstep",
        "sd_nonstep",
        "n_step",
        "n_nonstep",
        "converged",
    ]
    results = results[ordered_cols].sort_values(["p_fdr", "p_value", "dv"], na_position="last")
    results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  saved: {out_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()
