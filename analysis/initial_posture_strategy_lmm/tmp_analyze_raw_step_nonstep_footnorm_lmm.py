"""Temporary raw analysis: foot-length normalized xCOM/COM metrics (step vs nonstep).

Input:
- Raw CSV from `main.py --no-meta_prefilter`
- Platform labels from `data/perturb_inform.xlsm` (sheet=platform)
- Anthropometrics from `data/perturb_inform.xlsm` (sheet=meta)

Events:
- onset: MocapFrame == platform_onset_local
- 300ms: MocapFrame == platform_onset_local + 30 (100 Hz)

DVs:
- DV1_norm = (xCOM_X - BOS_minX) / foot_len_m
- DV1_abs_cm = (xCOM_X - BOS_minX) * 100
- DV2_norm = (COM_X - BOS_minX) / foot_len_m

Model:
- DV ~ step_TF + (1|subject), REML, lmerTest
- BH-FDR across all tested DVs
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from statsmodels.stats.multitest import multipletests

pl.Config.set_tbl_rows(999)
pl.Config.set_tbl_cols(999)
pl.Config.set_tbl_width_chars(120)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TRIAL_KEYS = ["subject", "velocity", "trial"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCRIPT_DIR / "tmp_raw_nometa_output" / "all_trials_timeseries_raw_nometa.csv",
    )
    parser.add_argument(
        "--platform_xlsm",
        type=Path,
        default=REPO_ROOT / "data" / "perturb_inform.xlsm",
    )
    parser.add_argument("--frames_300ms", type=int, default=30)
    parser.add_argument("--timeout_sec", type=int, default=900)
    parser.add_argument("--out_dir", type=Path, default=SCRIPT_DIR)
    return parser.parse_args()


def load_raw_timeseries(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"input csv not found: {path}")

    df = pl.read_csv(str(path), encoding="utf8-lossy", infer_schema_length=10000)
    if "trial" not in df.columns and "trial_num" in df.columns:
        df = df.rename({"trial_num": "trial"})

    required = {
        "subject",
        "velocity",
        "trial",
        "MocapFrame",
        "platform_onset_local",
        "analysis_end_local",
        "COM_X",
        "xCOM_X",
        "BOS_minX",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"input csv missing required columns: {missing}")

    return df.with_columns(
        pl.col("subject").cast(pl.Utf8).str.strip_chars(),
        pl.col("velocity").cast(pl.Float64, strict=False),
        pl.col("trial").cast(pl.Int64, strict=False),
        pl.col("MocapFrame").cast(pl.Int64, strict=False),
        pl.col("platform_onset_local").cast(pl.Int64, strict=False),
        pl.col("analysis_end_local").cast(pl.Int64, strict=False),
    )


def load_platform_labels(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"platform_xlsm not found: {path}")
    platform = pl.read_excel(str(path), sheet_name="platform")
    required = {"subject", "velocity", "trial", "step_TF"}
    missing = sorted(required - set(platform.columns))
    if missing:
        raise ValueError(f"platform sheet missing required columns: {missing}")
    return (
        platform.select(["subject", "velocity", "trial", "step_TF"])
        .with_columns(
            pl.col("subject").cast(pl.Utf8).str.strip_chars(),
            pl.col("velocity").cast(pl.Float64, strict=False),
            pl.col("trial").cast(pl.Int64, strict=False),
            pl.col("step_TF").cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase(),
        )
        .unique()
    )


def load_foot_length_from_meta(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(str(path), sheet_name="meta")
    label_col = raw.columns[0]
    raw[label_col] = raw[label_col].astype(str).str.strip()

    rows = raw[raw[label_col].isin(["발길이_왼", "발길이_오른"])].copy()
    missing = [r for r in ["발길이_왼", "발길이_오른"] if r not in rows[label_col].tolist()]
    if missing:
        raise ValueError(f"meta sheet missing required rows: {missing}")

    records: list[dict] = []
    for _, row in rows.iterrows():
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
    wide["발길이_왼"] = pd.to_numeric(wide["발길이_왼"], errors="coerce")
    wide["발길이_오른"] = pd.to_numeric(wide["발길이_오른"], errors="coerce")
    wide["foot_len_m"] = ((wide["발길이_왼"] + wide["발길이_오른"]) / 2.0) / 1000.0
    wide = wide[["subject", "foot_len_m"]].copy()

    return wide


def validate_minimum(df: pl.DataFrame) -> None:
    key_nulls = (
        df.select(
            pl.col("subject").is_null().sum().alias("subject"),
            pl.col("velocity").is_null().sum().alias("velocity"),
            pl.col("trial").is_null().sum().alias("trial"),
        )
        .row(0)
    )
    if any(int(x) > 0 for x in key_nulls):
        raise ValueError(f"null keys detected: {key_nulls}")

    monotonic_bad = (
        df.sort(TRIAL_KEYS + ["MocapFrame"])
        .group_by(TRIAL_KEYS)
        .agg(pl.col("MocapFrame").diff().drop_nulls().min().alias("min_diff"))
        .filter(pl.col("min_diff") < 1)
    )
    if monotonic_bad.height > 0:
        sample = monotonic_bad.head(5).to_dicts()
        raise ValueError(f"MocapFrame not monotonic. sample={sample}")

    onset_bad = (
        df.group_by(TRIAL_KEYS)
        .agg(
            pl.col("platform_onset_local").drop_nulls().first().alias("platform_onset_local"),
            pl.col("analysis_end_local").drop_nulls().first().alias("analysis_end_local"),
        )
        .filter((pl.col("platform_onset_local") < 1) | (pl.col("platform_onset_local") > pl.col("analysis_end_local")))
    )
    if onset_bad.height > 0:
        sample = onset_bad.head(5).to_dicts()
        raise ValueError(f"platform_onset_local out of range. sample={sample}")


def build_snapshot(df: pl.DataFrame, frame_expr: pl.Expr, suffix: str) -> pl.DataFrame:
    snap = (
        df.filter(pl.col("MocapFrame") == frame_expr)
        .select(
            TRIAL_KEYS
            + [
                "dv1_norm",
                "dv1_abs_cm",
                "dv2_norm",
            ]
        )
        .rename(
            {
                "dv1_norm": f"DV1_norm_{suffix}",
                "dv1_abs_cm": f"DV1_abs_cm_{suffix}",
                "dv2_norm": f"DV2_norm_{suffix}",
            }
        )
    )
    dup = snap.group_by(TRIAL_KEYS).len().filter(pl.col("len") != 1)
    if dup.height > 0:
        sample = dup.head(5).to_dicts()
        raise ValueError(f"non-unique rows at snapshot={suffix}. sample={sample}")
    return snap


def resolve_rscript() -> str:
    which = shutil.which("Rscript")
    if which:
        return which
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "Rscript"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Rscript not found.")


def build_r_code(csv_path: str, out_path: str, dv_cols: list[str]) -> str:
    dv_vector = ", ".join([f"'{dv}'" for dv in dv_cols])
    return f"""
library(lmerTest)

data <- read.csv("{csv_path}", stringsAsFactors = FALSE)
data$step_TF <- tolower(trimws(as.character(data$step_TF)))
data$step_TF <- factor(data$step_TF, levels = c("nonstep", "step"))
dv_cols <- c({dv_vector})

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
      dv = dv, estimate = NA, SE = NA, df = NA, t_value = NA, p_value = NA,
      mean_step = m_s, sd_step = sd_s, mean_nonstep = m_ns, sd_nonstep = sd_ns,
      n_step = n_s, n_nonstep = n_ns, converged = FALSE, stringsAsFactors = FALSE
    ))
    next
  }}

  formula_str <- paste0("`", dv, "` ~ step_TF + (1|subject)")
  tryCatch({{
    model <- lmer(as.formula(formula_str), data = sub, REML = TRUE)
    co <- coef(summary(model))
    row_name <- grep("^step_TFstep$", rownames(co), value = TRUE)[1]
    if (!is.na(row_name)) {{
      est <- co[row_name, "Estimate"]
      se <- co[row_name, "Std. Error"]
      dfv <- co[row_name, "df"]
      tv <- co[row_name, "t value"]
      pv <- co[row_name, "Pr(>|t|)"]
    }} else {{
      est <- NA; se <- NA; dfv <- NA; tv <- NA; pv <- NA
    }}
    results <- rbind(results, data.frame(
      dv = dv, estimate = est, SE = se, df = dfv, t_value = tv, p_value = pv,
      mean_step = m_s, sd_step = sd_s, mean_nonstep = m_ns, sd_nonstep = sd_ns,
      n_step = n_s, n_nonstep = n_ns, converged = TRUE, stringsAsFactors = FALSE
    ))
  }}, error = function(e) {{
    results <<- rbind(results, data.frame(
      dv = dv, estimate = NA, SE = NA, df = NA, t_value = NA, p_value = NA,
      mean_step = m_s, sd_step = sd_s, mean_nonstep = m_ns, sd_nonstep = sd_ns,
      n_step = n_s, n_nonstep = n_ns, converged = FALSE, stringsAsFactors = FALSE
    ))
  }})
}}

write.csv(results, "{out_path}", row.names = FALSE)
cat("LMM fitting complete:", nrow(results), "models\\n")
"""


def fit_lmm(df: pd.DataFrame, dv_cols: list[str], timeout_sec: int) -> pd.DataFrame:
    export = df[TRIAL_KEYS + ["step_TF"] + dv_cols].copy()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, dir=tempfile.gettempdir(), newline="") as f:
        export.to_csv(f, index=False)
        input_csv = f.name

    result_csv = tempfile.mktemp(suffix="_footnorm_raw_lmm.csv", dir=tempfile.gettempdir())
    r_script = tempfile.mktemp(suffix="_footnorm_raw_lmm.R", dir=tempfile.gettempdir())

    try:
        Path(r_script).write_text(
            build_r_code(input_csv.replace("\\", "/"), result_csv.replace("\\", "/"), dv_cols),
            encoding="utf-8",
        )
        rscript = resolve_rscript()
        proc = subprocess.run(
            [rscript, r_script],
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Rscript failed code={proc.returncode}\n{proc.stderr[:2000]}")
        out = pd.read_csv(result_csv)
    finally:
        for path in [input_csv, result_csv, r_script]:
            try:
                os.unlink(path)
            except OSError:
                pass

    out["p_fdr"] = np.nan
    valid = out["p_value"].notna()
    if int(valid.sum()) > 0:
        _, p_fdr, _, _ = multipletests(out.loc[valid, "p_value"].values, method="fdr_bh")
        out.loc[valid, "p_fdr"] = p_fdr

    def to_sig(p: float) -> str:
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    out["sig"] = out["p_fdr"].apply(to_sig)
    return out


def write_utf8_sig_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def write_report(
    out_md: Path,
    input_csv: Path,
    counts: dict[str, int],
    results: pd.DataFrame,
) -> None:
    view_cols = [
        "dv",
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
    table = results[view_cols].copy()
    float_cols = [
        "estimate",
        "SE",
        "df",
        "t_value",
        "p_value",
        "p_fdr",
        "mean_step",
        "sd_step",
        "mean_nonstep",
        "sd_nonstep",
    ]
    for c in float_cols:
        table[c] = table[c].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")

    lines: list[str] = []
    lines.append("# TEMP: Raw Foot-Length Normalized LMM (Step vs Nonstep)")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- Input CSV: `{input_csv}`")
    lines.append(
        f"- Trials={counts['n_trials']}, Subjects={counts['n_subjects']}, step={counts['n_step']}, nonstep={counts['n_nonstep']}"
    )
    lines.append("")
    lines.append("## DVs")
    lines.append("")
    lines.append("- `DV1_norm = (xCOM_X - BOS_minX) / foot_len_m`")
    lines.append("- `DV1_abs_cm = (xCOM_X - BOS_minX) * 100`")
    lines.append("- `DV2_norm = (COM_X - BOS_minX) / foot_len_m`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(table.to_markdown(index=False))
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = load_raw_timeseries(args.csv)
    validate_minimum(ts)
    platform = load_platform_labels(args.platform_xlsm)
    foot = load_foot_length_from_meta(args.platform_xlsm)
    foot_pl = pl.from_pandas(foot)

    ts_with_dv = (
        ts.join(foot_pl, on="subject", how="left")
        .with_columns(
            pl.when(pl.col("foot_len_m") > 0)
            .then((pl.col("xCOM_X") - pl.col("BOS_minX")) / pl.col("foot_len_m"))
            .otherwise(None)
            .alias("dv1_norm"),
            ((pl.col("xCOM_X") - pl.col("BOS_minX")) * 100.0).alias("dv1_abs_cm"),
            pl.when(pl.col("foot_len_m") > 0)
            .then((pl.col("COM_X") - pl.col("BOS_minX")) / pl.col("foot_len_m"))
            .otherwise(None)
            .alias("dv2_norm"),
        )
    )

    # Validate foot length only for subjects present in this raw analysis set.
    foot_check = (
        ts_with_dv.select(["subject", "foot_len_m"])
        .unique()
        .filter(pl.col("foot_len_m").is_null() | (pl.col("foot_len_m") <= 0))
    )
    if foot_check.height > 0:
        bad_subj = ", ".join(sorted(set(foot_check["subject"].to_list())))
        raise ValueError(f"invalid foot_len_m for analysis subjects: {bad_subj}")

    onset_expr = pl.col("platform_onset_local")
    frame_300_expr = pl.col("platform_onset_local") + int(args.frames_300ms)
    snap_onset = build_snapshot(ts_with_dv, onset_expr, "onset")
    snap_300 = build_snapshot(ts_with_dv, frame_300_expr, "300ms")

    merged = (
        snap_onset.join(snap_300, on=TRIAL_KEYS, how="full", coalesce=True)
        .join(platform, on=TRIAL_KEYS, how="left")
        .with_columns(pl.col("step_TF").cast(pl.Utf8, strict=False))
    )
    usable = merged.filter(pl.col("step_TF").is_in(["step", "nonstep"]))
    pdf = usable.to_pandas()

    counts = {
        "n_trials": int(len(pdf)),
        "n_subjects": int(pdf["subject"].nunique(dropna=True)),
        "n_step": int((pdf["step_TF"] == "step").sum()),
        "n_nonstep": int((pdf["step_TF"] == "nonstep").sum()),
    }

    dv_cols = [
        "DV1_norm_onset",
        "DV1_norm_300ms",
        "DV1_abs_cm_onset",
        "DV1_abs_cm_300ms",
        "DV2_norm_onset",
        "DV2_norm_300ms",
    ]
    lmm = fit_lmm(pdf, dv_cols, timeout_sec=int(args.timeout_sec))

    out_csv = out_dir / "tmp_raw_footnorm_lmm_results.csv"
    out_md = out_dir / "tmp_report_raw_footnorm_lmm.md"
    write_utf8_sig_csv(lmm, out_csv)
    write_report(out_md, args.csv, counts, lmm)

    print(f"[OK] results: {out_csv}")
    print(f"[OK] report:  {out_md}")
    print(
        f"[INFO] counts: trials={counts['n_trials']}, subjects={counts['n_subjects']}, step={counts['n_step']}, nonstep={counts['n_nonstep']}"
    )


if __name__ == "__main__":
    main()
