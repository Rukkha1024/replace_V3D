"""COM vs xCOM Stepping Predictor Analysis.

Answers: "Is it correct that COM must leave BOS for stepping to occur?"

Key finding: Stepping is proactive/anticipatory — MoS < 0 is NOT a trigger.

Produces:
  - 6 publication-quality figures (saved alongside this script)
  - stdout summary statistics

Usage:
    conda run -n module python analysis/com_vs_xcom_stepping/analyze_com_vs_xcom_stepping.py
"""

from __future__ import annotations

import argparse
import sys
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
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, roc_curve

from replace_v3d.geometry.geometry2d import convex_hull_2d, signed_min_distance_point_to_polygon
from replace_v3d.io.c3d_reader import read_c3d_points
from replace_v3d.mos.core import BOS_MARKERS_DEFAULT

matplotlib.use("Agg")

# Korean font support: find an available Korean font
_KO_FONTS = ("Malgun Gothic", "NanumGothic", "AppleGothic")
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _fname in _KO_FONTS:
    if _fname in _available:
        plt.rcParams["font.family"] = _fname
        break
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_C3D_DIR = REPO_ROOT / "data" / "all_data"
DEFAULT_OUT_DIR = SCRIPT_DIR  # figures saved alongside script

TRIAL_KEYS = ["subject", "velocity", "trial"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--c3d_dir", type=Path, default=DEFAULT_C3D_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    return ap.parse_args()


def load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(str(path), encoding="utf8-lossy", infer_schema_length=10000)


def load_platform_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_excel(str(path), sheet_name="platform")
    df["subject"] = df["subject"].astype(str).str.strip()
    df["velocity"] = pd.to_numeric(df["velocity"], errors="coerce")
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce").astype("Int64")
    df["step_TF"] = df["step_TF"].astype(str).str.strip()
    df["state"] = df["state"].astype(str).str.strip()
    return df


def _resolve_c3d_for_trial(
    c3d_dir: Path,
    event_xlsm: Path,
    subject: str,
    velocity: float,
    trial: int,
) -> Path | None:
    """Locate matching C3D via the existing batch_utils pattern."""
    from replace_v3d.cli.batch_utils import iter_c3d_files
    from replace_v3d.io.events_excel import (
        parse_subject_velocity_trial_from_filename,
        resolve_subject_from_token,
    )

    if not c3d_dir.exists():
        return None
    for path in iter_c3d_files(c3d_dir):
        try:
            token, vel, tri = parse_subject_velocity_trial_from_filename(path.name)
        except Exception:
            continue
        if int(tri) != int(trial):
            continue
        if not np.isclose(float(vel), float(velocity), atol=1e-9):
            continue
        try:
            subj = resolve_subject_from_token(event_xlsm, token)
        except Exception:
            continue
        if str(subj).strip() == str(subject).strip():
            return path
    return None


def compute_com_to_hull_signed_dist(
    c3d_path: Path,
    mocap_frame: int,
    com_xy: np.ndarray,
) -> float:
    """Compute signed distance from COM(XY) to BOS convex hull at a single frame."""
    c3d = read_c3d_points(c3d_path)
    label_to_idx = {lab: i for i, lab in enumerate(c3d.labels)}
    idx_list = []
    for m in BOS_MARKERS_DEFAULT:
        if m not in label_to_idx:
            return float("nan")
        idx_list.append(label_to_idx[m])

    c3d_idx = int(mocap_frame) - 1
    if c3d_idx < 0 or c3d_idx >= c3d.points.shape[0]:
        return float("nan")

    foot_pts = c3d.points[c3d_idx, idx_list, :2].astype(float)
    hull = convex_hull_2d(foot_pts)
    if len(hull) < 3:
        return float("nan")
    return signed_min_distance_point_to_polygon(np.asarray(com_xy, dtype=float), hull)


# ---------------------------------------------------------------------------
# Milestone 1: data preparation
# ---------------------------------------------------------------------------

def build_trial_summary(
    df: pl.DataFrame,
    platform: pd.DataFrame,
    c3d_dir: Path,
    event_xlsm: Path,
) -> pd.DataFrame:
    """Build per-trial snapshot at reference timepoint."""

    # unique trials from CSV
    trials = (
        df.select(TRIAL_KEYS + ["step_onset_local"])
        .group_by(TRIAL_KEYS)
        .agg(pl.col("step_onset_local").drop_nulls().first().alias("step_onset_local"))
        .sort(TRIAL_KEYS)
        .to_pandas()
    )
    trials["subject"] = trials["subject"].astype(str).str.strip()
    trials["trial"] = trials["trial"].astype(int)

    # join step_TF and state from platform sheet
    plat_sub = platform[["subject", "velocity", "trial", "step_TF", "state"]].copy()
    trials = trials.merge(plat_sub, on=["subject", "velocity", "trial"], how="left")
    trials["step_TF"] = trials["step_TF"].fillna("unknown")
    trials["state"] = trials["state"].fillna("unknown")

    # Compute ref_frame:
    # stepping -> step_onset_local
    # nonstep -> mean step_onset_local of stepping trials in same (subject, velocity)
    trials["ref_frame"] = np.nan

    # For stepping trials with step_onset_local
    step_mask = (trials["step_TF"] == "step") & trials["step_onset_local"].notna()
    trials.loc[step_mask, "ref_frame"] = trials.loc[step_mask, "step_onset_local"]

    # Mean step_onset per (subject, velocity) from stepping trials
    step_means = (
        trials.loc[step_mask]
        .groupby(["subject", "velocity"])["step_onset_local"]
        .mean()
        .reset_index()
        .rename(columns={"step_onset_local": "mean_step_onset"})
    )

    # For nonstep or stepping-without-onset: use mean
    needs_ref = trials["ref_frame"].isna()
    if needs_ref.any():
        trials = trials.merge(step_means, on=["subject", "velocity"], how="left")
        trials.loc[needs_ref & trials["mean_step_onset"].notna(), "ref_frame"] = \
            trials.loc[needs_ref & trials["mean_step_onset"].notna(), "mean_step_onset"]
        trials.drop(columns=["mean_step_onset"], inplace=True)

    # Round ref_frame to nearest integer
    trials["ref_frame"] = trials["ref_frame"].round().astype("Int64")

    # Drop trials with no ref_frame
    n_before = len(trials)
    trials = trials.dropna(subset=["ref_frame"]).reset_index(drop=True)
    n_after = len(trials)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} trials with no computable ref_frame")

    # Extract snapshot from CSV at ref_frame
    snapshot_cols = [
        "COM_X", "COM_Y", "xCOM_X", "xCOM_Y",
        "BOS_minX", "BOS_maxX", "BOS_minY", "BOS_maxY",
        "MOS_minDist_signed", "MOS_AP_v3d", "MOS_ML_v3d",
    ]
    snap_records = []
    df_pd = df.select(TRIAL_KEYS + ["MocapFrame"] + snapshot_cols).to_pandas()

    for _, row in trials.iterrows():
        subj, vel, tri, ref = row["subject"], row["velocity"], int(row["trial"]), int(row["ref_frame"])
        mask = (
            (df_pd["subject"].astype(str).str.strip() == subj)
            & (np.isclose(df_pd["velocity"], vel, atol=1e-9))
            & (df_pd["trial"] == tri)
            & (df_pd["MocapFrame"] == ref)
        )
        matched = df_pd.loc[mask]
        if matched.empty:
            # Try nearest frame
            trial_mask = (
                (df_pd["subject"].astype(str).str.strip() == subj)
                & (np.isclose(df_pd["velocity"], vel, atol=1e-9))
                & (df_pd["trial"] == tri)
            )
            trial_data = df_pd.loc[trial_mask]
            if trial_data.empty:
                snap_records.append({c: np.nan for c in snapshot_cols})
                continue
            nearest_idx = (trial_data["MocapFrame"] - ref).abs().idxmin()
            matched = trial_data.loc[[nearest_idx]]

        vals = matched.iloc[0][snapshot_cols].to_dict()
        snap_records.append(vals)

    snap_df = pd.DataFrame(snap_records)
    for col in snapshot_cols:
        trials[col] = snap_df[col].values

    # Compute COM-to-hull signed distance from C3D
    print("  Computing COM-to-hull distances from C3D files...")
    com_signed_dists = []
    c3d_cache: dict[str, Path | None] = {}
    for i, row in trials.iterrows():
        subj = row["subject"]
        vel = row["velocity"]
        tri = int(row["trial"])
        ref = int(row["ref_frame"])
        com_xy = np.array([row["COM_X"], row["COM_Y"]])

        if np.any(np.isnan(com_xy)):
            com_signed_dists.append(np.nan)
            continue

        cache_key = f"{subj}__{vel}__{tri}"
        if cache_key not in c3d_cache:
            c3d_cache[cache_key] = _resolve_c3d_for_trial(
                c3d_dir, event_xlsm, subj, vel, tri
            )
        c3d_path = c3d_cache[cache_key]
        if c3d_path is None:
            com_signed_dists.append(np.nan)
            continue

        dist = compute_com_to_hull_signed_dist(c3d_path, ref, com_xy)
        com_signed_dists.append(dist)

    trials["COM_signed_dist"] = com_signed_dists

    # Derived boolean columns
    trials["COM_inside_BOS_hull"] = trials["COM_signed_dist"] > 0
    trials["xCOM_inside_BOS_hull"] = trials["MOS_minDist_signed"] > 0
    trials["COM_inside_BOS_AABB"] = (
        (trials["COM_X"] >= trials["BOS_minX"])
        & (trials["COM_X"] <= trials["BOS_maxX"])
        & (trials["COM_Y"] >= trials["BOS_minY"])
        & (trials["COM_Y"] <= trials["BOS_maxY"])
    )

    print(f"  Trial summary: {len(trials)} trials, "
          f"step={sum(trials['step_TF'] == 'step')}, nonstep={sum(trials['step_TF'] == 'nonstep')}")

    return trials


# ---------------------------------------------------------------------------
# Milestone 1b: MoS < 0 timing analysis
# ---------------------------------------------------------------------------

def analyze_mos_negative_timing(
    df: pl.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """For stepping trials, analyze when MoS < 0 occurs relative to step onset.

    Returns a DataFrame with per-stepping-trial timing info.
    """
    step_trials = trials[
        (trials["step_TF"] == "step") & trials["step_onset_local"].notna()
    ].copy()

    if step_trials.empty:
        return pd.DataFrame()

    df_pd = df.select(
        TRIAL_KEYS + ["MocapFrame", "MOS_minDist_signed", "time_from_platform_onset_s"]
    ).to_pandas()

    records = []
    for _, row in step_trials.iterrows():
        subj = row["subject"]
        vel = row["velocity"]
        tri = int(row["trial"])
        step_onset = int(row["step_onset_local"])

        trial_mask = (
            (df_pd["subject"].astype(str).str.strip() == subj)
            & (np.isclose(df_pd["velocity"], vel, atol=1e-9))
            & (df_pd["trial"] == tri)
        )
        tdata = df_pd.loc[trial_mask].sort_values("MocapFrame")
        if tdata.empty:
            continue

        mos_at_onset = tdata.loc[
            (tdata["MocapFrame"] - step_onset).abs().idxmin(), "MOS_minDist_signed"
        ]

        # MoS values after step onset
        after_onset = tdata[tdata["MocapFrame"] >= step_onset]
        mos_min_after = after_onset["MOS_minDist_signed"].min() if not after_onset.empty else np.nan
        mos_ever_negative_after = bool(mos_min_after < 0) if np.isfinite(mos_min_after) else False

        # Frame where MoS reaches minimum after step onset
        if not after_onset.empty and np.isfinite(mos_min_after):
            min_idx = after_onset["MOS_minDist_signed"].idxmin()
            min_frame = after_onset.loc[min_idx, "MocapFrame"]
            # Time difference in ms (mocap 100 Hz → 10 ms per frame)
            mos_min_delay_ms = (min_frame - step_onset) * 10.0
        else:
            mos_min_delay_ms = np.nan

        records.append({
            "subject": subj,
            "velocity": vel,
            "trial": tri,
            "step_onset_local": step_onset,
            "mos_at_onset": mos_at_onset,
            "mos_negative_at_onset": bool(mos_at_onset < 0) if np.isfinite(mos_at_onset) else False,
            "mos_ever_negative_after": mos_ever_negative_after,
            "mos_min_after_onset": mos_min_after,
            "mos_min_delay_ms": mos_min_delay_ms,
        })

    return pd.DataFrame(records)


def print_mos_timing_summary(timing_df: pd.DataFrame) -> None:
    """Print MoS < 0 timing analysis to stdout."""
    if timing_df.empty:
        print("  No stepping trials with step_onset for MoS timing analysis.")
        return

    n = len(timing_df)
    n_neg_onset = timing_df["mos_negative_at_onset"].sum()
    n_neg_ever = timing_df["mos_ever_negative_after"].sum()
    median_delay = timing_df.loc[timing_df["mos_ever_negative_after"], "mos_min_delay_ms"].median()

    print(f"  Stepping trials analyzed: {n}")
    print(f"  MoS < 0 at step onset:   {n_neg_onset}/{n} ({100*n_neg_onset/n:.1f}%)")
    print(f"  MoS < 0 ever after onset: {n_neg_ever}/{n} ({100*n_neg_ever/n:.1f}%)")
    if np.isfinite(median_delay):
        print(f"  Median MoS-min delay:     {median_delay:.0f} ms after step onset")

    # Per-velocity breakdown
    print("\n  Per-velocity breakdown:")
    print(f"  {'vel':>5s} | {'n_step':>6s} | {'step_rate':>9s} | {'MoS<0@onset':>12s} | {'MoS<0_ever':>10s}")
    print(f"  {'-'*5}-+-{'-'*6}-+-{'-'*9}-+-{'-'*12}-+-{'-'*10}")
    for vel in sorted(timing_df["velocity"].unique()):
        vdf = timing_df[timing_df["velocity"] == vel]
        nv = len(vdf)
        neg_onset = vdf["mos_negative_at_onset"].sum()
        neg_ever = vdf["mos_ever_negative_after"].sum()
        print(f"  {vel:5.0f} | {nv:6d} | {'—':>9s} | {neg_onset:3d}/{nv:3d} ({100*neg_onset/nv:4.1f}%) | {neg_ever:3d}/{nv:3d} ({100*neg_ever/nv:4.1f}%)")


# ---------------------------------------------------------------------------
# Milestone 2: statistics
# ---------------------------------------------------------------------------

def compute_contingency(trials: pd.DataFrame, inside_col: str, label: str) -> dict:
    """2x2 contingency: inside_col x step_TF."""
    valid = trials.dropna(subset=[inside_col])
    valid = valid[valid["step_TF"].isin(["step", "nonstep"])]

    ct = pd.crosstab(valid[inside_col], valid["step_TF"])
    ct.index = ct.index.map({True: "inside", False: "outside"})
    ct = ct.reindex(index=["inside", "outside"], columns=["step", "nonstep"]).fillna(0).astype(int)

    table = ct.values
    if table.size == 4 and table.min() >= 0:
        chi2, chi2_p, dof, _ = sp_stats.chi2_contingency(table, correction=True)
        _, fisher_p = sp_stats.fisher_exact(table)
    else:
        chi2, chi2_p, dof, fisher_p = np.nan, np.nan, np.nan, np.nan

    return {
        "label": label,
        "table": ct,
        "chi2": chi2,
        "chi2_p": chi2_p,
        "dof": dof,
        "fisher_p": fisher_p,
        "n": len(valid),
    }


def compute_mann_whitney(trials: pd.DataFrame, col: str, label: str) -> dict:
    """Mann-Whitney U: col between step vs nonstep."""
    valid = trials.dropna(subset=[col])
    valid = valid[valid["step_TF"].isin(["step", "nonstep"])]
    step_vals = valid.loc[valid["step_TF"] == "step", col].values.astype(float)
    nonstep_vals = valid.loc[valid["step_TF"] == "nonstep", col].values.astype(float)

    if len(step_vals) < 2 or len(nonstep_vals) < 2:
        return {"label": label, "U": np.nan, "p": np.nan, "rbc": np.nan,
                "n_step": len(step_vals), "n_nonstep": len(nonstep_vals),
                "median_step": np.nan, "median_nonstep": np.nan}

    U, p = sp_stats.mannwhitneyu(step_vals, nonstep_vals, alternative="two-sided")
    n1, n2 = len(step_vals), len(nonstep_vals)
    rbc = 1 - (2 * U) / (n1 * n2)

    return {
        "label": label,
        "U": U,
        "p": p,
        "rbc": rbc,
        "n_step": n1,
        "n_nonstep": n2,
        "median_step": float(np.median(step_vals)),
        "median_nonstep": float(np.median(nonstep_vals)),
    }


def compute_roc(trials: pd.DataFrame, col: str, label: str) -> dict:
    """ROC/AUC for stepping prediction. Lower values -> more likely to step."""
    valid = trials.dropna(subset=[col])
    valid = valid[valid["step_TF"].isin(["step", "nonstep"])]
    y_true = (valid["step_TF"] == "step").astype(int).values
    scores = -valid[col].values.astype(float)  # negate: lower stability -> higher score

    if len(np.unique(y_true)) < 2:
        return {"label": label, "auc": np.nan, "fpr": np.array([]), "tpr": np.array([]),
                "thresholds": np.array([]), "n": len(valid)}

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    # optimal threshold (Youden's J)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_threshold = -thresholds[opt_idx]  # un-negate

    return {
        "label": label,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "opt_threshold": opt_threshold,
        "opt_sensitivity": tpr[opt_idx],
        "opt_specificity": 1 - fpr[opt_idx],
        "n": len(valid),
    }


# ---------------------------------------------------------------------------
# Milestone 3: figures
# ---------------------------------------------------------------------------

def _step_nonstep_colors():
    return {"step": "#E74C3C", "nonstep": "#3498DB"}


def fig1_contingency_bars(
    cont_com: dict, cont_xcom: dict, out_dir: Path, dpi: int
) -> None:
    """Side-by-side grouped bar charts for COM and xCOM contingency tables."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, cont, title in zip(
        axes,
        [cont_com, cont_xcom],
        ["COM inside BOS hull", "xCOM inside BOS hull"],
    ):
        ct = cont["table"]
        categories = ct.index.tolist()
        step_vals = ct["step"].values if "step" in ct.columns else np.zeros(len(categories))
        nonstep_vals = ct["nonstep"].values if "nonstep" in ct.columns else np.zeros(len(categories))

        x = np.arange(len(categories))
        width = 0.35
        colors = _step_nonstep_colors()
        bars1 = ax.bar(x - width / 2, step_vals, width, label="step", color=colors["step"], alpha=0.85)
        bars2 = ax.bar(x + width / 2, nonstep_vals, width, label="nonstep", color=colors["nonstep"], alpha=0.85)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            str(int(h)), ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

        p_text = f"Fisher p={cont['fisher_p']:.4f}" if np.isfinite(cont['fisher_p']) else "Fisher p=N/A"
        ax.text(0.98, 0.95, p_text, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    fig.suptitle("Contingency: Position Inside BOS vs Stepping Outcome", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_contingency_bars.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig2_mos_signed_distribution(trials: pd.DataFrame, mw: dict, out_dir: Path, dpi: int) -> None:
    """Violin + strip of MOS_minDist_signed by step_TF."""
    import seaborn as sns

    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].dropna(subset=["MOS_minDist_signed"])
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = _step_nonstep_colors()
    palette = [colors["step"], colors["nonstep"]]

    sns.violinplot(data=valid, x="step_TF", y="MOS_minDist_signed", hue="step_TF",
                   order=["step", "nonstep"], hue_order=["step", "nonstep"],
                   palette=palette, inner=None, alpha=0.3, legend=False, ax=ax)
    sns.stripplot(data=valid, x="step_TF", y="MOS_minDist_signed", hue="step_TF",
                  order=["step", "nonstep"], hue_order=["step", "nonstep"],
                  palette=palette, size=4, alpha=0.6, jitter=0.2, legend=False, ax=ax)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="MoS = 0 (boundary)")
    ax.set_ylabel("MoS (xCOM-to-hull signed distance, m)", fontsize=10)
    ax.set_xlabel("Stepping outcome", fontsize=10)
    ax.set_title("xCOM-based MoS at Reference Timepoint", fontsize=11, fontweight="bold")

    stat_text = (
        f"Mann-Whitney U={mw['U']:.0f}\n"
        f"p={mw['p']:.2e}\n"
        f"r={mw['rbc']:.3f} (rank-biserial)"
    )
    ax.text(0.98, 0.02, stat_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_mos_signed_distribution.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig3_com_signed_distribution(trials: pd.DataFrame, mw: dict, out_dir: Path, dpi: int) -> None:
    """Violin + strip of COM_signed_dist by step_TF."""
    import seaborn as sns

    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].dropna(subset=["COM_signed_dist"])
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = _step_nonstep_colors()
    palette = [colors["step"], colors["nonstep"]]

    sns.violinplot(data=valid, x="step_TF", y="COM_signed_dist", hue="step_TF",
                   order=["step", "nonstep"], hue_order=["step", "nonstep"],
                   palette=palette, inner=None, alpha=0.3, legend=False, ax=ax)
    sns.stripplot(data=valid, x="step_TF", y="COM_signed_dist", hue="step_TF",
                  order=["step", "nonstep"], hue_order=["step", "nonstep"],
                  palette=palette, size=4, alpha=0.6, jitter=0.2, legend=False, ax=ax)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Distance = 0 (boundary)")
    ax.set_ylabel("COM-to-hull signed distance (m)", fontsize=10)
    ax.set_xlabel("Stepping outcome", fontsize=10)
    ax.set_title("COM-based Distance to BOS Hull at Reference Timepoint", fontsize=11, fontweight="bold")

    stat_text = (
        f"Mann-Whitney U={mw['U']:.0f}\n"
        f"p={mw['p']:.2e}\n"
        f"r={mw['rbc']:.3f} (rank-biserial)"
    )
    ax.text(0.98, 0.02, stat_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_com_signed_distribution.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4_scatter_com_vs_xcom(trials: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    """Scatter: COM_signed_dist vs MOS_minDist_signed, colored by step_TF."""
    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].dropna(
        subset=["COM_signed_dist", "MOS_minDist_signed"]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = _step_nonstep_colors()

    for grp, color in colors.items():
        sub = valid[valid["step_TF"] == grp]
        ax.scatter(sub["COM_signed_dist"], sub["MOS_minDist_signed"],
                   c=color, label=grp, alpha=0.65, s=40, edgecolors="white", linewidths=0.3)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    qstyle = dict(fontsize=7, alpha=0.5, ha="center", va="center", style="italic")
    ax.text(np.mean([max(0, xlim[0]), xlim[1]]) * 0.5 + xlim[1] * 0.5,
            np.mean([max(0, ylim[0]), ylim[1]]) * 0.5 + ylim[1] * 0.5,
            "COM in, xCOM in", **qstyle)
    ax.text(np.mean([xlim[0], min(0, xlim[1])]) * 0.5 + xlim[0] * 0.5,
            np.mean([max(0, ylim[0]), ylim[1]]) * 0.5 + ylim[1] * 0.5,
            "COM out, xCOM in", **qstyle)
    ax.text(np.mean([max(0, xlim[0]), xlim[1]]) * 0.5 + xlim[1] * 0.5,
            np.mean([ylim[0], min(0, ylim[1])]) * 0.5 + ylim[0] * 0.5,
            "COM in, xCOM out", **qstyle)
    ax.text(np.mean([xlim[0], min(0, xlim[1])]) * 0.5 + xlim[0] * 0.5,
            np.mean([ylim[0], min(0, ylim[1])]) * 0.5 + ylim[0] * 0.5,
            "COM out, xCOM out", **qstyle)

    ax.set_xlabel("COM-to-hull signed distance (m)", fontsize=10)
    ax.set_ylabel("xCOM-to-hull MoS (m)", fontsize=10)
    ax.set_title("COM vs xCOM Distance to BOS at Reference Timepoint", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_scatter_com_vs_xcom.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig5_roc_curve(roc_mos: dict, roc_com: dict, out_dir: Path, dpi: int) -> None:
    """ROC curves for MOS_signed and COM_signed_dist as stepping predictors."""
    fig, ax = plt.subplots(figsize=(6, 5.5))

    if len(roc_mos["fpr"]) > 0:
        ax.plot(roc_mos["fpr"], roc_mos["tpr"], color="#E74C3C", linewidth=2,
                label=f"xCOM MoS (AUC={roc_mos['auc']:.3f})")
    if len(roc_com["fpr"]) > 0:
        ax.plot(roc_com["fpr"], roc_com["tpr"], color="#3498DB", linewidth=2, linestyle="--",
                label=f"COM dist (AUC={roc_com['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title("ROC: Stepping Prediction by Stability Metrics", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_roc_curve.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig6_example_timeseries(
    df: pl.DataFrame,
    trials: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    """Time series of COM_X and xCOM_X vs BOS bounds for a representative stepping trial."""
    # Pick a stepping trial with step_onset_local
    step_trials = trials[(trials["step_TF"] == "step") & trials["step_onset_local"].notna()]
    if step_trials.empty:
        print("  Warning: No stepping trial with step_onset for fig6. Skipping.")
        return

    # Pick one with median MOS_minDist_signed for representativeness
    step_trials = step_trials.sort_values("MOS_minDist_signed")
    pick = step_trials.iloc[len(step_trials) // 2]
    subj, vel, tri = pick["subject"], pick["velocity"], int(pick["trial"])

    trial_data = df.filter(
        (pl.col("subject") == subj)
        & (pl.col("velocity") == vel)
        & (pl.col("trial") == tri)
    ).sort("MocapFrame").to_pandas()

    if trial_data.empty:
        print("  Warning: Trial data empty for fig6. Skipping.")
        return

    t = trial_data["time_from_platform_onset_s"].values
    com_x = trial_data["COM_X"].values
    xcom_x = trial_data["xCOM_X"].values
    bos_min = trial_data["BOS_minX"].values
    bos_max = trial_data["BOS_maxX"].values

    platform_onset = trial_data["platform_onset_local"].iloc[0]
    step_onset = trial_data["step_onset_local"].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(t, bos_min, bos_max, alpha=0.15, color="gray", label="BOS range (X)")
    ax.plot(t, bos_min, color="gray", linewidth=0.8, alpha=0.5)
    ax.plot(t, bos_max, color="gray", linewidth=0.8, alpha=0.5)
    ax.plot(t, com_x, color="#3498DB", linewidth=2, label="COM X")
    ax.plot(t, xcom_x, color="#E74C3C", linewidth=2, linestyle="--", label="xCOM X")

    # Event lines
    onset_t = trial_data.loc[
        trial_data["MocapFrame"] == int(platform_onset), "time_from_platform_onset_s"
    ]
    if not onset_t.empty:
        ax.axvline(onset_t.iloc[0], color="black", linestyle=":", linewidth=1.2, alpha=0.7, label="Platform onset")

    if pd.notna(step_onset):
        step_t = trial_data.loc[
            trial_data["MocapFrame"] == int(step_onset), "time_from_platform_onset_s"
        ]
        if not step_t.empty:
            ax.axvline(step_t.iloc[0], color="darkorange", linestyle="-", linewidth=2, alpha=0.8, label="Step onset")

    ax.set_xlabel("Time from platform onset (s)", fontsize=10)
    ax.set_ylabel("A/P position X (m)", fontsize=10)
    ax.set_title(
        f"Example Stepping Trial: {subj}, vel={vel}, trial={tri}\n"
        f"COM vs xCOM X trajectory relative to BOS",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_example_timeseries.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    print("=" * 60)
    print("COM vs xCOM Stepping Predictor Analysis")
    print("=" * 60)

    # --- Milestone 1 ---
    print("\n[Milestone 1] Loading data and computing distances...")
    df = load_csv(args.csv)
    platform = load_platform_sheet(args.platform_xlsm)
    trials = build_trial_summary(df, platform, Path(args.c3d_dir), Path(args.platform_xlsm))

    # --- Milestone 1b: MoS < 0 timing ---
    print("\n[Milestone 1b] MoS < 0 timing analysis (stepping trials)...")
    timing_df = analyze_mos_negative_timing(df, trials)
    print_mos_timing_summary(timing_df)

    # --- Milestone 2 ---
    print("\n[Milestone 2] Statistical tests...")
    cont_com = compute_contingency(trials, "COM_inside_BOS_hull", "COM_inside_BOS_hull")
    cont_xcom = compute_contingency(trials, "xCOM_inside_BOS_hull", "xCOM_inside_BOS_hull")

    print(f"  COM contingency:\n{cont_com['table']}")
    print(f"  Fisher p={cont_com['fisher_p']:.4f}")
    print(f"  xCOM contingency:\n{cont_xcom['table']}")
    print(f"  Fisher p={cont_xcom['fisher_p']:.4f}")

    mw_mos = compute_mann_whitney(trials, "MOS_minDist_signed", "MOS_minDist_signed")
    mw_com = compute_mann_whitney(trials, "COM_signed_dist", "COM_signed_dist")
    print(f"  MoS Mann-Whitney: U={mw_mos['U']:.0f}, p={mw_mos['p']:.2e}, r={mw_mos['rbc']:.3f}")
    print(f"  COM Mann-Whitney: U={mw_com['U']:.0f}, p={mw_com['p']:.2e}, r={mw_com['rbc']:.3f}")

    roc_mos = compute_roc(trials, "MOS_minDist_signed", "MOS_minDist_signed")
    roc_com = compute_roc(trials, "COM_signed_dist", "COM_signed_dist")
    print(f"  MoS ROC AUC={roc_mos['auc']:.3f}")
    print(f"  COM ROC AUC={roc_com['auc']:.3f}")

    # --- Milestone 3 ---
    print("\n[Milestone 3] Generating figures...")
    fig1_contingency_bars(cont_com, cont_xcom, out_dir, dpi)
    print("  fig1_contingency_bars.png")

    fig2_mos_signed_distribution(trials, mw_mos, out_dir, dpi)
    print("  fig2_mos_signed_distribution.png")

    fig3_com_signed_distribution(trials, mw_com, out_dir, dpi)
    print("  fig3_com_signed_distribution.png")

    fig4_scatter_com_vs_xcom(trials, out_dir, dpi)
    print("  fig4_scatter_com_vs_xcom.png")

    fig5_roc_curve(roc_mos, roc_com, out_dir, dpi)
    print("  fig5_roc_curve.png")

    fig6_example_timeseries(df, trials, out_dir, dpi)
    print("  fig6_example_timeseries.png")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Output directory: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
