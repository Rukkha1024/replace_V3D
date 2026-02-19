"""eBOS & FSR State-Space Analysis: Why Stepping Occurs Before the Textbook Threshold.

Answers: "Can effective BOS (eBOS) and COM velocity-position state space (FSR)
explain why stepping occurs when xCOM is still within the anatomical BOS?"

Extends: analysis/com_vs_xcom_stepping (MoS AUC=0.783, 93.6% stepping with MoS>0)

Analysis 1 — Effective BOS (eBOS):
  Hof & Curtze (2016): eBOS ~ 30% of physical BOS. COP excursion envelope
  from nonstep trials defines the functional boundary. eBOS-MoS may better
  predict stepping than physical-BOS-MoS.

Analysis 2 — COM Velocity-Position State Space (FSR):
  Pai & Patton (1997): Stepping is driven by COM velocity, not just position.
  2D (position x velocity) GLMM outperforms 1D MoS for stepping prediction.

Produces:
  - 8 publication-quality figures (saved alongside this script)
  - stdout summary statistics

Usage:
    conda run -n module python analysis/why_stepping_before_threshold/analyze_ebos_and_fsr.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

# Korean font support
_KO_FONTS = ("Malgun Gothic", "NanumGothic", "AppleGothic")
_available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _fname in _KO_FONTS:
    if _fname in _available:
        plt.rcParams["font.family"] = _fname
        break
plt.rcParams["axes.unicode_minus"] = False

from replace_v3d.geometry.geometry2d import (
    convex_hull_2d,
    polygon_area,
    signed_min_distance_point_to_polygon,
)
from replace_v3d.io.c3d_reader import read_c3d_points
from replace_v3d.io.events_excel import load_subject_leg_length_cm
from replace_v3d.mos.core import BOS_MARKERS_DEFAULT

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_C3D_DIR = REPO_ROOT / "data" / "all_data"
DEFAULT_OUT_DIR = SCRIPT_DIR

TRIAL_KEYS = ["subject", "velocity", "trial"]
G = 9.81  # m/s^2


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
    ap.add_argument("--c3d_dir", type=Path, default=DEFAULT_C3D_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=200)
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers (adapted from com_vs_xcom_stepping)
# ═══════════════════════════════════════════════════════════════════════════


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


def _get_physical_bos_hull_at_frame(
    c3d_path: Path, mocap_frame: int
) -> np.ndarray | None:
    """Return BOS convex hull (N,2) at a single mocap frame from C3D."""
    c3d = read_c3d_points(c3d_path)
    label_to_idx = {lab: i for i, lab in enumerate(c3d.labels)}
    idx_list = []
    for m in BOS_MARKERS_DEFAULT:
        if m not in label_to_idx:
            return None
        idx_list.append(label_to_idx[m])
    c3d_idx = int(mocap_frame) - 1
    if c3d_idx < 0 or c3d_idx >= c3d.points.shape[0]:
        return None
    foot_pts = c3d.points[c3d_idx, idx_list, :2].astype(float)
    hull = convex_hull_2d(foot_pts)
    if len(hull) < 3:
        return None
    return hull


# ═══════════════════════════════════════════════════════════════════════════
# Milestone 1: Data Preparation
# ═══════════════════════════════════════════════════════════════════════════


def build_trial_summary(
    df: pl.DataFrame,
    platform: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-trial snapshot at reference timepoint (reused from com_vs_xcom_stepping)."""

    trials = (
        df.select(TRIAL_KEYS + ["step_onset_local"])
        .group_by(TRIAL_KEYS)
        .agg(pl.col("step_onset_local").drop_nulls().first().alias("step_onset_local"))
        .sort(TRIAL_KEYS)
        .to_pandas()
    )
    trials["subject"] = trials["subject"].astype(str).str.strip()
    trials["trial"] = trials["trial"].astype(int)

    plat_sub = platform[["subject", "velocity", "trial", "step_TF", "state"]].copy()
    trials = trials.merge(plat_sub, on=["subject", "velocity", "trial"], how="left")
    trials["step_TF"] = trials["step_TF"].fillna("unknown")
    trials["state"] = trials["state"].fillna("unknown")

    # ref_frame: stepping -> step_onset_local, nonstep -> mean of stepping in same (subject, velocity)
    trials["ref_frame"] = np.nan
    step_mask = (trials["step_TF"] == "step") & trials["step_onset_local"].notna()
    trials.loc[step_mask, "ref_frame"] = trials.loc[step_mask, "step_onset_local"]

    step_means = (
        trials.loc[step_mask]
        .groupby(["subject", "velocity"])["step_onset_local"]
        .mean()
        .reset_index()
        .rename(columns={"step_onset_local": "mean_step_onset"})
    )
    needs_ref = trials["ref_frame"].isna()
    if needs_ref.any():
        trials = trials.merge(step_means, on=["subject", "velocity"], how="left")
        trials.loc[needs_ref & trials["mean_step_onset"].notna(), "ref_frame"] = (
            trials.loc[needs_ref & trials["mean_step_onset"].notna(), "mean_step_onset"]
        )
        trials.drop(columns=["mean_step_onset"], inplace=True)

    trials["ref_frame"] = trials["ref_frame"].round().astype("Int64")
    n_before = len(trials)
    trials = trials.dropna(subset=["ref_frame"]).reset_index(drop=True)
    if n_before != len(trials):
        print(f"  Dropped {n_before - len(trials)} trials with no computable ref_frame")

    # Extract snapshot from CSV at ref_frame
    snapshot_cols = [
        "COM_X", "COM_Y", "vCOM_X", "vCOM_Y",
        "xCOM_X", "xCOM_Y",
        "BOS_minX", "BOS_maxX", "BOS_minY", "BOS_maxY", "BOS_area",
        "MOS_minDist_signed", "MOS_AP_v3d", "MOS_ML_v3d",
        "COP_X_m", "COP_Y_m",
    ]
    df_pd = df.select(TRIAL_KEYS + ["MocapFrame"] + snapshot_cols).to_pandas()
    snap_records = []

    for _, row in trials.iterrows():
        subj, vel, tri, ref = (
            row["subject"], row["velocity"], int(row["trial"]), int(row["ref_frame"])
        )
        mask = (
            (df_pd["subject"].astype(str).str.strip() == subj)
            & (np.isclose(df_pd["velocity"], vel, atol=1e-9))
            & (df_pd["trial"] == tri)
            & (df_pd["MocapFrame"] == ref)
        )
        matched = df_pd.loc[mask]
        if matched.empty:
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

        snap_records.append(matched.iloc[0][snapshot_cols].to_dict())

    snap_df = pd.DataFrame(snap_records)
    for col in snapshot_cols:
        trials[col] = snap_df[col].values

    print(
        f"  Trial summary: {len(trials)} trials, "
        f"step={sum(trials['step_TF'] == 'step')}, "
        f"nonstep={sum(trials['step_TF'] == 'nonstep')}"
    )
    return trials


def load_leg_lengths(xlsm_path: Path, subjects: list[str]) -> dict[str, float]:
    """Load leg_length_m for each subject."""
    result = {}
    for subj in subjects:
        val = load_subject_leg_length_cm(xlsm_path, subj)
        if val is not None:
            result[subj] = float(val) / 100.0
        else:
            warnings.warn(f"No leg length for subject '{subj}', using default 0.9 m")
            result[subj] = 0.9
    return result


def extract_cop_timeseries_nonstep(
    df: pl.DataFrame,
    trials: pd.DataFrame,
) -> dict[str, list[np.ndarray]]:
    """Extract COP (X, Y) timeseries for nonstep trials, per subject.

    Returns dict[subject -> list of (N, 2) arrays].
    """
    nonstep = trials[trials["step_TF"] == "nonstep"]
    df_pd = df.select(
        TRIAL_KEYS + ["MocapFrame", "COP_X_m", "COP_Y_m", "platform_onset_local"]
    ).to_pandas()

    result: dict[str, list[np.ndarray]] = {}
    for _, row in nonstep.iterrows():
        subj = row["subject"]
        vel = row["velocity"]
        tri = int(row["trial"])

        mask = (
            (df_pd["subject"].astype(str).str.strip() == subj)
            & (np.isclose(df_pd["velocity"], vel, atol=1e-9))
            & (df_pd["trial"] == tri)
        )
        tdata = df_pd.loc[mask].sort_values("MocapFrame")
        if tdata.empty:
            continue

        onset = tdata["platform_onset_local"].iloc[0]
        if pd.notna(onset):
            tdata = tdata[tdata["MocapFrame"] >= int(onset)]

        cop_xy = tdata[["COP_X_m", "COP_Y_m"]].dropna().values.astype(float)
        if len(cop_xy) > 0:
            result.setdefault(subj, []).append(cop_xy)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Milestone 2: Effective BOS (eBOS)
# ═══════════════════════════════════════════════════════════════════════════


def compute_ebos_per_subject(
    cop_ts: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Compute eBOS convex hull per subject from pooled nonstep COP trajectories.

    Returns dict[subject -> (M, 2) hull polygon].
    """
    ebos = {}
    for subj, arrays in cop_ts.items():
        pooled = np.vstack(arrays)
        if len(pooled) < 3:
            continue
        hull = convex_hull_2d(pooled)
        if len(hull) >= 3:
            ebos[subj] = hull
    return ebos


def compute_physical_bos_area_per_subject(
    trials: pd.DataFrame,
    c3d_dir: Path,
    event_xlsm: Path,
) -> dict[str, float]:
    """Compute mean physical BOS area per subject from C3D at ref_frame."""
    c3d_cache: dict[str, Path | None] = {}
    subj_areas: dict[str, list[float]] = {}

    for _, row in trials.iterrows():
        subj = row["subject"]
        vel = row["velocity"]
        tri = int(row["trial"])
        ref = int(row["ref_frame"])

        cache_key = f"{subj}__{vel}__{tri}"
        if cache_key not in c3d_cache:
            c3d_cache[cache_key] = _resolve_c3d_for_trial(
                c3d_dir, event_xlsm, subj, vel, tri
            )
        c3d_path = c3d_cache[cache_key]
        if c3d_path is None:
            continue

        hull = _get_physical_bos_hull_at_frame(c3d_path, ref)
        if hull is not None:
            area = polygon_area(hull)
            subj_areas.setdefault(subj, []).append(area)

    return {s: float(np.mean(a)) for s, a in subj_areas.items()}


def compute_ebos_mos(
    trials: pd.DataFrame,
    ebos_hulls: dict[str, np.ndarray],
) -> pd.Series:
    """Compute eBOS-MoS for each trial: signed distance from xCOM to eBOS hull."""
    values = []
    for _, row in trials.iterrows():
        subj = row["subject"]
        xcom_xy = np.array([row["xCOM_X"], row["xCOM_Y"]])
        if subj not in ebos_hulls or np.any(np.isnan(xcom_xy)):
            values.append(np.nan)
            continue
        dist = signed_min_distance_point_to_polygon(xcom_xy, ebos_hulls[subj])
        values.append(dist)
    return pd.Series(values, index=trials.index, name="eBOS_MoS")


# ═══════════════════════════════════════════════════════════════════════════
# Milestone 3: FSR State Space
# ═══════════════════════════════════════════════════════════════════════════


def compute_fsr_features(
    trials: pd.DataFrame,
    leg_lengths: dict[str, float],
) -> pd.DataFrame:
    """Compute normalized COM position and velocity for FSR analysis."""
    pos_norm = []
    vel_norm = []

    for _, row in trials.iterrows():
        bos_len = row["BOS_maxX"] - row["BOS_minX"]
        subj = row["subject"]
        ll = leg_lengths.get(subj, 0.9)
        omega_0 = np.sqrt(G / ll)

        if bos_len > 0 and np.isfinite(bos_len):
            pn = (row["COM_X"] - row["BOS_minX"]) / bos_len
            vn = row["vCOM_X"] / (omega_0 * bos_len)
        else:
            pn = np.nan
            vn = np.nan

        pos_norm.append(pn)
        vel_norm.append(vn)

    trials = trials.copy()
    trials["COM_pos_norm"] = pos_norm
    trials["COM_vel_norm"] = vel_norm
    return trials


# ═══════════════════════════════════════════════════════════════════════════
# GLMM statistics
# ═══════════════════════════════════════════════════════════════════════════


def fit_glmm(
    data: pd.DataFrame,
    formula: str,
    groups: str = "subject",
) -> dict:
    """Fit Binomial GLMM (Bayesian) and return summary dict.

    Uses statsmodels BinomialBayesMixedGLM for binary outcome with
    random intercept per subject.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    # Parse formula: "y ~ x1 + x2"
    lhs, rhs = formula.split("~")
    y_col = lhs.strip()
    x_cols = [c.strip() for c in rhs.strip().split("+")]

    clean = data[[y_col] + x_cols + [groups]].dropna().copy()
    if clean[y_col].dtype == object or clean[y_col].dtype == bool:
        clean[y_col] = (clean[y_col].astype(str).str.strip() == "step").astype(int)
    else:
        clean[y_col] = clean[y_col].astype(int)

    y = clean[y_col].values
    X = clean[x_cols].values.astype(float)
    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])
    col_names = ["Intercept"] + x_cols

    # Random effects: subject as groups
    group_labels = clean[groups].values
    unique_groups = np.unique(group_labels)
    # Random intercept matrix: one column per group
    ident = np.zeros((len(clean), len(unique_groups)))
    for i, g in enumerate(unique_groups):
        ident[group_labels == g, i] = 1.0

    try:
        model = BinomialBayesMixedGLM(
            endog=y,
            exog=X_with_const,
            exog_vc=ident,
            ident=np.zeros(len(unique_groups), dtype=int),
        )
        result = model.fit_vb()

        coefs = result.fe_mean
        se = result.fe_sd
        or_vals = np.exp(coefs)
        or_ci_lo = np.exp(coefs - 1.96 * se)
        or_ci_hi = np.exp(coefs + 1.96 * se)
        # z-test p-values
        z_vals = coefs / se
        p_vals = 2 * (1 - sp_stats.norm.cdf(np.abs(z_vals)))

        return {
            "converged": True,
            "coef_names": col_names,
            "coefs": coefs,
            "se": se,
            "or": or_vals,
            "or_ci_lo": or_ci_lo,
            "or_ci_hi": or_ci_hi,
            "z": z_vals,
            "p": p_vals,
            "n": len(clean),
            "n_groups": len(unique_groups),
            "result": result,
        }
    except Exception as e:
        warnings.warn(f"GLMM fitting failed: {e}")
        return {"converged": False, "error": str(e), "n": len(clean)}


def compute_loso_cv_auc(
    data: pd.DataFrame,
    feature_cols: list[str],
    y_col: str = "step_binary",
    group_col: str = "subject",
) -> dict:
    """LOSO-CV AUC using sklearn LogisticRegression as proxy for GLMM predictions."""
    clean = data[feature_cols + [y_col, group_col]].dropna()
    X = clean[feature_cols].values.astype(float)
    y = clean[y_col].values.astype(int)
    groups = clean[group_col].values

    logo = LeaveOneGroupOut()
    scaler = StandardScaler()

    aucs = []
    all_y_true = []
    all_y_prob = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_train_s, y_train)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

        if len(np.unique(y_test)) == 2:
            aucs.append(roc_auc_score(y_test, y_prob))

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    overall_auc = roc_auc_score(all_y_true, all_y_prob) if len(np.unique(all_y_true)) == 2 else np.nan
    fpr, tpr, thresholds = roc_curve(all_y_true, all_y_prob)

    return {
        "overall_auc": overall_auc,
        "fold_aucs": aucs,
        "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
        "ci95_lo": float(np.percentile(aucs, 2.5)) if len(aucs) > 1 else np.nan,
        "ci95_hi": float(np.percentile(aucs, 97.5)) if len(aucs) > 1 else np.nan,
        "fpr": fpr,
        "tpr": tpr,
        "n": len(clean),
        "y_true": all_y_true,
        "y_prob": all_y_prob,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════


def _step_nonstep_colors():
    return {"step": "#E74C3C", "nonstep": "#3498DB"}


def fig1_cop_excursion_envelope(
    cop_ts: dict[str, list[np.ndarray]],
    ebos_hulls: dict[str, np.ndarray],
    trials: pd.DataFrame,
    c3d_dir: Path,
    event_xlsm: Path,
    out_dir: Path,
    dpi: int,
) -> None:
    """COP excursion envelope for representative subjects."""
    subjects = sorted(ebos_hulls.keys())
    # Pick up to 3 subjects with most nonstep trials
    nonstep_counts = (
        trials[trials["step_TF"] == "nonstep"]
        .groupby("subject")
        .size()
        .sort_values(ascending=False)
    )
    pick_subjects = [s for s in nonstep_counts.index if s in ebos_hulls][:3]
    if not pick_subjects:
        print("  Warning: No subjects for fig1. Skipping.")
        return

    n_sub = len(pick_subjects)
    fig, axes = plt.subplots(1, n_sub, figsize=(5 * n_sub, 5))
    if n_sub == 1:
        axes = [axes]

    for ax, subj in zip(axes, pick_subjects):
        # Plot nonstep COP trajectories
        if subj in cop_ts:
            for cop_arr in cop_ts[subj]:
                ax.plot(cop_arr[:, 0], cop_arr[:, 1], color="#3498DB",
                        alpha=0.15, linewidth=0.5)

        # Plot eBOS hull
        ebos = ebos_hulls[subj]
        hull_closed = np.vstack([ebos, ebos[0]])
        ax.plot(hull_closed[:, 0], hull_closed[:, 1], color="#E74C3C",
                linewidth=2.5, label="eBOS hull")
        ax.fill(hull_closed[:, 0], hull_closed[:, 1], color="#E74C3C", alpha=0.08)

        # Plot physical BOS from one trial
        subj_trials = trials[trials["subject"] == subj]
        if not subj_trials.empty:
            sample = subj_trials.iloc[0]
            c3d_path = _resolve_c3d_for_trial(
                c3d_dir, event_xlsm, subj, sample["velocity"], int(sample["trial"])
            )
            if c3d_path is not None:
                phys_hull = _get_physical_bos_hull_at_frame(
                    c3d_path, int(sample["ref_frame"])
                )
                if phys_hull is not None:
                    ph_closed = np.vstack([phys_hull, phys_hull[0]])
                    ax.plot(ph_closed[:, 0], ph_closed[:, 1], color="gray",
                            linewidth=2, linestyle="--", label="Physical BOS")
                    ax.fill(ph_closed[:, 0], ph_closed[:, 1], color="gray", alpha=0.05)

        # Plot stepping trial xCOM at ref_frame
        step_trials = subj_trials[subj_trials["step_TF"] == "step"]
        if not step_trials.empty:
            ax.scatter(
                step_trials["xCOM_X"], step_trials["xCOM_Y"],
                c="#E74C3C", s=30, alpha=0.7, zorder=5,
                marker="x", linewidths=1.5, label="xCOM (step)",
            )
        nonstep_trials = subj_trials[subj_trials["step_TF"] == "nonstep"]
        if not nonstep_trials.empty:
            ax.scatter(
                nonstep_trials["xCOM_X"], nonstep_trials["xCOM_Y"],
                c="#3498DB", s=30, alpha=0.7, zorder=5,
                marker="o", linewidths=1, label="xCOM (nonstep)",
            )

        ax.set_xlabel("X (A/P, m)")
        ax.set_ylabel("Y (M/L, m)")
        ax.set_title(f"Subject: {subj}", fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.set_aspect("equal")
        ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.suptitle(
        "COP Excursion Envelope & Effective BOS",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "fig1_cop_excursion_envelope.png",
        dpi=dpi, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def fig2_ebos_area_ratio(
    ebos_hulls: dict[str, np.ndarray],
    phys_bos_areas: dict[str, float],
    out_dir: Path,
    dpi: int,
) -> dict[str, float]:
    """Bar chart: eBOS area / physical BOS area per subject."""
    subjects = sorted(set(ebos_hulls.keys()) & set(phys_bos_areas.keys()))
    ratios = {}
    for s in subjects:
        ebos_area = polygon_area(ebos_hulls[s])
        phys_area = phys_bos_areas[s]
        if phys_area > 0:
            ratios[s] = ebos_area / phys_area

    if not ratios:
        print("  Warning: No data for fig2. Skipping.")
        return {}

    fig, ax = plt.subplots(figsize=(max(8, len(ratios) * 0.8), 5))
    subj_names = list(ratios.keys())
    ratio_vals = [ratios[s] for s in subj_names]

    bars = ax.bar(range(len(subj_names)), ratio_vals, color="#2ECC71", alpha=0.8,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0.30, color="#E74C3C", linestyle="--", linewidth=2,
               label="Hof & Curtze (2016): ~30%", alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xticks(range(len(subj_names)))
    ax.set_xticklabels(subj_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("eBOS Area / Physical BOS Area")
    ax.set_title("Effective BOS as Proportion of Physical BOS", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    mean_ratio = np.mean(ratio_vals)
    ax.text(
        0.98, 0.95,
        f"Mean ratio: {mean_ratio:.3f}\n(N={len(ratios)} subjects)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(
        out_dir / "fig2_ebos_area_ratio.png",
        dpi=dpi, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    return ratios


def fig3_mos_ebos_distribution(
    trials: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    """Violin+strip: eBOS-MoS vs physical-BOS-MoS by step/nonstep."""
    import seaborn as sns

    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].copy()
    colors = _step_nonstep_colors()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, col, title in zip(
        axes,
        ["MOS_minDist_signed", "eBOS_MoS"],
        ["Physical BOS MoS", "Effective BOS MoS"],
    ):
        sub = valid.dropna(subset=[col])
        palette = [colors["step"], colors["nonstep"]]

        sns.violinplot(
            data=sub, x="step_TF", y=col, hue="step_TF",
            order=["step", "nonstep"], hue_order=["step", "nonstep"],
            palette=palette, inner=None, alpha=0.3, legend=False, ax=ax,
        )
        sns.stripplot(
            data=sub, x="step_TF", y=col, hue="step_TF",
            order=["step", "nonstep"], hue_order=["step", "nonstep"],
            palette=palette, size=4, alpha=0.6, jitter=0.2, legend=False, ax=ax,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("MoS (m)" if ax == axes[0] else "")
        ax.set_xlabel("Stepping outcome")

    # Count outside eBOS
    step_trials = valid[valid["step_TF"] == "step"]
    n_outside_phys = (step_trials["MOS_minDist_signed"] < 0).sum()
    n_outside_ebos = (step_trials["eBOS_MoS"].dropna() < 0).sum()
    n_step = len(step_trials)

    axes[1].text(
        0.98, 0.02,
        f"Step outside eBOS: {n_outside_ebos}/{n_step}\n"
        f"Step outside phys: {n_outside_phys}/{n_step}",
        transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        "MoS Distribution: Physical BOS vs Effective BOS",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        out_dir / "fig3_mos_ebos_distribution.png",
        dpi=dpi, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def fig4_roc_ebos_vs_physical(
    trials: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> dict[str, float]:
    """ROC curves: eBOS-MoS vs physical-BOS-MoS."""
    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].copy()
    valid["step_binary"] = (valid["step_TF"] == "step").astype(int)

    roc_data = {}
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for col, label, color, ls in [
        ("eBOS_MoS", "eBOS MoS", "#E74C3C", "-"),
        ("MOS_minDist_signed", "Physical BOS MoS", "#3498DB", "--"),
    ]:
        sub = valid.dropna(subset=[col])
        y_true = sub["step_binary"].values
        scores = -sub[col].values.astype(float)  # lower MoS -> higher stepping score

        if len(np.unique(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, scores)
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_data[col] = auc

        ax.plot(fpr, tpr, color=color, linewidth=2, linestyle=ls,
                label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: eBOS-MoS vs Physical-BOS-MoS", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        out_dir / "fig4_roc_ebos_vs_physical.png",
        dpi=dpi, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    return roc_data


def fig5_state_space_scatter(
    trials: pd.DataFrame,
    glmm_result: dict | None,
    out_dir: Path,
    dpi: int,
) -> None:
    """State-space scatter: COM_pos_norm vs COM_vel_norm with GLMM boundary."""
    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].dropna(
        subset=["COM_pos_norm", "COM_vel_norm"]
    )
    colors = _step_nonstep_colors()

    fig, ax = plt.subplots(figsize=(8, 6))

    for grp, color in colors.items():
        sub = valid[valid["step_TF"] == grp]
        ax.scatter(
            sub["COM_pos_norm"], sub["COM_vel_norm"],
            c=color, label=grp, alpha=0.65, s=40,
            edgecolors="white", linewidths=0.3,
        )

    # Theoretical FSR boundary: approximate line from (0, 1.0) to (1, 0)
    ax.plot([0, 1], [1.0, 0], color="gray", linestyle=":", linewidth=2,
            alpha=0.5, label="Approx FSR boundary")

    # GLMM decision boundary (P(step)=0.5 contour)
    if glmm_result and glmm_result.get("converged"):
        coefs = glmm_result["coefs"]  # [intercept, pos, vel]
        if len(coefs) == 3:
            # Decision boundary: intercept + b1*pos + b2*vel = 0
            # vel = -(intercept + b1*pos) / b2
            x_range = np.linspace(
                valid["COM_pos_norm"].min() - 0.1,
                valid["COM_pos_norm"].max() + 0.1,
                100,
            )
            if abs(coefs[2]) > 1e-6:
                y_boundary = -(coefs[0] + coefs[1] * x_range) / coefs[2]
                ax.plot(x_range, y_boundary, color="black", linewidth=2,
                        linestyle="-", label="GLMM P(step)=0.5")

    ax.set_xlabel("COM position (normalized to BOS, 0=post, 1=ant)")
    ax.set_ylabel("COM velocity (normalized, dimensionless)")
    ax.set_title("COM Velocity-Position State Space (FSR)", fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        out_dir / "fig5_state_space_scatter.png",
        dpi=dpi, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def fig6_state_space_marginals(
    trials: pd.DataFrame,
    out_dir: Path,
    dpi: int,
) -> None:
    """State-space scatter with marginal histograms."""
    import seaborn as sns

    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].dropna(
        subset=["COM_pos_norm", "COM_vel_norm"]
    ).copy()
    colors = _step_nonstep_colors()

    g = sns.JointGrid(
        data=valid, x="COM_pos_norm", y="COM_vel_norm",
        hue="step_TF", hue_order=["step", "nonstep"],
        palette=colors, height=7,
    )
    g.plot_joint(sns.scatterplot, alpha=0.6, s=40, edgecolor="white", linewidth=0.3)
    g.plot_marginals(sns.histplot, kde=True, alpha=0.4, common_norm=False)

    g.ax_joint.set_xlabel("COM position (norm)", fontsize=10)
    g.ax_joint.set_ylabel("COM velocity (norm)", fontsize=10)
    g.figure.suptitle(
        "State Space with Marginal Distributions",
        fontsize=12, fontweight="bold", y=1.02,
    )
    g.figure.tight_layout()
    g.savefig(
        out_dir / "fig6_state_space_marginals.png",
        dpi=dpi, bbox_inches="tight",
    )
    plt.close(g.figure)


def fig7_roc_2d_vs_1d(
    roc_results: dict[str, dict],
    out_dir: Path,
    dpi: int,
) -> None:
    """ROC curves: 2D GLMM vs 1D models."""
    fig, ax = plt.subplots(figsize=(7, 6))

    style_map = {
        "2D (pos+vel)": ("#E74C3C", "-", 2.5),
        "1D MoS": ("#3498DB", "--", 2),
        "1D velocity": ("#2ECC71", "-.", 1.5),
        "1D position": ("#9B59B6", ":", 1.5),
    }

    for label, rdata in roc_results.items():
        color, ls, lw = style_map.get(label, ("gray", "-", 1))
        auc = rdata["overall_auc"]
        if np.isfinite(auc):
            ax.plot(rdata["fpr"], rdata["tpr"], color=color, linestyle=ls,
                    linewidth=lw, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: 2D State-Space vs 1D Models (LOSO-CV)", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        out_dir / "fig7_roc_2d_vs_1d.png",
        dpi=dpi, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def fig8_summary_auc_comparison(
    auc_summary: dict[str, dict],
    out_dir: Path,
    dpi: int,
) -> None:
    """Summary AUC bar chart with 95% CI error bars."""
    labels = list(auc_summary.keys())
    aucs = [auc_summary[k]["overall_auc"] for k in labels]
    ci_lo = [auc_summary[k].get("ci95_lo", np.nan) for k in labels]
    ci_hi = [auc_summary[k].get("ci95_hi", np.nan) for k in labels]

    # Error bars
    yerr_lo = [max(0, a - lo) if np.isfinite(lo) else 0 for a, lo in zip(aucs, ci_lo)]
    yerr_hi = [max(0, hi - a) if np.isfinite(hi) else 0 for a, hi in zip(aucs, ci_hi)]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bar_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C"]
    colors = bar_colors[: len(labels)]

    bars = ax.bar(
        range(len(labels)), aucs,
        yerr=[yerr_lo, yerr_hi],
        capsize=5, color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5,
    )

    for i, (bar, auc_val) in enumerate(zip(bars, aucs)):
        if np.isfinite(auc_val):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{auc_val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Random")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC (LOSO-CV)")
    ax.set_title("Stepping Prediction: Model Comparison", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        out_dir / "fig8_summary_auc_comparison.png",
        dpi=dpi, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    print("=" * 70)
    print("eBOS & FSR State-Space Analysis")
    print("Why Stepping Occurs Before the Textbook Threshold")
    print("=" * 70)

    # ── Milestone 1: Data Preparation ──────────────────────────────────
    print("\n[Milestone 1] Loading data...")
    df = load_csv(args.csv)
    platform = load_platform_sheet(args.platform_xlsm)
    trials = build_trial_summary(df, platform)

    subjects = sorted(trials["subject"].unique())
    print(f"  Subjects: {len(subjects)}")

    leg_lengths = load_leg_lengths(args.platform_xlsm, subjects)
    print(f"  Leg lengths loaded: {len(leg_lengths)} subjects")
    for s, ll in sorted(leg_lengths.items()):
        omega0 = np.sqrt(G / ll)
        print(f"    {s}: {ll:.3f} m, omega_0={omega0:.2f} rad/s")

    # COP timeseries for nonstep trials
    print("  Extracting COP timeseries from nonstep trials...")
    cop_ts = extract_cop_timeseries_nonstep(df, trials)
    for s in sorted(cop_ts.keys()):
        total_pts = sum(len(a) for a in cop_ts[s])
        print(f"    {s}: {len(cop_ts[s])} nonstep trials, {total_pts} COP points")

    # ── Milestone 2: eBOS Analysis ─────────────────────────────────────
    print("\n[Milestone 2] Effective BOS (eBOS) analysis...")

    # 2a: eBOS hulls
    ebos_hulls = compute_ebos_per_subject(cop_ts)
    print(f"  eBOS computed for {len(ebos_hulls)} subjects")

    # Physical BOS areas
    print("  Computing physical BOS areas from C3D...")
    phys_bos_areas = compute_physical_bos_area_per_subject(
        trials, Path(args.c3d_dir), Path(args.platform_xlsm)
    )

    # 2b: eBOS-MoS
    trials["eBOS_MoS"] = compute_ebos_mos(trials, ebos_hulls)

    # 2c: Statistics
    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].copy()
    valid["step_binary"] = (valid["step_TF"] == "step").astype(int)

    # eBOS area ratios
    print("\n  --- eBOS Area Ratios ---")
    area_ratios = {}
    for s in sorted(set(ebos_hulls.keys()) & set(phys_bos_areas.keys())):
        ebos_a = polygon_area(ebos_hulls[s])
        phys_a = phys_bos_areas[s]
        ratio = ebos_a / phys_a if phys_a > 0 else np.nan
        area_ratios[s] = ratio
        print(f"    {s}: eBOS={ebos_a:.6f} m^2, phys={phys_a:.6f} m^2, ratio={ratio:.3f}")
    if area_ratios:
        print(f"  Mean eBOS/BOS ratio: {np.mean(list(area_ratios.values())):.3f}")

    # Contingency: xCOM inside eBOS vs step_TF
    valid["xCOM_inside_eBOS"] = valid["eBOS_MoS"] > 0
    valid_ct = valid.dropna(subset=["eBOS_MoS"])
    ct = pd.crosstab(valid_ct["xCOM_inside_eBOS"], valid_ct["step_TF"])
    ct.index = ct.index.map({True: "inside_eBOS", False: "outside_eBOS"})
    print(f"\n  --- Contingency: xCOM inside eBOS vs step_TF ---")
    print(ct)
    if ct.shape == (2, 2):
        _, fisher_p = sp_stats.fisher_exact(ct.values)
        print(f"  Fisher exact p = {fisher_p:.4f}")

    # GLMM: eBOS-MoS
    print("\n  --- GLMM: step_TF ~ eBOS_MoS + (1|subject) ---")
    glmm_ebos = fit_glmm(valid, "step_binary ~ eBOS_MoS")
    if glmm_ebos.get("converged"):
        for i, name in enumerate(glmm_ebos["coef_names"]):
            print(
                f"    {name}: coef={glmm_ebos['coefs'][i]:.4f}, "
                f"OR={glmm_ebos['or'][i]:.4f} "
                f"[{glmm_ebos['or_ci_lo'][i]:.4f}, {glmm_ebos['or_ci_hi'][i]:.4f}], "
                f"p={glmm_ebos['p'][i]:.4e}"
            )
    else:
        print(f"    GLMM failed: {glmm_ebos.get('error', 'unknown')}")

    # GLMM: physical MoS
    print("\n  --- GLMM: step_TF ~ MOS_minDist_signed + (1|subject) ---")
    glmm_phys = fit_glmm(valid, "step_binary ~ MOS_minDist_signed")
    if glmm_phys.get("converged"):
        for i, name in enumerate(glmm_phys["coef_names"]):
            print(
                f"    {name}: coef={glmm_phys['coefs'][i]:.4f}, "
                f"OR={glmm_phys['or'][i]:.4f} "
                f"[{glmm_phys['or_ci_lo'][i]:.4f}, {glmm_phys['or_ci_hi'][i]:.4f}], "
                f"p={glmm_phys['p'][i]:.4e}"
            )

    # Mann-Whitney (auxiliary)
    step_ebos = valid.loc[valid["step_TF"] == "step", "eBOS_MoS"].dropna()
    nonstep_ebos = valid.loc[valid["step_TF"] == "nonstep", "eBOS_MoS"].dropna()
    if len(step_ebos) >= 2 and len(nonstep_ebos) >= 2:
        U, p = sp_stats.mannwhitneyu(step_ebos, nonstep_ebos, alternative="two-sided")
        rbc = 1 - (2 * U) / (len(step_ebos) * len(nonstep_ebos))
        print(f"\n  Mann-Whitney (eBOS-MoS): U={U:.0f}, p={p:.2e}, r={rbc:.3f}")

    # Figures 1-4
    print("\n  Generating figures 1-4...")
    fig1_cop_excursion_envelope(
        cop_ts, ebos_hulls, trials,
        Path(args.c3d_dir), Path(args.platform_xlsm), out_dir, dpi,
    )
    print("    fig1_cop_excursion_envelope.png")

    fig2_ebos_area_ratio(ebos_hulls, phys_bos_areas, out_dir, dpi)
    print("    fig2_ebos_area_ratio.png")

    fig3_mos_ebos_distribution(trials, out_dir, dpi)
    print("    fig3_mos_ebos_distribution.png")

    roc_data = fig4_roc_ebos_vs_physical(trials, out_dir, dpi)
    print("    fig4_roc_ebos_vs_physical.png")
    for k, v in roc_data.items():
        print(f"      {k} AUC = {v:.3f}")

    # ── Milestone 3: FSR State Space ───────────────────────────────────
    print("\n[Milestone 3] FSR State-Space analysis...")

    # 3a: Normalize
    trials = compute_fsr_features(trials, leg_lengths)
    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].copy()
    valid["step_binary"] = (valid["step_TF"] == "step").astype(int)

    fsr_valid = valid.dropna(subset=["COM_pos_norm", "COM_vel_norm"])
    print(f"  FSR features computed: {len(fsr_valid)} valid trials")
    print(f"    COM_pos_norm: mean={fsr_valid['COM_pos_norm'].mean():.3f}, "
          f"std={fsr_valid['COM_pos_norm'].std():.3f}")
    print(f"    COM_vel_norm: mean={fsr_valid['COM_vel_norm'].mean():.3f}, "
          f"std={fsr_valid['COM_vel_norm'].std():.3f}")

    # 3c: GLMM 2D
    print("\n  --- GLMM: step_TF ~ COM_pos_norm + COM_vel_norm + (1|subject) ---")
    glmm_2d = fit_glmm(valid, "step_binary ~ COM_pos_norm + COM_vel_norm")
    if glmm_2d.get("converged"):
        for i, name in enumerate(glmm_2d["coef_names"]):
            print(
                f"    {name}: coef={glmm_2d['coefs'][i]:.4f}, "
                f"OR={glmm_2d['or'][i]:.4f} "
                f"[{glmm_2d['or_ci_lo'][i]:.4f}, {glmm_2d['or_ci_hi'][i]:.4f}], "
                f"p={glmm_2d['p'][i]:.4e}"
            )

    # GLMM 1D: velocity only
    print("\n  --- GLMM: step_TF ~ COM_vel_norm + (1|subject) ---")
    glmm_vel = fit_glmm(valid, "step_binary ~ COM_vel_norm")
    if glmm_vel.get("converged"):
        for i, name in enumerate(glmm_vel["coef_names"]):
            print(
                f"    {name}: coef={glmm_vel['coefs'][i]:.4f}, "
                f"OR={glmm_vel['or'][i]:.4f}, p={glmm_vel['p'][i]:.4e}"
            )

    # GLMM 1D: position only
    print("\n  --- GLMM: step_TF ~ COM_pos_norm + (1|subject) ---")
    glmm_pos = fit_glmm(valid, "step_binary ~ COM_pos_norm")
    if glmm_pos.get("converged"):
        for i, name in enumerate(glmm_pos["coef_names"]):
            print(
                f"    {name}: coef={glmm_pos['coefs'][i]:.4f}, "
                f"OR={glmm_pos['or'][i]:.4f}, p={glmm_pos['p'][i]:.4e}"
            )

    # LOSO-CV AUC for all models
    print("\n  --- LOSO-CV AUC comparison ---")
    roc_results = {}

    roc_2d = compute_loso_cv_auc(
        valid, ["COM_pos_norm", "COM_vel_norm"], "step_binary", "subject"
    )
    roc_results["2D (pos+vel)"] = roc_2d
    print(f"  2D (pos+vel): AUC={roc_2d['overall_auc']:.3f} "
          f"(mean={roc_2d['mean_auc']:.3f}, "
          f"CI=[{roc_2d['ci95_lo']:.3f}, {roc_2d['ci95_hi']:.3f}])")

    roc_mos = compute_loso_cv_auc(
        valid, ["MOS_minDist_signed"], "step_binary", "subject"
    )
    roc_results["1D MoS"] = roc_mos
    print(f"  1D MoS:       AUC={roc_mos['overall_auc']:.3f} "
          f"(mean={roc_mos['mean_auc']:.3f})")

    roc_vel = compute_loso_cv_auc(
        valid, ["COM_vel_norm"], "step_binary", "subject"
    )
    roc_results["1D velocity"] = roc_vel
    print(f"  1D velocity:  AUC={roc_vel['overall_auc']:.3f} "
          f"(mean={roc_vel['mean_auc']:.3f})")

    roc_pos = compute_loso_cv_auc(
        valid, ["COM_pos_norm"], "step_binary", "subject"
    )
    roc_results["1D position"] = roc_pos
    print(f"  1D position:  AUC={roc_pos['overall_auc']:.3f} "
          f"(mean={roc_pos['mean_auc']:.3f})")

    # eBOS-MoS LOSO-CV
    roc_ebos_cv = compute_loso_cv_auc(
        valid, ["eBOS_MoS"], "step_binary", "subject"
    )
    roc_results["eBOS MoS"] = roc_ebos_cv
    print(f"  eBOS MoS:     AUC={roc_ebos_cv['overall_auc']:.3f} "
          f"(mean={roc_ebos_cv['mean_auc']:.3f})")

    # Figures 5-8
    print("\n  Generating figures 5-8...")
    fig5_state_space_scatter(trials, glmm_2d, out_dir, dpi)
    print("    fig5_state_space_scatter.png")

    fig6_state_space_marginals(trials, out_dir, dpi)
    print("    fig6_state_space_marginals.png")

    fig7_roc_2d_vs_1d(roc_results, out_dir, dpi)
    print("    fig7_roc_2d_vs_1d.png")

    fig8_summary_auc_comparison(roc_results, out_dir, dpi)
    print("    fig8_summary_auc_comparison.png")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nAnalysis 1 — Effective BOS:")
    if area_ratios:
        print(f"  Mean eBOS/BOS area ratio: {np.mean(list(area_ratios.values())):.3f}")
    print(f"  eBOS-MoS AUC (simple):   {roc_data.get('eBOS_MoS', np.nan):.3f}")
    print(f"  Physical MoS AUC (simple): {roc_data.get('MOS_minDist_signed', np.nan):.3f}")

    step_trials = valid[valid["step_TF"] == "step"]
    n_outside_ebos = (step_trials["eBOS_MoS"].dropna() < 0).sum()
    n_outside_phys = (step_trials["MOS_minDist_signed"] < 0).sum()
    n_step = len(step_trials)
    print(f"  Step outside eBOS: {n_outside_ebos}/{n_step} ({100*n_outside_ebos/n_step:.1f}%)")
    print(f"  Step outside phys: {n_outside_phys}/{n_step} ({100*n_outside_phys/n_step:.1f}%)")

    print(f"\nAnalysis 2 — FSR State Space:")
    print(f"  2D (pos+vel) LOSO AUC: {roc_2d['overall_auc']:.3f}")
    print(f"  1D MoS LOSO AUC:       {roc_mos['overall_auc']:.3f}")
    print(f"  1D velocity LOSO AUC:  {roc_vel['overall_auc']:.3f}")
    print(f"  1D position LOSO AUC:  {roc_pos['overall_auc']:.3f}")

    if glmm_2d.get("converged") and len(glmm_2d["coefs"]) == 3:
        vel_coef = abs(glmm_2d["coefs"][2])
        pos_coef = abs(glmm_2d["coefs"][1])
        dom = "velocity" if vel_coef > pos_coef else "position"
        print(f"  Dominant predictor: {dom} (|coef_vel|={vel_coef:.3f}, |coef_pos|={pos_coef:.3f})")

    print(f"\nOutput directory: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
