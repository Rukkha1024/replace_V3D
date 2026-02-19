"""FSR-only stepping analysis.

Answers: "Can COM velocity-position state variables explain stepping better than
MoS baseline without any COP-based boundary assumptions?"

Produces:
  - 4 publication-quality figures (saved alongside this script)
  - stdout summary statistics

Usage:
    conda run -n module python analysis/why_stepping_before_threshold/analyze_fsr_only.py
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
import polars as pl
import pandas as pd
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

from replace_v3d.io.events_excel import load_subject_leg_length_cm

DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_PLATFORM_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_OUT_DIR = SCRIPT_DIR

TRIAL_KEYS = ["subject", "velocity", "trial"]
G = 9.81  # m/s^2


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--platform_xlsm", type=Path, default=DEFAULT_PLATFORM_XLSM)
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


def build_trial_summary(
    df: pl.DataFrame,
    platform: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-trial snapshot at reference timepoint.

    ref_frame:
    - step trial: step_onset_local
    - nonstep trial: mean step_onset_local of same (subject, velocity)
    """

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

    snapshot_cols = [
        "COM_X",
        "vCOM_X",
        "BOS_minX",
        "BOS_maxX",
        "MOS_minDist_signed",
    ]

    df_pd = df.select(TRIAL_KEYS + ["MocapFrame"] + snapshot_cols).to_pandas()
    snap_records = []

    for _, row in trials.iterrows():
        subj, vel, tri, ref = (
            row["subject"],
            row["velocity"],
            int(row["trial"]),
            int(row["ref_frame"]),
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
    result: dict[str, float] = {}
    for subj in subjects:
        val = load_subject_leg_length_cm(xlsm_path, subj)
        if val is not None:
            result[subj] = float(val) / 100.0
        else:
            warnings.warn(f"No leg length for subject '{subj}', using default 0.9 m")
            result[subj] = 0.9
    return result


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


def fit_glmm(
    data: pd.DataFrame,
    formula: str,
    groups: str = "subject",
) -> dict:
    """Fit Binomial GLMM (Bayesian) and return summary dict."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

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
    X_with_const = np.column_stack([np.ones(len(X)), X])
    col_names = ["Intercept"] + x_cols

    group_labels = clean[groups].values
    unique_groups = np.unique(group_labels)
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
    """LOSO-CV AUC using sklearn LogisticRegression as proxy model."""
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
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)

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


def _step_nonstep_colors() -> dict[str, str]:
    return {"step": "#E74C3C", "nonstep": "#3498DB"}


def fig1_state_space_scatter(
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
            sub["COM_pos_norm"],
            sub["COM_vel_norm"],
            c=color,
            label=grp,
            alpha=0.65,
            s=40,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.plot(
        [0, 1],
        [1.0, 0],
        color="gray",
        linestyle=":",
        linewidth=2,
        alpha=0.5,
        label="Approx FSR boundary",
    )

    if glmm_result and glmm_result.get("converged"):
        coefs = glmm_result["coefs"]
        if len(coefs) == 3 and abs(coefs[2]) > 1e-6:
            x_range = np.linspace(
                valid["COM_pos_norm"].min() - 0.1,
                valid["COM_pos_norm"].max() + 0.1,
                100,
            )
            y_boundary = -(coefs[0] + coefs[1] * x_range) / coefs[2]
            ax.plot(x_range, y_boundary, color="black", linewidth=2, linestyle="-", label="GLMM P(step)=0.5")

    ax.set_xlabel("COM position (normalized to BOS, 0=post, 1=ant)")
    ax.set_ylabel("COM velocity (normalized, dimensionless)")
    ax.set_title("COM Velocity-Position State Space (FSR)", fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        out_dir / "fig1_state_space_scatter.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def fig2_state_space_marginals(
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
        data=valid,
        x="COM_pos_norm",
        y="COM_vel_norm",
        hue="step_TF",
        hue_order=["step", "nonstep"],
        palette=colors,
        height=7,
    )
    g.plot_joint(sns.scatterplot, alpha=0.6, s=40, edgecolor="white", linewidth=0.3)
    g.plot_marginals(sns.histplot, kde=True, alpha=0.4, common_norm=False)

    g.ax_joint.set_xlabel("COM position (norm)", fontsize=10)
    g.ax_joint.set_ylabel("COM velocity (norm)", fontsize=10)
    g.figure.suptitle(
        "State Space with Marginal Distributions",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    g.figure.tight_layout()
    g.savefig(out_dir / "fig2_state_space_marginals.png", dpi=dpi, bbox_inches="tight")
    plt.close(g.figure)


def fig3_roc_2d_vs_1d(
    roc_results: dict[str, dict],
    out_dir: Path,
    dpi: int,
) -> None:
    """ROC curves: 2D state-space vs 1D baselines."""
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
            ax.plot(
                rdata["fpr"],
                rdata["tpr"],
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=f"{label} (AUC={auc:.3f})",
            )

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
        out_dir / "fig3_roc_2d_vs_1d.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def fig4_summary_auc_comparison(
    auc_summary: dict[str, dict],
    out_dir: Path,
    dpi: int,
) -> None:
    """Summary AUC bar chart with 95% CI error bars."""
    labels = list(auc_summary.keys())
    aucs = [auc_summary[k]["overall_auc"] for k in labels]
    ci_lo = [auc_summary[k].get("ci95_lo", np.nan) for k in labels]
    ci_hi = [auc_summary[k].get("ci95_hi", np.nan) for k in labels]

    yerr_lo = [max(0, a - lo) if np.isfinite(lo) else 0 for a, lo in zip(aucs, ci_lo)]
    yerr_hi = [max(0, hi - a) if np.isfinite(hi) else 0 for a, hi in zip(aucs, ci_hi)]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bar_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

    bars = ax.bar(
        range(len(labels)),
        aucs,
        yerr=[yerr_lo, yerr_hi],
        capsize=5,
        color=bar_colors[: len(labels)],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, auc_val in zip(bars, aucs):
        if np.isfinite(auc_val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{auc_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
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
        out_dir / "fig4_summary_auc_comparison.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    print("=" * 70)
    print("FSR-only stepping analysis")
    print("COP-based boundaries excluded by coordinate-system constraints")
    print("=" * 70)

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

    print("\n[Milestone 2] FSR feature engineering...")
    trials = compute_fsr_features(trials, leg_lengths)
    valid = trials[trials["step_TF"].isin(["step", "nonstep"])].copy()
    valid["step_binary"] = (valid["step_TF"] == "step").astype(int)

    fsr_valid = valid.dropna(subset=["COM_pos_norm", "COM_vel_norm"])
    print(f"  FSR features computed: {len(fsr_valid)} valid trials")
    print(
        f"    COM_pos_norm: mean={fsr_valid['COM_pos_norm'].mean():.3f}, "
        f"std={fsr_valid['COM_pos_norm'].std():.3f}"
    )
    print(
        f"    COM_vel_norm: mean={fsr_valid['COM_vel_norm'].mean():.3f}, "
        f"std={fsr_valid['COM_vel_norm'].std():.3f}"
    )

    print("\n[Milestone 3] GLMM models...")
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

    print("\n  --- GLMM: step_TF ~ COM_vel_norm + (1|subject) ---")
    glmm_vel = fit_glmm(valid, "step_binary ~ COM_vel_norm")
    if glmm_vel.get("converged"):
        for i, name in enumerate(glmm_vel["coef_names"]):
            print(
                f"    {name}: coef={glmm_vel['coefs'][i]:.4f}, "
                f"OR={glmm_vel['or'][i]:.4f}, p={glmm_vel['p'][i]:.4e}"
            )

    print("\n  --- GLMM: step_TF ~ COM_pos_norm + (1|subject) ---")
    glmm_pos = fit_glmm(valid, "step_binary ~ COM_pos_norm")
    if glmm_pos.get("converged"):
        for i, name in enumerate(glmm_pos["coef_names"]):
            print(
                f"    {name}: coef={glmm_pos['coefs'][i]:.4f}, "
                f"OR={glmm_pos['or'][i]:.4f}, p={glmm_pos['p'][i]:.4e}"
            )

    print("\n  --- GLMM: step_TF ~ MOS_minDist_signed + (1|subject) ---")
    glmm_mos = fit_glmm(valid, "step_binary ~ MOS_minDist_signed")
    if glmm_mos.get("converged"):
        for i, name in enumerate(glmm_mos["coef_names"]):
            print(
                f"    {name}: coef={glmm_mos['coefs'][i]:.4f}, "
                f"OR={glmm_mos['or'][i]:.4f}, p={glmm_mos['p'][i]:.4e}"
            )

    print("\n[Milestone 4] LOSO-CV AUC comparison...")
    roc_results: dict[str, dict] = {}

    roc_2d = compute_loso_cv_auc(valid, ["COM_pos_norm", "COM_vel_norm"], "step_binary", "subject")
    roc_results["2D (pos+vel)"] = roc_2d
    print(
        f"  2D (pos+vel): AUC={roc_2d['overall_auc']:.3f} "
        f"(mean={roc_2d['mean_auc']:.3f}, CI=[{roc_2d['ci95_lo']:.3f}, {roc_2d['ci95_hi']:.3f}])"
    )

    roc_mos = compute_loso_cv_auc(valid, ["MOS_minDist_signed"], "step_binary", "subject")
    roc_results["1D MoS"] = roc_mos
    print(f"  1D MoS:       AUC={roc_mos['overall_auc']:.3f} (mean={roc_mos['mean_auc']:.3f})")

    roc_vel = compute_loso_cv_auc(valid, ["COM_vel_norm"], "step_binary", "subject")
    roc_results["1D velocity"] = roc_vel
    print(f"  1D velocity:  AUC={roc_vel['overall_auc']:.3f} (mean={roc_vel['mean_auc']:.3f})")

    roc_pos = compute_loso_cv_auc(valid, ["COM_pos_norm"], "step_binary", "subject")
    roc_results["1D position"] = roc_pos
    print(f"  1D position:  AUC={roc_pos['overall_auc']:.3f} (mean={roc_pos['mean_auc']:.3f})")

    print("\n[Milestone 5] Generating figures 1-4...")
    fig1_state_space_scatter(trials, glmm_2d, out_dir, dpi)
    print("    fig1_state_space_scatter.png")

    fig2_state_space_marginals(trials, out_dir, dpi)
    print("    fig2_state_space_marginals.png")

    fig3_roc_2d_vs_1d(roc_results, out_dir, dpi)
    print("    fig3_roc_2d_vs_1d.png")

    fig4_summary_auc_comparison(roc_results, out_dir, dpi)
    print("    fig4_summary_auc_comparison.png")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("  COP-based boundary analyses were excluded by design constraints.")
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
