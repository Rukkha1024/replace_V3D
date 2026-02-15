"""Grid-plot visualisation of biomechanical time-series.

Usage
-----
# Sample mode (1 trial, for layout confirmation)
conda run -n module python scripts/plot_grid_timeseries.py --sample

# All-trials overlay (spaghetti plot)
conda run -n module python scripts/plot_grid_timeseries.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap

_bootstrap.ensure_src_on_path()

import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Korean font fallback
# ---------------------------------------------------------------------------
try:
    import matplotlib.font_manager as fm

    _nanum = [f.name for f in fm.fontManager.ttflist if "NanumGothic" in f.name]
    if _nanum:
        plt.rcParams["font.family"] = _nanum[0]
        plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

REPO_ROOT = _bootstrap.REPO_ROOT
DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "figures"

# ---------------------------------------------------------------------------
# Category specifications
# ---------------------------------------------------------------------------
# Each spec: tag, title, nrows, ncols, figsize, subplots
# subplot: (row, col, column_name, label)  OR  (row, col, [(col, label, style), ...], ylabel)
# The second form overlays multiple lines on one axes (e.g. L/R).

CATEGORIES: list[dict] = [
    # 1. MOS & BOS
    {
        "tag": "mos_bos",
        "title": "MOS & BOS",
        "nrows": 2,
        "ncols": 2,
        "figsize": (10, 8),
        "subplots": [
            (0, 0, "MOS_minDist_signed", "MOS minDist (signed)"),
            (0, 1, "MOS_AP_dir", "MOS AP"),
            (1, 0, "MOS_ML_dir", "MOS ML"),
            (1, 1, "BOS_area", "BOS area"),
        ],
    },
    # 2. COM / vCOM / xCOM
    {
        "tag": "com",
        "title": "COM / vCOM / xCOM",
        "nrows": 3,
        "ncols": 3,
        "figsize": (14, 10),
        "subplots": [
            (0, 0, "COM_X", "COM X"),
            (0, 1, "COM_Y", "COM Y"),
            (0, 2, "COM_Z", "COM Z"),
            (1, 0, "vCOM_X", "vCOM X"),
            (1, 1, "vCOM_Y", "vCOM Y"),
            (1, 2, "vCOM_Z", "vCOM Z"),
            (2, 0, "xCOM_X", "xCOM X"),
            (2, 1, "xCOM_Y", "xCOM Y"),
            (2, 2, "xCOM_Z", "xCOM Z"),
        ],
    },
    # 3. Joint Angles – Lower (L/R overlay)
    {
        "tag": "joint_lower",
        "title": "Joint Angles - Lower Extremity",
        "nrows": 3,
        "ncols": 3,
        "figsize": (14, 10),
        "subplots": [
            (0, 0, [("Hip_L_X_deg", "L", "-"), ("Hip_R_X_deg", "R", "--")], "Hip X (deg)"),
            (0, 1, [("Hip_L_Y_deg", "L", "-"), ("Hip_R_Y_deg", "R", "--")], "Hip Y (deg)"),
            (0, 2, [("Hip_L_Z_deg", "L", "-"), ("Hip_R_Z_deg", "R", "--")], "Hip Z (deg)"),
            (1, 0, [("Knee_L_X_deg", "L", "-"), ("Knee_R_X_deg", "R", "--")], "Knee X (deg)"),
            (1, 1, [("Knee_L_Y_deg", "L", "-"), ("Knee_R_Y_deg", "R", "--")], "Knee Y (deg)"),
            (1, 2, [("Knee_L_Z_deg", "L", "-"), ("Knee_R_Z_deg", "R", "--")], "Knee Z (deg)"),
            (2, 0, [("Ankle_L_X_deg", "L", "-"), ("Ankle_R_X_deg", "R", "--")], "Ankle X (deg)"),
            (2, 1, [("Ankle_L_Y_deg", "L", "-"), ("Ankle_R_Y_deg", "R", "--")], "Ankle Y (deg)"),
            (2, 2, [("Ankle_L_Z_deg", "L", "-"), ("Ankle_R_Z_deg", "R", "--")], "Ankle Z (deg)"),
        ],
    },
    # 4. Joint Angles – Trunk & Neck
    {
        "tag": "joint_upper",
        "title": "Joint Angles - Trunk & Neck",
        "nrows": 2,
        "ncols": 3,
        "figsize": (14, 7),
        "subplots": [
            (0, 0, "Trunk_X_deg", "Trunk X (deg)"),
            (0, 1, "Trunk_Y_deg", "Trunk Y (deg)"),
            (0, 2, "Trunk_Z_deg", "Trunk Z (deg)"),
            (1, 0, "Neck_X_deg", "Neck X (deg)"),
            (1, 1, "Neck_Y_deg", "Neck Y (deg)"),
            (1, 2, "Neck_Z_deg", "Neck Z (deg)"),
        ],
    },
    # 5. GRF
    {
        "tag": "grf",
        "title": "Ground Reaction Force",
        "nrows": 1,
        "ncols": 3,
        "figsize": (14, 4),
        "subplots": [
            (0, 0, "GRF_X_N", "GRF X (N)"),
            (0, 1, "GRF_Y_N", "GRF Y (N)"),
            (0, 2, "GRF_Z_N", "GRF Z (N)"),
        ],
    },
    # 6. Ankle Torque (internal)
    {
        "tag": "ankle_torque_int",
        "title": "Ankle Torque (Internal)",
        "nrows": 3,
        "ncols": 3,
        "figsize": (14, 10),
        "subplots": [
            (0, 0, "AnkleTorqueMid_int_X_Nm", "Mid int X (Nm)"),
            (0, 1, "AnkleTorqueMid_int_Y_Nm", "Mid int Y (Nm)"),
            (0, 2, "AnkleTorqueMid_int_Z_Nm", "Mid int Z (Nm)"),
            (1, 0, "AnkleTorqueL_int_X_Nm", "L int X (Nm)"),
            (1, 1, "AnkleTorqueL_int_Y_Nm", "L int Y (Nm)"),
            (1, 2, "AnkleTorqueL_int_Z_Nm", "L int Z (Nm)"),
            (2, 0, "AnkleTorqueR_int_X_Nm", "R int X (Nm)"),
            (2, 1, "AnkleTorqueR_int_Y_Nm", "R int Y (Nm)"),
            (2, 2, "AnkleTorqueR_int_Z_Nm", "R int Z (Nm)"),
        ],
    },
    # 7. COP
    {
        "tag": "cop",
        "title": "Center of Pressure",
        "nrows": 1,
        "ncols": 3,
        "figsize": (14, 4),
        "subplots": [
            (0, 0, "COP_X_m", "COP X (m)"),
            (0, 1, "COP_Y_m", "COP Y (m)"),
            (0, 2, "COP_Z_m", "COP Z (m)"),
        ],
    },
]

TRIAL_KEYS = ["subject", "velocity", "trial"]
TIME_COL = "time_from_platform_onset_s"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-plot biomechanical timeseries")
    p.add_argument("--sample", action="store_true", help="Plot 1 trial only (layout check)")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    return p.parse_args()


def load_data(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path, encoding="utf8-lossy", infer_schema_length=10000)
    # Filter: MocapFrame >= 81
    df = df.filter(pl.col("MocapFrame") >= 81)

    # Compute step_onset time for each trial
    # step_onset_local is in MocapFrame units; find the time at that frame
    # We use the row where Is_step_onset_frame == 1 to get the exact time value
    step_times = (
        df.filter(pl.col("Is_step_onset_frame") == 1)
        .select(TRIAL_KEYS + [TIME_COL])
        .rename({TIME_COL: "step_onset_time_s"})
        .unique(subset=TRIAL_KEYS)
    )
    df = df.join(step_times, on=TRIAL_KEYS, how="left")
    return df


def get_sample_trial(df: pl.DataFrame) -> pl.DataFrame:
    """Pick the first trial that has a non-null step_onset."""
    candidates = (
        df.filter(pl.col("step_onset_time_s").is_not_null())
        .select(TRIAL_KEYS)
        .unique()
        .sort(TRIAL_KEYS)
    )
    if candidates.height == 0:
        # fallback: any trial
        candidates = df.select(TRIAL_KEYS).unique().sort(TRIAL_KEYS)
    first = candidates.row(0, named=True)
    return df.filter(
        (pl.col("subject") == first["subject"])
        & (pl.col("velocity") == first["velocity"])
        & (pl.col("trial") == first["trial"])
    )


def _draw_events(ax: plt.Axes, trial_df: pl.DataFrame, alpha: float = 1.0) -> None:
    """Draw platform_onset (red) and step_onset (blue dashed) vlines."""
    # platform_onset is always at time=0
    ax.axvline(0, color="red", linewidth=0.8, alpha=alpha, linestyle="-")
    # step_onset
    step_t = trial_df["step_onset_time_s"].drop_nulls().unique()
    if step_t.len() > 0:
        ax.axvline(
            step_t[0], color="blue", linewidth=0.8, alpha=alpha, linestyle="--"
        )


def plot_category(
    df: pl.DataFrame,
    spec: dict,
    out_dir: Path,
    sample: bool,
    dpi: int,
) -> Path:
    nrows, ncols = spec["nrows"], spec["ncols"]
    fig, axes = plt.subplots(nrows, ncols, figsize=spec["figsize"], squeeze=False)

    # Group by trial
    trials = df.select(TRIAL_KEYS).unique().sort(TRIAL_KEYS)

    if sample:
        trial_df = get_sample_trial(df)
        trial_list = [trial_df]
        info = trial_df.select(TRIAL_KEYS).unique().row(0, named=True)
        suptitle = f"{spec['title']}  —  {info['subject']} / v{info['velocity']} / trial {info['trial']}"
    else:
        # build list of per-trial dataframes
        trial_list = []
        for row in trials.iter_rows(named=True):
            tdf = df.filter(
                (pl.col("subject") == row["subject"])
                & (pl.col("velocity") == row["velocity"])
                & (pl.col("trial") == row["trial"])
            )
            trial_list.append(tdf)
        suptitle = f"{spec['title']}  ({trials.height} trials)"

    lw = 1.2 if sample else 0.5
    alpha_line = 1.0 if sample else 0.3
    alpha_event = 1.0 if sample else 0.2

    for sp in spec["subplots"]:
        r, c = sp[0], sp[1]
        ax = axes[r][c]
        col_spec = sp[2]  # str or list
        ylabel = sp[3]

        if isinstance(col_spec, str):
            # single variable
            for tdf in trial_list:
                pd_tdf = tdf.select([TIME_COL, col_spec]).to_pandas()
                ax.plot(
                    pd_tdf[TIME_COL],
                    pd_tdf[col_spec],
                    color="gray" if not sample else None,
                    linewidth=lw,
                    alpha=alpha_line,
                )
                _draw_events(ax, tdf, alpha=alpha_event)
        else:
            # L/R overlay list: [(col, label, linestyle), ...]
            colors_lr = {"L": "blue", "R": "red"}
            for tdf in trial_list:
                pd_tdf = tdf.to_pandas()
                for col_name, side_label, ls in col_spec:
                    if col_name not in pd_tdf.columns:
                        continue
                    color = colors_lr.get(side_label, "gray") if sample else "gray"
                    ax.plot(
                        pd_tdf[TIME_COL],
                        pd_tdf[col_name],
                        color=color,
                        linestyle=ls if sample else "-",
                        linewidth=lw,
                        alpha=alpha_line,
                        label=side_label if sample else None,
                    )
                _draw_events(ax, tdf, alpha=alpha_event)
            if sample:
                ax.legend(fontsize=8, loc="best")

        ax.set_title(ylabel, fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.5)

        # xlabel only on bottom row
        if r == nrows - 1:
            ax.set_xlabel("Time from platform onset (s)", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)

    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    suffix = "sample" if sample else "all"
    out_path = out_dir / f"{spec['tag']}_{suffix}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.csv} ...")
    df = load_data(args.csv)
    n_trials = df.select(TRIAL_KEYS).unique().height
    print(f"  {df.height} rows, {n_trials} trials (after MocapFrame>=81 filter)")

    mode = "sample" if args.sample else "all"
    print(f"Mode: {mode}")

    for spec in CATEGORIES:
        out_path = plot_category(df, spec, args.out_dir, args.sample, args.dpi)
        print(f"  Saved: {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
