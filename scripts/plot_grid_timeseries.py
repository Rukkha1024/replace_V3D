"""Grid-plot visualisation of biomechanical time-series.

Usage
-----
# Sample mode (subject-wise preview)
conda run -n module python scripts/plot_grid_timeseries.py --sample

# All subjects
conda run -n module python scripts/plot_grid_timeseries.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import _bootstrap

_bootstrap.ensure_src_on_path()

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = _bootstrap.REPO_ROOT
DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "figures" / "grid_timeseries"
TRIAL_KEYS = ["subject", "velocity", "trial"]
TIME_COL = "time_from_platform_onset_s"

CATEGORIES: list[dict] = [
    {
        "tag": "mos_bos",
        "title": "MOS / BOS",
        "nrows": 2,
        "ncols": 2,
        "figsize": (11, 8),
        "subplots": [
            (0, 0, "MOS_minDist_signed", "MOS minDist (signed)"),
            (0, 1, "MOS_AP_dir", "MOS AP"),
            (1, 0, "MOS_ML_dir", "MOS ML"),
            (1, 1, "BOS_area", "BOS area"),
        ],
    },
    {
        "tag": "com_family",
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
    {
        "tag": "joint_angles_lower",
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
    {
        "tag": "joint_angles_upper",
        "title": "Joint Angles - Trunk / Neck",
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
    {
        "tag": "ankle_torque_internal",
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
    {
        "tag": "grf_cop",
        "title": "GRF / COP",
        "nrows": 2,
        "ncols": 3,
        "figsize": (14, 7),
        "subplots": [
            (0, 0, "GRF_X_N", "GRF X (N)"),
            (0, 1, "GRF_Y_N", "GRF Y (N)"),
            (0, 2, "GRF_Z_N", "GRF Z (N)"),
            (1, 0, "COP_X_m", "COP X (m)"),
            (1, 1, "COP_Y_m", "COP Y (m)"),
            (1, 2, "COP_Z_m", "COP Z (m)"),
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-plot biomechanical timeseries")
    parser.add_argument("--sample", action="store_true", help="Generate preview only")
    parser.add_argument(
        "--sample_subjects",
        type=int,
        default=3,
        help="Number of subjects to render in sample mode",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument(
        "--resample_hz",
        type=float,
        default=100.0,
        help="Common x-axis resampling rate (Hz)",
    )
    parser.add_argument(
        "--xtick_sec",
        type=float,
        default=0.2,
        help="Common x-axis tick spacing (sec)",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", str(text))
    value = re.sub(r"\s+", "_", value).strip("_")
    return value if value else "unknown"


def load_data(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path, encoding="utf8-lossy", infer_schema_length=10000)
    required = TRIAL_KEYS + ["MocapFrame", "platform_onset_local", "Is_step_onset_frame", TIME_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Requested data range: [platform_onset - 20, end].
    df = df.filter(pl.col("MocapFrame") >= (pl.col("platform_onset_local") - 20))

    step_times = (
        df.filter(pl.col("Is_step_onset_frame") == 1)
        .group_by(TRIAL_KEYS)
        .agg(pl.col(TIME_COL).first().alias("step_onset_time_s"))
    )
    return df.join(step_times, on=TRIAL_KEYS, how="left")


def build_common_x_grid(df: pl.DataFrame, resample_hz: float) -> np.ndarray:
    if resample_hz <= 0:
        raise ValueError("--resample_hz must be > 0")
    time_col = df.get_column(TIME_COL)
    t_min = float(time_col.min())
    t_max = float(time_col.max())
    step = 1.0 / resample_hz
    grid_min = np.floor(t_min / step) * step
    grid_max = np.ceil(t_max / step) * step
    return np.arange(grid_min, grid_max + (step * 0.5), step, dtype=float)


def build_common_xticks(x_grid: np.ndarray, xtick_sec: float) -> np.ndarray:
    if xtick_sec <= 0:
        raise ValueError("--xtick_sec must be > 0")
    x_min = float(x_grid[0])
    x_max = float(x_grid[-1])
    tick_start = np.floor(x_min / xtick_sec) * xtick_sec
    tick_end = np.ceil(x_max / xtick_sec) * xtick_sec
    ticks = np.arange(tick_start, tick_end + (xtick_sec * 0.5), xtick_sec, dtype=float)
    return np.round(ticks, 6)


def resample_trial_column(trial_df: pl.DataFrame, col_name: str, x_grid: np.ndarray) -> np.ndarray:
    if col_name not in trial_df.columns:
        return np.full(x_grid.shape, np.nan, dtype=float)

    x_raw = np.asarray(trial_df.get_column(TIME_COL).to_list(), dtype=float)
    y_raw = np.asarray(trial_df.get_column(col_name).to_list(), dtype=float)
    valid = np.isfinite(x_raw) & np.isfinite(y_raw)
    if np.count_nonzero(valid) == 0:
        return np.full(x_grid.shape, np.nan, dtype=float)

    x = x_raw[valid]
    y = y_raw[valid]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    # np.interp requires strictly increasing x; keep first sample for duplicates.
    keep = np.concatenate(([True], np.diff(x) > 0.0))
    x = x[keep]
    y = y[keep]
    if x.size == 1:
        out = np.full(x_grid.shape, np.nan, dtype=float)
        out[int(np.argmin(np.abs(x_grid - x[0])))] = y[0]
        return out
    return np.interp(x_grid, x, y, left=np.nan, right=np.nan)


def draw_events(ax: plt.Axes, trial_df: pl.DataFrame, alpha: float, label_once: bool) -> None:
    ax.axvline(
        0.0,
        color="red",
        linewidth=0.9,
        alpha=alpha,
        linestyle="-",
        label="platform_onset" if label_once else None,
    )
    step_times = trial_df.get_column("step_onset_time_s").drop_nulls().unique()
    if step_times.len() > 0:
        ax.axvline(
            float(step_times[0]),
            color="blue",
            linewidth=0.9,
            alpha=alpha,
            linestyle="--",
            label="step_onset" if label_once else None,
        )


def plot_single_col(
    ax: plt.Axes,
    trials: list[pl.DataFrame],
    col_name: str,
    sample: bool,
    x_grid: np.ndarray,
) -> None:
    if not trials or col_name not in trials[0].columns:
        ax.text(0.5, 0.5, f"Missing: {col_name}", transform=ax.transAxes, ha="center", va="center")
        return

    line_width = 1.2 if sample else 0.6
    line_alpha = 0.95 if sample else 0.30
    event_alpha = 1.0 if sample else 0.20

    for idx, trial_df in enumerate(trials):
        y_vals = resample_trial_column(trial_df, col_name, x_grid)
        ax.plot(
            x_grid,
            y_vals,
            color="gray",
            linewidth=line_width,
            alpha=line_alpha,
            label="trial lines" if idx == 0 else None,
        )
        draw_events(ax, trial_df, alpha=event_alpha, label_once=(idx == 0))


def plot_lr_overlay(
    ax: plt.Axes,
    trials: list[pl.DataFrame],
    col_specs: list[tuple[str, str, str]],
    sample: bool,
    x_grid: np.ndarray,
) -> None:
    if not trials:
        return

    line_width = 1.2 if sample else 0.6
    line_alpha = 0.90 if sample else 0.30
    event_alpha = 1.0 if sample else 0.20
    side_color = {"L": "tab:blue", "R": "tab:orange"}

    for trial_idx, trial_df in enumerate(trials):
        trial_cols = set(trial_df.columns)
        for col_name, side, style in col_specs:
            if col_name not in trial_cols:
                continue
            y_vals = resample_trial_column(trial_df, col_name, x_grid)
            ax.plot(
                x_grid,
                y_vals,
                color=side_color.get(side, "gray"),
                linestyle=style if sample else "-",
                linewidth=line_width,
                alpha=line_alpha,
                label=side if trial_idx == 0 else None,
            )
        draw_events(ax, trial_df, alpha=event_alpha, label_once=(trial_idx == 0))


def build_trial_list(subject_df: pl.DataFrame) -> list[pl.DataFrame]:
    trials = subject_df.select(TRIAL_KEYS).unique().sort(["velocity", "trial"])
    return [
        subject_df.filter(
            (pl.col("subject") == row["subject"])
            & (pl.col("velocity") == row["velocity"])
            & (pl.col("trial") == row["trial"])
        )
        for row in trials.iter_rows(named=True)
    ]


def plot_subject_category(
    subject_df: pl.DataFrame,
    subject_value: str,
    spec: dict,
    out_dir: Path,
    sample: bool,
    dpi: int,
    x_grid: np.ndarray,
    x_ticks: np.ndarray,
) -> Path:
    nrows, ncols = spec["nrows"], spec["ncols"]
    fig, axes = plt.subplots(nrows, ncols, figsize=spec["figsize"], squeeze=False)

    trials = build_trial_list(subject_df)
    trial_count = len(trials)

    for subplot in spec["subplots"]:
        r, c = subplot[0], subplot[1]
        col_spec = subplot[2]
        ylabel = subplot[3]
        ax = axes[r][c]

        if isinstance(col_spec, str):
            plot_single_col(ax, trials, col_spec, sample, x_grid)
        else:
            plot_lr_overlay(ax, trials, col_spec, sample, x_grid)

        ax.set_title(ylabel, fontsize=9)
        ax.grid(True, linewidth=0.35, alpha=0.5)
        ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
        ax.set_xticks(x_ticks)
        ax.margins(x=0.0)
        if r == nrows - 1:
            ax.set_xlabel("Time from platform onset (s)", fontsize=8)
        else:
            ax.tick_params(labelbottom=False)
        ax.legend(loc="best", fontsize=7, frameon=True)

    total_axes = nrows * ncols
    used_axes = len(spec["subplots"])
    for idx in range(used_axes, total_axes):
        rr = idx // ncols
        cc = idx % ncols
        axes[rr][cc].axis("off")

    mode_label = "sample" if sample else "all"
    fig.suptitle(
        f"{spec['title']} | subject overlay ({trial_count} subject-velocity-trial lines) | {mode_label}",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_name = f"{spec['tag']}__subject-{safe_name(subject_value)}__{mode_label}.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {args.csv}")
    df = load_data(args.csv)
    x_grid = build_common_x_grid(df, args.resample_hz)
    x_ticks = build_common_xticks(x_grid, args.xtick_sec)
    trial_count = df.select(TRIAL_KEYS).unique().height
    subjects = df.select("subject").unique().sort("subject").get_column("subject").to_list()

    if args.sample:
        subjects = subjects[: max(1, args.sample_subjects)]

    print(f"Rows: {df.height}, Trials: {trial_count}, Subjects to render: {len(subjects)}")
    print(f"Mode: {'sample' if args.sample else 'all'}")
    print(
        "Common x-axis range: "
        f"[{x_grid[0]:.3f}, {x_grid[-1]:.3f}] sec "
        f"({x_grid.size} points @ {args.resample_hz:g} Hz)"
    )
    print(f"Common x-axis ticks: every {args.xtick_sec:g} sec ({x_ticks.size} ticks)")

    for subject_value in subjects:
        subject_df = df.filter(pl.col("subject") == subject_value)
        for spec in CATEGORIES:
            out_path = plot_subject_category(
                subject_df=subject_df,
                subject_value=str(subject_value),
                spec=spec,
                out_dir=args.out_dir,
                sample=args.sample,
                dpi=args.dpi,
                x_grid=x_grid,
                x_ticks=x_ticks,
            )
            print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
