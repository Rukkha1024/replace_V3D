"""Grid-plot visualisation of biomechanical time-series.

Usage
-----
# Sample mode (subject×velocity-wise preview; recommended for sanity checks)
conda run -n module python scripts/plot_grid_timeseries.py --sample

# All subject×velocity groups (default)
conda run -n module python scripts/plot_grid_timeseries.py

# Legacy: subject-wise overlay (all velocities together)
conda run -n module python scripts/plot_grid_timeseries.py --group_by subject

# Raw seconds x-axis (disable piecewise normalization)
conda run -n module python scripts/plot_grid_timeseries.py --no-x_piecewise
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
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = _bootstrap.REPO_ROOT
DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "figures" / "grid_timeseries"
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
TRIAL_KEYS = ["subject", "velocity", "trial"]
TIME_COL = "time_from_platform_onset_s"
DEFAULT_SEGMENT_FRAMES = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-plot biomechanical timeseries")
    parser.add_argument("--sample", action="store_true", help="Generate preview only")
    parser.add_argument(
        "--sample_subjects",
        type=int,
        default=3,
        help="Number of subjects to render in sample mode",
    )
    parser.add_argument(
        "--sample_velocities",
        type=int,
        default=2,
        help="Number of velocities per subject in sample mode (used with --group_by subject_velocity)",
    )
    parser.add_argument(
        "--group_by",
        choices=["subject_velocity", "subject"],
        default="subject_velocity",
        help="Render one figure per grouping unit (default: subject_velocity)",
    )
    parser.add_argument(
        "--only_subjects",
        type=str,
        default=None,
        help="Comma-separated subject whitelist (e.g. 김우연,가윤호). Default: all",
    )
    parser.add_argument(
        "--only_velocities",
        type=str,
        default=None,
        help="Comma-separated velocity whitelist (e.g. 60,70). Default: all",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML config path (reads plot_grid_timeseries.*)",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output directory (overrides config.yaml if provided)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument(
        "--resample_hz",
        type=float,
        default=100.0,
        help="Common x-axis resampling rate (Hz)",
    )
    parser.add_argument(
        "--segment_frames",
        type=int,
        default=DEFAULT_SEGMENT_FRAMES,
        help="Plot window frames: [platform_onset-frames, platform_offset+frames] (default: 100).",
    )
    parser.add_argument(
        "--xtick_sec",
        type=float,
        default=0.2,
        help="Common x-axis tick spacing (sec)",
    )
    parser.add_argument(
        "--x_piecewise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Piecewise-normalize the displayed x-axis (default: enabled). "
            "Segments: [onset-frames, onset] (raw), [onset, offset] (normalized), "
            "[offset, offset+frames] (raw). Disable with --no-x_piecewise."
        ),
    )
    parser.add_argument(
        "--x_norm01",
        action="store_true",
        help="Normalize displayed x-axis to 0-1 per trial",
    )
    parser.add_argument(
        "--xtick_norm",
        type=float,
        default=0.1,
        help="Tick spacing for normalized x-axis (used with --x_norm01)",
    )
    parser.add_argument(
        "--separate_step_nonstep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate separate figures for step/nonstep trials (default: enabled)",
    )
    return parser.parse_args()


def safe_name(text: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", str(text))
    value = re.sub(r"\s+", "_", value).strip("_")
    return value if value else "unknown"


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_csv_list(text: str | None) -> list[str] | None:
    if text is None:
        return None
    items = [chunk.strip() for chunk in str(text).split(",")]
    items = [item for item in items if item]
    return items if items else None


def parse_float_list(text: str | None) -> list[float] | None:
    items = parse_csv_list(text)
    if not items:
        return None
    out: list[float] = []
    for raw in items:
        try:
            out.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"Invalid float in list: {raw!r}") from exc
    return out


def format_velocity(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return str(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    text = f"{v:.6f}".rstrip("0").rstrip(".")
    return text if text else str(v)


def load_plot_specs(config_path: Path) -> tuple[Path | None, list[dict]]:
    config_path = resolve_repo_path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}. Expected a YAML file with plot_grid_timeseries.*"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must be a mapping at the top level")
    root = raw.get("plot_grid_timeseries")
    if not isinstance(root, dict):
        raise ValueError("config.yaml is missing required key: plot_grid_timeseries")

    out_dir_raw = root.get("out_dir")
    out_dir: Path | None = None
    if out_dir_raw is not None:
        out_dir = resolve_repo_path(Path(str(out_dir_raw)))

    categories_raw = root.get("categories")
    if not isinstance(categories_raw, list) or not categories_raw:
        raise ValueError("plot_grid_timeseries.categories must be a non-empty list")

    categories: list[dict] = []
    for cat_idx, cat in enumerate(categories_raw):
        if not isinstance(cat, dict):
            raise ValueError(f"categories[{cat_idx}] must be a mapping")
        tag = str(cat.get("tag", "")).strip()
        title = str(cat.get("title", "")).strip()
        if not tag or not title:
            raise ValueError(f"categories[{cat_idx}] requires 'tag' and 'title'")
        try:
            nrows = int(cat.get("nrows"))
            ncols = int(cat.get("ncols"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"categories[{cat_idx}] requires integer nrows/ncols") from exc
        figsize_raw = cat.get("figsize")
        if not isinstance(figsize_raw, (list, tuple)) or len(figsize_raw) != 2:
            raise ValueError(f"categories[{cat_idx}].figsize must be a 2-item list like [14, 10]")
        try:
            figsize = (float(figsize_raw[0]), float(figsize_raw[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"categories[{cat_idx}].figsize must be numeric") from exc

        subplots_raw = cat.get("subplots")
        if not isinstance(subplots_raw, list) or not subplots_raw:
            raise ValueError(f"categories[{cat_idx}].subplots must be a non-empty list")
        subplots: list[tuple[int, int, object, str]] = []
        for sp_idx, sp in enumerate(subplots_raw):
            if not isinstance(sp, dict):
                raise ValueError(f"categories[{cat_idx}].subplots[{sp_idx}] must be a mapping")
            try:
                r = int(sp.get("row"))
                c = int(sp.get("col"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"subplots[{sp_idx}] requires integer row/col") from exc
            ylabel = str(sp.get("ylabel", "")).strip()
            if not ylabel:
                raise ValueError(f"subplots[{sp_idx}] requires 'ylabel'")
            series_raw = sp.get("series")
            if not isinstance(series_raw, list) or not series_raw:
                raise ValueError(f"subplots[{sp_idx}].series must be a non-empty list")

            series_specs: list[tuple[str, str, str]] = []
            for ser_idx, ser in enumerate(series_raw):
                if not isinstance(ser, dict) or "col" not in ser:
                    raise ValueError(f"subplots[{sp_idx}].series[{ser_idx}] must be a mapping with 'col'")
                col_name = str(ser.get("col", "")).strip()
                if not col_name:
                    raise ValueError(f"subplots[{sp_idx}].series[{ser_idx}].col must be a non-empty string")
                side = str(ser.get("side", "")).strip()
                linestyle = str(ser.get("linestyle", "-")).strip() or "-"
                series_specs.append((col_name, side, linestyle))

            if len(series_specs) == 1 and not series_specs[0][1] and series_specs[0][2] == "-":
                col_spec: object = series_specs[0][0]
            else:
                col_spec = series_specs
            subplots.append((r, c, col_spec, ylabel))

        categories.append(
            {
                "tag": tag,
                "title": title,
                "nrows": nrows,
                "ncols": ncols,
                "figsize": figsize,
                "subplots": subplots,
            }
        )

    return out_dir, categories


def load_data(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(csv_path, encoding="utf8-lossy", infer_schema_length=10000)
    required = TRIAL_KEYS + [
        "MocapFrame",
        "platform_onset_local",
        "platform_offset_local",
        "step_onset_local",
        TIME_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


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


def normalize_x_values(x_values: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    span = x_max - x_min
    if span <= 0.0:
        return np.zeros_like(x_values, dtype=float)
    return (x_values - x_min) / span


def normalize_x_scalar(x_value: float, x_min: float, x_max: float) -> float:
    span = x_max - x_min
    if span <= 0.0:
        return 0.0
    return float((x_value - x_min) / span)


def get_trial_time_bounds(trial_df: pl.DataFrame) -> tuple[float, float] | None:
    x_raw = np.asarray(trial_df.get_column(TIME_COL).to_list(), dtype=float)
    valid = np.isfinite(x_raw)
    if np.count_nonzero(valid) == 0:
        return None
    x = x_raw[valid]
    return float(np.min(x)), float(np.max(x))


def build_normalized_xticks(xtick_norm: float) -> np.ndarray:
    if xtick_norm <= 0:
        raise ValueError("--xtick_norm must be > 0")
    ticks = np.arange(0.0, 1.0 + (xtick_norm * 0.5), xtick_norm, dtype=float)
    ticks = np.clip(ticks, 0.0, 1.0)
    ticks = np.unique(np.concatenate([ticks, np.array([1.0], dtype=float)]))
    return np.round(ticks, 6)


def estimate_dt_seconds(df: pl.DataFrame) -> float:
    if df.height < 2 or TIME_COL not in df.columns:
        return 0.01
    time_series = df.get_column(TIME_COL).drop_nulls().unique().sort()
    if time_series.len() < 2:
        return 0.01
    values = np.asarray(time_series.to_list(), dtype=float)
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 0.01
    return float(np.median(diffs))


def piecewise_warp_times(
    t_raw_s: np.ndarray,
    *,
    offset_time_s: float,
    mid_duration_s: float,
) -> np.ndarray:
    out = np.asarray(t_raw_s, dtype=float).copy()
    if not np.isfinite(offset_time_s) or offset_time_s <= 0.0:
        return out
    if not np.isfinite(mid_duration_s) or mid_duration_s <= 0.0:
        return out
    mid_mask = (out >= 0.0) & (out <= offset_time_s)
    post_mask = out > offset_time_s
    out[mid_mask] = (out[mid_mask] / offset_time_s) * mid_duration_s
    out[post_mask] = mid_duration_s + (out[post_mask] - offset_time_s)
    return out


def piecewise_warp_scalar(
    t_raw_s: float,
    *,
    offset_time_s: float,
    mid_duration_s: float,
) -> float:
    if not np.isfinite(t_raw_s):
        return float("nan")
    if not np.isfinite(offset_time_s) or offset_time_s <= 0.0:
        return float(t_raw_s)
    if not np.isfinite(mid_duration_s) or mid_duration_s <= 0.0:
        return float(t_raw_s)
    if t_raw_s < 0.0:
        return float(t_raw_s)
    if t_raw_s <= offset_time_s:
        return float((t_raw_s / offset_time_s) * mid_duration_s)
    return float(mid_duration_s + (t_raw_s - offset_time_s))


def get_trial_scalar(trial_df: pl.DataFrame, col_name: str) -> float | None:
    if col_name not in trial_df.columns:
        return None
    values = trial_df.get_column(col_name).drop_nulls().unique()
    if values.len() == 0:
        return None
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return None


def resample_xy(x_raw: np.ndarray, y_raw: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
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


def resample_trial_column(
    trial_df: pl.DataFrame,
    col_name: str,
    x_grid: np.ndarray,
    x_mode: str,
    piecewise_mid_duration_s: float | None,
) -> np.ndarray:
    if col_name not in trial_df.columns:
        return np.full(x_grid.shape, np.nan, dtype=float)

    x_raw = np.asarray(trial_df.get_column(TIME_COL).to_list(), dtype=float)
    y_raw = np.asarray(trial_df.get_column(col_name).to_list(), dtype=float)
    if x_mode == "norm01":
        bounds = get_trial_time_bounds(trial_df)
        if bounds is None:
            return np.full(x_grid.shape, np.nan, dtype=float)
        x_raw = normalize_x_values(x_raw, bounds[0], bounds[1])
    elif x_mode == "piecewise":
        offset_time_s = get_trial_scalar(trial_df, "platform_offset_time_s")
        if offset_time_s is not None:
            mid_s = float(piecewise_mid_duration_s) if piecewise_mid_duration_s is not None else 0.0
            x_raw = piecewise_warp_times(
                x_raw,
                offset_time_s=float(offset_time_s),
                mid_duration_s=mid_s,
            )
    return resample_xy(x_raw=x_raw, y_raw=y_raw, x_grid=x_grid)


def draw_events(
    ax: plt.Axes,
    trial_df: pl.DataFrame,
    alpha: float,
    label_once: bool,
    x_mode: str,
    piecewise_mid_duration_s: float | None,
    x_norm_bounds: tuple[float, float] | None = None,
) -> None:
    onset_time_s = 0.0
    offset_time_s = get_trial_scalar(trial_df, "platform_offset_time_s")
    step_time_s = get_trial_scalar(trial_df, "step_onset_time_s")

    mid_s = float(piecewise_mid_duration_s) if piecewise_mid_duration_s is not None else 0.0

    def map_time(t_raw_s: float) -> float:
        if x_mode == "norm01" and x_norm_bounds is not None:
            return normalize_x_scalar(t_raw_s, x_norm_bounds[0], x_norm_bounds[1])
        if x_mode == "piecewise" and offset_time_s is not None:
            return piecewise_warp_scalar(
                t_raw_s,
                offset_time_s=float(offset_time_s),
                mid_duration_s=mid_s,
            )
        return float(t_raw_s)

    platform_x = map_time(onset_time_s)
    ax.axvline(
        platform_x,
        color="red",
        linewidth=0.9,
        alpha=alpha,
        linestyle="-",
        label="platform_onset" if label_once else None,
    )

    if offset_time_s is not None:
        offset_x = map_time(float(offset_time_s))
        ax.axvline(
            offset_x,
            color="green",
            linewidth=0.9,
            alpha=alpha,
            linestyle="-",
            label="platform_offset" if label_once else None,
        )

    if step_time_s is not None:
        step_x = map_time(float(step_time_s))
        ax.axvline(
            step_x,
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
    x_plot: np.ndarray,
    x_mode: str,
    piecewise_mid_duration_s: float | None,
) -> None:
    if not trials or col_name not in trials[0].columns:
        ax.text(0.5, 0.5, f"Missing: {col_name}", transform=ax.transAxes, ha="center", va="center")
        return

    line_width = 1.2 if sample else 0.6
    line_alpha = 0.95 if sample else 0.30
    event_alpha = 1.0 if sample else 0.20

    for idx, trial_df in enumerate(trials):
        trial_bounds = get_trial_time_bounds(trial_df) if x_mode == "norm01" else None
        y_vals = resample_trial_column(
            trial_df=trial_df,
            col_name=col_name,
            x_grid=x_plot,
            x_mode=x_mode,
            piecewise_mid_duration_s=piecewise_mid_duration_s,
        )
        ax.plot(
            x_plot,
            y_vals,
            color="gray",
            linewidth=line_width,
            alpha=line_alpha,
            label="trial lines" if idx == 0 else None,
        )
        draw_events(
            ax,
            trial_df,
            alpha=event_alpha,
            label_once=(idx == 0),
            x_mode=x_mode,
            piecewise_mid_duration_s=piecewise_mid_duration_s,
            x_norm_bounds=trial_bounds,
        )


def plot_lr_overlay(
    ax: plt.Axes,
    trials: list[pl.DataFrame],
    col_specs: list[tuple[str, str, str]],
    sample: bool,
    x_plot: np.ndarray,
    x_mode: str,
    piecewise_mid_duration_s: float | None,
) -> None:
    if not trials:
        return

    line_width = 1.2 if sample else 0.6
    line_alpha = 0.90 if sample else 0.30
    event_alpha = 1.0 if sample else 0.20
    side_color = {"L": "tab:blue", "R": "tab:orange"}

    for trial_idx, trial_df in enumerate(trials):
        trial_bounds = get_trial_time_bounds(trial_df) if x_mode == "norm01" else None
        trial_cols = set(trial_df.columns)
        for col_name, side, style in col_specs:
            if col_name not in trial_cols:
                continue
            y_vals = resample_trial_column(
                trial_df=trial_df,
                col_name=col_name,
                x_grid=x_plot,
                x_mode=x_mode,
                piecewise_mid_duration_s=piecewise_mid_duration_s,
            )
            ax.plot(
                x_plot,
                y_vals,
                color=side_color.get(side, "gray"),
                linestyle=style if sample else "-",
                linewidth=line_width,
                alpha=line_alpha,
                label=side if trial_idx == 0 else None,
            )
        draw_events(
            ax,
            trial_df,
            alpha=event_alpha,
            label_once=(trial_idx == 0),
            x_mode=x_mode,
            piecewise_mid_duration_s=piecewise_mid_duration_s,
            x_norm_bounds=trial_bounds,
        )


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


def trial_has_step(trial_df: pl.DataFrame) -> bool:
    if "step_onset_local" not in trial_df.columns:
        return False
    return trial_df.get_column("step_onset_local").drop_nulls().len() > 0


def split_trials_by_step(trials: list[pl.DataFrame]) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
    step_trials: list[pl.DataFrame] = []
    nonstep_trials: list[pl.DataFrame] = []
    for trial_df in trials:
        if trial_has_step(trial_df):
            step_trials.append(trial_df)
        else:
            nonstep_trials.append(trial_df)
    return step_trials, nonstep_trials


def plot_subject_category(
    subject_df: pl.DataFrame,
    subject_value: str,
    velocity_value: object | None,
    group_by: str,
    spec: dict,
    out_dir: Path,
    sample: bool,
    dpi: int,
    x_plot: np.ndarray,
    x_ticks: np.ndarray,
    x_axis_label: str,
    x_mode: str,
    piecewise_mid_duration_s: float | None,
    step_group: str = "all",
) -> Path | None:
    nrows, ncols = spec["nrows"], spec["ncols"]
    fig, axes = plt.subplots(nrows, ncols, figsize=spec["figsize"], squeeze=False)

    all_trials = build_trial_list(subject_df)
    if step_group == "all":
        trials = all_trials
    else:
        step_trials, nonstep_trials = split_trials_by_step(all_trials)
        trials = step_trials if step_group == "step" else nonstep_trials
    trial_count = len(trials)
    if trial_count == 0:
        plt.close(fig)
        return None

    for subplot in spec["subplots"]:
        r, c = subplot[0], subplot[1]
        col_spec = subplot[2]
        ylabel = subplot[3]
        ax = axes[r][c]

        if isinstance(col_spec, str):
            plot_single_col(
                ax,
                trials,
                col_spec,
                sample,
                x_plot,
                x_mode,
                piecewise_mid_duration_s,
            )
        else:
            plot_lr_overlay(
                ax,
                trials,
                col_spec,
                sample,
                x_plot,
                x_mode,
                piecewise_mid_duration_s,
            )

        ax.set_title(ylabel, fontsize=9)
        ax.grid(True, linewidth=0.35, alpha=0.5)
        x_left = float(x_plot[0])
        x_right = float(x_plot[-1])
        if np.isclose(x_left, x_right):
            x_left -= 0.5
            x_right += 0.5
        ax.set_xlim(x_left, x_right)
        ax.set_xticks(x_ticks)
        ax.margins(x=0.0)
        if r == nrows - 1:
            ax.set_xlabel(x_axis_label, fontsize=8)
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
    group_label = "all trials" if step_group == "all" else f"{step_group} only"
    if group_by == "subject":
        suptitle = (
            f"{spec['title']} | {group_label} | subject overlay ({trial_count} subject-velocity-trial lines) | {mode_label}"
        )
    else:
        velocity_token = "unknown"
        if velocity_value is not None:
            velocity_token = format_velocity(velocity_value)
        suptitle = f"{spec['title']} | velocity={velocity_token} | {group_label} | trial overlay ({trial_count} trial lines) | {mode_label}"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    suffix = f"__{step_group}" if step_group != "all" else ""
    if group_by == "subject":
        out_name = f"{spec['tag']}__subject-{safe_name(subject_value)}{suffix}__{mode_label}.png"
    else:
        velocity_token = "unknown"
        if velocity_value is not None:
            velocity_token = format_velocity(velocity_value)
        out_name = (
            f"{spec['tag']}__subject-{safe_name(subject_value)}__velocity-{safe_name(velocity_token)}{suffix}__{mode_label}.png"
        )
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    config_out_dir, categories = load_plot_specs(args.config)
    if args.out_dir is None:
        args.out_dir = config_out_dir if config_out_dir is not None else DEFAULT_OUT
    args.out_dir = resolve_repo_path(args.out_dir)
    args.csv = resolve_repo_path(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {args.csv}")
    df = load_data(args.csv)
    if args.segment_frames <= 0:
        raise ValueError("--segment_frames must be >= 1")

    # Plot window (local mocap frames): [platform_onset-frames, platform_offset+frames]
    df = df.filter(pl.col("MocapFrame") >= (pl.col("platform_onset_local") - int(args.segment_frames)))
    df = df.filter(pl.col("MocapFrame") <= (pl.col("platform_offset_local") + int(args.segment_frames)))

    only_subjects = parse_csv_list(args.only_subjects)
    if only_subjects:
        df = df.filter(pl.col("subject").is_in(only_subjects))
        print(f"Filter: only_subjects={only_subjects} (remaining rows={df.height})")
    only_velocities = parse_float_list(args.only_velocities)
    if only_velocities:
        df = df.filter(pl.col("velocity").is_in(only_velocities))
        print(f"Filter: only_velocities={only_velocities} (remaining rows={df.height})")

    if df.height == 0:
        raise ValueError("No rows to plot after applying filters.")

    dt_s = estimate_dt_seconds(df)
    df = df.with_columns(
        [
            ((pl.col("platform_offset_local") - pl.col("platform_onset_local")) * float(dt_s)).alias(
                "platform_offset_time_s"
            ),
            pl.when(pl.col("step_onset_local").is_not_null())
            .then((pl.col("step_onset_local") - pl.col("platform_onset_local")) * float(dt_s))
            .otherwise(None)
            .alias("step_onset_time_s"),
        ]
    )

    piecewise_mid_duration_s: float | None = None
    if args.x_norm01:
        x_mode = "norm01"
        x_grid = build_common_x_grid(df, args.resample_hz)
        x_ticks = build_normalized_xticks(args.xtick_norm)
        x_plot = normalize_x_values(x_grid, float(x_grid[0]), float(x_grid[-1]))
        x_axis_label = "Normalized time (0-1)"
    else:
        if args.x_piecewise:
            x_mode = "piecewise"
            segment_window_s = float(args.segment_frames) * float(dt_s)
            piecewise_mid_duration_s = segment_window_s
            step = 1.0 / float(args.resample_hz)
            x_min = -segment_window_s
            x_max = piecewise_mid_duration_s + segment_window_s
            x_grid = np.arange(x_min, x_max + (step * 0.5), step, dtype=float)
            x_plot = x_grid
            x_ticks = build_common_xticks(x_grid, args.xtick_sec)
            x_axis_label = "Piecewise-normalized time (s)"
        else:
            x_mode = "seconds"
            x_grid = build_common_x_grid(df, args.resample_hz)
            x_ticks = build_common_xticks(x_grid, args.xtick_sec)
            x_plot = x_grid
            x_axis_label = "Time from platform onset (s)"
    trial_count = df.select(TRIAL_KEYS).unique().height
    subjects_all = df.select("subject").unique().sort("subject").get_column("subject").to_list()
    if args.sample:
        subjects = subjects_all[: max(1, args.sample_subjects)]
    else:
        subjects = subjects_all

    print(f"Rows: {df.height}, Trials: {trial_count}, Subjects to render: {len(subjects)}")
    print(f"Mode: {'sample' if args.sample else 'all'}")
    print(f"Grouping: {args.group_by}")
    print(f"X-axis mode: {x_mode}")
    print(
        "Common x-axis range: "
        f"[{x_grid[0]:.3f}, {x_grid[-1]:.3f}] sec "
        f"({x_grid.size} points @ {args.resample_hz:g} Hz)"
    )
    if x_mode == "norm01":
        print(f"Normalized x-axis enabled (per trial): [0.000, 1.000] with {x_ticks.size} ticks")
        print(f"Normalized x-axis tick spacing: {args.xtick_norm:g}")
    else:
        print(f"Common x-axis ticks: every {args.xtick_sec:g} sec ({x_ticks.size} ticks)")
    if args.separate_step_nonstep:
        trial_flags = (
            df.select(TRIAL_KEYS + ["step_onset_local"])
            .unique(TRIAL_KEYS)
            .with_columns(pl.col("step_onset_local").is_not_null().alias("has_step"))
        )
        step_trial_count = trial_flags.filter(pl.col("has_step")).height
        nonstep_trial_count = trial_flags.filter(~pl.col("has_step")).height
        print(
            "Step split enabled: "
            f"step={step_trial_count}, nonstep={nonstep_trial_count} (criterion: step_onset_local non-null)"
        )

    if args.group_by == "subject":
        for subject_value in subjects:
            subject_df = df.filter(pl.col("subject") == subject_value)
            for spec in categories:
                if args.separate_step_nonstep:
                    for step_group in ["step", "nonstep"]:
                        out_path = plot_subject_category(
                            subject_df=subject_df,
                            subject_value=str(subject_value),
                            velocity_value=None,
                            group_by=args.group_by,
                            spec=spec,
                            out_dir=args.out_dir,
                            sample=args.sample,
                            dpi=args.dpi,
                            x_plot=x_plot,
                            x_ticks=x_ticks,
                            x_axis_label=x_axis_label,
                            x_mode=x_mode,
                            piecewise_mid_duration_s=piecewise_mid_duration_s,
                            step_group=step_group,
                        )
                        if out_path is not None:
                            print(f"Saved: {out_path}")
                else:
                    out_path = plot_subject_category(
                        subject_df=subject_df,
                        subject_value=str(subject_value),
                        velocity_value=None,
                        group_by=args.group_by,
                        spec=spec,
                        out_dir=args.out_dir,
                        sample=args.sample,
                        dpi=args.dpi,
                        x_plot=x_plot,
                        x_ticks=x_ticks,
                        x_axis_label=x_axis_label,
                        x_mode=x_mode,
                        piecewise_mid_duration_s=piecewise_mid_duration_s,
                        step_group="all",
                    )
                    if out_path is not None:
                        print(f"Saved: {out_path}")
    else:
        if args.sample and args.sample_velocities <= 0:
            raise ValueError("--sample_velocities must be >= 1 (sample mode)")
        for subject_value in subjects:
            subject_df_all = df.filter(pl.col("subject") == subject_value)
            velocities_all = (
                subject_df_all.select("velocity").unique().sort("velocity").get_column("velocity").to_list()
            )
            if args.sample:
                velocities = velocities_all[: max(1, args.sample_velocities)]
            else:
                velocities = velocities_all
            for velocity_value in velocities:
                group_df = subject_df_all.filter(pl.col("velocity") == velocity_value)
                for spec in categories:
                    if args.separate_step_nonstep:
                        for step_group in ["step", "nonstep"]:
                            out_path = plot_subject_category(
                                subject_df=group_df,
                                subject_value=str(subject_value),
                                velocity_value=velocity_value,
                                group_by=args.group_by,
                                spec=spec,
                                out_dir=args.out_dir,
                                sample=args.sample,
                                dpi=args.dpi,
                                x_plot=x_plot,
                                x_ticks=x_ticks,
                                x_axis_label=x_axis_label,
                                x_mode=x_mode,
                                piecewise_mid_duration_s=piecewise_mid_duration_s,
                                step_group=step_group,
                            )
                            if out_path is not None:
                                print(f"Saved: {out_path}")
                    else:
                        out_path = plot_subject_category(
                            subject_df=group_df,
                            subject_value=str(subject_value),
                            velocity_value=velocity_value,
                            group_by=args.group_by,
                            spec=spec,
                            out_dir=args.out_dir,
                            sample=args.sample,
                            dpi=args.dpi,
                            x_plot=x_plot,
                            x_ticks=x_ticks,
                            x_axis_label=x_axis_label,
                            x_mode=x_mode,
                            piecewise_mid_duration_s=piecewise_mid_duration_s,
                            step_group="all",
                        )
                        if out_path is not None:
                            print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
