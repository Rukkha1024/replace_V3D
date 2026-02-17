"""Create BOS/COM XY sample visualizations as static PNG + GIF.

This script reads `output/all_trials_timeseries.csv` and renders one trial:
- Static PNG: aggregated BOS outlines + COM trajectory + inside/outside markers
- GIF: frame-by-frame BOS rectangle + cumulative COM trail + current COM state

Default behavior picks the first trial by sorted (subject, velocity, trial).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import _bootstrap

_bootstrap.ensure_src_on_path()

import matplotlib
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

matplotlib.use("Agg")

REPO_ROOT = _bootstrap.REPO_ROOT
DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "figures" / "bos_com_xy_sample"
TRIAL_KEYS = ["subject", "velocity", "trial"]
REQUIRED_COLUMNS = [
    "subject",
    "velocity",
    "trial",
    "MocapFrame",
    "platform_onset_local",
    "platform_offset_local",
    "step_onset_local",
    "COM_X",
    "COM_Y",
    "BOS_minX",
    "BOS_maxX",
    "BOS_minY",
    "BOS_maxY",
]


@dataclass(frozen=True)
class TrialSeries:
    subject: str
    velocity: float
    trial: int
    mocap_frame: np.ndarray
    com_x: np.ndarray
    com_y: np.ndarray
    bos_minx: np.ndarray
    bos_maxx: np.ndarray
    bos_miny: np.ndarray
    bos_maxy: np.ndarray
    valid_mask: np.ndarray
    inside_mask: np.ndarray
    nan_invalid_count: int
    bos_invalid_count: int
    platform_onset_local: int
    platform_offset_local: int
    step_onset_local: int | None
    time_from_onset_s: np.ndarray | None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render BOS/COM XY sample as static PNG + GIF (inside/outside visible)."
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input long CSV path")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT, help="Output directory")
    ap.add_argument("--subject", type=str, default=None, help="Subject selector (must pair with velocity/trial)")
    ap.add_argument("--velocity", type=float, default=None, help="Velocity selector (must pair with subject/trial)")
    ap.add_argument("--trial", type=int, default=None, help="Trial selector (must pair with subject/velocity)")
    ap.add_argument("--fps", type=int, default=20, help="GIF FPS")
    ap.add_argument("--frame_step", type=int, default=1, help="Use every Nth valid frame for GIF")
    ap.add_argument("--dpi", type=int, default=180, help="Output DPI")
    ap.add_argument(
        "--gif_name_suffix",
        type=str,
        default="bos_com_xy_anim",
        help="Output GIF filename suffix",
    )
    ap.add_argument(
        "--png_name_suffix",
        type=str,
        default="bos_com_xy_static",
        help="Output PNG filename suffix",
    )
    ap.add_argument(
        "--save_png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save static PNG output (default: enabled; disable with --no-save_png).",
    )
    ap.add_argument(
        "--save_gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save GIF output (default: enabled; disable with --no-save_gif).",
    )
    return ap.parse_args()


def safe_name(text: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", "_", str(text))
    value = re.sub(r"\s+", "_", value).strip("_")
    return value if value else "unknown"


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def format_velocity(value: object) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return str(value)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def load_data(csv_path: Path) -> pl.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pl.read_csv(csv_path, encoding="utf8-lossy", infer_schema_length=10000)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def resolve_trial_selection(args: argparse.Namespace, df: pl.DataFrame) -> tuple[str, float, int, bool]:
    flags = [args.subject is not None, args.velocity is not None, args.trial is not None]
    if any(flags) and not all(flags):
        raise ValueError("Ambiguous selection: provide all of --subject, --velocity, --trial together.")

    if all(flags):
        return str(args.subject), float(args.velocity), int(args.trial), False

    first = df.select(TRIAL_KEYS).unique().sort(TRIAL_KEYS).row(0, named=True)
    return str(first["subject"]), float(first["velocity"]), int(first["trial"]), True


def get_optional_int_scalar(trial_df: pl.DataFrame, col_name: str) -> int | None:
    values = trial_df.get_column(col_name).drop_nulls().unique()
    if values.len() == 0:
        return None
    return int(values[0])


def build_trial_series(trial_df: pl.DataFrame, subject: str, velocity: float, trial: int) -> TrialSeries:
    trial_df = trial_df.sort("MocapFrame")

    mocap = np.asarray(trial_df.get_column("MocapFrame").to_list(), dtype=int)
    com_x = np.asarray(trial_df.get_column("COM_X").to_list(), dtype=float)
    com_y = np.asarray(trial_df.get_column("COM_Y").to_list(), dtype=float)
    bos_minx = np.asarray(trial_df.get_column("BOS_minX").to_list(), dtype=float)
    bos_maxx = np.asarray(trial_df.get_column("BOS_maxX").to_list(), dtype=float)
    bos_miny = np.asarray(trial_df.get_column("BOS_minY").to_list(), dtype=float)
    bos_maxy = np.asarray(trial_df.get_column("BOS_maxY").to_list(), dtype=float)
    time_from_onset = None
    if "time_from_platform_onset_s" in trial_df.columns:
        time_from_onset = np.asarray(trial_df.get_column("time_from_platform_onset_s").to_list(), dtype=float)

    finite_mask = (
        np.isfinite(com_x)
        & np.isfinite(com_y)
        & np.isfinite(bos_minx)
        & np.isfinite(bos_maxx)
        & np.isfinite(bos_miny)
        & np.isfinite(bos_maxy)
    )
    bos_order_mask = (bos_minx <= bos_maxx) & (bos_miny <= bos_maxy)
    valid_mask = finite_mask & bos_order_mask
    nan_invalid_count = int(np.count_nonzero(~finite_mask))
    bos_invalid_count = int(np.count_nonzero(finite_mask & (~bos_order_mask)))

    inside_mask = np.zeros(valid_mask.shape, dtype=bool)
    inside_mask[valid_mask] = (
        (com_x[valid_mask] >= bos_minx[valid_mask])
        & (com_x[valid_mask] <= bos_maxx[valid_mask])
        & (com_y[valid_mask] >= bos_miny[valid_mask])
        & (com_y[valid_mask] <= bos_maxy[valid_mask])
    )

    platform_onset = get_optional_int_scalar(trial_df, "platform_onset_local")
    platform_offset = get_optional_int_scalar(trial_df, "platform_offset_local")
    if platform_onset is None or platform_offset is None:
        raise ValueError("platform_onset_local/platform_offset_local are required per trial.")
    step_onset = get_optional_int_scalar(trial_df, "step_onset_local")

    return TrialSeries(
        subject=subject,
        velocity=velocity,
        trial=trial,
        mocap_frame=mocap,
        com_x=com_x,
        com_y=com_y,
        bos_minx=bos_minx,
        bos_maxx=bos_maxx,
        bos_miny=bos_miny,
        bos_maxy=bos_maxy,
        valid_mask=valid_mask,
        inside_mask=inside_mask,
        nan_invalid_count=nan_invalid_count,
        bos_invalid_count=bos_invalid_count,
        platform_onset_local=platform_onset,
        platform_offset_local=platform_offset,
        step_onset_local=step_onset,
        time_from_onset_s=time_from_onset,
    )


def compute_axis_limits(series: TrialSeries) -> tuple[tuple[float, float], tuple[float, float]]:
    valid_idx = np.flatnonzero(series.valid_mask)
    if valid_idx.size == 0:
        raise ValueError("No valid frame remains after filtering NaN/invalid BOS bounds.")

    x_values = np.concatenate(
        (
            series.com_x[valid_idx],
            series.bos_minx[valid_idx],
            series.bos_maxx[valid_idx],
        )
    )
    y_values = np.concatenate(
        (
            series.com_y[valid_idx],
            series.bos_miny[valid_idx],
            series.bos_maxy[valid_idx],
        )
    )

    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))

    x_span = x_max - x_min
    y_span = y_max - y_min
    x_margin = max(0.05 * x_span, 1e-3)
    y_margin = max(0.05 * y_span, 1e-3)

    if x_span <= 1e-12:
        x_margin = max(x_margin, 0.05)
    if y_span <= 1e-12:
        y_margin = max(y_margin, 0.05)

    return (x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin)


def draw_bos_outline(
    ax: plt.Axes,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    *,
    color: str = "0.65",
    alpha: float = 0.14,
    linewidth: float = 0.7,
) -> None:
    x = np.asarray([min_x, max_x, max_x, min_x, min_x], dtype=float)
    y = np.asarray([min_y, min_y, max_y, max_y, min_y], dtype=float)
    ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)


def get_com_point_for_frame(series: TrialSeries, event_frame: int | None) -> tuple[float, float] | None:
    if event_frame is None:
        return None
    idx = np.flatnonzero((series.mocap_frame == int(event_frame)) & series.valid_mask)
    if idx.size == 0:
        return None
    i = int(idx[0])
    return float(series.com_x[i]), float(series.com_y[i])


def save_figure(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    if out_path.exists():
        out_path.unlink()
        print(f"Overwrote: {out_path}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")


def render_static_png(
    series: TrialSeries,
    out_path: Path,
    dpi: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    valid_idx = np.flatnonzero(series.valid_mask)

    for i in valid_idx:
        draw_bos_outline(
            ax,
            float(series.bos_minx[i]),
            float(series.bos_maxx[i]),
            float(series.bos_miny[i]),
            float(series.bos_maxy[i]),
        )

    ax.plot(
        series.com_x[valid_idx],
        series.com_y[valid_idx],
        color="tab:blue",
        linewidth=1.9,
        alpha=0.95,
        label="COM trajectory",
    )

    inside_idx = np.flatnonzero(series.valid_mask & series.inside_mask)
    outside_idx = np.flatnonzero(series.valid_mask & (~series.inside_mask))
    if inside_idx.size:
        ax.scatter(
            series.com_x[inside_idx],
            series.com_y[inside_idx],
            s=14,
            c="tab:green",
            alpha=0.75,
            label="COM inside BOS",
            zorder=4,
        )
    if outside_idx.size:
        ax.scatter(
            series.com_x[outside_idx],
            series.com_y[outside_idx],
            s=18,
            c="tab:red",
            alpha=0.85,
            label="COM outside BOS",
            zorder=4,
        )

    event_specs = [
        ("platform_onset", series.platform_onset_local, "o", "black"),
        ("platform_offset", series.platform_offset_local, "s", "tab:orange"),
        ("step_onset", series.step_onset_local, "^", "tab:purple"),
    ]
    for label, event_frame, marker, color in event_specs:
        event_point = get_com_point_for_frame(series, event_frame)
        if event_point is None:
            continue
        ax.scatter(
            [event_point[0]],
            [event_point[1]],
            s=95,
            marker=marker,
            c=color,
            edgecolors="white",
            linewidths=0.7,
            label=label,
            zorder=6,
        )

    valid_count = int(valid_idx.size)
    inside_count = int(np.count_nonzero(series.valid_mask & series.inside_mask))
    outside_count = valid_count - inside_count
    inside_ratio = (100.0 * inside_count / valid_count) if valid_count > 0 else float("nan")
    summary_text = (
        f"velocity={format_velocity(series.velocity)}, trial={series.trial}\n"
        f"frames(valid/total)={valid_count}/{series.mocap_frame.size}\n"
        f"inside={inside_count}, outside={outside_count} ({inside_ratio:.1f}%)"
    )
    ax.text(
        0.99,
        0.99,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"},
    )

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.55)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"BOS + COM XY (static) | velocity={format_velocity(series.velocity)}, trial={series.trial}")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def render_gif(
    series: TrialSeries,
    out_path: Path,
    fps: int,
    frame_step: int,
    dpi: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
) -> int:
    if fps <= 0:
        raise ValueError("--fps must be >= 1")
    if frame_step <= 0:
        raise ValueError("--frame_step must be >= 1")

    valid_indices = np.flatnonzero(series.valid_mask)
    if valid_indices.size == 0:
        raise ValueError("No valid frame available for GIF generation.")

    frame_indices = valid_indices[::frame_step]
    if frame_indices[-1] != valid_indices[-1]:
        frame_indices = np.append(frame_indices, valid_indices[-1])

    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    bos_rect = Rectangle(
        (0.0, 0.0),
        width=1.0,
        height=1.0,
        facecolor="lightskyblue",
        edgecolor="tab:blue",
        alpha=0.25,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(bos_rect)

    trail_line, = ax.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.0,
        alpha=0.95,
        zorder=3,
    )
    current_point, = ax.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=8.5,
        markerfacecolor="tab:green",
        markeredgecolor="black",
        markeredgewidth=0.6,
        zorder=5,
    )
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"},
    )

    legend_handles = [
        Patch(facecolor="lightskyblue", edgecolor="tab:blue", alpha=0.25, label="Current BOS"),
        Line2D([0], [0], color="tab:blue", lw=2, label="COM cumulative trajectory"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="tab:green",
            markeredgecolor="black",
            label="Current COM (inside)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="tab:red",
            markeredgecolor="black",
            label="Current COM (outside)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.55)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"BOS + COM XY animation | velocity={format_velocity(series.velocity)}, trial={series.trial}")
    fig.tight_layout()

    valid_count = int(valid_indices.size)
    inside_count = int(np.count_nonzero(series.inside_mask[valid_indices]))
    outside_count = valid_count - inside_count
    inside_ratio = 100.0 * inside_count / valid_count

    def event_state(frame_value: int) -> str:
        labels: list[str] = []
        if frame_value == int(series.platform_onset_local):
            labels.append("platform_onset")
        if frame_value == int(series.platform_offset_local):
            labels.append("platform_offset")
        if series.step_onset_local is not None and frame_value == int(series.step_onset_local):
            labels.append("step_onset")
        return ",".join(labels) if labels else "-"

    def update(frame_no: int):
        idx = int(frame_indices[frame_no])
        frame_value = int(series.mocap_frame[idx])
        history = valid_indices[valid_indices <= idx]

        trail_line.set_data(series.com_x[history], series.com_y[history])
        cx = float(series.com_x[idx])
        cy = float(series.com_y[idx])
        current_point.set_data([cx], [cy])

        is_inside = bool(series.inside_mask[idx])
        color = "tab:green" if is_inside else "tab:red"
        current_point.set_markerfacecolor(color)
        current_point.set_markeredgecolor("black")

        min_x = float(series.bos_minx[idx])
        max_x = float(series.bos_maxx[idx])
        min_y = float(series.bos_miny[idx])
        max_y = float(series.bos_maxy[idx])
        bos_rect.set_xy((min_x, min_y))
        bos_rect.set_width(max_x - min_x)
        bos_rect.set_height(max_y - min_y)

        time_info = f"idx={idx + 1}/{series.mocap_frame.size}"
        if series.time_from_onset_s is not None and np.isfinite(series.time_from_onset_s[idx]):
            time_info = f"t={series.time_from_onset_s[idx]:.3f} s"

        info_text.set_text(
            f"frame={frame_value} ({frame_no + 1}/{frame_indices.size})\n"
            f"{time_info}\n"
            f"status={'inside' if is_inside else 'outside'}\n"
            f"event={event_state(frame_value)}\n"
            f"inside ratio={inside_ratio:.1f}% ({inside_count}/{valid_count})\n"
            f"outside={outside_count}"
        )
        return trail_line, current_point, bos_rect, info_text

    def init():
        return update(0)

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frame_indices.size,
        interval=max(1, int(round(1000.0 / float(fps)))),
        blit=False,
        repeat=False,
    )
    if out_path.exists():
        out_path.unlink()
        print(f"Overwrote: {out_path}")
    ani.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return int(frame_indices.size)


def main() -> None:
    args = parse_args()
    if (not bool(args.save_png)) and (not bool(args.save_gif)):
        raise ValueError("At least one output must be enabled: --save_png and/or --save_gif.")

    args.csv = resolve_repo_path(Path(args.csv))
    args.out_dir = resolve_repo_path(Path(args.out_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.csv)
    subject, velocity, trial, auto_selected = resolve_trial_selection(args, df)

    trial_df = df.filter(
        (pl.col("subject") == subject) & (pl.col("velocity") == float(velocity)) & (pl.col("trial") == int(trial))
    )
    if trial_df.height == 0:
        raise ValueError(
            f"Selected trial has no rows: subject={subject}, velocity={format_velocity(velocity)}, trial={trial}"
        )

    series = build_trial_series(trial_df=trial_df, subject=subject, velocity=velocity, trial=trial)
    x_lim, y_lim = compute_axis_limits(series)

    base_name = (
        f"{safe_name(subject)}__velocity-{safe_name(format_velocity(velocity))}"
        f"__trial-{int(trial)}"
    )
    png_out = args.out_dir / f"{base_name}__{safe_name(args.png_name_suffix)}.png"
    gif_out = args.out_dir / f"{base_name}__{safe_name(args.gif_name_suffix)}.gif"

    print(
        "Trial selection: "
        f"subject={subject}, velocity={format_velocity(velocity)}, trial={trial}, auto_selected={auto_selected}"
    )
    print(
        "Validity summary: "
        f"total_frames={series.mocap_frame.size}, valid_frames={int(np.count_nonzero(series.valid_mask))}, "
        f"nan_invalid={series.nan_invalid_count}, bos_invalid={series.bos_invalid_count}"
    )
    print(
        "Events: "
        f"platform_onset_local={series.platform_onset_local}, "
        f"platform_offset_local={series.platform_offset_local}, "
        f"step_onset_local={series.step_onset_local}"
    )

    gif_frames: int | None = None
    if bool(args.save_png):
        render_static_png(
            series=series,
            out_path=png_out,
            dpi=int(args.dpi),
            x_lim=x_lim,
            y_lim=y_lim,
        )
    if bool(args.save_gif):
        gif_frames = render_gif(
            series=series,
            out_path=gif_out,
            fps=int(args.fps),
            frame_step=int(args.frame_step),
            dpi=int(args.dpi),
            x_lim=x_lim,
            y_lim=y_lim,
        )

    valid_count = int(np.count_nonzero(series.valid_mask))
    inside_count = int(np.count_nonzero(series.valid_mask & series.inside_mask))
    outside_count = valid_count - inside_count
    inside_ratio = (100.0 * inside_count / valid_count) if valid_count > 0 else float("nan")
    print(
        "Inside/outside summary: "
        f"inside={inside_count}, outside={outside_count}, inside_ratio={inside_ratio:.2f}%"
    )
    if bool(args.save_png):
        print(f"Saved PNG: {png_out}")
    if bool(args.save_gif):
        print(f"Saved GIF: {gif_out} (frames={gif_frames}, fps={args.fps}, frame_step={args.frame_step})")


if __name__ == "__main__":
    main()
