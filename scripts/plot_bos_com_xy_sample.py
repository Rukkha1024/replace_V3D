"""Create BOS/COM XY sample visualizations as GIF.

This script reads `output/all_trials_timeseries.csv` and renders one trial:
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
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from replace_v3d.cli.batch_utils import iter_c3d_files
from replace_v3d.geometry.geometry2d import convex_hull_2d
from replace_v3d.io.c3d_reader import read_c3d_points
from replace_v3d.io.events_excel import (
    parse_subject_velocity_trial_from_filename,
    resolve_subject_from_token,
)

matplotlib.use("Agg")

REPO_ROOT = _bootstrap.REPO_ROOT
DEFAULT_CSV = REPO_ROOT / "output" / "all_trials_timeseries.csv"
DEFAULT_OUT = REPO_ROOT / "output" / "figures" / "bos_com_xy_sample"
DEFAULT_EVENT_XLSM = REPO_ROOT / "data" / "perturb_inform.xlsm"
DEFAULT_C3D_DIR = REPO_ROOT / "data" / "all_data"
GIF_BOS_MODES = ("freeze", "live")
RIGHT1COL_SUFFIX = "right1col"
TRIAL_KEYS = ["subject", "velocity", "trial"]
GIF_FIGSIZE = (8.2, 8.0)
GIF_LAYOUT_WIDTH_RATIOS = (3.45, 1.15)
GIF_LAYOUT_WSPACE = 0.05
BOS_MARKERS_ALL = [
    "LHEE",
    "LTOE",
    "LANK",
    "LFoot_3",
    "RHEE",
    "RTOE",
    "RANK",
    "RFoot_3",
]
BOS_MARKERS_L = ["LHEE", "LTOE", "LANK", "LFoot_3"]
BOS_MARKERS_R = ["RHEE", "RTOE", "RANK", "RFoot_3"]
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


@dataclass(frozen=True)
class DisplaySeries:
    rotate_ccw_deg: int
    com_x: np.ndarray
    com_y: np.ndarray
    bos_minx: np.ndarray
    bos_maxx: np.ndarray
    bos_miny: np.ndarray
    bos_maxy: np.ndarray
    x_lim: tuple[float, float]
    y_lim: tuple[float, float]


@dataclass(frozen=True)
class BOSPolylines:
    """Optional BOS overlay polylines already rotated into display frame."""

    source_c3d: Path
    hull_x: list[np.ndarray]
    hull_y: list[np.ndarray]
    union_x: list[np.ndarray]
    union_y: list[np.ndarray]


_PLATFORM_SHEET_CACHE: dict[Path, pd.DataFrame] = {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render BOS/COM XY sample as GIF (inside/outside visible)."
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input long CSV path")
    ap.add_argument(
        "--event_xlsm",
        type=Path,
        default=DEFAULT_EVENT_XLSM,
        help="Event workbook path (used for trial state subtitle; default: data/perturb_inform.xlsm).",
    )
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
        "--save_gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save GIF output (default: enabled; disable with --no-save_gif).",
    )
    ap.add_argument(
        "--rotate_ccw_deg",
        type=int,
        default=90,
        help="Display rotation in degrees CCW. Allowed: 0, 90, 180, 270 (default: 90).",
    )
    ap.add_argument(
        "--show_trial_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show step/nonstep and stepping-foot info in title subtitle (default: enabled).",
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


def load_platform_sheet(event_xlsm: Path) -> pd.DataFrame:
    resolved = event_xlsm.resolve()
    cached = _PLATFORM_SHEET_CACHE.get(resolved)
    if cached is not None:
        return cached

    if not resolved.exists():
        raise FileNotFoundError(f"Event workbook not found: {resolved}")

    df = pd.read_excel(resolved, sheet_name="platform")
    required = {"subject", "velocity", "trial", "state"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"platform sheet missing required columns: {', '.join(missing)}")

    out = df[["subject", "velocity", "trial", "state"]].copy()
    out["subject"] = out["subject"].astype(str).str.strip()
    out["velocity"] = pd.to_numeric(out["velocity"], errors="coerce")
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out["state"] = out["state"].astype(str).str.strip()
    _PLATFORM_SHEET_CACHE[resolved] = out
    return out


def canonicalize_trial_state(raw_state: str | None) -> str:
    if raw_state is None:
        return "unknown"
    text = str(raw_state).strip()
    if not text:
        return "unknown"
    low = text.lower()
    if low == "step_l":
        return "step_L"
    if low == "step_r":
        return "step_R"
    if low == "nonstep":
        return "nonstep"
    if low == "footlift":
        return "footlift"
    return text


def resolve_trial_state(
    event_xlsm: Path,
    *,
    subject: str,
    velocity: float,
    trial: int,
) -> str:
    try:
        platform = load_platform_sheet(event_xlsm)
    except Exception as exc:
        print(f"Warning: trial state lookup failed ({exc}). Using unknown.")
        return "unknown"

    subset = platform[platform["subject"] == str(subject).strip()]
    subset = subset[np.isclose(subset["velocity"].to_numpy(dtype=float), float(velocity), rtol=0.0, atol=1e-9)]
    subset = subset[subset["trial"] == float(int(trial))]
    if subset.empty:
        print(
            "Warning: no platform.state row for "
            f"subject={subject}, velocity={format_velocity(velocity)}, trial={trial}. Using unknown."
        )
        return "unknown"
    if len(subset) > 1:
        print(
            "Warning: multiple platform.state rows for "
            f"subject={subject}, velocity={format_velocity(velocity)}, trial={trial}. Using first row."
        )
    return canonicalize_trial_state(subset.iloc[0]["state"])


def format_trial_state_label(state: str) -> str:
    if state == "step_R":
        return "trial_type=step (R foot)"
    if state == "step_L":
        return "trial_type=step (L foot)"
    if state == "nonstep":
        return "trial_type=nonstep"
    if state == "footlift":
        return "trial_type=footlift (nonstep)"
    return f"trial_type=unknown ({state})" if state != "unknown" else "trial_type=unknown"


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


def normalize_rotate_ccw_deg(value: int) -> int:
    try:
        deg = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rotate_ccw_deg must be an integer. Got: {value!r}") from exc
    if deg not in {0, 90, 180, 270}:
        raise ValueError(f"Unsupported rotation: {deg}. Allowed values are 0, 90, 180, 270.")
    return deg


def rotate_xy(x: np.ndarray, y: np.ndarray, rotate_ccw_deg: int) -> tuple[np.ndarray, np.ndarray]:
    deg = normalize_rotate_ccw_deg(rotate_ccw_deg)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if deg == 0:
        return x_arr.copy(), y_arr.copy()
    if deg == 90:
        return -y_arr, x_arr
    if deg == 180:
        return -x_arr, -y_arr
    return y_arr, -x_arr


def rotate_box_bounds(
    min_x: np.ndarray,
    max_x: np.ndarray,
    min_y: np.ndarray,
    max_y: np.ndarray,
    rotate_ccw_deg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_corners = np.stack((min_x, max_x, max_x, min_x), axis=-1)
    y_corners = np.stack((min_y, min_y, max_y, max_y), axis=-1)
    x_rot, y_rot = rotate_xy(x_corners, y_corners, rotate_ccw_deg=rotate_ccw_deg)
    out_min_x = np.nanmin(x_rot, axis=-1)
    out_max_x = np.nanmax(x_rot, axis=-1)
    out_min_y = np.nanmin(y_rot, axis=-1)
    out_max_y = np.nanmax(y_rot, axis=-1)
    return out_min_x, out_max_x, out_min_y, out_max_y


def compute_axis_limits_from_arrays(
    *,
    com_x: np.ndarray,
    com_y: np.ndarray,
    bos_minx: np.ndarray,
    bos_maxx: np.ndarray,
    bos_miny: np.ndarray,
    bos_maxy: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        raise ValueError("No valid frame remains after filtering NaN/invalid BOS bounds.")

    x_values = np.concatenate(
        (
            com_x[valid_idx],
            bos_minx[valid_idx],
            bos_maxx[valid_idx],
        )
    )
    y_values = np.concatenate(
        (
            com_y[valid_idx],
            bos_miny[valid_idx],
            bos_maxy[valid_idx],
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


def build_display_series(series: TrialSeries, rotate_ccw_deg: int) -> DisplaySeries:
    deg = normalize_rotate_ccw_deg(rotate_ccw_deg)
    com_x, com_y = rotate_xy(series.com_x, series.com_y, rotate_ccw_deg=deg)
    bos_minx, bos_maxx, bos_miny, bos_maxy = rotate_box_bounds(
        series.bos_minx,
        series.bos_maxx,
        series.bos_miny,
        series.bos_maxy,
        rotate_ccw_deg=deg,
    )
    x_lim, y_lim = compute_axis_limits_from_arrays(
        com_x=com_x,
        com_y=com_y,
        bos_minx=bos_minx,
        bos_maxx=bos_maxx,
        bos_miny=bos_miny,
        bos_maxy=bos_maxy,
        valid_mask=series.valid_mask,
    )
    return DisplaySeries(
        rotate_ccw_deg=deg,
        com_x=com_x,
        com_y=com_y,
        bos_minx=bos_minx,
        bos_maxx=bos_maxx,
        bos_miny=bos_miny,
        bos_maxy=bos_maxy,
        x_lim=x_lim,
        y_lim=y_lim,
    )


def resolve_c3d_for_trial(
    *,
    c3d_dir: Path,
    event_xlsm: Path,
    subject: str,
    velocity: float,
    trial: int,
) -> Path | None:
    """Locate matching C3D for (subject, velocity, trial) using token resolution."""
    c3d_dir = Path(c3d_dir)
    if not c3d_dir.exists():
        return None

    matches: list[Path] = []
    for path in iter_c3d_files(c3d_dir):
        try:
            token, vel, tri = parse_subject_velocity_trial_from_filename(path.name)
        except Exception:
            continue

        if int(tri) != int(trial):
            continue
        if not np.isclose(float(vel), float(velocity), rtol=0.0, atol=1e-9):
            continue

        try:
            subj = resolve_subject_from_token(event_xlsm, token)
        except Exception:
            continue

        if str(subj).strip() != str(subject).strip():
            continue
        matches.append(path)

    if not matches:
        return None
    matches = sorted(matches)
    if len(matches) > 1:
        print(
            "Warning: multiple C3D matches for "
            f"subject={subject}, velocity={format_velocity(velocity)}, trial={trial}. "
            f"Using: {matches[0].name}"
        )
    return matches[0]


def _poly_to_closed_xy(poly_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poly = np.asarray(poly_xy, dtype=float)
    if poly.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = np.append(poly[:, 0], poly[0, 0])
    y = np.append(poly[:, 1], poly[0, 1])
    return x, y


def _join_polylines(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y in parts:
        if x.size == 0 or y.size == 0:
            continue
        if xs:
            xs.append(np.asarray([np.nan], dtype=float))
            ys.append(np.asarray([np.nan], dtype=float))
        xs.append(np.asarray(x, dtype=float))
        ys.append(np.asarray(y, dtype=float))
    if not xs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return np.concatenate(xs), np.concatenate(ys)


def compute_bos_polylines_from_c3d(
    *,
    c3d_path: Path,
    n_frames: int,
    rotate_ccw_deg: int,
) -> BOSPolylines:
    """Compute BOS hull and left/right-union outlines from C3D markers."""
    c3d = read_c3d_points(c3d_path)
    label_to_idx = {label: i for i, label in enumerate(c3d.labels)}

    def idx_for(markers: list[str]) -> list[int]:
        missing = [marker for marker in markers if marker not in label_to_idx]
        if missing:
            raise KeyError(f"Missing BoS markers in {c3d_path.name}: {', '.join(missing)}")
        return [int(label_to_idx[marker]) for marker in markers]

    idx_all = idx_for(BOS_MARKERS_ALL)
    idx_l = idx_for(BOS_MARKERS_L)
    idx_r = idx_for(BOS_MARKERS_R)

    n_csv = int(n_frames)
    n_c3d = int(c3d.points.shape[0])
    end = min(n_csv, n_c3d)
    if n_c3d != n_csv:
        print(
            f"Warning: C3D frames ({n_c3d}) != CSV frames ({n_csv}). "
            f"Using first {end} frames for BOS hull/union."
        )

    hull_x: list[np.ndarray] = []
    hull_y: list[np.ndarray] = []
    union_x: list[np.ndarray] = []
    union_y: list[np.ndarray] = []
    for t in range(n_csv):
        if t >= end:
            hull_x.append(np.asarray([], dtype=float))
            hull_y.append(np.asarray([], dtype=float))
            union_x.append(np.asarray([], dtype=float))
            union_y.append(np.asarray([], dtype=float))
            continue

        pts_all = c3d.points[t, idx_all, :2]
        poly_hull = convex_hull_2d(pts_all)
        hx, hy = _poly_to_closed_xy(poly_hull)
        hx, hy = rotate_xy(hx, hy, rotate_ccw_deg=rotate_ccw_deg)
        hull_x.append(hx)
        hull_y.append(hy)

        pts_l = c3d.points[t, idx_l, :2]
        pts_r = c3d.points[t, idx_r, :2]
        poly_l = convex_hull_2d(pts_l)
        poly_r = convex_hull_2d(pts_r)
        lx, ly = _poly_to_closed_xy(poly_l)
        rx, ry = _poly_to_closed_xy(poly_r)
        lx, ly = rotate_xy(lx, ly, rotate_ccw_deg=rotate_ccw_deg)
        rx, ry = rotate_xy(rx, ry, rotate_ccw_deg=rotate_ccw_deg)

        ux, uy = _join_polylines([(lx, ly), (rx, ry)])
        union_x.append(ux)
        union_y.append(uy)

    return BOSPolylines(
        source_c3d=Path(c3d_path),
        hull_x=hull_x,
        hull_y=hull_y,
        union_x=union_x,
        union_y=union_y,
    )


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


def build_gif_legend_handles() -> list[object]:
    return [
        Patch(facecolor="lightskyblue", edgecolor="tab:blue", alpha=0.25, label="Current BOS (bbox)"),
        Line2D([0], [0], color="0.25", lw=1.4, linestyle="--", label="BOS hull (all-foot convex)"),
        Line2D([0], [0], color="tab:purple", lw=1.4, label="BOS union (L/R hull)"),
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


def resolve_gif_trial_state_line(trial_state_label: str | None) -> str:
    if trial_state_label is None:
        return "trial_type=unknown"
    text = str(trial_state_label).strip()
    return text if text else "trial_type=unknown"


def create_gif_canvas() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=GIF_FIGSIZE)
    grid = fig.add_gridspec(1, 2, width_ratios=GIF_LAYOUT_WIDTH_RATIOS, wspace=GIF_LAYOUT_WSPACE)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_side = fig.add_subplot(grid[0, 1])
    ax_side.axis("off")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.085, top=0.88)
    return fig, ax_main, ax_side


def apply_gif_right_panel(ax_side: plt.Axes) -> object:
    info_text = ax_side.text(
        0.02,
        0.98,
        "",
        transform=ax_side.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "0.7"},
    )
    ax_side.legend(
        handles=build_gif_legend_handles(),
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=1,
        fontsize=8,
        frameon=True,
        borderaxespad=0.25,
        handlelength=2.0,
        columnspacing=0.9,
    )
    return info_text


def _collect_finite_polyline_values(polylines: list[np.ndarray], valid_mask: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for i, arr in enumerate(polylines):
        if i >= valid_mask.size or (not bool(valid_mask[i])):
            continue
        values = np.asarray(arr, dtype=float)
        if values.size == 0:
            continue
        finite = values[np.isfinite(values)]
        if finite.size > 0:
            chunks.append(finite)
    if not chunks:
        return np.asarray([], dtype=float)
    return np.concatenate(chunks)


def compute_fixed_gif_axis_limits(
    *,
    series: TrialSeries,
    display: DisplaySeries,
    bos_polylines: BOSPolylines | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    valid_idx = np.flatnonzero(series.valid_mask)
    if valid_idx.size == 0:
        raise ValueError("No valid frame remains for fixed GIF axis calculation.")

    x_parts: list[np.ndarray] = [
        display.com_x[valid_idx],
        display.bos_minx[valid_idx],
        display.bos_maxx[valid_idx],
    ]
    y_parts: list[np.ndarray] = [
        display.com_y[valid_idx],
        display.bos_miny[valid_idx],
        display.bos_maxy[valid_idx],
    ]
    if bos_polylines is not None:
        hull_x = _collect_finite_polyline_values(bos_polylines.hull_x, series.valid_mask)
        hull_y = _collect_finite_polyline_values(bos_polylines.hull_y, series.valid_mask)
        union_x = _collect_finite_polyline_values(bos_polylines.union_x, series.valid_mask)
        union_y = _collect_finite_polyline_values(bos_polylines.union_y, series.valid_mask)
        if hull_x.size > 0:
            x_parts.append(hull_x)
        if hull_y.size > 0:
            y_parts.append(hull_y)
        if union_x.size > 0:
            x_parts.append(union_x)
        if union_y.size > 0:
            y_parts.append(union_y)

    x_values = np.concatenate(x_parts)
    y_values = np.concatenate(y_parts)
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


def get_com_point_for_frame(
    series: TrialSeries,
    display: DisplaySeries,
    event_frame: int | None,
) -> tuple[float, float] | None:
    if event_frame is None:
        return None
    idx = np.flatnonzero((series.mocap_frame == int(event_frame)) & series.valid_mask)
    if idx.size == 0:
        return None
    i = int(idx[0])
    return float(display.com_x[i]), float(display.com_y[i])


def save_figure(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    if out_path.exists():
        out_path.unlink()
        print(f"Overwrote: {out_path}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")


def set_title_and_subtitle(
    ax: plt.Axes,
    *,
    title: str,
    subtitle: str | None,
) -> None:
    if subtitle:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=20)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    else:
        ax.set_title(title, fontsize=11, fontweight="bold")


def render_static_png(
    series: TrialSeries,
    display: DisplaySeries,
    trial_state_label: str | None,
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    valid_idx = np.flatnonzero(series.valid_mask)

    for i in valid_idx:
        draw_bos_outline(
            ax,
            float(display.bos_minx[i]),
            float(display.bos_maxx[i]),
            float(display.bos_miny[i]),
            float(display.bos_maxy[i]),
        )

    ax.plot(
        display.com_x[valid_idx],
        display.com_y[valid_idx],
        color="tab:blue",
        linewidth=1.9,
        alpha=0.95,
        label="COM trajectory",
    )

    inside_idx = np.flatnonzero(series.valid_mask & series.inside_mask)
    outside_idx = np.flatnonzero(series.valid_mask & (~series.inside_mask))
    if inside_idx.size:
        ax.scatter(
            display.com_x[inside_idx],
            display.com_y[inside_idx],
            s=14,
            c="tab:green",
            alpha=0.75,
            label="COM inside BOS",
            zorder=4,
        )
    if outside_idx.size:
        ax.scatter(
            display.com_x[outside_idx],
            display.com_y[outside_idx],
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
        event_point = get_com_point_for_frame(series, display, event_frame)
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

    ax.set_xlim(*display.x_lim)
    ax.set_ylim(*display.y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.55)
    ax.set_xlabel("X (m) [- Left / + Right]")
    ax.set_ylabel("Y (m) [+ Anterior / - Posterior]")
    set_title_and_subtitle(
        ax,
        title=(
            "BOS + COM XY (static) | "
            f"velocity={format_velocity(series.velocity)}, trial={series.trial}, "
            f"view=CCW{display.rotate_ccw_deg}"
        ),
        subtitle=trial_state_label,
    )
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if trial_state_label else None)
    save_figure(fig, out_path, dpi=dpi)
    plt.close(fig)


def render_gif(
    series: TrialSeries,
    display: DisplaySeries,
    trial_state_label: str | None,
    out_path: Path,
    fps: int,
    frame_step: int,
    dpi: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    bos_polylines: BOSPolylines | None = None,
    bos_mode: str = "freeze",
) -> int:
    if fps <= 0:
        raise ValueError("--fps must be >= 1")
    if frame_step <= 0:
        raise ValueError("--frame_step must be >= 1")
    mode = str(bos_mode).strip().lower()
    if mode not in GIF_BOS_MODES:
        raise ValueError(f"Unsupported bos_mode={bos_mode!r}. Allowed: {', '.join(GIF_BOS_MODES)}")

    valid_indices = np.flatnonzero(series.valid_mask)
    if valid_indices.size == 0:
        raise ValueError("No valid frame available for GIF generation.")

    frame_indices = valid_indices[::frame_step]
    if frame_indices[-1] != valid_indices[-1]:
        frame_indices = np.append(frame_indices, valid_indices[-1])

    bos_freeze_idx: int | None = None
    if mode == "freeze" and series.step_onset_local is not None:
        step_frame = int(series.step_onset_local)
        step_exact = np.flatnonzero((series.mocap_frame == step_frame) & series.valid_mask)
        if step_exact.size > 0:
            bos_freeze_idx = int(step_exact[0])
        else:
            # Fallback: first valid frame at/after step onset.
            step_after = np.flatnonzero(series.valid_mask & (series.mocap_frame >= step_frame))
            if step_after.size > 0:
                bos_freeze_idx = int(step_after[0])
                print(
                    "Warning: step_onset frame not found in valid rows. "
                    f"Using first valid frame >= step_onset: {series.mocap_frame[bos_freeze_idx]}"
                )
            else:
                print("Warning: no valid frames at/after step onset. BOS freeze disabled for this trial.")

    fig, ax, ax_side = create_gif_canvas()

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
    bos_union_line = None
    bos_hull_line = None
    if bos_polylines is not None:
        bos_union_line, = ax.plot(
            [],
            [],
            color="tab:purple",
            linewidth=1.4,
            alpha=0.9,
            zorder=4,
        )
        bos_hull_line, = ax.plot(
            [],
            [],
            color="0.25",
            linewidth=1.4,
            alpha=0.9,
            linestyle="--",
            zorder=4,
        )
    info_text = apply_gif_right_panel(ax_side)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.55)
    ax.set_xlabel("X (m) [- Left / + Right]")
    ax.set_ylabel("Y (m) [+ Anterior / - Posterior]")
    gif_trial_state_line = resolve_gif_trial_state_line(trial_state_label)
    set_title_and_subtitle(
        ax,
        title="BOS + COM XY animation",
        subtitle=gif_trial_state_line,
    )

    valid_count = int(valid_indices.size)
    inside_count = int(np.count_nonzero(series.inside_mask[valid_indices]))
    outside_count = valid_count - inside_count
    inside_ratio = 100.0 * inside_count / valid_count
    panel_header = (
        f"subject={series.subject}\n"
        f"velocity={format_velocity(series.velocity)}, trial={series.trial}\n"
        f"view=CCW{display.rotate_ccw_deg}\n"
        f"bos_mode={mode}"
    )

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

        trail_line.set_data(display.com_x[history], display.com_y[history])
        cx = float(display.com_x[idx])
        cy = float(display.com_y[idx])
        current_point.set_data([cx], [cy])

        is_inside = bool(series.inside_mask[idx])
        color = "tab:green" if is_inside else "tab:red"
        current_point.set_markerfacecolor(color)
        current_point.set_markeredgecolor("black")

        bos_idx = idx
        bos_state = "live(no-freeze)" if mode == "live" else "live"
        if mode == "freeze" and bos_freeze_idx is not None and idx >= bos_freeze_idx:
            bos_idx = int(bos_freeze_idx)
            bos_state = "frozen@step_onset"

        min_x = float(display.bos_minx[bos_idx])
        max_x = float(display.bos_maxx[bos_idx])
        min_y = float(display.bos_miny[bos_idx])
        max_y = float(display.bos_maxy[bos_idx])
        bos_rect.set_xy((min_x, min_y))
        bos_rect.set_width(max_x - min_x)
        bos_rect.set_height(max_y - min_y)
        if bos_polylines is not None and bos_union_line is not None and bos_hull_line is not None:
            bos_union_line.set_data(
                bos_polylines.union_x[bos_idx],
                bos_polylines.union_y[bos_idx],
            )
            bos_hull_line.set_data(
                bos_polylines.hull_x[bos_idx],
                bos_polylines.hull_y[bos_idx],
            )

        time_info = f"idx={idx + 1}/{series.mocap_frame.size}"
        if series.time_from_onset_s is not None and np.isfinite(series.time_from_onset_s[idx]):
            time_info = f"t={series.time_from_onset_s[idx]:.3f} s"

        info_text.set_text(
            f"{panel_header}\n\n"
            f"frame={frame_value} ({frame_no + 1}/{frame_indices.size})\n"
            f"{time_info}\n"
            f"status={'inside' if is_inside else 'outside'}\n"
            f"event={event_state(frame_value)}\n"
            f"bos={bos_state}\n"
            f"inside ratio={inside_ratio:.1f}% ({inside_count}/{valid_count})\n"
            f"outside={outside_count}"
        )
        artists: list[object] = [trail_line, current_point, bos_rect, info_text]
        if bos_union_line is not None and bos_hull_line is not None:
            artists.extend([bos_union_line, bos_hull_line])
        return tuple(artists)

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
    if not bool(args.save_gif):
        raise ValueError("GIF output is required. Use --save_gif.")
    rotate_ccw_deg = normalize_rotate_ccw_deg(int(args.rotate_ccw_deg))

    args.csv = resolve_repo_path(Path(args.csv))
    args.event_xlsm = resolve_repo_path(Path(args.event_xlsm))
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
    display = build_display_series(series, rotate_ccw_deg=rotate_ccw_deg)
    trial_state = resolve_trial_state(
        event_xlsm=args.event_xlsm,
        subject=subject,
        velocity=float(velocity),
        trial=int(trial),
    )
    trial_state_label = format_trial_state_label(trial_state) if bool(args.show_trial_state) else None

    base_name = (
        f"{safe_name(subject)}__velocity-{safe_name(format_velocity(velocity))}"
        f"__trial-{int(trial)}"
    )
    gif_base = args.out_dir / f"{base_name}__{safe_name(args.gif_name_suffix)}"

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
    print(f"Display rotation: CCW {rotate_ccw_deg} deg")
    print(f"Trial state: {trial_state}")

    bos_polylines: BOSPolylines | None = None
    try:
        c3d_path = resolve_c3d_for_trial(
            c3d_dir=DEFAULT_C3D_DIR,
            event_xlsm=args.event_xlsm,
            subject=subject,
            velocity=float(velocity),
            trial=int(trial),
        )
        if c3d_path is None:
            print("[BOS overlay] matching C3D not found; hull/union overlay disabled.")
        else:
            bos_polylines = compute_bos_polylines_from_c3d(
                c3d_path=c3d_path,
                n_frames=series.mocap_frame.size,
                rotate_ccw_deg=display.rotate_ccw_deg,
            )
            print(f"[BOS overlay] using C3D: {c3d_path}")
    except Exception as exc:
        print(f"[BOS overlay] disabled due to error: {exc}")
        bos_polylines = None

    fixed_x_lim, fixed_y_lim = compute_fixed_gif_axis_limits(
        series=series,
        display=display,
        bos_polylines=bos_polylines,
    )
    print(
        "Fixed axis limits: "
        f"x=({fixed_x_lim[0]:.4f}, {fixed_x_lim[1]:.4f}), "
        f"y=({fixed_y_lim[0]:.4f}, {fixed_y_lim[1]:.4f})"
    )

    gif_outputs: list[tuple[Path, int, str]] = []
    print("GIF outputs: right1col with bos_mode=freeze/live")
    for bos_mode in GIF_BOS_MODES:
        gif_out = Path(f"{gif_base}__{RIGHT1COL_SUFFIX}__{bos_mode}.gif")
        frames = render_gif(
            series=series,
            display=display,
            trial_state_label=trial_state_label,
            out_path=gif_out,
            fps=int(args.fps),
            frame_step=int(args.frame_step),
            dpi=int(args.dpi),
            x_lim=fixed_x_lim,
            y_lim=fixed_y_lim,
            bos_polylines=bos_polylines,
            bos_mode=bos_mode,
        )
        gif_outputs.append((gif_out, int(frames), bos_mode))

    valid_count = int(np.count_nonzero(series.valid_mask))
    inside_count = int(np.count_nonzero(series.valid_mask & series.inside_mask))
    outside_count = valid_count - inside_count
    inside_ratio = (100.0 * inside_count / valid_count) if valid_count > 0 else float("nan")
    print(
        "Inside/outside summary: "
        f"inside={inside_count}, outside={outside_count}, inside_ratio={inside_ratio:.2f}%"
    )
    for out_path, frames, bos_mode in gif_outputs:
        print(
            f"Saved GIF[{bos_mode}]: {out_path} "
            f"(frames={frames}, fps={args.fps}, frame_step={args.frame_step})"
        )


if __name__ == "__main__":
    main()
