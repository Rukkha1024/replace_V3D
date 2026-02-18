"""Create BOS/COM XY sample visualizations as GIF.

This script reads `output/all_trials_timeseries.csv` and renders one trial:
- GIF: frame-by-frame BOS rectangle + cumulative COM trail + current COM state

Default behavior picks the first trial by sorted (subject, velocity, trial).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import re
import time
from collections import deque
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
STEP_VIS_TEMPLATES = ("phase_trail", "bos_phase", "star_only", "phase_bos")
TRIAL_KEYS = ["subject", "velocity", "trial"]
GIF_FIGSIZE = (8.2, 8.0)
GIF_LAYOUT_WIDTH_RATIOS = (3.45, 1.15)
GIF_LAYOUT_WSPACE = 0.05
XCOM_COLUMNS = ("xCOM_X", "xCOM_Y")
XCOM_TRAIL_COLOR = "teal"
XCOM_INSIDE_COLOR = "teal"
XCOM_OUTSIDE_COLOR = "mediumvioletred"
XCOM_GHOST_COLOR = "purple"
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
    xcom_x: np.ndarray | None
    xcom_y: np.ndarray | None
    xcom_valid_mask: np.ndarray | None
    xcom_inside_mask: np.ndarray | None
    platform_onset_local: int
    platform_offset_local: int
    step_onset_local: int | None
    time_from_onset_s: np.ndarray | None


@dataclass(frozen=True)
class DisplaySeries:
    rotate_ccw_deg: int
    com_x: np.ndarray
    com_y: np.ndarray
    xcom_x: np.ndarray | None
    xcom_y: np.ndarray | None
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


@dataclass(frozen=True)
class TrialKey:
    subject: str
    velocity: float
    trial: int


@dataclass(frozen=True)
class RenderConfig:
    csv_path: Path
    event_xlsm: Path
    out_dir: Path
    c3d_dir: Path
    fps: int
    frame_step: int
    dpi: int
    gif_name_suffix: str
    rotate_ccw_deg: int
    show_trial_state: bool
    start_from_platform_onset_offset: int | None = None
    step_vis: str = "none"


@dataclass(frozen=True)
class TrialRenderResult:
    key: TrialKey
    gif_outputs: tuple[tuple[str, int, str], ...]
    valid_count: int
    inside_count: int
    outside_count: int
    inside_ratio: float
    elapsed_sec: float


_PLATFORM_SHEET_CACHE: dict[Path, pd.DataFrame] = {}
_WORKER_CSV_CACHE: dict[Path, pl.DataFrame] = {}
_XCOM_MISSING_WARNED = False


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
    ap.add_argument(
        "--all",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render all unique (subject, velocity, trial) keys from CSV (default: disabled).",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker count for --all mode (default: CPU count).",
    )
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
    ap.add_argument(
        "--step_vis",
        default="phase_bos",
        choices=["none", "phase_trail", "bos_phase", "star_only", "phase_bos", "all"],
        help=(
            "Step-onset visualization style (default: phase_bos). "
            "'none' = no step_onset overlay (legacy). "
            "'phase_trail' = color-split COM trail at step onset. "
            "'bos_phase' = BOS rect color changes at step onset. "
            "'star_only' = (no-op; star removed). "
            "'phase_bos' = trail split + BOS color flash (default). "
            "'all' = render all 4 template styles for comparison."
        ),
    )
    ap.add_argument(
        "--start_from_platform_onset_offset",
        type=int,
        default=None,
        help=(
            "Optional per-trial start frame offset from platform_onset_local. "
            "Example: -20 trims to [platform_onset_local-20, end]."
        ),
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


def load_data_for_worker(csv_path: Path) -> pl.DataFrame:
    resolved = csv_path.resolve()
    cached = _WORKER_CSV_CACHE.get(resolved)
    if cached is not None:
        return cached
    df = load_data(resolved)
    _WORKER_CSV_CACHE[resolved] = df
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


def resolve_trial_selection(args: argparse.Namespace, df: pl.DataFrame) -> tuple[TrialKey, bool]:
    flags = [args.subject is not None, args.velocity is not None, args.trial is not None]
    if any(flags) and not all(flags):
        raise ValueError("Ambiguous selection: provide all of --subject, --velocity, --trial together.")

    if all(flags):
        return TrialKey(subject=str(args.subject), velocity=float(args.velocity), trial=int(args.trial)), False

    first = df.select(TRIAL_KEYS).unique().sort(TRIAL_KEYS).row(0, named=True)
    return (
        TrialKey(
            subject=str(first["subject"]),
            velocity=float(first["velocity"]),
            trial=int(first["trial"]),
        ),
        True,
    )


def collect_trial_keys(df: pl.DataFrame) -> list[TrialKey]:
    rows = df.select(TRIAL_KEYS).unique().sort(TRIAL_KEYS).iter_rows(named=True)
    out: list[TrialKey] = []
    for row in rows:
        out.append(
            TrialKey(
                subject=str(row["subject"]),
                velocity=float(row["velocity"]),
                trial=int(row["trial"]),
            )
        )
    return out


def resolve_jobs(requested_jobs: int | None) -> int:
    if requested_jobs is None:
        return max(1, int(os.cpu_count() or 1))
    jobs = int(requested_jobs)
    if jobs < 1:
        raise ValueError("--jobs must be >= 1.")
    return jobs


def warn_missing_xcom_columns_once(columns: list[str]) -> None:
    global _XCOM_MISSING_WARNED
    if _XCOM_MISSING_WARNED:
        return
    missing = [col for col in XCOM_COLUMNS if col not in columns]
    if missing:
        print(f"Warning: xCOM overlay disabled. Missing columns: {', '.join(missing)}")
        _XCOM_MISSING_WARNED = True


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

    xcom_x: np.ndarray | None = None
    xcom_y: np.ndarray | None = None
    xcom_valid_mask: np.ndarray | None = None
    xcom_inside_mask: np.ndarray | None = None
    has_xcom = all(col in trial_df.columns for col in XCOM_COLUMNS)
    if has_xcom:
        xcom_x = np.asarray(trial_df.get_column("xCOM_X").to_list(), dtype=float)
        xcom_y = np.asarray(trial_df.get_column("xCOM_Y").to_list(), dtype=float)
        xcom_finite_mask = np.isfinite(xcom_x) & np.isfinite(xcom_y)
        xcom_valid_mask = valid_mask & xcom_finite_mask
        xcom_inside_mask = np.zeros(valid_mask.shape, dtype=bool)
        xcom_inside_mask[xcom_valid_mask] = (
            (xcom_x[xcom_valid_mask] >= bos_minx[xcom_valid_mask])
            & (xcom_x[xcom_valid_mask] <= bos_maxx[xcom_valid_mask])
            & (xcom_y[xcom_valid_mask] >= bos_miny[xcom_valid_mask])
            & (xcom_y[xcom_valid_mask] <= bos_maxy[xcom_valid_mask])
        )
    else:
        warn_missing_xcom_columns_once(trial_df.columns)

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
        xcom_x=xcom_x,
        xcom_y=xcom_y,
        xcom_valid_mask=xcom_valid_mask,
        xcom_inside_mask=xcom_inside_mask,
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
    xcom_x: np.ndarray | None = None,
    xcom_y: np.ndarray | None = None,
    xcom_valid_mask: np.ndarray | None = None,
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
    if xcom_x is not None and xcom_y is not None and xcom_valid_mask is not None:
        xcom_valid_idx = np.flatnonzero(xcom_valid_mask)
        if xcom_valid_idx.size > 0:
            x_values = np.concatenate((x_values, xcom_x[xcom_valid_idx]))
            y_values = np.concatenate((y_values, xcom_y[xcom_valid_idx]))

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
    xcom_x: np.ndarray | None = None
    xcom_y: np.ndarray | None = None
    if series.xcom_x is not None and series.xcom_y is not None:
        xcom_x, xcom_y = rotate_xy(series.xcom_x, series.xcom_y, rotate_ccw_deg=deg)
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
        xcom_x=xcom_x,
        xcom_y=xcom_y,
        xcom_valid_mask=series.xcom_valid_mask,
    )
    return DisplaySeries(
        rotate_ccw_deg=deg,
        com_x=com_x,
        com_y=com_y,
        xcom_x=xcom_x,
        xcom_y=xcom_y,
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
    mocap_frames: np.ndarray,
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

    mocap = np.asarray(mocap_frames, dtype=int)
    n_csv = int(mocap.size)
    n_c3d = int(c3d.points.shape[0])
    out_of_range_count = 0

    hull_x: list[np.ndarray] = []
    hull_y: list[np.ndarray] = []
    union_x: list[np.ndarray] = []
    union_y: list[np.ndarray] = []
    for t in range(n_csv):
        c3d_idx = int(mocap[t]) - 1
        if c3d_idx < 0 or c3d_idx >= n_c3d:
            out_of_range_count += 1
            hull_x.append(np.asarray([], dtype=float))
            hull_y.append(np.asarray([], dtype=float))
            union_x.append(np.asarray([], dtype=float))
            union_y.append(np.asarray([], dtype=float))
            continue

        pts_all = c3d.points[c3d_idx, idx_all, :2]
        poly_hull = convex_hull_2d(pts_all)
        hx, hy = _poly_to_closed_xy(poly_hull)
        hx, hy = rotate_xy(hx, hy, rotate_ccw_deg=rotate_ccw_deg)
        hull_x.append(hx)
        hull_y.append(hy)

        pts_l = c3d.points[c3d_idx, idx_l, :2]
        pts_r = c3d.points[c3d_idx, idx_r, :2]
        poly_l = convex_hull_2d(pts_l)
        poly_r = convex_hull_2d(pts_r)
        lx, ly = _poly_to_closed_xy(poly_l)
        rx, ry = _poly_to_closed_xy(poly_r)
        lx, ly = rotate_xy(lx, ly, rotate_ccw_deg=rotate_ccw_deg)
        rx, ry = rotate_xy(rx, ry, rotate_ccw_deg=rotate_ccw_deg)

        ux, uy = _join_polylines([(lx, ly), (rx, ry)])
        union_x.append(ux)
        union_y.append(uy)

    if out_of_range_count > 0:
        print(
            "Warning: C3D overlay frame mapping out-of-range "
            f"({out_of_range_count}/{n_csv}). "
            f"CSV MocapFrame range={int(mocap.min())}..{int(mocap.max())}, "
            f"C3D valid frame range=1..{n_c3d}"
        )

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


def build_gif_legend_handles(*, has_xcom: bool, show_step_ghost: bool) -> list[object]:
    handles: list[object] = [
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
    if show_step_ghost:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                linestyle="None",
                markerfacecolor="darkorange",
                markeredgecolor="black",
                label="Step-onset COM ghost",
            )
        )
    if has_xcom:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=XCOM_TRAIL_COLOR,
                    lw=2,
                    linestyle=":",
                    label="xCOM cumulative trajectory",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    linestyle="None",
                    markerfacecolor=XCOM_INSIDE_COLOR,
                    markeredgecolor="black",
                    label="Current xCOM (inside)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    linestyle="None",
                    markerfacecolor=XCOM_OUTSIDE_COLOR,
                    markeredgecolor="black",
                    label="Current xCOM (outside)",
                ),
            ]
        )
        if show_step_ghost:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    linestyle="None",
                    markerfacecolor=XCOM_GHOST_COLOR,
                    markeredgecolor="black",
                    label="Step-onset xCOM ghost",
                )
            )
    return handles


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


def apply_gif_right_panel(ax_side: plt.Axes, *, has_xcom: bool, show_step_ghost: bool) -> object:
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
        handles=build_gif_legend_handles(has_xcom=has_xcom, show_step_ghost=show_step_ghost),
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


def add_timeline_inset(
    ax_side: plt.Axes,
    series: TrialSeries,
) -> Line2D:
    """Add a horizontal timeline strip at the bottom of ax_side.

    Marks platform_onset, step_onset, platform_offset as vertical lines.
    Returns the animated cursor Line2D that must be updated each frame via
    cursor_line.set_xdata([frame_value, frame_value]).
    """
    ax_tl = ax_side.inset_axes([0.04, 0.37, 0.92, 0.12])
    first_frame = int(series.mocap_frame[0])
    last_frame = int(series.mocap_frame[-1])
    ax_tl.set_xlim(first_frame, last_frame)
    ax_tl.set_ylim(0.0, 1.0)
    ax_tl.axis("off")
    # Base bar
    ax_tl.axhline(0.5, color="0.55", linewidth=2.5, solid_capstyle="round")
    # Platform onset/offset
    ax_tl.axvline(series.platform_onset_local, color="0.40", linewidth=1.3, linestyle="--")
    ax_tl.axvline(series.platform_offset_local, color="0.40", linewidth=1.3, linestyle="--")
    ax_tl.text(series.platform_onset_local, 0.12, "on", fontsize=5,
               ha="center", va="bottom", color="0.40")
    ax_tl.text(series.platform_offset_local, 0.12, "off", fontsize=5,
               ha="center", va="bottom", color="0.40")
    # Step onset (orange)
    if series.step_onset_local is not None:
        ax_tl.axvline(int(series.step_onset_local), color="tab:orange", linewidth=2.5)
        ax_tl.text(int(series.step_onset_local), 0.85, "step", fontsize=5,
                   ha="center", va="top", color="tab:orange", fontweight="bold")
    # Animated cursor (blue vertical line)
    cursor_line = ax_tl.axvline(first_frame, color="tab:blue", linewidth=1.8, alpha=0.85)
    return cursor_line


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
    if (
        series.xcom_valid_mask is not None
        and display.xcom_x is not None
        and display.xcom_y is not None
    ):
        xcom_valid_idx = np.flatnonzero(series.xcom_valid_mask)
        if xcom_valid_idx.size > 0:
            x_parts.append(display.xcom_x[xcom_valid_idx])
            y_parts.append(display.xcom_y[xcom_valid_idx])
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


def get_xcom_point_for_frame(
    series: TrialSeries,
    display: DisplaySeries,
    event_frame: int | None,
) -> tuple[float, float] | None:
    if (
        event_frame is None
        or series.xcom_valid_mask is None
        or display.xcom_x is None
        or display.xcom_y is None
    ):
        return None
    idx = np.flatnonzero((series.mocap_frame == int(event_frame)) & series.xcom_valid_mask)
    if idx.size == 0:
        return None
    i = int(idx[0])
    return float(display.xcom_x[i]), float(display.xcom_y[i])


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
    has_xcom = (
        series.xcom_valid_mask is not None
        and series.xcom_inside_mask is not None
        and display.xcom_x is not None
        and display.xcom_y is not None
    )
    xcom_valid_idx = np.asarray([], dtype=int)
    if has_xcom:
        xcom_valid_idx = np.flatnonzero(series.xcom_valid_mask)
        if xcom_valid_idx.size == 0:
            has_xcom = False

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
    if has_xcom:
        ax.plot(
            display.xcom_x[xcom_valid_idx],
            display.xcom_y[xcom_valid_idx],
            color=XCOM_TRAIL_COLOR,
            linewidth=1.9,
            alpha=0.90,
            linestyle=":",
            label="xCOM trajectory",
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
    if has_xcom:
        xcom_inside_idx = np.flatnonzero(series.xcom_valid_mask & series.xcom_inside_mask)
        xcom_outside_idx = np.flatnonzero(series.xcom_valid_mask & (~series.xcom_inside_mask))
        if xcom_inside_idx.size:
            ax.scatter(
                display.xcom_x[xcom_inside_idx],
                display.xcom_y[xcom_inside_idx],
                s=20,
                marker="^",
                c=XCOM_INSIDE_COLOR,
                alpha=0.80,
                label="xCOM inside BOS",
                zorder=5,
            )
        if xcom_outside_idx.size:
            ax.scatter(
                display.xcom_x[xcom_outside_idx],
                display.xcom_y[xcom_outside_idx],
                s=22,
                marker="^",
                c=XCOM_OUTSIDE_COLOR,
                alpha=0.90,
                label="xCOM outside BOS",
                zorder=5,
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
    xcom_step_point = get_xcom_point_for_frame(series, display, series.step_onset_local)
    if xcom_step_point is not None:
        ax.scatter(
            [xcom_step_point[0]],
            [xcom_step_point[1]],
            s=98,
            marker="X",
            c=XCOM_GHOST_COLOR,
            edgecolors="white",
            linewidths=0.8,
            label="xCOM step_onset ghost",
            zorder=7,
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
    static_title = "BOS + COM/xCOM XY (static)" if has_xcom else "BOS + COM XY (static)"
    set_title_and_subtitle(
        ax,
        title=(
            f"{static_title} | "
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
    step_vis: str = "none",
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

    has_xcom = (
        series.xcom_valid_mask is not None
        and series.xcom_inside_mask is not None
        and display.xcom_x is not None
        and display.xcom_y is not None
    )
    xcom_valid_indices = np.asarray([], dtype=int)
    if has_xcom:
        xcom_valid_indices = np.flatnonzero(series.xcom_valid_mask)
        if xcom_valid_indices.size == 0:
            has_xcom = False

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
    xcom_trail_line: Line2D | None = None
    xcom_current_point: Line2D | None = None
    if has_xcom:
        xcom_trail_line, = ax.plot(
            [],
            [],
            color=XCOM_TRAIL_COLOR,
            linewidth=2.0,
            alpha=0.9,
            linestyle=":",
            zorder=3,
        )
        xcom_current_point, = ax.plot(
            [],
            [],
            marker="^",
            linestyle="None",
            markersize=8.5,
            markerfacecolor=XCOM_INSIDE_COLOR,
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
    # ---- step_vis template setup ----
    # Always compute step_onset_idx independently (bos_freeze_idx is only set for freeze mode)
    step_onset_idx: int | None = None
    if series.step_onset_local is not None:
        _step_frame = int(series.step_onset_local)
        _exact = np.flatnonzero((series.mocap_frame == _step_frame) & series.valid_mask)
        if _exact.size > 0:
            step_onset_idx = int(_exact[0])
        else:
            _after = np.flatnonzero(series.valid_mask & (series.mocap_frame >= _step_frame))
            if _after.size > 0:
                step_onset_idx = int(_after[0])

    show_step_ghost = mode == "live" and step_onset_idx is not None
    info_text = apply_gif_right_panel(
        ax_side,
        has_xcom=has_xcom,
        show_step_ghost=show_step_ghost,
    )

    timeline_cursor: Line2D | None = None
    if step_vis != "none":
        timeline_cursor = add_timeline_inset(ax_side, series)

    trail_pre: object | None = None
    trail_post: object | None = None
    xcom_trail_pre: object | None = None
    xcom_trail_post: object | None = None
    if step_vis in ("phase_trail", "phase_bos"):
        trail_line.set_visible(False)
        (trail_pre,) = ax.plot([], [], color="tab:blue", linewidth=2.0, alpha=0.95, zorder=3)
        (trail_post,) = ax.plot([], [], color="tab:orange", linewidth=2.0, alpha=0.95, zorder=3)
        if xcom_trail_line is not None:
            xcom_trail_line.set_visible(False)
            (xcom_trail_pre,) = ax.plot(
                [],
                [],
                color=XCOM_TRAIL_COLOR,
                linewidth=2.0,
                alpha=0.9,
                linestyle=":",
                zorder=3,
            )
            (xcom_trail_post,) = ax.plot(
                [],
                [],
                color=XCOM_OUTSIDE_COLOR,
                linewidth=2.0,
                alpha=0.9,
                linestyle=":",
                zorder=3,
            )
    # ---- end step_vis setup ----

    # ---- ghost snapshot setup (live mode only) ----
    ghost_bos_rect: Rectangle | None = None
    ghost_com_pt = None
    ghost_xcom_pt = None
    ghost_label = None
    ghost_bos_union_line = None
    ghost_bos_hull_line = None
    if show_step_ghost:
        ghost_bos_rect = Rectangle(
            (0.0, 0.0),
            width=0.0,
            height=0.0,
            facecolor="none",
            edgecolor="darkorange",
            alpha=0.0,
            linewidth=2.0,
            linestyle="--",
            zorder=6,
        )
        ax.add_patch(ghost_bos_rect)
        (ghost_com_pt,) = ax.plot(
            [],
            [],
            marker="X",
            linestyle="None",
            markersize=9,
            markerfacecolor="darkorange",
            markeredgecolor="black",
            markeredgewidth=0.8,
            alpha=0.0,
            zorder=7,
        )
        if has_xcom:
            (ghost_xcom_pt,) = ax.plot(
                [],
                [],
                marker="X",
                linestyle="None",
                markersize=9,
                markerfacecolor=XCOM_GHOST_COLOR,
                markeredgecolor="black",
                markeredgewidth=0.8,
                alpha=0.0,
                zorder=7,
            )
        ghost_label = ax.text(
            0.0,
            0.0,
            "",
            fontsize=6.5,
            color="darkorange",
            ha="left",
            va="bottom",
            zorder=8,
            visible=False,
        )
        if bos_polylines is not None:
            (ghost_bos_union_line,) = ax.plot(
                [],
                [],
                color="tab:purple",
                linewidth=1.0,
                alpha=0.0,
                linestyle="--",
                zorder=5,
            )
            (ghost_bos_hull_line,) = ax.plot(
                [],
                [],
                color="0.45",
                linewidth=1.0,
                alpha=0.0,
                linestyle=":",
                zorder=5,
            )
    # ---- end ghost setup ----

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.55)
    ax.set_xlabel("X (m) [- Left / + Right]")
    ax.set_ylabel("Y (m) [+ Anterior / - Posterior]")
    gif_trial_state_line = resolve_gif_trial_state_line(trial_state_label)
    main_title = "BOS + COM/xCOM XY animation" if has_xcom else "BOS + COM XY animation"
    set_title_and_subtitle(
        ax,
        title=main_title,
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
        xcom_history = xcom_valid_indices[xcom_valid_indices <= idx] if has_xcom else np.asarray([], dtype=int)

        trail_line.set_data(display.com_x[history], display.com_y[history])
        cx = float(display.com_x[idx])
        cy = float(display.com_y[idx])
        current_point.set_data([cx], [cy])

        is_inside = bool(series.inside_mask[idx])
        color = "tab:green" if is_inside else "tab:red"
        current_point.set_markerfacecolor(color)
        current_point.set_markeredgecolor("black")

        if has_xcom and xcom_trail_line is not None and display.xcom_x is not None and display.xcom_y is not None:
            xcom_trail_line.set_data(display.xcom_x[xcom_history], display.xcom_y[xcom_history])
        if has_xcom and xcom_current_point is not None and series.xcom_valid_mask is not None:
            if bool(series.xcom_valid_mask[idx]) and display.xcom_x is not None and display.xcom_y is not None:
                xcx = float(display.xcom_x[idx])
                xcy = float(display.xcom_y[idx])
                xcom_current_point.set_data([xcx], [xcy])
                xcom_current_point.set_alpha(1.0)
                xcom_inside = bool(series.xcom_inside_mask[idx]) if series.xcom_inside_mask is not None else True
                xcom_color = XCOM_INSIDE_COLOR if xcom_inside else XCOM_OUTSIDE_COLOR
                xcom_current_point.set_markerfacecolor(xcom_color)
                xcom_current_point.set_markeredgecolor("black")
            else:
                xcom_current_point.set_data([], [])
                xcom_current_point.set_alpha(0.0)

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

        local_idx = frame_no + 1
        time_info = f"frame_local={local_idx}/{frame_indices.size}"
        if series.time_from_onset_s is not None and np.isfinite(series.time_from_onset_s[idx]):
            time_info = f"t={series.time_from_onset_s[idx]:.3f} s"

        info_text.set_text(
            f"{panel_header}\n\n"
            f"frame_local={local_idx}/{frame_indices.size}\n"
            f"{time_info}\n"
            f"status={'inside' if is_inside else 'outside'}\n"
            f"event={event_state(frame_value)}\n"
            f"bos={bos_state}\n"
            f"inside ratio={inside_ratio:.1f}% ({inside_count}/{valid_count})\n"
            f"outside={outside_count}"
        )

        # ---- step_vis per-frame updates ----
        if timeline_cursor is not None:
            timeline_cursor.set_xdata([frame_value, frame_value])

        if trail_pre is not None and trail_post is not None:
            pre_hist = history[history <= step_onset_idx] if step_onset_idx is not None else history
            post_hist = history[history > step_onset_idx] if step_onset_idx is not None else np.array([], dtype=int)
            trail_pre.set_data(display.com_x[pre_hist], display.com_y[pre_hist])
            trail_post.set_data(display.com_x[post_hist], display.com_y[post_hist])
        if (
            xcom_trail_pre is not None
            and xcom_trail_post is not None
            and has_xcom
            and display.xcom_x is not None
            and display.xcom_y is not None
        ):
            xcom_pre = xcom_history[xcom_history <= step_onset_idx] if step_onset_idx is not None else xcom_history
            xcom_post = (
                xcom_history[xcom_history > step_onset_idx]
                if step_onset_idx is not None
                else np.array([], dtype=int)
            )
            xcom_trail_pre.set_data(display.xcom_x[xcom_pre], display.xcom_y[xcom_pre])
            xcom_trail_post.set_data(display.xcom_x[xcom_post], display.xcom_y[xcom_post])

        if step_vis in ("bos_phase", "phase_bos") and step_onset_idx is not None:
            if idx == step_onset_idx:
                # 정확히 step_onset 프레임: 강한 빨강으로 flash
                bos_rect.set_facecolor("tomato")
                bos_rect.set_edgecolor("red")
                bos_rect.set_alpha(0.65)
                bos_rect.set_linewidth(3.0)
            elif idx > step_onset_idx:
                # step_onset 이후: 주황색 유지
                bos_rect.set_facecolor("moccasin")
                bos_rect.set_edgecolor("tab:orange")
                bos_rect.set_alpha(0.40)
                bos_rect.set_linewidth(2.0)
            else:
                # step_onset 이전: 기본 파란색
                bos_rect.set_facecolor("lightskyblue")
                bos_rect.set_edgecolor("tab:blue")
                bos_rect.set_alpha(0.25)
                bos_rect.set_linewidth(1.2)
        # ---- end step_vis per-frame ----

        # ---- ghost snapshot per-frame ----
        if ghost_bos_rect is not None and ghost_com_pt is not None and step_onset_idx is not None:
            if idx >= step_onset_idx:
                g_minx = float(display.bos_minx[step_onset_idx])
                g_maxx = float(display.bos_maxx[step_onset_idx])
                g_miny = float(display.bos_miny[step_onset_idx])
                g_maxy = float(display.bos_maxy[step_onset_idx])
                ghost_bos_rect.set_xy((g_minx, g_miny))
                ghost_bos_rect.set_width(g_maxx - g_minx)
                ghost_bos_rect.set_height(g_maxy - g_miny)
                ghost_bos_rect.set_alpha(0.75)
                g_cx = float(display.com_x[step_onset_idx])
                g_cy = float(display.com_y[step_onset_idx])
                ghost_com_pt.set_data([g_cx], [g_cy])
                ghost_com_pt.set_alpha(1.0)
                if (
                    ghost_xcom_pt is not None
                    and series.xcom_valid_mask is not None
                    and bool(series.xcom_valid_mask[step_onset_idx])
                    and display.xcom_x is not None
                    and display.xcom_y is not None
                ):
                    gx_xcom = float(display.xcom_x[step_onset_idx])
                    gy_xcom = float(display.xcom_y[step_onset_idx])
                    ghost_xcom_pt.set_data([gx_xcom], [gy_xcom])
                    ghost_xcom_pt.set_alpha(1.0)
                elif ghost_xcom_pt is not None:
                    ghost_xcom_pt.set_data([], [])
                    ghost_xcom_pt.set_alpha(0.0)
                if ghost_label is not None:
                    ghost_label.set_position((g_cx + 0.01, g_cy + 0.01))
                    ghost_label.set_text(f"step@{int(series.mocap_frame[step_onset_idx])}")
                    ghost_label.set_visible(True)
                if ghost_bos_union_line is not None and bos_polylines is not None:
                    ghost_bos_union_line.set_data(
                        bos_polylines.union_x[step_onset_idx],
                        bos_polylines.union_y[step_onset_idx],
                    )
                    ghost_bos_union_line.set_alpha(0.5)
                if ghost_bos_hull_line is not None and bos_polylines is not None:
                    ghost_bos_hull_line.set_data(
                        bos_polylines.hull_x[step_onset_idx],
                        bos_polylines.hull_y[step_onset_idx],
                    )
                    ghost_bos_hull_line.set_alpha(0.5)
            else:
                ghost_bos_rect.set_alpha(0.0)
                ghost_com_pt.set_alpha(0.0)
                if ghost_xcom_pt is not None:
                    ghost_xcom_pt.set_alpha(0.0)
                if ghost_label is not None:
                    ghost_label.set_visible(False)
                if ghost_bos_union_line is not None:
                    ghost_bos_union_line.set_alpha(0.0)
                if ghost_bos_hull_line is not None:
                    ghost_bos_hull_line.set_alpha(0.0)
        # ---- end ghost per-frame ----

        artists: list[object] = [trail_line, current_point, bos_rect, info_text]
        if xcom_trail_line is not None:
            artists.append(xcom_trail_line)
        if xcom_current_point is not None:
            artists.append(xcom_current_point)
        if bos_union_line is not None and bos_hull_line is not None:
            artists.extend([bos_union_line, bos_hull_line])
        if trail_pre is not None:
            artists.append(trail_pre)
        if trail_post is not None:
            artists.append(trail_post)
        if xcom_trail_pre is not None:
            artists.append(xcom_trail_pre)
        if xcom_trail_post is not None:
            artists.append(xcom_trail_post)
        if timeline_cursor is not None:
            artists.append(timeline_cursor)
        if ghost_bos_rect is not None:
            artists.append(ghost_bos_rect)
        if ghost_com_pt is not None:
            artists.append(ghost_com_pt)
        if ghost_xcom_pt is not None:
            artists.append(ghost_xcom_pt)
        if ghost_label is not None:
            artists.append(ghost_label)
        if ghost_bos_union_line is not None:
            artists.append(ghost_bos_union_line)
        if ghost_bos_hull_line is not None:
            artists.append(ghost_bos_hull_line)
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


def select_trial_df(df: pl.DataFrame, key: TrialKey) -> pl.DataFrame:
    trial_df = df.filter(
        (pl.col("subject") == key.subject)
        & (pl.col("velocity") == float(key.velocity))
        & (pl.col("trial") == int(key.trial))
    )
    if trial_df.height == 0:
        raise ValueError(
            "Selected trial has no rows: "
            f"subject={key.subject}, velocity={format_velocity(key.velocity)}, trial={key.trial}"
        )
    return trial_df


def build_render_config(args: argparse.Namespace, rotate_ccw_deg: int) -> RenderConfig:
    start_offset = None
    if args.start_from_platform_onset_offset is not None:
        start_offset = int(args.start_from_platform_onset_offset)
    return RenderConfig(
        csv_path=Path(args.csv),
        event_xlsm=Path(args.event_xlsm),
        out_dir=Path(args.out_dir),
        c3d_dir=resolve_repo_path(DEFAULT_C3D_DIR),
        fps=int(args.fps),
        frame_step=int(args.frame_step),
        dpi=int(args.dpi),
        gif_name_suffix=str(args.gif_name_suffix),
        rotate_ccw_deg=int(rotate_ccw_deg),
        show_trial_state=bool(args.show_trial_state),
        start_from_platform_onset_offset=start_offset,
        step_vis=str(args.step_vis),
    )


def render_one_trial(
    *,
    df: pl.DataFrame,
    config: RenderConfig,
    key: TrialKey,
    auto_selected: bool,
    verbose: bool,
) -> TrialRenderResult:
    started_at = time.perf_counter()
    trial_df = select_trial_df(df, key)
    if config.start_from_platform_onset_offset is not None:
        platform_onset = get_optional_int_scalar(trial_df, "platform_onset_local")
        if platform_onset is None:
            raise ValueError("platform_onset_local is required when using --start_from_platform_onset_offset.")
        start_frame = int(platform_onset + int(config.start_from_platform_onset_offset))
        trial_df = trial_df.filter(pl.col("MocapFrame") >= start_frame)
        if trial_df.is_empty():
            raise ValueError(
                "No rows remain after start trim: "
                f"start_frame={start_frame}, subject={key.subject}, "
                f"velocity={format_velocity(key.velocity)}, trial={key.trial}"
            )
        if verbose:
            print(
                "Start trim: "
                f"platform_onset_local={platform_onset}, "
                f"offset={config.start_from_platform_onset_offset}, "
                f"start_frame={start_frame}"
            )

    series = build_trial_series(
        trial_df=trial_df,
        subject=key.subject,
        velocity=key.velocity,
        trial=key.trial,
    )
    display = build_display_series(series, rotate_ccw_deg=config.rotate_ccw_deg)
    trial_state = resolve_trial_state(
        event_xlsm=config.event_xlsm,
        subject=key.subject,
        velocity=float(key.velocity),
        trial=int(key.trial),
    )
    trial_state_label = format_trial_state_label(trial_state) if bool(config.show_trial_state) else None

    base_name = (
        f"{safe_name(key.subject)}__velocity-{safe_name(format_velocity(key.velocity))}"
        f"__trial-{int(key.trial)}"
    )
    subject_out_dir = config.out_dir / str(key.subject)
    subject_out_dir.mkdir(parents=True, exist_ok=True)
    gif_base = subject_out_dir / f"{base_name}__{safe_name(config.gif_name_suffix)}"

    if verbose:
        print(
            "Trial selection: "
            f"subject={key.subject}, velocity={format_velocity(key.velocity)}, "
            f"trial={key.trial}, auto_selected={auto_selected}"
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
        print(f"Display rotation: CCW {config.rotate_ccw_deg} deg")
        print(f"Trial state: {trial_state}")
        print(f"Output root: {config.out_dir}")
        print(f"Subject output directory: {subject_out_dir}")

    bos_polylines: BOSPolylines | None = None
    try:
        c3d_path = resolve_c3d_for_trial(
            c3d_dir=config.c3d_dir,
            event_xlsm=config.event_xlsm,
            subject=key.subject,
            velocity=float(key.velocity),
            trial=int(key.trial),
        )
        if c3d_path is None:
            if verbose:
                print("[BOS overlay] matching C3D not found; hull/union overlay disabled.")
        else:
            bos_polylines = compute_bos_polylines_from_c3d(
                c3d_path=c3d_path,
                mocap_frames=series.mocap_frame,
                rotate_ccw_deg=display.rotate_ccw_deg,
            )
            if verbose:
                print(f"[BOS overlay] using C3D: {c3d_path}")
    except Exception as exc:
        print(f"[BOS overlay] disabled due to error: {exc}")
        bos_polylines = None

    fixed_x_lim, fixed_y_lim = compute_fixed_gif_axis_limits(
        series=series,
        display=display,
        bos_polylines=bos_polylines,
    )
    if verbose:
        print(
            "Fixed axis limits: "
            f"x=({fixed_x_lim[0]:.4f}, {fixed_x_lim[1]:.4f}), "
            f"y=({fixed_y_lim[0]:.4f}, {fixed_y_lim[1]:.4f})"
        )

    step_vis = config.step_vis
    vis_list: list[str] = list(STEP_VIS_TEMPLATES) if step_vis == "all" else [step_vis]

    gif_outputs: list[tuple[str, int, str]] = []
    if verbose:
        print(f"GIF outputs: step_vis={step_vis}, bos_mode=freeze/live")
    for sv in vis_list:
        # step_vis 템플릿 모드는 live만 렌더링 (freeze는 기존 none 모드에서 사용)
        bos_modes_for_sv = GIF_BOS_MODES if sv == "none" else ("live",)
        for bos_mode in bos_modes_for_sv:
            if sv == "none":
                gif_out = Path(f"{gif_base}__{RIGHT1COL_SUFFIX}__{bos_mode}.gif")
            else:
                gif_out = gif_base.parent / f"{gif_base.name}__step_vis-{sv}__{bos_mode}.gif"
            frames = render_gif(
                series=series,
                display=display,
                trial_state_label=trial_state_label,
                out_path=gif_out,
                fps=int(config.fps),
                frame_step=int(config.frame_step),
                dpi=int(config.dpi),
                x_lim=fixed_x_lim,
                y_lim=fixed_y_lim,
                bos_polylines=bos_polylines,
                bos_mode=bos_mode,
                step_vis=sv,
            )
            gif_outputs.append((str(gif_out), int(frames), bos_mode))

    valid_count = int(np.count_nonzero(series.valid_mask))
    inside_count = int(np.count_nonzero(series.valid_mask & series.inside_mask))
    outside_count = valid_count - inside_count
    inside_ratio = (100.0 * inside_count / valid_count) if valid_count > 0 else float("nan")
    if verbose:
        print(
            "Inside/outside summary: "
            f"inside={inside_count}, outside={outside_count}, inside_ratio={inside_ratio:.2f}%"
        )
        for out_path, frames, bos_mode in gif_outputs:
            print(
                f"Saved GIF[{bos_mode}]: {out_path} "
                f"(frames={frames}, fps={config.fps}, frame_step={config.frame_step})"
            )

    return TrialRenderResult(
        key=key,
        gif_outputs=tuple(gif_outputs),
        valid_count=valid_count,
        inside_count=inside_count,
        outside_count=outside_count,
        inside_ratio=float(inside_ratio),
        elapsed_sec=float(time.perf_counter() - started_at),
    )


def render_one_trial_worker(config: RenderConfig, key: TrialKey) -> TrialRenderResult:
    df = load_data_for_worker(config.csv_path)
    return render_one_trial(
        df=df,
        config=config,
        key=key,
        auto_selected=False,
        verbose=False,
    )


def run_all_trials(
    *,
    config: RenderConfig,
    trial_keys: list[TrialKey],
    jobs: int | None,
) -> tuple[list[TrialRenderResult], float]:
    total_trials = len(trial_keys)
    if total_trials == 0:
        raise ValueError("No trials found in CSV.")

    max_workers = resolve_jobs(jobs)
    print(
        "Batch mode enabled: "
        f"trials={total_trials}, workers={max_workers}, bos_modes={','.join(GIF_BOS_MODES)}"
    )
    started_at = time.perf_counter()
    pending = deque(trial_keys)
    inflight: dict[cf.Future[TrialRenderResult], TrialKey] = {}
    results: list[TrialRenderResult] = []

    executor = cf.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
    )
    aborted = False
    try:
        while pending and len(inflight) < max_workers:
            key = pending.popleft()
            future = executor.submit(render_one_trial_worker, config, key)
            inflight[future] = key

        while inflight:
            done_set, _ = cf.wait(tuple(inflight.keys()), return_when=cf.FIRST_COMPLETED)
            for done in done_set:
                key = inflight.pop(done)
                try:
                    result = done.result()
                except Exception as exc:
                    aborted = True
                    for future in inflight:
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(
                        "Batch aborted on first error at "
                        f"subject={key.subject}, velocity={format_velocity(key.velocity)}, trial={key.trial}"
                    ) from exc

                results.append(result)
                print(
                    f"[{len(results)}/{total_trials}] "
                    f"subject={result.key.subject}, velocity={format_velocity(result.key.velocity)}, "
                    f"trial={result.key.trial}, gifs={len(result.gif_outputs)}, "
                    f"elapsed={result.elapsed_sec:.2f}s"
                )
                if pending:
                    next_key = pending.popleft()
                    next_future = executor.submit(render_one_trial_worker, config, next_key)
                    inflight[next_future] = next_key
    finally:
        if not aborted:
            executor.shutdown(wait=True, cancel_futures=False)

    total_elapsed = float(time.perf_counter() - started_at)
    results.sort(key=lambda item: (item.key.subject, item.key.velocity, item.key.trial))
    return results, total_elapsed


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
    config = build_render_config(args, rotate_ccw_deg=rotate_ccw_deg)

    selection_flags = [args.subject is not None, args.velocity is not None, args.trial is not None]
    if bool(args.all) and any(selection_flags):
        raise ValueError("--all cannot be combined with --subject/--velocity/--trial.")

    if bool(args.all):
        trial_keys = collect_trial_keys(df)
        results, total_elapsed = run_all_trials(
            config=config,
            trial_keys=trial_keys,
            jobs=args.jobs,
        )
        total_gifs = sum(len(item.gif_outputs) for item in results)
        avg_elapsed = (total_elapsed / len(results)) if results else float("nan")
        print(
            "Batch summary: "
            f"trials={len(results)}, total_gifs={total_gifs}, total_elapsed_sec={total_elapsed:.2f}, "
            f"avg_elapsed_per_trial_sec={avg_elapsed:.2f}"
        )
        return

    trial_key, auto_selected = resolve_trial_selection(args, df)
    render_one_trial(
        df=df,
        config=config,
        key=trial_key,
        auto_selected=auto_selected,
        verbose=True,
    )


if __name__ == "__main__":
    main()
