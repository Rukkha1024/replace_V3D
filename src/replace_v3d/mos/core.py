from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..geometry.geometry2d import (
    convex_hull_2d,
    polygon_area,
    polygon_bounds,
    signed_min_distance_point_to_polygon,
)


BOS_MARKERS_DEFAULT = [
    "LHEE",
    "LTOE",
    "LANK",
    "LFoot_3",
    "RHEE",
    "RTOE",
    "RANK",
    "RFoot_3",
]


@dataclass(frozen=True)
class MOSResult:
    """MoS / BoS time series container.

    Notes
    -----
    Visual3D's MoS tutorial ("Original definition") uses the distance from xCoM to
    the *closest* BoS boundary at each frame (via the Closest_Bound meta-command).

    Your old implementation used a *velocity-sign switch* (pick minX vs maxX based
    on vCOM sign). That is not Visual3D's logic, and it produces step-like jumps
    whenever vCOM crosses 0 (which can happen frequently due to filtering noise).

    This repo now exposes:

    - MOS_AP_v3d / MOS_ML_v3d / MOS_v3d:
        Visual3D-style closest-bound distances (recommended).

    - MOS_AP_dir / MOS_ML_dir:
        Backward-compatible aliases of the Visual3D-style values above.
        (Kept because downstream scripts/plots often referenced "*_dir".)

    - MOS_AP_velDir / MOS_ML_velDir:
        The previous velocity-switching version (debug only).

    - MOS_signed:
        Polygon-based signed min distance to the convex hull boundary (+inside).
        This is *not* the same as Visual3D's original scalar MoS, but can be
        useful for convex-hull distance analysis.
    """

    # Polygon signed min distance (convex hull boundary)
    MOS_signed: np.ndarray  # (T,)

    # Visual3D (original) style: closest bound in each direction
    MOS_AP_v3d: np.ndarray  # (T,)
    MOS_ML_v3d: np.ndarray  # (T,)
    MOS_v3d: np.ndarray  # (T,)

    # Backward-compatible aliases (same as MOS_*_v3d)
    MOS_AP_dir: np.ndarray  # (T,)
    MOS_ML_dir: np.ndarray  # (T,)

    # Legacy: velocity-direction based bound selection (debug; discontinuous)
    MOS_AP_velDir: np.ndarray  # (T,)
    MOS_ML_velDir: np.ndarray  # (T,)

    # BoS geometry summaries
    BOS_area: np.ndarray  # (T,)
    BOS_minX: np.ndarray  # (T,)
    BOS_maxX: np.ndarray  # (T,)
    BOS_minY: np.ndarray  # (T,)
    BOS_maxY: np.ndarray  # (T,)


def _closest_bound_1d(x: float, xmin: float, xmax: float) -> float:
    """Signed distance to the closest of two bounds.

    Returns
    -------
    d : float
        Positive if xmin <= x <= xmax, negative otherwise.

    Notes
    -----
    This matches Visual3D's "Closest_Bound" behavior for two candidate bounds
    at each frame (take the smaller of the two distances).
    """

    return min(x - xmin, xmax - x)


def compute_mos_timeseries(
    points: np.ndarray,
    labels: List[str],
    xcom: np.ndarray,
    vcom: np.ndarray,
    end_frame: Optional[int] = None,
    bos_markers: List[str] = BOS_MARKERS_DEFAULT,
) -> MOSResult:
    """Compute MoS for each frame.

    BoS
    ---
    - BoS polygon is the convex hull of foot-landmark markers (ground plane: X-Y)
    - Also returns axis-aligned bounds (minX/maxX/minY/maxY) of that hull

    MoS outputs
    ---
    - MOS_signed: signed min distance from xCOM(XY) to hull boundary (+inside)

    - MOS_AP_v3d / MOS_ML_v3d:
        Visual3D "original" style closest-bound distance.

    - MOS_v3d:
        min(MOS_AP_v3d, MOS_ML_v3d)  (closest boundary overall)

    - MOS_AP_dir / MOS_ML_dir:
        Aliases of MOS_AP_v3d / MOS_ML_v3d (for backward compatibility)

    - MOS_AP_velDir / MOS_ML_velDir:
        Legacy velocity-direction switching (debug only)

    Parameters
    ----------
    points:
        (T, N, 3)
    labels:
        list of marker labels of length N
    xcom:
        (T, 3)
    vcom:
        (T, 3)
    end_frame:
        If provided, compute only frames [1..end_frame] (1-based)
    bos_markers:
        Marker names used to form the BoS hull.

    Returns
    -------
    MOSResult
    """

    T = points.shape[0]
    if end_frame is None:
        end = T
    else:
        end = max(1, min(int(end_frame), T))

    # marker indices for BoS
    idx = []
    for m in bos_markers:
        if m not in labels:
            raise KeyError(f"BoS marker not found: {m}")
        idx.append(labels.index(m))

    MOS_signed = np.zeros(end, dtype=float)

    MOS_AP_v3d = np.zeros(end, dtype=float)
    MOS_ML_v3d = np.zeros(end, dtype=float)
    MOS_v3d = np.zeros(end, dtype=float)

    # Backward-compat aliases (filled at the end)
    MOS_AP_dir = np.zeros(end, dtype=float)
    MOS_ML_dir = np.zeros(end, dtype=float)

    # Legacy velocity-switching (debug)
    MOS_AP_velDir = np.zeros(end, dtype=float)
    MOS_ML_velDir = np.zeros(end, dtype=float)

    area = np.zeros(end, dtype=float)
    minX = np.zeros(end, dtype=float)
    maxX = np.zeros(end, dtype=float)
    minY = np.zeros(end, dtype=float)
    maxY = np.zeros(end, dtype=float)

    for t in range(end):
        pts_xy = points[t, idx, :2]  # (M,2)

        hull = convex_hull_2d(pts_xy)
        area[t] = polygon_area(hull)
        bminx, bmaxx, bminy, bmaxy = polygon_bounds(hull)
        minX[t], maxX[t], minY[t], maxY[t] = bminx, bmaxx, bminy, bmaxy

        p = xcom[t, :2]
        MOS_signed[t] = signed_min_distance_point_to_polygon(p, hull)

        # Visual3D "original" style = closest bound (no velocity switching)
        # NOTE: In this repo's coordinate setup, X is A/P and Y is M/L (see plan.md).
        MOS_AP_v3d[t] = _closest_bound_1d(float(p[0]), bminx, bmaxx)
        MOS_ML_v3d[t] = _closest_bound_1d(float(p[1]), bminy, bmaxy)
        MOS_v3d[t] = min(MOS_AP_v3d[t], MOS_ML_v3d[t])

        # Legacy: velocity-direction switching (debug only)
        if vcom[t, 0] < 0:
            MOS_AP_velDir[t] = p[0] - bminx
        else:
            MOS_AP_velDir[t] = bmaxx - p[0]

        if vcom[t, 1] < 0:
            MOS_ML_velDir[t] = p[1] - bminy
        else:
            MOS_ML_velDir[t] = bmaxy - p[1]

    # Backward-compat aliases
    MOS_AP_dir[:] = MOS_AP_v3d
    MOS_ML_dir[:] = MOS_ML_v3d

    return MOSResult(
        MOS_signed=MOS_signed,
        MOS_AP_v3d=MOS_AP_v3d,
        MOS_ML_v3d=MOS_ML_v3d,
        MOS_v3d=MOS_v3d,
        MOS_AP_dir=MOS_AP_dir,
        MOS_ML_dir=MOS_ML_dir,
        MOS_AP_velDir=MOS_AP_velDir,
        MOS_ML_velDir=MOS_ML_velDir,
        BOS_area=area,
        BOS_minX=minX,
        BOS_maxX=maxX,
        BOS_minY=minY,
        BOS_maxY=maxY,
    )
