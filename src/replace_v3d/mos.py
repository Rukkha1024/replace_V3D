from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .geometry2d import (
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
    MOS_signed: np.ndarray  # (T,)
    MOS_AP_dir: np.ndarray  # (T,)
    MOS_ML_dir: np.ndarray  # (T,)
    BOS_area: np.ndarray  # (T,)
    BOS_minX: np.ndarray  # (T,)
    BOS_maxX: np.ndarray  # (T,)
    BOS_minY: np.ndarray  # (T,)
    BOS_maxY: np.ndarray  # (T,)


def compute_mos_timeseries(
    points: np.ndarray,
    labels: List[str],
    xcom: np.ndarray,
    vcom: np.ndarray,
    end_frame: Optional[int] = None,
    bos_markers: List[str] = BOS_MARKERS_DEFAULT,
) -> MOSResult:
    """Compute MoS for each frame using:

    - xCOM (Hof) as the test point
    - BoS as the convex hull of foot-landmark markers (ground plane: X-Y)
    - MoS = signed minimum distance to the BoS boundary

    Parameters
    ----------
    points: (T, N, 3)
    xcom:   (T, 3)
    vcom:   (T, 3)
    end_frame: if provided, compute only frames [1..end_frame] (1-based)
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
    MOS_AP = np.zeros(end, dtype=float)
    MOS_ML = np.zeros(end, dtype=float)
    area = np.zeros(end, dtype=float)
    minX = np.zeros(end, dtype=float)
    maxX = np.zeros(end, dtype=float)
    minY = np.zeros(end, dtype=float)
    maxY = np.zeros(end, dtype=float)

    for t in range(end):
        pts_xy = points[t, idx, :2]  # (8,2)

        hull = convex_hull_2d(pts_xy)
        area[t] = polygon_area(hull)
        bminx, bmaxx, bminy, bmaxy = polygon_bounds(hull)
        minX[t], maxX[t], minY[t], maxY[t] = bminx, bmaxx, bminy, bmaxy

        p = xcom[t, :2]
        MOS_signed[t] = signed_min_distance_point_to_polygon(p, hull)

        # Directional margins (optional but useful): boundary in direction of COM velocity
        # X axis: A/P, negative X = forward in your coordinate description.
        if vcom[t, 0] < 0:
            MOS_AP[t] = p[0] - bminx
        else:
            MOS_AP[t] = bmaxx - p[0]

        # Y axis: M/L (sign depends on setup; keep consistent with raw data)
        if vcom[t, 1] < 0:
            MOS_ML[t] = p[1] - bminy
        else:
            MOS_ML[t] = bmaxy - p[1]

    return MOSResult(
        MOS_signed=MOS_signed,
        MOS_AP_dir=MOS_AP,
        MOS_ML_dir=MOS_ML,
        BOS_area=area,
        BOS_minX=minX,
        BOS_maxX=maxX,
        BOS_minY=minY,
        BOS_maxY=maxY,
    )

