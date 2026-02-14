from __future__ import annotations

from typing import Tuple

import numpy as np


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute 2D convex hull (Monotone chain).

    Parameters
    ----------
    points : (N,2) array

    Returns
    -------
    hull : (M,2) array in CCW order, M>=1
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return pts

    # unique
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts

    # sort by x, then y
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def polygon_area(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_convex_polygon(p: np.ndarray, poly: np.ndarray, eps: float = 1e-12) -> bool:
    """Check if point is inside/on a convex polygon given in CCW order."""
    p = np.asarray(p, dtype=float)
    poly = np.asarray(poly, dtype=float)
    if len(poly) == 0:
        return False
    if len(poly) == 1:
        return np.linalg.norm(p - poly[0]) <= eps
    if len(poly) == 2:
        # on segment
        a, b = poly
        return distance_point_to_segment(p, a, b) <= eps

    def cross2(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        if cross2(a, b, p) < -eps:
            return False
    return True


def distance_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    ap = p - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-20:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def signed_min_distance_point_to_polygon(p: np.ndarray, poly: np.ndarray) -> float:
    """Signed min distance to polygon boundary.

    Positive if inside/on polygon, negative if outside.
    """
    p = np.asarray(p, dtype=float)
    poly = np.asarray(poly, dtype=float)

    if len(poly) == 0:
        return float("nan")
    if len(poly) == 1:
        d = float(np.linalg.norm(p - poly[0]))
        return -d
    if len(poly) == 2:
        d = distance_point_to_segment(p, poly[0], poly[1])
        # treat as outside unless exactly on segment
        return d if d == 0 else -d

    # distance to edges
    dmin = float("inf")
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        d = distance_point_to_segment(p, a, b)
        if d < dmin:
            dmin = d

    inside = point_in_convex_polygon(p, poly)
    return dmin if inside else -dmin


def polygon_bounds(poly: np.ndarray) -> Tuple[float, float, float, float]:
    poly = np.asarray(poly, dtype=float)
    if len(poly) == 0:
        return (float("nan"),) * 4
    return float(np.min(poly[:, 0])), float(np.max(poly[:, 0])), float(np.min(poly[:, 1])), float(np.max(poly[:, 1]))

