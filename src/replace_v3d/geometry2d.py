"""Compatibility wrapper for 2D geometry utilities.

Implementation lives in `replace_v3d.geometry.geometry2d`.
"""

from __future__ import annotations

from .geometry.geometry2d import (
    convex_hull_2d,
    distance_point_to_segment,
    point_in_convex_polygon,
    polygon_area,
    polygon_bounds,
    signed_min_distance_point_to_polygon,
)

__all__ = [
    "convex_hull_2d",
    "polygon_area",
    "point_in_convex_polygon",
    "distance_point_to_segment",
    "signed_min_distance_point_to_polygon",
    "polygon_bounds",
]

