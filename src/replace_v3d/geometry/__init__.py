"""Geometry helpers."""

from .geometry2d import (
    convex_hull_2d,
    distance_point_to_segment,
    point_in_convex_polygon,
    polygon_area,
    polygon_bounds,
    signed_min_distance_point_to_polygon,
)

__all__ = [
    "convex_hull_2d",
    "distance_point_to_segment",
    "point_in_convex_polygon",
    "polygon_area",
    "polygon_bounds",
    "signed_min_distance_point_to_polygon",
]

