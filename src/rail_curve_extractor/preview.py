from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import cumulative_arc_length


@dataclass(slots=True)
class ViewBounds:
    minimum: np.ndarray
    maximum: np.ndarray


def downsample_points(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[downsample_indices(len(points), limit)]


def downsample_indices(length: int, limit: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=int)
    if length <= limit:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, num=limit, dtype=int)


def visible_downsample_indices(
    points_xy: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    limit: int,
) -> np.ndarray:
    if len(points_xy) == 0:
        return np.empty(0, dtype=int)

    x_min, x_max = sorted((float(x_range[0]), float(x_range[1])))
    y_min, y_max = sorted((float(y_range[0]), float(y_range[1])))
    visible_mask = (
        (points_xy[:, 0] >= x_min)
        & (points_xy[:, 0] <= x_max)
        & (points_xy[:, 1] >= y_min)
        & (points_xy[:, 1] <= y_max)
    )
    visible_indices = np.flatnonzero(visible_mask)
    if len(visible_indices) <= limit:
        return visible_indices
    return visible_indices[downsample_indices(len(visible_indices), limit)]


def combine_bounds(*point_sets: np.ndarray) -> ViewBounds:
    valid_sets = [points for points in point_sets if len(points) > 0]
    if not valid_sets:
        return ViewBounds(minimum=np.array([0.0, 0.0]), maximum=np.array([1.0, 1.0]))

    stacked = np.vstack(valid_sets)
    minimum = stacked.min(axis=0).astype(float)
    maximum = stacked.max(axis=0).astype(float)
    if np.isclose(minimum[0], maximum[0]):
        minimum[0] -= 0.5
        maximum[0] += 0.5
    if np.isclose(minimum[1], maximum[1]):
        minimum[1] -= 0.5
        maximum[1] += 0.5
    return ViewBounds(minimum=minimum, maximum=maximum)


def fit_points_to_canvas(
    points: np.ndarray,
    bounds: ViewBounds,
    width: float,
    height: float,
    padding: float = 24.0,
) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)

    inner_width = max(width - 2.0 * padding, 1.0)
    inner_height = max(height - 2.0 * padding, 1.0)
    span = np.maximum(bounds.maximum - bounds.minimum, 1e-6)
    scale = min(inner_width / span[0], inner_height / span[1])
    scaled_span = span * scale
    offset_x = padding + (inner_width - scaled_span[0]) / 2.0
    offset_y = padding + (inner_height - scaled_span[1]) / 2.0

    normalized = points - bounds.minimum
    x_values = normalized[:, 0] * scale + offset_x
    y_values = height - (normalized[:, 1] * scale + offset_y)
    return np.column_stack([x_values, y_values])


def build_profile_points(centerline_world: np.ndarray) -> np.ndarray:
    if len(centerline_world) == 0:
        return np.empty((0, 2), dtype=float)
    arc_length = cumulative_arc_length(centerline_world)
    return np.column_stack([arc_length, centerline_world[:, 2]])
