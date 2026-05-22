from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class LocalFrame:
    origin: np.ndarray
    rotation: np.ndarray


def estimate_local_frame(points: np.ndarray) -> LocalFrame:
    xy = points[:, :2]
    center = xy.mean(axis=0)
    shifted = xy - center
    cov = np.cov(shifted.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    forward = eigenvectors[:, np.argmax(eigenvalues)]
    forward = forward / np.linalg.norm(forward)
    lateral = np.array([-forward[1], forward[0]])
    rotation = np.vstack([forward, lateral])
    origin = np.array([center[0], center[1], float(np.median(points[:, 2]))], dtype=float)
    return LocalFrame(origin=origin, rotation=rotation)


def world_to_local(points: np.ndarray, frame: LocalFrame) -> np.ndarray:
    shifted_xy = points[:, :2] - frame.origin[:2]
    st = shifted_xy @ frame.rotation.T
    z = (points[:, 2] - frame.origin[2]).reshape(-1, 1)
    return np.hstack([st, z])


def local_to_world(points: np.ndarray, frame: LocalFrame) -> np.ndarray:
    xy = points[:, :2] @ frame.rotation + frame.origin[:2]
    z = points[:, 2] + frame.origin[2]
    return np.column_stack([xy, z])


def compute_extent(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=float)
    if len(points) == 1:
        return np.array([0.0], dtype=float)
    deltas = np.diff(points, axis=0)
    steps = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(value, minimum)
    return value if value % 2 == 1 else value + 1


def heading_from_points(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    delta = points[-1, :2] - points[0, :2]
    return math.atan2(delta[1], delta[0])
