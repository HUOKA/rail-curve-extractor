from __future__ import annotations

import math
import unittest

import numpy as np

from rail_curve_extractor.pipeline import analyze_point_cloud, prepare_config


class AdvancedPipelineTest(unittest.TestCase):
    def test_advanced_las_like_preprocessing_uses_intensity_and_returns_centerline(self) -> None:
        points, intensity = build_las_like_track()
        config = prepare_config(
            overrides={
                "height_filter": {"enabled": False},
                "slice_length": 0.8,
                "min_points_per_slice": 20,
                "peak_search_bins": 48,
                "peak_window_radius": 0.12,
                "savgol_window": 7,
                "advanced_las": {
                    "enabled": True,
                    "sample_max_points": 20000,
                    "ground_cell_size": 0.8,
                    "ground_percentile": 0.10,
                    "rail_height_min": 0.05,
                    "rail_height_max": 0.22,
                    "intensity_quantile_max": 0.55,
                    "occupancy_threshold": 3,
                    "min_component_points": 300,
                    "component_cell_size": 0.8,
                    "corridor_quantile_low": 0.02,
                    "corridor_quantile_high": 0.98,
                    "corridor_margin": 1.5,
                },
            }
        )

        result = analyze_point_cloud(raw_points=points, config=config, intensity=intensity, source_format=".las")

        self.assertEqual(result.summary["preprocessing_mode"], "advanced_las_corridor")
        self.assertGreater(result.summary["centerline_points"], 20)
        self.assertGreater(result.summary["curve_length_m"], 25.0)
        self.assertLess(len(result.filtered_points_world), len(points))


def build_las_like_track(seed: int = 23) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 36.0, 120)
    center_t = 0.10 * np.sin(s / 9.0)
    center_z = 0.01 * np.sin(s / 10.0)
    left_t = center_t - 0.75
    right_t = center_t + 0.75

    left_points = _sample_surface(rng, s, left_t, center_z + 0.08, lateral_sigma=0.02, z_sigma=0.008, repeats=6)
    right_points = _sample_surface(rng, s, right_t, center_z + 0.08, lateral_sigma=0.02, z_sigma=0.008, repeats=6)
    ballast = _sample_ballast(rng, s, center_t)
    clutter = _sample_clutter(rng, s)

    local_points = np.vstack([left_points, right_points, ballast, clutter])
    world_points = _rotate_and_translate(local_points, angle_deg=32.0, translation=(500.0, 800.0, 12.0))

    left_int = rng.normal(5200.0, 600.0, size=len(left_points))
    right_int = rng.normal(5200.0, 600.0, size=len(right_points))
    ballast_int = rng.normal(12500.0, 2200.0, size=len(ballast))
    clutter_int = rng.normal(16500.0, 2600.0, size=len(clutter))
    intensity = np.concatenate([left_int, right_int, ballast_int, clutter_int]).clip(500.0, 65000.0).astype(np.float32)
    return world_points, intensity


def _sample_surface(
    rng: np.random.Generator,
    s: np.ndarray,
    t_center: np.ndarray,
    z_center: np.ndarray,
    lateral_sigma: float,
    z_sigma: float,
    repeats: int,
) -> np.ndarray:
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.06, size=len(s) * repeats)
    t_values = np.repeat(t_center, repeats) + rng.normal(0.0, lateral_sigma, size=len(s) * repeats)
    z_values = np.repeat(z_center, repeats) + rng.normal(0.0, z_sigma, size=len(s) * repeats)
    return np.column_stack([s_values, t_values, z_values])


def _sample_ballast(rng: np.random.Generator, s: np.ndarray, center_t: np.ndarray) -> np.ndarray:
    repeats = 8
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.12, size=len(s) * repeats)
    t_values = np.repeat(center_t, repeats) + rng.uniform(-3.6, 3.6, size=len(s) * repeats)
    z_values = rng.normal(-0.03, 0.03, size=len(s) * repeats)
    return np.column_stack([s_values, t_values, z_values])


def _sample_clutter(rng: np.random.Generator, s: np.ndarray) -> np.ndarray:
    repeats = 4
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.25, size=len(s) * repeats)
    t_values = rng.uniform(-8.0, 8.0, size=len(s) * repeats)
    z_values = rng.normal(0.55, 0.12, size=len(s) * repeats)
    return np.column_stack([s_values, t_values, z_values])


def _rotate_and_translate(points: np.ndarray, angle_deg: float, translation: tuple[float, float, float]) -> np.ndarray:
    angle = math.radians(angle_deg)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    xy = points[:, :2] @ rotation.T
    xyz = np.column_stack([xy, points[:, 2]])
    xyz[:, 0] += translation[0]
    xyz[:, 1] += translation[1]
    xyz[:, 2] += translation[2]
    return xyz


if __name__ == "__main__":
    unittest.main()
