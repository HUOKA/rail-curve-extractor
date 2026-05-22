from __future__ import annotations

import unittest

import numpy as np

from rail_curve_extractor.pipeline import (
    SliceDetection,
    _detect_slice,
    _estimate_track_confidence,
    analyze_point_cloud,
    prepare_config,
)


class PipelineSingleRailFallbackTest(unittest.TestCase):
    def test_single_rail_slice_requires_reference_center(self) -> None:
        config = _single_track_config()
        points = _sample_one_slice_rail(t_center=-0.75, seed=1)

        no_reference = _detect_slice(points, 0.0, 0.8, config, previous_center_t=None)
        self.assertIsNone(no_reference)

        detection = _detect_slice(points, 0.0, 0.8, config, previous_center_t=0.0)
        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(detection.pair_mode, "single_left")
        self.assertAlmostEqual(float(detection.center_local[1]), 0.0, delta=0.08)
        self.assertAlmostEqual(detection.right_peak_t - detection.left_peak_t, 1.5, delta=0.01)

    def test_single_rail_slice_rejects_center_guard_peak(self) -> None:
        config = _single_track_config()
        points = _sample_one_slice_rail(t_center=0.28, seed=2)

        detection = _detect_slice(points, 0.0, 0.8, config, previous_center_t=0.0)

        self.assertIsNone(detection)

    def test_single_rail_modes_reduce_confidence(self) -> None:
        dual = [_fake_detection(s_value=float(index), pair_mode="dual") for index in range(12)]
        partial = [
            _fake_detection(s_value=float(index), pair_mode="dual" if index < 6 else "single_left")
            for index in range(12)
        ]
        centerline = np.array([[float(index), 0.0, 0.0] for index in range(12)], dtype=float)
        rail_points = np.vstack([detection.rail_points_local for detection in dual])

        dual_confidence = _estimate_track_confidence(dual, centerline, rail_points)
        partial_confidence = _estimate_track_confidence(partial, centerline, rail_points)

        self.assertLess(partial_confidence, dual_confidence)
        self.assertGreater(partial_confidence, 0.9)

    def test_pipeline_marks_inferred_single_rail_gap(self) -> None:
        config = _single_track_config()
        full_result = analyze_point_cloud(_sample_track_with_single_rail_gap(gap=None, seed=3), config)
        partial_result = analyze_point_cloud(_sample_track_with_single_rail_gap(gap=(14.0, 26.0), seed=3), config)

        self.assertGreaterEqual(partial_result.summary["single_rail_inferred_slices"], 8)
        self.assertLess(partial_result.summary["confidence"], full_result.summary["confidence"])
        self.assertLess(float(np.max(np.abs(partial_result.centerline_world[:, 1]))), 0.25)


def _single_track_config() -> dict[str, object]:
    return prepare_config(
        overrides={
            "height_filter": {"enabled": True, "keep_top_percent": 0.50},
            "slice_length": 0.8,
            "min_points_per_slice": 20,
            "rail_pair_spacing_min": 1.2,
            "rail_pair_spacing_max": 1.8,
            "rail_pair_spacing_target": 1.5,
            "peak_search_bins": 48,
            "peak_window_radius": 0.12,
            "savgol_window": 7,
            "curve_width": 0.05,
        }
    )


def _sample_one_slice_rail(t_center: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s_values = rng.uniform(0.02, 0.78, size=80)
    t_values = rng.normal(t_center, 0.025, size=len(s_values))
    z_values = rng.normal(0.08, 0.01, size=len(s_values))
    return np.column_stack([s_values, t_values, z_values])


def _fake_detection(s_value: float, pair_mode: str) -> SliceDetection:
    rail_points = np.array(
        [
            [s_value, -0.75, 0.08],
            [s_value, 0.75, 0.08],
        ],
        dtype=float,
    )
    return SliceDetection(
        center_local=np.array([s_value, 0.0, 0.08], dtype=float),
        rail_points_local=rail_points,
        left_peak_t=-0.75,
        right_peak_t=0.75,
        pair_mode=pair_mode,
    )


def _sample_track_with_single_rail_gap(
    gap: tuple[float, float] | None,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 40.0, 180)
    center_t = np.zeros_like(s)
    center_z = 0.02 * np.sin(s / 12.0)
    left_points = _sample_rail_surface(rng, s, center_t - 0.75, center_z)
    if gap is None:
        right_s = s
        right_center_z = center_z
    else:
        right_mask = (s < gap[0]) | (s > gap[1])
        right_s = s[right_mask]
        right_center_z = center_z[right_mask]
    right_points = _sample_rail_surface(rng, right_s, np.full_like(right_s, 0.75), right_center_z)
    ballast = _sample_ballast(rng, s, center_t)
    return np.vstack([left_points, right_points, ballast])


def _sample_rail_surface(
    rng: np.random.Generator,
    s: np.ndarray,
    t_center: np.ndarray,
    z_center: np.ndarray,
) -> np.ndarray:
    repeats = 6
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.08, size=len(s) * repeats)
    t_values = np.repeat(t_center, repeats) + rng.normal(0.0, 0.025, size=len(s) * repeats)
    z_values = np.repeat(z_center + 0.08, repeats) + rng.normal(0.0, 0.01, size=len(s) * repeats)
    return np.column_stack([s_values, t_values, z_values])


def _sample_ballast(rng: np.random.Generator, s: np.ndarray, center_t: np.ndarray) -> np.ndarray:
    repeats = 8
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.12, size=len(s) * repeats)
    t_values = np.repeat(center_t, repeats) + rng.uniform(-1.5, 1.5, size=len(s) * repeats)
    z_values = rng.normal(-0.04, 0.025, size=len(s) * repeats)
    return np.column_stack([s_values, t_values, z_values])


if __name__ == "__main__":
    unittest.main()
