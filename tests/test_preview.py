from __future__ import annotations

import unittest

import numpy as np

from rail_curve_extractor.preview import (
    build_profile_points,
    combine_bounds,
    downsample_indices,
    downsample_points,
    fit_points_to_canvas,
    visible_downsample_indices,
)


class PreviewHelpersTest(unittest.TestCase):
    def test_downsample_points_keeps_limit(self) -> None:
        points = np.arange(300, dtype=float).reshape(100, 3)
        sampled = downsample_points(points, limit=12)
        self.assertEqual(len(sampled), 12)
        self.assertTrue(np.allclose(sampled[0], points[0]))
        self.assertTrue(np.allclose(sampled[-1], points[-1]))

    def test_downsample_indices_keeps_aligned_arrays(self) -> None:
        points = np.arange(300, dtype=float).reshape(100, 3)
        colors = np.arange(300, 600, dtype=np.uint16).reshape(100, 3)
        indices = downsample_indices(len(points), limit=12)
        sampled_points = points[indices]
        sampled_colors = colors[indices]

        self.assertEqual(len(indices), 12)
        self.assertTrue(np.allclose(sampled_points[:, 0] + 300, sampled_colors[:, 0]))

    def test_visible_downsample_indices_filters_current_view(self) -> None:
        points = np.array(
            [
                [-5.0, 0.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [10.0, 0.0],
            ]
        )

        indices = visible_downsample_indices(points, x_range=(0.0, 3.0), y_range=(0.0, 3.0), limit=2)

        self.assertEqual(indices.tolist(), [1, 4])

    def test_fit_points_to_canvas_keeps_points_inside_frame(self) -> None:
        points = np.array([[0.0, 0.0], [10.0, 5.0], [3.0, 2.0]])
        bounds = combine_bounds(points)
        fitted = fit_points_to_canvas(points, bounds, width=400, height=200, padding=20)
        self.assertTrue(np.all(fitted[:, 0] >= 20.0))
        self.assertTrue(np.all(fitted[:, 0] <= 380.0))
        self.assertTrue(np.all(fitted[:, 1] >= 20.0))
        self.assertTrue(np.all(fitted[:, 1] <= 180.0))

    def test_build_profile_points_uses_arc_length(self) -> None:
        centerline = np.array([[0.0, 0.0, 5.0], [3.0, 4.0, 6.0], [6.0, 4.0, 7.0]])
        profile = build_profile_points(centerline)
        self.assertEqual(profile.shape, (3, 2))
        self.assertAlmostEqual(profile[1, 0], np.sqrt(26.0))
        self.assertAlmostEqual(profile[2, 0], np.sqrt(26.0) + np.sqrt(10.0))
        self.assertAlmostEqual(profile[0, 1], 5.0)
        self.assertAlmostEqual(profile[2, 1], 7.0)


if __name__ == "__main__":
    unittest.main()
