from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rail_curve_extractor.pipeline import analyze_point_cloud, prepare_config, run_pipeline


MULTI_TRACK_ANGLE_DEG = 0.0
MULTI_TRACK_TRANSLATION = (100.0, -30.0, 5.0)


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_exports_centerline_and_usda(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "synthetic_track.csv"
            output_dir = root / "output"
            config_path = root / "config.json"

            points = build_synthetic_track()
            np.savetxt(input_path, points, delimiter=",", fmt="%.6f")

            config = {
                "height_filter": {"enabled": True, "keep_top_percent": 0.45},
                "slice_length": 0.8,
                "min_points_per_slice": 24,
                "rail_pair_spacing_min": 1.2,
                "rail_pair_spacing_max": 1.8,
                "peak_search_bins": 48,
                "peak_window_radius": 0.12,
                "savgol_window": 7,
                "curve_width": 0.05,
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_pipeline(input_path=input_path, output_dir=output_dir, config_path=config_path)

            centerline_path = output_dir / "centerline_points.xyz"
            usda_path = output_dir / "rail_centerline.usda"
            summary_path = output_dir / "run_summary.json"

            self.assertTrue(centerline_path.exists())
            self.assertTrue(usda_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertGreater(len(result.centerline_world), 10)
            self.assertGreater(result.summary["curve_length_m"], 20.0)

            truth_midpoints = build_truth_centerline()
            rmse = nearest_polyline_rmse(result.centerline_world[:, :2], truth_midpoints[:, :2])
            self.assertLess(rmse, 0.25)

    def test_xy_straight_constraint_removes_lateral_jitter_and_preserves_z(self) -> None:
        points = build_jittery_straight_track()
        base_overrides = {
            "height_filter": {"enabled": True, "keep_top_percent": 0.45},
            "slice_length": 0.8,
            "min_points_per_slice": 24,
            "rail_pair_spacing_min": 1.2,
            "rail_pair_spacing_max": 1.8,
            "peak_search_bins": 64,
            "peak_window_radius": 0.12,
            "savgol_window": 7,
            "curve_width": 0.05,
        }
        free_config = prepare_config(overrides={**base_overrides, "xy_constraint": {"mode": "free"}})
        straight_config = prepare_config(overrides={**base_overrides, "xy_constraint": {"mode": "straight"}})

        free_result = analyze_point_cloud(points, free_config)
        straight_result = analyze_point_cloud(points, straight_config)

        free_jitter = _xy_line_residual_std(free_result.centerline_world[:, :2])
        straight_jitter = _xy_line_residual_std(straight_result.centerline_world[:, :2])
        self.assertGreater(free_jitter, 0.03)
        self.assertLess(straight_jitter, 0.005)
        self.assertLess(straight_jitter, free_jitter * 0.2)
        np.testing.assert_allclose(straight_result.centerline_world[:, 2], free_result.centerline_world[:, 2], atol=1e-9)
        self.assertGreater(float(np.ptp(straight_result.centerline_world[:, 2])), 0.02)

    def test_manual_anchor_mode_follows_selected_parallel_track(self) -> None:
        points = build_synthetic_parallel_tracks((0.0, 4.0), seed=91)
        config = prepare_config(
            overrides={
                "height_filter": {"enabled": True, "keep_top_percent": 0.50},
                "slice_length": 0.8,
                "min_points_per_slice": 24,
                "rail_pair_spacing_min": 1.2,
                "rail_pair_spacing_max": 1.8,
                "peak_search_bins": 72,
                "peak_window_radius": 0.12,
                "savgol_window": 7,
                "curve_width": 0.05,
                "roi": {
                    "x_min": 99.0,
                    "x_max": 141.0,
                    "y_min": -32.0,
                    "y_max": -24.0,
                    "z_min": None,
                    "z_max": None,
                },
                "manual_anchor": {
                    "enabled": True,
                    "points": [[100.0, -30.0], [140.0, -30.0]],
                    "snap_distance": 1.0,
                    "score_weight": 20.0,
                },
            }
        )

        result = analyze_point_cloud(points, config)

        self.assertTrue(result.summary["manual_anchor_enabled"])
        self.assertEqual(result.summary["manual_anchor_points"], 2)
        self.assertLess(abs(float(np.median(result.centerline_world[:, 1])) + 30.0), 0.25)

    def test_guided_path_mode_extracts_selected_parallel_track(self) -> None:
        points = build_synthetic_parallel_tracks((0.0, 4.0), seed=93)
        anchor_world = _rotate_and_translate(
            np.array(
                [
                    [0.0, 3.55, 0.0],
                    [18.0, 3.70, 0.0],
                    [40.0, 3.90, 0.0],
                ]
            ),
            angle_deg=MULTI_TRACK_ANGLE_DEG,
            translation=MULTI_TRACK_TRANSLATION,
        )
        config = prepare_config(
            overrides={
                "height_filter": {"enabled": True, "keep_top_percent": 0.50},
                "slice_length": 0.8,
                "min_points_per_slice": 24,
                "rail_pair_spacing_min": 1.2,
                "rail_pair_spacing_max": 1.8,
                "peak_search_bins": 72,
                "peak_window_radius": 0.12,
                "savgol_window": 7,
                "guided_paths": {
                    "enabled": True,
                    "default_corridor_width": 4.2,
                    "tracks": [
                        {
                            "id": 4,
                            "points": anchor_world[:, :2].tolist(),
                        }
                    ],
                },
            }
        )

        result = analyze_point_cloud(points, config)

        self.assertTrue(result.summary["guided_path_mode"])
        self.assertEqual(result.summary["track_count"], 1)
        self.assertEqual(result.track_results[0].track_id, 4)
        local_center = _inverse_rotate_translate(
            result.track_results[0].centerline_world,
            angle_deg=MULTI_TRACK_ANGLE_DEG,
            translation=MULTI_TRACK_TRANSLATION,
        )
        self.assertLess(abs(float(np.median(local_center[:, 1])) - 4.0), 0.25)

    def test_oriented_roi_rejects_clutter_inside_axis_aligned_bbox(self) -> None:
        rng = np.random.default_rng(29)
        angle_deg = 28.0
        translation = (100.0, -30.0, 5.0)
        points = build_synthetic_track()
        clutter_s = rng.uniform(5.0, 35.0, size=5000)
        clutter_local = np.column_stack(
            [
                clutter_s,
                rng.normal(8.0, 0.18, size=len(clutter_s)),
                rng.normal(0.45, 0.04, size=len(clutter_s)),
            ]
        )
        points = np.vstack([points, _rotate_and_translate(clutter_local, angle_deg=angle_deg, translation=translation)])

        config = prepare_config(
            overrides={
                "roi": _world_roi_for_local_rect(0.0, 40.0, -1.8, 1.8, angle_deg, translation, margin=0.2),
                "oriented_roi": _oriented_roi_for_local_rect(0.0, 40.0, -1.8, 1.8, angle_deg, translation),
                "height_filter": {"enabled": True, "keep_top_percent": 0.45},
                "slice_length": 0.8,
                "min_points_per_slice": 24,
                "rail_pair_spacing_min": 1.2,
                "rail_pair_spacing_max": 1.8,
                "peak_search_bins": 64,
                "peak_window_radius": 0.12,
                "savgol_window": 7,
                "curve_width": 0.05,
            }
        )

        result = analyze_point_cloud(points, config)

        self.assertLess(result.summary["roi_points"], len(points) - 3000)
        self.assertEqual(result.summary["local_frame_source"], "oriented_roi")
        expected_axis = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
        self.assertGreater(float(result.frame.rotation[0] @ expected_axis), 0.999)
        truth_midpoints = build_truth_centerline()
        rmse = nearest_polyline_rmse(result.centerline_world[:, :2], truth_midpoints[:, :2])
        self.assertLess(rmse, 0.25)

    def test_multi_track_exports_three_centerlines_and_rejects_guard_rails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "synthetic_multi_track.csv"
            output_dir = root / "output"
            config_path = root / "config.json"

            points = build_synthetic_multi_track_with_guard_rails()
            np.savetxt(input_path, points, delimiter=",", fmt="%.6f")

            config = _multi_track_config()
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_pipeline(input_path=input_path, output_dir=output_dir, config_path=config_path)

            self.assertEqual(len(result.track_results), 3)
            self.assertTrue((output_dir / "rail_centerlines.usda").exists())
            for track_id in (1, 2, 3):
                self.assertTrue((output_dir / f"track_{track_id}_centerline_points.xyz").exists())
                self.assertTrue((output_dir / f"track_{track_id}_rail_points.xyz").exists())

            expected_offsets = [-4.0, 0.0, 4.0]
            for track_result, expected_t in zip(result.track_results, expected_offsets):
                local_center = _inverse_rotate_translate(
                    track_result.centerline_world,
                    angle_deg=MULTI_TRACK_ANGLE_DEG,
                    translation=MULTI_TRACK_TRANSLATION,
                )
                self.assertLess(abs(float(np.median(local_center[:, 1])) - expected_t), 0.25)

    def test_multi_track_accepts_oriented_roi_without_axis_roi(self) -> None:
        points = build_synthetic_multi_track_with_guard_rails()
        config = _multi_track_config()
        for track_config, (t_min, t_max) in zip(config["tracks"], [(-5.5, -2.5), (-1.5, 1.5), (2.5, 5.5)]):
            track_config["roi"] = {}
            track_config["oriented_roi"] = _oriented_roi_for_local_rect(
                0.0,
                40.0,
                t_min,
                t_max,
                MULTI_TRACK_ANGLE_DEG,
                MULTI_TRACK_TRANSLATION,
            )

        result = analyze_point_cloud(points, prepare_config(overrides=config))

        self.assertEqual(len(result.track_results), 3)
        self.assertEqual(result.summary["track_count"], 3)

    def test_auto_track_split_outputs_requested_track_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "synthetic_five_track.csv"
            output_dir = root / "output"
            config_path = root / "config.json"

            points = build_synthetic_parallel_tracks(offsets=(-8.0, -4.0, 0.0, 4.0, 8.0))
            np.savetxt(input_path, points, delimiter=",", fmt="%.6f")

            config = _multi_track_config()
            config["tracks"] = []
            config["auto_track_split"] = {
                "enabled": True,
                "count": 5,
                "oriented_roi": _oriented_roi_for_local_rect(
                    0.0,
                    40.0,
                    -10.5,
                    10.5,
                    MULTI_TRACK_ANGLE_DEG,
                    MULTI_TRACK_TRANSLATION,
                ),
                "band_overlap_ratio": 0.0,
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_pipeline(input_path=input_path, output_dir=output_dir, config_path=config_path)

            self.assertEqual(len(result.track_results), 5)
            self.assertEqual(result.summary["track_count"], 5)
            self.assertTrue(result.summary["auto_track_split"])
            self.assertTrue((output_dir / "track_5_centerline_points.xyz").exists())

            for track_result, expected_t in zip(result.track_results, (-8.0, -4.0, 0.0, 4.0, 8.0)):
                local_center = _inverse_rotate_translate(
                    track_result.centerline_world,
                    angle_deg=MULTI_TRACK_ANGLE_DEG,
                    translation=MULTI_TRACK_TRANSLATION,
                )
                self.assertLess(abs(float(np.median(local_center[:, 1])) - expected_t), 0.25)

    def test_multi_track_skips_failed_roi_and_exports_successful_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "synthetic_multi_track.csv"
            output_dir = root / "output"
            config_path = root / "config.json"

            points = build_synthetic_multi_track_with_guard_rails()
            np.savetxt(input_path, points, delimiter=",", fmt="%.6f")

            config = _multi_track_config()
            config["tracks"][2]["roi"] = {"x_min": 500.0, "x_max": 510.0, "y_min": 500.0, "y_max": 510.0, "z_min": None, "z_max": None}
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_pipeline(input_path=input_path, output_dir=output_dir, config_path=config_path)

            self.assertEqual(len(result.track_results), 2)
            self.assertEqual(result.summary["failed_track_count"], 1)
            self.assertTrue((output_dir / "track_1_centerline_points.xyz").exists())
            self.assertTrue((output_dir / "track_2_centerline_points.xyz").exists())

    def test_turnout_mode_exports_main_and_branch_centerlines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "synthetic_turnout.csv"
            output_dir = root / "output"
            config_path = root / "config.json"

            points = build_synthetic_turnout()
            np.savetxt(input_path, points, delimiter=",", fmt="%.6f")

            config = _turnout_config()
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = run_pipeline(input_path=input_path, output_dir=output_dir, config_path=config_path)

            self.assertEqual(len(result.turnout_results), 1)
            turnout = result.turnout_results[0]
            self.assertGreater(len(turnout.main_centerline_world), 10)
            self.assertGreater(len(turnout.branch_centerline_world), 6)
            self.assertTrue((output_dir / "turnout_1_main_centerline_points.xyz").exists())
            self.assertTrue((output_dir / "turnout_1_branch_centerline_points.xyz").exists())
            self.assertTrue((output_dir / "turnout_1_switch_point.xyz").exists())
            self.assertTrue((output_dir / "turnout_1_centerlines.usda").exists())

            main_local = _inverse_rotate_translate(turnout.main_centerline_world, angle_deg=0.0, translation=(10.0, 20.0, 2.0))
            branch_local = _inverse_rotate_translate(turnout.branch_centerline_world, angle_deg=0.0, translation=(10.0, 20.0, 2.0))
            self.assertLess(abs(float(np.median(main_local[-5:, 1]))), 0.35)
            self.assertGreater(float(np.median(branch_local[-5:, 1])), 1.4)

    def test_turnout_graph_search_keeps_branch_off_parallel_distractor(self) -> None:
        points = build_synthetic_turnout_with_parallel_distractor()
        config = prepare_config(
            overrides={
                **_turnout_config(),
                "turnout": {
                    **_turnout_config()["turnout"],
                    "roi": {
                        "x_min": 9.0,
                        "x_max": 70.0,
                        "y_min": 18.0,
                        "y_max": 28.0,
                        "z_min": None,
                        "z_max": None,
                    },
                },
            }
        )

        result = analyze_point_cloud(points, config)

        turnout = result.turnout_results[0]
        branch_local = _inverse_rotate_translate(turnout.branch_centerline_world, angle_deg=0.0, translation=(10.0, 20.0, 2.0))
        self.assertEqual(turnout.summary["turnout_trace_mode"], "graph_search")
        self.assertGreater(float(np.median(branch_local[-5:, 1])), 1.3)
        self.assertLess(float(np.median(branch_local[-5:, 1])), 2.8)

    def test_guided_path_mode_extracts_turnout_branch(self) -> None:
        points = build_synthetic_turnout_with_parallel_distractor()
        main_points = _rotate_and_translate(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [24.0, 0.0, 0.0],
                    [58.0, 0.0, 0.0],
                ]
            ),
            angle_deg=0.0,
            translation=(10.0, 20.0, 2.0),
        )
        branch_points = _rotate_and_translate(
            np.array(
                [
                    [14.0, 0.0, 0.0],
                    [34.0, 1.30, 0.0],
                    [58.0, 2.86, 0.0],
                ]
            ),
            angle_deg=0.0,
            translation=(10.0, 20.0, 2.0),
        )
        config = prepare_config(
            overrides={
                "height_filter": {"enabled": True, "keep_top_percent": 0.58},
                "slice_length": 0.8,
                "min_points_per_slice": 24,
                "rail_pair_spacing_min": 1.2,
                "rail_pair_spacing_max": 1.8,
                "peak_search_bins": 72,
                "peak_window_radius": 0.12,
                "savgol_window": 7,
                "guided_paths": {
                    "enabled": True,
                    "default_corridor_width": 7.0,
                    "turnouts": [
                        {
                            "id": 3,
                            "main_points": main_points[:, :2].tolist(),
                            "branch_points": branch_points[:, :2].tolist(),
                            "turnout": {
                                "branch_min_separation": 0.45,
                                "trace_min_branch_length_m": 8.0,
                            },
                        }
                    ],
                },
            }
        )

        result = analyze_point_cloud(points, config)

        self.assertTrue(result.summary["guided_path_mode"])
        self.assertEqual(len(result.turnout_results), 1)
        self.assertEqual(result.summary["track_count"], 2)
        turnout = result.turnout_results[0]
        branch_local = _inverse_rotate_translate(turnout.branch_centerline_world, angle_deg=0.0, translation=(10.0, 20.0, 2.0))
        self.assertGreater(float(np.median(branch_local[-5:, 1])), 1.3)
        self.assertLess(float(np.median(branch_local[-5:, 1])), 3.2)


def build_synthetic_track(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 40.0, 160)
    center_t = 0.12 * np.sin(s / 10.0)
    center_z = 0.02 * np.sin(s / 14.0)
    left_t = center_t - 0.75
    right_t = center_t + 0.75

    left_points = _sample_rail_surface(rng, s, left_t, center_z)
    right_points = _sample_rail_surface(rng, s, right_t, center_z)
    ballast = _sample_ballast(rng, s, center_t)
    clutter = _sample_clutter(rng, s)

    local_points = np.vstack([left_points, right_points, ballast, clutter])
    return _rotate_and_translate(local_points, angle_deg=28.0, translation=(100.0, -30.0, 5.0))


def build_jittery_straight_track(seed: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 55.0, 220)
    lateral_jitter = 0.11 * np.sin(s / 2.2) + 0.04 * np.sin(s / 0.9)
    center_z = 0.04 * np.sin(s / 9.0)
    left_points = _sample_rail_surface(rng, s, lateral_jitter - 0.75, center_z)
    right_points = _sample_rail_surface(rng, s, lateral_jitter + 0.75, center_z)
    ballast = _sample_ballast(rng, s, lateral_jitter)
    local_points = np.vstack([left_points, right_points, ballast])
    return _rotate_and_translate(local_points, angle_deg=14.0, translation=(20.0, 10.0, 3.0))


def build_synthetic_multi_track_with_guard_rails(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 40.0, 160)
    tracks = []
    for offset in (-4.0, 0.0, 4.0):
        center_t = offset + 0.08 * np.sin(s / 12.0)
        center_z = 0.02 * np.sin(s / 14.0)
        tracks.append(_sample_rail_surface(rng, s, center_t - 0.75, center_z))
        tracks.append(_sample_rail_surface(rng, s, center_t + 0.75, center_z))
        if offset != 0.0:
            tracks.append(_sample_guard_rail_surface(rng, s, center_t - 0.28, center_z))
            tracks.append(_sample_guard_rail_surface(rng, s, center_t + 0.28, center_z))
        tracks.append(_sample_ballast(rng, s, center_t))
    tracks.append(_sample_clutter(rng, s))
    local_points = np.vstack(tracks)
    return _rotate_and_translate(local_points, angle_deg=MULTI_TRACK_ANGLE_DEG, translation=MULTI_TRACK_TRANSLATION)


def build_synthetic_parallel_tracks(offsets: tuple[float, ...], seed: int = 31) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 40.0, 160)
    tracks = []
    for offset in offsets:
        center_t = offset + 0.05 * np.sin(s / 15.0)
        center_z = 0.018 * np.sin(s / 13.0)
        tracks.append(_sample_rail_surface(rng, s, center_t - 0.75, center_z))
        tracks.append(_sample_rail_surface(rng, s, center_t + 0.75, center_z))
        tracks.append(_sample_ballast(rng, s, center_t))
    tracks.append(_sample_clutter(rng, s))
    local_points = np.vstack(tracks)
    return _rotate_and_translate(local_points, angle_deg=MULTI_TRACK_ANGLE_DEG, translation=MULTI_TRACK_TRANSLATION)


def build_synthetic_turnout(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 42.0, 168)
    center_t = np.zeros_like(s)
    center_z = 0.015 * np.sin(s / 12.0)
    main_left = _sample_rail_surface(rng, s, center_t - 0.75, center_z)
    main_right = _sample_rail_surface(rng, s, center_t + 0.75, center_z)

    branch_mask = s >= 14.0
    branch_s = s[branch_mask]
    branch_center = 0.095 * (branch_s - 14.0)
    branch_center_z = center_z[branch_mask] + 0.005 * np.sin(branch_s / 8.0)
    branch_left = _sample_rail_surface(rng, branch_s, branch_center - 0.75, branch_center_z)
    branch_right = _sample_rail_surface(rng, branch_s, branch_center + 0.75, branch_center_z)
    guard = _sample_guard_rail_surface(rng, branch_s, branch_center + 0.28, branch_center_z)
    ballast = _sample_ballast(rng, s, center_t)
    clutter = _sample_clutter(rng, s)
    local_points = np.vstack([main_left, main_right, branch_left, branch_right, guard, ballast, clutter])
    return _rotate_and_translate(local_points, angle_deg=0.0, translation=(10.0, 20.0, 2.0))


def build_synthetic_turnout_with_parallel_distractor(seed: int = 37) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 58.0, 232)
    center_z = 0.012 * np.sin(s / 12.0)
    main_left = _sample_rail_surface(rng, s, np.full_like(s, -0.75), center_z)
    main_right = _sample_rail_surface(rng, s, np.full_like(s, 0.75), center_z)

    branch_mask = s >= 14.0
    branch_s = s[branch_mask]
    branch_center = 0.065 * (branch_s - 14.0)
    branch_center_z = center_z[branch_mask]
    branch_left = _sample_rail_surface(rng, branch_s, branch_center - 0.75, branch_center_z)
    branch_right = _sample_rail_surface(rng, branch_s, branch_center + 0.75, branch_center_z)
    guard = _sample_guard_rail_surface(rng, branch_s, branch_center + 0.28, branch_center_z)

    distractor_mask = s >= 28.0
    distractor_s = s[distractor_mask]
    distractor_center = np.full_like(distractor_s, 3.8)
    distractor_z = center_z[distractor_mask]
    distractor_left = _sample_rail_surface(rng, distractor_s, distractor_center - 0.75, distractor_z)
    distractor_right = _sample_rail_surface(rng, distractor_s, distractor_center + 0.75, distractor_z)

    ballast = _sample_ballast(rng, s, np.zeros_like(s))
    clutter = _sample_clutter(rng, s)
    local_points = np.vstack(
        [
            main_left,
            main_right,
            branch_left,
            branch_right,
            guard,
            distractor_left,
            distractor_right,
            ballast,
            clutter,
        ]
    )
    return _rotate_and_translate(local_points, angle_deg=0.0, translation=(10.0, 20.0, 2.0))


def build_truth_centerline() -> np.ndarray:
    s = np.linspace(0.0, 40.0, 80)
    center_t = 0.12 * np.sin(s / 10.0)
    center_z = 0.02 * np.sin(s / 14.0)
    local = np.column_stack([s, center_t, center_z])
    return _rotate_and_translate(local, angle_deg=28.0, translation=(100.0, -30.0, 5.0))


def _sample_rail_surface(rng: np.random.Generator, s: np.ndarray, t_center: np.ndarray, z_center: np.ndarray) -> np.ndarray:
    repeats = 6
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.08, size=len(s) * repeats)
    t_values = np.repeat(t_center, repeats) + rng.normal(0.0, 0.025, size=len(s) * repeats)
    z_values = np.repeat(z_center + 0.08, repeats) + rng.normal(0.0, 0.01, size=len(s) * repeats)
    return np.column_stack([s_values, t_values, z_values])


def _sample_guard_rail_surface(rng: np.random.Generator, s: np.ndarray, t_center: np.ndarray, z_center: np.ndarray) -> np.ndarray:
    repeats = 4
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.08, size=len(s) * repeats)
    t_values = np.repeat(t_center, repeats) + rng.normal(0.0, 0.022, size=len(s) * repeats)
    z_values = np.repeat(z_center + 0.085, repeats) + rng.normal(0.0, 0.01, size=len(s) * repeats)
    return np.column_stack([s_values, t_values, z_values])


def _sample_ballast(rng: np.random.Generator, s: np.ndarray, center_t: np.ndarray) -> np.ndarray:
    repeats = 8
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.12, size=len(s) * repeats)
    lateral = rng.uniform(-1.5, 1.5, size=len(s) * repeats)
    t_values = np.repeat(center_t, repeats) + lateral
    z_values = rng.normal(-0.04, 0.025, size=len(s) * repeats)
    return np.column_stack([s_values, t_values, z_values])


def _sample_clutter(rng: np.random.Generator, s: np.ndarray) -> np.ndarray:
    repeats = 2
    s_values = np.repeat(s, repeats) + rng.normal(0.0, 0.2, size=len(s) * repeats)
    t_values = rng.uniform(-2.0, 2.0, size=len(s) * repeats)
    z_values = rng.normal(0.45, 0.05, size=len(s) * repeats)
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


def _inverse_rotate_translate(points: np.ndarray, angle_deg: float, translation: tuple[float, float, float]) -> np.ndarray:
    shifted = points.copy()
    shifted[:, 0] -= translation[0]
    shifted[:, 1] -= translation[1]
    shifted[:, 2] -= translation[2]
    angle = math.radians(angle_deg)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
    )
    xy = shifted[:, :2] @ rotation
    return np.column_stack([xy, shifted[:, 2]])


def _multi_track_config() -> dict[str, object]:
    return {
        "height_filter": {"enabled": True, "keep_top_percent": 0.50},
        "slice_length": 0.8,
        "min_points_per_slice": 24,
        "rail_pair_spacing_min": 1.2,
        "rail_pair_spacing_max": 1.8,
        "peak_search_bins": 72,
        "peak_window_radius": 0.12,
        "savgol_window": 7,
        "curve_width": 0.05,
        "tracks": [
            {"id": 1, "enabled": True, "roi": _world_roi_for_local_t(-5.5, -2.5)},
            {"id": 2, "enabled": True, "roi": _world_roi_for_local_t(-1.5, 1.5)},
            {"id": 3, "enabled": True, "roi": _world_roi_for_local_t(2.5, 5.5)},
        ],
    }


def _turnout_config() -> dict[str, object]:
    return {
        "height_filter": {"enabled": True, "keep_top_percent": 0.58},
        "slice_length": 0.8,
        "min_points_per_slice": 24,
        "rail_pair_spacing_min": 1.2,
        "rail_pair_spacing_max": 1.8,
        "peak_search_bins": 72,
        "peak_window_radius": 0.12,
        "savgol_window": 7,
        "curve_width": 0.05,
        "turnout": {
            "enabled": True,
            "branch_min_separation": 0.45,
            "roi": {
                "x_min": 9.0,
                "x_max": 54.0,
                "y_min": 18.0,
                "y_max": 25.8,
                "z_min": None,
                "z_max": None,
            },
        },
    }


def _world_roi_for_local_t(t_min: float, t_max: float) -> dict[str, float | None]:
    return _world_roi_for_local_rect(0.0, 40.0, t_min, t_max, MULTI_TRACK_ANGLE_DEG, MULTI_TRACK_TRANSLATION)


def _world_roi_for_local_rect(
    s_min: float,
    s_max: float,
    t_min: float,
    t_max: float,
    angle_deg: float,
    translation: tuple[float, float, float],
    margin: float = 0.8,
) -> dict[str, float | None]:
    corners = np.array([[s_min, t_min, -1.0], [s_max, t_min, 1.0], [s_min, t_max, -1.0], [s_max, t_max, 1.0]])
    world = _rotate_and_translate(corners, angle_deg=angle_deg, translation=translation)
    return {
        "x_min": float(world[:, 0].min() - margin),
        "x_max": float(world[:, 0].max() + margin),
        "y_min": float(world[:, 1].min() - margin),
        "y_max": float(world[:, 1].max() + margin),
        "z_min": None,
        "z_max": None,
    }


def _oriented_roi_for_local_rect(
    s_min: float,
    s_max: float,
    t_min: float,
    t_max: float,
    angle_deg: float,
    translation: tuple[float, float, float],
) -> dict[str, object]:
    angle = math.radians(angle_deg)
    axis_s = [math.cos(angle), math.sin(angle)]
    axis_t = [-math.sin(angle), math.cos(angle)]
    margin = 0.8
    return {
        "enabled": True,
        "origin": [translation[0], translation[1]],
        "axis_s": axis_s,
        "axis_t": axis_t,
        "s_min": float(s_min - margin),
        "s_max": float(s_max + margin),
        "t_min": float(t_min - margin),
        "t_max": float(t_max + margin),
        "z_min": None,
        "z_max": None,
    }


def nearest_polyline_rmse(observed_xy: np.ndarray, truth_xy: np.ndarray) -> float:
    deltas = observed_xy[:, None, :] - truth_xy[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    nearest = distances.min(axis=1)
    return float(np.sqrt(np.mean(nearest**2)))


def _xy_line_residual_std(xy: np.ndarray) -> float:
    centered = xy - xy.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projection = centered @ direction
    fitted = np.outer(projection, direction)
    residuals = centered - fitted
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


if __name__ == "__main__":
    unittest.main()
