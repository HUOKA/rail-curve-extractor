from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np

from rail_curve_extractor.backend.app import _result_overlay, app
from rail_curve_extractor.geometry import LocalFrame
from rail_curve_extractor.pipeline import PipelineResult, TrackResult


class BackendAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_token = os.environ.get("RAIL_CURVE_BACKEND_TOKEN")
        os.environ.pop("RAIL_CURVE_BACKEND_TOKEN", None)

    def tearDown(self) -> None:
        if self.previous_token is None:
            os.environ.pop("RAIL_CURVE_BACKEND_TOKEN", None)
        else:
            os.environ["RAIL_CURVE_BACKEND_TOKEN"] = self.previous_token

    def test_health_and_default_config(self) -> None:
        client = TestClient(app)

        health = client.get("/api/health")
        config = client.get("/api/config/default")

        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ok"])
        self.assertEqual(config.status_code, 200)
        self.assertIn("rail_pair_spacing_min", config.json())
        self.assertIn("tracks", config.json())

    def test_embedded_viewer_status_is_readable_when_idle(self) -> None:
        client = TestClient(app)

        response = client.get("/api/viewer/embedded/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["state"], {"idle", "stopped"})
        self.assertIn("progress_path", payload)

    def test_embedded_viewer_start_is_disabled_by_default(self) -> None:
        previous_flag = os.environ.pop("RAIL_CURVE_ENABLE_OPEN3D_WEBRTC", None)
        client = TestClient(app)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "points.npy"
                np.save(path, np.zeros((4, 3)))

                response = client.post(
                    "/api/viewer/embedded/start",
                    json={"input_path": str(path), "max_points": 4},
                )
        finally:
            if previous_flag is not None:
                os.environ["RAIL_CURVE_ENABLE_OPEN3D_WEBRTC"] = previous_flag

        self.assertEqual(response.status_code, 501)
        self.assertIn("Open3D WebRTC", response.json()["detail"])

    def test_token_blocks_api_when_configured(self) -> None:
        os.environ["RAIL_CURVE_BACKEND_TOKEN"] = "secret"
        client = TestClient(app)

        blocked = client.get("/api/health")
        allowed = client.get("/api/health", headers={"x-local-token": "secret"})

        self.assertEqual(blocked.status_code, 403)
        self.assertEqual(allowed.status_code, 200)

    def test_open_viewer_passes_bounds_and_point_size(self) -> None:
        class DummyProcess:
            pid = 12345

        client = TestClient(app)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points.npy"
            np.save(path, np.zeros((4, 3)))
            with patch("rail_curve_extractor.backend.app.subprocess.Popen", return_value=DummyProcess()) as popen:
                response = client.post(
                    "/api/viewer/open",
                    json={
                        "input_path": str(path),
                        "max_points": 12_000_000,
                        "point_size": 2,
                        "bounds": {"x_min": 1.0, "x_max": 2.0, "y_min": 3.0, "y_max": 4.0},
                    },
                )

        self.assertEqual(response.status_code, 200)
        command = popen.call_args.args[0]
        self.assertIn("--point-size", command)
        self.assertIn("2.0", command)
        self.assertIn("--bounds", command)
        self.assertEqual(command[-4:], ["1.0", "2.0", "3.0", "4.0"])
        self.assertEqual(response.json()["bounds"], [1.0, 2.0, 3.0, 4.0])

    def test_point_cloud_preview_samples_points(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points.npy"
            points = np.column_stack(
                [
                    np.linspace(0.0, 99.0, 100),
                    np.linspace(20.0, 119.0, 100),
                    np.ones(100),
                ]
            )
            np.save(path, points)

            response = client.post(
                "/api/point-cloud/preview",
                json={"input_path": str(path), "max_points": 12},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["input_points"], 100)
        self.assertEqual(payload["sample_points"], 12)
        self.assertEqual(len(payload["points_xy"]), 12)
        self.assertEqual(payload["bounds"]["minimum"], [0.0, 20.0])

    def test_point_cloud_preview_filters_bounds(self) -> None:
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "points.npy"
            points = np.column_stack(
                [
                    np.linspace(0.0, 99.0, 100),
                    np.linspace(20.0, 119.0, 100),
                    np.ones(100),
                ]
            )
            np.save(path, points)

            response = client.post(
                "/api/point-cloud/preview",
                json={
                    "input_path": str(path),
                    "max_points": 50,
                    "bounds": {"x_min": 40.0, "x_max": 45.0, "y_min": 60.0, "y_max": 65.0},
                },
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["input_points"], 100)
        self.assertEqual(payload["sample_points"], 6)
        self.assertEqual(payload["bounds"]["minimum"], [40.0, 60.0])
        self.assertEqual(payload["bounds"]["maximum"], [45.0, 65.0])
        self.assertTrue(all(40.0 <= point[0] <= 45.0 for point in payload["points_xy"]))
        self.assertTrue(all(60.0 <= point[1] <= 65.0 for point in payload["points_xy"]))

    def test_result_overlay_contains_track_centerline_and_sampled_rails(self) -> None:
        centerline = np.column_stack([np.linspace(0.0, 19.0, 20), np.zeros(20), np.ones(20)])
        rail_points = np.column_stack([np.linspace(0.0, 99.0, 100), np.ones(100), np.ones(100)])
        frame = LocalFrame(origin=np.zeros(3), rotation=np.eye(2))
        track = TrackResult(
            track_id=7,
            frame=frame,
            filtered_points_world=rail_points,
            rail_points_world=rail_points,
            centerline_world=centerline,
            summary={"source": "guided_path", "label": "装卸线 1"},
            confidence=0.82,
        )
        result = PipelineResult(
            config={},
            raw_points_world=rail_points,
            frame=frame,
            filtered_points_world=rail_points,
            rail_points_world=rail_points,
            centerline_world=centerline,
            summary={},
            track_results=[track],
        )

        overlay = _result_overlay(result)

        self.assertEqual(overlay["track_count"], 1)
        self.assertEqual(overlay["tracks"][0]["id"], 7)
        self.assertEqual(overlay["tracks"][0]["label"], "装卸线 1")
        self.assertEqual(len(overlay["tracks"][0]["centerline_xy"]), 20)
        self.assertEqual(len(overlay["tracks"][0]["rail_points_xy"]), 100)
        self.assertEqual(overlay["tracks"][0]["centerline_xy"][0], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
