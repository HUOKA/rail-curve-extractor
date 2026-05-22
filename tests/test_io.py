from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rail_curve_extractor.io import (
    discover_point_cloud_files,
    load_point_cloud_data,
    summarize_point_cloud_input,
)


class PointCloudDirectoryIoTest(unittest.TestCase):
    def test_dji_terra_las_directory_is_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            terra_las_dir = project_dir / "lidars" / "terra_las"
            temp_block_dir = project_dir / "lidars" / ".temp" / "PointCloud" / "Block1"
            _write_las(terra_las_dir / "cloud10.las", _points(4, x_offset=10.0))
            _write_las(terra_las_dir / "cloud2.las", _points(3, x_offset=2.0))
            _write_las(temp_block_dir / "lidar_cloud.las", _points(5, x_offset=100.0))

            files = discover_point_cloud_files(project_dir)

        self.assertEqual([path.name for path in files], ["cloud2.las", "cloud10.las"])

    def test_summarize_directory_uses_las_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            terra_las_dir = project_dir / "lidars" / "terra_las"
            _write_las(terra_las_dir / "cloud0.las", _points(4, x_offset=0.0))
            _write_las(terra_las_dir / "cloud1.las", _points(6, x_offset=10.0))

            summary = summarize_point_cloud_input(project_dir)

        self.assertEqual(summary["source_file_count"], 2)
        self.assertEqual(summary["points"], 10)
        self.assertEqual(summary["source_format"], "directory:.las")
        self.assertTrue(summary["has_intensity"])
        self.assertTrue(summary["has_rgb"])
        self.assertEqual(summary["bounds"]["minimum"], [0.0, 0.0, 1.0])
        self.assertEqual(summary["bounds"]["maximum"], [15.0, 5.0, 1.5])

    def test_directory_loader_virtual_merges_and_samples_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            terra_las_dir = project_dir / "lidars" / "terra_las"
            _write_las(terra_las_dir / "cloud0.las", _points(4, x_offset=0.0))
            _write_las(terra_las_dir / "cloud1.las", _points(8, x_offset=10.0))

            point_cloud = load_point_cloud_data(project_dir, max_points=6)

        self.assertEqual(point_cloud.source_point_count, 12)
        self.assertEqual(len(point_cloud.source_paths), 2)
        self.assertEqual(point_cloud.source_format, "directory:.las")
        self.assertEqual(point_cloud.points.shape, (6, 3))
        self.assertIsNotNone(point_cloud.intensity)
        self.assertIsNotNone(point_cloud.rgb)
        self.assertEqual(len(point_cloud.intensity), 6)
        self.assertEqual(point_cloud.rgb.shape, (6, 3))

    def test_directory_loader_filters_to_xy_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir)
            terra_las_dir = project_dir / "lidars" / "terra_las"
            _write_las(terra_las_dir / "cloud0.las", _points(5, x_offset=0.0))
            _write_las(terra_las_dir / "cloud1.las", _points(5, x_offset=10.0))

            point_cloud = load_point_cloud_data(
                project_dir,
                max_points=3,
                xy_bounds=(10.0, 14.0, 0.0, 4.0),
            )

        self.assertEqual(point_cloud.source_point_count, 10)
        self.assertEqual(len(point_cloud.source_paths), 1)
        self.assertLessEqual(len(point_cloud.points), 3)
        self.assertTrue(np.all(point_cloud.points[:, 0] >= 10.0))
        self.assertTrue(np.all(point_cloud.points[:, 0] <= 14.0))


def _points(count: int, x_offset: float) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(x_offset, x_offset + count - 1, count),
            np.linspace(0.0, count - 1, count),
            np.linspace(1.0, 1.5, count),
        ]
    )


def _write_las(path: Path, points: np.ndarray) -> None:
    import laspy

    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.intensity = np.arange(len(points), dtype=np.uint16) + 100
    las.red = np.full(len(points), 1000, dtype=np.uint16)
    las.green = np.full(len(points), 2000, dtype=np.uint16)
    las.blue = np.full(len(points), 3000, dtype=np.uint16)
    las.write(path)


if __name__ == "__main__":
    unittest.main()
