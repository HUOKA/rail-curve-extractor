from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rail_curve_extractor.dom_tiler import (
    DomAlignmentOptions,
    DomTileOptions,
    align_dom_to_axis,
    align_dom_to_axis_from_map_points,
    align_dom_to_corridor_from_map_points,
    auto_detect_dom_corridor_points,
    create_dom_preview,
    discover_dom_file,
    options_from_overlap,
    suggest_annotation_tile_size,
    stride_from_overlap,
    tile_dom,
)


class DomTilerTest(unittest.TestCase):
    def test_discover_prefers_dji_terra_dom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preferred = root / "lidars" / "terra_dom" / "dom.tif"
            other = root / "other_dom.tif"
            preferred.parent.mkdir(parents=True)
            preferred.write_bytes(b"preferred")
            other.write_bytes(b"other")

            self.assertEqual(discover_dom_file(root), preferred)

    def test_tile_dom_writes_images_and_georef_metadata(self) -> None:
        rasterio = _require_rasterio()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dom.tif"
            output = root / "tiles"
            transform = rasterio.transform.from_origin(100.0, 200.0, 0.5, 0.5)
            data = np.zeros((3, 20, 20), dtype=np.uint8)
            data[:, 2:18, 3:17] = np.array([[[80]], [[120]], [[160]]], dtype=np.uint8)
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=20,
                height=20,
                count=3,
                dtype="uint8",
                crs="EPSG:32651",
                transform=transform,
            ) as dataset:
                dataset.write(data)

            result = tile_dom(
                source,
                output,
                DomTileOptions(tile_size=8, stride=6, prefix="test", skip_empty=False),
            )

            self.assertEqual(result.tile_count, 9)
            self.assertEqual(result.epsg, 32651)
            self.assertTrue((output / "images" / "test_r000000_c000000.png").exists())
            self.assertTrue((output / "tile_georef.csv").exists())
            self.assertTrue((output / "tile_georef.json").exists())

            with Image.open(output / "images" / "test_r000000_c000000.png") as image:
                self.assertEqual(image.size, (8, 8))
                self.assertEqual(image.mode, "RGB")

            with (output / "tile_georef.csv").open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 9)
            self.assertEqual(rows[0]["row_off"], "0")
            self.assertEqual(rows[0]["col_off"], "0")
            self.assertEqual(json.loads(rows[0]["tile_transform"]), [0.5, 0.0, 100.0, 0.0, -0.5, 200.0])

            payload = json.loads((output / "tile_georef.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["tile_count"], 9)
            self.assertEqual(payload["records"][0]["tile_name"], "test_r000000_c000000.png")

    def test_rectangular_tiles_can_be_built_from_overlap_ratio(self) -> None:
        rasterio = _require_rasterio()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dom.tif"
            output = root / "tiles"
            transform = rasterio.transform.from_origin(10.0, 20.0, 1.0, 1.0)
            data = np.full((3, 18, 24), 120, dtype=np.uint8)
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=24,
                height=18,
                count=3,
                dtype="uint8",
                crs="EPSG:32651",
                transform=transform,
            ) as dataset:
                dataset.write(data)

            result = tile_dom(
                source,
                output,
                options_from_overlap(10, 8, 0.5, prefix="rect", skip_empty=False),
            )

            self.assertEqual(stride_from_overlap(10, 0.5), 5)
            self.assertEqual(result.tile_count, 16)
            self.assertEqual(result.records[0].width, 10)
            self.assertEqual(result.records[0].height, 8)
            self.assertEqual(result.records[-1].col_off, 14)
            self.assertEqual(result.records[-1].row_off, 10)

    def test_create_dom_preview_returns_scaled_rgb_image(self) -> None:
        rasterio = _require_rasterio()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dom.tif"
            transform = rasterio.transform.from_origin(100.0, 200.0, 0.5, 0.5)
            data = np.zeros((3, 20, 10), dtype=np.uint8)
            data[:, 4:16, 2:8] = np.array([[[80]], [[120]], [[160]]], dtype=np.uint8)
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=10,
                height=20,
                count=3,
                dtype="uint8",
                crs="EPSG:32651",
                transform=transform,
                nodata=0,
            ) as dataset:
                dataset.write(data)

            preview = create_dom_preview(source, max_width=5, max_height=8)

            self.assertEqual(preview.source_width, 10)
            self.assertEqual(preview.source_height, 20)
            self.assertEqual(preview.preview_width, 4)
            self.assertEqual(preview.preview_height, 8)
            self.assertEqual(preview.image.shape, (8, 4, 3))

    def test_create_dom_preview_uses_dji_overview_when_available(self) -> None:
        rasterio = _require_rasterio()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "lidars" / "terra_dom" / "dom.tif"
            overview = root / "lidars" / ".temp" / "Reconstruction2d" / "dom_overview.tif"
            source.parent.mkdir(parents=True)
            overview.parent.mkdir(parents=True)
            source_transform = rasterio.transform.from_origin(100.0, 200.0, 1.0, 1.0)
            overview_transform = rasterio.transform.from_origin(100.0, 200.0, 10.0, 10.0)
            source_data = np.full((3, 200, 100), 100, dtype=np.uint8)
            overview_data = np.full((3, 20, 10), 120, dtype=np.uint8)
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=100,
                height=200,
                count=3,
                dtype="uint8",
                crs="EPSG:32651",
                transform=source_transform,
            ) as dataset:
                dataset.write(source_data)
            with rasterio.open(
                overview,
                "w",
                driver="GTiff",
                width=10,
                height=20,
                count=3,
                dtype="uint8",
                transform=overview_transform,
            ) as dataset:
                dataset.write(overview_data)

            preview = create_dom_preview(root, max_width=10, max_height=20)
            a, b, c, d, e, f = preview.preview_to_source_transform
            source_col = a * 5.0 + b * 10.0 + c
            source_row = d * 5.0 + e * 10.0 + f

            self.assertEqual(preview.preview_path, str(overview.resolve()))
            self.assertAlmostEqual(source_col, 50.0)
            self.assertAlmostEqual(source_row, 100.0)

    def test_align_dom_to_axis_preserves_georeference(self) -> None:
        rasterio = _require_rasterio()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dom.tif"
            output = root / "aligned.tif"
            transform = rasterio.transform.from_origin(1000.0, 2000.0, 1.0, 1.0)
            data = np.zeros((3, 100, 100), dtype=np.uint8)
            for index in range(15, 86):
                row = 100 - index
                col = index
                data[:, max(0, row - 1) : min(100, row + 2), max(0, col - 1) : min(100, col + 2)] = np.array(
                    [[[90]], [[140]], [[190]]],
                    dtype=np.uint8,
                )
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=100,
                height=100,
                count=3,
                dtype="uint8",
                crs="EPSG:32651",
                transform=transform,
                nodata=0,
            ) as dataset:
                dataset.write(data)

            result = align_dom_to_axis(
                source,
                output,
                (15.0, 85.0),
                (85.0, 15.0),
                DomAlignmentOptions(padding_pixels=4, mask_sample_max_width=100, mask_sample_max_height=100),
            )

            self.assertTrue(output.exists())
            self.assertTrue(Path(result.metadata_path).exists())
            with rasterio.open(output) as aligned:
                self.assertEqual(aligned.crs.to_epsg(), 32651)
                self.assertGreater(aligned.height, aligned.width)
                self.assertNotEqual(aligned.transform.b, 0.0)
                point1_col, _ = (~aligned.transform) * result.point1_map
                point2_col, _ = (~aligned.transform) * result.point2_map
                self.assertAlmostEqual(point1_col, point2_col, places=6)

    def test_align_dom_to_axis_from_map_points_matches_pixel_points(self) -> None:
        rasterio = _require_rasterio()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dom.tif"
            output = root / "aligned.tif"
            transform = rasterio.transform.from_origin(1000.0, 2000.0, 1.0, 1.0)
            data = np.zeros((3, 100, 100), dtype=np.uint8)
            data[:, 20:80, 45:55] = 180
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=100,
                height=100,
                count=3,
                dtype="uint8",
                crs="EPSG:32651",
                transform=transform,
                nodata=0,
            ) as dataset:
                dataset.write(data)

            result = align_dom_to_axis_from_map_points(
                source,
                output,
                (1050.0, 1980.0),
                (1050.0, 1920.0),
                DomAlignmentOptions(padding_pixels=4),
            )

            self.assertEqual(result.point1_pixel, (50.0, 20.0))
            self.assertEqual(result.point2_pixel, (50.0, 80.0))
            self.assertTrue(output.exists())

    def test_align_dom_to_corridor_from_map_points_uses_four_point_bounds(self) -> None:
        rasterio = _require_rasterio()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dom.tif"
            output = root / "corridor.tif"
            transform = rasterio.transform.from_origin(1000.0, 2000.0, 1.0, 1.0)
            data = np.zeros((3, 120, 80), dtype=np.uint8)
            data[:, 10:110, 30:50] = 180
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=80,
                height=120,
                count=3,
                dtype="uint8",
                crs="EPSG:32651",
                transform=transform,
                nodata=0,
            ) as dataset:
                dataset.write(data)

            result = align_dom_to_corridor_from_map_points(
                source,
                output,
                [
                    (1030.0, 1990.0),
                    (1050.0, 1990.0),
                    (1030.0, 1890.0),
                    (1050.0, 1890.0),
                ],
                DomAlignmentOptions(padding_pixels=0, crop_to_valid_data=False),
            )

            self.assertEqual(result.output_width, 20)
            self.assertEqual(result.output_height, 100)
            with rasterio.open(output) as aligned:
                self.assertEqual(aligned.crs.to_epsg(), 32651)
                tl_col, tl_row = (~aligned.transform) * (1030.0, 1990.0)
                br_col, br_row = (~aligned.transform) * (1050.0, 1890.0)
                self.assertAlmostEqual(tl_col, 0.0, places=6)
                self.assertAlmostEqual(tl_row, 0.0, places=6)
                self.assertAlmostEqual(br_col, 20.0, places=6)
                self.assertAlmostEqual(br_row, 100.0, places=6)

    def test_auto_detect_dom_corridor_points_uses_non_black_region(self) -> None:
        rasterio = _require_rasterio()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dom.tif"
            output = root / "auto_corridor.tif"
            transform = rasterio.transform.from_origin(1000.0, 2000.0, 1.0, 1.0)
            height, width = 180, 120
            rows, cols = np.mgrid[0:height, 0:width]
            center_col, center_row = 60.0, 90.0
            angle = np.deg2rad(18.0)
            long_axis = np.array([np.sin(angle), np.cos(angle)])
            short_axis = np.array([np.cos(angle), -np.sin(angle)])
            centered = np.stack([cols - center_col, rows - center_row], axis=0)
            long_values = centered[0] * long_axis[0] + centered[1] * long_axis[1]
            short_values = centered[0] * short_axis[0] + centered[1] * short_axis[1]
            corridor_mask = (np.abs(long_values) <= 72.0) & (np.abs(short_values) <= 16.0)
            data = np.zeros((3, height, width), dtype=np.uint8)
            data[0, corridor_mask] = 90
            data[1, corridor_mask] = 140
            data[2, corridor_mask] = 190
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=3,
                dtype="uint8",
                crs="EPSG:32651",
                transform=transform,
                nodata=0,
            ) as dataset:
                dataset.write(data)

            detected = auto_detect_dom_corridor_points(
                source,
                max_width=width,
                max_height=height,
                quantile_margin=0.0,
                morphology_iterations=0,
            )

            self.assertEqual(len(detected.points_map), 4)
            top_left, top_right, bottom_left, bottom_right = detected.points_pixel
            top_width = np.hypot(top_right[0] - top_left[0], top_right[1] - top_left[1])
            left_length = np.hypot(bottom_left[0] - top_left[0], bottom_left[1] - top_left[1])
            self.assertGreater(left_length, top_width * 3.0)
            self.assertLess((top_left[1] + top_right[1]) / 2.0, (bottom_left[1] + bottom_right[1]) / 2.0)

            result = align_dom_to_corridor_from_map_points(
                source,
                output,
                detected.points_map,
                DomAlignmentOptions(padding_pixels=0, crop_to_valid_data=False),
            )
            self.assertGreater(result.output_height, result.output_width * 3)
            self.assertTrue(output.exists())

    def test_suggest_annotation_tile_size_uses_multiples_of_32(self) -> None:
        suggestion = suggest_annotation_tile_size(789, 87554)

        self.assertEqual(suggestion.tile_width, 768)
        self.assertEqual(suggestion.tile_height, 3072)
        self.assertEqual(suggestion.stride_x, 768)
        self.assertEqual(suggestion.stride_y, 1536)


def _require_rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - dependency check path
        raise unittest.SkipTest("rasterio is not installed") from exc
    return rasterio


if __name__ == "__main__":
    unittest.main()
