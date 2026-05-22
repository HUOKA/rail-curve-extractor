from __future__ import annotations

import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "extract_centerline_candidates.py"
    spec = importlib.util.spec_from_file_location("extract_centerline_candidates", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CenterlineCandidatesTest(unittest.TestCase):
    def test_extract_track_candidates_pairs_parallel_rails(self) -> None:
        module = _load_module()
        mask = np.zeros((80, 64), dtype=bool)
        mask[:, 10:13] = True
        mask[:, 30:33] = True

        rail_rows = module.extract_rail_centers(mask, row_step=16, column_threshold=0.2, min_run_pixels=2)
        target_gap = module.estimate_target_gap({"synthetic.png": rail_rows}, 10.0, 40.0)
        candidates = module.extract_track_candidates(rail_rows, 10.0, 40.0, target_gap)

        self.assertGreater(len(rail_rows), 0)
        self.assertGreater(len(candidates), 0)
        first = candidates[0]
        self.assertAlmostEqual(first["gap_px"], 20.0, delta=1.0)
        self.assertAlmostEqual(first["x"], 21.0, delta=1.0)

    def test_group_candidate_tracks_filters_short_fragments(self) -> None:
        module = _load_module()
        rows = [
            {"candidate_id": 0.0, "y": 8.0, "x": 20.0, "left_x": 10.0, "right_x": 30.0, "gap_px": 20.0, "confidence": 0.9},
            {"candidate_id": 0.0, "y": 24.0, "x": 21.0, "left_x": 11.0, "right_x": 31.0, "gap_px": 20.0, "confidence": 0.9},
            {"candidate_id": 0.0, "y": 40.0, "x": 22.0, "left_x": 12.0, "right_x": 32.0, "gap_px": 20.0, "confidence": 0.9},
            {"candidate_id": 1.0, "y": 8.0, "x": 55.0, "left_x": 45.0, "right_x": 65.0, "gap_px": 20.0, "confidence": 0.8},
        ]

        grouped = module.group_candidate_tracks(rows, row_step=16, max_x_jump=8.0, max_row_gap=1, min_track_points=3)

        self.assertEqual(len(grouped), 3)
        self.assertEqual({row["candidate_id"] for row in grouped}, {0.0})
        self.assertEqual([row["x"] for row in grouped], [20.0, 21.0, 22.0])

    def test_extract_track_candidates_uses_non_overlapping_pairs(self) -> None:
        module = _load_module()
        rail_rows = [
            {"y": 8.0, "x": 10.0, "score": 1.0},
            {"y": 8.0, "x": 30.0, "score": 1.0},
            {"y": 8.0, "x": 50.0, "score": 1.0},
            {"y": 8.0, "x": 70.0, "score": 1.0},
        ]

        candidates = module.extract_track_candidates(rail_rows, min_gap=15.0, max_gap=25.0, target_gap=20.0)

        self.assertEqual(len(candidates), 2)
        self.assertEqual([row["x"] for row in candidates], [20.0, 60.0])

    def test_extract_track_candidates_can_filter_by_map_gauge(self) -> None:
        module = _load_module()
        rail_rows = [
            {"y": 8.0, "x": 10.0, "score": 1.0},
            {"y": 8.0, "x": 30.0, "score": 1.0},
            {"y": 8.0, "x": 90.0, "score": 1.0},
        ]
        transform = (0.05, 0.0, 1000.0, 0.0, -0.05, 2000.0)

        candidates = module.extract_track_candidates(
            rail_rows,
            min_gap=15.0,
            max_gap=70.0,
            target_gap=20.0,
            map_transform=transform,
            target_gauge_m=1.0,
            gauge_tolerance_m=0.1,
        )

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["gap_m"], 1.0, places=6)
        self.assertAlmostEqual(candidates[0]["x"], 20.0, delta=1.0)

    def test_apply_yolo_ignore_labels_clears_ignored_polygon(self) -> None:
        module = _load_module()
        mask = np.ones((10, 10), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = Path(tmp_dir) / "sample.txt"
            label_path.write_text("1 0 0 0.5 0 0.5 1 0 1\n", encoding="utf-8")

            filtered = module.apply_yolo_ignore_labels(mask, label_path, {1})

        self.assertFalse(filtered[:, :5].any())
        self.assertTrue(filtered[:, 6:].all())

    def test_write_overlay_uses_png_for_lossless_qa(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "tile.png"
            Image.new("RGB", (32, 24), (20, 30, 40)).save(image_path)
            output_path = root / "overlay.png"
            rail_rows = [
                {"x": 8.0, "y": 4.0, "score": 1.0},
                {"x": 24.0, "y": 4.0, "score": 1.0},
                {"x": 8.0, "y": 20.0, "score": 1.0},
                {"x": 24.0, "y": 20.0, "score": 1.0},
            ]
            candidates = [
                {"candidate_id": 0.0, "x": 16.0, "y": 4.0, "left_x": 8.0, "right_x": 24.0, "gap_px": 16.0, "confidence": 0.9},
                {"candidate_id": 0.0, "x": 16.0, "y": 20.0, "left_x": 8.0, "right_x": 24.0, "gap_px": 16.0, "confidence": 0.9},
            ]

            module.write_overlay(image_path, rail_rows, candidates, output_path)

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as overlay:
                self.assertEqual(overlay.format, "PNG")

    def test_load_tile_lookup_uses_selected_tile_index_image_name(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir)
            selected_csv = dataset_dir / "selected_tile_index.csv"
            (dataset_dir / "summary.json").write_text(
                json.dumps({"selected_tile_index_csv": str(selected_csv)}),
                encoding="utf-8",
            )
            with selected_csv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "tile_name",
                        "image_name",
                        "tile_transform",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "tile_name": "raw_dom_roi_r000000_c000000.png",
                        "image_name": "raw_dom_roi_r000000_c000000.jpg",
                        "tile_transform": "[0.5, 0.0, 100.0, 0.0, -0.5, 200.0]",
                    }
                )

            lookup = module.load_tile_lookup(dataset_dir)

        record = module.match_tile_record(lookup, "raw_dom_roi_r000000_c000000.jpg")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["tile_transform"], (0.5, 0.0, 100.0, 0.0, -0.5, 200.0))


if __name__ == "__main__":
    unittest.main()
