from __future__ import annotations

import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_raw_dom_roi_sample_tiles.py"
    spec = importlib.util.spec_from_file_location("export_raw_dom_roi_sample_tiles", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RawDomRoiSampleTilesTest(unittest.TestCase):
    def test_select_tiles_prioritizes_focus_then_row_bins(self) -> None:
        module = _load_module()
        rows = [
            module.TileRow(i, f"tile_{i}.jpg", i * 10, 0, 10, 10, Path("dom.tif"), 0, 0, 1, 1, float(i + 1), 0.0)
            for i in range(8)
        ]
        rows[4] = rows[4]._replace(focus_intersection_area_m2=100.0)
        rows[5] = rows[5]._replace(focus_intersection_area_m2=50.0)

        selected = module.select_tiles(rows, max_tiles=4, focus_quota=1, row_bins=3)

        self.assertIn(4, {row.tile_id for row in selected})
        self.assertEqual(next(row for row in selected if row.tile_id == 4).selection_reason, "focus_workzone")
        self.assertIn("row_bin", {row.selection_reason for row in selected})
        self.assertEqual(len(selected), 4)

    def test_load_tile_rows_preserves_map_transform_for_fullpass(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "raw_dom_tile_index.csv"
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "tile_id",
                        "tile_name",
                        "row_off",
                        "col_off",
                        "width",
                        "height",
                        "source_path",
                        "x_min",
                        "y_min",
                        "x_max",
                        "y_max",
                        "roi_intersection_area_m2",
                        "tile_transform",
                        "source_transform",
                        "crs",
                        "epsg",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "tile_id": "7",
                        "tile_name": "raw_dom_roi_r000000_c000000.png",
                        "row_off": "0",
                        "col_off": "0",
                        "width": "32",
                        "height": "32",
                        "source_path": "dom.tif",
                        "x_min": "100.0",
                        "y_min": "190.0",
                        "x_max": "110.0",
                        "y_max": "200.0",
                        "roi_intersection_area_m2": "50.0",
                        "tile_transform": "[0.5, 0.0, 100.0, 0.0, -0.5, 200.0]",
                        "source_transform": "[0.5, 0.0, 100.0, 0.0, -0.5, 200.0]",
                        "crs": "EPSG:32651",
                        "epsg": "32651",
                    }
                )

            rows = module.load_tile_rows(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].tile_transform, "[0.5, 0.0, 100.0, 0.0, -0.5, 200.0]")
        self.assertEqual(rows[0].crs, "EPSG:32651")

    def test_select_tiles_max_zero_means_all_tiles(self) -> None:
        module = _load_module()
        rows = [
            module.TileRow(i, f"tile_{i}.png", i * 10, 0, 10, 10, Path("dom.tif"), 0, 0, 1, 1, float(i + 1))
            for i in range(3)
        ]

        selected = module.select_tiles(rows, max_tiles=0, focus_quota=0, row_bins=1)

        self.assertEqual([row.tile_id for row in selected], [0, 1, 2])
        self.assertEqual({row.selection_reason for row in selected}, {"all"})


if __name__ == "__main__":
    unittest.main()
