from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from affine import Affine


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_raw_dom_corridor_roi.py"
    spec = importlib.util.spec_from_file_location("build_raw_dom_corridor_roi", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RawDomCorridorRoiTest(unittest.TestCase):
    def test_axis_offsets_cover_interval_with_final_tile(self) -> None:
        module = _load_module()

        offsets = module.axis_offsets_for_interval(start=100, stop=900, max_length=1000, tile_size=300, stride=200)

        self.assertEqual(offsets, [100, 300, 500, 600])

    def test_raster_footprint_uses_rotated_corners_not_axis_aligned_bounds(self) -> None:
        module = _load_module()

        class Dataset:
            width = 10
            height = 100
            transform = Affine(1, -0.2, 100, 0.2, 1, 200)

        polygon = module.raster_footprint_polygon(Dataset())

        self.assertAlmostEqual(polygon.area, 1040.0, places=6)
        self.assertEqual(len(list(polygon.exterior.coords)), 5)


if __name__ == "__main__":
    unittest.main()
