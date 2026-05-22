from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_handheld_las_constraints.py"
    spec = importlib.util.spec_from_file_location("analyze_handheld_las_constraints", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HandheldLasConstraintsTest(unittest.TestCase):
    def test_find_profile_peaks_ignores_empty_profiles(self) -> None:
        module = _load_module()
        profile = [
            {"value_m": -1.0, "count": 0, "smooth_count": 0.0},
            {"value_m": 0.0, "count": 0, "smooth_count": 0.0},
            {"value_m": 1.0, "count": 0, "smooth_count": 0.0},
        ]

        peaks = module.find_profile_peaks(profile, min_distance_m=0.5, min_prominence_fraction=0.1)

        self.assertEqual(peaks, [])

    def test_exclusion_windows_mask_projected_points(self) -> None:
        module = _load_module()
        axis = module.Axis(
            origin=np.asarray([0.0, 0.0]),
            longitudinal=np.asarray([1.0, 0.0]),
            lateral=np.asarray([0.0, 1.0]),
        )
        geojson = """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"workzone_id": "switch_001"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[10, 20], [12, 20], [12, 22], [10, 22], [10, 20]]]
      }
    }
  ]
}
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "workzones.geojson"
            path.write_text(geojson, encoding="utf-8")
            windows = module.load_exclusion_windows(path, axis, buffer_m=1.0)

        s_values = np.asarray([8.9, 9.5, 11.0, 13.1])
        t_values = np.asarray([21.0, 21.0, 21.0, 21.0])
        mask = module.points_in_exclusion_windows(s_values, t_values, windows)

        self.assertEqual(len(windows), 1)
        self.assertEqual(mask.tolist(), [False, True, True, False])


if __name__ == "__main__":
    unittest.main()
