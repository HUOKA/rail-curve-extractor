from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "refine_centerline_graph.py"
    spec = importlib.util.spec_from_file_location("refine_centerline_graph", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RefineCenterlineGraphTest(unittest.TestCase):
    def test_overlapping_segments_with_same_lateral_position_are_compatible(self) -> None:
        module = _load_module()
        coords = [(0.0, y) for y in range(0, 100, 10)] + [(0.2, y) for y in range(50, 150, 10)]
        axes = module.estimate_axes(coords)
        first = module.build_segment(0, [(0.0, y) for y in range(0, 100, 10)], {"mean_confidence": 0.9}, axes)
        second = module.build_segment(1, [(0.2, y) for y in range(50, 150, 10)], {"mean_confidence": 0.8}, axes)

        self.assertTrue(module.are_segments_compatible(first, second, max_lateral_gap=1.0, max_longitudinal_gap=5.0, max_angle_deg=10.0))

    def test_parallel_segments_with_large_lateral_gap_are_not_compatible(self) -> None:
        module = _load_module()
        coords = [(0.0, y) for y in range(0, 100, 10)] + [(5.0, y) for y in range(50, 150, 10)]
        axes = module.estimate_axes(coords)
        first = module.build_segment(0, [(0.0, y) for y in range(0, 100, 10)], {"mean_confidence": 0.9}, axes)
        second = module.build_segment(1, [(5.0, y) for y in range(50, 150, 10)], {"mean_confidence": 0.8}, axes)

        self.assertFalse(module.are_segments_compatible(first, second, max_lateral_gap=1.0, max_longitudinal_gap=5.0, max_angle_deg=10.0))

    def test_build_chains_selects_longest_main_chain(self) -> None:
        module = _load_module()
        raw = [
            [(0.0, y) for y in range(0, 100, 10)],
            [(0.1, y) for y in range(80, 180, 10)],
            [(6.0, y) for y in range(0, 60, 10)],
        ]
        axes = module.estimate_axes([point for line in raw for point in line])
        segments = [module.build_segment(index, coords, {"mean_confidence": 0.9}, axes) for index, coords in enumerate(raw)]
        components = module.build_components(segments, max_lateral_gap=1.0, max_longitudinal_gap=5.0, max_angle_deg=10.0)
        chains = module.build_chains(components, segments, axes, bin_size=1.0, min_chain_extent=10.0)
        main = module.select_main_chain(chains)

        self.assertIsNotNone(main)
        assert main is not None
        self.assertGreater(main["properties"]["s_extent_m"], 150.0)
        self.assertEqual(main["properties"]["role"], "main")


if __name__ == "__main__":
    unittest.main()
