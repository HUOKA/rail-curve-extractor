from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_crossover_with_las.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("validate_crossover_with_las", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _feature(coords):
    return {
        "type": "Feature",
        "properties": {"connector_id": "CX_TEST"},
        "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in coords]},
    }


def test_rail_line_features_offset_from_centerline_by_half_gauge() -> None:
    module = _load_module()
    samples = module.build_samples(_feature([(0.0, 0.0), (10.0, 0.0)]), gauge_m=1.6, sample_step_m=5.0)

    rails = module.rail_line_features(samples, correction=np.zeros(samples.station.size), gauge_m=1.6, source="test")

    left = rails[0]["geometry"]["coordinates"]
    right = rails[1]["geometry"]["coordinates"]
    assert left[0] == [0.0, -0.8]
    assert right[0] == [0.0, 0.8]
    assert left[-1] == [10.0, -0.8]
    assert right[-1] == [10.0, 0.8]


def test_analyze_connector_uses_both_rail_residuals_as_center_shift() -> None:
    module = _load_module()
    samples = module.build_samples(_feature([(0.0, 0.0), (9.0, 0.0)]), gauge_m=1.6, sample_step_m=1.0)
    residual_edges = np.arange(-0.31, 0.33, 0.02)
    centers = (residual_edges[:-1] + residual_edges[1:]) / 2.0
    peak_index = int(np.argmin(np.abs(centers - 0.11)))
    hist = np.zeros((samples.station.size, 2, residual_edges.size - 1), dtype=np.int64)
    hist[:, :, peak_index] = 30

    analysis = module.analyze_connector(
        samples,
        residual_hist=hist,
        residual_edges=residual_edges,
        corridor_count=np.full(samples.station.size, 100, dtype=np.int64),
        rail_like_count=np.full(samples.station.size, 40, dtype=np.int64),
        gauge_m=1.6,
        min_side_points=10,
        min_correction_samples=3,
        max_correction_m=0.35,
    )

    assert analysis["summary"]["support_samples_both_rails"] == samples.station.size
    assert abs(analysis["summary"]["median_raw_correction_m"] - 0.11) < 0.011
    assert np.allclose(analysis["correction"], analysis["correction"][0])


def test_endpoint_locked_correction_preserves_ends_and_keeps_middle() -> None:
    module = _load_module()
    station = np.arange(0.0, 41.0, 1.0)
    correction = np.full(station.size, 0.2, dtype=float)

    locked = module.endpoint_locked_correction(station, correction, endpoint_lock_m=5.0, endpoint_taper_m=5.0)

    assert np.allclose(locked[:6], 0.0)
    assert np.allclose(locked[-6:], 0.0)
    assert abs(locked[len(locked) // 2] - 0.2) < 1e-9
    assert 0.0 < locked[7] < 0.2
