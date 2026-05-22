from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_mainline_prior.py"
    spec = importlib.util.spec_from_file_location("build_mainline_prior", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vertical_line(x: float, y0: float, y1: float, *, step: float = 10.0, **props):
    count = int((y1 - y0) / step) + 1
    coords = [[x, y0 + index * step] for index in range(count)]
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def test_semseg_auto_guide_selects_two_sided_through_track() -> None:
    module = _load_module()
    features = [
        _vertical_line(-5.0, 0.0, 1000.0, candidate_id=1),
        _vertical_line(0.0, 0.0, 1000.0, candidate_id=2),
        _vertical_line(5.0, 250.0, 900.0, candidate_id=3),
    ]

    guide, report = module.build_semseg_auto_guide(
        features,
        min_peak_count=10,
        neighbor_overlap_m=200.0,
    )

    assert abs(guide.start[0]) < 0.25
    assert abs(guide.end[0]) < 0.25
    assert guide.length > 950.0
    assert report["track_peak_count"] == 3
    assert report["track_peaks"][report["selected_cluster_index"]]["two_sided_neighbor"] is True


def test_semseg_auto_guide_falls_back_to_longest_single_peak() -> None:
    module = _load_module()
    features = [
        _vertical_line(-5.0, 0.0, 200.0, candidate_id=1),
        _vertical_line(0.0, 0.0, 1000.0, candidate_id=2),
    ]

    guide, report = module.build_semseg_auto_guide(features, min_peak_count=10)

    assert abs(guide.start[0]) < 0.25
    assert abs(guide.end[0]) < 0.25
    assert report["track_peaks"][report["selected_cluster_index"]]["coverage_m"] > 950.0
