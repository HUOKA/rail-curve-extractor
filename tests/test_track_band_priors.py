from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_track_band_priors.py"
    spec = importlib.util.spec_from_file_location("build_track_band_priors", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature(coords, *, candidate_id: int = 0):
    return {
        "type": "Feature",
        "properties": {"candidate_id": candidate_id, "point_count": len(coords), "mean_confidence": 0.8},
        "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in coords]},
    }


def test_assign_band_uses_nearest_configured_offset() -> None:
    module = _load_module()

    assert module.assign_band(-5.2)["band_id"] == "parallel_minus_5m"
    assert module.assign_band(0.4)["band_id"] == "mainline_2_track"
    assert module.assign_band(5.1)["band_id"] == "parallel_plus_5m"
    assert module.assign_band(8.5) is None


def test_merge_intervals_bridges_small_gaps_and_preserves_large_gaps() -> None:
    module = _load_module()

    merged = module.merge_intervals([(0.0, 10.0, 1), (15.0, 20.0, 1), (120.0, 140.0, 1)], gap_m=20.0)

    assert merged == [(0.0, 20.0, 2), (120.0, 140.0, 1)]


def test_reviewed_bridge_gap_is_disabled_by_default() -> None:
    module = _load_module()

    merged = module.merge_intervals_with_review_bridges(
        [(1442.0, 1723.0, 1), (1848.0, 1875.0, 1), (1946.0, 1980.0, 1)],
        gap_m=40.0,
        band_id="parallel_minus_5m",
    )

    assert len(merged) == 3
    assert all(item["bridge_gap_count"] == 0 for item in merged)


def test_reviewed_bridge_gap_requires_explicit_compatibility_flag() -> None:
    module = _load_module()

    merged = module.merge_intervals_with_review_bridges(
        [(1442.0, 1723.0, 1), (1848.0, 1875.0, 1), (1946.0, 1980.0, 1)],
        gap_m=40.0,
        band_id="parallel_minus_5m",
        allow_reviewed_bridges=True,
    )

    assert len(merged) == 2
    assert merged[0]["station_min_m"] == 1442.0
    assert merged[0]["station_max_m"] == 1875.0
    assert merged[0]["bridge_gap_count"] == 1
    assert merged[0]["bridge_mid_m"] == 1785.5
    assert merged[0]["qa_note"] == "reviewed_s1810_straight_gap"
    assert merged[1]["station_min_m"] == 1946.0


def test_local_offset_profile_interpolates_inside_bridge_without_global_drift() -> None:
    module = _load_module()
    profile = module.build_offset_profile([(1700.0, -5.2), (1720.0, -5.1), (1850.0, -4.9)])

    assert module.estimate_offset_at(1710.0, profile=profile, default_offset=-5.0, window_m=20.0) == -5.15
    assert round(module.estimate_offset_at(1785.0, profile=profile, default_offset=-5.0, window_m=20.0), 3) == -5.0
    assert module.estimate_offset_at(2500.0, profile=profile, default_offset=-5.0, window_m=20.0) == -5.0


def test_classify_candidates_records_station_and_offset_properties() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (100.0, 0.0))
    features = [
        _feature([(0.0, -5.0), (50.0, -5.1)], candidate_id=1),
        _feature([(0.0, 8.0), (50.0, 8.0)], candidate_id=2),
    ]

    classified, band_offsets = module.classify_candidates(features, guide=guide, min_station_span_m=5.0)

    assert len(classified) == 1
    props = classified[0]["properties"]
    assert props["band_id"] == "parallel_minus_5m"
    assert props["station_span_m"] == 50.0
    assert band_offsets["parallel_minus_5m"]


def test_build_band_centerlines_drops_short_side_band_intervals_before_numbering() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (120.0, 0.0))
    short = _feature([(0.0, 5.0), (20.0, 5.0)], candidate_id=1)
    short["properties"].update({"band_id": "parallel_plus_5m", "station_min_m": 0.0, "station_max_m": 20.0})
    long = _feature([(60.0, 5.0), (110.0, 5.0)], candidate_id=2)
    long["properties"].update({"band_id": "parallel_plus_5m", "station_min_m": 60.0, "station_max_m": 110.0})

    centerlines = module.build_band_centerlines(
        [short, long],
        guide=guide,
        band_offsets={"parallel_plus_5m": [5.0]},
        sample_step_m=10.0,
        merge_gap_m=5.0,
        local_offset_window_m=20.0,
        min_centerline_span_m=35.0,
    )

    plus = [feature for feature in centerlines if feature["properties"]["band_id"] == "parallel_plus_5m"]
    assert len(plus) == 1
    assert plus[0]["properties"]["interval_id"] == 0
    assert plus[0]["properties"]["station_min_m"] == 60.0


def test_qa_targets_do_not_include_user_review_points_by_default() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (100.0, 0.0))
    feature = _feature([(10.0, 5.0), (50.0, 5.0)], candidate_id=1)
    feature["properties"].update({"band_id": "parallel_plus_5m", "station_min_m": 10.0, "station_max_m": 50.0})

    targets = module.build_qa_targets([feature], guide=guide)

    assert targets
    assert not [target for target in targets if str(target["name"]).startswith("user_")]


def test_mainline_centerline_uses_robust_straight_support_fit() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (100.0, 0.0))
    mainline = _feature([(0.0, -0.03), (50.0, -0.03), (100.0, -0.03)], candidate_id=1)
    mainline["properties"].update({"band_id": "mainline_2_track", "station_min_m": 0.0, "station_max_m": 100.0})

    centerlines = module.build_band_centerlines(
        [mainline],
        guide=guide,
        band_offsets={"mainline_2_track": [-0.03]},
        sample_step_m=25.0,
        merge_gap_m=5.0,
        local_offset_window_m=20.0,
        mainline_local_offset_window_m=30.0,
        mainline_max_local_correction_m=0.08,
        min_centerline_span_m=35.0,
    )

    main = [feature for feature in centerlines if feature["properties"]["band_id"] == "mainline_2_track"][0]
    ys = [coord[1] for coord in main["geometry"]["coordinates"]]
    assert main["properties"]["fit_mode"] == "robust_straight_line"
    assert max(abs(y + 0.03) for y in ys) < 1e-6


def test_parallel_straight_fit_outputs_collinear_line_not_local_wiggle() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (100.0, 0.0))
    wiggly = _feature([(0.0, 5.0), (50.0, 5.2), (100.0, 5.0)], candidate_id=1)
    wiggly["properties"].update({"band_id": "parallel_plus_5m", "station_min_m": 0.0, "station_max_m": 100.0})

    centerlines = module.build_band_centerlines(
        [wiggly],
        guide=guide,
        band_offsets={"parallel_plus_5m": [5.0, 5.2, 5.0]},
        sample_step_m=25.0,
        merge_gap_m=5.0,
        local_offset_window_m=20.0,
        min_centerline_span_m=35.0,
    )

    plus = [feature for feature in centerlines if feature["properties"]["band_id"] == "parallel_plus_5m"][0]
    ys = [coord[1] for coord in plus["geometry"]["coordinates"]]
    assert plus["properties"]["fit_mode"] == "robust_straight_line"
    assert max(ys) - min(ys) < 1e-6
    assert abs(ys[len(ys) // 2] - 5.2) > 0.05
