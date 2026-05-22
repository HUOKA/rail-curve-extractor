from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_auto_turnout_crossover_evidence.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("build_auto_turnout_crossover_evidence", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _line(points, **props):
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[x, y] for x, y in points]},
    }


def test_fragment_cluster_builds_tangent_connector_between_active_bands() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (100.0, 0.0))
    features = [
        _line(
            [(30.0, 1.0), (40.0, 2.0), (50.0, 3.0), (60.0, 4.0)],
            candidate_id="frag",
            mean_confidence=0.8,
        )
    ]

    transitions = module.detect_fragment_cluster_transitions(
        features,
        guide=guide,
        band_centers={"mainline_2_track": 0.0, "parallel_plus_5m": 5.0},
        band_intervals=[
            {"band_id": "mainline_2_track", "station_min_m": 0.0, "station_max_m": 100.0, "role": "accepted_mainline"},
            {"band_id": "parallel_plus_5m", "station_min_m": 0.0, "station_max_m": 100.0, "role": "parallel_track"},
        ],
        min_station_span_m=5.0,
        min_offset_span_m=0.75,
        min_abs_slope=0.012,
        max_abs_slope=0.35,
        min_points=4,
        cluster_gap_m=45.0,
        band_margin_m=2.3,
        local_context_trend_max_distance_m=1.35,
        local_context_station_margin_m=3.0,
        local_context_offset_margin_m=0.75,
        endpoint_tangent_padding_m=10.0,
        curve_step_m=5.0,
    )

    assert len(transitions) == 1
    transition = transitions[0]
    assert transition["start_band"] == "mainline_2_track"
    assert transition["end_band"] == "parallel_plus_5m"
    assert transition["station_min_m"] < 30.0
    assert transition["station_max_m"] > 60.0
    assert transition["offset_start_m"] == 0.0
    assert transition["offset_end_m"] == 5.0
    sampled = [guide.station_offset(point) for point in transition["coords"]]
    station, offset = min(sampled, key=lambda item: abs(item[0] - 50.0))
    assert abs(station - 50.0) < 3.0
    assert abs(offset - 3.0) < 0.35
    evidence_sampled = [guide.station_offset(point) for point in transition["evidence_coords"]]
    assert min(offset for _, offset in evidence_sampled) > 0.75
    assert max(offset for _, offset in evidence_sampled) < 4.25


def test_fragment_cluster_skips_inactive_intermediate_band_for_outer_transition() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (200.0, 0.0))
    features = [
        _line(
            [(70.0, 9.5), (80.0, 8.5), (90.0, 7.0), (100.0, 5.5)],
            candidate_id="outer_frag",
            mean_confidence=0.8,
        )
    ]

    transitions = module.detect_fragment_cluster_transitions(
        features,
        guide=guide,
        band_centers={"mainline_2_track": 0.0, "parallel_plus_5m": 5.0, "possible_outer_plus_10m": 10.0},
        band_intervals=[
            {"band_id": "mainline_2_track", "station_min_m": 0.0, "station_max_m": 200.0, "role": "accepted_mainline"},
            {"band_id": "possible_outer_plus_10m", "station_min_m": 40.0, "station_max_m": 75.0, "role": "diagnostic_candidate"},
        ],
        min_station_span_m=5.0,
        min_offset_span_m=0.75,
        min_abs_slope=0.012,
        max_abs_slope=0.35,
        min_points=4,
        cluster_gap_m=45.0,
        band_margin_m=2.3,
        local_context_trend_max_distance_m=1.35,
        local_context_station_margin_m=3.0,
        local_context_offset_margin_m=0.75,
        endpoint_tangent_padding_m=10.0,
        curve_step_m=5.0,
    )

    assert len(transitions) == 1
    transition = transitions[0]
    assert transition["start_band"] == "possible_outer_plus_10m"
    assert transition["end_band"] == "mainline_2_track"
    assert transition["station_min_m"] == 40.0
    assert transition["offset_start_m"] == 10.0
    assert transition["offset_end_m"] == 0.0


def test_fragment_cluster_uses_short_local_candidates_as_curve_constraints() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (100.0, 0.0))
    features = [
        _line(
            [(20.0, 1.0), (30.0, 1.4), (70.0, 3.6), (80.0, 4.0)],
            candidate_id="long_fragment",
            mean_gap_m=1.55,
            mean_confidence=0.8,
        ),
        _line(
            [(49.0, 3.45), (51.0, 3.55)],
            candidate_id="short_local",
            mean_gap_m=1.55,
            mean_confidence=0.7,
        ),
        _line(
            [(49.0, -2.5), (51.0, -2.4)],
            candidate_id="unrelated_local",
            mean_gap_m=1.55,
            mean_confidence=0.7,
        ),
    ]

    transitions = module.detect_fragment_cluster_transitions(
        features,
        guide=guide,
        band_centers={"mainline_2_track": 0.0, "parallel_plus_5m": 5.0},
        band_intervals=[
            {"band_id": "mainline_2_track", "station_min_m": 0.0, "station_max_m": 100.0, "role": "accepted_mainline"},
            {"band_id": "parallel_plus_5m", "station_min_m": 0.0, "station_max_m": 100.0, "role": "parallel_track"},
        ],
        min_station_span_m=5.0,
        min_offset_span_m=0.75,
        min_abs_slope=0.012,
        max_abs_slope=0.35,
        min_points=4,
        cluster_gap_m=45.0,
        band_margin_m=2.3,
        local_context_trend_max_distance_m=1.35,
        local_context_station_margin_m=3.0,
        local_context_offset_margin_m=0.75,
        endpoint_tangent_padding_m=0.0,
        transition_curve_mode="evidence_guided",
        curve_step_m=2.0,
    )

    assert len(transitions) == 1
    transition = transitions[0]
    sampled = [guide.station_offset(point) for point in transition["coords"]]
    station, offset = min(sampled, key=lambda item: abs(item[0] - 50.0))
    assert abs(station - 50.0) <= 1.0
    assert offset > 3.25
    assert "short_local" in transition["local_context_candidate_id"]
    assert "unrelated_local" not in transition["local_context_candidate_id"]


def test_route_curve_does_not_snap_to_short_local_frog_candidate() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (100.0, 0.0))
    features = [
        _line(
            [(20.0, 1.0), (30.0, 1.4), (70.0, 3.6), (80.0, 4.0)],
            candidate_id="long_fragment",
            mean_gap_m=1.55,
            mean_confidence=0.8,
        ),
        _line(
            [(49.0, 3.45), (51.0, 3.55)],
            candidate_id="short_local",
            mean_gap_m=1.55,
            mean_confidence=0.7,
        ),
    ]

    transitions = module.detect_fragment_cluster_transitions(
        features,
        guide=guide,
        band_centers={"mainline_2_track": 0.0, "parallel_plus_5m": 5.0},
        band_intervals=[
            {"band_id": "mainline_2_track", "station_min_m": 0.0, "station_max_m": 100.0, "role": "accepted_mainline"},
            {"band_id": "parallel_plus_5m", "station_min_m": 0.0, "station_max_m": 100.0, "role": "parallel_track"},
        ],
        min_station_span_m=5.0,
        min_offset_span_m=0.75,
        min_abs_slope=0.012,
        max_abs_slope=0.35,
        min_points=4,
        cluster_gap_m=45.0,
        band_margin_m=2.3,
        local_context_trend_max_distance_m=1.35,
        local_context_station_margin_m=3.0,
        local_context_offset_margin_m=0.75,
        endpoint_tangent_padding_m=0.0,
        transition_curve_mode="route_curve",
        route_curve_fraction=0.34,
        curve_step_m=2.0,
    )

    assert len(transitions) == 1
    transition = transitions[0]
    sampled = [guide.station_offset(point) for point in transition["coords"]]
    station, offset = min(sampled, key=lambda item: abs(item[0] - 50.0))
    assert abs(station - 50.0) <= 1.0
    assert offset < 2.9
    assert transition["transition_curve_mode"] == "route_curve"
    assert "short_local" in transition["local_context_candidate_id"]
