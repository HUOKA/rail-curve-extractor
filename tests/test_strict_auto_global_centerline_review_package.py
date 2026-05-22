from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "package_strict_auto_global_centerline_review.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("package_strict_auto_global_centerline_review", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _line(coords, **props):
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[x, y] for x, y in coords]},
    }


def test_replacement_keeps_global_line_id_and_uses_refined_geometry() -> None:
    module = _load_module()
    original = _line(
        [(0.0, 0.0), (10.0, 0.0)],
        line_id="TURNOUT_AUTO_001",
        branch_id="AUTO_001",
        network_role="turnout_connector",
        start_band="mainline_2_track",
        end_band="parallel_plus_5m",
    )
    refined = _line(
        [(0.0, 0.0), (5.0, 1.0), (10.0, 2.0)],
        line_id="AUTO_001_outside_rail_offset_centerline",
        branch_id="AUTO_001",
    )

    feature, row = module.build_replacement_feature(
        original,
        refined,
        {
            "valid_outer_rail_ratio": "1.0",
            "invalid_ratio": "0.0",
            "single_left_ratio": "0.1",
            "single_right_ratio": "0.2",
            "max_turn_deg": "0.5",
        },
    )

    props = feature["properties"]
    assert props["line_id"] == "TURNOUT_AUTO_001"
    assert props["network_role"] == "turnout_connector"
    assert props["geom_kind"] == "outside_rail_offset_centerline"
    assert feature["geometry"]["coordinates"] == [[0.0, 0.0], [5.0, 1.0], [10.0, 2.0]]
    assert row["single_fallback_ratio"] == 0.3


def test_boundary_bridge_endpoint_snaps_to_refined_turnout_endpoint() -> None:
    module = _load_module()
    bridge = _line(
        [(-1.0, 0.0), (0.0, 0.0)],
        line_id="BRIDGE_BOUNDARY_BAND_parallel_plus_5m_0_TURNOUT_AUTO_001",
        branch_id="AUTO_001",
        network_role="turnout_boundary_bridge",
    )
    old_turnout = _line([(0.0, 0.0), (10.0, 0.0)], branch_id="AUTO_001")
    refined_turnout = _line([(0.2, 0.1), (10.0, 0.0)], branch_id="AUTO_001")

    row = module.adjust_boundary_bridge(bridge, old_turnout=old_turnout, refined_turnout=refined_turnout)

    assert row is not None
    assert row["adjusted_endpoint"] == "end"
    assert row["turnout_endpoint"] == "start"
    assert bridge["geometry"]["coordinates"][-1] == [0.2, 0.1]
    assert bridge["properties"]["boundary_adjusted_to_refined_turnout"] == 1


def test_boundary_bridge_is_dropped_when_tangent_snap_makes_it_degenerate() -> None:
    module = _load_module()
    bridge = _line(
        [(0.0, 0.0), (0.5, 0.2), (1.0, 0.0)],
        line_id="BRIDGE_BOUNDARY_BAND_parallel_minus_5m_1_TURNOUT_AUTO_001",
        branch_id="AUTO_001",
        network_role="turnout_boundary_bridge",
    )
    old_turnout = _line([(1.0, 0.0), (2.0, 0.0)], branch_id="AUTO_001")
    refined_turnout = _line([(0.0, 0.0), (2.0, 0.0)], branch_id="AUTO_001")

    row = module.adjust_boundary_bridge(bridge, old_turnout=old_turnout, refined_turnout=refined_turnout)

    assert row is not None
    assert row["dropped_bridge"] == 1
    assert bridge["geometry"]["coordinates"] == [[0.0, 0.0], [0.0, 0.0]]


def test_turnout_endpoint_tangency_uses_connected_band() -> None:
    module = _load_module()
    original = _line(
        [(0.0, 1.0), (10.0, 2.5)],
        line_id="TURNOUT_AUTO_001",
        branch_id="AUTO_001",
        network_role="turnout_connector",
        start_band="parallel_minus_5m",
        end_band="mainline_2_track",
    )
    refined = _line(
        [(0.0, 1.0), (1.0, 1.35), (2.0, 1.55), (5.0, 1.8), (10.0, 2.2)],
        line_id="AUTO_001_outside_rail_offset_centerline",
        branch_id="AUTO_001",
    )
    bands = [
        _line(
            [(-5.0, 0.0), (0.0, 0.0), (20.0, 0.0)],
            line_id="BAND_parallel_minus_5m_0",
            band_id="parallel_minus_5m",
            network_role="parallel_straight_track",
        ),
        _line(
            [(-5.0, 5.0), (20.0, 5.0)],
            line_id="BAND_mainline_2_track_0",
            band_id="mainline_2_track",
            network_role="main_through_track",
        ),
    ]

    feature, row = module.build_replacement_feature(
        original,
        refined,
        {},
        band_index=module.build_straight_band_index(bands),
        tangent_snap_max_m=1.75,
        tangent_smooth_taper_m=6.0,
        tangent_window_m=3.0,
    )

    assert feature["geometry"]["coordinates"][0] == [0.0, 0.0]
    assert row["start_tangent_applied"] == 1
    assert row["start_tangent_band_id"] == "parallel_minus_5m"
    assert row["start_tangent_endpoint_shift_m"] == 1.0
    assert row["start_tangent_after_angle_deg"] < row["start_tangent_before_angle_deg"]
    assert row["start_tangent_after_angle_deg"] < 2.0
    assert row["end_tangent_applied"] == 0


def test_same_band_occlusion_bridge_is_added_for_long_internal_gap() -> None:
    module = _load_module()
    features = [
        _line(
            [(0.0, 0.0), (100.0, 0.0)],
            line_id="BAND_parallel_minus_5m_1",
            band_id="parallel_minus_5m",
            network_role="parallel_straight_track",
            station_min_m=0.0,
            station_max_m=100.0,
        ),
        _line(
            [(170.0, 0.0), (260.0, 0.0)],
            line_id="BAND_parallel_minus_5m_2",
            band_id="parallel_minus_5m",
            network_role="parallel_straight_track",
            station_min_m=170.0,
            station_max_m=260.0,
        ),
    ]

    bridges, rows = module.build_same_band_occlusion_bridges(features, min_gap_m=60.0, max_gap_m=85.0)

    assert len(bridges) == 1
    assert rows[0]["gap_m"] == 70.0
    props = bridges[0]["properties"]
    assert props["bridge_kind"] == "occlusion_bridge"
    assert props["qa_status"] == "bridge_needs_review"
    assert bridges[0]["geometry"]["coordinates"] == [[100.0, 0.0], [170.0, 0.0]]


def test_same_band_occlusion_bridge_skips_short_gap() -> None:
    module = _load_module()
    features = [
        _line(
            [(0.0, 0.0), (100.0, 0.0)],
            line_id="left",
            band_id="parallel_minus_5m",
            network_role="parallel_straight_track",
            station_min_m=0.0,
            station_max_m=100.0,
        ),
        _line(
            [(140.0, 0.0), (260.0, 0.0)],
            line_id="right",
            band_id="parallel_minus_5m",
            network_role="parallel_straight_track",
            station_min_m=140.0,
            station_max_m=260.0,
        ),
    ]

    bridges, rows = module.build_same_band_occlusion_bridges(features, min_gap_m=60.0, max_gap_m=85.0)

    assert bridges == []
    assert rows == []
