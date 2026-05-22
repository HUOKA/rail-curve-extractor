from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_deeplab_topology_centerline_network.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("build_deeplab_topology_centerline_network", script_path)
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


def test_default_gauge_pair_input_is_all_turnout_evidence() -> None:
    module = _load_module()

    assert "deeplab_gauge_pair_turnouts_v1" in str(module.DEFAULT_GAUGE_PAIR)
    assert "ta08" not in str(module.DEFAULT_GAUGE_PAIR).lower()


def test_select_track_band_features_excludes_diagnostic_outer_band() -> None:
    module = _load_module()
    features = [
        _line([(0, 0), (10, 0)], band_id="mainline_2_track"),
        _line([(0, 5), (10, 5)], band_id="parallel_plus_5m"),
        _line([(0, 10), (10, 10)], band_id="possible_outer_plus_10m"),
    ]

    selected = module.select_track_band_features(features)

    assert [feature["properties"]["band_id"] for feature in selected] == ["mainline_2_track", "parallel_plus_5m"]


def test_measure_support_detects_close_and_far_evidence() -> None:
    module = _load_module()
    target = [(0.0, 0.0), (20.0, 0.0)]
    close_evidence = [((0.0, 0.2), (20.0, 0.2))]
    far_evidence = [((0.0, 5.0), (20.0, 5.0))]

    close = module.measure_support(target, evidence_segments=close_evidence, threshold_m=0.85, sample_step_m=5.0)
    far = module.measure_support(target, evidence_segments=far_evidence, threshold_m=0.85, sample_step_m=5.0)

    assert close["support_ratio"] == 1.0
    assert close["mean_distance_m"] < 0.25
    assert far["support_ratio"] == 0.0
    assert far["max_unsupported_gap_m"] >= 20.0


def test_build_topology_features_keeps_turnout_but_flags_low_support() -> None:
    module = _load_module()
    band = _line([(0, 0), (20, 0)], band_id="mainline_2_track", interval_id=0)
    turnout = _line([(0, 5), (20, 5)], branch_id="TA08", qa_status="self_review_pass_low_support")
    evidence = [_line([(0, 0.1), (20, 0.1)], role="mainline")]

    features = module.build_topology_features(
        band_features=[band],
        turnout_features=[turnout],
        evidence_features=evidence,
        support_threshold_m=0.85,
        sample_step_m=5.0,
    )

    assert [feature["properties"]["network_role"] for feature in features] == ["main_through_track", "turnout_connector"]
    assert features[0]["properties"]["deeplab_support_ratio"] == 1.0
    assert features[1]["properties"]["risk_flag"] == "review_priority_low_support_turnout"


def test_build_topology_features_promotes_same_band_gap_with_evidence() -> None:
    module = _load_module()
    left = _line([(0, 0), (10, 0)], band_id="parallel_plus_5m", interval_id=0, station_min_m=0.0, station_max_m=10.0)
    right = _line([(20, 0), (30, 0)], band_id="parallel_plus_5m", interval_id=1, station_min_m=20.0, station_max_m=30.0)
    evidence = [_line([(10, 0.1), (20, 0.1)], role="support")]

    features = module.build_topology_features(
        band_features=[left, right],
        turnout_features=[],
        evidence_features=evidence,
        support_threshold_m=0.85,
        sample_step_m=2.5,
        bridge_min_gap_m=2.0,
        bridge_max_gap_m=20.0,
        bridge_evidence_support=0.45,
    )

    bridges = [feature for feature in features if feature["properties"]["network_role"] == "straight_gap_bridge"]
    assert len(bridges) == 1
    assert bridges[0]["properties"]["risk_flag"] == "evidence_promoted_bridge"
    assert bridges[0]["properties"]["deeplab_support_ratio"] == 1.0


def test_build_topology_features_bridges_short_same_band_gap_as_occlusion_when_evidence_is_missing() -> None:
    module = _load_module()
    left = _line([(0, 0), (10, 0)], band_id="parallel_minus_5m", interval_id=0, station_min_m=0.0, station_max_m=10.0)
    right = _line([(20, 0), (30, 0)], band_id="parallel_minus_5m", interval_id=1, station_min_m=20.0, station_max_m=30.0)

    features = module.build_topology_features(
        band_features=[left, right],
        turnout_features=[],
        evidence_features=[],
        support_threshold_m=0.85,
        sample_step_m=2.5,
        bridge_min_gap_m=2.0,
        bridge_max_gap_m=20.0,
        bridge_evidence_support=0.45,
    )

    bridges = [feature for feature in features if feature["properties"]["network_role"] == "straight_gap_bridge"]
    assert len(bridges) == 1
    assert bridges[0]["properties"]["risk_flag"] == "occlusion_bridge"
    assert bridges[0]["properties"]["gap_m"] == 10.0


def test_build_topology_features_drops_long_weak_same_band_gap_when_strict() -> None:
    module = _load_module()
    left = _line([(0, 0), (10, 0)], band_id="parallel_minus_5m", interval_id=0, station_min_m=0.0, station_max_m=10.0)
    right = _line([(80, 0), (90, 0)], band_id="parallel_minus_5m", interval_id=1, station_min_m=80.0, station_max_m=90.0)

    features = module.build_topology_features(
        band_features=[left, right],
        turnout_features=[],
        evidence_features=[],
        support_threshold_m=0.85,
        sample_step_m=5.0,
        bridge_min_gap_m=2.0,
        bridge_max_gap_m=95.0,
        bridge_evidence_support=0.45,
        weak_gap_bridge_max_gap_m=60.0,
    )

    assert not [feature for feature in features if feature["properties"]["network_role"] == "straight_gap_bridge"]


def test_build_topology_features_does_not_bridge_through_turnout_range() -> None:
    module = _load_module()
    left = _line([(0, 0), (10, 0)], band_id="parallel_plus_5m", interval_id=0, station_min_m=0.0, station_max_m=10.0)
    right = _line([(20, 0), (30, 0)], band_id="parallel_plus_5m", interval_id=1, station_min_m=20.0, station_max_m=30.0)
    turnout = _line([(12, 0), (18, 0)], branch_id="TA99", station_min_m=12.0, station_max_m=18.0)

    features = module.build_topology_features(
        band_features=[left, right],
        turnout_features=[turnout],
        evidence_features=[_line([(10, 0.1), (20, 0.1)], role="support")],
        support_threshold_m=0.85,
        sample_step_m=2.5,
        bridge_min_gap_m=2.0,
        bridge_max_gap_m=20.0,
        bridge_evidence_support=0.45,
        bridge_turnout_clearance_m=0.0,
    )

    assert not [feature for feature in features if feature["properties"]["network_role"] == "straight_gap_bridge"]


def test_parallel_band_endpoints_are_trimmed_out_of_turnout_zones() -> None:
    module = _load_module()
    mainline = _line(
        [(0, 0), (100, 0)],
        band_id="mainline_2_track",
        interval_id=0,
        station_min_m=0.0,
        station_max_m=100.0,
    )
    parallel = _line(
        [(10, 5), (80, 5)],
        band_id="parallel_plus_5m",
        interval_id=0,
        station_min_m=10.0,
        station_max_m=80.0,
    )
    start_turnout = _line([(5, 5), (30, 0)], branch_id="TA01", station_min_m=5.0, station_max_m=30.0)
    end_turnout = _line([(70, 5), (90, 0)], branch_id="TA02", station_min_m=70.0, station_max_m=90.0)

    features = module.build_topology_features(
        band_features=[mainline, parallel],
        turnout_features=[start_turnout, end_turnout],
        evidence_features=[],
        support_threshold_m=0.85,
        sample_step_m=5.0,
    )

    trimmed = next(feature for feature in features if feature["properties"]["line_id"] == "BAND_parallel_plus_5m_0")
    coords = trimmed["geometry"]["coordinates"]
    assert trimmed["properties"]["station_min_m"] == 31.0
    assert trimmed["properties"]["station_max_m"] == 69.0
    assert coords[0] == [31.0, 5.0]
    assert coords[-1] == [69.0, 5.0]
    assert trimmed["properties"]["endpoint_trim_rule"] == "straight_band_endpoint_inside_turnout_zone"


def test_build_topology_features_adds_evidence_supported_turnout_boundary_bridge() -> None:
    module = _load_module()
    mainline = _line(
        [(0, 0), (100, 0)],
        band_id="mainline_2_track",
        interval_id=0,
        station_min_m=0.0,
        station_max_m=100.0,
    )
    parallel = _line(
        [(55, 5), (90, 5)],
        band_id="parallel_plus_5m",
        interval_id=0,
        station_min_m=55.0,
        station_max_m=90.0,
    )
    turnout = _line(
        [(20, 5), (40, 5)],
        branch_id="TA01",
        station_min_m=20.0,
        station_max_m=40.0,
    )
    evidence = [_line([(40, 5.1), (55, 5.1)], role="support", seq_id="TA01_GP01")]

    features = module.build_topology_features(
        band_features=[mainline, parallel],
        turnout_features=[turnout],
        evidence_features=evidence,
        support_threshold_m=0.85,
        sample_step_m=2.5,
    )

    bridges = [feature for feature in features if feature["properties"]["network_role"] == "turnout_boundary_bridge"]
    assert len(bridges) == 1
    assert bridges[0]["properties"]["risk_flag"] == "evidence_promoted_turnout_boundary_bridge"
    assert bridges[0]["properties"]["bridge_evidence"] == "TA01_GP01"
    assert bridges[0]["properties"]["deeplab_support_ratio"] == 1.0
    assert tuple(bridges[0]["geometry"]["coordinates"][0]) == (55.0, 5.0)
    assert tuple(bridges[0]["geometry"]["coordinates"][-1]) == (40.0, 5.0)


def test_rebuild_crossover_connector_uses_semseg_boundary_evidence() -> None:
    module = _load_module()
    mainline = _line(
        [(0, 0), (80, 0)],
        line_id="BAND_mainline_2_track_0",
        network_role="main_through_track",
        band_id="mainline_2_track",
    )
    old_crossover = _line(
        [(0, 0), (8, -0.2), (20, -2.2), (32, -3.6), (50, -5)],
        line_id="TURNOUT_CX_TEST",
        network_role="turnout_connector",
        connector_id="CX_TEST",
        branch_id="CX_TEST",
        shape_model="curve_straight_curve_raw_dom_fit",
    )
    start_evidence = _line([(0, 0.0), (8, -0.05)], branch_id="CX_TEST", seq_id="CX_TEST_GP_START")
    end_evidence = _line([(42, -4.95), (50, -5.0)], branch_id="CX_TEST", seq_id="CX_TEST_GP_END")
    evidence = [start_evidence, end_evidence]

    rebuilt = module.rebuild_crossover_connectors_with_evidence(
        [mainline, old_crossover],
        evidence_features=evidence,
        evidence_segments=module.build_segments(evidence),
        support_threshold_m=0.85,
        sample_step_m=5.0,
    )

    cx = next(feature for feature in rebuilt if feature["properties"]["line_id"] == "TURNOUT_CX_TEST")
    assert cx["properties"]["crossover_rebuild_status"] == "accepted"
    assert cx["properties"]["crossover_start_evidence"] == "CX_TEST_GP_START"
    assert cx["properties"]["crossover_end_evidence"] == "CX_TEST_GP_END"
    assert cx["properties"]["crossover_middle_len_m"] == 34.0
    assert cx["properties"]["shape_model"] == "semseg_evidence_curve_straight_curve"
    assert cx["properties"]["crossover_endpoint_tangent_slope"] == 0.0
    offsets = [point[1] for point in module.station_offsets_for_coords(module.line_coords(cx), guide=module.LinearGuide((0, 0), (80, 0)))]
    assert max(offsets) <= 0.0
    assert min(offsets) >= -5.0


def test_build_topology_features_promotes_diagnostic_track_and_turnout_tail_bridge() -> None:
    module = _load_module()
    diagnostic = _line(
        [(100, 10), (120, 10)],
        role="diagnostic_candidate",
        band_id="possible_outer_plus_10m",
        interval_id=0,
        station_min_m=100.0,
        station_max_m=120.0,
    )
    turnout = _line([(125, 10), (140, 5)], branch_id="TA10", station_min_m=125.0, station_max_m=140.0)
    evidence = [
        _line([(100, 10.1), (120, 10.1)], role="support"),
        _line([(120, 10.1), (125, 10.1)], role="support"),
    ]

    features = module.build_topology_features(
        band_features=[],
        diagnostic_band_features=[diagnostic],
        turnout_features=[turnout],
        evidence_features=evidence,
        support_threshold_m=0.85,
        sample_step_m=2.5,
        diagnostic_promote_support=0.65,
        diagnostic_min_length_m=10.0,
        turnout_tail_bridge_max_gap_m=10.0,
        turnout_tail_bridge_support=0.5,
    )

    promoted = [feature for feature in features if feature["properties"]["network_role"] == "promoted_straight_track"]
    tail_bridges = [feature for feature in features if feature["properties"]["network_role"] == "turnout_tail_bridge"]

    assert len(promoted) == 1
    assert promoted[0]["properties"]["risk_flag"] == "evidence_promoted_diagnostic_track"
    assert promoted[0]["properties"]["deeplab_support_ratio"] == 1.0
    assert len(tail_bridges) == 1
    assert tail_bridges[0]["properties"]["risk_flag"] == "evidence_promoted_turnout_tail_bridge"
    assert tail_bridges[0]["properties"]["gap_m"] == 5.0
    assert len(tail_bridges[0]["geometry"]["coordinates"]) > 2


def test_curved_endpoint_bridge_uses_endpoint_tangents() -> None:
    module = _load_module()

    coords = module.curved_endpoint_bridge_coords(
        [(0.0, 0.0), (10.0, 0.0)],
        -1,
        [(20.0, 10.0), (20.0, 20.0)],
        0,
    )

    assert len(coords) > 2
    assert coords[0] == [10.0, 0.0]
    assert coords[-1] == [20.0, 10.0]
    assert coords[1][0] > coords[0][0]
    assert coords[-1][1] > coords[-2][1]


def test_rebuild_ta08_curved_branch_replaces_segmented_parts() -> None:
    module = _load_module()
    features = [
        _line([(0, 0), (160, 0)], line_id="BAND_mainline_2_track_0", network_role="main_through_track"),
        _line(
            [(0, 10), (50, 10)],
            line_id="PROMOTED_possible_outer_plus_10m_0",
            network_role="promoted_straight_track",
            band_id="possible_outer_plus_10m",
        ),
        _line(
            [(50, 10), (80, 7)],
            line_id="BRIDGE_PROMOTED_possible_outer_plus_10m_0_TURNOUT_TA08",
            network_role="turnout_tail_bridge",
            branch_id="TA08",
        ),
        _line([(80, 7), (120, 2), (150, 0)], line_id="TURNOUT_TA08", network_role="turnout_connector", branch_id="TA08"),
    ]
    gp02 = _line([(80, 7), (92, 5.8), (104, 4.0), (120, 2)], seq_id="TA08_GP02", branch_id="TA08")
    gp01 = _line([(70, 0), (160, 0)], seq_id="TA08_GP01", branch_id="TA08")

    rebuilt = module.rebuild_ta08_curved_branch(
        features,
        evidence_features=[gp02, gp01],
        evidence_segments=module.build_segments([gp02, gp01]),
        support_threshold_m=1.0,
        sample_step_m=5.0,
    )

    line_ids = [feature["properties"].get("line_id") for feature in rebuilt]
    assert "PROMOTED_possible_outer_plus_10m_0" not in line_ids
    assert "BRIDGE_PROMOTED_possible_outer_plus_10m_0_TURNOUT_TA08" not in line_ids
    assert line_ids.count("TURNOUT_TA08") == 1
    ta08 = next(feature for feature in rebuilt if feature["properties"].get("line_id") == "TURNOUT_TA08")
    assert ta08["properties"]["curve_model"] == "curve_straight_curve_parallel_from_switch"
    assert ta08["properties"]["station_order_model"] == "parallel_curve_straight_curve_to_switch"
    assert ta08["properties"]["parallel_s0_m"] == 0.0
    assert ta08["properties"]["parallel_s1_m"] == 50.0
    assert ta08["properties"]["outer_curve_s0_m"] == 50.0
    assert ta08["properties"]["outer_curve_s1_m"] == 80.0
    assert ta08["properties"]["straight_middle_s0_m"] == 80.0
    assert ta08["properties"]["straight_middle_s1_m"] == 120.0
    assert ta08["properties"]["switch_curve_s0_m"] == 120.0
    assert ta08["properties"]["switch_curve_s1_m"] == 150.0
    assert ta08["properties"]["station_min_m"] == 0.0
    assert ta08["properties"]["station_max_m"] == 150.0
    assert len(ta08["geometry"]["coordinates"]) > 60


def test_build_topology_features_does_not_apply_named_turnout_rebuild_by_default() -> None:
    module = _load_module()
    features = [
        _line([(0, 0), (160, 0)], line_id="BAND_mainline_2_track_0", network_role="main_through_track"),
        _line(
            [(0, 10), (50, 10)],
            line_id="PROMOTED_possible_outer_plus_10m_0",
            network_role="promoted_straight_track",
            band_id="possible_outer_plus_10m",
        ),
        _line([(80, 7), (120, 2), (150, 0)], line_id="TURNOUT_TA08", network_role="turnout_connector", branch_id="TA08"),
    ]
    gp02 = _line([(80, 7), (92, 5.8), (104, 4.0), (120, 2)], seq_id="TA08_GP02", branch_id="TA08")
    gp01 = _line([(70, 0), (160, 0)], seq_id="TA08_GP01", branch_id="TA08")

    rebuilt = module.build_topology_features(
        band_features=features[:1],
        diagnostic_band_features=features[1:2],
        turnout_features=features[2:],
        evidence_features=[gp02, gp01],
        support_threshold_m=1.0,
        sample_step_m=5.0,
        diagnostic_promote_support=0.0,
        diagnostic_min_length_m=1.0,
        allow_specialized_turnout_rebuilds=False,
    )

    ta08 = next(feature for feature in rebuilt if feature["properties"].get("line_id") == "TURNOUT_TA08")
    assert ta08["geometry"]["coordinates"] == [[80, 7], [120, 2], [150, 0]]
    assert ta08["properties"].get("curve_model") != "curve_straight_curve_parallel_from_switch"


def test_rebuild_ta08_outer_curve_uses_deeplab_support_chain() -> None:
    module = _load_module()
    features = [
        _line([(0, 0), (160, 0)], line_id="BAND_mainline_2_track_0", network_role="main_through_track"),
        _line(
            [(0, 10), (50, 10)],
            line_id="PROMOTED_possible_outer_plus_10m_0",
            network_role="promoted_straight_track",
            band_id="possible_outer_plus_10m",
        ),
        _line(
            [(50, 10), (80, 7)],
            line_id="BRIDGE_PROMOTED_possible_outer_plus_10m_0_TURNOUT_TA08",
            network_role="turnout_tail_bridge",
            branch_id="TA08",
        ),
        _line([(80, 7), (120, 2), (150, 0)], line_id="TURNOUT_TA08", network_role="turnout_connector", branch_id="TA08"),
    ]
    support = _line(
        [(40, 10), (50, 10), (60, 8.6), (70, 7.6), (80, 7.0)],
        line_id="DLV1_SUPPORT_TEST",
        role="support",
        network_source="deeplab_v1_refined_support_chain",
    )
    gp02 = _line([(80, 7), (92, 5.8), (104, 4.0), (120, 2)], seq_id="TA08_GP02", branch_id="TA08")
    gp01 = _line([(70, 0), (160, 0)], seq_id="TA08_GP01", branch_id="TA08")

    rebuilt = module.rebuild_ta08_curved_branch(
        features,
        evidence_features=[support, gp02, gp01],
        evidence_segments=module.build_segments([support, gp02, gp01]),
        support_threshold_m=1.0,
        sample_step_m=5.0,
    )

    ta08 = next(feature for feature in rebuilt if feature["properties"].get("line_id") == "TURNOUT_TA08")
    guide = module.LinearGuide((0.0, 0.0), (160.0, 0.0))
    st = module.station_offsets_for_coords(module.line_coords(ta08), guide=guide)
    assert ta08["properties"]["outer_curve_mode"] == "deeplab_support_chain_constrained"
    assert ta08["properties"]["outer_curve_source"] == "DLV1_SUPPORT_TEST"
    assert ta08["properties"]["parallel_s1_m"] < 55.0
    assert abs(module.interpolated_offset(st, 60.0) - 8.6) < 0.25


def test_smooth_station_offsets_updates_each_unlocked_point() -> None:
    module = _load_module()

    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 9.0),
        (3.0, 0.0),
        (4.0, 9.0),
        (5.0, 0.0),
        (6.0, 0.0),
    ]

    smoothed = module.smooth_station_offsets(points, window_size=3, passes=1)

    assert smoothed[0] == points[0]
    assert smoothed[1] == points[1]
    assert smoothed[5] == points[5]
    assert smoothed[6] == points[6]
    assert smoothed[2][1] != points[2][1]
    assert smoothed[3][1] != points[3][1]
    assert smoothed[4][1] != points[4][1]


def test_snap_connector_endpoints_moves_nearby_turnout_endpoint_to_track_line() -> None:
    module = _load_module()
    track = _line(
        [(0, 0), (20, 0)],
        line_id="BAND_mainline_2_track_0",
        network_role="main_through_track",
    )
    turnout = _line(
        [(10, 0.4), (14, 5)],
        line_id="TURNOUT_TA99",
        network_role="turnout_connector",
    )

    snapped = module.snap_connector_endpoints([track, turnout], max_distance_m=1.0)

    snapped_turnout = snapped[1]
    assert snapped_turnout["geometry"]["coordinates"][0] == [10.0, 0.0]
    assert snapped_turnout["geometry"]["coordinates"][1] == [14, 5]
    assert snapped_turnout["properties"]["endpoint_snap_count"] == 1
    assert snapped_turnout["properties"]["endpoint_snap_max_m"] == 0.4


def test_write_centerline_shapefile_preserves_long_line_id(tmp_path) -> None:
    module = _load_module()
    import shapefile

    line_id = "BRIDGE_PROMOTED_possible_outer_plus_10m_0_TURNOUT_TA08"
    output_path = tmp_path / "network.shp"
    feature = _line(
        [(0, 0), (1, 1)],
        line_id=line_id,
        network_role="turnout_tail_bridge",
        risk_flag="evidence_promoted_turnout_tail_bridge",
    )

    module.write_centerline_shapefile([feature], output_path, epsg=32651)

    reader = shapefile.Reader(str(output_path), encoding="utf-8")
    fields = [field[0] for field in reader.fields[1:]]
    record = reader.records()[0]
    assert record[fields.index("line_id")] == line_id
