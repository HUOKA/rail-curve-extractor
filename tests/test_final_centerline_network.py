from __future__ import annotations

import importlib.util
from pathlib import Path

from shapely.geometry import LineString


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_final_centerline_network.py"
    spec = importlib.util.spec_from_file_location("build_final_centerline_network", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _feature(coords, *, chain_id: str, source: str = "automatic_refined"):
    return {
        "type": "Feature",
        "properties": {"chain_id": chain_id, "network_source": source},
        "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in coords]},
    }


def _segment(module, segment_id: str, coords):
    return module.TopologySegment(
        segment_id=segment_id,
        parent_feature_index=1,
        parent_feature_id=segment_id,
        properties={"network_source": "test"},
        line=LineString(coords),
    )


def test_qgis_accepted_override_removes_nearby_duplicate_but_keeps_parallel_track() -> None:
    module = _load_module()
    features = [
        _feature([(0, 0), (100, 0)], chain_id="accepted", source="qgis_accepted_centerline"),
        _feature([(0, 0.45), (100, 0.45)], chain_id="biased_duplicate"),
        _feature([(0, 5.0), (100, 5.0)], chain_id="real_parallel_track"),
    ]

    kept, removed = module.apply_qgis_accepted_overrides(
        module.load_line_features_from_geojson_payload({"type": "FeatureCollection", "features": features}),
        duplicate_distance_m=1.0,
        duplicate_coverage=0.95,
    )

    kept_ids = {item.feature_id for item in kept}
    removed_ids = {item.feature_id for item in removed}
    assert "accepted" in kept_ids
    assert "real_parallel_track" in kept_ids
    assert "biased_duplicate" not in kept_ids
    assert removed_ids == {"biased_duplicate"}


def test_materialize_endpoint_to_line_interior_creates_shared_junction_node() -> None:
    module = _load_module()
    features = [
        _feature([(0, 0), (10, 0)], chain_id="main"),
        _feature([(5, -4), (5, 0.2)], chain_id="branch"),
    ]
    lines = module.load_line_features_from_geojson_payload({"type": "FeatureCollection", "features": features})

    segments, snap_events = module.materialize_topology(lines, snap_tolerance_m=0.5)
    stats = module.build_graph_summary(segments, node_tolerance_m=0.05)

    assert any(event["target_feature_id"] == "main" and event["snap_kind"] == "line_interior" for event in snap_events)
    main_segments = [segment for segment in segments if segment.parent_feature_id == "main"]
    assert len(main_segments) == 2
    junction_nodes = [node for node in stats["nodes"] if node["degree"] == 3]
    assert len(junction_nodes) == 1
    junction = junction_nodes[0]
    assert abs(junction["x"] - 5.0) < 0.05
    assert abs(junction["y"] - 0.0) < 0.05


def test_materialized_segments_preserve_total_line_length_except_removed_duplicates() -> None:
    module = _load_module()
    features = [
        _feature([(0, 0), (10, 0)], chain_id="main"),
        _feature([(5, -3), (5, 0)], chain_id="branch"),
    ]
    lines = module.load_line_features_from_geojson_payload({"type": "FeatureCollection", "features": features})

    segments, _ = module.materialize_topology(lines, snap_tolerance_m=0.25)

    assert abs(sum(segment.line.length for segment in segments) - 13.0) < 1e-6
    assert all(isinstance(segment.line, LineString) for segment in segments)


def test_graph_summary_clusters_nearby_endpoints_without_grid_boundary_false_split() -> None:
    module = _load_module()
    # These two endpoints are 0.02m apart but fall on opposite sides of a simple
    # round(coord/tolerance) grid boundary at tolerance=0.25.
    segments = [
        _segment(module, "a", [(0.00, 0.00), (10.12, 0.00)]),
        _segment(module, "b", [(10.14, 0.00), (20.00, 0.00)]),
    ]

    stats = module.build_graph_summary(segments, node_tolerance_m=0.25)

    assert stats["component_count"] == 1
    assert any(node["degree"] == 2 for node in stats["nodes"])


def test_post_segment_materialization_reconnects_dead_endpoint_exposed_by_duplicate_removal() -> None:
    module = _load_module()
    # This mirrors the real QGIS-fid4/hint27 failure mode: after segment-level
    # duplicate removal, a segment endpoint can be only ~1m from another segment
    # endpoint even though the original feature endpoint was elsewhere.
    segments = [
        _segment(module, "accepted_patch", [(0.0, 0.0), (0.0, 20.0)]),
        _segment(module, "remaining_route", [(0.8, 20.6), (0.8, 40.0)]),
    ]

    rematerialized, snap_events = module.rematerialize_segments_after_filtering(segments, snap_tolerance_m=1.25)
    stats = module.build_graph_summary(rematerialized, node_tolerance_m=0.25)

    assert any(event["source_feature_id"] == "remaining_route" for event in snap_events)
    assert stats["component_count"] == 1
    assert stats["dead_end_node_count"] == 2


def test_materialize_preserves_tiny_qgis_accepted_connector_segments() -> None:
    module = _load_module()
    features = [
        _feature([(0.0, 0.0), (0.05, 0.0)], chain_id="qgis_tiny", source="qgis_accepted_centerline"),
        _feature([(1.0, 0.0), (2.0, 0.0)], chain_id="regular"),
    ]
    lines = module.load_line_features_from_geojson_payload({"type": "FeatureCollection", "features": features})

    segments, _ = module.materialize_topology(lines, snap_tolerance_m=0.01)

    assert any(segment.parent_feature_id == "qgis_tiny" and segment.line.length < 0.1 for segment in segments)
