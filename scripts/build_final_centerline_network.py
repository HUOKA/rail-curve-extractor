#!/usr/bin/env python3
"""Build a topology-materialized final rail centerline candidate network.

The script is intentionally conservative:

1. Start from the QGIS-supervised centerline layer.
2. Let note-accepted QGIS centerline geometry override nearby biased/duplicate
   automatic/manual/hint features, while preserving real parallel tracks farther
   away.
3. Snap endpoints to nearby endpoints or line interiors and split the target line
   at those points so the exported GeoJSON has real shared graph nodes instead
   of metadata-only endpoint snaps.
4. Export graph segments plus a machine-readable topology summary for QA.

This does not claim final acceptance by itself; it produces a better candidate
for DOM/point-cloud visual QA and follow-up route stitching.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, NamedTuple

from shapely.geometry import LineString, Point
from shapely.ops import substring


class LineFeature(NamedTuple):
    feature_index: int
    feature_id: str
    properties: dict[str, Any]
    line: LineString


class TopologySegment(NamedTuple):
    segment_id: str
    parent_feature_index: int
    parent_feature_id: str
    properties: dict[str, Any]
    line: LineString


DEFAULT_INPUT = Path("output/rail_centerline_refined_v4_adaptive_gauge_qgis_supervised/hybrid_centerline_network_qgis_supervised.geojson")
DEFAULT_OUT_DIR = Path("output/rail_centerline_final_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build topology-materialized final rail centerline candidate network.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--snap-tolerance-m", type=float, default=0.75)
    parser.add_argument("--node-tolerance-m", type=float, default=0.25)
    parser.add_argument("--qgis-duplicate-distance-m", type=float, default=2.0)
    parser.add_argument("--qgis-duplicate-coverage", type=float, default=0.95)
    parser.add_argument("--segment-duplicate-distance-m", type=float, default=2.0)
    parser.add_argument("--segment-duplicate-coverage", type=float, default=0.9)
    parser.add_argument(
        "--extra-bridges",
        type=Path,
        default=None,
        help="Optional GeoJSON LineString layer of QA-supervised bridge/extension segments to append before final topology materialization.",
    )
    parser.add_argument("--write-shapefile", action="store_true")
    return parser.parse_args()


def load_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_line_features_from_geojson_payload(payload: dict[str, Any]) -> list[LineFeature]:
    features: list[LineFeature] = []
    for index, feature in enumerate(payload.get("features", []) or [], start=1):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates", [])]
        if len(coords) < 2:
            continue
        props = dict(feature.get("properties") or {})
        feature_id = stable_feature_id(index, props)
        features.append(LineFeature(index, feature_id, props, LineString(coords)))
    return features


def stable_feature_id(index: int, props: dict[str, Any]) -> str:
    for key in ("chain_id", "id_text", "candidate_id", "bridge_id", "fid", "qgis_fid"):
        value = props.get(key)
        if value not in (None, ""):
            return str(value)
    return f"feature_{index}"


def load_extra_bridge_segments(path: Path | None, *, start_index: int) -> list[TopologySegment]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"extra bridge GeoJSON not found: {path}")
    bridge_features = load_line_features_from_geojson_payload(load_geojson(path))
    segments: list[TopologySegment] = []
    for offset, feature in enumerate(bridge_features, start=1):
        props = dict(feature.properties)
        props.setdefault("network_source", "qa_supervised_gap_bridge")
        props.setdefault("normalization_role", "qa_supervised_bridge_or_extension")
        props.setdefault("bridge_source_path", str(path))
        segment_id = f"qa_bridge_{feature.feature_id}_seg_001"
        segments.append(
            TopologySegment(
                segment_id=segment_id,
                parent_feature_index=start_index + offset,
                parent_feature_id=feature.feature_id,
                properties=props,
                line=feature.line,
            )
        )
    return segments


def directed_coverage(line: LineString, target: LineString, *, threshold_m: float, sample_spacing_m: float = 1.0) -> dict[str, float]:
    sample_count = max(2, int(math.ceil(max(line.length, 1e-6) / sample_spacing_m)) + 1)
    distances: list[float] = []
    hits = 0
    for index in range(sample_count):
        point = line.interpolate(line.length * index / max(sample_count - 1, 1))
        distance = float(point.distance(target))
        distances.append(distance)
        if distance <= threshold_m:
            hits += 1
    return {
        "coverage": hits / sample_count,
        "mean_distance_m": sum(distances) / len(distances),
        "max_distance_m": max(distances),
        "sample_count": sample_count,
    }


def apply_qgis_accepted_overrides(
    features: list[LineFeature],
    *,
    duplicate_distance_m: float = 1.0,
    duplicate_coverage: float = 0.95,
) -> tuple[list[LineFeature], list[LineFeature]]:
    """Remove non-QGIS features that are effectively replaced by accepted QGIS geometry."""
    accepted = [feature for feature in features if feature.properties.get("network_source") == "qgis_accepted_centerline"]
    if not accepted:
        return features, []

    kept: list[LineFeature] = []
    removed: list[LineFeature] = []
    for feature in features:
        if feature.properties.get("network_source") == "qgis_accepted_centerline":
            kept.append(feature)
            continue
        replacement = best_qgis_replacement(feature, accepted, duplicate_distance_m)
        if replacement is not None and replacement["coverage"] >= duplicate_coverage:
            # Avoid deleting long through-routes when a short accepted segment only
            # overlaps a local patch.  Those are split/trimmed later at segment level.
            accepted_length = float(replacement["accepted"].line.length)
            if feature.line.length <= accepted_length * 1.5 + 5.0:
                removed.append(feature)
                continue
        kept.append(feature)
    return kept, removed


def best_qgis_replacement(feature: LineFeature, accepted: list[LineFeature], threshold_m: float) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for accepted_feature in accepted:
        score = directed_coverage(feature.line, accepted_feature.line, threshold_m=threshold_m, sample_spacing_m=2.0)
        candidate = {**score, "accepted": accepted_feature}
        if best is None or (candidate["coverage"], -candidate["mean_distance_m"]) > (best["coverage"], -best["mean_distance_m"]):
            best = candidate
    return best


def collect_intersection_points(geometry) -> list[tuple[float, float]]:
    if geometry.is_empty:
        return []
    geom_type = geometry.geom_type
    if geom_type == "Point":
        return [(float(geometry.x), float(geometry.y))]
    if geom_type == "MultiPoint":
        return [(float(point.x), float(point.y)) for point in geometry.geoms]
    if geom_type == "LineString":
        coords = list(geometry.coords)
        if not coords:
            return []
        return [(float(coords[0][0]), float(coords[0][1])), (float(coords[-1][0]), float(coords[-1][1]))]
    points: list[tuple[float, float]] = []
    for item in getattr(geometry, "geoms", []) or []:
        points.extend(collect_intersection_points(item))
    return points


def materialize_topology(features: list[LineFeature], *, snap_tolerance_m: float = 0.75) -> tuple[list[TopologySegment], list[dict[str, Any]]]:
    adjusted_coords = [list(feature.line.coords) for feature in features]
    split_distances: list[set[float]] = [{0.0, float(feature.line.length)} for feature in features]
    original_lines = [feature.line for feature in features]
    snap_events: list[dict[str, Any]] = []
    clustered_endpoint_keys, endpoint_cluster_events = cluster_nearby_feature_endpoints(features, adjusted_coords, snap_tolerance_m)
    snap_events.extend(endpoint_cluster_events)

    for source_index, feature in enumerate(features):
        for endpoint_name, coord_index in (("start", 0), ("end", -1)):
            if (source_index, coord_index) in clustered_endpoint_keys:
                continue
            point = Point(adjusted_coords[source_index][coord_index])
            best = nearest_snap_target(point, source_index, features, snap_tolerance_m)
            if best is None:
                continue
            target_index, target_point, distance_m, snap_kind = best
            adjusted_coords[source_index][coord_index] = target_point
            target_distance = original_lines[target_index].project(Point(target_point))
            split_distances[target_index].add(float(target_distance))
            split_distances[source_index].add(0.0 if coord_index == 0 else float(original_lines[source_index].length))
            snap_events.append(
                {
                    "source_feature_id": feature.feature_id,
                    "source_feature_index": feature.feature_index,
                    "endpoint": endpoint_name,
                    "target_feature_id": features[target_index].feature_id,
                    "target_feature_index": features[target_index].feature_index,
                    "snap_kind": snap_kind,
                    "distance_m": round(float(distance_m), 4),
                    "point": [round(float(target_point[0]), 6), round(float(target_point[1]), 6)],
                }
            )

    adjusted_lines = [LineString(coords) for coords in adjusted_coords]
    for left_index in range(len(adjusted_lines)):
        for right_index in range(left_index + 1, len(adjusted_lines)):
            intersection = adjusted_lines[left_index].intersection(adjusted_lines[right_index])
            for point in collect_intersection_points(intersection):
                p = Point(point)
                split_distances[left_index].add(float(adjusted_lines[left_index].project(p)))
                split_distances[right_index].add(float(adjusted_lines[right_index].project(p)))

    segments: list[TopologySegment] = []
    for feature_index, line in enumerate(adjusted_lines):
        feature = features[feature_index]
        split_dedupe_tolerance_m = 0.005 if feature.properties.get("network_source") == "qgis_accepted_centerline" else 0.05
        distances = dedupe_distances(split_distances[feature_index], line.length, tolerance_m=split_dedupe_tolerance_m)
        local_segment_index = 0
        for start_m, end_m in zip(distances, distances[1:]):
            min_segment_length_m = 0.01 if feature.properties.get("network_source") == "qgis_accepted_centerline" else 0.10
            if end_m - start_m < min_segment_length_m:
                continue
            segment_line = substring(line, start_m, end_m)
            if segment_line.geom_type != "LineString" or segment_line.length < min_segment_length_m:
                continue
            local_segment_index += 1
            segment_id = f"{feature.feature_id}_seg_{local_segment_index:03d}"
            segments.append(
                TopologySegment(
                    segment_id=segment_id,
                    parent_feature_index=feature.feature_index,
                    parent_feature_id=feature.feature_id,
                    properties=dict(feature.properties),
                    line=segment_line,
                )
            )
    return segments, snap_events


def cluster_nearby_feature_endpoints(
    features: list[LineFeature],
    adjusted_coords: list[list[tuple[float, float]]],
    snap_tolerance_m: float,
) -> tuple[set[tuple[int, int]], list[dict[str, Any]]]:
    endpoint_entries: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        coords = list(feature.line.coords)
        for endpoint_name, coord_index in (("start", 0), ("end", -1)):
            coord = coords[coord_index]
            endpoint_entries.append(
                {
                    "feature_index": feature_index,
                    "feature_id": feature.feature_id,
                    "feature_source_index": feature.feature_index,
                    "endpoint": endpoint_name,
                    "coord_index": coord_index,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "network_source": feature.properties.get("network_source"),
                }
            )

    parent = list(range(len(endpoint_entries)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, entry in enumerate(endpoint_entries):
        key = (math.floor(entry["x"] / snap_tolerance_m), math.floor(entry["y"] / snap_tolerance_m))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_index in buckets.get((key[0] + dx, key[1] + dy), []):
                    other = endpoint_entries[other_index]
                    if other["feature_index"] == entry["feature_index"]:
                        continue
                    if math.hypot(entry["x"] - other["x"], entry["y"] - other["y"]) <= snap_tolerance_m:
                        union(index, other_index)
        buckets.setdefault(key, []).append(index)

    groups_by_root: dict[int, list[dict[str, Any]]] = {}
    for index, entry in enumerate(endpoint_entries):
        groups_by_root.setdefault(find(index), []).append(entry)

    clustered_keys: set[tuple[int, int]] = set()
    events: list[dict[str, Any]] = []
    cluster_index = 0
    for group in groups_by_root.values():
        if len(group) < 2:
            continue
        cluster_index += 1
        qgis_entries = [entry for entry in group if entry.get("network_source") == "qgis_accepted_centerline"]
        if len(qgis_entries) == 1:
            anchor_x = qgis_entries[0]["x"]
            anchor_y = qgis_entries[0]["y"]
            anchor_feature_id = qgis_entries[0]["feature_id"]
        else:
            anchor_x = sum(entry["x"] for entry in group) / len(group)
            anchor_y = sum(entry["y"] for entry in group) / len(group)
            anchor_feature_id = "endpoint_cluster_centroid"
        for entry in group:
            feature_index = int(entry["feature_index"])
            coord_index = int(entry["coord_index"])
            clustered_keys.add((feature_index, coord_index))
            adjusted_coords[feature_index][coord_index] = (float(anchor_x), float(anchor_y))
            events.append(
                {
                    "source_feature_id": entry["feature_id"],
                    "source_feature_index": entry["feature_source_index"],
                    "endpoint": entry["endpoint"],
                    "target_feature_id": anchor_feature_id,
                    "target_feature_index": None,
                    "snap_kind": "endpoint_cluster",
                    "cluster_id": f"endpoint_cluster_{cluster_index:03d}",
                    "distance_m": round(math.hypot(entry["x"] - anchor_x, entry["y"] - anchor_y), 4),
                    "point": [round(float(anchor_x), 6), round(float(anchor_y), 6)],
                }
            )
    return clustered_keys, events


def nearest_snap_target(
    point: Point,
    source_index: int,
    features: list[LineFeature],
    snap_tolerance_m: float,
) -> tuple[int, tuple[float, float], float, str] | None:
    best_endpoint: tuple[int, tuple[float, float], float, str] | None = None
    best_line: tuple[int, tuple[float, float], float, str] | None = None
    for target_index, target in enumerate(features):
        if target_index == source_index:
            continue
        coords = list(target.line.coords)
        endpoint_candidates = [(coords[0], "endpoint"), (coords[-1], "endpoint")]
        for endpoint_coord, kind in endpoint_candidates:
            target_point = Point(endpoint_coord)
            distance = float(point.distance(target_point))
            if distance <= snap_tolerance_m and (best_endpoint is None or distance < best_endpoint[2]):
                best_endpoint = (target_index, (float(target_point.x), float(target_point.y)), distance, kind)

        distance_to_line = float(target.line.distance(point))
        if distance_to_line <= snap_tolerance_m and (best_line is None or distance_to_line < best_line[2]):
            projected = target.line.interpolate(target.line.project(point))
            projected_coord = (float(projected.x), float(projected.y))
            # If projection is effectively an endpoint, label it as endpoint.
            if min(Point(projected_coord).distance(Point(coords[0])), Point(projected_coord).distance(Point(coords[-1]))) <= 0.05:
                kind = "endpoint"
            else:
                kind = "line_interior"
            best_line = (target_index, projected_coord, distance_to_line, kind)

    if best_endpoint is not None:
        # Prefer endpoint-to-endpoint when the endpoint is only slightly farther
        # than a projection to the neighbor segment interior.  This prevents two
        # near endpoints from snapping to different points on each other's lines,
        # while still allowing true branch-to-mainline interior materialization.
        if best_line is None or best_endpoint[2] <= best_line[2] + min(0.50, snap_tolerance_m * 0.35):
            return best_endpoint
    return best_line


def dedupe_distances(distances: set[float], line_length: float, tolerance_m: float = 0.05) -> list[float]:
    ordered = sorted(max(0.0, min(float(line_length), float(distance))) for distance in distances)
    deduped: list[float] = []
    for distance in ordered:
        if not deduped or abs(distance - deduped[-1]) > tolerance_m:
            deduped.append(distance)
    if not deduped or deduped[0] > 0.0:
        deduped.insert(0, 0.0)
    if abs(deduped[-1] - line_length) > tolerance_m:
        deduped.append(float(line_length))
    return deduped


def apply_segment_level_qgis_overrides(
    segments: list[TopologySegment],
    *,
    duplicate_distance_m: float = 1.25,
    duplicate_coverage: float = 0.9,
) -> tuple[list[TopologySegment], list[TopologySegment]]:
    accepted = [segment for segment in segments if segment.properties.get("network_source") == "qgis_accepted_centerline"]
    if not accepted:
        return segments, []
    kept: list[TopologySegment] = []
    removed: list[TopologySegment] = []
    for segment in segments:
        if segment.properties.get("network_source") == "qgis_accepted_centerline":
            kept.append(segment)
            continue
        best_coverage = 0.0
        for accepted_segment in accepted:
            if segment.line.length > accepted_segment.line.length * 1.75 + 1.0:
                continue
            score = directed_coverage(segment.line, accepted_segment.line, threshold_m=duplicate_distance_m, sample_spacing_m=0.75)
            best_coverage = max(best_coverage, float(score["coverage"]))
        if best_coverage >= duplicate_coverage:
            removed.append(segment)
        else:
            kept.append(segment)
    return kept, removed


def rematerialize_segments_after_filtering(
    segments: list[TopologySegment],
    *,
    snap_tolerance_m: float,
) -> tuple[list[TopologySegment], list[dict[str, Any]]]:
    """Run a second topology pass after duplicate segment suppression.

    Segment-level QGIS override can expose new segment endpoints that were not
    original feature endpoints.  A second conservative materialization pass snaps
    those newly exposed endpoints to nearby endpoints/line interiors and splits
    target segments so the final exported network does not retain avoidable
    micro-gaps.
    """
    features: list[LineFeature] = []
    for index, segment in enumerate(segments, start=1):
        props = dict(segment.properties)
        props.setdefault("pre_post_segment_id", segment.segment_id)
        props.setdefault("pre_post_parent_feature_id", segment.parent_feature_id)
        features.append(
            LineFeature(
                feature_index=index,
                feature_id=segment.segment_id,
                properties=props,
                line=segment.line,
            )
        )
    post_segments, snap_events = materialize_topology(features, snap_tolerance_m=snap_tolerance_m)
    return post_segments, snap_events


def node_key(coord: tuple[float, float], tolerance_m: float) -> tuple[int, int]:
    return (round(float(coord[0]) / tolerance_m), round(float(coord[1]) / tolerance_m))


def build_graph_summary(segments: list[TopologySegment], *, node_tolerance_m: float = 0.25) -> dict[str, Any]:
    endpoint_refs: list[dict[str, Any]] = []
    for segment in segments:
        coords = list(segment.line.coords)
        for endpoint_name, coord in (("start", coords[0]), ("end", coords[-1])):
            endpoint_refs.append(
                {
                    "segment_id": segment.segment_id,
                    "parent_feature_id": segment.parent_feature_id,
                    "endpoint": endpoint_name,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                }
            )

    endpoint_parent = list(range(len(endpoint_refs)))

    def endpoint_find(index: int) -> int:
        while endpoint_parent[index] != index:
            endpoint_parent[index] = endpoint_parent[endpoint_parent[index]]
            index = endpoint_parent[index]
        return index

    def endpoint_union(left: int, right: int) -> None:
        left_root = endpoint_find(left)
        right_root = endpoint_find(right)
        if left_root != right_root:
            endpoint_parent[right_root] = left_root

    # Cluster by true metric distance, not by a single rounded grid key.  A
    # round(coord/tol) key creates false topology breaks at cell boundaries; use
    # buckets only as a neighbor-search accelerator and check adjacent buckets.
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, ref in enumerate(endpoint_refs):
        key = (math.floor(ref["x"] / node_tolerance_m), math.floor(ref["y"] / node_tolerance_m))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_index in buckets.get((key[0] + dx, key[1] + dy), []):
                    other = endpoint_refs[other_index]
                    if math.hypot(ref["x"] - other["x"], ref["y"] - other["y"]) <= node_tolerance_m:
                        endpoint_union(index, other_index)
        buckets.setdefault(key, []).append(index)

    node_groups_by_root: dict[int, list[dict[str, Any]]] = {}
    for index, ref in enumerate(endpoint_refs):
        node_groups_by_root.setdefault(endpoint_find(index), []).append(ref)
    node_groups = list(node_groups_by_root.values())

    segment_index = {segment.segment_id: index for index, segment in enumerate(segments)}
    parent = list(range(len(segments)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for refs in node_groups:
        if len(refs) < 2:
            continue
        first = segment_index[refs[0]["segment_id"]]
        for ref in refs[1:]:
            union(first, segment_index[ref["segment_id"]])

    components: dict[int, list[TopologySegment]] = {}
    for segment in segments:
        components.setdefault(find(segment_index[segment.segment_id]), []).append(segment)

    nodes: list[dict[str, Any]] = []
    for node_index, refs in enumerate(node_groups, start=1):
        x = sum(ref["x"] for ref in refs) / len(refs)
        y = sum(ref["y"] for ref in refs) / len(refs)
        nodes.append(
            {
                "node_id": f"node_{node_index:04d}",
                "x": round(x, 6),
                "y": round(y, 6),
                "degree": len(refs),
                "refs": refs,
            }
        )

    component_summaries = []
    for component_index, component_segments in enumerate(sorted(components.values(), key=len, reverse=True), start=1):
        coords = [coord for segment in component_segments for coord in segment.line.coords]
        line = LineString(coords) if len(coords) >= 2 else LineString()
        component_summaries.append(
            {
                "component_id": f"component_{component_index:03d}",
                "segment_count": len(component_segments),
                "length_m": round(sum(segment.line.length for segment in component_segments), 3),
                "bounds": [round(float(value), 3) for value in line.bounds] if not line.is_empty else [],
                "parent_feature_ids": sorted({str(segment.parent_feature_id) for segment in component_segments}),
            }
        )

    degree_counts: dict[str, int] = {}
    for node in nodes:
        key = str(node["degree"])
        degree_counts[key] = degree_counts.get(key, 0) + 1

    return {
        "segment_count": len(segments),
        "node_count": len(nodes),
        "dead_end_node_count": sum(1 for node in nodes if node["degree"] == 1),
        "degree_counts": degree_counts,
        "component_count": len(component_summaries),
        "components": component_summaries,
        "nodes": sorted(nodes, key=lambda item: (-int(item["degree"]), item["y"], item["x"])),
    }


def segment_to_feature(segment: TopologySegment, index: int) -> dict[str, Any]:
    props = dict(segment.properties)
    props.update(
        {
            "final_id": segment.segment_id,
            "final_segment_index": index,
            "parent_feature_index": segment.parent_feature_index,
            "parent_feature_id": segment.parent_feature_id,
            "final_network_stage": "qgis_override_topology_materialized_v1",
            "length_m": round(float(segment.line.length), 3),
        }
    )
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[round(float(x), 6), round(float(y), 6)] for x, y, *_ in segment.line.coords]},
    }


def write_geojson(path: Path, segments: list[TopologySegment]) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": "EPSG:32651"}},
        "features": [segment_to_feature(segment, index) for index, segment in enumerate(segments, start=1)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_shapefile(path: Path, segments: list[TopologySegment]) -> None:
    try:
        import shapefile  # type: ignore
    except ImportError:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(path.with_suffix(".shp")), shapeType=shapefile.POLYLINE)
    writer.autoBalance = 1
    fields = [
        ("final_id", "C", 80),
        ("parent_id", "C", 80),
        ("src", "C", 40),
        ("role", "C", 50),
        ("qgis_fid", "C", 20),
        ("len_m", "N", 12, 3),
    ]
    for name, kind, size, *decimal in fields:
        writer.field(name, kind, size=size, decimal=decimal[0] if decimal else 0)
    for segment in segments:
        writer.line([[(float(x), float(y)) for x, y, *_ in segment.line.coords]])
        props = segment.properties
        writer.record(
            segment.segment_id[:80],
            str(segment.parent_feature_id)[:80],
            str(props.get("network_source") or "")[:40],
            str(props.get("normalization_role") or "")[:50],
            str(props.get("qgis_fid") or props.get("turnout_curve_qgis_fid") or "")[:20],
            round(float(segment.line.length), 3),
        )
    writer.close()
    source_prj = DEFAULT_INPUT.with_suffix(".prj")
    if source_prj.exists():
        shutil.copyfile(source_prj, path.with_suffix(".prj"))
    else:
        path.with_suffix(".prj").write_text(
            'PROJCS["WGS_1984_UTM_Zone_51N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",123.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]',
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    payload = load_geojson(args.input)
    features = load_line_features_from_geojson_payload(payload)
    kept_features, feature_removed = apply_qgis_accepted_overrides(
        features,
        duplicate_distance_m=args.qgis_duplicate_distance_m,
        duplicate_coverage=args.qgis_duplicate_coverage,
    )
    segments, snap_events = materialize_topology(kept_features, snap_tolerance_m=args.snap_tolerance_m)
    kept_segments, segment_removed = apply_segment_level_qgis_overrides(
        segments,
        duplicate_distance_m=args.segment_duplicate_distance_m,
        duplicate_coverage=args.segment_duplicate_coverage,
    )
    extra_bridge_segments = load_extra_bridge_segments(args.extra_bridges, start_index=len(kept_segments))
    segments_before_post = kept_segments + extra_bridge_segments
    final_segments, post_snap_events = rematerialize_segments_after_filtering(segments_before_post, snap_tolerance_m=args.snap_tolerance_m)
    graph_summary = build_graph_summary(final_segments, node_tolerance_m=args.node_tolerance_m)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = args.out_dir / "final_centerline_network.geojson"
    write_geojson(geojson_path, final_segments)
    if args.write_shapefile:
        write_shapefile(args.out_dir / "final_centerline_network.shp", final_segments)

    summary = {
        "input": str(args.input),
        "input_feature_count": len(features),
        "feature_removed_by_qgis_accepted_count": len(feature_removed),
        "features_removed_by_qgis_accepted": [feature.feature_id for feature in feature_removed],
        "materialized_segment_count_before_segment_override": len(segments),
        "segment_removed_by_qgis_accepted_count": len(segment_removed),
        "segments_removed_by_qgis_accepted": [segment.segment_id for segment in segment_removed],
        "segment_count_before_post_materialize": len(segments_before_post),
        "extra_bridge_segment_count": len(extra_bridge_segments),
        "extra_bridge_segment_ids": [segment.segment_id for segment in extra_bridge_segments],
        "extra_bridges_path": str(args.extra_bridges) if args.extra_bridges else None,
        "post_materialized_segment_count": len(final_segments),
        "snap_event_count": len(snap_events),
        "snap_events": snap_events,
        "post_snap_event_count": len(post_snap_events),
        "post_snap_events": post_snap_events,
        "output_geojson": str(geojson_path),
        "graph_summary": graph_summary,
        "parameters": {
            "snap_tolerance_m": args.snap_tolerance_m,
            "node_tolerance_m": args.node_tolerance_m,
            "qgis_duplicate_distance_m": args.qgis_duplicate_distance_m,
            "qgis_duplicate_coverage": args.qgis_duplicate_coverage,
            "segment_duplicate_distance_m": args.segment_duplicate_distance_m,
            "segment_duplicate_coverage": args.segment_duplicate_coverage,
            "extra_bridges": str(args.extra_bridges) if args.extra_bridges else None,
        },
    }
    (args.out_dir / "final_centerline_network_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "output_geojson": str(geojson_path), "graph_summary": {k: graph_summary[k] for k in ["segment_count", "node_count", "dead_end_node_count", "component_count", "degree_counts"]}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
