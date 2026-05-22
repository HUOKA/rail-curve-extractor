#!/usr/bin/env python3
"""Package a full strict-auto 2D centerline review network with refined turnouts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import build_deeplab_topology_centerline_network as topo


DEFAULT_BASE_NETWORK = Path("output/dom_centerline_strict_auto_v1/10_centerline_2d/deeplab_topology_centerline_network.geojson")
DEFAULT_BASE_EVIDENCE = Path("output/dom_centerline_strict_auto_v1/10_centerline_2d/deeplab_topology_evidence.geojson")
DEFAULT_REFINED_TURNOUTS = Path(
    "output/dom_centerline_strict_auto_v1/experiments/outer_rail_generalization_audit_kinkfix/all_outer_rail_centerlines.geojson"
)
DEFAULT_TURNOUT_AUDIT = Path("output/dom_centerline_strict_auto_v1/experiments/outer_rail_generalization_audit_kinkfix")
DEFAULT_OUT_DIR = Path("output/dom_centerline_strict_auto_v1/global_centerline_review_tangent_occlusion")
DEFAULT_EPSG = 32651


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a full-DOM 2D review package using refined strict-auto turnout connectors.")
    parser.add_argument("--base-network", type=Path, default=DEFAULT_BASE_NETWORK)
    parser.add_argument("--base-evidence", type=Path, default=DEFAULT_BASE_EVIDENCE)
    parser.add_argument("--refined-turnouts", type=Path, default=DEFAULT_REFINED_TURNOUTS)
    parser.add_argument("--turnout-audit-dir", type=Path, default=DEFAULT_TURNOUT_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--occlusion-bridge-min-gap-m", type=float, default=60.0)
    parser.add_argument("--occlusion-bridge-max-gap-m", type=float, default=85.0)
    parser.add_argument("--tangent-snap-max-m", type=float, default=1.75)
    parser.add_argument("--tangent-smooth-taper-m", type=float, default=12.0)
    parser.add_argument("--tangent-window-m", type=float, default=6.0)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_features = load_features(args.base_network.expanduser().resolve())
    evidence_features = load_features(args.base_evidence.expanduser().resolve()) if args.base_evidence.exists() else []
    refined_turnouts = load_features(args.refined_turnouts.expanduser().resolve())
    audit = load_turnout_audit(args.turnout_audit_dir.expanduser().resolve())

    packaged, replacement_rows, bridge_rows, occlusion_rows = build_packaged_features(
        base_features,
        refined_turnouts=refined_turnouts,
        audit=audit,
        occlusion_bridge_min_gap_m=args.occlusion_bridge_min_gap_m,
        occlusion_bridge_max_gap_m=args.occlusion_bridge_max_gap_m,
        tangent_snap_max_m=args.tangent_snap_max_m,
        tangent_smooth_taper_m=args.tangent_smooth_taper_m,
        tangent_window_m=args.tangent_window_m,
    )

    network_geojson = out_dir / "global_centerline_2d.geojson"
    network_shp = out_dir / "global_centerline_2d.shp"
    evidence_geojson = out_dir / "global_centerline_evidence.geojson"
    evidence_shp = out_dir / "global_centerline_evidence.shp"
    summary_json = out_dir / "summary.json"
    review_md = out_dir / "REVIEW.md"
    replacements_csv = out_dir / "turnout_replacements.csv"
    bridges_csv = out_dir / "boundary_bridge_adjustments.csv"
    occlusion_csv = out_dir / "occlusion_bridges.csv"

    topo.write_geojson(network_geojson, packaged, epsg=args.epsg)
    topo.write_centerline_shapefile(packaged, network_shp, epsg=args.epsg)
    topo.write_centerline_qml(network_geojson.with_suffix(".qml"))
    if evidence_features:
        topo.write_geojson(evidence_geojson, evidence_features, epsg=args.epsg)
        topo.write_evidence_shapefile(evidence_features, evidence_shp, epsg=args.epsg)
        topo.write_evidence_qml(evidence_geojson.with_suffix(".qml"))

    write_dict_csv(replacements_csv, replacement_rows)
    write_dict_csv(bridges_csv, bridge_rows)
    write_dict_csv(occlusion_csv, occlusion_rows)
    summary = build_summary(
        args=args,
        out_dir=out_dir,
        features=packaged,
        replacement_rows=replacement_rows,
        bridge_rows=bridge_rows,
        occlusion_rows=occlusion_rows,
        outputs={
            "network_geojson": network_geojson,
            "network_shp": network_shp,
            "network_qml": network_geojson.with_suffix(".qml"),
            "evidence_geojson": evidence_geojson if evidence_features else None,
            "evidence_shp": evidence_shp if evidence_features else None,
            "turnout_replacements_csv": replacements_csv,
            "boundary_bridge_adjustments_csv": bridges_csv,
            "occlusion_bridges_csv": occlusion_csv,
            "review_md": review_md,
        },
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(review_md, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        feature
        for feature in payload.get("features", []) or []
        if feature.get("geometry", {}).get("type") == "LineString" and len(feature.get("geometry", {}).get("coordinates", [])) >= 2
    ]


def load_turnout_audit(audit_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = audit_dir / "outer_rail_generalization_summary.csv"
    support_path = audit_dir / "outer_rail_support_kind_summary.csv"
    geometry_path = audit_dir / "outer_rail_geometry_audit.csv"
    data: dict[str, dict[str, Any]] = {}
    for path in (summary_path, support_path, geometry_path):
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                branch_id = str(row.get("branch_id", ""))
                if not branch_id:
                    continue
                data.setdefault(branch_id, {}).update(row)
    return data


def build_packaged_features(
    base_features: list[dict[str, Any]],
    *,
    refined_turnouts: list[dict[str, Any]],
    audit: dict[str, dict[str, Any]],
    occlusion_bridge_min_gap_m: float,
    occlusion_bridge_max_gap_m: float,
    tangent_snap_max_m: float,
    tangent_smooth_taper_m: float,
    tangent_window_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    band_index = build_straight_band_index(base_features)
    refined_by_branch = {
        str((feature.get("properties") or {}).get("branch_id", "")): feature
        for feature in refined_turnouts
        if str((feature.get("properties") or {}).get("branch_id", ""))
    }
    old_turnouts = {
        str((feature.get("properties") or {}).get("branch_id", "")): feature
        for feature in base_features
        if str((feature.get("properties") or {}).get("network_role", "")) == "turnout_connector"
    }
    packaged: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    replacements_by_branch: dict[str, dict[str, Any]] = {}
    for feature in base_features:
        props = feature.get("properties") or {}
        branch_id = str(props.get("branch_id", ""))
        if str(props.get("network_role", "")) == "turnout_connector" and branch_id in refined_by_branch:
            replacement, row = build_replacement_feature(
                feature,
                refined_by_branch[branch_id],
                audit.get(branch_id, {}),
                band_index=band_index,
                tangent_snap_max_m=tangent_snap_max_m,
                tangent_smooth_taper_m=tangent_smooth_taper_m,
                tangent_window_m=tangent_window_m,
            )
            packaged.append(replacement)
            replacement_rows.append(row)
            replacements_by_branch[branch_id] = replacement
        else:
            packaged.append(clone_feature(feature))

    bridge_rows: list[dict[str, Any]] = []
    bridge_adjusted_packaged: list[dict[str, Any]] = []
    for feature in packaged:
        props = feature.get("properties") or {}
        branch_id = str(props.get("branch_id", ""))
        if str(props.get("network_role", "")) != "turnout_boundary_bridge" or branch_id not in replacements_by_branch:
            bridge_adjusted_packaged.append(feature)
            continue
        old_turnout = old_turnouts.get(branch_id)
        if old_turnout is None:
            bridge_adjusted_packaged.append(feature)
            continue
        row = adjust_boundary_bridge(feature, old_turnout=old_turnout, refined_turnout=replacements_by_branch[branch_id])
        if row:
            bridge_rows.append(row)
            if int(row.get("dropped_bridge", 0)) == 1:
                continue
        bridge_adjusted_packaged.append(feature)
    packaged = bridge_adjusted_packaged

    occlusion_bridges, occlusion_rows = build_same_band_occlusion_bridges(
        packaged,
        min_gap_m=occlusion_bridge_min_gap_m,
        max_gap_m=occlusion_bridge_max_gap_m,
    )
    packaged.extend(occlusion_bridges)
    packaged.sort(key=topo.feature_sort_key)
    return packaged, replacement_rows, bridge_rows, occlusion_rows


def build_replacement_feature(
    original: dict[str, Any],
    refined: dict[str, Any],
    audit_row: dict[str, Any],
    *,
    band_index: dict[str, list[dict[str, Any]]] | None = None,
    tangent_snap_max_m: float = 0.0,
    tangent_smooth_taper_m: float = 0.0,
    tangent_window_m: float = 6.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    branch_id = str((original.get("properties") or {}).get("branch_id", ""))
    original_coords = line_coords(original)
    refined_coords = line_coords(refined)
    original_props = dict(original.get("properties") or {})
    refined_props = dict(refined.get("properties") or {})
    adjusted_coords, tangent_rows = smooth_turnout_to_connected_bands(
        refined_coords,
        original_props,
        band_index=band_index or {},
        snap_max_m=tangent_snap_max_m,
        taper_m=tangent_smooth_taper_m,
        tangent_window_m=tangent_window_m,
    )
    tangent_applied = [row for row in tangent_rows if row["applied"]]
    max_tangent_shift = max((row["endpoint_shift_m"] for row in tangent_applied), default=0.0)
    props = {
        **original_props,
        "line_id": original_props.get("line_id", f"TURNOUT_{branch_id}"),
        "branch_id": branch_id,
        "network_role": "turnout_connector",
        "geom_kind": "outside_rail_offset_centerline",
        "source": "deeplab_outer_rail_postprocess",
        "source_layer": "strict_auto_outer_rail_kinkfix",
        "review_status": "not_user_reviewed",
        "postprocess_source_line_id": refined_props.get("line_id", ""),
        "length_m": round(polyline_length(adjusted_coords), 3),
        "outer_valid_ratio": safe_float(audit_row.get("valid_outer_rail_ratio")),
        "outer_correction_p95_abs_m": safe_float(audit_row.get("correction_p95_abs_m")),
        "outer_invalid_ratio": safe_float(audit_row.get("invalid_ratio")),
        "outer_single_left_ratio": safe_float(audit_row.get("single_left_ratio")),
        "outer_single_right_ratio": safe_float(audit_row.get("single_right_ratio")),
        "outer_max_turn_deg": safe_float(audit_row.get("max_turn_deg")),
        "outer_p95_turn_deg": safe_float(audit_row.get("p95_turn_deg")),
        "endpoint_tangent_policy": "connected_band_tangent",
        "endpoint_tangent_applied_count": len(tangent_applied),
        "endpoint_tangent_max_shift_m": round(max_tangent_shift, 4),
        "endpoint_snap_count": len(tangent_applied),
        "endpoint_snap_max_m": round(max_tangent_shift, 4),
    }
    if tangent_applied:
        props["self_note"] = "turnout endpoints tangent-smoothed to connected straight band"
    feature = {
        "type": "Feature",
        "properties": props,
        "geometry": {
            "type": "LineString",
            "coordinates": [[round(x, 6), round(y, 6)] for x, y in adjusted_coords],
        },
    }
    tangent_by_endpoint = {row["endpoint"]: row for row in tangent_rows}
    row = {
        "branch_id": branch_id,
        "line_id": props["line_id"],
        "old_point_count": len(original_coords),
        "new_point_count": len(adjusted_coords),
        "old_length_m": round(polyline_length(original_coords), 3),
        "new_length_m": round(polyline_length(adjusted_coords), 3),
        "start_shift_m": round(point_distance(original_coords[0], adjusted_coords[0]), 4),
        "end_shift_m": round(point_distance(original_coords[-1], adjusted_coords[-1]), 4),
        "valid_outer_rail_ratio": props["outer_valid_ratio"],
        "invalid_ratio": props["outer_invalid_ratio"],
        "single_fallback_ratio": round(props["outer_single_left_ratio"] + props["outer_single_right_ratio"], 4),
        "max_turn_deg": props["outer_max_turn_deg"],
        "tangent_applied_count": len(tangent_applied),
    }
    for endpoint in ("start", "end"):
        endpoint_row = tangent_by_endpoint.get(endpoint, {})
        prefix = f"{endpoint}_tangent"
        row[f"{prefix}_applied"] = int(bool(endpoint_row.get("applied", False)))
        row[f"{prefix}_band_id"] = endpoint_row.get("band_id", "")
        row[f"{prefix}_endpoint_shift_m"] = endpoint_row.get("endpoint_shift_m", 0.0)
        row[f"{prefix}_before_angle_deg"] = endpoint_row.get("before_angle_deg", 0.0)
        row[f"{prefix}_after_angle_deg"] = endpoint_row.get("after_angle_deg", 0.0)
    return feature, row


def build_straight_band_index(features: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_band: dict[str, list[dict[str, Any]]] = {}
    for feature in features:
        props = feature.get("properties") or {}
        if str(props.get("network_role", "")) not in {"main_through_track", "parallel_straight_track"}:
            continue
        band_id = str(props.get("band_id", ""))
        if not band_id:
            continue
        by_band.setdefault(band_id, []).append(feature)
    return by_band


def smooth_turnout_to_connected_bands(
    coords: list[tuple[float, float]],
    props: dict[str, Any],
    *,
    band_index: dict[str, list[dict[str, Any]]],
    snap_max_m: float,
    taper_m: float,
    tangent_window_m: float,
) -> tuple[list[tuple[float, float]], list[dict[str, Any]]]:
    adjusted = list(coords)
    rows: list[dict[str, Any]] = []
    if len(adjusted) < 2 or snap_max_m <= 0.0 or taper_m <= 0.0:
        return adjusted, rows

    for endpoint, band_key in (("start", "start_band"), ("end", "end_band")):
        band_id = str(props.get(band_key, ""))
        band_features = band_index.get(band_id, [])
        adjusted, row = apply_endpoint_tangency(
            adjusted,
            endpoint=endpoint,
            band_id=band_id,
            band_features=band_features,
            snap_max_m=snap_max_m,
            taper_m=taper_m,
            tangent_window_m=tangent_window_m,
        )
        rows.append(row)
    return adjusted, rows


def apply_endpoint_tangency(
    coords: list[tuple[float, float]],
    *,
    endpoint: str,
    band_id: str,
    band_features: list[dict[str, Any]],
    snap_max_m: float,
    taper_m: float,
    tangent_window_m: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    empty_row = {
        "endpoint": endpoint,
        "band_id": band_id,
        "applied": False,
        "endpoint_shift_m": 0.0,
        "before_angle_deg": 0.0,
        "after_angle_deg": 0.0,
    }
    if not band_id or not band_features or len(coords) < 2:
        return coords, empty_row

    endpoint_point = coords[0] if endpoint == "start" else coords[-1]
    candidates = []
    for band_feature in band_features:
        band_coords = line_coords(band_feature)
        if len(band_coords) < 2:
            continue
        nearest = nearest_point_on_polyline(band_coords, endpoint_point)
        if nearest is None:
            continue
        distance_m, station_m, point = nearest
        candidates.append((distance_m, station_m, point, band_coords, band_feature))
    if not candidates:
        return coords, empty_row

    distance_m, station_m, anchor, band_coords, band_feature = min(candidates, key=lambda item: item[0])
    if distance_m > snap_max_m:
        row = dict(empty_row)
        row.update(
            {
                "band_line_id": str((band_feature.get("properties") or {}).get("line_id", "")),
                "endpoint_shift_m": round(distance_m, 4),
            }
        )
        return coords, row

    diagnostic_window_m = min(max(tangent_window_m, 0.5), 1.5)
    line_direction = endpoint_direction(coords, endpoint=endpoint, window_m=diagnostic_window_m)
    band_tangent = tangent_at_station(band_coords, station_m, window_m=tangent_window_m)
    if dot_product(line_direction, band_tangent) < 0.0:
        band_tangent = (-band_tangent[0], -band_tangent[1])
    before_angle = vector_angle_deg(line_direction, band_tangent)

    adjusted = smooth_endpoint_to_tangent(
        coords,
        endpoint=endpoint,
        anchor=anchor,
        tangent=band_tangent,
        taper_m=taper_m,
    )
    after_direction = endpoint_direction(adjusted, endpoint=endpoint, window_m=diagnostic_window_m)
    after_angle = vector_angle_deg(after_direction, band_tangent)

    row = {
        "endpoint": endpoint,
        "band_id": band_id,
        "band_line_id": str((band_feature.get("properties") or {}).get("line_id", "")),
        "applied": True,
        "endpoint_shift_m": round(distance_m, 4),
        "before_angle_deg": round(before_angle, 4),
        "after_angle_deg": round(after_angle, 4),
    }
    return adjusted, row


def smooth_endpoint_to_tangent(
    coords: list[tuple[float, float]],
    *,
    endpoint: str,
    anchor: tuple[float, float],
    tangent: tuple[float, float],
    taper_m: float,
) -> list[tuple[float, float]]:
    lengths = cumulative_lengths(coords)
    total = lengths[-1]
    if total <= 0.0:
        return coords
    max_taper = min(max(taper_m, 0.0), total)
    if max_taper <= 0.0:
        return coords

    adjusted = list(coords)
    tangent_lock_m = min(3.0, max_taper * 0.5)
    for index, point in enumerate(coords):
        endpoint_distance = lengths[index] if endpoint == "start" else total - lengths[index]
        if endpoint_distance > max_taper:
            continue
        if endpoint_distance <= tangent_lock_m:
            weight = 1.0
        else:
            fade_span = max(max_taper - tangent_lock_m, 1e-9)
            u = min(max((endpoint_distance - tangent_lock_m) / fade_span, 0.0), 1.0)
            weight = 1.0 - smoothstep(u)
        if endpoint == "start":
            target = (anchor[0] + tangent[0] * endpoint_distance, anchor[1] + tangent[1] * endpoint_distance)
        else:
            target = (anchor[0] - tangent[0] * endpoint_distance, anchor[1] - tangent[1] * endpoint_distance)
        adjusted[index] = (
            point[0] * (1.0 - weight) + target[0] * weight,
            point[1] * (1.0 - weight) + target[1] * weight,
        )
    if endpoint == "start":
        adjusted[0] = anchor
    else:
        adjusted[-1] = anchor
    return adjusted


def nearest_point_on_polyline(
    coords: list[tuple[float, float]],
    point: tuple[float, float],
) -> tuple[float, float, tuple[float, float]] | None:
    best: tuple[float, float, tuple[float, float]] | None = None
    station_start = 0.0
    for start, end in zip(coords, coords[1:]):
        segment = (end[0] - start[0], end[1] - start[1])
        length_sq = dot_product(segment, segment)
        if length_sq <= 0.0:
            continue
        t = max(0.0, min(1.0, dot_product((point[0] - start[0], point[1] - start[1]), segment) / length_sq))
        projected = (start[0] + segment[0] * t, start[1] + segment[1] * t)
        distance_m = point_distance(point, projected)
        station_m = station_start + math.sqrt(length_sq) * t
        if best is None or distance_m < best[0]:
            best = (distance_m, station_m, projected)
        station_start += math.sqrt(length_sq)
    return best


def tangent_at_station(coords: list[tuple[float, float]], station_m: float, *, window_m: float) -> tuple[float, float]:
    lengths = cumulative_lengths(coords)
    total = lengths[-1]
    if total <= 0.0:
        return (1.0, 0.0)
    before = point_at_station(coords, lengths, max(0.0, station_m - window_m))
    after = point_at_station(coords, lengths, min(total, station_m + window_m))
    tangent = unit_vector((after[0] - before[0], after[1] - before[1]))
    if tangent == (0.0, 0.0):
        return unit_vector((coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1]))
    return tangent


def endpoint_direction(coords: list[tuple[float, float]], *, endpoint: str, window_m: float) -> tuple[float, float]:
    lengths = cumulative_lengths(coords)
    total = lengths[-1]
    if total <= 0.0:
        return (1.0, 0.0)
    if endpoint == "start":
        before = point_at_station(coords, lengths, 0.0)
        after = point_at_station(coords, lengths, min(total, window_m))
    else:
        before = point_at_station(coords, lengths, max(0.0, total - window_m))
        after = point_at_station(coords, lengths, total)
    direction = unit_vector((after[0] - before[0], after[1] - before[1]))
    if direction == (0.0, 0.0):
        return (1.0, 0.0)
    return direction


def point_at_station(coords: list[tuple[float, float]], lengths: list[float], station_m: float) -> tuple[float, float]:
    station_m = min(max(station_m, 0.0), lengths[-1])
    for index in range(len(lengths) - 1):
        if lengths[index + 1] >= station_m:
            segment_length = max(lengths[index + 1] - lengths[index], 1e-9)
            t = (station_m - lengths[index]) / segment_length
            start = coords[index]
            end = coords[index + 1]
            return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
    return coords[-1]


def cumulative_lengths(coords: list[tuple[float, float]]) -> list[float]:
    lengths = [0.0]
    total = 0.0
    for start, end in zip(coords, coords[1:]):
        total += point_distance(start, end)
        lengths.append(total)
    return lengths


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def unit_vector(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    if length <= 0.0:
        return (0.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def dot_product(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def vector_angle_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    a_unit = unit_vector(a)
    b_unit = unit_vector(b)
    if a_unit == (0.0, 0.0) or b_unit == (0.0, 0.0):
        return 0.0
    cosine = max(-1.0, min(1.0, dot_product(a_unit, b_unit)))
    return math.degrees(math.acos(cosine))


def adjust_boundary_bridge(
    bridge: dict[str, Any],
    *,
    old_turnout: dict[str, Any],
    refined_turnout: dict[str, Any],
    min_bridge_length_m: float = 0.05,
) -> dict[str, Any] | None:
    bridge_coords = line_coords(bridge)
    old_coords = line_coords(old_turnout)
    refined_coords = line_coords(refined_turnout)
    candidates = [
        (0, "start", point_distance(bridge_coords[0], old_coords[0]), refined_coords[0]),
        (0, "end", point_distance(bridge_coords[0], old_coords[-1]), refined_coords[-1]),
        (-1, "start", point_distance(bridge_coords[-1], old_coords[0]), refined_coords[0]),
        (-1, "end", point_distance(bridge_coords[-1], old_coords[-1]), refined_coords[-1]),
    ]
    endpoint_index, turnout_endpoint, old_distance, replacement = min(candidates, key=lambda item: item[2])
    if old_distance > 2.5:
        return None
    before = bridge_coords[endpoint_index]
    bridge_coords[endpoint_index] = replacement
    bridge_coords = [bridge_coords[0], bridge_coords[-1]]
    bridge_length = polyline_length(bridge_coords)
    dropped_bridge = bridge_length <= min_bridge_length_m
    bridge["geometry"]["coordinates"] = [[round(x, 6), round(y, 6)] for x, y in bridge_coords]
    props = bridge.setdefault("properties", {})
    props["boundary_adjusted_to_refined_turnout"] = 1
    props["boundary_adjusted_endpoint"] = "start" if endpoint_index == 0 else "end"
    props["boundary_turnout_endpoint"] = turnout_endpoint
    props["boundary_endpoint_shift_m"] = round(point_distance(before, replacement), 4)
    props["boundary_bridge_simplified"] = 1
    props["boundary_bridge_dropped"] = int(dropped_bridge)
    props["length_m"] = round(bridge_length, 3)
    return {
        "line_id": props.get("line_id", ""),
        "branch_id": props.get("branch_id", ""),
        "adjusted_endpoint": props["boundary_adjusted_endpoint"],
        "turnout_endpoint": turnout_endpoint,
        "old_distance_to_original_turnout_m": round(old_distance, 4),
        "endpoint_shift_m": props["boundary_endpoint_shift_m"],
        "new_length_m": props["length_m"],
        "dropped_bridge": int(dropped_bridge),
    }


def build_same_band_occlusion_bridges(
    features: list[dict[str, Any]],
    *,
    min_gap_m: float,
    max_gap_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_gap_m <= 0.0 or min_gap_m > max_gap_m:
        return [], []
    by_band: dict[str, list[dict[str, Any]]] = {}
    for feature in features:
        props = feature.get("properties") or {}
        if str(props.get("network_role", "")) not in {"main_through_track", "parallel_straight_track"}:
            continue
        band_id = str(props.get("band_id", ""))
        if not band_id:
            continue
        by_band.setdefault(band_id, []).append(feature)

    bridges: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for band_id, band_features in sorted(by_band.items()):
        ordered = sorted(band_features, key=lambda feature: station_min(feature))
        for left, right in zip(ordered, ordered[1:]):
            left_coords = line_coords(left)
            right_coords = line_coords(right)
            if not left_coords or not right_coords:
                continue
            station_gap = station_min(right) - station_max(left)
            endpoint_gap = point_distance(left_coords[-1], right_coords[0])
            if station_gap < min_gap_m or station_gap > max_gap_m:
                continue
            if abs(endpoint_gap - station_gap) > max(1.0, station_gap * 0.05):
                continue
            line_id = f"OCCLUSION_BRIDGE_{band_id}_{len(rows) + 1:02d}"
            bridge_coords = [left_coords[-1], right_coords[0]]
            props = {
                "line_id": line_id,
                "band_id": band_id,
                "network_role": "straight_gap_bridge",
                "source_layer": "topology_gap_bridge",
                "source": "same_band_occlusion_bridge",
                "role": "occlusion_bridge",
                "bridge_kind": "occlusion_bridge",
                "risk_flag": "occlusion_bridge",
                "qa_status": "bridge_needs_review",
                "length_m": round(polyline_length(bridge_coords), 3),
                "station_min_m": round(station_max(left), 3),
                "station_max_m": round(station_min(right), 3),
                "gap_m": round(station_gap, 3),
                "endpoint_gap_m": round(endpoint_gap, 3),
                "left_line_id": str((left.get("properties") or {}).get("line_id", "")),
                "right_line_id": str((right.get("properties") or {}).get("line_id", "")),
                "postprocess_policy": "same_band_long_occlusion_bridge",
                "review_note": "same-band internal gap bridged as likely vehicle/train occlusion; requires visual review",
            }
            bridges.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in bridge_coords]},
                }
            )
            rows.append(
                {
                    "line_id": line_id,
                    "band_id": band_id,
                    "left_line_id": props["left_line_id"],
                    "right_line_id": props["right_line_id"],
                    "station_min_m": props["station_min_m"],
                    "station_max_m": props["station_max_m"],
                    "gap_m": props["gap_m"],
                    "endpoint_gap_m": props["endpoint_gap_m"],
                    "bridge_kind": "occlusion_bridge",
                }
            )
    return bridges, rows


def station_min(feature: dict[str, Any]) -> float:
    props = feature.get("properties") or {}
    return safe_float(props.get("station_min_m", props.get("s_min_m", 0.0)))


def station_max(feature: dict[str, Any]) -> float:
    props = feature.get("properties") or {}
    return safe_float(props.get("station_max_m", props.get("s_max_m", 0.0)))


def clone_feature(feature: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(feature))


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "LineString":
        return []
    return [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates", [])]


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(point_distance(a, b) for a, b in zip(coords, coords[1:]))


def point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def safe_float(value: Any) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    features: list[dict[str, Any]],
    replacement_rows: list[dict[str, Any]],
    bridge_rows: list[dict[str, Any]],
    occlusion_rows: list[dict[str, Any]],
    outputs: dict[str, Path | None],
) -> dict[str, Any]:
    role_counts = Counter(str((feature.get("properties") or {}).get("network_role", "")) for feature in features)
    tangent_feature_count = sum(1 for row in replacement_rows if int(row.get("tangent_applied_count", 0)) > 0)
    tangent_endpoint_count = sum(int(row.get("tangent_applied_count", 0)) for row in replacement_rows)
    return {
        "mode": "strict_auto_global_centerline_review_tangent_occlusion",
        "policy": (
            "Reuse existing full-DOM strict-auto 2D network and replace all turnout_connector features "
            "with DeepLab outer-rail postprocessed kinkfix turnouts, then smooth turnout endpoints to the "
            "tangent of their connected straight-band centerlines. Semantic segmentation is not rerun."
        ),
        "base_network": str(args.base_network.expanduser().resolve()),
        "base_evidence": str(args.base_evidence.expanduser().resolve()) if args.base_evidence.exists() else None,
        "refined_turnouts": str(args.refined_turnouts.expanduser().resolve()),
        "turnout_audit_dir": str(args.turnout_audit_dir.expanduser().resolve()),
        "out_dir": str(out_dir),
        "feature_count": len(features),
        "network_role_counts": dict(sorted(role_counts.items())),
        "turnout_replacement_count": len(replacement_rows),
        "turnout_tangent_smoothed_feature_count": tangent_feature_count,
        "turnout_tangent_smoothed_endpoint_count": tangent_endpoint_count,
        "tangent_snap_max_m": args.tangent_snap_max_m,
        "tangent_smooth_taper_m": args.tangent_smooth_taper_m,
        "tangent_window_m": args.tangent_window_m,
        "boundary_bridge_adjustment_count": len(bridge_rows),
        "boundary_bridge_dropped_count": sum(1 for row in bridge_rows if int(row.get("dropped_bridge", 0)) == 1),
        "occlusion_bridge_count": len(occlusion_rows),
        "occlusion_bridge_min_gap_m": args.occlusion_bridge_min_gap_m,
        "occlusion_bridge_max_gap_m": args.occlusion_bridge_max_gap_m,
        "outputs": {key: None if path is None else str(path) for key, path in outputs.items()},
    }


def write_review(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Strict Auto Global Centerline Review",
        "",
        "This package is for full-DOM 2D visual review. It reuses the existing strict-auto full-DOM network, replaces the seven turnout connector features with the latest DeepLab outer-rail postprocessed kinkfix results, and smooths turnout endpoints to the tangent of their connected straight-band centerlines.",
        "",
        "## Boundaries",
        "",
        "- Semantic segmentation was not rerun.",
        "- `final_delivery` was not overwritten.",
        "- QA coordinates are not production constraints.",
        "- Z/3D export is still downstream of 2D acceptance.",
        "",
        "## Outputs",
        "",
        f"- 2D Shapefile: `{summary['outputs']['network_shp']}`",
        f"- 2D GeoJSON: `{summary['outputs']['network_geojson']}`",
        f"- QGIS style: `{summary['outputs']['network_qml']}`",
        f"- Turnout replacements: `{summary['outputs']['turnout_replacements_csv']}`",
        f"- Boundary bridge adjustments: `{summary['outputs']['boundary_bridge_adjustments_csv']}`",
        f"- Occlusion bridges: `{summary['outputs']['occlusion_bridges_csv']}`",
        "",
        "## Counts",
        "",
        f"- Feature count: {summary['feature_count']}",
        f"- Turnout replacements: {summary['turnout_replacement_count']}",
        f"- Turnout tangent-smoothed features: {summary['turnout_tangent_smoothed_feature_count']}",
        f"- Turnout tangent-smoothed endpoints: {summary['turnout_tangent_smoothed_endpoint_count']}",
        f"- Tangent snap tolerance: {summary['tangent_snap_max_m']}m",
        f"- Tangent smooth taper: {summary['tangent_smooth_taper_m']}m",
        f"- Boundary bridge endpoint adjustments: {summary['boundary_bridge_adjustment_count']}",
        f"- Boundary bridges dropped as degenerate: {summary['boundary_bridge_dropped_count']}",
        f"- Occlusion bridges: {summary['occlusion_bridge_count']}",
        f"- Occlusion bridge gap window: {summary['occlusion_bridge_min_gap_m']}m to {summary['occlusion_bridge_max_gap_m']}m",
    ]
    for role, count in summary["network_role_counts"].items():
        lines.append(f"- `{role}`: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
