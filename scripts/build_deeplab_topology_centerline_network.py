from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import rasterio.crs


DEFAULT_DEEPLAB_NETWORK = Path("output/raw_dom_roi_fullpass_v1/deeplab_centerline_network_v1/deeplab_centerline_network_v1.geojson")
DEFAULT_TRACK_BANDS = Path("output/raw_dom_roi_fullpass_v1/track_band_priors/track_band_centerline_priors.geojson")
DEFAULT_TURNOUTS = Path("output/raw_dom_roi_fullpass_v1/all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson")
DEFAULT_GAUGE_PAIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_turnouts_v1/deeplab_gauge_pair_centerlines.geojson")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_v1")
DEFAULT_EPSG = 32651

KEEP_BANDS = {"mainline_2_track", "parallel_minus_5m", "parallel_plus_5m"}
MAIN_BAND_ID = "mainline_2_track"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a topology-aware centerline network from DeepLab evidence and track priors.")
    parser.add_argument("--deeplab-network", type=Path, default=DEFAULT_DEEPLAB_NETWORK)
    parser.add_argument("--track-bands", type=Path, default=DEFAULT_TRACK_BANDS)
    parser.add_argument("--turnouts", type=Path, default=DEFAULT_TURNOUTS)
    parser.add_argument("--gauge-pair", type=Path, default=DEFAULT_GAUGE_PAIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--support-threshold-m", type=float, default=0.85)
    parser.add_argument("--sample-step-m", type=float, default=5.0)
    parser.add_argument("--bridge-min-gap-m", type=float, default=8.0)
    parser.add_argument("--bridge-max-gap-m", type=float, default=95.0)
    parser.add_argument("--bridge-evidence-support", type=float, default=0.45)
    parser.add_argument(
        "--weak-gap-bridge-max-gap-m",
        type=float,
        default=95.0,
        help=(
            "Maximum same-band gap that may be bridged by topology alone when DeepLab evidence is weak. "
            "Set below --bridge-max-gap-m for strict automatic delivery."
        ),
    )
    parser.add_argument("--bridge-turnout-clearance-m", type=float, default=12.0)
    parser.add_argument("--diagnostic-promote-support", type=float, default=0.65)
    parser.add_argument("--diagnostic-min-length-m", type=float, default=20.0)
    parser.add_argument("--turnout-tail-bridge-max-gap-m", type=float, default=45.0)
    parser.add_argument("--turnout-tail-bridge-support", type=float, default=0.35)
    parser.add_argument(
        "--snap-connector-endpoints-m",
        type=float,
        default=0.0,
        help="Optionally snap turnout/bridge endpoints to nearby accepted track lines for QGIS review topology.",
    )
    parser.add_argument(
        "--allow-specialized-turnout-rebuilds",
        action="store_true",
        help="Compatibility/debug only: allow legacy named-turnout rebuild logic.",
    )
    parser.add_argument("--exclude-low-support-turnouts", action="store_true", help="Drop turnout candidates that earlier QA marked as low support.")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    deeplab_features = load_line_features(args.deeplab_network.expanduser().resolve())
    gauge_features = load_line_features(args.gauge_pair.expanduser().resolve()) if args.gauge_pair.exists() else []
    evidence_features = deeplab_features + gauge_features

    all_band_features = load_line_features(args.track_bands.expanduser().resolve())
    band_features = select_track_band_features(all_band_features)
    diagnostic_band_features = select_diagnostic_track_band_features(all_band_features)
    turnout_features = select_turnout_features(load_line_features(args.turnouts.expanduser().resolve()), include_low_support=not args.exclude_low_support_turnouts)

    final_features = build_topology_features(
        band_features=band_features,
        diagnostic_band_features=diagnostic_band_features,
        turnout_features=turnout_features,
        evidence_features=evidence_features,
        support_threshold_m=args.support_threshold_m,
        sample_step_m=args.sample_step_m,
        bridge_min_gap_m=args.bridge_min_gap_m,
        bridge_max_gap_m=args.bridge_max_gap_m,
        bridge_evidence_support=args.bridge_evidence_support,
        weak_gap_bridge_max_gap_m=args.weak_gap_bridge_max_gap_m,
        bridge_turnout_clearance_m=args.bridge_turnout_clearance_m,
        diagnostic_promote_support=args.diagnostic_promote_support,
        diagnostic_min_length_m=args.diagnostic_min_length_m,
        turnout_tail_bridge_max_gap_m=args.turnout_tail_bridge_max_gap_m,
        turnout_tail_bridge_support=args.turnout_tail_bridge_support,
        allow_specialized_turnout_rebuilds=args.allow_specialized_turnout_rebuilds,
    )
    if args.snap_connector_endpoints_m > 0:
        final_features = snap_connector_endpoints(final_features, max_distance_m=args.snap_connector_endpoints_m)

    network_geojson = out_dir / "deeplab_topology_centerline_network.geojson"
    evidence_geojson = out_dir / "deeplab_topology_evidence.geojson"
    summary_path = out_dir / "summary.json"
    review_path = out_dir / "REVIEW.md"

    write_geojson(network_geojson, final_features, epsg=args.epsg)
    write_geojson(evidence_geojson, normalize_evidence_features(evidence_features), epsg=args.epsg)
    write_centerline_shapefile(final_features, network_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_evidence_shapefile(normalize_evidence_features(evidence_features), evidence_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_centerline_qml(network_geojson.with_suffix(".qml"))
    write_evidence_qml(evidence_geojson.with_suffix(".qml"))

    summary = summarize_run(
        final_features,
        deeplab_features=deeplab_features,
        gauge_features=gauge_features,
        args=args,
        out_dir=out_dir,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review_markdown(review_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_topology_features(
    *,
    band_features: list[dict[str, Any]],
    diagnostic_band_features: list[dict[str, Any]] | None = None,
    turnout_features: list[dict[str, Any]],
    evidence_features: list[dict[str, Any]],
    support_threshold_m: float,
    sample_step_m: float,
    bridge_min_gap_m: float = 8.0,
    bridge_max_gap_m: float = 95.0,
    bridge_evidence_support: float = 0.45,
    weak_gap_bridge_max_gap_m: float = 95.0,
    bridge_turnout_clearance_m: float = 12.0,
    diagnostic_promote_support: float = 0.65,
    diagnostic_min_length_m: float = 20.0,
    turnout_tail_bridge_max_gap_m: float = 45.0,
    turnout_tail_bridge_support: float = 0.35,
    allow_specialized_turnout_rebuilds: bool = False,
) -> list[dict[str, Any]]:
    evidence_segments = build_segments(evidence_features)
    band_features = trim_straight_band_endpoint_overlaps(
        band_features,
        turnout_features=turnout_features,
    )
    output: list[dict[str, Any]] = []
    for feature in band_features:
        output.append(
            annotate_feature(
                feature,
                evidence_segments=evidence_segments,
                network_role=band_network_role(feature),
                source_layer="track_band_prior",
                line_id=band_line_id(feature),
                support_threshold_m=support_threshold_m,
                sample_step_m=sample_step_m,
            )
        )
    promoted_diagnostics = build_promoted_diagnostic_tracks(
        diagnostic_band_features or [],
        evidence_segments=evidence_segments,
        support_threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
        min_support_ratio=diagnostic_promote_support,
        min_length_m=diagnostic_min_length_m,
    )
    promoted_diagnostics = filter_promoted_diagnostics_used_by_turnouts(
        promoted_diagnostics,
        turnout_features=turnout_features,
    )
    output.extend(promoted_diagnostics)
    for feature in build_straight_gap_bridges(
        band_features + promoted_diagnostics,
        turnout_features=turnout_features,
        evidence_segments=evidence_segments,
        support_threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
        min_gap_m=bridge_min_gap_m,
        max_gap_m=bridge_max_gap_m,
        evidence_support_threshold=bridge_evidence_support,
        weak_gap_max_m=weak_gap_bridge_max_gap_m,
        turnout_clearance_m=bridge_turnout_clearance_m,
    ):
        output.append(feature)
    output.extend(
        build_turnout_tail_bridges(
            promoted_diagnostics,
            turnout_features=turnout_features,
            evidence_segments=evidence_segments,
            support_threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
            max_gap_m=turnout_tail_bridge_max_gap_m,
            min_support_ratio=turnout_tail_bridge_support,
        )
    )
    for feature in turnout_features:
        output.append(
            annotate_feature(
                feature,
                evidence_segments=evidence_segments,
                network_role="turnout_connector",
                source_layer="turnout_connector_prior",
                line_id=turnout_line_id(feature),
                support_threshold_m=support_threshold_m,
                sample_step_m=sample_step_m,
            )
        )
    if allow_specialized_turnout_rebuilds:
        output = rebuild_ta08_curved_branch(
            output,
            evidence_features=evidence_features,
            evidence_segments=evidence_segments,
            support_threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
        )
    output = smooth_turnout_connectors_with_evidence(
        output,
        evidence_segments=evidence_segments,
        support_threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    output = rebuild_crossover_connectors_with_evidence(
        output,
        evidence_features=evidence_features,
        evidence_segments=evidence_segments,
        support_threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    output.extend(
        build_turnout_boundary_evidence_bridges(
            output,
            evidence_features=evidence_features,
            evidence_segments=evidence_segments,
            support_threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
        )
    )
    output.sort(key=feature_sort_key)
    return output


def rebuild_crossover_connectors_with_evidence(
    features: list[dict[str, Any]],
    *,
    evidence_features: list[dict[str, Any]],
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
) -> list[dict[str, Any]]:
    mainline = find_feature_by_line_id(features, "BAND_mainline_2_track_0")
    if mainline is None:
        return features
    guide = LinearGuide(line_coords(mainline)[0], line_coords(mainline)[-1])
    rebuilt: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties") or {}
        if not is_crossover_connector(feature):
            rebuilt.append(feature)
            continue
        evidence = crossover_evidence_for_feature(feature, evidence_features)
        candidate = build_semseg_crossover_candidate(
            feature,
            evidence_features=evidence,
            guide=guide,
            evidence_segments=evidence_segments,
            support_threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
        )
        rebuilt.append(candidate or feature)
    return rebuilt


def is_crossover_connector(feature: dict[str, Any]) -> bool:
    props = feature.get("properties") or {}
    if str(props.get("network_role", "")) != "turnout_connector":
        return False
    connector_id = str(props.get("connector_id", props.get("branch_id", props.get("line_id", "")))).upper()
    shape_model = str(props.get("shape_model", ""))
    pair_id = str(props.get("pair_id", ""))
    return connector_id.startswith("CX") or "crossover" in shape_model or pair_id == "minus_to_main"


def crossover_evidence_for_feature(feature: dict[str, Any], evidence_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    props = feature.get("properties") or {}
    connector_id = str(props.get("connector_id", props.get("branch_id", ""))).strip()
    if not connector_id:
        return []
    matched: list[dict[str, Any]] = []
    for evidence in evidence_features:
        ev_props = evidence.get("properties") or {}
        branch_id = str(ev_props.get("branch_id", ev_props.get("source_branch_id", ""))).strip()
        seq_id = str(ev_props.get("seq_id", ev_props.get("line_id", ""))).strip()
        if branch_id == connector_id or seq_id.startswith(f"{connector_id}_"):
            if len(line_coords(evidence)) >= 2:
                matched.append(evidence)
    return matched


def build_semseg_crossover_candidate(
    feature: dict[str, Any],
    *,
    evidence_features: list[dict[str, Any]],
    guide: LinearGuide,
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
) -> dict[str, Any] | None:
    if not evidence_features:
        return None
    original_coords = line_coords(feature)
    original_st = station_offsets_for_coords(original_coords, guide=guide)
    if len(original_st) < 2:
        return None
    start = original_st[0]
    end = original_st[-1]
    if end[0] - start[0] < 20.0:
        return None
    start_choice = select_crossover_start_breakpoint(feature, evidence_features, guide=guide)
    end_choice = select_crossover_end_breakpoint(feature, evidence_features, guide=guide)
    if start_choice is None or end_choice is None:
        return None
    start_break = start_choice["point"]
    end_break = end_choice["point"]
    if not (start[0] < start_break[0] < end_break[0] < end[0]):
        return None
    middle_span = end_break[0] - start_break[0]
    total_span = end[0] - start[0]
    if middle_span < 10.0 or middle_span < 0.25 * total_span:
        return None
    middle_slope = slope_between([start_break, end_break])
    start_slope = 0.0
    end_slope = 0.0
    parts: list[list[tuple[float, float]]] = []
    if start_break[0] - start[0] > 0.75:
        parts.append(cubic_hermite_station_offsets(start, start_break, start_slope, middle_slope, step_m=0.75))
    else:
        parts.append([start])
    parts.append(paired_rail_linear_segment([start_break, end_break], step_m=0.75))
    if end[0] - end_break[0] > 0.75:
        parts.append(cubic_hermite_station_offsets(end_break, end, middle_slope, end_slope, step_m=0.75))
    else:
        parts.append([end])
    rebuilt_st = clamp_offsets_to_endpoint_range(merge_station_offset_parts(parts), start[1], end[1])
    if len(rebuilt_st) < 4:
        return None
    coords = [[round(x, 6), round(y, 6)] for x, y in (guide.point_at(station, offset) for station, offset in rebuilt_st)]
    old_support = measure_support(
        original_coords,
        evidence_segments=evidence_segments,
        threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    new_support = measure_support(
        [(float(x), float(y)) for x, y in coords],
        evidence_segments=evidence_segments,
        threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    old_angle = max_local_angle_deg(original_coords)
    new_angle = max_local_angle_deg([(float(x), float(y)) for x, y in coords])
    if (
        new_support["mean_distance_m"] > old_support["mean_distance_m"] + 0.08
        or new_support["max_unsupported_gap_m"] > old_support["max_unsupported_gap_m"] + 5.0
        or new_angle > max(2.5, old_angle + 0.5)
    ):
        return build_endpoint_tangent_crossover_candidate(
            feature,
            start_choice=start_choice,
            end_choice=end_choice,
            guide=guide,
            evidence_segments=evidence_segments,
            support_threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
            old_support=old_support,
            old_angle=old_angle,
        )
    props = dict(feature.get("properties") or {})
    curvature = station_offset_curve_summary(rebuilt_st)
    props.update(
        {
            "geometry": "semseg_evidence_crossover_rebuild",
            "shape_model": "semseg_evidence_curve_straight_curve",
            "crossover_rebuild_status": "accepted",
            "crossover_start_evidence": start_choice["evidence_id"],
            "crossover_end_evidence": end_choice["evidence_id"],
            "crossover_start_break_s": round(start_break[0], 3),
            "crossover_end_break_s": round(end_break[0], 3),
            "crossover_middle_len_m": round(middle_span, 3),
            "crossover_middle_slope": round(middle_slope, 5),
            "crossover_endpoint_tangent_slope": 0.0,
            "length_m": round(polyline_length([(float(x), float(y)) for x, y in coords]), 3),
            "deeplab_support_ratio": new_support["support_ratio"],
            "deeplab_mean_distance_m": new_support["mean_distance_m"],
            "deeplab_max_unsupported_gap_m": new_support["max_unsupported_gap_m"],
            "deeplab_sample_count": new_support["sample_count"],
            "crossover_old_support_ratio": old_support["support_ratio"],
            "crossover_old_mean_distance_m": old_support["mean_distance_m"],
            "crossover_old_angle_deg": round(old_angle, 3),
            "crossover_new_angle_deg": round(new_angle, 3),
            "crossover_rebuild_mode": "semseg_boundary_straight_middle",
            "curve_start_s_m": round(curvature["curve_start_s_m"], 3),
            "curve_end_s_m": round(curvature["curve_end_s_m"], 3),
            "max_abs_slope": round(curvature["max_abs_slope"], 4),
            "max_abs_curvature": round(curvature["max_abs_curvature"], 5),
            "postprocess_policy": "semseg_evidence_crossover_curve_straight_curve_rebuild_with_straight_endpoint_tangent",
            "review_note": "crossover connector rebuilt from DeepLab gauge-pair boundary evidence; endpoints are tangent to straight track centers and clamped against centerline overshoot",
        }
    )
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def build_endpoint_tangent_crossover_candidate(
    feature: dict[str, Any],
    *,
    start_choice: dict[str, Any],
    end_choice: dict[str, Any],
    guide: LinearGuide,
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
    old_support: dict[str, Any],
    old_angle: float,
) -> dict[str, Any] | None:
    original_st = station_offsets_for_coords(line_coords(feature), guide=guide)
    if len(original_st) < 2:
        return None
    start = original_st[0]
    end = original_st[-1]
    span = end[0] - start[0]
    if span <= 20.0:
        return None
    candidates: list[dict[str, Any]] = []
    for curve_fraction in (0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44):
        station_offsets = curve_straight_curve_station_offsets(start, end, curve_fraction=curve_fraction, step_m=0.75)
        coords = [[round(x, 6), round(y, 6)] for x, y in (guide.point_at(station, offset) for station, offset in station_offsets)]
        metric_coords = [(float(x), float(y)) for x, y in coords]
        support = measure_support(
            metric_coords,
            evidence_segments=evidence_segments,
            threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
        )
        angle = max_local_angle_deg(metric_coords)
        if support["support_ratio"] + 0.08 < old_support["support_ratio"]:
            continue
        if support["mean_distance_m"] > old_support["mean_distance_m"] + 0.08:
            continue
        if support["max_unsupported_gap_m"] > old_support["max_unsupported_gap_m"] + 5.0:
            continue
        if angle > max(2.5, old_angle + 0.5):
            continue
        candidates.append(
            {
                "curve_fraction": curve_fraction,
                "station_offsets": station_offsets,
                "coords": coords,
                "support": support,
                "angle": angle,
            }
        )
    if not candidates:
        return None
    best_support = max(item["support"]["support_ratio"] for item in candidates)
    best_mean = min(item["support"]["mean_distance_m"] for item in candidates if item["support"]["support_ratio"] >= best_support - 0.01)
    viable = [
        item
        for item in candidates
        if item["support"]["support_ratio"] >= best_support - 0.01 and item["support"]["mean_distance_m"] <= best_mean + 0.04
    ]
    chosen = sorted(viable, key=lambda item: (item["curve_fraction"], item["angle"]))[0]
    station_offsets = chosen["station_offsets"]
    coords = chosen["coords"]
    support = chosen["support"]
    angle = chosen["angle"]
    curve_fraction = float(chosen["curve_fraction"])
    curvature = station_offset_curve_summary(station_offsets)
    props = dict(feature.get("properties") or {})
    middle_len = span * max(0.0, 1.0 - 2.0 * curve_fraction)
    props.update(
        {
            "geometry": "semseg_scored_endpoint_tangent_crossover_rebuild",
            "shape_model": "semseg_scored_endpoint_tangent_curve_straight_curve",
            "crossover_rebuild_status": "accepted",
            "crossover_rebuild_mode": "endpoint_tangent_fullspan_semseg_scored",
            "crossover_start_evidence": start_choice["evidence_id"],
            "crossover_end_evidence": end_choice["evidence_id"],
            "crossover_start_break_s": round(start[0] + span * curve_fraction, 3),
            "crossover_end_break_s": round(end[0] - span * curve_fraction, 3),
            "crossover_middle_len_m": round(middle_len, 3),
            "crossover_curve_fraction": round(curve_fraction, 3),
            "crossover_middle_slope": round(slope_between([station_offsets[0], station_offsets[-1]]) / max(1.0 - curve_fraction, 1e-6), 5),
            "crossover_endpoint_tangent_slope": 0.0,
            "length_m": round(polyline_length([(float(x), float(y)) for x, y in coords]), 3),
            "deeplab_support_ratio": support["support_ratio"],
            "deeplab_mean_distance_m": support["mean_distance_m"],
            "deeplab_max_unsupported_gap_m": support["max_unsupported_gap_m"],
            "deeplab_sample_count": support["sample_count"],
            "crossover_old_support_ratio": old_support["support_ratio"],
            "crossover_old_mean_distance_m": old_support["mean_distance_m"],
            "crossover_old_angle_deg": round(old_angle, 3),
            "crossover_new_angle_deg": round(angle, 3),
            "curve_start_s_m": round(curvature["curve_start_s_m"], 3),
            "curve_end_s_m": round(curvature["curve_end_s_m"], 3),
            "max_abs_slope": round(curvature["max_abs_slope"], 4),
            "max_abs_curvature": round(curvature["max_abs_curvature"], 5),
            "postprocess_policy": "semseg_scored_crossover_rebuild_with_straight_endpoint_tangent_and_no_overshoot",
            "review_note": "crossover connector rebuilt as an endpoint-tangent monotone curve-straight-curve candidate selected by DeepLab support metrics",
        }
    )
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def curve_straight_curve_station_offsets(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    curve_fraction: float,
    step_m: float,
) -> list[tuple[float, float]]:
    station0, offset0 = start
    station1, offset1 = end
    span = station1 - station0
    if span <= 0.0:
        return [start, end]
    c = min(max(curve_fraction, 0.01), 0.49)
    count = max(12, int(math.ceil(span / max(step_m, 0.1))) + 1)
    points: list[tuple[float, float]] = []
    middle_slope = 1.0 / (1.0 - c)
    for index in range(count):
        u = index / (count - 1)
        if u <= c:
            fraction = 0.5 * middle_slope * u * u / c
        elif u >= 1.0 - c:
            fraction = 1.0 - 0.5 * middle_slope * (1.0 - u) * (1.0 - u) / c
        else:
            fraction = middle_slope * (u - 0.5 * c)
        offset = offset0 + (offset1 - offset0) * fraction
        points.append((station0 + span * u, offset))
    return clamp_offsets_to_endpoint_range(points, offset0, offset1)


def clamp_offsets_to_endpoint_range(
    points: list[tuple[float, float]],
    start_offset: float,
    end_offset: float,
) -> list[tuple[float, float]]:
    low = min(start_offset, end_offset)
    high = max(start_offset, end_offset)
    return [(station, min(high, max(low, offset))) for station, offset in points]


def select_crossover_start_breakpoint(
    feature: dict[str, Any],
    evidence_features: list[dict[str, Any]],
    *,
    guide: LinearGuide,
) -> dict[str, Any] | None:
    original_st = station_offsets_for_coords(line_coords(feature), guide=guide)
    start = original_st[0]
    end = original_st[-1]
    midpoint_s = (start[0] + end[0]) / 2.0
    span = end[0] - start[0]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for evidence in evidence_features:
        points = [
            point
            for point in station_offsets_for_coords(line_coords(evidence), guide=guide)
            if start[0] - 5.0 <= point[0] <= end[0] + 5.0
        ]
        if len(points) < 2:
            continue
        anchor_points = [point for point in points if point[0] <= start[0] + 30.0]
        if not anchor_points:
            continue
        offset_delta = min(abs(point[1] - start[1]) for point in anchor_points)
        anchor_distance = min(math.hypot(point[0] - start[0], 8.0 * (point[1] - start[1])) for point in anchor_points)
        if offset_delta > 1.25 or anchor_distance > 35.0:
            continue
        inside = [point for point in points if start[0] <= point[0] <= midpoint_s]
        if not inside:
            continue
        breakpoint = max(inside, key=lambda point: point[0])
        if breakpoint[0] - start[0] < 4.0:
            continue
        score = anchor_distance + 0.03 * abs(breakpoint[0] - (start[0] + 0.25 * span))
        candidates.append(
            (
                score,
                {
                    "point": breakpoint,
                    "points": points,
                    "evidence_id": evidence_identifier(evidence),
                    "anchor_distance": anchor_distance,
                    "offset_delta": offset_delta,
                },
            )
        )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def select_crossover_end_breakpoint(
    feature: dict[str, Any],
    evidence_features: list[dict[str, Any]],
    *,
    guide: LinearGuide,
) -> dict[str, Any] | None:
    original_st = station_offsets_for_coords(line_coords(feature), guide=guide)
    start = original_st[0]
    end = original_st[-1]
    midpoint_s = (start[0] + end[0]) / 2.0
    span = end[0] - start[0]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for evidence in evidence_features:
        points = [
            point
            for point in station_offsets_for_coords(line_coords(evidence), guide=guide)
            if start[0] - 5.0 <= point[0] <= end[0] + 5.0
        ]
        if len(points) < 2:
            continue
        anchor_points = [point for point in points if point[0] >= end[0] - 30.0]
        if not anchor_points:
            continue
        offset_delta = min(abs(point[1] - end[1]) for point in anchor_points)
        anchor_distance = min(math.hypot(point[0] - end[0], 8.0 * (point[1] - end[1])) for point in anchor_points)
        if offset_delta > 1.25 or anchor_distance > 35.0:
            continue
        inside = [point for point in points if midpoint_s <= point[0] <= end[0]]
        if not inside:
            continue
        if points[-1][0] >= end[0] - 1.0:
            breakpoint = min(inside, key=lambda point: point[0])
        else:
            breakpoint = max(inside, key=lambda point: point[0])
        if end[0] - breakpoint[0] < 4.0:
            continue
        score = anchor_distance + 0.03 * abs(breakpoint[0] - (end[0] - 0.25 * span))
        candidates.append(
            (
                score,
                {
                    "point": breakpoint,
                    "points": points,
                    "evidence_id": evidence_identifier(evidence),
                    "anchor_distance": anchor_distance,
                    "offset_delta": offset_delta,
                },
            )
        )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def trim_straight_band_endpoint_overlaps(
    band_features: list[dict[str, Any]],
    *,
    turnout_features: list[dict[str, Any]],
    margin_m: float = 1.0,
) -> list[dict[str, Any]]:
    mainline = next(
        (feature for feature in band_features if str((feature.get("properties") or {}).get("band_id", "")) == MAIN_BAND_ID),
        None,
    )
    if mainline is None:
        return band_features
    guide = LinearGuide(line_coords(mainline)[0], line_coords(mainline)[-1])
    trimmed: list[dict[str, Any]] = []
    for feature in band_features:
        props = feature.get("properties") or {}
        band_id = str(props.get("band_id", ""))
        if band_id == MAIN_BAND_ID:
            trimmed.append(feature)
            continue
        original_s0, original_s1 = station_range(feature)
        new_s0 = original_s0
        new_s1 = original_s1
        trim_notes: list[str] = []
        for turnout in turnout_features:
            turnout_s0, turnout_s1 = station_range(turnout)
            turnout_id = turnout_line_id(turnout)
            if turnout_s0 - margin_m <= original_s0 <= turnout_s1 + margin_m and turnout_s1 < original_s1:
                candidate_s0 = turnout_s1 + margin_m
                if candidate_s0 > new_s0:
                    new_s0 = candidate_s0
                    trim_notes.append(f"start_after_{turnout_id}")
            if turnout_s0 - margin_m <= original_s1 <= turnout_s1 + margin_m and original_s0 < turnout_s0:
                candidate_s1 = turnout_s0 - margin_m
                if candidate_s1 < new_s1:
                    new_s1 = candidate_s1
                    trim_notes.append(f"end_before_{turnout_id}")
        if new_s1 - new_s0 < 5.0:
            continue
        if abs(new_s0 - original_s0) <= 1e-6 and abs(new_s1 - original_s1) <= 1e-6:
            trimmed.append(feature)
            continue
        clipped = clip_feature_by_station(feature, guide=guide, station_min_m=new_s0, station_max_m=new_s1)
        clipped_props = dict(clipped.get("properties") or {})
        clipped_props.update(
            {
                "station_min_m": round(new_s0, 3),
                "station_max_m": round(new_s1, 3),
                "length_m": round(polyline_length(line_coords(clipped)), 3),
                "endpoint_trim_count": len(trim_notes),
                "endpoint_trim_rule": "straight_band_endpoint_inside_turnout_zone",
                "endpoint_trim_note": ";".join(trim_notes)[:96],
            }
        )
        clipped["properties"] = clipped_props
        trimmed.append(clipped)
    return trimmed


def clip_feature_by_station(
    feature: dict[str, Any],
    *,
    guide: LinearGuide,
    station_min_m: float,
    station_max_m: float,
) -> dict[str, Any]:
    points = station_offsets_for_coords(line_coords(feature), guide=guide)
    if len(points) < 2:
        return feature
    clipped_points = [(station_min_m, interpolated_offset(points, station_min_m))]
    clipped_points.extend((station, offset) for station, offset in points if station_min_m < station < station_max_m)
    clipped_points.append((station_max_m, interpolated_offset(points, station_max_m)))
    coords = [[round(x, 6), round(y, 6)] for x, y in (guide.point_at(station, offset) for station, offset in clipped_points)]
    return {
        "type": "Feature",
        "properties": dict(feature.get("properties") or {}),
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def smooth_turnout_connectors_with_evidence(
    features: list[dict[str, Any]],
    *,
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
) -> list[dict[str, Any]]:
    mainline = find_feature_by_line_id(features, "BAND_mainline_2_track_0")
    if mainline is None:
        return features
    guide = LinearGuide(line_coords(mainline)[0], line_coords(mainline)[-1])
    smoothed: list[dict[str, Any]] = []
    for feature in features:
        props = dict(feature.get("properties") or {})
        if str(props.get("network_role", "")) != "turnout_connector":
            smoothed.append(feature)
            continue
        coords = line_coords(feature)
        old_angle = max_local_angle_deg(coords)
        if old_angle < 1.0:
            smoothed.append(feature)
            continue
        old_support = measure_support(
            coords,
            evidence_segments=evidence_segments,
            threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
        )
        best: tuple[float, dict[str, Any], list[tuple[float, float]], int, int] | None = None
        for passes in (1, 2, 3):
            for window_size in (5, 7, 9, 13):
                candidate_coords = smooth_turnout_coords(coords, guide=guide, window_size=window_size, passes=passes)
                new_support = measure_support(
                    candidate_coords,
                    evidence_segments=evidence_segments,
                    threshold_m=support_threshold_m,
                    sample_step_m=sample_step_m,
                )
                if new_support["support_ratio"] < old_support["support_ratio"] - 0.08:
                    continue
                if new_support["mean_distance_m"] > old_support["mean_distance_m"] + 0.15:
                    continue
                new_angle = max_local_angle_deg(candidate_coords)
                if new_angle >= old_angle - 0.25:
                    continue
                score = new_angle + max(0.0, new_support["mean_distance_m"] - old_support["mean_distance_m"])
                if best is None or score < best[0]:
                    best = (score, new_support, candidate_coords, passes, window_size)
        if best is None:
            smoothed.append(feature)
            continue
        _, new_support, candidate_coords, passes, window_size = best
        new_angle = max_local_angle_deg(candidate_coords)
        props.update(
            {
                "length_m": round(polyline_length(candidate_coords), 3),
                "deeplab_support_ratio": new_support["support_ratio"],
                "deeplab_mean_distance_m": new_support["mean_distance_m"],
                "deeplab_max_unsupported_gap_m": new_support["max_unsupported_gap_m"],
                "deeplab_sample_count": new_support["sample_count"],
                "turnout_smooth_status": "accepted",
                "turnout_smooth_old_angle": round(old_angle, 3),
                "turnout_smooth_new_angle": round(new_angle, 3),
                "turnout_smooth_passes": passes,
                "turnout_smooth_window": window_size,
                "postprocess_policy": "evidence_constrained_turnout_polyline_smoothing",
            }
        )
        smoothed.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[round(x, 6), round(y, 6)] for x, y in candidate_coords],
                },
            }
        )
    return smoothed


def smooth_turnout_coords(
    coords: list[tuple[float, float]],
    *,
    guide: LinearGuide,
    window_size: int,
    passes: int,
) -> list[tuple[float, float]]:
    station_offsets = station_offsets_for_coords(coords, guide=guide)
    sampled = resample_station_offsets(station_offsets, step_m=0.75)
    if len(sampled) < 5:
        return coords
    smoothed = smooth_station_offsets(sampled, window_size=window_size, passes=passes)
    smoothed[0] = sampled[0]
    smoothed[-1] = sampled[-1]
    return [guide.point_at(station, offset) for station, offset in smoothed]


def max_local_angle_deg(coords: list[tuple[float, float]]) -> float:
    values: list[float] = []
    for left, center, right in zip(coords, coords[1:], coords[2:]):
        ax = center[0] - left[0]
        ay = center[1] - left[1]
        bx = right[0] - center[0]
        by = right[1] - center[1]
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la <= 1e-9 or lb <= 1e-9:
            continue
        cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        values.append(math.degrees(math.acos(cosine)))
    return max(values, default=0.0)


def build_turnout_boundary_evidence_bridges(
    features: list[dict[str, Any]],
    *,
    evidence_features: list[dict[str, Any]],
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
    max_gap_m: float = 45.0,
    max_offset_delta_m: float = 0.85,
    min_support_ratio: float = 0.65,
) -> list[dict[str, Any]]:
    mainline = find_feature_by_line_id(features, "BAND_mainline_2_track_0")
    if mainline is None:
        return []
    guide = LinearGuide(line_coords(mainline)[0], line_coords(mainline)[-1])
    bands = [
        feature
        for feature in features
        if str((feature.get("properties") or {}).get("network_role", "")) == "parallel_straight_track"
    ]
    turnouts = [
        feature
        for feature in features
        if str((feature.get("properties") or {}).get("network_role", "")) == "turnout_connector"
    ]
    candidates: list[tuple[float, dict[str, Any], int, dict[str, Any], int, dict[str, Any], list[tuple[float, float]]]] = []
    for band in bands:
        band_id = str((band.get("properties") or {}).get("line_id", "band"))
        for band_endpoint_index, band_endpoint in enumerate(endpoint_records(band, guide=guide)):
            for turnout in turnouts:
                turnout_id = str((turnout.get("properties") or {}).get("line_id", turnout_line_id(turnout)))
                for turnout_endpoint_index, turnout_endpoint in enumerate(endpoint_records(turnout, guide=guide)):
                    station_gap = abs(turnout_endpoint["station"] - band_endpoint["station"])
                    offset_delta = abs(turnout_endpoint["offset"] - band_endpoint["offset"])
                    if station_gap <= 1e-6 or station_gap > max_gap_m or offset_delta > max_offset_delta_m:
                        continue
                    bridge_coords, evidence_id = evidence_bridge_coords(
                        band_endpoint,
                        turnout_endpoint,
                        guide=guide,
                        evidence_features=evidence_features,
                    )
                    if bridge_coords is None:
                        continue
                    support = measure_support(
                        bridge_coords,
                        evidence_segments=evidence_segments,
                        threshold_m=support_threshold_m,
                        sample_step_m=sample_step_m,
                    )
                    if support["support_ratio"] < min_support_ratio:
                        continue
                    score = station_gap + 4.0 * offset_delta + max(0.0, support["mean_distance_m"])
                    props = {
                        "line_id": f"BRIDGE_BOUNDARY_{short_id(band_id)}_{short_id(turnout_id)}",
                        "network_role": "turnout_boundary_bridge",
                        "source_layer": "semseg_turnout_boundary_bridge",
                        "band_id": (band.get("properties") or {}).get("band_id", ""),
                        "branch_id": (turnout.get("properties") or {}).get("branch_id", ""),
                        "risk_flag": "evidence_promoted_turnout_boundary_bridge",
                        "qa_status": "bridge_evidence_supported",
                        "length_m": round(polyline_length(bridge_coords), 3),
                        "station_min_m": round(min(band_endpoint["station"], turnout_endpoint["station"]), 3),
                        "station_max_m": round(max(band_endpoint["station"], turnout_endpoint["station"]), 3),
                        "gap_m": round(station_gap, 3),
                        "offset_delta_m": round(offset_delta, 3),
                        "bridge_evidence": evidence_id,
                        "deeplab_support_ratio": support["support_ratio"],
                        "deeplab_mean_distance_m": support["mean_distance_m"],
                        "deeplab_max_unsupported_gap_m": support["max_unsupported_gap_m"],
                        "deeplab_sample_count": support["sample_count"],
                        "support_threshold_m": support_threshold_m,
                        "postprocess_policy": "evidence_supported_trimmed_band_to_turnout_boundary_bridge",
                        "review_note": "trimmed straight-track endpoint connected to turnout endpoint using DeepLab/gauge evidence",
                    }
                    candidates.append(
                        (
                            score,
                            {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": bridge_coords}},
                            band_endpoint_index,
                            band,
                            turnout_endpoint_index,
                            turnout,
                            bridge_coords,
                        )
                    )
    bridges: list[dict[str, Any]] = []
    used_band_endpoints: set[tuple[str, int]] = set()
    used_turnout_endpoints: set[tuple[str, int]] = set()
    used_line_ids: set[str] = set()
    for _score, bridge, band_endpoint_index, band, turnout_endpoint_index, turnout, _coords in sorted(candidates, key=lambda item: item[0]):
        band_key = (str((band.get("properties") or {}).get("line_id", "")), band_endpoint_index)
        turnout_key = (str((turnout.get("properties") or {}).get("line_id", "")), turnout_endpoint_index)
        line_id = str((bridge.get("properties") or {}).get("line_id", ""))
        if band_key in used_band_endpoints or turnout_key in used_turnout_endpoints or line_id in used_line_ids:
            continue
        used_band_endpoints.add(band_key)
        used_turnout_endpoints.add(turnout_key)
        used_line_ids.add(line_id)
        bridges.append(bridge)
    return bridges


def endpoint_records(feature: dict[str, Any], *, guide: LinearGuide) -> list[dict[str, Any]]:
    coords = line_coords(feature)
    if len(coords) < 2:
        return []
    records: list[dict[str, Any]] = []
    for index, point in ((0, coords[0]), (len(coords) - 1, coords[-1])):
        station, offset = guide.station_offset(point)
        records.append({"index": index, "point": point, "station": station, "offset": offset})
    return records


def evidence_bridge_coords(
    endpoint_a: dict[str, Any],
    endpoint_b: dict[str, Any],
    *,
    guide: LinearGuide,
    evidence_features: list[dict[str, Any]],
) -> tuple[list[tuple[float, float]] | None, str]:
    start = endpoint_a
    end = endpoint_b
    start_s = float(start["station"])
    end_s = float(end["station"])
    lo_s = min(start_s, end_s)
    hi_s = max(start_s, end_s)
    best: tuple[float, str, list[tuple[float, float]]] | None = None
    for evidence in evidence_features:
        evidence_id = evidence_identifier(evidence)
        points = station_offsets_for_coords(line_coords(evidence), guide=guide)
        if len(points) < 2:
            continue
        if points[0][0] > lo_s + 1.5 or points[-1][0] < hi_s - 1.5:
            continue
        start_delta = abs(interpolated_offset(points, start_s) - float(start["offset"]))
        end_delta = abs(interpolated_offset(points, end_s) - float(end["offset"]))
        if start_delta > 1.0 or end_delta > 1.0:
            continue
        score = start_delta + end_delta + evidence_role_penalty(evidence)
        if best is None or score < best[0]:
            best = (score, evidence_id, points)
    if best is None:
        if abs(end_s - start_s) <= 2.0:
            return [start["point"], end["point"]], "short_endpoint_gap"
        return None, ""
    _score, evidence_id, evidence_points = best
    span = hi_s - lo_s
    count = max(3, int(math.ceil(span / 0.75)) + 1)
    stations = [lo_s + span * index / (count - 1) for index in range(count)]
    station_offsets = [(station, interpolated_offset(evidence_points, station)) for station in stations]
    if start_s > end_s:
        station_offsets.reverse()
    station_offsets[0] = (start_s, float(start["offset"]))
    station_offsets[-1] = (end_s, float(end["offset"]))
    return [guide.point_at(station, offset) for station, offset in station_offsets], evidence_id


def evidence_identifier(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    return str(props.get("line_id", props.get("evidence_id", props.get("seq_id", "evidence"))))


def evidence_role_penalty(feature: dict[str, Any]) -> float:
    props = feature.get("properties") or {}
    role = str(props.get("role", props.get("evidence_role", "")))
    source = str(props.get("source_layer", props.get("network_source", "")))
    if role == "support" or "support_chain" in source:
        return 0.05
    if "gauge_pair" in source or str(props.get("seq_id", "")).endswith("GP01") or str(props.get("seq_id", "")).endswith("GP02"):
        return 0.0
    if role == "mainline":
        return 0.35
    return 0.2


def snap_connector_endpoints(features: list[dict[str, Any]], *, max_distance_m: float) -> list[dict[str, Any]]:
    if max_distance_m <= 0:
        return features
    target_roles = {"main_through_track", "parallel_straight_track", "promoted_straight_track"}
    snap_roles = {"turnout_connector", "turnout_tail_bridge", "straight_gap_bridge"}
    targets = [feature for feature in features if str((feature.get("properties") or {}).get("network_role", "")) in target_roles]
    snapped_features: list[dict[str, Any]] = []
    for feature in features:
        props = dict(feature.get("properties") or {})
        role = str(props.get("network_role", ""))
        coords = line_coords(feature)
        if role not in snap_roles or len(coords) < 2:
            snapped_features.append(feature)
            continue
        snap_distances: list[float] = []
        updated_coords = list(coords)
        for endpoint_index in (0, len(updated_coords) - 1):
            best = nearest_projection_on_features(updated_coords[endpoint_index], targets)
            if best is None:
                continue
            snap_distance, projected, _target_id = best
            if 1e-6 < snap_distance <= max_distance_m:
                updated_coords[endpoint_index] = projected
                snap_distances.append(snap_distance)
        if not snap_distances:
            snapped_features.append(feature)
            continue
        props.update(
            {
                "endpoint_snap_count": int(len(snap_distances)),
                "endpoint_snap_max_m": round(max(snap_distances), 4),
                "endpoint_snap_tolerance_m": round(float(max_distance_m), 3),
            }
        )
        snapped_features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[round(x, 6), round(y, 6)] for x, y in updated_coords],
                },
            }
        )
    return snapped_features


def nearest_projection_on_features(
    point: tuple[float, float],
    features: list[dict[str, Any]],
) -> tuple[float, tuple[float, float], str] | None:
    best: tuple[float, tuple[float, float], str] | None = None
    for feature in features:
        target_id = str((feature.get("properties") or {}).get("line_id", ""))
        coords = line_coords(feature)
        for start, end in zip(coords, coords[1:]):
            candidate = point_segment_projection(point, start, end)
            if best is None or candidate[0] < best[0]:
                best = (candidate[0], candidate[1], target_id)
    return best


def build_promoted_diagnostic_tracks(
    diagnostic_band_features: list[dict[str, Any]],
    *,
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
    min_support_ratio: float,
    min_length_m: float,
) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    for feature in diagnostic_band_features:
        coords = line_coords(feature)
        length_m = polyline_length(coords)
        if length_m < min_length_m:
            continue
        support = measure_support(
            coords,
            evidence_segments=evidence_segments,
            threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
        )
        if support["support_ratio"] < min_support_ratio:
            continue
        props = dict(feature.get("properties") or {})
        band_id = str(props.get("band_id", "diagnostic_track"))
        props.update(
            {
                "line_id": f"PROMOTED_{band_id}_{safe_interval_id(feature)}",
                "network_role": "promoted_straight_track",
                "source_layer": "diagnostic_track_band_promoted",
                "length_m": round(length_m, 3),
                "deeplab_support_ratio": support["support_ratio"],
                "deeplab_mean_distance_m": support["mean_distance_m"],
                "deeplab_max_unsupported_gap_m": support["max_unsupported_gap_m"],
                "deeplab_sample_count": support["sample_count"],
                "support_threshold_m": support_threshold_m,
                "risk_flag": "evidence_promoted_diagnostic_track",
                "qa_status": "promoted_needs_review",
                "postprocess_policy": "diagnostic_track_band_evidence_promotion",
                "review_note": "diagnostic track-band candidate promoted because DeepLab/gauge evidence strongly supports it",
            }
        )
        promoted.append({"type": "Feature", "properties": props, "geometry": feature["geometry"]})
    return promoted


def build_turnout_tail_bridges(
    promoted_tracks: list[dict[str, Any]],
    *,
    turnout_features: list[dict[str, Any]],
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
    max_gap_m: float,
    min_support_ratio: float,
) -> list[dict[str, Any]]:
    bridges: list[dict[str, Any]] = []
    used_pairs: set[tuple[str, str]] = set()
    for track in promoted_tracks:
        best = nearest_turnout_endpoint_bridge(track, turnout_features)
        if best is None:
            continue
        turnout, track_endpoint_index, turnout_endpoint_index, gap_m = best
        if gap_m > max_gap_m:
            continue
        bridge_coords = curved_endpoint_bridge_coords(
            line_coords(track),
            track_endpoint_index,
            line_coords(turnout),
            turnout_endpoint_index,
        )
        support = measure_support(
            bridge_coords,
            evidence_segments=evidence_segments,
            threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
        )
        if support["support_ratio"] < min_support_ratio:
            continue
        track_id = str((track.get("properties") or {}).get("line_id", "track"))
        turnout_id = turnout_line_id(turnout)
        pair_key = tuple(sorted([track_id, turnout_id]))
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)
        track_props = track.get("properties") or {}
        turnout_props = turnout.get("properties") or {}
        props = {
            "line_id": f"BRIDGE_{short_id(track_id)}_{short_id(turnout_id)}",
            "network_role": "turnout_tail_bridge",
            "source_layer": "diagnostic_track_to_turnout_bridge",
            "band_id": track_props.get("band_id", ""),
            "branch_id": turnout_props.get("branch_id", turnout_props.get("anchor_id", "")),
            "risk_flag": "evidence_promoted_turnout_tail_bridge",
            "qa_status": "bridge_evidence_supported",
            "length_m": round(polyline_length(bridge_coords), 3),
            "station_min_m": round(min(station_range(track)[track_endpoint_index], station_range(turnout)[turnout_endpoint_index]), 3),
            "station_max_m": round(max(station_range(track)[track_endpoint_index], station_range(turnout)[turnout_endpoint_index]), 3),
            "gap_m": round(gap_m, 3),
            "deeplab_support_ratio": support["support_ratio"],
            "deeplab_mean_distance_m": support["mean_distance_m"],
            "deeplab_max_unsupported_gap_m": support["max_unsupported_gap_m"],
            "deeplab_sample_count": support["sample_count"],
            "support_threshold_m": support_threshold_m,
            "postprocess_policy": "promoted_diagnostic_track_to_turnout_tail_bridge",
            "review_note": "evidence-supported curved bridge between promoted diagnostic track and nearby turnout tail",
        }
        bridges.append({"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": bridge_coords}})
    return bridges


def curved_endpoint_bridge_coords(
    track_coords: list[tuple[float, float]],
    track_endpoint_index: int,
    turnout_coords: list[tuple[float, float]],
    turnout_endpoint_index: int,
) -> list[list[float]]:
    start = track_coords[track_endpoint_index]
    end = turnout_coords[turnout_endpoint_index]
    gap_m = distance(start, end)
    if gap_m <= 1e-6:
        return [[round(start[0], 6), round(start[1], 6)], [round(end[0], 6), round(end[1], 6)]]
    start_tangent = endpoint_extension_tangent(track_coords, track_endpoint_index)
    end_tangent = endpoint_entry_tangent(turnout_coords, turnout_endpoint_index)
    chord = normalize((end[0] - start[0], end[1] - start[1]))
    if dot(start_tangent, chord) <= 0.0 or dot(end_tangent, chord) <= 0.0:
        start_tangent = chord
        end_tangent = chord
    handle_m = min(gap_m / 3.0, 18.0)
    p1 = (start[0] + start_tangent[0] * handle_m, start[1] + start_tangent[1] * handle_m)
    p2 = (end[0] - end_tangent[0] * handle_m, end[1] - end_tangent[1] * handle_m)
    count = max(8, int(math.ceil(gap_m / 2.0)) + 1)
    coords: list[list[float]] = []
    for index in range(count):
        u = index / (count - 1)
        v = 1.0 - u
        x = v * v * v * start[0] + 3.0 * v * v * u * p1[0] + 3.0 * v * u * u * p2[0] + u * u * u * end[0]
        y = v * v * v * start[1] + 3.0 * v * v * u * p1[1] + 3.0 * v * u * u * p2[1] + u * u * u * end[1]
        coords.append([round(x, 6), round(y, 6)])
    return coords


def endpoint_extension_tangent(coords: list[tuple[float, float]], endpoint_index: int) -> tuple[float, float]:
    if len(coords) < 2:
        return (1.0, 0.0)
    if endpoint_index == 0:
        return normalize((coords[0][0] - coords[1][0], coords[0][1] - coords[1][1]))
    return normalize((coords[-1][0] - coords[-2][0], coords[-1][1] - coords[-2][1]))


def endpoint_entry_tangent(coords: list[tuple[float, float]], endpoint_index: int) -> tuple[float, float]:
    if len(coords) < 2:
        return (1.0, 0.0)
    if endpoint_index == 0:
        return normalize((coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]))
    return normalize((coords[-2][0] - coords[-1][0], coords[-2][1] - coords[-1][1]))


def normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm <= 1e-9:
        return (1.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def nearest_turnout_endpoint_bridge(
    track: dict[str, Any],
    turnout_features: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, int, float] | None:
    track_coords = line_coords(track)
    if len(track_coords) < 2:
        return None
    track_endpoints = [(0, track_coords[0]), (-1, track_coords[-1])]
    best: tuple[dict[str, Any], int, int, float] | None = None
    for turnout in turnout_features:
        turnout_coords = line_coords(turnout)
        if len(turnout_coords) < 2:
            continue
        turnout_endpoints = [(0, turnout_coords[0]), (-1, turnout_coords[-1])]
        for track_index, track_point in track_endpoints:
            for turnout_index, turnout_point in turnout_endpoints:
                gap_m = distance(track_point, turnout_point)
                if best is None or gap_m < best[3]:
                    best = (turnout, track_index, turnout_index, gap_m)
    return best


def build_straight_gap_bridges(
    band_features: list[dict[str, Any]],
    *,
    turnout_features: list[dict[str, Any]],
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
    min_gap_m: float,
    max_gap_m: float,
    evidence_support_threshold: float,
    weak_gap_max_m: float,
    turnout_clearance_m: float,
) -> list[dict[str, Any]]:
    bridges: list[dict[str, Any]] = []
    by_band: dict[str, list[dict[str, Any]]] = {}
    for feature in band_features:
        band_id = str((feature.get("properties") or {}).get("band_id", ""))
        if not band_id or band_id == MAIN_BAND_ID:
            continue
        by_band.setdefault(band_id, []).append(feature)

    for band_id, features in by_band.items():
        ordered = sorted(features, key=lambda item: station_range(item)[0])
        for index, (left, right) in enumerate(zip(ordered, ordered[1:]), start=1):
            left_s0, left_s1 = station_range(left)
            right_s0, right_s1 = station_range(right)
            gap_m = right_s0 - left_s1
            if gap_m < min_gap_m or gap_m > max_gap_m:
                continue
            if gap_overlaps_turnout(turnout_features, left_s1, right_s0, clearance_m=turnout_clearance_m):
                continue
            left_coords = line_coords(left)
            right_coords = line_coords(right)
            if len(left_coords) < 2 or len(right_coords) < 2:
                continue
            bridge_coords = [left_coords[-1], right_coords[0]]
            support = measure_support(
                bridge_coords,
                evidence_segments=evidence_segments,
                threshold_m=support_threshold_m,
                sample_step_m=sample_step_m,
            )
            if support["support_ratio"] >= evidence_support_threshold:
                bridge_kind = "evidence_promoted_bridge"
            elif gap_m <= weak_gap_max_m:
                bridge_kind = "occlusion_bridge"
            else:
                continue
            props = {
                "line_id": f"BRIDGE_{band_id}_{safe_interval_id(left)}_{safe_interval_id(right)}",
                "network_role": "straight_gap_bridge",
                "source_layer": "topology_gap_bridge",
                "band_id": band_id,
                "bridge_kind": bridge_kind,
                "risk_flag": bridge_kind,
                "qa_status": "bridge_needs_review" if bridge_kind == "occlusion_bridge" else "bridge_evidence_supported",
                "length_m": round(polyline_length(bridge_coords), 3),
                "station_min_m": round(left_s1, 3),
                "station_max_m": round(right_s0, 3),
                "gap_m": round(gap_m, 3),
                "deeplab_support_ratio": support["support_ratio"],
                "deeplab_mean_distance_m": support["mean_distance_m"],
                "deeplab_max_unsupported_gap_m": support["max_unsupported_gap_m"],
                "deeplab_sample_count": support["sample_count"],
                "support_threshold_m": support_threshold_m,
                "weak_gap_bridge_max_gap_m": weak_gap_max_m,
                "postprocess_policy": "same_band_internal_gap_bridge",
                "review_note": (
                    "same-band internal gap bridged by DeepLab evidence"
                    if bridge_kind == "evidence_promoted_bridge"
                    else "same-band internal gap bridged by topology; likely occlusion or missing segmentation"
                ),
            }
            bridges.append({"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": bridge_coords}})
    return bridges


def filter_promoted_diagnostics_used_by_turnouts(
    promoted_features: list[dict[str, Any]],
    *,
    turnout_features: list[dict[str, Any]],
    min_overlap_ratio: float = 0.45,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for feature in promoted_features:
        props = feature.get("properties") or {}
        band_id = str(props.get("band_id", ""))
        station_min, station_max = station_range(feature)
        length = max(0.0, station_max - station_min)
        if not band_id or length <= 0:
            filtered.append(feature)
            continue
        used_by_turnout = False
        for turnout in turnout_features:
            turnout_props = turnout.get("properties") or {}
            if band_id not in {
                str(turnout_props.get("start_band", "")),
                str(turnout_props.get("end_band", "")),
            }:
                continue
            turnout_min, turnout_max = station_range(turnout)
            overlap = max(0.0, min(station_max, turnout_max) - max(station_min, turnout_min))
            if overlap / length >= min_overlap_ratio:
                used_by_turnout = True
                break
        if not used_by_turnout:
            filtered.append(feature)
    return filtered


def gap_overlaps_turnout(
    turnout_features: list[dict[str, Any]],
    gap_start_m: float,
    gap_end_m: float,
    *,
    clearance_m: float,
) -> bool:
    query_start = gap_start_m - clearance_m
    query_end = gap_end_m + clearance_m
    for feature in turnout_features:
        s0, s1 = station_range(feature)
        if s1 >= query_start and s0 <= query_end:
            return True
    return False


def rebuild_ta08_curved_branch(
    features: list[dict[str, Any]],
    *,
    evidence_features: list[dict[str, Any]],
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
) -> list[dict[str, Any]]:
    mainline = find_feature_by_line_id(features, "BAND_mainline_2_track_0")
    promoted = find_feature_by_line_id(features, "PROMOTED_possible_outer_plus_10m_0")
    turnout = find_feature_by_line_id(features, "TURNOUT_TA08")
    gp02 = find_evidence_by_seq_id(evidence_features, "TA08_GP02")
    if mainline is None or promoted is None or turnout is None or gp02 is None:
        return features

    guide = LinearGuide(line_coords(mainline)[0], line_coords(mainline)[-1])
    promoted_st = station_offsets_for_coords(line_coords(promoted), guide=guide)
    gp02_st = station_offsets_for_coords(line_coords(gp02), guide=guide)
    turnout_st = station_offsets_for_coords(line_coords(turnout), guide=guide)
    if len(promoted_st) < 2 or len(gp02_st) < 4 or len(turnout_st) < 2:
        return features

    promoted_curve = resample_station_offsets(promoted_st, step_m=1.5)
    straight_middle = paired_rail_linear_segment(gp02_st, step_m=0.75)
    straight_middle_slope, straight_middle_intercept = fit_station_offset_line(gp02_st)
    straight_middle_rms = linear_fit_rms(gp02_st, straight_middle_slope, straight_middle_intercept)
    curve_start = straight_middle[0]
    curve_end = straight_middle[-1]
    end = max(turnout_st, key=lambda item: item[0])
    if not (promoted_curve[-1][0] < curve_start[0] < curve_end[0] < end[0]):
        return features

    outer_curve_source = ""
    outer_curve_mode = "hermite_endpoint_transition"
    constrained_outer_curve = build_evidence_constrained_outer_curve(
        promoted_curve,
        curve_start,
        evidence_features=evidence_features,
        guide=guide,
        middle_start_slope=slope_at_start(straight_middle),
        step_m=0.75,
    )
    if constrained_outer_curve is not None:
        promoted_curve, transition_in, outer_curve_source = constrained_outer_curve
        outer_curve_mode = "deeplab_support_chain_constrained"
    else:
        start_slope = slope_at_end(promoted_curve)
        middle_slope = slope_at_start(straight_middle)
        transition_in = cubic_hermite_station_offsets(promoted_curve[-1], curve_start, start_slope, middle_slope, step_m=0.75)

    start = promoted_curve[-1]
    middle_slope = slope_at_start(straight_middle)
    tangent_slope = mainline_tangent_slope(evidence_features, guide=guide, station=end[0])

    transition_out = cubic_hermite_station_offsets(curve_end, end, middle_slope, tangent_slope, step_m=0.75)
    merged_st = merge_station_offset_parts([promoted_curve, transition_in, straight_middle, transition_out])
    coords = [[round(x, 6), round(y, 6)] for x, y in (guide.point_at(station, offset) for station, offset in merged_st)]
    support = measure_support(
        [(float(x), float(y)) for x, y in coords],
        evidence_segments=evidence_segments,
        threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    stations = [station for station, _ in merged_st]
    offsets = [offset for _, offset in merged_st]
    curvature = station_offset_curve_summary(merged_st)
    props = dict(turnout.get("properties") or {})
    props.update(
        {
            "line_id": "TURNOUT_TA08",
            "network_role": "turnout_connector",
            "source_layer": "paired_rail_piecewise_rebuild",
            "source_type": "paired_rail_piecewise_rebuild",
            "geom_kind": "continuous_piecewise_paired_rail_centerline",
            "branch_id": "TA08",
            "band_id": "possible_outer_plus_10m",
            "risk_flag": "review_priority_low_support_turnout",
            "qa_status": "self_review_needs_visual_check",
            "length_m": round(polyline_length([(float(x), float(y)) for x, y in coords]), 3),
            "station_min_m": round(min(stations), 3),
            "station_max_m": round(max(stations), 3),
            "offset_start_m": round(offsets[0], 3),
            "offset_end_m": round(offsets[-1], 3),
            "offset_min_m": round(min(offsets), 3),
            "offset_max_m": round(max(offsets), 3),
            "deeplab_support_ratio": support["support_ratio"],
            "deeplab_mean_distance_m": support["mean_distance_m"],
            "deeplab_max_unsupported_gap_m": support["max_unsupported_gap_m"],
            "deeplab_sample_count": support["sample_count"],
            "support_threshold_m": support_threshold_m,
            "curve_model": "curve_straight_curve_parallel_from_switch",
            "station_order_model": "parallel_curve_straight_curve_to_switch",
            "outer_curve_mode": outer_curve_mode,
            "outer_curve_source": outer_curve_source,
            "parallel_s0_m": round(promoted_curve[0][0], 3),
            "parallel_s1_m": round(promoted_curve[-1][0], 3),
            "outer_curve_s0_m": round(start[0], 3),
            "outer_curve_s1_m": round(curve_start[0], 3),
            "straight_middle_s0_m": round(straight_middle[0][0], 3),
            "straight_middle_s1_m": round(straight_middle[-1][0], 3),
            "straight_middle_slope": round(middle_slope, 5),
            "straight_middle_fit_rms_m": round(straight_middle_rms, 4),
            "switch_curve_s0_m": round(curve_end[0], 3),
            "switch_curve_s1_m": round(end[0], 3),
            "curve_start_s_m": round(curvature["curve_start_s_m"], 3),
            "curve_end_s_m": round(curvature["curve_end_s_m"], 3),
            "max_abs_slope": round(curvature["max_abs_slope"], 4),
            "max_abs_curvature": round(curvature["max_abs_curvature"], 5),
            "postprocess_policy": "continuous_ta08_piecewise_paired_rail_rebuild",
            "review_note": "TA08 rebuilt from switch as curve-straight-curve-parallel using paired rail evidence.",
            "self_note": "continuous TA08 curve-straight-curve-parallel rebuild; inspect against DOM before acceptance",
        }
    )
    rebuilt = {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}
    remove_ids = {
        "PROMOTED_possible_outer_plus_10m_0",
        "BRIDGE_PROMOTED_possible_outer_plus_10m_0_TURNOUT_TA08",
        "TURNOUT_TA08",
    }
    kept = [feature for feature in features if str((feature.get("properties") or {}).get("line_id", "")) not in remove_ids]
    kept.append(rebuilt)
    return kept


def find_feature_by_line_id(features: list[dict[str, Any]], line_id: str) -> dict[str, Any] | None:
    for feature in features:
        if str((feature.get("properties") or {}).get("line_id", "")) == line_id:
            return feature
    return None


def find_evidence_by_seq_id(features: list[dict[str, Any]], seq_id: str) -> dict[str, Any] | None:
    for feature in features:
        if str((feature.get("properties") or {}).get("seq_id", "")) == seq_id:
            return feature
    return None


def build_evidence_constrained_outer_curve(
    promoted_curve: list[tuple[float, float]],
    middle_start: tuple[float, float],
    *,
    evidence_features: list[dict[str, Any]],
    guide: LinearGuide,
    middle_start_slope: float,
    step_m: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], str] | None:
    evidence = find_outer_curve_support_feature(
        evidence_features,
        guide=guide,
        promoted_curve=promoted_curve,
        middle_start=middle_start,
    )
    if evidence is None:
        return None
    evidence_id, evidence_st = evidence
    curve_start_s = outer_curve_start_station(evidence_st, promoted_curve=promoted_curve, middle_start_s=middle_start[0])
    if not (promoted_curve[0][0] < curve_start_s < middle_start[0] - 3.0):
        return None

    promoted_end = (curve_start_s, interpolated_offset(promoted_curve, curve_start_s))
    promoted_part = resample_station_offsets(
        [point for point in promoted_curve if point[0] <= curve_start_s] + [promoted_end],
        step_m=1.5,
    )
    span = middle_start[0] - curve_start_s
    blend_span = min(10.0, max(4.0, span * 0.25))
    blend_end_s = min(curve_start_s + blend_span, middle_start[0] - 2.0)
    blend_end = (blend_end_s, interpolated_offset(evidence_st, blend_end_s))
    support_slope = local_evidence_slope(evidence_st, station=blend_end_s, radius_m=3.0)
    blend = cubic_hermite_station_offsets(
        promoted_end,
        blend_end,
        slope_at_end(promoted_part),
        support_slope,
        step_m=step_m,
    )
    end_blend_span = min(6.0, max(3.0, span * 0.15))
    tail_end_s = max(blend_end_s + 2.0, middle_start[0] - end_blend_span)
    tail_span = tail_end_s - blend_end_s
    tail_count = max(3, int(math.ceil(tail_span / max(step_m, 0.1))) + 1)
    tail_stations = [blend_end_s + tail_span * index / (tail_count - 1) for index in range(tail_count)]
    tail = [(station, interpolated_offset(evidence_st, station)) for station in tail_stations]
    tail[0] = blend_end
    tail_end = (tail_end_s, interpolated_offset(evidence_st, tail_end_s))
    tail[-1] = tail_end
    tail = smooth_station_offsets(tail, window_size=5, passes=1)
    tail[0] = blend_end
    tail[-1] = tail_end
    end_blend = cubic_hermite_station_offsets(
        tail_end,
        middle_start,
        local_evidence_slope(evidence_st, station=tail_end_s, radius_m=3.0),
        middle_start_slope,
        step_m=step_m,
    )
    curve = merge_station_offset_parts([blend, tail, end_blend])
    return promoted_part, curve, evidence_id


def find_outer_curve_support_feature(
    evidence_features: list[dict[str, Any]],
    *,
    guide: LinearGuide,
    promoted_curve: list[tuple[float, float]],
    middle_start: tuple[float, float],
) -> tuple[str, list[tuple[float, float]]] | None:
    best: tuple[float, str, list[tuple[float, float]]] | None = None
    promoted_start_s = promoted_curve[0][0]
    promoted_end_s = promoted_curve[-1][0]
    middle_s, middle_t = middle_start
    for feature in evidence_features:
        props = feature.get("properties") or {}
        line_id = str(props.get("line_id", ""))
        role = str(props.get("role", ""))
        source = str(props.get("network_source", ""))
        if role != "support" and not line_id.startswith("DLV1_SUPPORT") and "support_chain" not in source:
            continue
        points = station_offsets_for_coords(line_coords(feature), guide=guide)
        if len(points) < 5:
            continue
        if points[-1][0] < promoted_end_s or points[0][0] > middle_s:
            continue
        overlap = min(points[-1][0], middle_s) - max(points[0][0], promoted_start_s)
        if overlap < 12.0:
            continue
        middle_offset = interpolated_offset(points, middle_s)
        promoted_offset = interpolated_offset(points, promoted_end_s)
        score = abs(middle_offset - middle_t) + 0.5 * abs(promoted_offset - promoted_curve[-1][1])
        if abs(middle_offset - middle_t) > 0.85:
            continue
        if best is None or score < best[0]:
            best = (score, line_id, points)
    if best is None:
        return None
    return best[1], best[2]


def outer_curve_start_station(
    evidence_st: list[tuple[float, float]],
    *,
    promoted_curve: list[tuple[float, float]],
    middle_start_s: float,
) -> float:
    search_start = max(evidence_st[0][0], promoted_curve[0][0])
    search_end = min(evidence_st[-1][0], middle_start_s)
    samples = [
        point
        for point in resample_station_offsets(evidence_st, step_m=0.75)
        if search_start <= point[0] <= search_end
    ]
    if len(samples) < 5:
        return promoted_curve[-1][0]
    max_index = max(range(len(samples)), key=lambda index: samples[index][1])
    max_offset = samples[max_index][1]
    for index in range(max_index + 1, len(samples)):
        if max_offset - samples[index][1] >= 0.12:
            return samples[max(max_index, index - 2)][0]
    return promoted_curve[-1][0]


def local_evidence_slope(points: list[tuple[float, float]], *, station: float, radius_m: float) -> float:
    left_s = max(points[0][0], station - radius_m)
    right_s = min(points[-1][0], station + radius_m)
    if right_s - left_s <= 1e-6:
        return 0.0
    left = (left_s, interpolated_offset(points, left_s))
    right = (right_s, interpolated_offset(points, right_s))
    return slope_between([left, right])


class LinearGuide:
    def __init__(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        self.start = start
        self.end = end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        self.length = math.hypot(dx, dy)
        if self.length <= 1e-9:
            raise ValueError("Guide endpoints must be different.")
        self.ux = dx / self.length
        self.uy = dy / self.length

    def station_offset(self, point: tuple[float, float]) -> tuple[float, float]:
        vx = point[0] - self.start[0]
        vy = point[1] - self.start[1]
        station = vx * self.ux + vy * self.uy
        offset = vx * (-self.uy) + vy * self.ux
        return station, offset

    def point_at(self, station: float, offset: float) -> tuple[float, float]:
        return (
            self.start[0] + self.ux * station - self.uy * offset,
            self.start[1] + self.uy * station + self.ux * offset,
        )


def station_offsets_for_coords(coords: list[tuple[float, float]], *, guide: LinearGuide) -> list[tuple[float, float]]:
    return sorted((guide.station_offset(coord) for coord in coords), key=lambda item: item[0])


def resample_station_offsets(points: list[tuple[float, float]], *, step_m: float) -> list[tuple[float, float]]:
    points = dedupe_station_offsets(points)
    if len(points) < 2:
        return points
    start_s, end_s = points[0][0], points[-1][0]
    span = end_s - start_s
    if span <= 0:
        return points
    count = max(2, int(math.ceil(span / max(step_m, 0.1))) + 1)
    stations = [start_s + span * index / (count - 1) for index in range(count)]
    return [(station, interpolated_offset(points, station)) for station in stations]


def smooth_station_offsets(points: list[tuple[float, float]], *, window_size: int, passes: int) -> list[tuple[float, float]]:
    if len(points) < 5:
        return points
    stations = [station for station, _ in points]
    offsets = [offset for _, offset in points]
    half = max(1, int(window_size) // 2)
    locked = {0, 1, len(offsets) - 2, len(offsets) - 1}
    for _ in range(max(0, int(passes))):
        previous = offsets[:]
        for index in range(len(offsets)):
            if index in locked:
                continue
            lo = max(0, index - half)
            hi = min(len(offsets), index + half + 1)
            offsets[index] = sum(previous[lo:hi]) / (hi - lo)
    return list(zip(stations, offsets))


def paired_rail_curvature_curve(points: list[tuple[float, float]], *, step_m: float) -> list[tuple[float, float]]:
    sampled = resample_station_offsets(points, step_m=step_m)
    if len(sampled) < 4:
        return sampled
    slope_window = max(4, min(15, len(sampled) // 3))
    start_slope = slope_at_start(sampled, sample_count=slope_window)
    end_slope = slope_at_end(sampled, sample_count=slope_window)
    return cubic_hermite_station_offsets(sampled[0], sampled[-1], start_slope, end_slope, step_m=step_m)


def paired_rail_linear_segment(points: list[tuple[float, float]], *, step_m: float) -> list[tuple[float, float]]:
    sampled = resample_station_offsets(points, step_m=step_m)
    if len(sampled) < 2:
        return sampled
    slope, intercept = fit_station_offset_line(sampled)
    start_s = sampled[0][0]
    end_s = sampled[-1][0]
    count = max(2, int(math.ceil((end_s - start_s) / max(step_m, 0.1))) + 1)
    return [
        (station, slope * station + intercept)
        for station in (start_s + (end_s - start_s) * index / (count - 1) for index in range(count))
    ]


def fit_station_offset_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, points[0][1] if points else 0.0
    mean_s = sum(station for station, _ in points) / len(points)
    mean_t = sum(offset for _, offset in points) / len(points)
    denom = sum((station - mean_s) ** 2 for station, _ in points)
    if denom <= 1e-9:
        return 0.0, mean_t
    slope = sum((station - mean_s) * (offset - mean_t) for station, offset in points) / denom
    intercept = mean_t - slope * mean_s
    return slope, intercept


def linear_fit_rms(points: list[tuple[float, float]], slope: float, intercept: float) -> float:
    if not points:
        return 0.0
    return math.sqrt(sum((offset - (slope * station + intercept)) ** 2 for station, offset in points) / len(points))


def cubic_hermite_station_offsets(
    start: tuple[float, float],
    end: tuple[float, float],
    start_slope: float,
    end_slope: float,
    *,
    step_m: float,
) -> list[tuple[float, float]]:
    station0, offset0 = start
    station1, offset1 = end
    span = station1 - station0
    if span <= 0:
        return [start, end]
    count = max(5, int(math.ceil(span / max(step_m, 0.1))) + 1)
    points: list[tuple[float, float]] = []
    for index in range(count):
        u = index / (count - 1)
        h00 = 2.0 * u * u * u - 3.0 * u * u + 1.0
        h10 = u * u * u - 2.0 * u * u + u
        h01 = -2.0 * u * u * u + 3.0 * u * u
        h11 = u * u * u - u * u
        offset = h00 * offset0 + h10 * span * start_slope + h01 * offset1 + h11 * span * end_slope
        points.append((station0 + span * u, offset))
    return points


def slope_at_start(points: list[tuple[float, float]], sample_count: int = 5) -> float:
    return slope_between(points[: max(2, min(len(points), sample_count))])


def slope_at_end(points: list[tuple[float, float]], sample_count: int = 5) -> float:
    return slope_between(points[max(0, len(points) - sample_count) :])


def slope_between(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    start = points[0]
    end = points[-1]
    span = end[0] - start[0]
    if abs(span) <= 1e-9:
        return 0.0
    return (end[1] - start[1]) / span


def mainline_tangent_slope(evidence_features: list[dict[str, Any]], *, guide: LinearGuide, station: float) -> float:
    gp01 = find_evidence_by_seq_id(evidence_features, "TA08_GP01")
    if gp01 is None:
        return 0.0
    points = station_offsets_for_coords(line_coords(gp01), guide=guide)
    nearby = [point for point in points if abs(point[0] - station) <= 15.0]
    if len(nearby) < 2:
        return 0.0
    return slope_between(nearby)


def merge_station_offset_parts(parts: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for part in parts:
        for point in part:
            if merged and abs(point[0] - merged[-1][0]) < 1e-6:
                merged[-1] = point
            elif not merged or point[0] > merged[-1][0]:
                merged.append(point)
    return dedupe_station_offsets(merged)


def dedupe_station_offsets(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for station, offset in sorted(points, key=lambda item: item[0]):
        if deduped and abs(station - deduped[-1][0]) < 1e-6:
            deduped[-1] = (station, offset)
        else:
            deduped.append((station, offset))
    return deduped


def interpolated_offset(points: list[tuple[float, float]], station: float) -> float:
    points = dedupe_station_offsets(points)
    if station <= points[0][0]:
        return points[0][1]
    if station >= points[-1][0]:
        return points[-1][1]
    for (s0, t0), (s1, t1) in zip(points, points[1:]):
        if s0 <= station <= s1:
            if abs(s1 - s0) <= 1e-9:
                return t1
            u = (station - s0) / (s1 - s0)
            return t0 + (t1 - t0) * u
    return points[-1][1]


def station_offset_curve_summary(points: list[tuple[float, float]]) -> dict[str, float]:
    slopes: list[tuple[float, float]] = []
    curvatures: list[tuple[float, float]] = []
    for left, right in zip(points, points[1:]):
        span = right[0] - left[0]
        if abs(span) <= 1e-9:
            continue
        slopes.append(((left[0] + right[0]) / 2.0, (right[1] - left[1]) / span))
    for left, right in zip(slopes, slopes[1:]):
        span = right[0] - left[0]
        if abs(span) <= 1e-9:
            continue
        curvatures.append(((left[0] + right[0]) / 2.0, (right[1] - left[1]) / span))
    active_slopes = [item for item in slopes if abs(item[1]) > 0.015]
    max_abs_slope = max((abs(value) for _, value in slopes), default=0.0)
    max_abs_curvature = max((abs(value) for _, value in curvatures), default=0.0)
    return {
        "curve_start_s_m": active_slopes[0][0] if active_slopes else points[0][0],
        "curve_end_s_m": active_slopes[-1][0] if active_slopes else points[-1][0],
        "max_abs_slope": max_abs_slope,
        "max_abs_curvature": max_abs_curvature,
    }


def safe_interval_id(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    value = str(props.get("interval_id", props.get("line_id", "x")))
    return "".join(ch if ch.isalnum() else "_" for ch in value) or "x"


def short_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)[:42] or "id"


def select_track_band_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for feature in features:
        band_id = str((feature.get("properties") or {}).get("band_id", ""))
        if band_id in KEEP_BANDS:
            selected.append(feature)
    return selected


def select_diagnostic_track_band_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties") or {}
        band_id = str(props.get("band_id", ""))
        role = str(props.get("role", ""))
        if band_id in KEEP_BANDS:
            continue
        if role == "diagnostic_candidate" or band_id.startswith(("possible_", "diagnostic_", "candidate_")):
            selected.append(feature)
    return selected


def select_turnout_features(features: list[dict[str, Any]], *, include_low_support: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties") or {}
        status = str(props.get("qa_status", ""))
        if include_low_support or status != "self_review_pass_low_support":
            selected.append(feature)
    return selected


def annotate_feature(
    feature: dict[str, Any],
    *,
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    network_role: str,
    source_layer: str,
    line_id: str,
    support_threshold_m: float,
    sample_step_m: float,
) -> dict[str, Any]:
    coords = line_coords(feature)
    props = dict(feature.get("properties") or {})
    support = measure_support(
        coords,
        evidence_segments=evidence_segments,
        threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    risk_flag = classify_risk(props, network_role=network_role, support_ratio=support["support_ratio"])
    props.update(
        {
            "line_id": line_id,
            "network_role": network_role,
            "source_layer": source_layer,
            "length_m": round(polyline_length(coords), 3),
            "deeplab_support_ratio": support["support_ratio"],
            "deeplab_mean_distance_m": support["mean_distance_m"],
            "deeplab_max_unsupported_gap_m": support["max_unsupported_gap_m"],
            "deeplab_sample_count": support["sample_count"],
            "support_threshold_m": support_threshold_m,
            "risk_flag": risk_flag,
            "postprocess_policy": "topology_skeleton_with_deeplab_support_metric",
        }
    )
    return {"type": "Feature", "properties": props, "geometry": feature["geometry"]}


def classify_risk(props: dict[str, Any], *, network_role: str, support_ratio: float) -> str:
    if network_role == "main_through_track":
        return "accepted_mainline_check_deeplab_support"
    qa_status = str(props.get("qa_status", ""))
    branch_id = str(props.get("branch_id", props.get("anchor_id", "")))
    if qa_status == "self_review_pass_low_support" or branch_id == "TA08":
        return "review_priority_low_support_turnout"
    if network_role == "turnout_connector" and support_ratio < 0.25:
        return "review_priority_low_deeplab_support"
    if support_ratio < 0.20:
        return "weak_deeplab_support_keep_by_topology"
    return "normal_review"


def band_network_role(feature: dict[str, Any]) -> str:
    band_id = str((feature.get("properties") or {}).get("band_id", ""))
    if band_id == MAIN_BAND_ID:
        return "main_through_track"
    return "parallel_straight_track"


def band_line_id(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    band_id = str(props.get("band_id", "band"))
    interval_id = str(props.get("interval_id", "0"))
    return f"BAND_{band_id}_{interval_id}"


def turnout_line_id(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    branch_id = str(props.get("branch_id", props.get("anchor_id", props.get("connector_id", "turnout"))))
    return f"TURNOUT_{branch_id}"


def station_range(feature: dict[str, Any]) -> tuple[float, float]:
    props = feature.get("properties") or {}
    values = [
        safe_float(props.get("station_min_m", props.get("s_min_m", 0.0))),
        safe_float(props.get("station_max_m", props.get("s_max_m", 0.0))),
    ]
    return min(values), max(values)


def feature_sort_key(feature: dict[str, Any]) -> tuple[int, float, str]:
    props = feature.get("properties") or {}
    role = str(props.get("network_role", ""))
    role_rank = {
        "main_through_track": 0,
        "parallel_straight_track": 1,
        "promoted_straight_track": 2,
        "straight_gap_bridge": 3,
        "turnout_tail_bridge": 4,
        "turnout_boundary_bridge": 5,
        "turnout_connector": 6,
    }.get(role, 9)
    station = safe_float(props.get("station_min_m", props.get("s_min_m", 0.0)))
    return role_rank, station, str(props.get("line_id", ""))


def measure_support(
    coords: list[tuple[float, float]],
    *,
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    threshold_m: float,
    sample_step_m: float,
) -> dict[str, Any]:
    samples = sample_polyline(coords, step_m=sample_step_m)
    if not samples or not evidence_segments:
        return {
            "support_ratio": 0.0,
            "mean_distance_m": 0.0,
            "max_unsupported_gap_m": 0.0,
            "sample_count": 0,
        }
    distances: list[float] = []
    supported = 0
    current_gap = 0
    max_gap = 0
    for point in samples:
        distance = nearest_segment_distance(point, evidence_segments)
        distances.append(distance)
        if distance <= threshold_m:
            supported += 1
            max_gap = max(max_gap, current_gap)
            current_gap = 0
        else:
            current_gap += 1
    max_gap = max(max_gap, current_gap)
    return {
        "support_ratio": round(supported / len(samples), 4),
        "mean_distance_m": round(sum(distances) / len(distances), 4),
        "max_unsupported_gap_m": round(max_gap * sample_step_m, 3),
        "sample_count": len(samples),
    }


def sample_polyline(coords: list[tuple[float, float]], *, step_m: float) -> list[tuple[float, float]]:
    if len(coords) < 2:
        return coords
    step_m = max(float(step_m), 0.25)
    samples = [coords[0]]
    carry = 0.0
    previous = coords[0]
    for current in coords[1:]:
        segment_length = distance(previous, current)
        if segment_length <= 1e-9:
            previous = current
            continue
        travelled = step_m - carry
        while travelled <= segment_length:
            ratio = travelled / segment_length
            samples.append(
                (
                    previous[0] + (current[0] - previous[0]) * ratio,
                    previous[1] + (current[1] - previous[1]) * ratio,
                )
            )
            travelled += step_m
        carry = segment_length - (travelled - step_m)
        previous = current
    if distance(samples[-1], coords[-1]) > 1e-6:
        samples.append(coords[-1])
    return samples


def nearest_segment_distance(point: tuple[float, float], segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> float:
    best = float("inf")
    px, py = point
    for a, b in segments:
        min_x = min(a[0], b[0]) - best
        max_x = max(a[0], b[0]) + best
        min_y = min(a[1], b[1]) - best
        max_y = max(a[1], b[1]) + best
        if px < min_x or px > max_x or py < min_y or py > max_y:
            continue
        candidate = point_segment_distance(point, a, b)
        if candidate < best:
            best = candidate
    return best


def point_segment_distance(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return point_segment_projection(point, a, b)[0]


def point_segment_projection(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, tuple[float, float]]:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        projected = (ax, ay)
        return math.hypot(px - ax, py - ay), projected
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx = ax + t * dx
    qy = ay + t * dy
    projected = (qx, qy)
    return math.hypot(px - qx, py - qy), projected


def build_segments(features: Iterable[dict[str, Any]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for feature in features:
        coords = line_coords(feature)
        segments.extend(zip(coords, coords[1:]))
    return segments


def normalize_evidence_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        props = dict(feature.get("properties") or {})
        source = str(props.get("source", props.get("network_source", "")))
        role = str(props.get("role", "deeplab_evidence"))
        props.update(
            {
                "evidence_id": str(props.get("line_id", props.get("seq_id", f"E{index:03d}"))),
                "evidence_role": role,
                "source_layer": source or "deeplab_centerline_network",
                "length_m": round(polyline_length(line_coords(feature)), 3),
            }
        )
        normalized.append({"type": "Feature", "properties": props, "geometry": feature["geometry"]})
    return normalized


def summarize_run(
    final_features: list[dict[str, Any]],
    *,
    deeplab_features: list[dict[str, Any]],
    gauge_features: list[dict[str, Any]],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    role_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    for feature in final_features:
        props = feature.get("properties") or {}
        role_counts[str(props.get("network_role", ""))] = role_counts.get(str(props.get("network_role", "")), 0) + 1
        risk_counts[str(props.get("risk_flag", ""))] = risk_counts.get(str(props.get("risk_flag", "")), 0) + 1
    low_support = [
        summarize_feature(feature)
        for feature in final_features
        if str((feature.get("properties") or {}).get("risk_flag", "")).startswith("review_priority")
        or str((feature.get("properties") or {}).get("risk_flag", "")) == "weak_deeplab_support_keep_by_topology"
    ]
    return {
        "mode": "deeplab_topology_centerline_network_v1",
        "policy": {
            "segmentation_first": "DeepLab-derived centerlines are used as evidence and support metrics.",
            "geometry_priority": "accepted mainline and support-bounded parallel bands define straight-track topology; turnout candidates are connector edges.",
            "terminal_port_assumption": "mainline_2_track is kept as the uninterrupted through-track.",
            "low_support_rule": "low DeepLab support does not delete topology-required lines, but marks them for QGIS review.",
            "straight_gap_bridge_rule": "same-band internal straight-track gaps are bridged when they are away from turnouts and either DeepLab-supported or short enough for topology-only occlusion bridging.",
            "diagnostic_track_promotion_rule": "diagnostic or possible outer track-band candidates are promoted only when DeepLab/gauge evidence strongly supports them; short evidence-supported bridges can connect them to nearby turnout tails.",
            "connector_endpoint_snap_rule": "optional QGIS review packaging can snap turnout/bridge endpoints to nearby accepted track lines without changing the track-band geometry.",
        },
        "inputs": {
            "deeplab_network": str(args.deeplab_network.expanduser().resolve()),
            "track_bands": str(args.track_bands.expanduser().resolve()),
            "turnouts": str(args.turnouts.expanduser().resolve()),
            "gauge_pair": str(args.gauge_pair.expanduser().resolve()) if args.gauge_pair.exists() else None,
        },
        "outputs": {
            "network_geojson": str(out_dir / "deeplab_topology_centerline_network.geojson"),
            "network_shp": str(out_dir / "deeplab_topology_centerline_network.shp"),
            "network_qml": str(out_dir / "deeplab_topology_centerline_network.qml"),
            "evidence_geojson": str(out_dir / "deeplab_topology_evidence.geojson"),
            "evidence_shp": str(out_dir / "deeplab_topology_evidence.shp"),
            "review_md": str(out_dir / "REVIEW.md"),
            "summary_json": str(out_dir / "summary.json"),
        },
        "feature_count": len(final_features),
        "role_counts": role_counts,
        "risk_counts": risk_counts,
        "deeplab_evidence_feature_count": len(deeplab_features),
        "gauge_pair_evidence_feature_count": len(gauge_features),
        "support_threshold_m": args.support_threshold_m,
        "sample_step_m": args.sample_step_m,
        "bridge_max_gap_m": args.bridge_max_gap_m,
        "bridge_evidence_support": args.bridge_evidence_support,
        "weak_gap_bridge_max_gap_m": args.weak_gap_bridge_max_gap_m,
        "snap_connector_endpoints_m": args.snap_connector_endpoints_m,
        "low_support_or_review_priority": low_support,
        "features": [summarize_feature(feature) for feature in final_features],
    }


def summarize_feature(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties") or {}
    return {
        "line_id": props.get("line_id", ""),
        "network_role": props.get("network_role", ""),
        "band_id": props.get("band_id", ""),
        "branch_id": props.get("branch_id", props.get("anchor_id", "")),
        "station_min_m": props.get("station_min_m", props.get("s_min_m", None)),
        "station_max_m": props.get("station_max_m", props.get("s_max_m", None)),
        "length_m": props.get("length_m", round(polyline_length(line_coords(feature)), 3)),
        "deeplab_support_ratio": props.get("deeplab_support_ratio", 0.0),
        "deeplab_mean_distance_m": props.get("deeplab_mean_distance_m", 0.0),
        "deeplab_max_unsupported_gap_m": props.get("deeplab_max_unsupported_gap_m", 0.0),
        "risk_flag": props.get("risk_flag", ""),
        "qa_status": props.get("qa_status", ""),
        "endpoint_snap_count": props.get("endpoint_snap_count", 0),
        "endpoint_snap_max_m": props.get("endpoint_snap_max_m", 0.0),
    }


def write_review_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DeepLab Topology Centerline V1",
        "",
        "This layer is a topology-aware postprocess output. DeepLab evidence is used for support metrics, while the main through-track, parallel straight tracks, and turnout connectors remain explicit topology objects.",
        "",
        "## Outputs",
        "",
        f"- Network Shapefile: `{summary['outputs']['network_shp']}`",
        f"- Evidence Shapefile: `{summary['outputs']['evidence_shp']}`",
        f"- QGIS style: `{summary['outputs']['network_qml']}`",
        "",
        "## Counts",
        "",
    ]
    for role, count in sorted((summary.get("role_counts") or {}).items()):
        lines.append(f"- `{role}`: {count}")
    lines.extend(["", "## Review Priority", ""])
    priority = summary.get("low_support_or_review_priority") or []
    if not priority:
        lines.append("- No priority features were flagged by the support metric.")
    else:
        for item in priority:
            lines.append(
                f"- `{item['line_id']}`: support={item['deeplab_support_ratio']}, "
                f"max_gap={item['deeplab_max_unsupported_gap_m']}m, risk=`{item['risk_flag']}`"
            )
    promoted = [item for item in summary.get("features", []) if item.get("network_role") == "promoted_straight_track"]
    lines.extend(["", "## Promoted Diagnostic Tracks", ""])
    if not promoted:
        lines.append("- No diagnostic track-band candidates were promoted.")
    else:
        for item in promoted:
            lines.append(
                f"- `{item['line_id']}`: band=`{item['band_id']}`, "
                f"station={item['station_min_m']}..{item['station_max_m']}m, "
                f"support={item['deeplab_support_ratio']}, risk=`{item['risk_flag']}`"
            )
    bridges = [
        item
        for item in summary.get("features", [])
        if item.get("network_role") in {"straight_gap_bridge", "turnout_tail_bridge", "turnout_boundary_bridge"}
    ]
    lines.extend(["", "## Bridge Features", ""])
    if not bridges:
        lines.append("- No bridge features were created.")
    else:
        for item in bridges:
            lines.append(
                f"- `{item['line_id']}`: band=`{item['band_id']}`, "
                f"station={item['station_min_m']}..{item['station_max_m']}m, "
                f"support={item['deeplab_support_ratio']}, risk=`{item['risk_flag']}`"
            )
    lines.extend(
        [
            "",
            "## QGIS Review Guidance",
            "",
            "- Use the network layer as the candidate centerline network.",
            "- Load the evidence layer underneath it to see what DeepLab directly supports.",
            "- Do not reject the accepted mainline only because the DeepLab support metric is imperfect; it is retained by topology.",
            "- Treat `evidence_promoted_bridge` as DeepLab-supported straight-track completion and `occlusion_bridge` as topology-supported completion that needs visual review.",
            "- Treat `evidence_promoted_diagnostic_track` as a formerly diagnostic outer track candidate that earned promotion from local evidence; it should still receive a visual pass.",
            "- Prioritize low-support turnout connectors, especially TA08, for DOM and point-cloud review.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_centerline_shapefile(features: list[dict[str, Any]], path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("line_id", "C", size=96)
    writer.field("net_role", "C", size=32)
    writer.field("src_layer", "C", size=32)
    writer.field("band_id", "C", size=32)
    writer.field("branch_id", "C", size=24)
    writer.field("risk", "C", size=48)
    writer.field("qa_status", "C", size=36)
    writer.field("len_m", "F", decimal=3)
    writer.field("s0_m", "F", decimal=3)
    writer.field("s1_m", "F", decimal=3)
    writer.field("dl_sup", "F", decimal=4)
    writer.field("dl_mean", "F", decimal=4)
    writer.field("dl_gap", "F", decimal=3)
    writer.field("score", "F", decimal=4)
    writer.field("trans_cov", "F", decimal=4)
    writer.field("snap_n", "N", size=4)
    writer.field("snap_m", "F", decimal=4)
    writer.field("note", "C", size=160)
    for feature in features:
        props = feature.get("properties") or {}
        coords = line_coords(feature)
        writer.line([coords])
        writer.record(
            str(props.get("line_id", ""))[:96],
            str(props.get("network_role", ""))[:32],
            str(props.get("source_layer", ""))[:32],
            str(props.get("band_id", ""))[:32],
            str(props.get("branch_id", props.get("anchor_id", "")))[:24],
            str(props.get("risk_flag", ""))[:48],
            str(props.get("qa_status", ""))[:36],
            safe_float(props.get("length_m", polyline_length(coords))),
            safe_float(props.get("station_min_m", props.get("s_min_m", 0.0))),
            safe_float(props.get("station_max_m", props.get("s_max_m", 0.0))),
            safe_float(props.get("deeplab_support_ratio", 0.0)),
            safe_float(props.get("deeplab_mean_distance_m", 0.0)),
            safe_float(props.get("deeplab_max_unsupported_gap_m", 0.0)),
            safe_float(props.get("connector_score", props.get("template_score", 0.0))),
            safe_float(props.get("trans_cov", props.get("transition_cov", 0.0))),
            int(safe_float(props.get("endpoint_snap_count", 0.0))),
            safe_float(props.get("endpoint_snap_max_m", 0.0)),
            str(props.get("self_note", props.get("review_note", "")))[:160],
        )
    writer.close()
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    write_projection(path.with_suffix(".prj"), epsg)


def write_evidence_shapefile(features: list[dict[str, Any]], path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("evid_id", "C", size=48)
    writer.field("role", "C", size=32)
    writer.field("source", "C", size=48)
    writer.field("len_m", "F", decimal=3)
    writer.field("conf", "F", decimal=4)
    for feature in features:
        props = feature.get("properties") or {}
        coords = line_coords(feature)
        writer.line([coords])
        writer.record(
            str(props.get("evidence_id", ""))[:48],
            str(props.get("evidence_role", ""))[:32],
            str(props.get("source_layer", ""))[:48],
            safe_float(props.get("length_m", polyline_length(coords))),
            safe_float(props.get("mean_confidence", props.get("mean_score", 0.0))),
        )
    writer.close()
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    write_projection(path.with_suffix(".prj"), epsg)


def write_projection(path: Path, epsg: int) -> None:
    path.write_text(rasterio.crs.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")


def write_centerline_qml(path: Path) -> None:
    categories = [
        ("main_through_track", "255,0,0,255", "main through track", 0.78),
        ("parallel_straight_track", "0,114,178,255", "parallel straight track", 0.62),
        ("promoted_straight_track", "140,80,255,255", "promoted diagnostic straight track", 0.70),
        ("straight_gap_bridge", "255,140,0,255", "straight gap bridge", 0.70),
        ("turnout_tail_bridge", "255,80,180,255", "turnout tail bridge", 0.76),
        ("turnout_boundary_bridge", "0,190,190,255", "turnout boundary bridge", 0.76),
        ("turnout_connector", "0,170,80,255", "turnout connector", 0.82),
    ]
    write_categorized_line_qml(path, attr="net_role", categories=categories)


def write_evidence_qml(path: Path) -> None:
    categories = [
        ("mainline", "255,40,220,170", "DeepLab mainline evidence", 0.35),
        ("support", "20,220,90,150", "DeepLab support evidence", 0.28),
        ("deeplab_gauge_pair_centerline", "255,210,0,210", "gauge-pair evidence", 0.50),
    ]
    write_categorized_line_qml(path, attr="role", categories=categories)


def write_categorized_line_qml(path: Path, *, attr: str, categories: list[tuple[str, str, str, float]]) -> None:
    cat_lines: list[str] = []
    symbol_lines: list[str] = []
    for index, (value, color, label, width) in enumerate(categories):
        cat_lines.append(f'      <category value="{value}" symbol="{index}" label="{label}" render="true"/>')
        symbol_lines.append(
            f"""      <symbol name="{index}" type="line" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="capstyle" type="QString" value="round"/>
            <Option name="joinstyle" type="QString" value="round"/>
            <Option name="line_color" type="QString" value="{color}"/>
            <Option name="line_style" type="QString" value="solid"/>
            <Option name="line_width" type="QString" value="{width}"/>
            <Option name="line_width_unit" type="QString" value="MM"/>
          </Option>
          <data_defined_properties><Option type="Map"/></data_defined_properties>
        </layer>
      </symbol>"""
        )
    path.write_text(
        f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="{attr}" enableorderby="0" forceraster="0" referencescale="-1" symbollevels="0">
    <categories>
{chr(10).join(cat_lines)}
    </categories>
    <symbols>
{chr(10).join(symbol_lines)}
    </symbols>
    <source-symbol><symbol name="0" type="line" alpha="1"/></source-symbol>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerGeometryType>1</layerGeometryType>
</qgis>
""",
        encoding="utf-8",
    )


def load_line_features(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [feature for feature in data.get("features", []) if line_coords(feature)]


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "LineString":
        return []
    coords: list[tuple[float, float]] = []
    for coord in geometry.get("coordinates", []):
        if len(coord) < 2:
            continue
        coords.append((float(coord[0]), float(coord[1])))
    return coords


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(distance(a, b) for a, b in zip(coords, coords[1:]))


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
