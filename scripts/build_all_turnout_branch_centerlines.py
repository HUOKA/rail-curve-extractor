from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import build_turnout_connector_candidates as btc


DEFAULT_TEMPLATE = Path("output/raw_dom_roi_fullpass_v1/turnout_template_connectors/turnout_template_connector_proposals.geojson")
DEFAULT_CROSSOVER = Path("output/raw_dom_roi_fullpass_v1/turnout_crossover_las_validation/turnout_crossover_las_endpoint_locked_centerlines.geojson")
DEFAULT_TRACK_BANDS = Path("output/raw_dom_roi_fullpass_v1/track_band_priors/track_band_centerline_priors.geojson")
DEFAULT_MAINLINE = Path("output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_GAUGE_PAIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_turnouts_v1/deeplab_gauge_pair_centerlines.geojson")
DEFAULT_DOM = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u65e0\u4eba\u673a\u6570\u636e" / "\u6b63\u5c04" / "dom.tif"
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/all_turnout_branch_centerlines")
DEFAULT_EPSG = 32651

CROSSOVER_REPLACED_ANCHORS = {"TA01", "TA02", "TA04", "TA05"}
KEEP_TEMPLATE_ANCHORS = {"TA03", "TA06", "TA07", "TA08", "TA09"}

SELF_REVIEW = {
    "CX01": (
        "self_review_pass_visual",
        "visual pass on full-res DOM crops; endpoints stay on track centers, middle remains inside branch corridor",
    ),
    "CX02": (
        "self_review_pass_visual",
        "visual pass on full-res DOM crops; endpoint-locked LAS line stays inside the crossover track corridor",
    ),
    "TA03": (
        "self_review_pass_visual",
        "visual pass on full-res DOM crops; template line follows the turnout branch center",
    ),
    "TA06": (
        "self_review_pass_visual",
        "visual pass on full-res DOM crops; template line follows the turnout branch center",
    ),
    "TA07": (
        "self_review_pass_visual",
        "visual pass on full-res DOM crops; template line follows the turnout branch center",
    ),
    "TA08": (
        "self_review_pass_low_support",
        "visual pass but low segmentation support; prioritize this branch during QGIS review",
    ),
    "TA09": (
        "self_review_pass_visual",
        "visual pass on full-res DOM crops; template line follows the turnout branch center",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one QGIS review layer for all turnout branch centerline candidates.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--crossover", type=Path, default=DEFAULT_CROSSOVER)
    parser.add_argument("--track-bands", type=Path, default=DEFAULT_TRACK_BANDS)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument("--gauge-pair", type=Path, default=DEFAULT_GAUGE_PAIR)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bounds-padding-m", type=float, default=11.0)
    parser.add_argument("--segment-crop-m", type=float, default=42.0)
    parser.add_argument("--line-width-px", type=int, default=3)
    parser.add_argument("--skip-qa-crops", action="store_true")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    template_features = btc.load_line_features(args.template.expanduser().resolve())
    crossover_features = btc.load_line_features(args.crossover.expanduser().resolve())
    mainline_features = btc.load_line_features(args.mainline.expanduser().resolve()) if args.mainline.exists() else []
    guide = build_guide(mainline_features[0]) if mainline_features else None
    gauge_pair_features = btc.load_line_features(args.gauge_pair.expanduser().resolve()) if args.gauge_pair.exists() else []
    selected = build_all_turnout_features(template_features, crossover_features, gauge_pair_features=gauge_pair_features, guide=guide)

    geojson_path = out_dir / "all_turnout_branch_centerlines.geojson"
    btc.write_geojson(geojson_path, selected, epsg=args.epsg)
    write_all_turnout_shapefile(selected, geojson_path.with_suffix(".shp"), epsg=args.epsg)
    write_all_turnout_qml(geojson_path.with_suffix(".qml"))

    qa_summary = None
    dom_path = args.dom.expanduser().resolve()
    if not args.skip_qa_crops:
        if not dom_path.exists():
            raise FileNotFoundError(f"DOM not found: {dom_path}")
        band_features = btc.load_line_features(args.track_bands.expanduser().resolve()) if args.track_bands.exists() else []
        qa_summary = write_fullres_qa_crops(
            dom_path,
            branch_features=selected,
            band_features=band_features,
            out_dir=out_dir / "qa_crops",
            bounds_padding_m=args.bounds_padding_m,
            segment_crop_m=args.segment_crop_m,
            line_width_px=args.line_width_px,
        )

    summary = {
        "mode": "all_turnout_branch_centerlines_review_package",
        "policy": {
            "status": "candidate_layer_for_user_qgis_review",
            "rule": "Use paired crossover geometry for TA01/TA02 and TA04/TA05; keep remaining template/special candidates as independent turnout branches.",
            "crossover_replaced_anchors": sorted(CROSSOVER_REPLACED_ANCHORS),
            "template_kept_anchors": sorted(KEEP_TEMPLATE_ANCHORS),
            "visual_qa": "Full-resolution DOM crops are generated from the original DOM without resizing.",
        },
        "inputs": {
            "template": str(args.template.expanduser().resolve()),
            "crossover": str(args.crossover.expanduser().resolve()),
            "track_bands": str(args.track_bands.expanduser().resolve()),
            "mainline": str(args.mainline.expanduser().resolve()),
            "gauge_pair": str(args.gauge_pair.expanduser().resolve()) if args.gauge_pair.exists() else None,
            "dom": str(dom_path),
        },
        "feature_count": len(selected),
        "features": [summarize_feature(feature) for feature in selected],
        "outputs": {
            "geojson": str(geojson_path),
            "shp": str(geojson_path.with_suffix(".shp")),
            "qml": str(geojson_path.with_suffix(".qml")),
            "qa_crops": str(out_dir / "qa_crops") if qa_summary else None,
            "visual_qa_md": str(out_dir / "VISUAL_QA.md"),
            "summary_json": str(out_dir / "summary.json"),
        },
        "qa_crops": qa_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_visual_qa(out_dir / "VISUAL_QA.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_all_turnout_features(
    template_features: list[dict[str, Any]],
    crossover_features: list[dict[str, Any]],
    *,
    gauge_pair_features: list[dict[str, Any]] | None = None,
    guide: btc.Guide | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for feature in crossover_features:
        props = feature.get("properties") or {}
        connector_id = str(props.get("connector_id", props.get("conn_id", "")))
        if connector_id not in {"CX01", "CX02"}:
            continue
        selected.append(normalize_feature(feature, source_type="crossover_las_endpoint_locked"))

    for feature in template_features:
        props = feature.get("properties") or {}
        anchor_id = str(props.get("anchor_id", ""))
        if anchor_id in CROSSOVER_REPLACED_ANCHORS:
            continue
        if KEEP_TEMPLATE_ANCHORS and anchor_id not in KEEP_TEMPLATE_ANCHORS:
            continue
        normalized = normalize_feature(feature, source_type="remaining_template_or_special")
        if guide is not None and gauge_pair_features:
            normalized = apply_gauge_pair_refinement(normalized, gauge_pair_features, guide=guide)
        selected.append(normalized)

    selected.sort(key=feature_sort_key)
    return selected


def build_guide(mainline_feature: dict[str, Any]) -> btc.Guide:
    coords = btc.line_coords(mainline_feature)
    if len(coords) < 2:
        raise ValueError("Mainline feature has too few coordinates.")
    return btc.Guide(coords[0], coords[-1])


def apply_gauge_pair_refinement(feature: dict[str, Any], gauge_pair_features: list[dict[str, Any]], *, guide: btc.Guide) -> dict[str, Any]:
    props = feature.get("properties") or {}
    if not needs_gauge_pair_refinement(props):
        return feature
    match = best_gauge_pair_match(feature, gauge_pair_features, guide=guide)
    if match is None:
        return feature
    return build_gauge_pair_constrained_feature(feature, match["feature"], guide=guide, match=match)


def needs_gauge_pair_refinement(props: dict[str, Any]) -> bool:
    qa_status = str(props.get("qa_status", ""))
    support_cov = safe_float_default(props.get("support_cov", 1.0), 1.0)
    transition_cov = safe_float_default(props.get("trans_cov", props.get("transition_cov", 1.0)), 1.0)
    return "low_support" in qa_status or support_cov < 0.35 or transition_cov < 0.35


def safe_float_default(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def best_gauge_pair_match(feature: dict[str, Any], gauge_pair_features: list[dict[str, Any]], *, guide: btc.Guide) -> dict[str, Any] | None:
    candidate_st = btc.station_offsets_for_feature(feature, guide=guide)
    if len(candidate_st) < 2:
        return None
    candidate_station_min = min(station for station, _ in candidate_st)
    candidate_station_max = max(station for station, _ in candidate_st)
    candidate_span = max(candidate_station_max - candidate_station_min, 1e-6)
    candidate_segments = list(zip(candidate_st, candidate_st[1:]))
    best: dict[str, Any] | None = None
    for gauge_feature in gauge_pair_features:
        gauge_st = btc.station_offsets_for_feature(gauge_feature, guide=guide)
        if len(gauge_st) < 2:
            continue
        gauge_station_min = min(station for station, _ in gauge_st)
        gauge_station_max = max(station for station, _ in gauge_st)
        gauge_span = gauge_station_max - gauge_station_min
        if gauge_span < 15.0:
            continue
        offsets = [offset for _, offset in gauge_st]
        offset_span = max(offsets) - min(offsets)
        max_abs_offset = max(abs(offset) for offset in offsets)
        if offset_span < 1.0 or max_abs_offset < 1.5:
            continue
        overlap = min(candidate_station_max, gauge_station_max) - max(candidate_station_min, gauge_station_min)
        min_overlap = max(12.0, min(candidate_span, gauge_span) * 0.25)
        if overlap < min_overlap:
            continue
        distances = [nearest_st_distance(point, candidate_segments) for point in gauge_st[:: max(1, len(gauge_st) // 40)]]
        mean_distance = sum(distances) / len(distances) if distances else 999.0
        score = (overlap / candidate_span) + min(offset_span / 4.0, 1.0) - min(mean_distance / 5.0, 1.0) * 0.25
        match = {
            "feature": gauge_feature,
            "score": score,
            "overlap_m": overlap,
            "mean_distance_m": mean_distance,
            "offset_span_m": offset_span,
            "gauge_span_m": gauge_span,
        }
        if best is None or score > best["score"]:
            best = match
    return best


def nearest_st_distance(point: tuple[float, float], segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> float:
    if not segments:
        return float("inf")
    return min(point_segment_distance(point, start, end) for start, end in segments)


def point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def build_gauge_pair_constrained_feature(
    feature: dict[str, Any],
    gauge_feature: dict[str, Any],
    *,
    guide: btc.Guide,
    match: dict[str, Any],
    max_completion_gap_m: float = 45.0,
    max_completion_offset_shift_m: float = 1.2,
) -> dict[str, Any]:
    props = dict(feature.get("properties") or {})
    candidate_st = btc.station_offsets_for_feature(feature, guide=guide)
    gauge_st = btc.station_offsets_for_feature(gauge_feature, guide=guide)
    if len(candidate_st) < 2 or len(gauge_st) < 2:
        return feature
    candidate_start = min(candidate_st, key=lambda item: item[0])
    candidate_end = max(candidate_st, key=lambda item: item[0])
    gauge_start = min(gauge_st, key=lambda item: item[0])
    gauge_end = max(gauge_st, key=lambda item: item[0])

    merged: list[tuple[float, float]] = []
    completion_m = 0.0
    if candidate_start[0] < gauge_start[0] and gauge_start[0] - candidate_start[0] <= max_completion_gap_m:
        prefix = interpolate_st(candidate_start, gauge_start, step_m=1.0)
        merged.extend(prefix[:-1])
        completion_m += gauge_start[0] - candidate_start[0]
    merged.extend(sorted(gauge_st, key=lambda item: item[0]))
    if candidate_end[0] > gauge_end[0] and candidate_end[0] - gauge_end[0] <= max_completion_gap_m:
        suffix_end = candidate_end
        suffix_mode = "template_endpoint"
        offset_shift = abs(candidate_end[1] - gauge_end[1])
        if offset_shift > max_completion_offset_shift_m:
            suffix_mode = "evidence_limited_tangent_endpoint"
            suffix = interpolate_st(gauge_end, suffix_end, step_m=1.0, progress_power=0.35)
        else:
            suffix = interpolate_st(gauge_end, suffix_end, step_m=1.0)
        merged.extend(suffix[1:])
        completion_m += candidate_end[0] - gauge_end[0]
    else:
        suffix_mode = "none"
        offset_shift = 0.0

    merged = dedupe_station_points(sorted(merged, key=lambda item: item[0]))
    smoothing_mode = "none"
    if suffix_mode == "evidence_limited_tangent_endpoint":
        merged = smooth_station_offset_curve(merged, step_m=0.75, window_size=5, passes=4)
        smoothing_mode = "station_offset_curvature_limited"
    coords = [guide.point_at(station, offset) for station, offset in merged]
    offsets = [offset for _, offset in merged]
    stations = [station for station, _ in merged]
    gauge_props = gauge_feature.get("properties") or {}
    self_note = "gauge-pair constrained candidate; review DOM and completion stubs before acceptance"
    if suffix_mode == "evidence_limited_tangent_endpoint":
        self_note = "branch gauge evidence ends before mainline tangent; suffix tapers early to avoid unsupported parallel centerline"
    props.update(
        {
            "source_type": "gauge_pair_evidence_constrained",
            "source": "deeplab_gauge_pair_evidence_constrained",
            "geom_kind": "gauge_pair_evidence_constrained_centerline",
            "review_status": "candidate_needs_dom_review",
            "qa_status": "self_review_needs_visual_check",
            "review_note": "low-support turnout geometry rebuilt from nearby DeepLab gauge-pair centerline evidence",
            "self_note": self_note,
            "gauge_seq": str(gauge_props.get("seq_id", "")),
            "gauge_overlap": round(float(match.get("overlap_m", 0.0)), 3),
            "gauge_dist": round(float(match.get("mean_distance_m", 0.0)), 4),
            "gauge_span": round(float(match.get("gauge_span_m", 0.0)), 3),
            "completion_m": round(completion_m, 3),
            "completion_mode": suffix_mode,
            "completion_offset_shift_m": round(offset_shift, 3),
            "curve_smoothing": smoothing_mode,
            "station_min_m": round(min(stations), 3),
            "station_max_m": round(max(stations), 3),
            "station_span_m": round(max(stations) - min(stations), 3),
            "offset_start_m": round(merged[0][1], 3),
            "offset_end_m": round(merged[-1][1], 3),
            "offset_min_m": round(min(offsets), 3),
            "offset_max_m": round(max(offsets), 3),
        }
    )
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
    }


def interpolate_st(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    step_m: float,
    progress_power: float = 1.0,
) -> list[tuple[float, float]]:
    station0, offset0 = start
    station1, offset1 = end
    span = station1 - station0
    count = max(2, int(math.ceil(abs(span) / max(step_m, 0.1))) + 1)
    points: list[tuple[float, float]] = []
    for index in range(count):
        u = index / (count - 1)
        delayed_u = math.pow(u, max(float(progress_power), 0.25))
        smooth = delayed_u * delayed_u * (3.0 - 2.0 * delayed_u)
        points.append((station0 + span * u, offset0 + (offset1 - offset0) * smooth))
    return points


def smooth_station_offset_curve(
    points: list[tuple[float, float]],
    *,
    step_m: float,
    window_size: int,
    passes: int,
) -> list[tuple[float, float]]:
    if len(points) < 5:
        return points
    points = dedupe_station_points(sorted(points, key=lambda item: item[0]))
    start_s, start_t = points[0]
    end_s, end_t = points[-1]
    span = end_s - start_s
    if span <= 0:
        return points
    count = max(5, int(math.ceil(span / max(step_m, 0.1))) + 1)
    stations = [start_s + span * index / (count - 1) for index in range(count)]
    offsets = [interpolated_offset(points, station) for station in stations]
    half_window = max(1, int(window_size) // 2)
    locked = {0, 1, len(offsets) - 2, len(offsets) - 1}
    for _ in range(max(0, int(passes))):
        previous = offsets[:]
        for index in range(len(offsets)):
            if index in locked:
                continue
            lo = max(0, index - half_window)
            hi = min(len(offsets), index + half_window + 1)
            offsets[index] = sum(previous[lo:hi]) / (hi - lo)
    offsets[0] = start_t
    offsets[-1] = end_t
    return list(zip(stations, offsets))


def interpolated_offset(points: list[tuple[float, float]], station: float) -> float:
    if station <= points[0][0]:
        return points[0][1]
    if station >= points[-1][0]:
        return points[-1][1]
    for (s0, t0), (s1, t1) in zip(points, points[1:]):
        if s0 <= station <= s1:
            if abs(s1 - s0) < 1e-9:
                return t1
            u = (station - s0) / (s1 - s0)
            return t0 + (t1 - t0) * u
    return points[-1][1]


def dedupe_station_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for station, offset in points:
        if deduped and abs(station - deduped[-1][0]) < 1e-6:
            deduped[-1] = (station, offset)
            continue
        deduped.append((station, offset))
    return deduped


def normalize_feature(feature: dict[str, Any], *, source_type: str) -> dict[str, Any]:
    props = dict(feature.get("properties") or {})
    connector_id = str(props.get("connector_id", props.get("conn_id", "branch"))).strip() or "branch"
    anchor_id = str(props.get("anchor_id", ""))
    south_anchor = str(props.get("south_anchor", ""))
    north_anchor = str(props.get("north_anchor", ""))
    if source_type == "crossover_las_endpoint_locked":
        branch_id = connector_id
        anchors = ",".join(item for item in (south_anchor, north_anchor) if item)
        qa_status, self_review_note = SELF_REVIEW.get(branch_id, ("self_review_pending", "pending full-res DOM self review"))
        review_status = "preferred_candidate_self_reviewed"
        source_rank = 1
    else:
        branch_id = anchor_id or connector_id
        anchors = anchor_id
        qa_status, self_review_note = SELF_REVIEW.get(branch_id, ("self_review_pending", "pending full-res DOM self review"))
        review_status = "candidate_self_reviewed" if qa_status.startswith("self_review_pass") else "candidate_needs_visual_review"
        source_rank = 2
    props.update(
        {
            "branch_id": branch_id,
            "connector_id": connector_id,
            "source_type": source_type,
            "source_rank": source_rank,
            "anchors": anchors,
            "qa_status": qa_status,
            "review_status": review_status,
            "review_note": default_review_note(source_type, props),
            "self_note": self_review_note,
        }
    )
    return {
        "type": "Feature",
        "properties": props,
        "geometry": feature["geometry"],
    }


def default_review_note(source_type: str, props: dict[str, Any]) -> str:
    if source_type == "crossover_las_endpoint_locked":
        return "paired crossover; endpoints locked and middle LAS-adjusted"
    if props.get("shape_model") == "curve_straight_reverse":
        return "special curve-straight-reverse template; high-risk visual check"
    return "remaining template branch; use full-res DOM visual check"


def feature_sort_key(feature: dict[str, Any]) -> tuple[int, float, str]:
    props = feature.get("properties") or {}
    station = btc.safe_float(props.get("station_min_m", 0.0))
    return int(btc.safe_float(props.get("source_rank", 9))), station, str(props.get("branch_id", ""))


def summarize_feature(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties") or {}
    return {
        "branch_id": props.get("branch_id", ""),
        "connector_id": props.get("connector_id", ""),
        "source_type": props.get("source_type", ""),
        "anchors": props.get("anchors", ""),
        "pair_id": props.get("pair_id", ""),
        "direction": props.get("direction", ""),
        "shape_model": props.get("shape_model", ""),
        "station_min_m": props.get("station_min_m", 0.0),
        "station_max_m": props.get("station_max_m", 0.0),
        "score": props.get("connector_score", props.get("template_score", 0.0)),
        "support_coverage": props.get("support_cov", 0.0),
        "transition_coverage": props.get("trans_cov", props.get("transition_cov", 0.0)),
        "las_any": props.get("las_any", 0),
        "las_both": props.get("las_both", 0),
        "qa_status": props.get("qa_status", ""),
        "review_note": props.get("review_note", ""),
        "self_note": props.get("self_note", ""),
    }


def write_all_turnout_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("branch_id", "C", size=16)
    writer.field("conn_id", "C", size=16)
    writer.field("src_type", "C", size=32)
    writer.field("anchors", "C", size=32)
    writer.field("pair_id", "C", size=24)
    writer.field("direction", "C", size=64)
    writer.field("shape", "C", size=40)
    writer.field("status", "C", size=40)
    writer.field("qa_status", "C", size=32)
    writer.field("s0_m", "F", decimal=3)
    writer.field("s1_m", "F", decimal=3)
    writer.field("score", "F", decimal=4)
    writer.field("sup_cov", "F", decimal=4)
    writer.field("trans_cov", "F", decimal=4)
    writer.field("las_any", "N", size=8)
    writer.field("las_both", "N", size=8)
    writer.field("max_shift", "F", decimal=4)
    writer.field("note", "C", size=120)
    writer.field("self_note", "C", size=160)
    for feature in features:
        props = feature.get("properties") or {}
        writer.line([btc.line_coords(feature)])
        writer.record(
            str(props.get("branch_id", ""))[:16],
            str(props.get("connector_id", ""))[:16],
            str(props.get("source_type", ""))[:32],
            str(props.get("anchors", ""))[:32],
            str(props.get("pair_id", ""))[:24],
            str(props.get("direction", ""))[:64],
            str(props.get("shape_model", ""))[:40],
            str(props.get("review_status", ""))[:40],
            str(props.get("qa_status", ""))[:32],
            btc.safe_float(props.get("station_min_m", 0.0)),
            btc.safe_float(props.get("station_max_m", 0.0)),
            btc.safe_float(props.get("connector_score", props.get("template_score", 0.0))),
            btc.safe_float(props.get("support_cov", 0.0)),
            btc.safe_float(props.get("trans_cov", props.get("transition_cov", 0.0))),
            int(btc.safe_float(props.get("las_any", 0))),
            int(btc.safe_float(props.get("las_both", 0))),
            btc.safe_float(props.get("max_shift", 0.0)),
            str(props.get("review_note", ""))[:120],
            str(props.get("self_note", ""))[:160],
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    btc.write_projection(output_path.with_suffix(".prj"), epsg)


def write_all_turnout_qml(path: Path) -> None:
    categories = [
        ("crossover_las_endpoint_locked", "0,180,80,255", "crossover LAS endpoint locked"),
        ("remaining_template_or_special", "230,110,0,255", "remaining template or special"),
        ("gauge_pair_evidence_constrained", "255,210,0,255", "gauge-pair evidence constrained"),
    ]
    symbols = []
    cats = []
    for index, (value, color, label) in enumerate(categories):
        symbols.append(
            f"""      <symbol name="{index}" type="line" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleLine" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="capstyle" type="QString" value="round"/>
            <Option name="joinstyle" type="QString" value="round"/>
            <Option name="line_color" type="QString" value="{color}"/>
            <Option name="line_style" type="QString" value="solid"/>
            <Option name="line_width" type="QString" value="0.85"/>
            <Option name="line_width_unit" type="QString" value="MM"/>
          </Option>
          <data_defined_properties><Option type="Map"/></data_defined_properties>
        </layer>
      </symbol>"""
        )
        cats.append(f'      <category value="{value}" symbol="{index}" label="{label}" render="true"/>')
    path.write_text(
        f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="src_type" enableorderby="0" forceraster="0" referencescale="-1" symbollevels="0">
    <categories>
{chr(10).join(cats)}
    </categories>
    <symbols>
{chr(10).join(symbols)}
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


def write_fullres_qa_crops(
    dom_path: Path,
    *,
    branch_features: list[dict[str, Any]],
    band_features: list[dict[str, Any]],
    out_dir: Path,
    bounds_padding_m: float,
    segment_crop_m: float,
    line_width_px: int,
) -> dict[str, Any]:
    import rasterio
    from rasterio.windows import Window, from_bounds

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in list(out_dir.glob("*.png")) + [out_dir / "qa_crops_index.json"]:
        if old_path.exists():
            old_path.unlink()

    paths: list[str] = []
    index_rows: list[dict[str, Any]] = []
    with rasterio.open(dom_path) as dataset:
        for feature in branch_features:
            branch_id = str((feature.get("properties") or {}).get("branch_id", "branch"))
            stem = btc.sanitize_filename(branch_id)
            windows: list[tuple[str, Any]] = []
            bounds_window = feature_bounds_window(dataset, feature, padding_m=bounds_padding_m, from_bounds_func=from_bounds)
            if bounds_window is not None:
                windows.append(("bounds", bounds_window))
            windows.extend(feature_segment_windows(dataset, feature, crop_m=segment_crop_m, window_type=Window))
            for name, window in windows:
                overlay_path = out_dir / f"{stem}_fullres_{name}_overlay.png"
                write_review_crop(
                    dataset,
                    window,
                    feature=feature,
                    band_features=band_features,
                    output_path=overlay_path,
                    label=f"{branch_id} {name}",
                    line_width_px=line_width_px,
                )
                paths.append(str(overlay_path))
                index_rows.append({"branch_id": branch_id, "crop": name, "path": str(overlay_path), "width_px": int(window.width), "height_px": int(window.height)})
    index = {
        "mode": "native_resolution_no_resize",
        "dom_path": str(dom_path),
        "count": len(paths),
        "bounds_padding_m": bounds_padding_m,
        "segment_crop_m": segment_crop_m,
        "line_width_px": line_width_px,
        "overlays": paths,
        "items": index_rows,
    }
    (out_dir / "qa_crops_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def feature_bounds_window(dataset: Any, feature: dict[str, Any], *, padding_m: float, from_bounds_func: Any) -> Any | None:
    coords = btc.line_coords(feature)
    if len(coords) < 2:
        return None
    min_x = min(x for x, _ in coords) - padding_m
    max_x = max(x for x, _ in coords) + padding_m
    min_y = min(y for _, y in coords) - padding_m
    max_y = max(y for _, y in coords) + padding_m
    return clamp_window(dataset, from_bounds_func(min_x, min_y, max_x, max_y, transform=dataset.transform))


def feature_segment_windows(dataset: Any, feature: dict[str, Any], *, crop_m: float, window_type: Any) -> list[tuple[str, Any]]:
    coords = btc.line_coords(feature)
    if len(coords) < 2:
        return []
    picks = [("start", coords[0]), ("middle", coords[len(coords) // 2]), ("end", coords[-1])]
    half = crop_m / 2.0
    windows: list[tuple[str, Any]] = []
    pixel_width = max(abs(float(dataset.transform.a)), 1e-6)
    pixel_height = max(abs(float(dataset.transform.e)), 1e-6)
    half_w = max(16, int(math.ceil(half / pixel_width)))
    half_h = max(16, int(math.ceil(half / pixel_height)))
    for name, (x, y) in picks:
        row, col = dataset.index(x, y)
        window = clamp_window(dataset, window_type(col - half_w, row - half_h, half_w * 2, half_h * 2))
        if window is not None:
            windows.append((name, window))
    return windows


def clamp_window(dataset: Any, window: Any) -> Any | None:
    from rasterio.windows import Window

    col_off = max(0, int(math.floor(window.col_off)))
    row_off = max(0, int(math.floor(window.row_off)))
    col_max = min(dataset.width, int(math.ceil(window.col_off + window.width)))
    row_max = min(dataset.height, int(math.ceil(window.row_off + window.height)))
    width = col_max - col_off
    height = row_max - row_off
    if width <= 1 or height <= 1:
        return None
    return Window(col_off, row_off, width, height)


def write_review_crop(
    dataset: Any,
    window: Any,
    *,
    feature: dict[str, Any],
    band_features: list[dict[str, Any]],
    output_path: Path,
    label: str,
    line_width_px: int,
) -> None:
    from PIL import Image, ImageDraw

    rgb = btc.read_rgb_window(dataset, window)
    overlay = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    transform = dataset.window_transform(window)
    draw_band_features(draw, band_features, transform)
    draw_branch(draw, feature, transform, width_px=line_width_px)
    draw_endpoints(draw, feature, transform)
    draw_small_label(draw, label)
    overlay.convert("RGB").save(output_path)


def draw_band_features(draw: Any, features: list[dict[str, Any]], transform: Any) -> None:
    for feature in features:
        props = feature.get("properties") or {}
        color = btc.BAND_COLORS.get(str(props.get("band_id", "")), (255, 255, 255, 90))
        rgba = (color[0], color[1], color[2], 135)
        coords = []
        for x, y in btc.line_coords(feature):
            col, row = ~transform * (x, y)
            coords.append((col, row))
        if len(coords) >= 2:
            draw.line(coords, fill=rgba, width=2, joint="curve")


def draw_branch(draw: Any, feature: dict[str, Any], transform: Any, *, width_px: int) -> None:
    coords = []
    for x, y in btc.line_coords(feature):
        col, row = ~transform * (x, y)
        coords.append((col, row))
    if len(coords) >= 2:
        draw.line(coords, fill=(0, 255, 70, 255), width=max(1, width_px), joint="curve")


def draw_endpoints(draw: Any, feature: dict[str, Any], transform: Any) -> None:
    coords = btc.line_coords(feature)
    if len(coords) < 2:
        return
    for (x, y), color in ((coords[0], (0, 120, 255, 235)), (coords[-1], (255, 0, 180, 235))):
        col, row = ~transform * (x, y)
        radius = 5
        draw.ellipse([col - radius, row - radius, col + radius, row + radius], fill=color, outline=(255, 255, 255, 255), width=2)


def draw_small_label(draw: Any, label: str) -> None:
    box = [8, 8, 260, 34]
    draw.rectangle(box, fill=(0, 0, 0, 185))
    draw.text((15, 13), label[:32], fill=(255, 255, 255, 255))


def write_visual_qa(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 全道岔支线中心线 QA",
        "",
        "本层是给 QGIS 人工验收前的候选包。裁图来自原始 DOM，按原始像素读取，不做缩略图重采样。",
        "",
        "## 主验收图层",
        "",
        f"- `{Path(summary['outputs']['shp']).name}`：全道岔支线中心线候选。",
        f"- `{Path(summary['outputs']['qml']).name}`：按来源着色；绿色类为成对渡线 LAS 端点锁定版，橙色类为剩余模板/特殊候选。",
        "",
        "## 自检清单",
        "",
        "| branch | source | anchors | score | support | status | note |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in summary.get("features", []):
        lines.append(
            f"| `{row['branch_id']}` | {row['source_type']} | {row['anchors']} | "
            f"{row.get('score', 0.0)} | {row.get('support_coverage', 0.0)} | {row.get('qa_status', '')} | {row.get('self_note', row.get('review_note', ''))} |"
        )
    lines.extend(
        [
            "",
            "## 裁图",
            "",
            f"- 裁图目录：`{summary['outputs'].get('qa_crops')}`",
            "- 每条支线输出 `bounds/start/middle/end` 四类叠加图；绿色线是当前支线中心线，蓝色端点为线起点，粉色端点为线终点。",
            "- 如果 `TA08` 仍表现不稳定，要优先回到原始分割/点云证据做局部重建，不应把它和已通过支线混成同等置信度。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
