from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import build_turnout_connector_candidates as btc


DEFAULT_ANCHORS = Path("data/manual_feedback/turnout_template_anchors.geojson")
DEFAULT_FEEDBACK = Path("data/manual_feedback/turnout_template_feedback.geojson")
DEFAULT_TEMPLATE_PROPOSALS = Path("output/raw_dom_roi_fullpass_v1/turnout_connector_candidates/turnout_connector_proposals.geojson")
DEFAULT_TEMPLATE_ID = "P003"
DEFAULT_MAINLINE = Path("output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_RAW_CANDIDATES = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_candidates/track_centerline_candidates.geojson")
DEFAULT_TRACK_BANDS = Path("output/raw_dom_roi_fullpass_v1/track_band_priors/track_band_centerline_priors.geojson")
DEFAULT_DOM = Path("data/生产数据/无人机数据/正射/dom.tif")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/turnout_template_connectors")
DEFAULT_EPSG = 32651


@dataclass(frozen=True)
class TemplateSample:
    distance_from_main_m: float
    offset_fraction: float


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    length_m: float
    samples: list[TemplateSample]
    source_station_min_m: float
    source_station_max_m: float
    source_offset_start_m: float
    source_offset_end_m: float


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    point: tuple[float, float]
    station: float
    offset: float
    source: str
    note: str


@dataclass(frozen=True)
class RawPoint:
    station: float
    offset: float
    confidence: float


@dataclass(frozen=True)
class CandidateScore:
    score: float
    support_coverage: float
    transition_coverage: float
    mean_distance_m: float
    mean_confidence: float


@dataclass(frozen=True)
class TurnoutFeedback:
    anchor_id: str
    branch_direction: str
    shape_model: str
    note: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project accepted P003 turnout geometry as a reusable template.")
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument("--template-proposals", type=Path, default=DEFAULT_TEMPLATE_PROPOSALS)
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument("--raw-candidates", type=Path, default=DEFAULT_RAW_CANDIDATES)
    parser.add_argument("--track-bands", type=Path, default=DEFAULT_TRACK_BANDS)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--side-offset-m", type=float, default=5.0)
    parser.add_argument("--main-offset-tolerance-m", type=float, default=1.5)
    parser.add_argument("--side-offset-tolerance-m", type=float, default=2.0)
    parser.add_argument("--support-distance-m", type=float, default=1.2)
    parser.add_argument("--support-station-window-m", type=float, default=3.0)
    parser.add_argument("--min-review-score", type=float, default=0.15)
    parser.add_argument("--qa-crop-width-m", type=float, default=155.0)
    parser.add_argument("--qa-crop-height-m", type=float, default=155.0)
    parser.add_argument("--skip-qa-crops", action="store_true")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mainline = btc.load_line_features(args.mainline.expanduser().resolve())[0]
    guide = btc.Guide(btc.line_coords(mainline)[0], btc.line_coords(mainline)[-1])
    template = load_template(args.template_proposals.expanduser().resolve(), template_id=args.template_id, guide=guide)
    anchors = load_anchors(args.anchors.expanduser().resolve(), guide=guide)
    feedback_by_anchor = load_feedback(args.feedback.expanduser().resolve())
    raw_index = build_raw_index(
        btc.load_line_features(args.raw_candidates.expanduser().resolve()),
        guide=guide,
        bin_size_m=args.support_station_window_m,
    )

    best_features: list[dict[str, Any]] = []
    alternative_features: list[dict[str, Any]] = []
    anchor_features = anchor_point_features(anchors, epsg=args.epsg)
    summary_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        feedback = feedback_by_anchor.get(anchor.anchor_id)
        alternatives = build_anchor_alternatives(
            anchor,
            template=template,
            guide=guide,
            raw_index=raw_index,
            side_offset_m=args.side_offset_m,
            main_offset_tolerance_m=args.main_offset_tolerance_m,
            side_offset_tolerance_m=args.side_offset_tolerance_m,
            support_distance_m=args.support_distance_m,
            support_station_window_m=args.support_station_window_m,
        )
        if feedback and feedback.shape_model == "curve_straight_reverse":
            special = build_curve_straight_reverse_candidate(
                anchor,
                feedback=feedback,
                guide=guide,
                raw_index=raw_index,
                side_offset_m=args.side_offset_m,
                total_length_m=template.length_m,
                support_distance_m=args.support_distance_m,
                support_station_window_m=args.support_station_window_m,
            )
            if special is not None:
                alternatives.append(special)
        alternatives.sort(key=lambda feature: btc.safe_float(feature["properties"].get("template_score", 0.0)), reverse=True)
        for rank, feature in enumerate(alternatives, start=1):
            props = feature["properties"]
            props["rank"] = rank
            props["connector_id"] = f"{anchor.anchor_id}_ALT{rank:02d}"
            alternative_features.append(feature)
        if alternatives:
            best_source = select_best_candidate(alternatives, feedback)
            best = clone_feature(best_source)
            best["properties"]["connector_id"] = f"{anchor.anchor_id}_BEST"
            best["properties"]["role"] = "turnout_template_connector_proposal"
            if feedback:
                best["properties"]["feedback_note"] = feedback.note
                best["properties"]["feedback_rule"] = feedback.shape_model
            best["properties"]["review_status"] = (
                "candidate_needs_dom_review"
                if btc.safe_float(best["properties"].get("template_score", 0.0)) >= args.min_review_score
                else "low_support_candidate_needs_dom_review"
            )
            best_features.append(best)
            summary_rows.append(summarize_best(anchor, best, alternatives))
        else:
            summary_rows.append({"anchor_id": anchor.anchor_id, "status": "no_candidate"})

    best_geojson = out_dir / "turnout_template_connector_proposals.geojson"
    alternatives_geojson = out_dir / "turnout_template_connector_alternatives.geojson"
    anchors_geojson = out_dir / "turnout_template_anchors.geojson"
    btc.write_geojson(best_geojson, best_features, epsg=args.epsg)
    btc.write_geojson(alternatives_geojson, alternative_features, epsg=args.epsg)
    btc.write_geojson(anchors_geojson, anchor_features, epsg=args.epsg)

    for path in (best_geojson, alternatives_geojson):
        write_template_connector_shapefile(btc.load_line_features(path), path.with_suffix(".shp"), epsg=args.epsg)
        btc.write_connector_qml(path.with_suffix(".qml"))
    write_anchor_shapefile(anchor_features, anchors_geojson.with_suffix(".shp"), epsg=args.epsg)

    qa_summary = None
    dom_path = args.dom.expanduser().resolve()
    if not args.skip_qa_crops and dom_path.exists():
        band_features = btc.load_line_features(args.track_bands.expanduser().resolve()) if args.track_bands.exists() else []
        qa_summary = btc.write_qa_crops(
            dom_path,
            connector_features=best_features,
            band_features=band_features,
            out_dir=out_dir / "qa_crops",
            crop_width_m=args.qa_crop_width_m,
            crop_height_m=args.qa_crop_height_m,
        )
        overlay_paths = [Path(path) for path in qa_summary.get("overlays", [])]
        write_contact_sheet(overlay_paths, out_dir / "qa_crops" / "_template_contact.png")
        zoom_summary = write_connector_zoom_crops(overlay_paths, out_dir / "qa_crops")
        qa_summary["zoom_contact"] = str(zoom_summary.get("contact_sheet", ""))
        qa_summary["zoom_overlays"] = [str(path) for path in zoom_summary.get("zoom_paths", [])]

    summary = {
        "template_id": template.template_id,
        "template_length_m": round(template.length_m, 3),
        "template_source": str(args.template_proposals.expanduser().resolve()),
        "anchor_count": len(anchors),
        "best_candidate_count": len(best_features),
        "alternative_count": len(alternative_features),
        "policy": {
            "status": "candidate_only_not_final_topology",
            "rule": "P003 geometry is reused as a turnout template, but every generated line still requires DOM/QGIS review.",
            "feedback_rule": "When user visual feedback exists, best-candidate selection must obey branch_direction and shape_model before score.",
            "support_distance_m": args.support_distance_m,
            "support_station_window_m": args.support_station_window_m,
            "min_review_score": args.min_review_score,
        },
        "anchors": [anchor_summary(anchor) for anchor in anchors],
        "feedback": [feedback_summary(item) for item in feedback_by_anchor.values()],
        "best_candidates": summary_rows,
        "outputs": {
            "proposals_geojson": str(best_geojson),
            "proposals_shp": str(best_geojson.with_suffix(".shp")),
            "alternatives_geojson": str(alternatives_geojson),
            "alternatives_shp": str(alternatives_geojson.with_suffix(".shp")),
            "anchors_geojson": str(anchors_geojson),
            "anchors_shp": str(anchors_geojson.with_suffix(".shp")),
            "qa_crops": str(out_dir / "qa_crops") if qa_summary else None,
        },
    }
    if qa_summary is not None:
        summary["qa_crops"] = qa_summary
    btc.write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_template(path: Path, *, template_id: str, guide: btc.Guide) -> TemplateSpec:
    features = btc.load_line_features(path)
    feature = next((item for item in features if str(item.get("properties", {}).get("connector_id", "")) == template_id), None)
    if feature is None:
        raise RuntimeError(f"Template connector {template_id!r} not found in {path}")
    station_offsets = btc.station_offsets_for_feature(feature, guide=guide)
    if len(station_offsets) < 2:
        raise RuntimeError(f"Template connector {template_id!r} has too few points.")
    endpoints = [station_offsets[0], station_offsets[-1]]
    main_endpoint = min(endpoints, key=lambda item: abs(item[1]))
    side_endpoint = max(endpoints, key=lambda item: abs(item[1]))
    length = abs(side_endpoint[0] - main_endpoint[0])
    offset_span = side_endpoint[1] - main_endpoint[1]
    if length <= 0.0 or abs(offset_span) < 1e-6:
        raise RuntimeError(f"Template connector {template_id!r} has invalid geometry.")
    samples = [
        TemplateSample(
            distance_from_main_m=abs(station - main_endpoint[0]),
            offset_fraction=(offset - main_endpoint[1]) / offset_span,
        )
        for station, offset in station_offsets
    ]
    samples.sort(key=lambda item: item.distance_from_main_m)
    return TemplateSpec(
        template_id=template_id,
        length_m=length,
        samples=samples,
        source_station_min_m=min(station for station, _ in station_offsets),
        source_station_max_m=max(station for station, _ in station_offsets),
        source_offset_start_m=station_offsets[0][1],
        source_offset_end_m=station_offsets[-1][1],
    )


def load_anchors(path: Path, *, guide: btc.Guide) -> list[Anchor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors: list[Anchor] = []
    for index, feature in enumerate(payload.get("features", []), start=1):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        point = (float(coordinates[0]), float(coordinates[1]))
        station, offset = guide.station_offset(point)
        props = feature.get("properties") or {}
        anchors.append(
            Anchor(
                anchor_id=str(props.get("anchor_id", f"TA{index:02d}")),
                point=point,
                station=station,
                offset=offset,
                source=str(props.get("source", "user_qgis_tangent_coordinate")),
                note=str(props.get("note", "")),
            )
        )
    return anchors


def load_feedback(path: Path) -> dict[str, TurnoutFeedback]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    feedback: dict[str, TurnoutFeedback] = {}
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        anchor_id = str(props.get("anchor_id", "")).strip()
        if not anchor_id:
            continue
        branch_direction = str(props.get("branch_direction", "")).strip().lower()
        if branch_direction not in {"north", "south", ""}:
            raise ValueError(f"Unsupported branch_direction for {anchor_id}: {branch_direction}")
        feedback[anchor_id] = TurnoutFeedback(
            anchor_id=anchor_id,
            branch_direction=branch_direction,
            shape_model=str(props.get("shape_model", "p003_template")).strip() or "p003_template",
            note=str(props.get("note", "")),
        )
    return feedback


def build_raw_index(features: list[dict[str, Any]], *, guide: btc.Guide, bin_size_m: float) -> dict[int, list[RawPoint]]:
    index: dict[int, list[RawPoint]] = {}
    for feature in features:
        confidence = btc.safe_float((feature.get("properties") or {}).get("mean_confidence", 0.0))
        for coord in btc.line_coords(feature):
            station, offset = guide.station_offset(coord)
            bucket = station_bucket(station, bin_size_m)
            index.setdefault(bucket, []).append(RawPoint(station=station, offset=offset, confidence=confidence))
    return index


def station_bucket(station: float, bin_size_m: float) -> int:
    return int(math.floor(station / max(bin_size_m, 1e-6)))


def build_anchor_alternatives(
    anchor: Anchor,
    *,
    template: TemplateSpec,
    guide: btc.Guide,
    raw_index: dict[int, list[RawPoint]],
    side_offset_m: float,
    main_offset_tolerance_m: float,
    side_offset_tolerance_m: float,
    support_distance_m: float,
    support_station_window_m: float,
) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    for role, side_offsets in endpoint_roles(anchor.offset, side_offset_m, main_offset_tolerance_m, side_offset_tolerance_m):
        for side_offset in side_offsets:
            for orientation in (-1, 1):
                st_points = project_template(template, anchor_station=anchor.station, side_offset=side_offset, orientation=orientation, endpoint_role=role)
                if not st_points or st_points[0][0] < -5.0 or st_points[-1][0] > guide.length + 5.0:
                    continue
                score = score_candidate(
                    st_points,
                    raw_index=raw_index,
                    support_distance_m=support_distance_m,
                    support_station_window_m=support_station_window_m,
                )
                feature = template_feature(anchor, template, st_points, score=score, side_offset=side_offset, orientation=orientation, endpoint_role=role, guide=guide)
                alternatives.append(feature)
    return alternatives


def endpoint_roles(
    anchor_offset: float,
    side_offset_m: float,
    main_offset_tolerance_m: float,
    side_offset_tolerance_m: float,
) -> list[tuple[str, list[float]]]:
    roles: list[tuple[str, list[float]]] = []
    if abs(anchor_offset) <= main_offset_tolerance_m:
        roles.append(("main", [-side_offset_m, side_offset_m]))
    if abs(anchor_offset + side_offset_m) <= side_offset_tolerance_m:
        roles.append(("side", [-side_offset_m]))
    if abs(anchor_offset - side_offset_m) <= side_offset_tolerance_m:
        roles.append(("side", [side_offset_m]))
    return roles or [("main", [-side_offset_m, side_offset_m])]


def project_template(
    template: TemplateSpec,
    *,
    anchor_station: float,
    side_offset: float,
    orientation: int,
    endpoint_role: str,
) -> list[tuple[float, float]]:
    if endpoint_role == "main":
        main_station = anchor_station
    elif endpoint_role == "side":
        main_station = anchor_station - orientation * template.length_m
    else:
        raise ValueError(f"Unknown endpoint role: {endpoint_role}")
    points = [
        (main_station + orientation * sample.distance_from_main_m, side_offset * sample.offset_fraction)
        for sample in template.samples
    ]
    points.sort(key=lambda item: item[0])
    return points


def branch_direction_for_points(st_points: list[tuple[float, float]], anchor_station: float) -> str:
    if not st_points:
        return ""
    mid_station = sum(station for station, _ in st_points) / len(st_points)
    return "north" if mid_station > anchor_station else "south"


def select_best_candidate(alternatives: list[dict[str, Any]], feedback: TurnoutFeedback | None) -> dict[str, Any]:
    candidates = alternatives
    if feedback and feedback.branch_direction:
        directed = [
            feature
            for feature in candidates
            if str(feature.get("properties", {}).get("branch_dir", "")) == feedback.branch_direction
        ]
        if directed:
            candidates = directed
    if feedback and feedback.shape_model != "p003_template":
        shaped = [
            feature
            for feature in candidates
            if str(feature.get("properties", {}).get("shape_model", "")) == feedback.shape_model
        ]
        if shaped:
            candidates = shaped
    return max(candidates, key=lambda feature: btc.safe_float(feature["properties"].get("template_score", 0.0)))


def build_curve_straight_reverse_candidate(
    anchor: Anchor,
    *,
    feedback: TurnoutFeedback,
    guide: btc.Guide,
    raw_index: dict[int, list[RawPoint]],
    side_offset_m: float,
    total_length_m: float,
    support_distance_m: float,
    support_station_window_m: float,
) -> dict[str, Any] | None:
    if feedback.branch_direction not in {"north", "south"}:
        return None
    sign = 1.0 if feedback.branch_direction == "north" else -1.0
    side_offset = side_offset_m
    sample_step_m = 1.0
    st_points = curve_straight_reverse_points(
        anchor_station=anchor.station,
        side_offset=side_offset,
        direction_sign=sign,
        total_length_m=total_length_m,
        sample_step_m=sample_step_m,
    )
    if st_points[0][0] < -5.0 or st_points[-1][0] > guide.length + 5.0:
        return None
    score = score_candidate(
        st_points,
        raw_index=raw_index,
        support_distance_m=support_distance_m,
        support_station_window_m=support_station_window_m,
    )
    return template_feature(
        anchor,
        TemplateSpec(
            template_id="curve_straight_reverse",
            length_m=total_length_m,
            samples=[],
            source_station_min_m=0.0,
            source_station_max_m=total_length_m,
            source_offset_start_m=0.0,
            source_offset_end_m=side_offset,
        ),
        st_points,
        score=score,
        side_offset=side_offset,
        orientation=1 if sign > 0 else -1,
        endpoint_role="main",
        guide=guide,
        shape_model="curve_straight_reverse",
        qa_note_override="feedback_curve_straight_reverse_check_dom",
    )


def curve_straight_reverse_points(
    *,
    anchor_station: float,
    side_offset: float,
    direction_sign: float,
    total_length_m: float,
    sample_step_m: float,
) -> list[tuple[float, float]]:
    curve_length = total_length_m * 0.24
    samples = max(8, int(math.ceil(total_length_m / max(sample_step_m, 0.1))) + 1)
    slope = side_offset / max(total_length_m - curve_length, 1e-6)
    points_from_main: list[tuple[float, float]] = []
    offset = 0.0
    previous_d = 0.0
    previous_factor = 0.0
    for index in range(samples):
        d = min(total_length_m, index * total_length_m / (samples - 1))
        factor = curve_straight_reverse_slope_factor(d, total_length_m=total_length_m, curve_length=curve_length)
        if index > 0:
            offset += slope * (d - previous_d) * (previous_factor + factor) * 0.5
        points_from_main.append((anchor_station + direction_sign * d, offset))
        previous_d = d
        previous_factor = factor
    final_offset = points_from_main[-1][1]
    if abs(final_offset) > 1e-9:
        scale = side_offset / final_offset
        points_from_main = [(station, offset * scale) for station, offset in points_from_main]
    points_from_main[-1] = (points_from_main[-1][0], side_offset)
    return sorted(points_from_main, key=lambda item: item[0])


def curve_straight_reverse_slope_factor(d: float, *, total_length_m: float, curve_length: float) -> float:
    if d <= curve_length:
        return smoothstep(d / max(curve_length, 1e-6))
    if d >= total_length_m - curve_length:
        return smoothstep((total_length_m - d) / max(curve_length, 1e-6))
    return 1.0


def smoothstep(value: float) -> float:
    x = max(0.0, min(1.0, value))
    return x * x * (3.0 - 2.0 * x)


def score_candidate(
    st_points: list[tuple[float, float]],
    *,
    raw_index: dict[int, list[RawPoint]],
    support_distance_m: float,
    support_station_window_m: float,
) -> CandidateScore:
    samples = st_points[:: max(1, len(st_points) // 80)]
    supported = 0
    transition_total = 0
    transition_supported = 0
    distance_sum = 0.0
    confidence_sum = 0.0
    for station, offset in samples:
        is_transition_sample = 1.0 < abs(offset) < 4.0
        if is_transition_sample:
            transition_total += 1
        best_distance = float("inf")
        best_confidence = 0.0
        bucket = station_bucket(station, support_station_window_m)
        for nearby_bucket in range(bucket - 1, bucket + 2):
            for raw_point in raw_index.get(nearby_bucket, []):
                if abs(raw_point.station - station) > support_station_window_m:
                    continue
                distance = math.hypot(raw_point.station - station, raw_point.offset - offset)
                if distance < best_distance:
                    best_distance = distance
                    best_confidence = raw_point.confidence
        if best_distance <= support_distance_m:
            supported += 1
            if is_transition_sample:
                transition_supported += 1
            distance_sum += best_distance
            confidence_sum += best_confidence
    coverage = supported / max(len(samples), 1)
    transition_coverage = transition_supported / transition_total if transition_total else coverage
    mean_distance = distance_sum / supported if supported else support_distance_m * 2.5
    mean_confidence = confidence_sum / supported if supported else 0.0
    score = (
        coverage * 0.45
        + transition_coverage * 0.45
        + min(mean_confidence, 1.0) * 0.15
        - min(mean_distance / max(support_distance_m, 1e-6), 1.0) * 0.15
    )
    return CandidateScore(
        score=round(score, 4),
        support_coverage=round(coverage, 4),
        transition_coverage=round(transition_coverage, 4),
        mean_distance_m=round(mean_distance, 4),
        mean_confidence=round(mean_confidence, 4),
    )


def template_feature(
    anchor: Anchor,
    template: TemplateSpec,
    st_points: list[tuple[float, float]],
    *,
    score: CandidateScore,
    side_offset: float,
    orientation: int,
    endpoint_role: str,
    guide: btc.Guide,
    shape_model: str = "p003_template",
    qa_note_override: str | None = None,
) -> dict[str, Any]:
    coords = [guide.point_at(station, offset) for station, offset in st_points]
    pair_id = "minus_to_main" if side_offset < 0.0 else "main_to_plus"
    side_band = "parallel_minus_5m" if side_offset < 0.0 else "parallel_plus_5m"
    if orientation < 0:
        direction = f"{side_band}->mainline_2_track"
    else:
        direction = f"mainline_2_track->{side_band}"
    qa_note = "template_candidate_check_dom"
    if score.score < 0.3:
        qa_note = "low_raw_support_template_candidate"
    if qa_note_override:
        qa_note = qa_note_override
    branch_dir = branch_direction_for_points(st_points, anchor.station)
    props = {
        "role": "turnout_template_connector_proposal",
        "geom_kind": "piecewise_curve_straight_reverse_proposal" if shape_model == "curve_straight_reverse" else "template_curve_proposal",
        "pair_id": pair_id,
        "direction": direction,
        "source": "user_feedback_piecewise_curve_straight_reverse" if shape_model == "curve_straight_reverse" else "accepted_p003_template_projection",
        "review_status": "candidate_needs_dom_review",
        "qa_note": qa_note,
        "template_id": template.template_id,
        "anchor_id": anchor.anchor_id,
        "endpoint_role": endpoint_role,
        "orient": "side_before_main" if orientation < 0 else "side_after_main",
        "branch_dir": branch_dir,
        "shape_model": shape_model,
        "side_offset_m": round(side_offset, 3),
        "station_min_m": round(st_points[0][0], 3),
        "station_max_m": round(st_points[-1][0], 3),
        "station_span_m": round(st_points[-1][0] - st_points[0][0], 3),
        "offset_start_m": round(st_points[0][1], 3),
        "offset_end_m": round(st_points[-1][1], 3),
        "offset_min_m": round(min(offset for _, offset in st_points), 3),
        "offset_max_m": round(max(offset for _, offset in st_points), 3),
        "anchor_s_m": round(anchor.station, 3),
        "anchor_t_m": round(anchor.offset, 3),
        "template_score": score.score,
        "connector_score": score.score,
        "support_cov": score.support_coverage,
        "transition_cov": score.transition_coverage,
        "support_dist": score.mean_distance_m,
        "support_conf": score.mean_confidence,
        "mean_confidence": score.mean_confidence,
    }
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
    }


def anchor_point_features(anchors: list[Anchor], *, epsg: int) -> list[dict[str, Any]]:
    return [
        {
            "type": "Feature",
            "properties": {
                "anchor_id": anchor.anchor_id,
                "source": anchor.source,
                "station_m": round(anchor.station, 3),
                "offset_m": round(anchor.offset, 3),
                "note": anchor.note,
                "epsg": epsg,
            },
            "geometry": {"type": "Point", "coordinates": [round(anchor.point[0], 6), round(anchor.point[1], 6)]},
        }
        for anchor in anchors
    ]


def write_anchor_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pyshp in the active virtual environment.") from exc
    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POINT, encoding="utf-8")
    writer.field("anchor_id", "C", size=16)
    writer.field("source", "C", size=48)
    writer.field("station_m", "F", decimal=3)
    writer.field("offset_m", "F", decimal=3)
    writer.field("note", "C", size=80)
    for feature in features:
        props = feature.get("properties") or {}
        x, y = feature["geometry"]["coordinates"][:2]
        writer.point(float(x), float(y))
        writer.record(
            str(props.get("anchor_id", ""))[:16],
            str(props.get("source", ""))[:48],
            btc.safe_float(props.get("station_m", 0.0)),
            btc.safe_float(props.get("offset_m", 0.0)),
            str(props.get("note", ""))[:80],
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    btc.write_projection(output_path.with_suffix(".prj"), epsg)


def write_template_connector_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pyshp in the active virtual environment.") from exc
    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("conn_id", "C", size=16)
    writer.field("role", "C", size=32)
    writer.field("kind", "C", size=32)
    writer.field("pair_id", "C", size=24)
    writer.field("direction", "C", size=48)
    writer.field("status", "C", size=32)
    writer.field("anchor_id", "C", size=16)
    writer.field("tmpl_id", "C", size=16)
    writer.field("end_role", "C", size=16)
    writer.field("orient", "C", size=20)
    writer.field("branch", "C", size=8)
    writer.field("shape", "C", size=40)
    writer.field("s0_m", "F", decimal=3)
    writer.field("s1_m", "F", decimal=3)
    writer.field("t0_m", "F", decimal=3)
    writer.field("t1_m", "F", decimal=3)
    writer.field("anchor_s", "F", decimal=3)
    writer.field("anchor_t", "F", decimal=3)
    writer.field("side_m", "F", decimal=3)
    writer.field("score", "F", decimal=4)
    writer.field("sup_cov", "F", decimal=4)
    writer.field("trans_cov", "F", decimal=4)
    writer.field("sup_dist", "F", decimal=4)
    writer.field("sup_conf", "F", decimal=4)
    writer.field("qa_note", "C", size=80)
    for index, feature in enumerate(features):
        props = feature.get("properties") or {}
        coords = btc.line_coords(feature)
        writer.line([coords])
        writer.record(
            str(props.get("connector_id", f"T{index + 1:03d}"))[:16],
            str(props.get("role", ""))[:32],
            str(props.get("geom_kind", ""))[:32],
            str(props.get("pair_id", ""))[:24],
            str(props.get("direction", ""))[:48],
            str(props.get("review_status", ""))[:32],
            str(props.get("anchor_id", ""))[:16],
            str(props.get("template_id", ""))[:16],
            str(props.get("endpoint_role", ""))[:16],
            str(props.get("orient", ""))[:20],
            str(props.get("branch_dir", ""))[:8],
            str(props.get("shape_model", ""))[:40],
            btc.safe_float(props.get("station_min_m", 0.0)),
            btc.safe_float(props.get("station_max_m", 0.0)),
            btc.safe_float(props.get("offset_start_m", props.get("offset_min_m", 0.0))),
            btc.safe_float(props.get("offset_end_m", props.get("offset_max_m", 0.0))),
            btc.safe_float(props.get("anchor_s_m", 0.0)),
            btc.safe_float(props.get("anchor_t_m", 0.0)),
            btc.safe_float(props.get("side_offset_m", 0.0)),
            btc.safe_float(props.get("template_score", props.get("connector_score", 0.0))),
            btc.safe_float(props.get("support_cov", 0.0)),
            btc.safe_float(props.get("transition_cov", 0.0)),
            btc.safe_float(props.get("support_dist", 0.0)),
            btc.safe_float(props.get("support_conf", 0.0)),
            str(props.get("qa_note", ""))[:80],
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    btc.write_projection(output_path.with_suffix(".prj"), epsg)


def write_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    from PIL import Image, ImageDraw

    thumbs: list[tuple[Path, Image.Image]] = []
    for path in paths:
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        image.thumbnail((620, 620))
        thumbs.append((path, image.copy()))
        image.close()
    if not thumbs:
        return
    columns = 3
    rows = math.ceil(len(thumbs) / columns)
    cell_w, cell_h = 640, 680
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for index, (path, image) in enumerate(thumbs):
        col = index % columns
        row = index // columns
        x = col * cell_w + 10
        y = row * cell_h + 48
        draw.text((col * cell_w + 10, row * cell_h + 12), path.stem.replace("_overlay", ""), fill=(255, 255, 255))
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def write_connector_zoom_crops(overlay_paths: list[Path], out_dir: Path) -> dict[str, Any]:
    if not overlay_paths:
        return {"zoom_paths": [], "contact_sheet": ""}
    from PIL import Image, ImageDraw
    import numpy as np

    connector_colors = np.array([(0, 158, 115), (213, 94, 0)], dtype=np.int16)
    zoom_paths: list[Path] = []
    for path in overlay_paths:
        image = Image.open(path).convert("RGB")
        arr = np.asarray(image, dtype=np.int16)
        color_distance = np.abs(arr[:, :, None, :] - connector_colors[None, None, :, :]).max(axis=3).min(axis=2)
        rows, cols = np.where(color_distance <= 6)
        if len(rows) == 0:
            continue
        pad = 360
        left = max(0, int(cols.min()) - pad)
        right = min(image.width, int(cols.max()) + pad)
        top = max(0, int(rows.min()) - pad)
        bottom = min(image.height, int(rows.max()) + pad)
        crop = image.crop((left, top, right, bottom))
        max_side = 1800
        scale = min(1.0, max_side / max(crop.size))
        if scale < 1.0:
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
        output_path = out_dir / f"{path.stem.replace('_overlay', '')}_connector_zoom.png"
        crop.save(output_path)
        image.close()
        zoom_paths.append(output_path)

    contact_sheet = out_dir / "_connector_zoom_contact.png"
    if zoom_paths:
        columns = 3
        cell_w, cell_h = 740, 900
        rows_count = math.ceil(len(zoom_paths) / columns)
        sheet = Image.new("RGB", (columns * cell_w, rows_count * cell_h), (18, 18, 18))
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(zoom_paths):
            image = Image.open(path).convert("RGB")
            image.thumbnail((cell_w - 20, cell_h - 60), Image.Resampling.LANCZOS)
            col = index % columns
            row = index // columns
            x = col * cell_w + 10
            y = row * cell_h + 48
            draw.text((col * cell_w + 10, row * cell_h + 16), path.stem.replace("_connector_zoom", ""), fill=(255, 255, 255))
            sheet.paste(image, (x, y))
            image.close()
        sheet.save(contact_sheet)
    return {"zoom_paths": zoom_paths, "contact_sheet": contact_sheet if zoom_paths else ""}


def clone_feature(feature: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(feature, ensure_ascii=False))


def anchor_summary(anchor: Anchor) -> dict[str, Any]:
    return {
        "anchor_id": anchor.anchor_id,
        "x": round(anchor.point[0], 3),
        "y": round(anchor.point[1], 3),
        "station_m": round(anchor.station, 3),
        "offset_m": round(anchor.offset, 3),
    }


def feedback_summary(feedback: TurnoutFeedback) -> dict[str, Any]:
    return {
        "anchor_id": feedback.anchor_id,
        "branch_direction": feedback.branch_direction,
        "shape_model": feedback.shape_model,
        "note": feedback.note,
    }


def summarize_best(anchor: Anchor, best: dict[str, Any], alternatives: list[dict[str, Any]]) -> dict[str, Any]:
    props = best.get("properties") or {}
    return {
        "anchor_id": anchor.anchor_id,
        "station_m": round(anchor.station, 3),
        "offset_m": round(anchor.offset, 3),
        "best_connector_id": props.get("connector_id", ""),
        "pair_id": props.get("pair_id", ""),
        "direction": props.get("direction", ""),
        "endpoint_role": props.get("endpoint_role", ""),
        "orientation": props.get("orient", ""),
        "branch_direction": props.get("branch_dir", ""),
        "shape_model": props.get("shape_model", ""),
        "station_min_m": props.get("station_min_m", 0.0),
        "station_max_m": props.get("station_max_m", 0.0),
        "template_score": props.get("template_score", 0.0),
        "support_coverage": props.get("support_cov", 0.0),
        "transition_coverage": props.get("transition_cov", 0.0),
        "support_distance_m": props.get("support_dist", 0.0),
        "alternative_count": len(alternatives),
        "qa_note": props.get("qa_note", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())
