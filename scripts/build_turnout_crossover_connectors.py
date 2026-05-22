from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import build_turnout_connector_candidates as btc
import build_turnout_template_connectors as btt


DEFAULT_PAIRS = Path("data/manual_feedback/turnout_crossover_pairs.geojson")
DEFAULT_MAINLINE = Path("output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_RAW_CANDIDATES = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_candidates/track_centerline_candidates.geojson")
DEFAULT_TRACK_BANDS = Path("output/raw_dom_roi_fullpass_v1/track_band_priors/track_band_centerline_priors.geojson")
DEFAULT_DOM = Path("data/生产数据/无人机数据/正射/dom.tif")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/turnout_crossover_connectors")
DEFAULT_EPSG = 32651


@dataclass(frozen=True)
class RawPoint:
    station: float
    offset: float
    confidence: float


@dataclass(frozen=True)
class RawEvidence:
    evidence_id: str
    points: list[RawPoint]
    station_min: float
    station_max: float
    offset_min: float
    offset_max: float
    station_span: float
    offset_span: float
    slope_dt_ds: float
    mean_confidence: float


@dataclass(frozen=True)
class CrossoverPair:
    crossover_id: str
    south_anchor_id: str
    north_anchor_id: str
    south_role: str
    north_role: str
    south_point: tuple[float, float]
    north_point: tuple[float, float]
    south_station: float
    south_offset: float
    north_station: float
    north_offset: float
    note: str


@dataclass(frozen=True)
class CrossoverScore:
    score: float
    support_coverage: float
    transition_coverage: float
    mean_distance_m: float
    mean_confidence: float
    evidence_points: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paired turnout crossover connector candidates.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument("--raw-candidates", type=Path, default=DEFAULT_RAW_CANDIDATES)
    parser.add_argument("--track-bands", type=Path, default=DEFAULT_TRACK_BANDS)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-step-m", type=float, default=0.75)
    parser.add_argument("--evidence-window-m", type=float, default=2.5)
    parser.add_argument("--evidence-corridor-m", type=float, default=1.35)
    parser.add_argument("--transition-station-pad-m", type=float, default=10.0)
    parser.add_argument("--transition-offset-pad-m", type=float, default=0.75)
    parser.add_argument("--min-transition-station-span-m", type=float, default=4.0)
    parser.add_argument("--min-transition-offset-span-m", type=float, default=0.35)
    parser.add_argument("--min-transition-slope", type=float, default=0.025)
    parser.add_argument("--max-transition-slope", type=float, default=0.25)
    parser.add_argument("--support-distance-m", type=float, default=1.2)
    parser.add_argument("--support-station-window-m", type=float, default=3.0)
    parser.add_argument("--qa-crop-width-m", type=float, default=130.0)
    parser.add_argument("--qa-crop-height-m", type=float, default=170.0)
    parser.add_argument("--qa-bounds-padding-m", type=float, default=9.0)
    parser.add_argument("--qa-segment-crop-m", type=float, default=32.0)
    parser.add_argument("--qa-line-width-px", type=int, default=3)
    parser.add_argument("--skip-qa-crops", action="store_true")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mainline = btc.load_line_features(args.mainline.expanduser().resolve())[0]
    guide = btc.Guide(btc.line_coords(mainline)[0], btc.line_coords(mainline)[-1])
    pairs = load_crossover_pairs(args.pairs.expanduser().resolve(), guide=guide)
    raw_features = btc.load_line_features(args.raw_candidates.expanduser().resolve())
    raw_points = raw_points_from_features(raw_features, guide=guide)
    raw_evidence = load_raw_evidence(raw_features, guide=guide)

    features: list[dict[str, Any]] = []
    summary_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        feature = build_crossover_feature(
            pair,
            raw_points=raw_points,
            raw_evidence=raw_evidence,
            guide=guide,
            sample_step_m=args.sample_step_m,
            evidence_window_m=args.evidence_window_m,
            evidence_corridor_m=args.evidence_corridor_m,
            transition_station_pad_m=args.transition_station_pad_m,
            transition_offset_pad_m=args.transition_offset_pad_m,
            min_transition_station_span_m=args.min_transition_station_span_m,
            min_transition_offset_span_m=args.min_transition_offset_span_m,
            min_transition_slope=args.min_transition_slope,
            max_transition_slope=args.max_transition_slope,
            support_distance_m=args.support_distance_m,
            support_station_window_m=args.support_station_window_m,
        )
        features.append(feature)
        summary_pairs.append(summarize_feature(feature))

    geojson_path = out_dir / "turnout_crossover_connector_proposals.geojson"
    btc.write_geojson(geojson_path, features, epsg=args.epsg)
    write_crossover_shapefile(features, geojson_path.with_suffix(".shp"), epsg=args.epsg)
    btc.write_connector_qml(geojson_path.with_suffix(".qml"))

    pairs_geojson = out_dir / "turnout_crossover_pairs.geojson"
    btc.write_geojson(pairs_geojson, pair_features(pairs), epsg=args.epsg)
    write_pair_shapefile(pair_features(pairs), pairs_geojson.with_suffix(".shp"), epsg=args.epsg)

    qa_summary = None
    dom_path = args.dom.expanduser().resolve()
    if not args.skip_qa_crops and dom_path.exists():
        band_features = btc.load_line_features(args.track_bands.expanduser().resolve()) if args.track_bands.exists() else []
        qa_summary = btc.write_qa_crops(
            dom_path,
            connector_features=features,
            band_features=band_features,
            out_dir=out_dir / "qa_crops",
            crop_width_m=args.qa_crop_width_m,
            crop_height_m=args.qa_crop_height_m,
        )
        overlay_paths = [Path(path) for path in qa_summary.get("overlays", [])]
        btt.write_contact_sheet(overlay_paths, out_dir / "qa_crops" / "_crossover_contact.png")
        zoom_summary = btt.write_connector_zoom_crops(overlay_paths, out_dir / "qa_crops")
        qa_summary["zoom_contact"] = str(zoom_summary.get("contact_sheet", ""))
        qa_summary["zoom_overlays"] = [str(path) for path in zoom_summary.get("zoom_paths", [])]
        fullres_summary = write_fullres_review_crops(
            dom_path,
            connector_features=features,
            band_features=band_features,
            out_dir=out_dir / "qa_crops",
            bounds_padding_m=args.qa_bounds_padding_m,
            segment_crop_m=args.qa_segment_crop_m,
            line_width_px=args.qa_line_width_px,
        )
        qa_summary["fullres_review"] = fullres_summary

    summary = {
        "pair_count": len(pairs),
        "candidate_count": len(features),
        "policy": {
            "status": "candidate_only_not_final_topology",
            "rule": "Each paired turnout crossover is generated as one shared connector between two anchors, not as two independent copied P003 templates.",
            "geometry": "curve_straight_curve_baseline_with_raw_dom_transition_evidence_fit",
            "transition_evidence": "feature_level_filter_by_pair_station_offset_and_slope_before_point_fitting",
            "support_distance_m": args.support_distance_m,
            "support_station_window_m": args.support_station_window_m,
        },
        "pairs": summary_pairs,
        "outputs": {
            "proposals_geojson": str(geojson_path),
            "proposals_shp": str(geojson_path.with_suffix(".shp")),
            "pairs_geojson": str(pairs_geojson),
            "pairs_shp": str(pairs_geojson.with_suffix(".shp")),
            "qa_crops": str(out_dir / "qa_crops"),
        },
        "qa_crops": qa_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_crossover_pairs(path: Path, *, guide: btc.Guide) -> list[CrossoverPair]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[CrossoverPair] = []
    for index, feature in enumerate(payload.get("features", []), start=1):
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue
        first = (float(coords[0][0]), float(coords[0][1]))
        second = (float(coords[-1][0]), float(coords[-1][1]))
        first_station, first_offset = guide.station_offset(first)
        second_station, second_offset = guide.station_offset(second)
        if first_station <= second_station:
            south_point, north_point = first, second
            south_station, south_offset = first_station, first_offset
            north_station, north_offset = second_station, second_offset
            south_anchor_id = str(props.get("south_anchor_id", ""))
            north_anchor_id = str(props.get("north_anchor_id", ""))
            south_role = str(props.get("south_role", ""))
            north_role = str(props.get("north_role", ""))
        else:
            south_point, north_point = second, first
            south_station, south_offset = second_station, second_offset
            north_station, north_offset = first_station, first_offset
            south_anchor_id = str(props.get("north_anchor_id", ""))
            north_anchor_id = str(props.get("south_anchor_id", ""))
            south_role = str(props.get("north_role", ""))
            north_role = str(props.get("south_role", ""))
        pairs.append(
            CrossoverPair(
                crossover_id=str(props.get("crossover_id", f"CX{index:02d}")),
                south_anchor_id=south_anchor_id,
                north_anchor_id=north_anchor_id,
                south_role=south_role,
                north_role=north_role,
                south_point=south_point,
                north_point=north_point,
                south_station=south_station,
                south_offset=south_offset,
                north_station=north_station,
                north_offset=north_offset,
                note=str(props.get("note", "")),
            )
        )
    return pairs


def load_raw_points(path: Path, *, guide: btc.Guide) -> list[RawPoint]:
    return raw_points_from_features(btc.load_line_features(path), guide=guide)


def raw_points_from_features(features: list[dict[str, Any]], *, guide: btc.Guide) -> list[RawPoint]:
    points: list[RawPoint] = []
    for feature in features:
        confidence = btc.safe_float((feature.get("properties") or {}).get("mean_confidence", 0.0))
        for coord in btc.line_coords(feature):
            station, offset = guide.station_offset(coord)
            points.append(RawPoint(station=station, offset=offset, confidence=confidence))
    points.sort(key=lambda item: item.station)
    return points


def load_raw_evidence(features: list[dict[str, Any]], *, guide: btc.Guide) -> list[RawEvidence]:
    evidence: list[RawEvidence] = []
    for index, feature in enumerate(features, start=1):
        coords = btc.line_coords(feature)
        if len(coords) < 2:
            continue
        confidence = btc.safe_float((feature.get("properties") or {}).get("mean_confidence", 0.0))
        points = []
        for coord in coords:
            station, offset = guide.station_offset(coord)
            points.append(RawPoint(station=station, offset=offset, confidence=confidence))
        points.sort(key=lambda item: item.station)
        station_values = [point.station for point in points]
        offset_values = [point.offset for point in points]
        station_span = max(station_values) - min(station_values)
        offset_span = max(offset_values) - min(offset_values)
        if station_span <= 1e-6:
            continue
        slope = (points[-1].offset - points[0].offset) / (points[-1].station - points[0].station)
        props = feature.get("properties") or {}
        evidence.append(
            RawEvidence(
                evidence_id=f"R{index:03d}_C{props.get('candidate_id', index)}",
                points=points,
                station_min=min(station_values),
                station_max=max(station_values),
                offset_min=min(offset_values),
                offset_max=max(offset_values),
                station_span=station_span,
                offset_span=offset_span,
                slope_dt_ds=slope,
                mean_confidence=confidence,
            )
        )
    evidence.sort(key=lambda item: item.station_min)
    return evidence


def build_crossover_feature(
    pair: CrossoverPair,
    *,
    raw_points: list[RawPoint],
    raw_evidence: list[RawEvidence],
    guide: btc.Guide,
    sample_step_m: float,
    evidence_window_m: float,
    evidence_corridor_m: float,
    transition_station_pad_m: float,
    transition_offset_pad_m: float,
    min_transition_station_span_m: float,
    min_transition_offset_span_m: float,
    min_transition_slope: float,
    max_transition_slope: float,
    support_distance_m: float,
    support_station_window_m: float,
) -> dict[str, Any]:
    selected_evidence = select_transition_evidence(
        pair,
        raw_evidence=raw_evidence,
        station_pad_m=transition_station_pad_m,
        offset_pad_m=transition_offset_pad_m,
        min_station_span_m=min_transition_station_span_m,
        min_offset_span_m=min_transition_offset_span_m,
        min_abs_slope=min_transition_slope,
        max_abs_slope=max_transition_slope,
    )
    transition_points = transition_points_from_evidence(
        pair,
        selected_evidence,
        station_pad_m=transition_station_pad_m,
        offset_pad_m=transition_offset_pad_m,
    )
    sample_st = fit_crossover_points(
        pair,
        transition_points=transition_points,
        sample_step_m=sample_step_m,
        evidence_window_m=evidence_window_m,
        evidence_corridor_m=evidence_corridor_m,
    )
    score = score_crossover(
        sample_st,
        raw_points=raw_points,
        support_distance_m=support_distance_m,
        support_station_window_m=support_station_window_m,
    )
    coords = [guide.point_at(station, offset) for station, offset in sample_st]
    props = {
        "role": "turnout_connector_proposal",
        "geom_kind": "crossover_evidence_fit_proposal",
        "connector_id": pair.crossover_id,
        "pair_id": "minus_to_main",
        "direction": f"{pair.south_role}->{pair.north_role}",
        "source": "paired_crossover_feedback_raw_dom_fit",
        "review_status": "candidate_needs_dom_review",
        "qa_note": "paired_crossover_check_dom_fullres",
        "shape_model": "curve_straight_curve_raw_dom_fit",
        "south_anchor": pair.south_anchor_id,
        "north_anchor": pair.north_anchor_id,
        "south_role": pair.south_role,
        "north_role": pair.north_role,
        "station_min_m": round(pair.south_station, 3),
        "station_max_m": round(pair.north_station, 3),
        "station_span_m": round(pair.north_station - pair.south_station, 3),
        "offset_start_m": round(pair.south_offset, 3),
        "offset_end_m": round(pair.north_offset, 3),
        "offset_span_m": round(pair.north_offset - pair.south_offset, 3),
        "connector_score": score.score,
        "support_cov": score.support_coverage,
        "trans_cov": score.transition_coverage,
        "support_dist": score.mean_distance_m,
        "support_conf": score.mean_confidence,
        "evidence_n": score.evidence_points,
        "trans_feat_n": len(selected_evidence),
        "trans_pt_n": len(transition_points),
        "trans_ids": ",".join(item.evidence_id for item in selected_evidence)[:180],
        "note": pair.note,
    }
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
    }


def fit_crossover_points(
    pair: CrossoverPair,
    *,
    transition_points: list[RawPoint],
    sample_step_m: float,
    evidence_window_m: float,
    evidence_corridor_m: float,
) -> list[tuple[float, float]]:
    span = max(pair.north_station - pair.south_station, sample_step_m)
    count = max(12, int(math.ceil(span / max(sample_step_m, 0.1))) + 1)
    samples: list[tuple[float, float]] = []
    for index in range(count):
        u = index / (count - 1)
        station = pair.south_station + span * u
        baseline = curve_straight_curve_baseline(pair.south_offset, pair.north_offset, u)
        evidence = nearby_evidence_offset(
            station,
            baseline,
            raw_points=transition_points,
            station_window_m=evidence_window_m,
            offset_window_m=evidence_corridor_m,
        )
        taper = endpoint_taper(u)
        if evidence is None:
            offset = baseline
        else:
            evidence_offset, evidence_weight = evidence
            weight = min(0.88, evidence_weight) * taper
            offset = baseline * (1.0 - weight) + evidence_offset * weight
        samples.append((station, offset))
    samples = smooth_offsets(samples, passes=2)
    samples = enforce_monotonic_offsets(samples, increasing=pair.north_offset > pair.south_offset)
    samples[0] = (pair.south_station, pair.south_offset)
    samples[-1] = (pair.north_station, pair.north_offset)
    return samples


def select_transition_evidence(
    pair: CrossoverPair,
    *,
    raw_evidence: list[RawEvidence],
    station_pad_m: float,
    offset_pad_m: float,
    min_station_span_m: float,
    min_offset_span_m: float,
    min_abs_slope: float,
    max_abs_slope: float,
) -> list[RawEvidence]:
    expected_sign = 1.0 if pair.north_offset > pair.south_offset else -1.0
    low_offset = min(pair.south_offset, pair.north_offset) - offset_pad_m
    high_offset = max(pair.south_offset, pair.north_offset) + offset_pad_m
    selected: list[RawEvidence] = []
    for evidence in raw_evidence:
        if evidence.station_max < pair.south_station - station_pad_m:
            continue
        if evidence.station_min > pair.north_station + station_pad_m:
            break
        if evidence.offset_max < low_offset or evidence.offset_min > high_offset:
            continue
        if evidence.station_span < min_station_span_m or evidence.offset_span < min_offset_span_m:
            continue
        abs_slope = abs(evidence.slope_dt_ds)
        if abs_slope < min_abs_slope or abs_slope > max_abs_slope:
            continue
        if evidence.slope_dt_ds * expected_sign <= 0.0:
            continue
        if not has_between_track_points(pair, evidence, station_pad_m=station_pad_m):
            continue
        selected.append(evidence)
    return selected


def has_between_track_points(pair: CrossoverPair, evidence: RawEvidence, *, station_pad_m: float) -> bool:
    low_offset = min(pair.south_offset, pair.north_offset)
    high_offset = max(pair.south_offset, pair.north_offset)
    inner_low = low_offset + min(0.8, max(0.0, (high_offset - low_offset) * 0.2))
    inner_high = high_offset - min(0.8, max(0.0, (high_offset - low_offset) * 0.2))
    station_low = pair.south_station - station_pad_m
    station_high = pair.north_station + station_pad_m
    for point in evidence.points:
        if station_low <= point.station <= station_high and inner_low <= point.offset <= inner_high:
            return True
    return False


def transition_points_from_evidence(
    pair: CrossoverPair,
    evidence_items: list[RawEvidence],
    *,
    station_pad_m: float,
    offset_pad_m: float,
) -> list[RawPoint]:
    low_offset = min(pair.south_offset, pair.north_offset) - offset_pad_m
    high_offset = max(pair.south_offset, pair.north_offset) + offset_pad_m
    station_low = pair.south_station - station_pad_m
    station_high = pair.north_station + station_pad_m
    by_bucket: dict[tuple[int, int], RawPoint] = {}
    for evidence in evidence_items:
        for point in evidence.points:
            if not (station_low <= point.station <= station_high and low_offset <= point.offset <= high_offset):
                continue
            key = (round(point.station / 0.15), round(point.offset / 0.15))
            previous = by_bucket.get(key)
            if previous is None or point.confidence > previous.confidence:
                by_bucket[key] = point
    points = sorted(by_bucket.values(), key=lambda item: item.station)
    return points


def tangent_baseline(start_offset: float, end_offset: float, u: float) -> float:
    h = btt.smoothstep(u)
    return start_offset + (end_offset - start_offset) * h


def curve_straight_curve_baseline(start_offset: float, end_offset: float, u: float, *, curve_fraction: float = 0.28) -> float:
    c = min(max(curve_fraction, 0.05), 0.45)
    middle_slope = 1.0 / (1.0 - c)
    if u <= c:
        fraction = 0.5 * middle_slope * u * u / c
    elif u >= 1.0 - c:
        fraction = 1.0 - 0.5 * middle_slope * (1.0 - u) * (1.0 - u) / c
    else:
        fraction = middle_slope * (u - 0.5 * c)
    return start_offset + (end_offset - start_offset) * fraction


def endpoint_taper(u: float) -> float:
    return btt.smoothstep(min(u, 1.0 - u) / 0.18)


def nearby_evidence_offset(
    station: float,
    baseline_offset: float,
    *,
    raw_points: list[RawPoint],
    station_window_m: float,
    offset_window_m: float,
) -> tuple[float, float] | None:
    weighted_offsets: list[tuple[float, float]] = []
    for point in raw_points:
        ds = abs(point.station - station)
        if ds > station_window_m:
            if point.station > station + station_window_m:
                break
            continue
        dt = abs(point.offset - baseline_offset)
        if dt > offset_window_m:
            continue
        weight = max(0.05, point.confidence) * (1.0 - ds / station_window_m) * (1.0 - dt / offset_window_m)
        if weight > 0.0:
            weighted_offsets.append((point.offset, weight))
    if not weighted_offsets:
        return None
    total_weight = sum(weight for _, weight in weighted_offsets)
    offset = sum(value * weight for value, weight in weighted_offsets) / total_weight
    evidence_weight = min(0.88, total_weight / (total_weight + 1.5))
    return offset, evidence_weight


def smooth_offsets(points: list[tuple[float, float]], *, passes: int) -> list[tuple[float, float]]:
    current = points[:]
    for _ in range(passes):
        if len(current) <= 4:
            break
        updated = current[:]
        for index in range(1, len(current) - 1):
            station = current[index][0]
            offset = current[index - 1][1] * 0.25 + current[index][1] * 0.5 + current[index + 1][1] * 0.25
            updated[index] = (station, offset)
        current = updated
    return current


def enforce_monotonic_offsets(points: list[tuple[float, float]], *, increasing: bool) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    output = points[:]
    if increasing:
        last = output[0][1]
        for index in range(1, len(output) - 1):
            station, offset = output[index]
            last = max(last - 0.03, offset)
            output[index] = (station, last)
    else:
        last = output[0][1]
        for index in range(1, len(output) - 1):
            station, offset = output[index]
            last = min(last + 0.03, offset)
            output[index] = (station, last)
    return output


def score_crossover(
    points: list[tuple[float, float]],
    *,
    raw_points: list[RawPoint],
    support_distance_m: float,
    support_station_window_m: float,
) -> CrossoverScore:
    supported = 0
    transition_total = 0
    transition_supported = 0
    distance_sum = 0.0
    confidence_sum = 0.0
    for station, offset in points:
        is_transition = 1.0 < abs(offset) < 4.0
        if is_transition:
            transition_total += 1
        best_distance = float("inf")
        best_confidence = 0.0
        for point in raw_points:
            ds = abs(point.station - station)
            if ds > support_station_window_m:
                if point.station > station + support_station_window_m:
                    break
                continue
            distance = math.hypot(point.station - station, point.offset - offset)
            if distance < best_distance:
                best_distance = distance
                best_confidence = point.confidence
        if best_distance <= support_distance_m:
            supported += 1
            distance_sum += best_distance
            confidence_sum += best_confidence
            if is_transition:
                transition_supported += 1
    coverage = supported / max(len(points), 1)
    transition_coverage = transition_supported / transition_total if transition_total else coverage
    mean_distance = distance_sum / supported if supported else support_distance_m * 2.5
    mean_confidence = confidence_sum / supported if supported else 0.0
    score = coverage * 0.45 + transition_coverage * 0.45 + min(mean_confidence, 1.0) * 0.15 - min(mean_distance / max(support_distance_m, 1e-6), 1.0) * 0.15
    return CrossoverScore(
        score=round(score, 4),
        support_coverage=round(coverage, 4),
        transition_coverage=round(transition_coverage, 4),
        mean_distance_m=round(mean_distance, 4),
        mean_confidence=round(mean_confidence, 4),
        evidence_points=supported,
    )


def pair_features(pairs: list[CrossoverPair]) -> list[dict[str, Any]]:
    return [
        {
            "type": "Feature",
            "properties": {
                "crossover_id": pair.crossover_id,
                "south_anchor": pair.south_anchor_id,
                "north_anchor": pair.north_anchor_id,
                "south_role": pair.south_role,
                "north_role": pair.north_role,
                "note": pair.note,
            },
            "geometry": {"type": "LineString", "coordinates": [list(pair.south_point), list(pair.north_point)]},
        }
        for pair in pairs
    ]


def write_crossover_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pyshp in the active virtual environment.") from exc
    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("conn_id", "C", size=16)
    writer.field("kind", "C", size=40)
    writer.field("pair_id", "C", size=24)
    writer.field("direction", "C", size=64)
    writer.field("shape", "C", size=40)
    writer.field("status", "C", size=32)
    writer.field("south_a", "C", size=16)
    writer.field("north_a", "C", size=16)
    writer.field("s0_m", "F", decimal=3)
    writer.field("s1_m", "F", decimal=3)
    writer.field("t0_m", "F", decimal=3)
    writer.field("t1_m", "F", decimal=3)
    writer.field("score", "F", decimal=4)
    writer.field("sup_cov", "F", decimal=4)
    writer.field("trans_cov", "F", decimal=4)
    writer.field("sup_dist", "F", decimal=4)
    writer.field("sup_conf", "F", decimal=4)
    writer.field("evid_n", "N", size=8)
    writer.field("tr_feat_n", "N", size=8)
    writer.field("tr_pt_n", "N", size=8)
    writer.field("tr_ids", "C", size=180)
    writer.field("qa_note", "C", size=80)
    for index, feature in enumerate(features):
        props = feature.get("properties") or {}
        writer.line([btc.line_coords(feature)])
        writer.record(
            str(props.get("connector_id", f"CX{index + 1:02d}"))[:16],
            str(props.get("geom_kind", ""))[:40],
            str(props.get("pair_id", ""))[:24],
            str(props.get("direction", ""))[:64],
            str(props.get("shape_model", ""))[:40],
            str(props.get("review_status", ""))[:32],
            str(props.get("south_anchor", ""))[:16],
            str(props.get("north_anchor", ""))[:16],
            btc.safe_float(props.get("station_min_m", 0.0)),
            btc.safe_float(props.get("station_max_m", 0.0)),
            btc.safe_float(props.get("offset_start_m", 0.0)),
            btc.safe_float(props.get("offset_end_m", 0.0)),
            btc.safe_float(props.get("connector_score", 0.0)),
            btc.safe_float(props.get("support_cov", 0.0)),
            btc.safe_float(props.get("trans_cov", 0.0)),
            btc.safe_float(props.get("support_dist", 0.0)),
            btc.safe_float(props.get("support_conf", 0.0)),
            int(btc.safe_float(props.get("evidence_n", 0))),
            int(btc.safe_float(props.get("trans_feat_n", 0))),
            int(btc.safe_float(props.get("trans_pt_n", 0))),
            str(props.get("trans_ids", ""))[:180],
            str(props.get("qa_note", ""))[:80],
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    btc.write_projection(output_path.with_suffix(".prj"), epsg)


def write_pair_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pyshp in the active virtual environment.") from exc
    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("cx_id", "C", size=16)
    writer.field("south_a", "C", size=16)
    writer.field("north_a", "C", size=16)
    writer.field("south_role", "C", size=32)
    writer.field("north_role", "C", size=32)
    writer.field("note", "C", size=100)
    for feature in features:
        props = feature.get("properties") or {}
        writer.line([btc.line_coords(feature)])
        writer.record(
            str(props.get("crossover_id", ""))[:16],
            str(props.get("south_anchor", ""))[:16],
            str(props.get("north_anchor", ""))[:16],
            str(props.get("south_role", ""))[:32],
            str(props.get("north_role", ""))[:32],
            str(props.get("note", ""))[:100],
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    btc.write_projection(output_path.with_suffix(".prj"), epsg)


def write_fullres_review_crops(
    dom_path: Path,
    *,
    connector_features: list[dict[str, Any]],
    band_features: list[dict[str, Any]],
    out_dir: Path,
    bounds_padding_m: float,
    segment_crop_m: float,
    line_width_px: int,
) -> dict[str, Any]:
    from PIL import Image, ImageDraw
    import rasterio
    from rasterio.windows import Window, from_bounds

    paths: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dom_path) as dataset:
        for feature in connector_features:
            stem = btc.sanitize_filename(str((feature.get("properties") or {}).get("connector_id", "crossover")))
            bounds_window = feature_bounds_window(dataset, feature, padding_m=bounds_padding_m)
            if bounds_window is not None:
                overlay_path = out_dir / f"{stem}_fullres_bounds_overlay.png"
                write_review_crop(
                    dataset,
                    bounds_window,
                    connector_features=[feature],
                    band_features=band_features,
                    output_path=overlay_path,
                    label=f"{stem} full bounds",
                    line_width_px=line_width_px,
                )
                paths.append(str(overlay_path))
            segment_windows = feature_segment_windows(dataset, feature, crop_m=segment_crop_m)
            for name, window in segment_windows:
                overlay_path = out_dir / f"{stem}_fullres_{name}_overlay.png"
                write_review_crop(
                    dataset,
                    window,
                    connector_features=[feature],
                    band_features=band_features,
                    output_path=overlay_path,
                    label=f"{stem} {name}",
                    line_width_px=line_width_px,
                )
                paths.append(str(overlay_path))
    return {
        "mode": "native_resolution_no_resize",
        "line_width_px": line_width_px,
        "bounds_padding_m": bounds_padding_m,
        "segment_crop_m": segment_crop_m,
        "overlays": paths,
    }


def feature_bounds_window(dataset: Any, feature: dict[str, Any], *, padding_m: float) -> Any | None:
    from rasterio.windows import from_bounds

    coords = btc.line_coords(feature)
    if len(coords) < 2:
        return None
    min_x = min(x for x, _ in coords) - padding_m
    max_x = max(x for x, _ in coords) + padding_m
    min_y = min(y for _, y in coords) - padding_m
    max_y = max(y for _, y in coords) + padding_m
    window = from_bounds(min_x, min_y, max_x, max_y, transform=dataset.transform)
    return clamp_window(dataset, window)


def feature_segment_windows(dataset: Any, feature: dict[str, Any], *, crop_m: float) -> list[tuple[str, Any]]:
    from rasterio.windows import Window

    coords = btc.line_coords(feature)
    if len(coords) < 2:
        return []
    picks = [("south", coords[0]), ("middle", coords[len(coords) // 2]), ("north", coords[-1])]
    windows: list[tuple[str, Any]] = []
    half = crop_m / 2.0
    for name, (x, y) in picks:
        row, col = dataset.index(x, y)
        pixel_width = max(abs(float(dataset.transform.a)), 1e-6)
        pixel_height = max(abs(float(dataset.transform.e)), 1e-6)
        half_w = max(16, int(math.ceil(half / pixel_width)))
        half_h = max(16, int(math.ceil(half / pixel_height)))
        windows.append((name, clamp_window(dataset, Window(col - half_w, row - half_h, half_w * 2, half_h * 2))))
    return [(name, window) for name, window in windows if window is not None]


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
    connector_features: list[dict[str, Any]],
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
    btc.draw_features(draw, band_features, transform, width_px=2)
    draw_review_connectors(draw, connector_features, transform, width_px=line_width_px)
    draw_review_endpoints(draw, connector_features, transform)
    draw_small_label(draw, label)
    overlay.convert("RGB").save(output_path)


def draw_review_connectors(draw: Any, features: list[dict[str, Any]], transform: Any, *, width_px: int) -> None:
    for feature in features:
        coords = []
        for x, y in btc.line_coords(feature):
            col, row = ~transform * (x, y)
            coords.append((col, row))
        if len(coords) >= 2:
            draw.line(coords, fill=(0, 255, 70, 255), width=max(1, width_px), joint="curve")


def draw_review_endpoints(draw: Any, features: list[dict[str, Any]], transform: Any) -> None:
    for feature in features:
        coords = btc.line_coords(feature)
        if len(coords) < 2:
            continue
        for (x, y), color in ((coords[0], (0, 120, 255, 235)), (coords[-1], (255, 0, 180, 235))):
            col, row = ~transform * (x, y)
            radius = 5
            draw.ellipse([col - radius, row - radius, col + radius, row + radius], fill=color, outline=(255, 255, 255, 255), width=2)


def draw_small_label(draw: Any, label: str) -> None:
    box = [8, 8, 300, 36]
    draw.rectangle(box, fill=(0, 0, 0, 190))
    draw.text((15, 14), label[:38], fill=(255, 255, 255, 255))


def summarize_feature(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties") or {}
    return {
        "crossover_id": props.get("connector_id", ""),
        "south_anchor": props.get("south_anchor", ""),
        "north_anchor": props.get("north_anchor", ""),
        "direction": props.get("direction", ""),
        "station_min_m": props.get("station_min_m", 0.0),
        "station_max_m": props.get("station_max_m", 0.0),
        "offset_start_m": props.get("offset_start_m", 0.0),
        "offset_end_m": props.get("offset_end_m", 0.0),
        "score": props.get("connector_score", 0.0),
        "support_coverage": props.get("support_cov", 0.0),
        "transition_coverage": props.get("trans_cov", 0.0),
        "support_distance_m": props.get("support_dist", 0.0),
        "evidence_points": props.get("evidence_n", 0),
        "transition_features": props.get("trans_feat_n", 0),
        "transition_points": props.get("trans_pt_n", 0),
        "transition_ids": props.get("trans_ids", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())
