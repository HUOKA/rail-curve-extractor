from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_CANDIDATES = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_candidates/track_centerline_candidates.geojson")
DEFAULT_MAINLINE = Path("output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_TRACK_BANDS = Path("output/raw_dom_roi_fullpass_v1/track_band_priors/track_band_centerline_priors.geojson")
DEFAULT_DOM = Path("data/生产数据/无人机数据/正射/dom.tif")
DEFAULT_SWITCH_ANCHORS = Path("data/manual_feedback/turnout_switch_anchors.geojson")
DEFAULT_CONNECTOR_SPLITS = Path("data/manual_feedback/turnout_connector_splits.geojson")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/turnout_connector_candidates")
DEFAULT_EPSG = 32651


CONNECTOR_PAIRS = [
    {
        "pair_id": "minus_to_main",
        "low_band": "parallel_minus_5m",
        "low_offset": -5.0,
        "high_band": "mainline_2_track",
        "high_offset": 0.0,
    },
    {
        "pair_id": "main_to_plus",
        "low_band": "mainline_2_track",
        "low_offset": 0.0,
        "high_band": "parallel_plus_5m",
        "high_offset": 5.0,
    },
]

PAIR_COLORS = {
    "minus_to_main": (0, 158, 115, 255),
    "main_to_plus": (213, 94, 0, 255),
}

BAND_COLORS = {
    "parallel_minus_5m": (0, 114, 178, 180),
    "mainline_2_track": (255, 0, 0, 180),
    "parallel_plus_5m": (230, 159, 0, 180),
    "possible_outer_plus_10m": (131, 56, 236, 160),
}


@dataclass(frozen=True)
class TransitionStats:
    feature_index: int
    s_min: float
    s_max: float
    station_span_m: float
    t_min: float
    t_max: float
    offset_span_m: float
    t_median: float
    t_at_s_min: float
    t_at_s_max: float
    slope_dt_ds: float
    point_count: int
    mean_confidence: float
    image_name: str


@dataclass(frozen=True)
class SwitchAnchor:
    anchor_id: str
    point: tuple[float, float]
    station: float
    offset: float
    pair_id: str
    target_band: str
    target_offset: float
    source: str


@dataclass(frozen=True)
class ConnectorSplit:
    split_id: str
    point: tuple[float, float]
    station: float
    offset: float
    pair_id: str
    evidence_id: str
    keep_connector_side: str
    straight_side: str
    straight_band: str
    source: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build QGIS-ready turnout/connector candidates from raw DOM centerline evidence.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument("--track-bands", type=Path, default=DEFAULT_TRACK_BANDS)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--switch-anchors", type=Path, default=DEFAULT_SWITCH_ANCHORS)
    parser.add_argument("--connector-splits", type=Path, default=DEFAULT_CONNECTOR_SPLITS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-station-span-m", type=float, default=8.0)
    parser.add_argument("--min-offset-span-m", type=float, default=1.2)
    parser.add_argument("--min-abs-slope", type=float, default=0.015)
    parser.add_argument("--max-abs-slope", type=float, default=0.25)
    parser.add_argument("--offset-pad-m", type=float, default=0.8)
    parser.add_argument("--anchor-tolerance-m", type=float, default=1.8)
    parser.add_argument("--min-confidence", type=float, default=0.28)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--max-extrapolate-m", type=float, default=80.0)
    parser.add_argument("--max-switch-anchor-distance-m", type=float, default=45.0)
    parser.add_argument("--endpoint-snap-tolerance-m", type=float, default=0.35)
    parser.add_argument("--max-proposal-span-m", type=float, default=220.0)
    parser.add_argument("--sample-step-m", type=float, default=5.0)
    parser.add_argument("--qa-crop-width-m", type=float, default=145.0)
    parser.add_argument("--qa-crop-height-m", type=float, default=145.0)
    parser.add_argument("--skip-qa-crops", action="store_true")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidates_path = args.candidates.expanduser().resolve()
    mainline_path = args.mainline.expanduser().resolve()
    track_bands_path = args.track_bands.expanduser().resolve()
    dom_path = args.dom.expanduser().resolve()
    switch_anchors_path = args.switch_anchors.expanduser().resolve()
    connector_splits_path = args.connector_splits.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mainline_features = load_line_features(mainline_path)
    if not mainline_features:
        raise RuntimeError(f"No LineString mainline found: {mainline_path}")
    mainline_coords = line_coords(mainline_features[0])
    guide = Guide(mainline_coords[0], mainline_coords[-1])
    switch_anchors = load_switch_anchors(switch_anchors_path, guide=guide) if switch_anchors_path.exists() else []
    connector_splits = load_connector_splits(connector_splits_path, guide=guide) if connector_splits_path.exists() else []

    raw_features = load_line_features(candidates_path)
    raw_evidence = find_transition_evidence(
        raw_features,
        guide=guide,
        min_station_span_m=args.min_station_span_m,
        min_offset_span_m=args.min_offset_span_m,
        min_abs_slope=args.min_abs_slope,
        max_abs_slope=args.max_abs_slope,
        offset_pad_m=args.offset_pad_m,
        anchor_tolerance_m=args.anchor_tolerance_m,
        min_confidence=args.min_confidence,
        min_points=args.min_points,
    )
    evidence_features = dedupe_transition_evidence(raw_evidence)
    proposal_features = build_evidence_curve_proposals(
        evidence_features,
        guide=guide,
        switch_anchors=switch_anchors,
        connector_splits=connector_splits,
        max_extrapolate_m=args.max_extrapolate_m,
        max_switch_anchor_distance_m=args.max_switch_anchor_distance_m,
        endpoint_snap_tolerance_m=args.endpoint_snap_tolerance_m,
        max_proposal_span_m=args.max_proposal_span_m,
        sample_step_m=args.sample_step_m,
    )
    package_features = [*evidence_features, *proposal_features]

    evidence_geojson = out_dir / "turnout_connector_evidence.geojson"
    proposal_geojson = out_dir / "turnout_connector_proposals.geojson"
    package_geojson = out_dir / "turnout_connector_package.geojson"
    write_geojson(evidence_geojson, evidence_features, epsg=args.epsg)
    write_geojson(proposal_geojson, proposal_features, epsg=args.epsg)
    write_geojson(package_geojson, package_features, epsg=args.epsg)
    for path in (evidence_geojson, proposal_geojson, package_geojson):
        write_connector_shapefile(load_line_features(path), path.with_suffix(".shp"), epsg=args.epsg)
        write_connector_qml(path.with_suffix(".qml"))

    qa_summary: dict[str, Any] | None = None
    if not args.skip_qa_crops and dom_path.exists():
        band_features = load_line_features(track_bands_path) if track_bands_path.exists() else []
        qa_summary = write_qa_crops(
            dom_path,
            connector_features=package_features,
            band_features=band_features,
            out_dir=out_dir / "qa_crops",
            crop_width_m=args.qa_crop_width_m,
            crop_height_m=args.qa_crop_height_m,
        )

    summary = summarize_connectors(
        evidence_features,
        proposal_features,
        candidate_feature_count=len(raw_features),
        guide_length_m=guide.length,
    )
    summary.update(
        {
            "mainline": str(mainline_path),
            "candidates": str(candidates_path),
            "track_bands": str(track_bands_path),
            "dom": str(dom_path),
            "policy": {
                "status": "candidate_only_not_final_topology",
                "rule": "raw transition fragments are evidence; proposals follow raw evidence curvature and only use short switch-anchor completion before DOM/QGIS review",
                "connector_pairs": CONNECTOR_PAIRS,
                "min_station_span_m": args.min_station_span_m,
                "min_offset_span_m": args.min_offset_span_m,
                "min_abs_slope": args.min_abs_slope,
                "max_abs_slope": args.max_abs_slope,
                "max_extrapolate_m": args.max_extrapolate_m,
                "max_switch_anchor_distance_m": args.max_switch_anchor_distance_m,
                "endpoint_snap_tolerance_m": args.endpoint_snap_tolerance_m,
                "switch_anchors": str(switch_anchors_path) if switch_anchors else None,
                "connector_splits": str(connector_splits_path) if connector_splits else None,
            },
            "outputs": {
                "evidence_geojson": str(evidence_geojson),
                "evidence_shp": str(evidence_geojson.with_suffix(".shp")),
                "proposal_geojson": str(proposal_geojson),
                "proposal_shp": str(proposal_geojson.with_suffix(".shp")),
                "package_geojson": str(package_geojson),
                "package_shp": str(package_geojson.with_suffix(".shp")),
                "qa_crops": str(out_dir / "qa_crops") if qa_summary else None,
            },
        }
    )
    if qa_summary is not None:
        summary["qa_crops"] = qa_summary
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


class Guide:
    def __init__(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        self.start = start
        self.end = end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        self.length = math.hypot(dx, dy)
        if self.length <= 0:
            raise ValueError("Guide endpoints must be different.")
        self.ux = dx / self.length
        self.uy = dy / self.length
        self.nx = -self.uy
        self.ny = self.ux

    def station_offset(self, point: tuple[float, float]) -> tuple[float, float]:
        dx = point[0] - self.start[0]
        dy = point[1] - self.start[1]
        return dx * self.ux + dy * self.uy, dx * self.nx + dy * self.ny

    def point_at(self, station: float, offset_m: float) -> tuple[float, float]:
        return (
            self.start[0] + self.ux * station + self.nx * offset_m,
            self.start[1] + self.uy * station + self.ny * offset_m,
        )


def find_transition_evidence(
    features: list[dict[str, Any]],
    *,
    guide: Guide,
    min_station_span_m: float,
    min_offset_span_m: float,
    min_abs_slope: float,
    max_abs_slope: float,
    offset_pad_m: float,
    anchor_tolerance_m: float,
    min_confidence: float,
    min_points: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        stats = compute_transition_stats(feature, feature_index=index, guide=guide)
        if stats is None:
            continue
        if stats.station_span_m < min_station_span_m or stats.offset_span_m < min_offset_span_m:
            continue
        if stats.point_count < min_points or stats.mean_confidence < min_confidence:
            continue
        slope_abs = abs(stats.slope_dt_ds)
        if slope_abs < min_abs_slope or slope_abs > max_abs_slope:
            continue
        for pair in CONNECTOR_PAIRS:
            score = score_transition_for_pair(
                stats,
                pair=pair,
                offset_pad_m=offset_pad_m,
                anchor_tolerance_m=anchor_tolerance_m,
            )
            if score is None:
                continue
            evidence.append(build_evidence_feature(feature, stats=stats, pair=pair, score=score))
    return evidence


def compute_transition_stats(feature: dict[str, Any], *, feature_index: int, guide: Guide) -> TransitionStats | None:
    stations_offsets = [guide.station_offset(coord) for coord in line_coords(feature)]
    stations_offsets = [(s, t) for s, t in stations_offsets if -5.0 <= s <= guide.length + 5.0]
    if len(stations_offsets) < 2:
        return None
    stations_offsets.sort(key=lambda item: item[0])
    stations = [item[0] for item in stations_offsets]
    offsets = [item[1] for item in stations_offsets]
    station_span = stations[-1] - stations[0]
    if station_span <= 0:
        return None
    offset_span = max(offsets) - min(offsets)
    props = feature.get("properties") or {}
    return TransitionStats(
        feature_index=feature_index,
        s_min=stations[0],
        s_max=stations[-1],
        station_span_m=station_span,
        t_min=min(offsets),
        t_max=max(offsets),
        offset_span_m=offset_span,
        t_median=float(median(offsets)),
        t_at_s_min=offsets[0],
        t_at_s_max=offsets[-1],
        slope_dt_ds=(offsets[-1] - offsets[0]) / station_span,
        point_count=int(props.get("point_count", len(offsets))),
        mean_confidence=safe_float(props.get("mean_confidence", 0.0)),
        image_name=str(props.get("image_name", "")),
    )


def score_transition_for_pair(
    stats: TransitionStats,
    *,
    pair: dict[str, Any],
    offset_pad_m: float,
    anchor_tolerance_m: float,
) -> float | None:
    low = float(pair["low_offset"])
    high = float(pair["high_offset"])
    width = high - low
    if width <= 0:
        return None
    if stats.t_min < low - offset_pad_m or stats.t_max > high + offset_pad_m:
        return None
    if stats.t_max <= low + 0.65 or stats.t_min >= high - 0.65:
        return None
    lower_anchor = max(0.0, 1.0 - abs(stats.t_min - low) / anchor_tolerance_m)
    upper_anchor = max(0.0, 1.0 - abs(stats.t_max - high) / anchor_tolerance_m)
    anchor_score = max(lower_anchor, upper_anchor)
    if anchor_score <= 0.0 and stats.offset_span_m < width * 0.45:
        return None
    coverage = min(1.0, stats.offset_span_m / width)
    length_score = min(1.0, stats.station_span_m / 80.0)
    slope_score = max(0.0, 1.0 - abs(abs(stats.slope_dt_ds) - 0.06) / 0.08)
    confidence_score = max(0.0, min(1.0, stats.mean_confidence))
    return round(0.38 * coverage + 0.24 * anchor_score + 0.18 * length_score + 0.12 * slope_score + 0.08 * confidence_score, 4)


def build_evidence_feature(
    feature: dict[str, Any],
    *,
    stats: TransitionStats,
    pair: dict[str, Any],
    score: float,
) -> dict[str, Any]:
    pair_id = str(pair["pair_id"])
    direction = direction_for_stats(stats, pair)
    props = dict(feature.get("properties") or {})
    props.update(
        {
            "role": "turnout_connector_evidence",
            "geom_kind": "raw_transition_evidence",
            "pair_id": pair_id,
            "direction": direction,
            "source": "raw_dom_transition_fragment",
            "review_status": "candidate_needs_dom_review",
            "qa_note": "raw_transition_not_final_topology",
            "feature_index": stats.feature_index,
            "station_min_m": round(stats.s_min, 3),
            "station_max_m": round(stats.s_max, 3),
            "station_span_m": round(stats.station_span_m, 3),
            "offset_min_m": round(stats.t_min, 3),
            "offset_max_m": round(stats.t_max, 3),
            "offset_span_m": round(stats.offset_span_m, 3),
            "offset_median_m": round(stats.t_median, 3),
            "slope_dt_ds": round(stats.slope_dt_ds, 5),
            "connector_score": score,
            "low_band": pair["low_band"],
            "high_band": pair["high_band"],
        }
    )
    return {"type": "Feature", "properties": props, "geometry": feature["geometry"]}


def direction_for_stats(stats: TransitionStats, pair: dict[str, Any]) -> str:
    if stats.slope_dt_ds >= 0:
        return f"{pair['low_band']}->{pair['high_band']}"
    return f"{pair['high_band']}->{pair['low_band']}"


def dedupe_transition_evidence(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_features = sorted(
        features,
        key=lambda feature: (
            -safe_float(feature["properties"].get("connector_score", 0.0)),
            -safe_float(feature["properties"].get("station_span_m", 0.0)),
            -safe_float(feature["properties"].get("offset_span_m", 0.0)),
        ),
    )
    kept: list[dict[str, Any]] = []
    for feature in sorted_features:
        if any(is_duplicate_transition(feature, existing) for existing in kept):
            continue
        kept.append(feature)
    kept.sort(key=lambda feature: (str(feature["properties"].get("pair_id", "")), safe_float(feature["properties"].get("station_min_m", 0.0))))
    for index, feature in enumerate(kept, start=1):
        feature["properties"]["connector_id"] = f"E{index:03d}"
    return kept


def is_duplicate_transition(a: dict[str, Any], b: dict[str, Any]) -> bool:
    pa = a.get("properties") or {}
    pb = b.get("properties") or {}
    if pa.get("pair_id") != pb.get("pair_id"):
        return False
    a_s = (safe_float(pa.get("station_min_m", 0.0)), safe_float(pa.get("station_max_m", 0.0)))
    b_s = (safe_float(pb.get("station_min_m", 0.0)), safe_float(pb.get("station_max_m", 0.0)))
    a_t = (safe_float(pa.get("offset_min_m", 0.0)), safe_float(pa.get("offset_max_m", 0.0)))
    b_t = (safe_float(pb.get("offset_min_m", 0.0)), safe_float(pb.get("offset_max_m", 0.0)))
    station_overlap = interval_overlap_ratio(a_s, b_s)
    offset_overlap = interval_overlap_ratio(a_t, b_t)
    return station_overlap >= 0.55 and offset_overlap >= 0.50


def interval_overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    a0, a1 = min(a), max(a)
    b0, b1 = min(b), max(b)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(min(a1 - a0, b1 - b0), 1e-9)
    return overlap / denom


def build_evidence_curve_proposals(
    evidence_features: list[dict[str, Any]],
    *,
    guide: Guide,
    switch_anchors: list[SwitchAnchor],
    connector_splits: list[ConnectorSplit],
    max_extrapolate_m: float,
    max_switch_anchor_distance_m: float,
    endpoint_snap_tolerance_m: float,
    max_proposal_span_m: float,
    sample_step_m: float,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for evidence in evidence_features:
        props = evidence.get("properties") or {}
        pair = pair_by_id(str(props.get("pair_id", "")))
        if pair is None:
            continue
        proposal = build_evidence_curve_proposal(
            evidence,
            pair=pair,
            guide=guide,
            switch_anchors=switch_anchors,
            connector_splits=connector_splits,
            max_extrapolate_m=max_extrapolate_m,
            max_switch_anchor_distance_m=max_switch_anchor_distance_m,
            endpoint_snap_tolerance_m=endpoint_snap_tolerance_m,
            max_proposal_span_m=max_proposal_span_m,
            sample_step_m=sample_step_m,
        )
        if proposal is None:
            continue
        proposal["properties"]["connector_id"] = f"P{len(proposals) + 1:03d}"
        proposals.append(proposal)
    return proposals


def build_evidence_curve_proposal(
    evidence: dict[str, Any],
    *,
    pair: dict[str, Any],
    guide: Guide,
    switch_anchors: list[SwitchAnchor],
    connector_splits: list[ConnectorSplit],
    max_extrapolate_m: float,
    max_switch_anchor_distance_m: float,
    endpoint_snap_tolerance_m: float,
    max_proposal_span_m: float,
    sample_step_m: float,
) -> dict[str, Any] | None:
    evidence_st = station_offsets_for_feature(evidence, guide=guide)
    if len(evidence_st) < 2:
        return None
    targets = pair_targets(pair)
    start_target = nearest_target(evidence_st[0][1], targets)
    end_target = nearest_target(evidence_st[-1][1], targets)

    start_result = complete_endpoint(
        evidence_st,
        at_start=True,
        target=start_target,
        pair=pair,
        switch_anchors=switch_anchors,
        max_extrapolate_m=max_extrapolate_m,
        max_switch_anchor_distance_m=max_switch_anchor_distance_m,
        endpoint_snap_tolerance_m=endpoint_snap_tolerance_m,
        sample_step_m=sample_step_m,
    )
    proposal_st = start_result["points"]
    end_result = complete_endpoint(
        proposal_st,
        at_start=False,
        target=end_target,
        pair=pair,
        switch_anchors=switch_anchors,
        max_extrapolate_m=max_extrapolate_m,
        max_switch_anchor_distance_m=max_switch_anchor_distance_m,
        endpoint_snap_tolerance_m=endpoint_snap_tolerance_m,
        sample_step_m=sample_step_m,
    )
    proposal_st = end_result["points"]
    split_result = apply_connector_split(
        proposal_st,
        pair=pair,
        evidence_id=str(evidence.get("properties", {}).get("connector_id", "")),
        connector_splits=connector_splits,
    )
    proposal_st = split_result["points"]

    proposal_span = proposal_st[-1][0] - proposal_st[0][0]
    if proposal_span <= 0.0 or proposal_span > max_proposal_span_m:
        return None

    coords = [guide.point_at(s, t) for s, t in proposal_st]
    evidence_props = evidence.get("properties") or {}
    completion_m = safe_float(start_result["completion_m"]) + safe_float(end_result["completion_m"])
    completed_endpoints = int(bool(start_result["complete"])) + int(bool(end_result["complete"]))
    base_score = safe_float(evidence_props.get("connector_score", 0.0))
    partial_penalty = 0.08 * (2 - completed_endpoints)
    completion_penalty = min(0.18, completion_m / max(max_extrapolate_m * 5.0, 1.0))
    proposal_score = max(0.0, round(base_score - partial_penalty - completion_penalty, 4))
    anchor_ids = [value for value in [start_result.get("anchor_id"), end_result.get("anchor_id")] if value]
    split_id = str(split_result.get("split_id", ""))
    straight_tail_m = safe_float(split_result.get("straight_tail_m", 0.0))
    qa_note = proposal_qa_note(start_result, end_result)
    if split_id:
        qa_note = "trimmed_straight_tail_at_user_split"
    proposal_props = {
        "role": "turnout_connector_proposal",
        "geom_kind": "evidence_curve_proposal",
        "pair_id": pair["pair_id"],
        "direction": f"{start_result['band']}->{end_result['band']}",
        "source": "raw_transition_evidence_curve_following",
        "review_status": "candidate_needs_dom_review",
        "qa_note": qa_note,
        "proposal_mode": "evidence_curve_with_optional_switch_anchor",
        "evidence_id": evidence_props.get("connector_id", ""),
        "evidence_idx": evidence_props.get("feature_index", ""),
        "anchor_id": ",".join(anchor_ids),
        "split_id": split_id,
        "station_min_m": round(proposal_st[0][0], 3),
        "station_max_m": round(proposal_st[-1][0], 3),
        "station_span_m": round(proposal_span, 3),
        "offset_start_m": round(proposal_st[0][1], 3),
        "offset_end_m": round(proposal_st[-1][1], 3),
        "offset_min_m": round(min(t for _, t in proposal_st), 3),
        "offset_max_m": round(max(t for _, t in proposal_st), 3),
        "observed_s0_m": evidence_props.get("station_min_m", 0.0),
        "observed_s1_m": evidence_props.get("station_max_m", 0.0),
        "extrap_before_m": round(safe_float(start_result["completion_m"]), 3),
        "extrap_after_m": round(safe_float(end_result["completion_m"]), 3),
        "completion_m": round(completion_m, 3),
        "straight_tail_m": round(straight_tail_m, 3),
        "connector_score": proposal_score,
        "low_band": pair["low_band"],
        "high_band": pair["high_band"],
        "mean_confidence": evidence_props.get("mean_confidence", 0.0),
    }
    return {
        "type": "Feature",
        "properties": proposal_props,
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
    }


def station_offsets_for_feature(feature: dict[str, Any], *, guide: Guide) -> list[tuple[float, float]]:
    points = sorted([guide.station_offset(coord) for coord in line_coords(feature)], key=lambda item: item[0])
    cleaned: list[tuple[float, float]] = []
    for station, offset in points:
        if cleaned and abs(station - cleaned[-1][0]) < 1e-6:
            continue
        cleaned.append((station, offset))
    return cleaned


def pair_targets(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"band": str(pair["low_band"]), "offset": float(pair["low_offset"])},
        {"band": str(pair["high_band"]), "offset": float(pair["high_offset"])},
    ]


def nearest_target(offset: float, targets: list[dict[str, Any]]) -> dict[str, Any]:
    return min(targets, key=lambda target: abs(offset - float(target["offset"])))


def complete_endpoint(
    points: list[tuple[float, float]],
    *,
    at_start: bool,
    target: dict[str, Any],
    pair: dict[str, Any],
    switch_anchors: list[SwitchAnchor],
    max_extrapolate_m: float,
    max_switch_anchor_distance_m: float,
    endpoint_snap_tolerance_m: float,
    sample_step_m: float,
) -> dict[str, Any]:
    endpoint = points[0] if at_start else points[-1]
    target_offset = float(target["offset"])
    target_band = str(target["band"])
    if abs(endpoint[1] - target_offset) <= endpoint_snap_tolerance_m:
        updated = list(points)
        snapped = (endpoint[0], target_offset)
        if at_start:
            updated[0] = snapped
        else:
            updated[-1] = snapped
        return {"points": updated, "complete": True, "band": target_band, "completion_m": 0.0, "mode": "snap_to_band"}

    anchor = best_switch_anchor(
        endpoint,
        at_start=at_start,
        target=target,
        pair=pair,
        switch_anchors=switch_anchors,
        max_switch_anchor_distance_m=max_switch_anchor_distance_m,
    )
    if anchor is None:
        return {"points": points, "complete": False, "band": target_band, "completion_m": 0.0, "mode": "partial_evidence_only"}

    completion_m = abs(anchor.station - endpoint[0])
    if completion_m > max_extrapolate_m:
        return {"points": points, "complete": False, "band": target_band, "completion_m": 0.0, "mode": "anchor_too_far"}

    evidence_slope = estimate_endpoint_slope(points, at_start=at_start)
    if at_start:
        segment = sample_hermite_station_offset(
            start_s=anchor.station,
            start_t=anchor.offset,
            start_slope=0.0,
            end_s=endpoint[0],
            end_t=endpoint[1],
            end_slope=evidence_slope,
            step_m=sample_step_m,
        )
        updated_points = segment[:-1] + list(points)
    else:
        segment = sample_hermite_station_offset(
            start_s=endpoint[0],
            start_t=endpoint[1],
            start_slope=evidence_slope,
            end_s=anchor.station,
            end_t=anchor.offset,
            end_slope=0.0,
            step_m=sample_step_m,
        )
        updated_points = list(points) + segment[1:]
    updated_points.sort(key=lambda item: item[0])
    return {
        "points": updated_points,
        "complete": True,
        "band": target_band,
        "completion_m": completion_m,
        "mode": "switch_anchor_hermite",
        "anchor_id": anchor.anchor_id,
    }


def best_switch_anchor(
    endpoint: tuple[float, float],
    *,
    at_start: bool,
    target: dict[str, Any],
    pair: dict[str, Any],
    switch_anchors: list[SwitchAnchor],
    max_switch_anchor_distance_m: float,
) -> SwitchAnchor | None:
    pair_id = str(pair["pair_id"])
    target_band = str(target["band"])
    target_offset = float(target["offset"])
    matches: list[tuple[float, SwitchAnchor]] = []
    for anchor in switch_anchors:
        if anchor.pair_id and anchor.pair_id != pair_id:
            continue
        if anchor.target_band and anchor.target_band != target_band:
            continue
        if abs(anchor.target_offset - target_offset) > 0.75:
            continue
        station_gap = abs(anchor.station - endpoint[0])
        if station_gap > max_switch_anchor_distance_m:
            continue
        if at_start and anchor.station > endpoint[0] + 5.0:
            continue
        if not at_start and anchor.station < endpoint[0] - 5.0:
            continue
        offset_gap = abs(anchor.offset - target_offset)
        matches.append((station_gap + 3.0 * offset_gap, anchor))
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]


def estimate_endpoint_slope(points: list[tuple[float, float]], *, at_start: bool, window_m: float = 20.0) -> float:
    if len(points) < 2:
        return 0.0
    endpoint_s = points[0][0] if at_start else points[-1][0]
    if at_start:
        window = [(s, t) for s, t in points if s <= endpoint_s + window_m]
    else:
        window = [(s, t) for s, t in points if s >= endpoint_s - window_m]
    if len(window) < 2:
        a, b = (points[0], points[1]) if at_start else (points[-2], points[-1])
        ds = b[0] - a[0]
        return 0.0 if abs(ds) < 1e-9 else (b[1] - a[1]) / ds
    count = len(window)
    sum_s = sum(s for s, _ in window)
    sum_t = sum(t for _, t in window)
    sum_ss = sum(s * s for s, _ in window)
    sum_st = sum(s * t for s, t in window)
    denom = count * sum_ss - sum_s * sum_s
    if abs(denom) < 1e-9:
        return 0.0
    return (count * sum_st - sum_s * sum_t) / denom


def sample_hermite_station_offset(
    *,
    start_s: float,
    start_t: float,
    start_slope: float,
    end_s: float,
    end_t: float,
    end_slope: float,
    step_m: float,
) -> list[tuple[float, float]]:
    span = end_s - start_s
    count = max(2, int(math.ceil(abs(span) / max(step_m, 1.0))) + 1)
    points: list[tuple[float, float]] = []
    for index in range(count):
        u = index / (count - 1)
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        station = start_s + span * u
        offset = h00 * start_t + h10 * span * start_slope + h01 * end_t + h11 * span * end_slope
        points.append((station, offset))
    return points


def proposal_qa_note(start_result: dict[str, Any], end_result: dict[str, Any]) -> str:
    modes = {str(start_result.get("mode", "")), str(end_result.get("mode", ""))}
    if "switch_anchor_hermite" in modes:
        return "raw_evidence_plus_switch_anchor_completion"
    if start_result.get("complete") and end_result.get("complete"):
        return "raw_evidence_observed_to_band"
    return "partial_raw_evidence_no_forced_smoothstep"


def apply_connector_split(
    points: list[tuple[float, float]],
    *,
    pair: dict[str, Any],
    evidence_id: str,
    connector_splits: list[ConnectorSplit],
) -> dict[str, Any]:
    if len(points) < 2:
        return {"points": points, "split_id": "", "straight_tail_m": 0.0}
    pair_id = str(pair["pair_id"])
    start_s = points[0][0]
    end_s = points[-1][0]
    candidates: list[tuple[float, ConnectorSplit]] = []
    for split in connector_splits:
        if split.pair_id and split.pair_id != pair_id:
            continue
        if split.evidence_id and split.evidence_id != evidence_id:
            continue
        if split.station < start_s + 1e-6 or split.station > end_s - 1e-6:
            continue
        offset_at_split = interpolate_offset_at_station(points, split.station)
        offset_gap = abs(offset_at_split - split.offset)
        if offset_gap > 1.5:
            continue
        candidates.append((offset_gap, split))
    if not candidates:
        return {"points": points, "split_id": "", "straight_tail_m": 0.0}

    split = min(candidates, key=lambda item: item[0])[1]
    split_point = (split.station, split.offset)
    side = split.keep_connector_side.lower()
    if side == "north":
        kept = [split_point, *[(s, t) for s, t in points if s > split.station]]
        straight_tail_m = split.station - start_s
    elif side == "south":
        kept = [*(item for item in points if item[0] < split.station), split_point]
        straight_tail_m = end_s - split.station
    else:
        return {"points": points, "split_id": "", "straight_tail_m": 0.0}
    if len(kept) < 2:
        return {"points": points, "split_id": "", "straight_tail_m": 0.0}
    kept.sort(key=lambda item: item[0])
    return {"points": kept, "split_id": split.split_id, "straight_tail_m": max(0.0, straight_tail_m)}


def interpolate_offset_at_station(points: list[tuple[float, float]], station: float) -> float:
    if station <= points[0][0]:
        return points[0][1]
    if station >= points[-1][0]:
        return points[-1][1]
    for (s0, t0), (s1, t1) in zip(points, points[1:]):
        if s0 <= station <= s1:
            if abs(s1 - s0) < 1e-9:
                return t0
            u = (station - s0) / (s1 - s0)
            return t0 + (t1 - t0) * u
    return points[-1][1]


def build_smooth_proposals(
    evidence_features: list[dict[str, Any]],
    *,
    guide: Guide,
    max_extrapolate_m: float,
    max_proposal_span_m: float,
    sample_step_m: float,
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for evidence in evidence_features:
        props = evidence.get("properties") or {}
        pair = pair_by_id(str(props.get("pair_id", "")))
        if pair is None:
            continue
        proposal = build_smooth_proposal(
            evidence,
            pair=pair,
            guide=guide,
            max_extrapolate_m=max_extrapolate_m,
            max_proposal_span_m=max_proposal_span_m,
            sample_step_m=sample_step_m,
        )
        if proposal is None:
            continue
        proposal["properties"]["connector_id"] = f"P{len(proposals) + 1:03d}"
        proposals.append(proposal)
    return proposals


def pair_by_id(pair_id: str) -> dict[str, Any] | None:
    for pair in CONNECTOR_PAIRS:
        if str(pair["pair_id"]) == pair_id:
            return pair
    return None


def build_smooth_proposal(
    evidence: dict[str, Any],
    *,
    pair: dict[str, Any],
    guide: Guide,
    max_extrapolate_m: float,
    max_proposal_span_m: float,
    sample_step_m: float,
) -> dict[str, Any] | None:
    props = evidence.get("properties") or {}
    s0 = safe_float(props.get("station_min_m", 0.0))
    s1 = safe_float(props.get("station_max_m", 0.0))
    t0 = endpoint_offset_at_station(evidence, guide=guide, station=s0)
    t1 = endpoint_offset_at_station(evidence, guide=guide, station=s1)
    if s1 <= s0:
        return None
    slope = (t1 - t0) / (s1 - s0)
    if abs(slope) < 1e-9:
        return None

    low_offset = float(pair["low_offset"])
    high_offset = float(pair["high_offset"])
    if slope >= 0:
        start_offset = low_offset
        end_offset = high_offset
        start_band = str(pair["low_band"])
        end_band = str(pair["high_band"])
    else:
        start_offset = high_offset
        end_offset = low_offset
        start_band = str(pair["high_band"])
        end_band = str(pair["low_band"])

    start_s = s0 + (start_offset - t0) / slope
    end_s = s0 + (end_offset - t0) / slope
    if end_s < start_s:
        start_s, end_s = end_s, start_s
        start_offset, end_offset = end_offset, start_offset
        start_band, end_band = end_band, start_band

    before_m = max(0.0, s0 - start_s)
    after_m = max(0.0, end_s - s1)
    if before_m > max_extrapolate_m or after_m > max_extrapolate_m:
        return None
    if start_s < -5.0 or end_s > guide.length + 5.0:
        return None
    proposal_span = end_s - start_s
    if proposal_span <= 0.0 or proposal_span > max_proposal_span_m:
        return None

    coords = sample_smoothstep_curve(
        guide,
        start_s=start_s,
        end_s=end_s,
        start_offset=start_offset,
        end_offset=end_offset,
        step_m=sample_step_m,
    )
    score = safe_float(props.get("connector_score", 0.0))
    completion_m = before_m + after_m
    completion_penalty = min(0.25, completion_m / max(max_extrapolate_m * 4.0, 1.0))
    proposal_score = max(0.0, round(score - completion_penalty, 4))
    proposal_props = {
        "role": "turnout_connector_proposal",
        "geom_kind": "smooth_tangent_proposal",
        "pair_id": pair["pair_id"],
        "direction": f"{start_band}->{end_band}",
        "source": "smoothstep_from_raw_transition_evidence",
        "review_status": "candidate_needs_dom_review",
        "qa_note": "proposal_not_final_topology",
        "evidence_id": props.get("connector_id", ""),
        "evidence_idx": props.get("feature_index", ""),
        "station_min_m": round(start_s, 3),
        "station_max_m": round(end_s, 3),
        "station_span_m": round(proposal_span, 3),
        "offset_start_m": round(start_offset, 3),
        "offset_end_m": round(end_offset, 3),
        "offset_min_m": round(min(start_offset, end_offset), 3),
        "offset_max_m": round(max(start_offset, end_offset), 3),
        "observed_s0_m": round(s0, 3),
        "observed_s1_m": round(s1, 3),
        "extrap_before_m": round(before_m, 3),
        "extrap_after_m": round(after_m, 3),
        "completion_m": round(completion_m, 3),
        "connector_score": proposal_score,
        "low_band": pair["low_band"],
        "high_band": pair["high_band"],
        "mean_confidence": props.get("mean_confidence", 0.0),
    }
    return {
        "type": "Feature",
        "properties": proposal_props,
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
    }


def endpoint_offset_at_station(feature: dict[str, Any], *, guide: Guide, station: float) -> float:
    station_offsets = [guide.station_offset(coord) for coord in line_coords(feature)]
    if not station_offsets:
        return 0.0
    return min(station_offsets, key=lambda item: abs(item[0] - station))[1]


def sample_smoothstep_curve(
    guide: Guide,
    *,
    start_s: float,
    end_s: float,
    start_offset: float,
    end_offset: float,
    step_m: float,
) -> list[tuple[float, float]]:
    length = max(end_s - start_s, 0.0)
    count = max(2, int(math.ceil(length / max(step_m, 1.0))) + 1)
    coords: list[tuple[float, float]] = []
    for index in range(count):
        u = index / (count - 1)
        smooth = 3.0 * u * u - 2.0 * u * u * u
        station = start_s + length * u
        offset = start_offset + (end_offset - start_offset) * smooth
        coords.append(guide.point_at(station, offset))
    return coords


def summarize_connectors(
    evidence_features: list[dict[str, Any]],
    proposal_features: list[dict[str, Any]],
    *,
    candidate_feature_count: int,
    guide_length_m: float,
) -> dict[str, Any]:
    pair_summaries: list[dict[str, Any]] = []
    for pair in CONNECTOR_PAIRS:
        pair_id = str(pair["pair_id"])
        evidence = [feature for feature in evidence_features if feature["properties"].get("pair_id") == pair_id]
        proposals = [feature for feature in proposal_features if feature["properties"].get("pair_id") == pair_id]
        pair_summaries.append(
            {
                "pair_id": pair_id,
                "low_band": pair["low_band"],
                "high_band": pair["high_band"],
                "evidence_count": len(evidence),
                "proposal_count": len(proposals),
                "station_ranges": [
                    [
                        feature["properties"].get("connector_id", ""),
                        feature["properties"].get("geom_kind", ""),
                        feature["properties"].get("station_min_m", 0.0),
                        feature["properties"].get("station_max_m", 0.0),
                        feature["properties"].get("connector_score", 0.0),
                    ]
                    for feature in [*evidence, *proposals]
                ],
            }
        )
    return {
        "guide_length_m": round(guide_length_m, 3),
        "candidate_feature_count": candidate_feature_count,
        "evidence_count": len(evidence_features),
        "proposal_count": len(proposal_features),
        "pairs": pair_summaries,
    }


def load_switch_anchors(path: Path, *, guide: Guide) -> list[SwitchAnchor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors: list[SwitchAnchor] = []
    for index, feature in enumerate(payload.get("features", []), start=1):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            continue
        coordinates = geometry.get("coordinates") or []
        if len(coordinates) < 2:
            continue
        point = (float(coordinates[0]), float(coordinates[1]))
        props = feature.get("properties") or {}
        pair_id = str(props.get("pair_id", ""))
        target_band = str(props.get("target_band", ""))
        target_offset_value = props.get("target_offset_m")
        if target_offset_value is None:
            inferred = infer_target_offset(pair_id=pair_id, target_band=target_band)
            if inferred is None:
                continue
            target_offset = inferred
        else:
            target_offset = safe_float(target_offset_value)
        station, offset = guide.station_offset(point)
        anchors.append(
            SwitchAnchor(
                anchor_id=str(props.get("anchor_id", f"switch_anchor_{index:03d}")),
                point=point,
                station=station,
                offset=offset,
                pair_id=pair_id,
                target_band=target_band,
                target_offset=target_offset,
                source=str(props.get("source", "manual_switch_anchor")),
            )
        )
    return anchors


def load_connector_splits(path: Path, *, guide: Guide) -> list[ConnectorSplit]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits: list[ConnectorSplit] = []
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
        splits.append(
            ConnectorSplit(
                split_id=str(props.get("split_id", f"connector_split_{index:03d}")),
                point=point,
                station=station,
                offset=offset,
                pair_id=str(props.get("pair_id", "")),
                evidence_id=str(props.get("evidence_id", "")),
                keep_connector_side=str(props.get("keep_connector_side", "")),
                straight_side=str(props.get("straight_side", "")),
                straight_band=str(props.get("straight_band", "")),
                source=str(props.get("source", "manual_connector_split")),
            )
        )
    return splits


def infer_target_offset(*, pair_id: str, target_band: str) -> float | None:
    pairs = CONNECTOR_PAIRS if not pair_id else [pair for pair in CONNECTOR_PAIRS if str(pair["pair_id"]) == pair_id]
    for pair in pairs:
        if str(pair["low_band"]) == target_band:
            return float(pair["low_offset"])
        if str(pair["high_band"]) == target_band:
            return float(pair["high_offset"])
    return None


def load_line_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = []
        for coord in geometry.get("coordinates") or []:
            if len(coord) < 2:
                continue
            x, y = coord[:2]
            if x is None or y is None:
                continue
            coords.append([float(x), float(y)])
        if len(coords) < 2:
            continue
        features.append({"type": "Feature", "properties": feature.get("properties") or {}, "geometry": {"type": "LineString", "coordinates": coords}})
    return features


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, *_ in feature["geometry"]["coordinates"]]


def line_length(coords: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(coords, coords[1:])) if len(coords) >= 2 else 0.0


def write_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    write_json(
        path,
        {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
            "features": features,
        },
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_connector_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
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
    writer.field("s0_m", "F", decimal=3)
    writer.field("s1_m", "F", decimal=3)
    writer.field("t0_m", "F", decimal=3)
    writer.field("t1_m", "F", decimal=3)
    writer.field("score", "F", decimal=4)
    writer.field("conf", "F", decimal=4)
    writer.field("evid_id", "C", size=16)
    writer.field("anchor_id", "C", size=40)
    writer.field("split_id", "C", size=40)
    writer.field("extra_m", "F", decimal=3)
    writer.field("tail_m", "F", decimal=3)
    writer.field("qa_note", "C", size=80)
    for index, feature in enumerate(features):
        props = feature.get("properties") or {}
        coords = line_coords(feature)
        writer.line([coords])
        writer.record(
            str(props.get("connector_id", f"C{index + 1:03d}"))[:16],
            str(props.get("role", ""))[:32],
            str(props.get("geom_kind", ""))[:32],
            str(props.get("pair_id", ""))[:24],
            str(props.get("direction", ""))[:48],
            str(props.get("review_status", ""))[:32],
            safe_float(props.get("station_min_m", 0.0)),
            safe_float(props.get("station_max_m", 0.0)),
            safe_float(props.get("offset_start_m", props.get("offset_min_m", 0.0))),
            safe_float(props.get("offset_end_m", props.get("offset_max_m", 0.0))),
            safe_float(props.get("connector_score", 0.0)),
            safe_float(props.get("mean_confidence", 0.0)),
            str(props.get("evidence_id", ""))[:16],
            str(props.get("anchor_id", ""))[:40],
            str(props.get("split_id", ""))[:40],
            safe_float(props.get("completion_m", 0.0)),
            safe_float(props.get("straight_tail_m", 0.0)),
            str(props.get("qa_note", ""))[:80],
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    write_projection(output_path.with_suffix(".prj"), epsg)


def write_projection(path: Path, epsg: int) -> None:
    import rasterio

    path.write_text(rasterio.crs.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")


def write_connector_qml(path: Path) -> None:
    categories = [
        ("minus_to_main", "0,158,115,255", "minus_to_main"),
        ("main_to_plus", "213,94,0,255", "main_to_plus"),
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
            <Option name="line_width" type="QString" value="0.75"/>
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
  <renderer-v2 type="categorizedSymbol" attr="pair_id" enableorderby="0" forceraster="0" referencescale="-1" symbollevels="0">
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


def write_qa_crops(
    dom_path: Path,
    *,
    connector_features: list[dict[str, Any]],
    band_features: list[dict[str, Any]],
    out_dir: Path,
    crop_width_m: float,
    crop_height_m: float,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw
    import rasterio
    from rasterio.windows import Window

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in list(out_dir.glob("*.png")) + [out_dir / "qa_crops_index.json"]:
        if old_path.exists():
            old_path.unlink()
    targets = build_qa_targets(connector_features)
    overlay_paths: list[str] = []

    with rasterio.open(dom_path) as dataset:
        pixel_width = max(abs(float(dataset.transform.a)), 1e-6)
        pixel_height = max(abs(float(dataset.transform.e)), 1e-6)
        crop_width_px = max(32, int(math.ceil(crop_width_m / pixel_width)))
        crop_height_px = max(32, int(math.ceil(crop_height_m / pixel_height)))
        for target in targets:
            x, y = target["point"]
            row, col = dataset.index(x, y)
            col_off = max(0, min(dataset.width - 1, col - crop_width_px // 2))
            row_off = max(0, min(dataset.height - 1, row - crop_height_px // 2))
            width = min(crop_width_px, dataset.width - col_off)
            height = min(crop_height_px, dataset.height - row_off)
            if width <= 1 or height <= 1:
                continue
            window = Window(col_off, row_off, width, height)
            rgb = read_rgb_window(dataset, window)
            raw = Image.fromarray(rgb, mode="RGB")
            overlay = raw.convert("RGBA")
            draw = ImageDraw.Draw(overlay, "RGBA")
            transform = dataset.window_transform(window)
            draw_features(draw, band_features, transform, width_px=4)
            draw_features(draw, connector_features, transform, width_px=9)
            marker_col, marker_row = ~transform * (x, y)
            draw.ellipse(
                [marker_col - 10, marker_row - 10, marker_col + 10, marker_row + 10],
                fill=(255, 0, 255, 215),
                outline=(255, 255, 255, 255),
                width=3,
            )
            draw_label(draw, str(target["label"]))
            stem = sanitize_filename(str(target["name"]))
            raw_path = out_dir / f"{stem}_raw.png"
            overlay_path = out_dir / f"{stem}_overlay.png"
            raw.save(raw_path)
            overlay.convert("RGB").save(overlay_path)
            overlay_paths.append(str(overlay_path))

    index = {"dom_path": str(dom_path), "count": len(overlay_paths), "overlays": overlay_paths}
    write_json(out_dir / "qa_crops_index.json", index)
    return index


def build_qa_targets(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    proposal_features = [
        feature
        for feature in features
        if feature.get("properties", {}).get("role") == "turnout_connector_proposal"
        or str(feature.get("properties", {}).get("geom_kind", "")).endswith("_proposal")
    ]
    target_features = proposal_features or features
    for feature in target_features:
        coords = line_coords(feature)
        if len(coords) < 2:
            continue
        props = feature.get("properties") or {}
        point = coords[len(coords) // 2]
        targets.append(
            {
                "name": f"{props.get('connector_id', 'connector')}_{props.get('pair_id', 'pair')}",
                "label": f"{props.get('connector_id', '')} {props.get('pair_id', '')} {props.get('geom_kind', '')}",
                "point": point,
            }
        )
    return targets


def read_rgb_window(dataset: Any, window: Any) -> Any:
    import numpy as np

    if dataset.count >= 3:
        arr = dataset.read([1, 2, 3], window=window)
    else:
        single = dataset.read(1, window=window)
        arr = np.stack([single, single, single], axis=0)
    arr = np.moveaxis(arr, 0, -1)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype("float32")
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        if hi > lo:
            arr = (arr - lo) * (255.0 / (hi - lo))
    return np.clip(arr, 0, 255).astype("uint8")


def draw_features(draw: Any, features: list[dict[str, Any]], transform: Any, *, width_px: int) -> None:
    for feature in features:
        props = feature.get("properties") or {}
        color = color_for_feature(props)
        coords = []
        for x, y in line_coords(feature):
            col, row = ~transform * (x, y)
            coords.append((col, row))
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=width_px, joint="curve")


def color_for_feature(props: dict[str, Any]) -> tuple[int, int, int, int]:
    if props.get("role") == "turnout_connector_proposal" or str(props.get("geom_kind", "")).endswith("_proposal"):
        base = PAIR_COLORS.get(str(props.get("pair_id", "")), (255, 255, 255, 255))
        return (base[0], base[1], base[2], 255)
    if props.get("geom_kind") == "raw_transition_evidence":
        return (255, 255, 255, 230)
    return BAND_COLORS.get(str(props.get("band_id", "")), (255, 255, 255, 160))


def draw_label(draw: Any, label: str) -> None:
    box = [8, 8, 720, 76]
    draw.rectangle(box, fill=(0, 0, 0, 210))
    draw.text((18, 20), label[:90], fill=(255, 255, 255, 255))


def sanitize_filename(name: str) -> str:
    safe = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "connector"


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
