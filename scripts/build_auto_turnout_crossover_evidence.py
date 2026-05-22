#!/usr/bin/env python3
"""Build automatic turnout/crossover transition evidence from current semantic candidates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_CANDIDATES = Path("output/dom_centerline_strict_auto_v1/03_rail_candidates/track_centerline_candidates.geojson")
DEFAULT_MAINLINE = Path("output/dom_centerline_strict_auto_v1/06_mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_TRACK_BANDS = Path("output/dom_centerline_strict_auto_v1/07_track_band_priors/track_band_centerline_priors.geojson")
DEFAULT_OUT_DIR = Path("output/dom_centerline_strict_auto_v1/08_auto_turnout_crossover_evidence")
DEFAULT_EPSG = 32651
TRANSITION_CURVE_MODES = ("route_curve", "endpoint_smooth", "evidence_guided")


class Guide:
    def __init__(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0.0:
            raise ValueError("Guide endpoints must be different.")
        self.start = start
        self.end = end
        self.length = length
        self.ux = dx / length
        self.uy = dy / length
        self.nx = -self.uy
        self.ny = self.ux

    def station_offset(self, point: tuple[float, float]) -> tuple[float, float]:
        dx = point[0] - self.start[0]
        dy = point[1] - self.start[1]
        return dx * self.ux + dy * self.uy, dx * self.nx + dy * self.ny

    def point_at(self, station: float, offset: float) -> tuple[float, float]:
        return (
            self.start[0] + station * self.ux + offset * self.nx,
            self.start[1] + station * self.uy + offset * self.ny,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build strict-auto transition evidence without manual turnout anchors.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument("--track-bands", type=Path, default=DEFAULT_TRACK_BANDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-station-span-m", type=float, default=8.0)
    parser.add_argument("--min-offset-span-m", type=float, default=1.2)
    parser.add_argument("--min-abs-slope", type=float, default=0.012)
    parser.add_argument("--max-abs-slope", type=float, default=0.35)
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--max-band-distance-m", type=float, default=1.6)
    parser.add_argument("--fragment-min-offset-span-m", type=float, default=0.75)
    parser.add_argument("--fragment-cluster-gap-m", type=float, default=45.0)
    parser.add_argument("--fragment-band-margin-m", type=float, default=2.3)
    parser.add_argument("--local-context-trend-max-distance-m", type=float, default=0.0)
    parser.add_argument("--local-context-station-margin-m", type=float, default=3.0)
    parser.add_argument("--local-context-offset-margin-m", type=float, default=0.75)
    parser.add_argument("--endpoint-tangent-padding-m", type=float, default=13.0)
    parser.add_argument(
        "--transition-curve-mode",
        choices=TRANSITION_CURVE_MODES,
        default="evidence_guided",
        help=(
            "route_curve locks turnout geometry to the detected track-band endpoints "
            "and treats local DeepLab pieces as evidence, not geometry anchors."
        ),
    )
    parser.add_argument("--route-curve-fraction", type=float, default=0.34)
    parser.add_argument("--curve-step-m", type=float, default=1.0)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    turnout_dir = out_dir / "all_turnout_branch_centerlines"
    gauge_dir = out_dir / "deeplab_gauge_pair_turnouts_v1"
    crossover_dir = out_dir / "deeplab_gauge_pair_crossovers_v1"
    for directory in (turnout_dir, gauge_dir, crossover_dir):
        directory.mkdir(parents=True, exist_ok=True)

    candidates = load_line_features(args.candidates.expanduser().resolve())
    mainline_features = load_line_features(args.mainline.expanduser().resolve())
    if not mainline_features:
        raise RuntimeError(f"No mainline feature found: {args.mainline}")
    guide = Guide(line_coords(mainline_features[0])[0], line_coords(mainline_features[0])[-1])
    band_context = load_band_context(args.track_bands.expanduser().resolve(), guide=guide)
    band_centers = band_context["centers"]
    transitions = detect_fragment_cluster_transitions(
        candidates,
        guide=guide,
        band_centers=band_centers,
        band_intervals=band_context["intervals"],
        min_station_span_m=args.min_station_span_m,
        min_offset_span_m=args.fragment_min_offset_span_m,
        min_abs_slope=args.min_abs_slope,
        max_abs_slope=args.max_abs_slope,
        min_points=args.min_points,
        cluster_gap_m=args.fragment_cluster_gap_m,
        band_margin_m=args.fragment_band_margin_m,
        local_context_trend_max_distance_m=args.local_context_trend_max_distance_m,
        local_context_station_margin_m=args.local_context_station_margin_m,
        local_context_offset_margin_m=args.local_context_offset_margin_m,
        endpoint_tangent_padding_m=args.endpoint_tangent_padding_m,
        transition_curve_mode=args.transition_curve_mode,
        route_curve_fraction=args.route_curve_fraction,
        curve_step_m=args.curve_step_m,
    )
    if not transitions:
        transitions = detect_transitions(
            candidates,
            guide=guide,
            band_centers=band_centers,
            min_station_span_m=args.min_station_span_m,
            min_offset_span_m=args.min_offset_span_m,
            min_abs_slope=args.min_abs_slope,
            max_abs_slope=args.max_abs_slope,
            min_points=args.min_points,
            max_band_distance_m=args.max_band_distance_m,
        )
        detection_mode = "whole_candidate_fallback"
    else:
        detection_mode = "fragment_cluster"
    transitions = dedupe_transitions(transitions)

    turnout_features = [build_turnout_feature(item, index) for index, item in enumerate(transitions, start=1)]
    gauge_features = [build_gauge_feature(item, index, kind="turnout") for index, item in enumerate(transitions, start=1)]
    crossover_features = [
        build_gauge_feature(item, index, kind="crossover")
        for index, item in enumerate(transitions, start=1)
        if item["connector_kind"] == "crossover"
    ]

    turnout_geojson = turnout_dir / "all_turnout_branch_centerlines.geojson"
    gauge_geojson = gauge_dir / "deeplab_gauge_pair_centerlines.geojson"
    crossover_geojson = crossover_dir / "deeplab_gauge_pair_centerlines.geojson"
    write_geojson(turnout_geojson, turnout_features, epsg=args.epsg)
    write_geojson(gauge_geojson, gauge_features, epsg=args.epsg)
    write_geojson(crossover_geojson, crossover_features, epsg=args.epsg)

    summary = {
        "mode": "strict_auto_turnout_crossover_evidence",
        "policy": "Transition evidence is derived from current DeepLab paired-rail candidates only. No manual anchors, review points, retained evidence, or TA-specific rules are used.",
        "inputs": {
            "candidates": str(args.candidates.expanduser().resolve()),
            "mainline": str(args.mainline.expanduser().resolve()),
            "track_bands": str(args.track_bands.expanduser().resolve()),
        },
        "transition_count": len(transitions),
        "detection_mode": detection_mode,
        "transition_curve_mode": args.transition_curve_mode,
        "endpoint_tangent_padding_m": args.endpoint_tangent_padding_m,
        "route_curve_fraction": args.route_curve_fraction,
        "turnout_feature_count": len(turnout_features),
        "gauge_pair_feature_count": len(gauge_features),
        "crossover_feature_count": len(crossover_features),
        "band_centers": band_centers,
        "band_intervals": band_context["intervals"],
        "outputs": {
            "turnouts": str(turnout_geojson),
            "turnout_gauge_pair": str(gauge_geojson),
            "crossover_gauge_pair": str(crossover_geojson),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_line_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        feature
        for feature in payload.get("features", []) or []
        if feature.get("geometry", {}).get("type") == "LineString" and len(feature.get("geometry", {}).get("coordinates", [])) >= 2
    ]


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, *_ in feature["geometry"]["coordinates"]]


def load_band_context(path: Path, *, guide: Guide) -> dict[str, Any]:
    defaults = {"parallel_minus_5m": -5.0, "mainline_2_track": 0.0, "parallel_plus_5m": 5.0, "possible_outer_plus_10m": 10.0}
    if not path.exists():
        return {"centers": defaults, "intervals": []}
    centers = dict(defaults)
    intervals: list[dict[str, Any]] = []
    for feature in load_line_features(path):
        props = feature.get("properties") or {}
        band_id = str(props.get("band_id", ""))
        if not band_id:
            continue
        station_offsets = [guide.station_offset(point) for point in line_coords(feature)]
        if station_offsets:
            offsets = [offset for _, offset in station_offsets]
            stations = [station for station, _ in station_offsets]
            centers[band_id] = float(median(offsets))
            intervals.append(
                {
                    "band_id": band_id,
                    "station_min_m": min(stations),
                    "station_max_m": max(stations),
                    "center_offset_m": float(median(offsets)),
                    "role": str(props.get("role", props.get("network_role", ""))),
                }
            )
    return {"centers": centers, "intervals": intervals}


def load_band_centers(path: Path, *, guide: Guide) -> dict[str, float]:
    return dict(load_band_context(path, guide=guide)["centers"])


def detect_fragment_cluster_transitions(
    features: list[dict[str, Any]],
    *,
    guide: Guide,
    band_centers: dict[str, float],
    band_intervals: list[dict[str, Any]],
    min_station_span_m: float,
    min_offset_span_m: float,
    min_abs_slope: float,
    max_abs_slope: float,
    min_points: int,
    cluster_gap_m: float,
    band_margin_m: float,
    local_context_trend_max_distance_m: float,
    local_context_station_margin_m: float,
    local_context_offset_margin_m: float,
    endpoint_tangent_padding_m: float,
    transition_curve_mode: str = "route_curve",
    route_curve_fraction: float = 0.34,
    curve_step_m: float = 1.0,
) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for feature in features:
        coords = line_coords(feature)
        if len(coords) < min_points:
            continue
        st = sorted((guide.station_offset(point)[0], guide.station_offset(point)[1], point) for point in coords)
        station_span = st[-1][0] - st[0][0]
        offsets = [item[1] for item in st]
        offset_span = max(offsets) - min(offsets)
        if station_span < min_station_span_m or offset_span < min_offset_span_m:
            continue
        slope, intercept = linear_fit([(item[0], item[1]) for item in st])
        if not (min_abs_slope <= abs(slope) <= max_abs_slope):
            continue
        pair = best_adjacent_band_pair(min(offsets), max(offsets), median(offsets), band_centers, margin_m=band_margin_m)
        if pair is None:
            continue
        props = dict(feature.get("properties") or {})
        fragments.append(
            {
                "feature": feature,
                "coords": coords,
                "station_offsets": [(item[0], item[1]) for item in st],
                "pair": pair,
                "station_min_m": st[0][0],
                "station_max_m": st[-1][0],
                "offset_min_m": min(offsets),
                "offset_max_m": max(offsets),
                "offset_span_m": offset_span,
                "slope": slope,
                "intercept": intercept,
                "point_count": len(coords),
                "mean_confidence": float(props.get("mean_confidence", props.get("confidence", 0.0)) or 0.0),
                "source_candidate_id": str(props.get("candidate_id", props.get("line_id", ""))),
            }
        )
    clusters = cluster_fragments(fragments, gap_m=cluster_gap_m)
    transitions = [
        build_cluster_transition(
            cluster,
            guide=guide,
            band_centers=band_centers,
            band_intervals=band_intervals,
            context_features=features,
            local_context_trend_max_distance_m=local_context_trend_max_distance_m,
            local_context_station_margin_m=local_context_station_margin_m,
            local_context_offset_margin_m=local_context_offset_margin_m,
            endpoint_tangent_padding_m=endpoint_tangent_padding_m,
            transition_curve_mode=transition_curve_mode,
            route_curve_fraction=route_curve_fraction,
            curve_step_m=curve_step_m,
        )
        for cluster in clusters
    ]
    return [item for item in transitions if item is not None]


def best_adjacent_band_pair(
    offset_min: float,
    offset_max: float,
    offset_median: float,
    band_centers: dict[str, float],
    *,
    margin_m: float,
) -> tuple[str, str] | None:
    ordered = sorted(band_centers.items(), key=lambda item: item[1])
    best: tuple[float, str, str] | None = None
    for (left_band, left_offset), (right_band, right_offset) in zip(ordered, ordered[1:]):
        lo = min(left_offset, right_offset)
        hi = max(left_offset, right_offset)
        overlap = max(0.0, min(offset_max, hi) - max(offset_min, lo))
        if overlap <= 0.0 and not (lo - margin_m <= offset_median <= hi + margin_m):
            continue
        score = overlap - 0.02 * abs(offset_median - (lo + hi) / 2.0)
        if best is None or score > best[0]:
            best = (score, left_band, right_band)
    if best is None:
        return None
    return best[1], best[2]


def cluster_fragments(fragments: list[dict[str, Any]], *, gap_m: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    pair_keys = sorted({tuple(item["pair"]) for item in fragments})
    for pair in pair_keys:
        current: list[dict[str, Any]] = []
        items = sorted([item for item in fragments if tuple(item["pair"]) == pair], key=lambda item: item["station_min_m"])
        for item in items:
            if not current or item["station_min_m"] - max(part["station_max_m"] for part in current) <= gap_m:
                current.append(item)
            else:
                clusters.append(current)
                current = [item]
        if current:
            clusters.append(current)
    return clusters


def build_cluster_transition(
    cluster: list[dict[str, Any]],
    *,
    guide: Guide,
    band_centers: dict[str, float],
    band_intervals: list[dict[str, Any]],
    context_features: list[dict[str, Any]],
    local_context_trend_max_distance_m: float,
    local_context_station_margin_m: float,
    local_context_offset_margin_m: float,
    endpoint_tangent_padding_m: float,
    transition_curve_mode: str,
    route_curve_fraction: float,
    curve_step_m: float,
) -> dict[str, Any] | None:
    pair = tuple(cluster[0]["pair"])
    points = [point for item in cluster for point in item["station_offsets"]]
    if len(points) < 4:
        return None
    left_band, right_band = pair
    left_offset = band_centers[left_band]
    right_offset = band_centers[right_band]
    low_offset = min(left_offset, right_offset)
    high_offset = max(left_offset, right_offset)
    inner_points = [(s, o) for s, o in points if low_offset + 0.45 <= o <= high_offset - 0.45]
    fit_points = inner_points if len(inner_points) >= 4 else points
    slope, intercept = linear_fit(fit_points)
    if abs(slope) < 1e-6:
        return None

    lower_band, upper_band = (left_band, right_band) if left_offset < right_offset else (right_band, left_band)
    start_band, end_band = (lower_band, upper_band) if slope > 0 else (upper_band, lower_band)

    start_station = station_for_offset(band_centers[start_band], slope, intercept, points)
    end_station = station_for_offset(band_centers[end_band], slope, intercept, points)
    if start_station > end_station:
        start_station, end_station = end_station, start_station
        start_band, end_band = end_band, start_band

    start_band = active_or_next_band(
        start_band,
        station=start_station,
        band_centers=band_centers,
        band_intervals=band_intervals,
        search_direction=-1 if band_centers[start_band] < band_centers[end_band] else 1,
    )
    end_band = active_or_next_band(
        end_band,
        station=end_station,
        band_centers=band_centers,
        band_intervals=band_intervals,
        search_direction=-1 if band_centers[end_band] < band_centers[start_band] else 1,
    )
    start_station = station_for_offset(band_centers[start_band], slope, intercept, points)
    end_station = station_for_offset(band_centers[end_band], slope, intercept, points)
    if start_station > end_station:
        start_station, end_station = end_station, start_station
        start_band, end_band = end_band, start_band

    start_diag_interval = diagnostic_interval_for_band(start_band, band_intervals, station=start_station)
    end_diag_interval = diagnostic_interval_for_band(end_band, band_intervals, station=end_station)
    if start_diag_interval is not None:
        start_station = min(start_station, float(start_diag_interval["station_min_m"]))
    else:
        start_station -= endpoint_tangent_padding_m
    if end_diag_interval is not None:
        end_station = max(end_station, float(end_diag_interval["station_max_m"]))
    else:
        end_station += endpoint_tangent_padding_m
    if end_station <= start_station:
        return None

    start_offset = band_centers[start_band]
    end_offset = band_centers[end_band]
    evidence_points = [(station, offset) for item in cluster for station, offset in item["station_offsets"]]
    local_context_points, local_context_ids = collect_local_context_points(
        context_features,
        guide=guide,
        station_start=start_station,
        station_end=end_station,
        offset_start=start_offset,
        offset_end=end_offset,
        slope=slope,
        intercept=intercept,
        max_trend_distance_m=local_context_trend_max_distance_m,
        station_margin_m=local_context_station_margin_m,
        offset_margin_m=local_context_offset_margin_m,
    )
    curve_evidence_points = dedupe_station_offset_points(evidence_points + local_context_points)
    coords = build_transition_curve(
        guide,
        station_start=start_station,
        offset_start=start_offset,
        station_end=end_station,
        offset_end=end_offset,
        evidence_points=curve_evidence_points,
        step_m=curve_step_m,
        endpoint_tangent_padding_m=endpoint_tangent_padding_m,
        transition_curve_mode=transition_curve_mode,
        route_curve_fraction=route_curve_fraction,
    )
    evidence_coords = [guide.point_at(station, offset) for station, offset in sorted(curve_evidence_points)]
    source_ids = ",".join(str(item["source_candidate_id"]) for item in cluster if item["source_candidate_id"])
    connector_kind = classify_connector(start_band, end_band)
    return {
        "feature": cluster[0]["feature"],
        "coords": coords,
        "evidence_coords": evidence_coords,
        "start_band": start_band,
        "end_band": end_band,
        "connector_kind": connector_kind,
        "station_min_m": start_station,
        "station_max_m": end_station,
        "offset_start_m": start_offset,
        "offset_end_m": end_offset,
        "offset_span_m": abs(end_offset - start_offset),
        "slope": slope,
        "point_count": sum(int(item["point_count"]) for item in cluster),
        "mean_confidence": sum(float(item["mean_confidence"]) for item in cluster) / len(cluster),
        "source_candidate_id": source_ids,
        "local_context_candidate_id": ",".join(local_context_ids),
        "local_context_point_count": len(local_context_points),
        "transition_curve_mode": transition_curve_mode,
        "route_curve_fraction": route_curve_fraction,
        "curve_evidence_point_count": len(curve_evidence_points),
        "fragment_count": len(cluster),
        "fragment_station_min_m": min(float(item["station_min_m"]) for item in cluster),
        "fragment_station_max_m": max(float(item["station_max_m"]) for item in cluster),
    }


def collect_local_context_points(
    features: list[dict[str, Any]],
    *,
    guide: Guide,
    station_start: float,
    station_end: float,
    offset_start: float,
    offset_end: float,
    slope: float,
    intercept: float,
    max_trend_distance_m: float,
    station_margin_m: float,
    offset_margin_m: float,
) -> tuple[list[tuple[float, float]], list[str]]:
    """Collect short raw candidate samples that support a transition corridor.

    The fragment cluster detector intentionally requires enough offset span to
    discover a turnout. Once the turnout is known, short candidate pieces are
    still valuable local center constraints and should stop the generated curve
    from interpolating across a detected rail-center segment.
    """

    if max_trend_distance_m <= 0.0:
        return [], []
    lo_offset = min(offset_start, offset_end) - offset_margin_m
    hi_offset = max(offset_start, offset_end) + offset_margin_m
    lo_station = station_start - station_margin_m
    hi_station = station_end + station_margin_m
    points: list[tuple[float, float]] = []
    source_ids: list[str] = []
    seen_ids: set[str] = set()
    for feature_index, feature in enumerate(features):
        props = dict(feature.get("properties") or {})
        mean_gap_m = props.get("mean_gap_m")
        if mean_gap_m is not None:
            try:
                gap_m = float(mean_gap_m)
            except (TypeError, ValueError):
                gap_m = 0.0
            if gap_m and not (1.25 <= gap_m <= 1.80):
                continue
        feature_points: list[tuple[float, float]] = []
        for point in line_coords(feature):
            station, offset = guide.station_offset(point)
            if not (lo_station <= station <= hi_station and lo_offset <= offset <= hi_offset):
                continue
            trend_offset = slope * station + intercept
            if abs(offset - trend_offset) > max_trend_distance_m:
                continue
            feature_points.append((station, offset))
        if not feature_points:
            continue
        points.extend(feature_points)
        source_id = context_source_id(props, feature_index)
        if source_id not in seen_ids:
            source_ids.append(source_id)
            seen_ids.add(source_id)
    return points, source_ids


def context_source_id(props: dict[str, Any], feature_index: int) -> str:
    image_name = str(props.get("image_name", "")).replace(".png", "")
    candidate_id = props.get("candidate_id", props.get("line_id", feature_index))
    if image_name:
        return f"{image_name}:{candidate_id}"
    return str(candidate_id)


def dedupe_station_offset_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    seen: set[tuple[int, int]] = set()
    result: list[tuple[float, float]] = []
    for station, offset in sorted(points):
        key = (round(station, 3), round(offset, 3))
        if key in seen:
            continue
        seen.add(key)
        result.append((station, offset))
    return result


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denom = sum((point[0] - mean_x) ** 2 for point in points)
    if denom <= 1e-9:
        return 0.0, mean_y
    slope = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points) / denom
    return slope, mean_y - slope * mean_x


def station_for_offset(offset: float, slope: float, intercept: float, fallback_points: list[tuple[float, float]]) -> float:
    if abs(slope) <= 1e-9:
        return median([point[0] for point in fallback_points])
    return (offset - intercept) / slope


def active_or_next_band(
    band_id: str,
    *,
    station: float,
    band_centers: dict[str, float],
    band_intervals: list[dict[str, Any]],
    search_direction: int,
) -> str:
    if band_active_near(band_id, station, band_intervals):
        return band_id
    ordered = [band for band, _ in sorted(band_centers.items(), key=lambda item: item[1])]
    if band_id not in ordered:
        return band_id
    index = ordered.index(band_id)
    step = 1 if search_direction >= 0 else -1
    while 0 <= index + step < len(ordered):
        index += step
        candidate = ordered[index]
        if band_active_near(candidate, station, band_intervals):
            return candidate
    return band_id


def band_active_near(band_id: str, station: float, band_intervals: list[dict[str, Any]], *, tolerance_m: float = 80.0) -> bool:
    return any(
        str(item.get("band_id", "")) == band_id
        and float(item.get("station_max_m", 0.0)) >= station - tolerance_m
        and float(item.get("station_min_m", 0.0)) <= station + tolerance_m
        for item in band_intervals
    )


def diagnostic_interval_for_band(
    band_id: str,
    band_intervals: list[dict[str, Any]],
    *,
    station: float,
    tolerance_m: float = 90.0,
) -> dict[str, Any] | None:
    for item in band_intervals:
        if str(item.get("band_id", "")) != band_id:
            continue
        if str(item.get("role", "")) != "diagnostic_candidate":
            continue
        if float(item.get("station_min_m", 0.0)) - tolerance_m <= station <= float(item.get("station_max_m", 0.0)) + tolerance_m:
            return item
    return None


def build_smooth_curve(
    guide: Guide,
    *,
    station_start: float,
    offset_start: float,
    station_end: float,
    offset_end: float,
    step_m: float,
) -> list[tuple[float, float]]:
    span = station_end - station_start
    count = max(8, int(math.ceil(span / max(step_m, 0.25))) + 1)
    coords: list[tuple[float, float]] = []
    for index in range(count):
        t = index / (count - 1)
        station = station_start + span * t
        smooth = t * t * (3.0 - 2.0 * t)
        offset = offset_start + (offset_end - offset_start) * smooth
        coords.append(guide.point_at(station, offset))
    return coords


def build_transition_curve(
    guide: Guide,
    *,
    station_start: float,
    offset_start: float,
    station_end: float,
    offset_end: float,
    evidence_points: list[tuple[float, float]],
    step_m: float,
    endpoint_tangent_padding_m: float,
    transition_curve_mode: str,
    route_curve_fraction: float,
) -> list[tuple[float, float]]:
    if transition_curve_mode == "evidence_guided":
        return build_evidence_guided_curve(
            guide,
            station_start=station_start,
            offset_start=offset_start,
            station_end=station_end,
            offset_end=offset_end,
            evidence_points=evidence_points,
            step_m=step_m,
            endpoint_tangent_padding_m=endpoint_tangent_padding_m,
        )
    if transition_curve_mode == "endpoint_smooth":
        return build_smooth_curve(
            guide,
            station_start=station_start,
            offset_start=offset_start,
            station_end=station_end,
            offset_end=offset_end,
            step_m=step_m,
        )
    if transition_curve_mode != "route_curve":
        raise ValueError(f"Unsupported transition curve mode: {transition_curve_mode}")
    station_offsets = curve_straight_curve_station_offsets(
        (station_start, offset_start),
        (station_end, offset_end),
        curve_fraction=route_curve_fraction,
        step_m=step_m,
    )
    return [guide.point_at(station, offset) for station, offset in station_offsets]


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
    middle_slope = 1.0 / (1.0 - c)
    points: list[tuple[float, float]] = []
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


def build_evidence_guided_curve(
    guide: Guide,
    *,
    station_start: float,
    offset_start: float,
    station_end: float,
    offset_end: float,
    evidence_points: list[tuple[float, float]],
    step_m: float,
    endpoint_tangent_padding_m: float,
) -> list[tuple[float, float]]:
    anchors = evidence_guided_anchors(
        station_start=station_start,
        offset_start=offset_start,
        station_end=station_end,
        offset_end=offset_end,
        evidence_points=evidence_points,
        endpoint_tangent_padding_m=endpoint_tangent_padding_m,
    )
    span = station_end - station_start
    count = max(8, int(math.ceil(span / max(step_m, 0.25))) + 1)
    coords: list[tuple[float, float]] = []
    for index in range(count):
        station = station_start + span * index / (count - 1)
        offset = interpolate_anchor_offset(anchors, station)
        coords.append(guide.point_at(station, offset))
    return coords


def evidence_guided_anchors(
    *,
    station_start: float,
    offset_start: float,
    station_end: float,
    offset_end: float,
    evidence_points: list[tuple[float, float]],
    endpoint_tangent_padding_m: float,
    bin_m: float = 2.0,
) -> list[tuple[float, float]]:
    lo_offset = min(offset_start, offset_end) - 0.75
    hi_offset = max(offset_start, offset_end) + 0.75
    filtered = [
        (station, offset)
        for station, offset in evidence_points
        if station_start < station < station_end and lo_offset <= offset <= hi_offset
    ]
    bins: dict[int, list[tuple[float, float]]] = {}
    for station, offset in filtered:
        bins.setdefault(int((station - station_start) / bin_m), []).append((station, offset))
    anchors: list[tuple[float, float]] = [(station_start, offset_start)]
    if endpoint_tangent_padding_m > 0 and station_start + endpoint_tangent_padding_m < station_end:
        anchors.append((station_start + endpoint_tangent_padding_m, offset_start))
    for bucket in sorted(bins):
        points = bins[bucket]
        anchors.append((median([point[0] for point in points]), median([point[1] for point in points])))
    if endpoint_tangent_padding_m > 0 and station_end - endpoint_tangent_padding_m > station_start:
        anchors.append((station_end - endpoint_tangent_padding_m, offset_end))
    anchors.append((station_end, offset_end))
    return enforce_anchor_monotonicity(dedupe_anchor_stations(anchors), decreasing=offset_end < offset_start)


def dedupe_anchor_stations(anchors: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not anchors:
        return []
    result: list[tuple[float, float]] = []
    for station, offset in sorted(anchors):
        if result and abs(station - result[-1][0]) < 1e-6:
            result[-1] = (station, offset)
        else:
            result.append((station, offset))
    return result


def enforce_anchor_monotonicity(anchors: list[tuple[float, float]], *, decreasing: bool) -> list[tuple[float, float]]:
    if not anchors:
        return anchors
    result: list[tuple[float, float]] = []
    last_offset = anchors[0][1]
    for station, offset in anchors:
        if decreasing:
            offset = min(offset, last_offset)
        else:
            offset = max(offset, last_offset)
        result.append((station, offset))
        last_offset = offset
    if result:
        result[-1] = anchors[-1]
    return result


def interpolate_anchor_offset(anchors: list[tuple[float, float]], station: float) -> float:
    if not anchors:
        return 0.0
    if station <= anchors[0][0]:
        return anchors[0][1]
    if station >= anchors[-1][0]:
        return anchors[-1][1]
    for left, right in zip(anchors, anchors[1:]):
        if left[0] <= station <= right[0]:
            span = right[0] - left[0]
            if span <= 1e-9:
                return right[1]
            ratio = (station - left[0]) / span
            return left[1] + (right[1] - left[1]) * ratio
    return anchors[-1][1]


def detect_transitions(
    features: list[dict[str, Any]],
    *,
    guide: Guide,
    band_centers: dict[str, float],
    min_station_span_m: float,
    min_offset_span_m: float,
    min_abs_slope: float,
    max_abs_slope: float,
    min_points: int,
    max_band_distance_m: float,
) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for feature in features:
        coords = line_coords(feature)
        if len(coords) < min_points:
            continue
        st = sorted((guide.station_offset(point)[0], guide.station_offset(point)[1], point) for point in coords)
        station_span = st[-1][0] - st[0][0]
        offsets = [item[1] for item in st]
        offset_span = max(offsets) - min(offsets)
        if station_span < min_station_span_m or offset_span < min_offset_span_m:
            continue
        slope = (st[-1][1] - st[0][1]) / station_span if station_span else 0.0
        if not (min_abs_slope <= abs(slope) <= max_abs_slope):
            continue
        start_band, start_dist = nearest_band(st[0][1], band_centers)
        end_band, end_dist = nearest_band(st[-1][1], band_centers)
        if start_band == end_band or start_dist > max_band_distance_m or end_dist > max_band_distance_m:
            continue
        props = dict(feature.get("properties") or {})
        connector_kind = classify_connector(start_band, end_band)
        transitions.append(
            {
                "feature": feature,
                "coords": [item[2] for item in st],
                "start_band": start_band,
                "end_band": end_band,
                "connector_kind": connector_kind,
                "station_min_m": st[0][0],
                "station_max_m": st[-1][0],
                "offset_start_m": st[0][1],
                "offset_end_m": st[-1][1],
                "offset_span_m": offset_span,
                "slope": slope,
                "point_count": len(coords),
                "mean_confidence": float(props.get("mean_confidence", props.get("confidence", 0.0)) or 0.0),
                "source_candidate_id": str(props.get("candidate_id", props.get("line_id", ""))),
            }
        )
    transitions.sort(key=lambda item: (float(item["station_min_m"]), float(item["offset_start_m"])))
    return dedupe_transitions(transitions)


def nearest_band(offset: float, band_centers: dict[str, float]) -> tuple[str, float]:
    band_id, center = min(band_centers.items(), key=lambda item: abs(offset - item[1]))
    return band_id, abs(offset - center)


def classify_connector(start_band: str, end_band: str) -> str:
    if start_band != "mainline_2_track" and end_band != "mainline_2_track":
        return "crossover"
    return "turnout"


def dedupe_transitions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        duplicate = False
        for kept in result:
            same_bands = item["start_band"] == kept["start_band"] and item["end_band"] == kept["end_band"]
            close_station = abs(float(item["station_min_m"]) - float(kept["station_min_m"])) < 6.0
            if same_bands and close_station:
                duplicate = True
                if float(item["point_count"]) > float(kept["point_count"]):
                    kept.update(item)
                break
        if not duplicate:
            result.append(item)
    return result


def build_turnout_feature(item: dict[str, Any], index: int) -> dict[str, Any]:
    branch_id = f"AUTO_{index:03d}"
    coords = [[round(x, 6), round(y, 6)] for x, y in item["coords"]]
    return {
        "type": "Feature",
        "properties": {
            "line_id": f"TURNOUT_{branch_id}",
            "branch_id": branch_id,
            "connector_id": branch_id,
            "network_role": "turnout_connector",
            "source_type": "strict_auto_semseg_transition",
            "source": "current_deeplab_candidate_transition",
            "geom_kind": "automatic_transition_centerline",
            "connector_kind": item["connector_kind"],
            "start_band": item["start_band"],
            "end_band": item["end_band"],
            "station_min_m": round(float(item["station_min_m"]), 3),
            "station_max_m": round(float(item["station_max_m"]), 3),
            "offset_start_m": round(float(item["offset_start_m"]), 3),
            "offset_end_m": round(float(item["offset_end_m"]), 3),
            "offset_span_m": round(float(item["offset_span_m"]), 3),
            "transition_slope": round(float(item["slope"]), 6),
            "source_candidate_id": item["source_candidate_id"],
            "local_context_candidate_id": item.get("local_context_candidate_id", ""),
            "local_context_point_count": int(item.get("local_context_point_count", 0)),
            "transition_curve_mode": item.get("transition_curve_mode", ""),
            "route_curve_fraction": round(float(item.get("route_curve_fraction", 0.0)), 3),
            "curve_evidence_point_count": int(item.get("curve_evidence_point_count", 0)),
            "point_count": int(item["point_count"]),
            "fragment_count": int(item.get("fragment_count", 1)),
            "fragment_station_min_m": round(float(item.get("fragment_station_min_m", item["station_min_m"])), 3),
            "fragment_station_max_m": round(float(item.get("fragment_station_max_m", item["station_max_m"])), 3),
            "qa_status": "strict_auto_unreviewed",
            "review_status": "not_user_reviewed",
        },
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def build_gauge_feature(item: dict[str, Any], index: int, *, kind: str) -> dict[str, Any]:
    branch_id = f"AUTO_{index:03d}"
    suffix = "CX" if kind == "crossover" else "GP"
    coords = [[round(x, 6), round(y, 6)] for x, y in item.get("evidence_coords", item["coords"])]
    return {
        "type": "Feature",
        "properties": {
            "seq_id": f"{branch_id}_{suffix}01",
            "branch_id": branch_id,
            "connector_id": branch_id,
            "role": "deeplab_gauge_pair_centerline",
            "source": "strict_auto_current_deeplab_gauge_pair",
            "source_type": "strict_auto_semseg_transition",
            "connector_kind": item["connector_kind"],
            "start_band": item["start_band"],
            "end_band": item["end_band"],
            "station_min_m": round(float(item["station_min_m"]), 3),
            "station_max_m": round(float(item["station_max_m"]), 3),
            "offset_span_m": round(float(item["offset_span_m"]), 3),
            "source_candidate_id": item["source_candidate_id"],
            "local_context_candidate_id": item.get("local_context_candidate_id", ""),
            "local_context_point_count": int(item.get("local_context_point_count", 0)),
            "transition_curve_mode": item.get("transition_curve_mode", ""),
            "route_curve_fraction": round(float(item.get("route_curve_fraction", 0.0)), 3),
            "curve_evidence_point_count": int(item.get("curve_evidence_point_count", 0)),
            "fragment_count": int(item.get("fragment_count", 1)),
        },
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def write_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
