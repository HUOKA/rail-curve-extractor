from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_CANDIDATES = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_candidates/track_centerline_candidates.geojson")
DEFAULT_MAINLINE = Path("output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/track_band_priors")
DEFAULT_DOM = Path("data/生产数据/无人机数据/正射/dom.tif")
DEFAULT_EPSG = 32651


BANDS = [
    {
        "band_id": "parallel_minus_5m",
        "track_hint": "t=-5m 平行股道（待定 1/3股道）",
        "center": -5.0,
        "half_width": 1.1,
        "role": "parallel_track",
    },
    {"band_id": "mainline_2_track", "track_hint": "2股道", "center": 0.0, "half_width": 1.1, "role": "accepted_mainline"},
    {
        "band_id": "parallel_plus_5m",
        "track_hint": "t=+5m 平行股道（待定 1/3股道）",
        "center": 5.0,
        "half_width": 1.1,
        "role": "parallel_track",
    },
    {
        "band_id": "possible_outer_plus_10m",
        "track_hint": "外侧局部候选待核查",
        "center": 10.0,
        "half_width": 0.9,
        "role": "diagnostic_candidate",
    },
]

BAND_COLORS = {
    "parallel_minus_5m": (0, 114, 178, 255),
    "mainline_2_track": (255, 0, 0, 255),
    "parallel_plus_5m": (230, 159, 0, 255),
    "possible_outer_plus_10m": (131, 56, 236, 255),
}

# These are deliberately narrow exceptions. They were checked on full-resolution
# DOM crops: the side tracks remain straight near s=1810m, but visual clutter
# or guard-rail-like internal structure suppresses raw centerline candidates.
REVIEWED_BRIDGE_ZONES = [
    {
        "band_id": "parallel_minus_5m",
        "station_min_m": 1700.0,
        "station_max_m": 1865.0,
        "max_gap_m": 140.0,
        "qa_note": "reviewed_s1810_straight_gap",
    },
    {
        "band_id": "parallel_plus_5m",
        "station_min_m": 1740.0,
        "station_max_m": 1865.0,
        "max_gap_m": 105.0,
        "qa_note": "reviewed_s1810_straight_gap",
    },
]

USER_REVIEW_POINTS = [
    {"name": "user_gap_minus_t5", "point": (315433.49, 3521254.97)},
    {"name": "user_gap_plus_t5", "point": (315425.61, 3521267.38)},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify raw DOM centerline candidates into station track bands.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument(
        "--turnout-exclusions",
        type=Path,
        default=None,
        help="Optional turnout/branch LineString layer; straight-band support inside these station windows is ignored.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM, help="Original DOM used for full-resolution QA crops.")
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument("--merge-gap-m", type=float, default=40.0)
    parser.add_argument("--local-offset-window-m", type=float, default=25.0)
    parser.add_argument(
        "--mainline-local-offset-window-m",
        type=float,
        default=50.0,
        help="Sliding station window used to center the 2-track mainline on semantic-segmentation support.",
    )
    parser.add_argument(
        "--mainline-max-local-correction-m",
        type=float,
        default=0.08,
        help="Clamp for local 2-track mainline correction relative to the automatic guide.",
    )
    parser.add_argument("--min-station-span-m", type=float, default=5.0)
    parser.add_argument(
        "--min-centerline-span-m",
        type=float,
        default=35.0,
        help="Minimum support-bounded side-band interval length promoted to a straight-track prior.",
    )
    parser.add_argument(
        "--allow-reviewed-bridges",
        action="store_true",
        help="Compatibility/debug only: allow previously reviewed straight-gap bridge zones.",
    )
    parser.add_argument(
        "--include-review-point-crops",
        action="store_true",
        help="QA only: include historical user review points in crop output.",
    )
    parser.add_argument("--qa-crop-width-m", type=float, default=105.0)
    parser.add_argument("--qa-crop-height-m", type=float, default=105.0)
    parser.add_argument("--skip-qa-crops", action="store_true")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidates_path = args.candidates.expanduser().resolve()
    mainline_path = args.mainline.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mainline_coords = line_coords(load_line_features(mainline_path)[0])
    guide = Guide(mainline_coords[0], mainline_coords[-1])
    candidate_features = load_line_features(candidates_path)
    classified, band_offsets = classify_candidates(
        candidate_features,
        guide=guide,
        min_station_span_m=args.min_station_span_m,
    )
    turnout_exclusions = load_turnout_exclusion_windows(args.turnout_exclusions, guide=guide)
    if turnout_exclusions:
        classified = filter_turnout_overlap_support(classified, turnout_exclusions)
    centerline_features = build_band_centerlines(
        classified,
        guide=guide,
        band_offsets=band_offsets,
        sample_step_m=args.sample_step_m,
        merge_gap_m=args.merge_gap_m,
        local_offset_window_m=args.local_offset_window_m,
        mainline_local_offset_window_m=args.mainline_local_offset_window_m,
        mainline_max_local_correction_m=args.mainline_max_local_correction_m,
        min_centerline_span_m=args.min_centerline_span_m,
        allow_reviewed_bridges=args.allow_reviewed_bridges,
    )

    centerline_geojson = out_dir / "track_band_centerline_priors.geojson"
    support_geojson = out_dir / "track_band_support_candidates.geojson"
    write_geojson(centerline_geojson, centerline_features, epsg=args.epsg)
    write_geojson(support_geojson, classified, epsg=args.epsg)
    write_track_band_shapefile(centerline_features, centerline_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_track_band_shapefile(classified, support_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_categorized_qml(centerline_geojson.with_suffix(".qml"))
    write_categorized_qml(support_geojson.with_suffix(".qml"), width_mm=0.35)

    dom_path = args.dom.expanduser().resolve()
    qa_summary: dict[str, Any] | None = None
    if not args.skip_qa_crops and dom_path.exists():
        qa_summary = write_qa_crops(
            dom_path,
            centerline_features,
            guide=guide,
            out_dir=out_dir / "qa_crops",
            crop_width_m=args.qa_crop_width_m,
            crop_height_m=args.qa_crop_height_m,
            include_review_points=args.include_review_point_crops,
        )

    summary = summarize_bands(centerline_features, classified, guide=guide)
    summary.update(
        {
            "mainline": str(mainline_path),
            "candidates": str(candidates_path),
            "dom": str(dom_path),
            "guide_length_m": round(guide.length, 3),
            "merge_gap_m": args.merge_gap_m,
            "local_offset_window_m": args.local_offset_window_m,
            "mainline_local_offset_window_m": args.mainline_local_offset_window_m,
            "mainline_max_local_correction_m": args.mainline_max_local_correction_m,
            "min_station_span_m": args.min_station_span_m,
            "min_centerline_span_m": args.min_centerline_span_m,
            "turnout_exclusion_count": len(turnout_exclusions),
            "side_band_policy": {
                "ordinary_merge_gap_m": args.merge_gap_m,
                "allow_reviewed_bridges": bool(args.allow_reviewed_bridges),
                "known_bridge_zones": REVIEWED_BRIDGE_ZONES if args.allow_reviewed_bridges else [],
                "rule": "side bands are support-bounded; reviewed bridge zones are compatibility/debug only",
            },
            "outputs": {
                "centerline_geojson": str(centerline_geojson),
                "centerline_shp": str(centerline_geojson.with_suffix(".shp")),
                "support_geojson": str(support_geojson),
                "support_shp": str(support_geojson.with_suffix(".shp")),
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


def classify_candidates(
    features: list[dict[str, Any]],
    *,
    guide: Guide,
    min_station_span_m: float,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    classified: list[dict[str, Any]] = []
    band_offsets: dict[str, list[float]] = {str(band["band_id"]): [] for band in BANDS}
    for feature in features:
        coords = line_coords(feature)
        station_offsets = [guide.station_offset(coord) for coord in coords]
        stations = [item[0] for item in station_offsets]
        offsets = [item[1] for item in station_offsets]
        if not stations:
            continue
        median_offset = float(median(offsets))
        band = assign_band(median_offset)
        if band is None:
            continue

        center = float(band["center"])
        half_width = float(band["half_width"])
        in_band = [(s, t) for s, t in station_offsets if 0.0 <= s <= guide.length and abs(t - center) <= half_width + 0.65]
        if len(in_band) >= 2:
            station_values = [item[0] for item in in_band]
            offset_values = [item[1] for item in in_band]
        else:
            station_values = stations
            offset_values = offsets
        station_min = max(0.0, min(station_values))
        station_max = min(guide.length, max(station_values))
        station_span = station_max - station_min
        if station_span < min_station_span_m:
            continue

        props = dict(feature.get("properties") or {})
        props.update(
            {
                "role": "track_band_support_candidate",
                "band_id": band["band_id"],
                "track_hint": band["track_hint"],
                "band_role": band["role"],
                "source": "raw_dom_centerline_candidate",
                "station_min_m": round(station_min, 3),
                "station_max_m": round(station_max, 3),
                "station_span_m": round(station_span, 3),
                "median_offset_m": round(median_offset, 4),
                "band_center_m": band["center"],
                "in_band_point_count": len(in_band),
            }
        )
        classified.append({"type": "Feature", "properties": props, "geometry": feature["geometry"]})
        band_offsets[str(band["band_id"])].extend(offset_values)
    return classified, band_offsets


def assign_band(offset_m: float) -> dict[str, Any] | None:
    matches = [band for band in BANDS if abs(offset_m - float(band["center"])) <= float(band["half_width"])]
    if not matches:
        return None
    return min(matches, key=lambda band: abs(offset_m - float(band["center"])))


def load_turnout_exclusion_windows(path: Path | None, *, guide: Guide) -> list[dict[str, float]]:
    if path is None:
        return []
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return []
    windows: list[dict[str, float]] = []
    for feature in load_line_features(resolved):
        props = feature.get("properties") or {}
        coords = line_coords(feature)
        stations = [guide.station_offset(coord)[0] for coord in coords]
        if not stations:
            continue
        station_min = safe_float(props.get("station_min_m", min(stations)))
        station_max = safe_float(props.get("station_max_m", max(stations)))
        if station_max < station_min:
            station_min, station_max = station_max, station_min
        padding = 8.0
        windows.append(
            {
                "station_min_m": max(0.0, station_min - padding),
                "station_max_m": min(guide.length, station_max + padding),
            }
        )
    return windows


def filter_turnout_overlap_support(
    classified: list[dict[str, Any]],
    turnout_windows: list[dict[str, float]],
    *,
    overlap_fraction: float = 0.25,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for feature in classified:
        props = feature.get("properties") or {}
        band_id = str(props.get("band_id", ""))
        if band_id == "mainline_2_track":
            filtered.append(feature)
            continue
        station_min = safe_float(props.get("station_min_m", 0.0))
        station_max = safe_float(props.get("station_max_m", 0.0))
        span = max(1e-6, station_max - station_min)
        overlap = 0.0
        for window in turnout_windows:
            overlap += max(0.0, min(station_max, window["station_max_m"]) - max(station_min, window["station_min_m"]))
        if overlap / span >= overlap_fraction:
            continue
        filtered.append(feature)
    return filtered


def build_band_centerlines(
    classified: list[dict[str, Any]],
    *,
    guide: Guide,
    band_offsets: dict[str, list[float]],
    sample_step_m: float,
    merge_gap_m: float,
    local_offset_window_m: float,
    min_centerline_span_m: float,
    mainline_local_offset_window_m: float = 50.0,
    mainline_max_local_correction_m: float = 0.08,
    allow_reviewed_bridges: bool = False,
) -> list[dict[str, Any]]:
    centerline_features: list[dict[str, Any]] = []
    by_band: dict[str, list[dict[str, Any]]] = {}
    for feature in classified:
        by_band.setdefault(str(feature["properties"]["band_id"]), []).append(feature)

    for band in BANDS:
        band_id = str(band["band_id"])
        features = by_band.get(band_id, [])
        if band_id == "mainline_2_track":
            samples = collect_station_offset_samples(features, guide=guide, band=band)
            interval_items = [
                {
                    "station_min_m": 0.0,
                    "station_max_m": guide.length,
                    "support_feature_count": len(features),
                    "bridge_gap_count": 0,
                    "max_bridged_gap_m": 0.0,
                    "qa_note": "semseg_support_straight_fit_mainline" if samples else "semseg_auto_mainline_no_local_support",
                }
            ]
            default_offset = 0.0
            fit_mode = "robust_straight_line" if samples else "fixed_mainline"
            source = "raw_dom_support_straight_line_fit" if samples else "semseg_auto_mainline"
        else:
            raw_intervals = [
                (
                    float(feature["properties"]["station_min_m"]),
                    float(feature["properties"]["station_max_m"]),
                    1,
                )
                for feature in features
            ]
            interval_items = merge_intervals_with_review_bridges(
                raw_intervals,
                gap_m=merge_gap_m,
                band_id=band_id,
                allow_reviewed_bridges=allow_reviewed_bridges,
            )
            interval_items = [
                item
                for item in interval_items
                if float(item["station_max_m"]) - float(item["station_min_m"]) >= min_centerline_span_m
            ]
            samples = collect_station_offset_samples(features, guide=guide, band=band)
            profile = build_offset_profile(samples)
            default_offset = profile_median_offset(profile, fallback_offsets=band_offsets.get(band_id, []), band=band)
            fit_mode = "robust_straight_line"
            source = "raw_dom_support_bounded_straight_line_fit"

        for interval_id, item in enumerate(interval_items):
            start_s = float(item["station_min_m"])
            end_s = float(item["station_max_m"])
            if end_s <= start_s:
                continue
            if fit_mode == "fixed_mainline":
                coords = sample_shifted_line(guide, start_s, end_s, offset_m=default_offset, step_m=sample_step_m)
                fit_report = straight_fit_report_from_constant(default_offset, start_s=start_s, end_s=end_s)
            elif fit_mode == "robust_straight_line":
                interval_samples = [sample for sample in samples if start_s <= sample[0] <= end_s]
                fit_report = fit_straight_offset_line(
                    interval_samples,
                    default_offset=default_offset,
                    start_s=start_s,
                    end_s=end_s,
                )
                coords = sample_straight_fit_line(
                    guide,
                    start_s,
                    end_s,
                    fit_report=fit_report,
                    step_m=sample_step_m,
                )
            else:
                coords = sample_profiled_line(
                    guide,
                    start_s,
                    end_s,
                    profile=profile,
                    default_offset=default_offset,
                    window_m=local_offset_window_m,
                    step_m=sample_step_m,
                )
                fit_report = straight_fit_report_from_constant(
                    float(median([guide.station_offset(coord)[1] for coord in coords])) if coords else default_offset,
                    start_s=start_s,
                    end_s=end_s,
                )
            offsets = [guide.station_offset(coord)[1] for coord in coords]
            centerline_features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "role": band["role"],
                        "band_id": band_id,
                        "track_hint": band["track_hint"],
                        "source": source,
                        "fit_mode": fit_mode,
                        "interval_id": interval_id,
                        "offset_m": round(float(median(offsets)) if offsets else default_offset, 4),
                        "straight_slope": round(float(fit_report["slope_offset_per_m"]), 8),
                        "straight_samples": int(fit_report["sample_count"]),
                        "straight_inliers": int(fit_report["inlier_count"]),
                        "straight_rms_m": round(float(fit_report["rms_m"]), 4),
                        "local_offset_window_m": round(
                            local_offset_window_m if fit_mode == "local_station_offset" else 0.0,
                            3,
                        ),
                        "max_local_correction_m": round(
                            mainline_max_local_correction_m if fit_mode == "guarded_mainline_local_offset" else 0.0,
                            3,
                        ),
                        "station_min_m": round(start_s, 3),
                        "station_max_m": round(end_s, 3),
                        "length_m": round(line_length(coords), 3),
                        "support_feature_count": int(item["support_feature_count"]),
                        "bridge_gap_count": int(item.get("bridge_gap_count", 0)),
                        "max_bridged_gap_m": round(float(item.get("max_bridged_gap_m", 0.0)), 3),
                        "bridge_mid_m": round(float(item.get("bridge_mid_m", 0.0)), 3),
                        "qa_note": item.get("qa_note", "support_bounded_no_turnout_extension"),
                    },
                    "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
                }
            )
    return centerline_features


def collect_station_offset_samples(
    features: list[dict[str, Any]],
    *,
    guide: Guide,
    band: dict[str, Any],
) -> list[tuple[float, float]]:
    center = float(band["center"])
    max_offset_delta = float(band["half_width"]) + 0.65
    samples: list[tuple[float, float]] = []
    for feature in features:
        for coord in line_coords(feature):
            station, offset = guide.station_offset(coord)
            if 0.0 <= station <= guide.length and abs(offset - center) <= max_offset_delta:
                samples.append((station, offset))
    return sorted(samples)


def build_offset_profile(samples: list[tuple[float, float]]) -> dict[str, list[float]]:
    return {
        "stations": [float(station) for station, _ in samples],
        "offsets": [float(offset) for _, offset in samples],
    }


def profile_median_offset(profile: dict[str, list[float]], *, fallback_offsets: list[float], band: dict[str, Any]) -> float:
    offsets = profile.get("offsets") or fallback_offsets
    if offsets:
        return float(median(offsets))
    return float(band["center"])


def fit_straight_offset_line(
    samples: list[tuple[float, float]],
    *,
    default_offset: float,
    start_s: float,
    end_s: float,
    min_inliers: int = 8,
) -> dict[str, float | int]:
    center_station = (start_s + end_s) / 2.0
    if len(samples) < 2:
        return straight_fit_report_from_constant(default_offset, start_s=start_s, end_s=end_s, sample_count=len(samples))

    active = list(samples)
    slope = 0.0
    intercept = default_offset
    for _iteration in range(5):
        slope, intercept = least_squares_offset_line(active, center_station=center_station)
        residuals = [offset - (intercept + slope * (station - center_station)) for station, offset in active]
        if len(residuals) < min_inliers:
            break
        residual_median = float(median(residuals))
        abs_deviation = [abs(value - residual_median) for value in residuals]
        mad = float(median(abs_deviation)) if abs_deviation else 0.0
        threshold = max(0.06, min(0.25, 3.0 * 1.4826 * mad))
        next_active = [
            sample
            for sample, residual in zip(active, residuals)
            if abs(residual - residual_median) <= threshold
        ]
        if len(next_active) < min(min_inliers, len(samples)):
            break
        if len(next_active) == len(active):
            break
        active = next_active

    slope, intercept = least_squares_offset_line(active, center_station=center_station)
    residuals = [offset - (intercept + slope * (station - center_station)) for station, offset in active]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals)) if residuals else 0.0
    return {
        "center_station_m": center_station,
        "intercept_offset_m": float(intercept),
        "slope_offset_per_m": float(slope),
        "sample_count": len(samples),
        "inlier_count": len(active),
        "rms_m": float(rms),
    }


def least_squares_offset_line(samples: list[tuple[float, float]], *, center_station: float) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    x_values = [station - center_station for station, _offset in samples]
    y_values = [offset for _station, offset in samples]
    if len(samples) < 2:
        return 0.0, float(y_values[0])
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    sxx = sum((x - mean_x) * (x - mean_x) for x in x_values)
    if sxx <= 1e-12:
        return 0.0, float(median(y_values))
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return float(slope), float(intercept)


def straight_fit_report_from_constant(
    offset_m: float,
    *,
    start_s: float,
    end_s: float,
    sample_count: int = 0,
) -> dict[str, float | int]:
    return {
        "center_station_m": (start_s + end_s) / 2.0,
        "intercept_offset_m": float(offset_m),
        "slope_offset_per_m": 0.0,
        "sample_count": sample_count,
        "inlier_count": sample_count,
        "rms_m": 0.0,
    }


def sample_straight_fit_line(
    guide: Guide,
    start_s: float,
    end_s: float,
    *,
    fit_report: dict[str, float | int],
    step_m: float,
) -> list[tuple[float, float]]:
    length = max(end_s - start_s, 0.0)
    count = max(2, int(math.ceil(length / max(step_m, 1.0))) + 1)
    center_station = float(fit_report["center_station_m"])
    intercept = float(fit_report["intercept_offset_m"])
    slope = float(fit_report["slope_offset_per_m"])
    coords: list[tuple[float, float]] = []
    for index in range(count):
        station = start_s + length * index / (count - 1)
        offset = intercept + slope * (station - center_station)
        coords.append(guide.point_at(station, offset_m=offset))
    return coords


def sample_profiled_line(
    guide: Guide,
    start_s: float,
    end_s: float,
    *,
    profile: dict[str, list[float]],
    default_offset: float,
    window_m: float,
    step_m: float,
    max_correction_m: float | None = None,
) -> list[tuple[float, float]]:
    length = max(end_s - start_s, 0.0)
    count = max(2, int(math.ceil(length / max(step_m, 1.0))) + 1)
    coords: list[tuple[float, float]] = []
    for index in range(count):
        station = start_s + length * index / (count - 1)
        offset = estimate_offset_at(station, profile=profile, default_offset=default_offset, window_m=window_m)
        if max_correction_m is not None:
            offset = clamp_local_offset(offset, default_offset=default_offset, max_correction_m=max_correction_m)
        coords.append(guide.point_at(station, offset_m=offset))
    return coords


def clamp_local_offset(offset: float, *, default_offset: float, max_correction_m: float) -> float:
    max_abs = abs(float(max_correction_m))
    return default_offset + max(-max_abs, min(max_abs, offset - default_offset))


def estimate_offset_at(
    station: float,
    *,
    profile: dict[str, list[float]],
    default_offset: float,
    window_m: float,
) -> float:
    stations = profile.get("stations") or []
    offsets = profile.get("offsets") or []
    if not stations:
        return default_offset

    left_index = bisect_left(stations, station - window_m)
    right_index = bisect_right(stations, station + window_m)
    if right_index > left_index:
        return float(median(offsets[left_index:right_index]))

    insertion = bisect_left(stations, station)
    left = insertion - 1
    right = insertion
    if left >= 0 and right < len(stations):
        span = stations[right] - stations[left]
        if span <= 180.0 and span > 0:
            ratio = (station - stations[left]) / span
            return float(offsets[left] + (offsets[right] - offsets[left]) * ratio)
    if left >= 0 and abs(station - stations[left]) <= 80.0:
        return float(offsets[left])
    if right < len(stations) and abs(stations[right] - station) <= 80.0:
        return float(offsets[right])
    return default_offset


def merge_intervals(intervals: list[tuple[float, float, int]], *, gap_m: float) -> list[tuple[float, float, int]]:
    if not intervals:
        return []
    intervals = sorted((min(a, b), max(a, b), count) for a, b, count in intervals)
    merged: list[tuple[float, float, int]] = [intervals[0]]
    for start, end, count in intervals[1:]:
        prev_start, prev_end, prev_count = merged[-1]
        if start <= prev_end + gap_m:
            merged[-1] = (prev_start, max(prev_end, end), prev_count + count)
        else:
            merged.append((start, end, count))
    return merged


def merge_intervals_with_review_bridges(
    intervals: list[tuple[float, float, int]],
    *,
    gap_m: float,
    band_id: str,
    allow_reviewed_bridges: bool = False,
) -> list[dict[str, Any]]:
    if not intervals:
        return []
    normalized = sorted((min(a, b), max(a, b), count) for a, b, count in intervals)
    merged: list[dict[str, Any]] = [
        {
            "station_min_m": normalized[0][0],
            "station_max_m": normalized[0][1],
            "support_feature_count": normalized[0][2],
            "bridge_gap_count": 0,
            "max_bridged_gap_m": 0.0,
            "bridge_mid_m": 0.0,
            "qa_note": "support_bounded_no_turnout_extension",
        }
    ]
    for start, end, count in normalized[1:]:
        current = merged[-1]
        prev_end = float(current["station_max_m"])
        gap = start - prev_end
        bridge = reviewed_bridge_for_gap(band_id, prev_end, start) if allow_reviewed_bridges else None
        if start <= prev_end + gap_m or bridge is not None:
            current["station_max_m"] = max(prev_end, end)
            current["support_feature_count"] = int(current["support_feature_count"]) + count
            if bridge is not None and gap > gap_m:
                current["bridge_gap_count"] = int(current["bridge_gap_count"]) + 1
                if gap >= float(current["max_bridged_gap_m"]):
                    current["max_bridged_gap_m"] = gap
                    current["bridge_mid_m"] = (prev_end + start) / 2.0
                current["qa_note"] = str(bridge["qa_note"])
        else:
            merged.append(
                {
                    "station_min_m": start,
                    "station_max_m": end,
                    "support_feature_count": count,
                    "bridge_gap_count": 0,
                    "max_bridged_gap_m": 0.0,
                    "bridge_mid_m": 0.0,
                    "qa_note": "support_bounded_no_turnout_extension",
                }
            )
    return merged


def reviewed_bridge_for_gap(band_id: str, gap_start_m: float, gap_end_m: float) -> dict[str, Any] | None:
    gap_len = gap_end_m - gap_start_m
    if gap_len <= 0:
        return None
    for zone in REVIEWED_BRIDGE_ZONES:
        if str(zone["band_id"]) != band_id:
            continue
        if gap_len > float(zone["max_gap_m"]):
            continue
        if gap_start_m >= float(zone["station_min_m"]) and gap_end_m <= float(zone["station_max_m"]):
            return zone
    return None


def sample_shifted_line(guide: Guide, start_s: float, end_s: float, *, offset_m: float, step_m: float) -> list[tuple[float, float]]:
    length = max(end_s - start_s, 0.0)
    count = max(2, int(math.ceil(length / max(step_m, 1.0))) + 1)
    return [guide.point_at(start_s + length * index / (count - 1), offset_m=offset_m) for index in range(count)]


def line_length(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(coords, coords[1:]))


def summarize_bands(centerline_features: list[dict[str, Any]], support_features: list[dict[str, Any]], *, guide: Guide) -> dict[str, Any]:
    band_summaries: list[dict[str, Any]] = []
    for band in BANDS:
        band_id = str(band["band_id"])
        centers = [feature for feature in centerline_features if feature["properties"]["band_id"] == band_id]
        supports = [feature for feature in support_features if feature["properties"]["band_id"] == band_id]
        band_summaries.append(
            {
                "band_id": band_id,
                "role": band["role"],
                "track_hint": band["track_hint"],
                "center_offset_m": band["center"],
                "support_feature_count": len(supports),
                "support_point_count": sum(int(feature["properties"].get("point_count", 0)) for feature in supports),
                "centerline_interval_count": len(centers),
                "centerline_total_length_m": round(sum(float(feature["properties"]["length_m"]) for feature in centers), 3),
                "bridge_gap_count": sum(int(feature["properties"].get("bridge_gap_count", 0)) for feature in centers),
                "max_bridged_gap_m": round(
                    max((float(feature["properties"].get("max_bridged_gap_m", 0.0)) for feature in centers), default=0.0),
                    3,
                ),
                "station_ranges": [
                    [
                        feature["properties"]["station_min_m"],
                        feature["properties"]["station_max_m"],
                        feature["properties"].get("qa_note", ""),
                    ]
                    for feature in centers
                ],
            }
        )
    return {"guide_length_m": round(guide.length, 3), "bands": band_summaries}


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
        features.append(
            {
                "type": "Feature",
                "properties": feature.get("properties") or {},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )
    return features


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, *_ in feature["geometry"]["coordinates"]]


def write_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    write_json(
        path,
        {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}} ,
            "features": features,
        },
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_track_band_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pyshp in the active virtual environment.") from exc
    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("band_id", "C", size=32)
    writer.field("role", "C", size=32)
    writer.field("trk_hint", "C", size=50)
    writer.field("source", "C", size=40)
    writer.field("fit_mode", "C", size=32)
    writer.field("int_id", "N", decimal=0)
    writer.field("off_m", "F", decimal=4)
    writer.field("s0_m", "F", decimal=3)
    writer.field("s1_m", "F", decimal=3)
    writer.field("sup_n", "N", decimal=0)
    writer.field("len_m", "F", decimal=3)
    writer.field("bridge_n", "N", decimal=0)
    writer.field("gap_max", "F", decimal=3)
    writer.field("br_mid", "F", decimal=3)
    writer.field("qa_note", "C", size=80)
    writer.field("conf", "F", decimal=4)
    for index, feature in enumerate(features):
        props = feature.get("properties") or {}
        coords = line_coords(feature)
        writer.line([coords])
        writer.record(
            str(props.get("band_id", ""))[:32],
            str(props.get("role", props.get("band_role", "")))[:32],
            str(props.get("track_hint", ""))[:50],
            str(props.get("source", ""))[:40],
            str(props.get("fit_mode", ""))[:32],
            safe_int(props.get("interval_id", index)),
            safe_float(props.get("offset_m", props.get("median_offset_m", 0.0))),
            safe_float(props.get("station_min_m", 0.0)),
            safe_float(props.get("station_max_m", 0.0)),
            safe_int(props.get("support_feature_count", props.get("near_point_count", 0))),
            safe_float(props.get("length_m", props.get("station_span_m", 0.0))),
            safe_int(props.get("bridge_gap_count", 0)),
            safe_float(props.get("max_bridged_gap_m", 0.0)),
            safe_float(props.get("bridge_mid_m", 0.0)),
            str(props.get("qa_note", ""))[:80],
            safe_float(props.get("mean_confidence", 0.0)),
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    write_projection(output_path.with_suffix(".prj"), epsg)


def safe_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_projection(path: Path, epsg: int) -> None:
    import rasterio

    path.write_text(rasterio.crs.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")


def write_categorized_qml(path: Path, *, width_mm: float = 0.65) -> None:
    categories = [
        ("parallel_minus_5m", "0,114,178,255", "parallel_minus_5m"),
        ("mainline_2_track", "255,0,0,255", "mainline_2_track"),
        ("parallel_plus_5m", "230,159,0,255", "parallel_plus_5m"),
        ("possible_outer_plus_10m", "131,56,236,255", "possible_outer_plus_10m"),
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
            <Option name="line_width" type="QString" value="{width_mm}"/>
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
  <renderer-v2 type="categorizedSymbol" attr="band_id" enableorderby="0" forceraster="0" referencescale="-1" symbollevels="0">
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
    features: list[dict[str, Any]],
    *,
    guide: Guide,
    out_dir: Path,
    crop_width_m: float,
    crop_height_m: float,
    include_review_points: bool = False,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw
    import rasterio
    from rasterio.windows import Window

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in list(out_dir.glob("*.png")) + [out_dir / "qa_crops_index.json"]:
        if old_path.exists():
            old_path.unlink()
    targets = build_qa_targets(features, guide=guide, include_review_points=include_review_points)
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
            window_transform = dataset.window_transform(window)
            draw_features_on_crop(draw, features, window_transform)
            marker_col, marker_row = ~window_transform * (x, y)
            draw.ellipse(
                [marker_col - 11, marker_row - 11, marker_col + 11, marker_row + 11],
                fill=(255, 0, 255, 210),
                outline=(255, 255, 255, 255),
                width=3,
            )
            label = f"{target['name']} s={target['station_m']:.1f} t={target['offset_m']:.1f}"
            draw_label(draw, label)
            stem = sanitize_filename(str(target["name"]))
            raw_path = out_dir / f"{stem}_raw.png"
            overlay_path = out_dir / f"{stem}_overlay.png"
            raw.save(raw_path)
            overlay.convert("RGB").save(overlay_path)
            overlay_paths.append(str(overlay_path))

    index = {"dom_path": str(dom_path), "count": len(overlay_paths), "overlays": overlay_paths}
    write_json(out_dir / "qa_crops_index.json", index)
    return index


def build_qa_targets(features: list[dict[str, Any]], *, guide: Guide, include_review_points: bool = False) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    if include_review_points and USER_REVIEW_POINTS:
        avg_x = sum(float(item["point"][0]) for item in USER_REVIEW_POINTS) / len(USER_REVIEW_POINTS)
        avg_y = sum(float(item["point"][1]) for item in USER_REVIEW_POINTS) / len(USER_REVIEW_POINTS)
        station, offset = guide.station_offset((avg_x, avg_y))
        targets.append({"name": "user_gap_pair_s1814", "point": (avg_x, avg_y), "station_m": station, "offset_m": offset})
        for item in USER_REVIEW_POINTS:
            station, offset = guide.station_offset(item["point"])
            targets.append({"name": item["name"], "point": item["point"], "station_m": station, "offset_m": offset})

    for feature in features:
        props = feature.get("properties") or {}
        band_id = str(props.get("band_id", ""))
        if band_id == "mainline_2_track":
            continue
        coords = line_coords(feature)
        if not coords:
            continue
        for edge, point in (("start", coords[0]), ("end", coords[-1])):
            station, offset = guide.station_offset(point)
            name = f"{band_id}_{edge}_s{int(round(station))}"
            targets.append({"name": name, "point": point, "station_m": station, "offset_m": offset})
        if int(props.get("bridge_gap_count", 0)) > 0:
            station = float(props.get("bridge_mid_m") or (float(props["station_min_m"]) + float(props["station_max_m"])) / 2.0)
            point = guide.point_at(station, float(props.get("offset_m", 0.0)))
            targets.append(
                {
                    "name": f"{band_id}_reviewed_bridge_s{int(round(station))}",
                    "point": point,
                    "station_m": station,
                    "offset_m": float(props.get("offset_m", 0.0)),
                }
            )
    return dedupe_targets(targets)


def dedupe_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for target in targets:
        name = str(target["name"])
        if name in seen:
            continue
        seen.add(name)
        result.append(target)
    return result


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


def draw_features_on_crop(draw: Any, features: list[dict[str, Any]], window_transform: Any) -> None:
    for feature in features:
        props = feature.get("properties") or {}
        band_id = str(props.get("band_id", ""))
        color = BAND_COLORS.get(band_id, (255, 255, 255, 255))
        coords = []
        for x, y in line_coords(feature):
            col, row = ~window_transform * (x, y)
            coords.append((col, row))
        if len(coords) >= 2:
            draw.line(coords, fill=color, width=7, joint="curve")


def draw_label(draw: Any, label: str) -> None:
    box = [8, 8, 560, 78]
    draw.rectangle(box, fill=(0, 0, 0, 210))
    draw.text((18, 20), label, fill=(255, 255, 255, 255))


def sanitize_filename(name: str) -> str:
    safe = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "crop"


if __name__ == "__main__":
    raise SystemExit(main())
