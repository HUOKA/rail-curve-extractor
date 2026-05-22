from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_CANDIDATES = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_candidates/track_centerline_candidates.geojson")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/mainline_prior")
DEFAULT_START = (315112.328, 3519475.270)
DEFAULT_END = (315617.422, 3522319.160)
DEFAULT_EPSG = 32651


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a QGIS-ready mainline prior from semantic-segmentation centerline candidates.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--mode",
        choices=["semseg-auto", "manual"],
        default="semseg-auto",
        help="semseg-auto derives the guide from candidate centerlines; manual uses the supplied endpoints.",
    )
    parser.add_argument("--start-x", type=float, default=DEFAULT_START[0])
    parser.add_argument("--start-y", type=float, default=DEFAULT_START[1])
    parser.add_argument("--end-x", type=float, default=DEFAULT_END[0])
    parser.add_argument("--end-y", type=float, default=DEFAULT_END[1])
    parser.add_argument("--track-id", default="2股道")
    parser.add_argument("--corridor-m", type=float, default=2.0)
    parser.add_argument("--min-near-points", type=int, default=2)
    parser.add_argument("--min-station-span-m", type=float, default=5.0)
    parser.add_argument("--sample-step-m", type=float, default=10.0)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    candidates_path = args.candidates.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_features = load_line_features(candidates_path)
    auto_report: dict[str, Any] | None = None
    if args.mode == "manual":
        guide = Guide((args.start_x, args.start_y), (args.end_x, args.end_y))
        guide_source = "user_qgis_pick"
        guide_role = "manual_mainline_guide"
    else:
        guide, auto_report = build_semseg_auto_guide(candidate_features)
        guide_source = "semseg_candidate_hough_track_peak"
        guide_role = "semseg_auto_mainline_guide"

    support_features, support_offsets = select_support_features(
        candidate_features,
        guide=guide,
        track_id=args.track_id,
        corridor_m=args.corridor_m,
        min_near_points=args.min_near_points,
        min_station_span_m=args.min_station_span_m,
    )
    fitted_offset = float(median(support_offsets)) if support_offsets else 0.0
    connected_line = guide.sample_shifted_line(offset_m=fitted_offset, step_m=args.sample_step_m)

    guide_feature = {
        "type": "Feature",
        "properties": {
            "role": guide_role,
            "track_id": args.track_id,
            "source": guide_source,
            "epsg": args.epsg,
            "corridor_m": args.corridor_m,
            "length_m": round(guide.length, 3),
        },
        "geometry": {"type": "LineString", "coordinates": [list(guide.start), list(guide.end)]},
    }
    connected_feature = {
        "type": "Feature",
        "properties": {
            "role": "connected_mainline_prior",
            "track_id": args.track_id,
            "source": f"{guide_source}_plus_candidate_fit",
            "epsg": args.epsg,
            "offset_m": round(fitted_offset, 4),
            "support_feature_count": len(support_features),
            "support_point_count": len(support_offsets),
            "corridor_m": args.corridor_m,
            "length_m": round(guide.length, 3),
        },
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in connected_line]},
    }

    guide_geojson = out_dir / "mainline_2_track_guide.geojson"
    connected_geojson = out_dir / "mainline_2_track_connected.geojson"
    support_geojson = out_dir / "mainline_2_track_support_candidates.geojson"
    package_geojson = out_dir / "mainline_2_track_package.geojson"
    write_geojson(guide_geojson, [guide_feature])
    write_geojson(connected_geojson, [connected_feature])
    write_geojson(support_geojson, support_features)
    write_geojson(package_geojson, [guide_feature, connected_feature, *support_features])

    for geojson_path, color, width_mm in [
        (guide_geojson, "255,170,0,255", 0.55),
        (connected_geojson, "255,0,0,255", 0.8),
        (support_geojson, "0,114,178,255", 0.35),
        (package_geojson, "255,0,0,255", 0.65),
    ]:
        shp_path = geojson_path.with_suffix(".shp")
        write_shapefile(load_line_features(geojson_path), shp_path, epsg=args.epsg)
        write_qgis_line_style(shp_path.with_suffix(".qml"), color=color, width_mm=width_mm)

    summary = {
        "mode": args.mode,
        "track_id": args.track_id,
        "start": [round(guide.start[0], 6), round(guide.start[1], 6)],
        "end": [round(guide.end[0], 6), round(guide.end[1], 6)],
        "epsg": args.epsg,
        "guide_source": guide_source,
        "guide_length_m": round(guide.length, 3),
        "corridor_m": args.corridor_m,
        "candidate_feature_count": len(candidate_features),
        "support_feature_count": len(support_features),
        "support_point_count": len(support_offsets),
        "fitted_offset_m": round(fitted_offset, 4),
        "auto_report": auto_report,
        "outputs": {
            "guide_geojson": str(guide_geojson),
            "connected_geojson": str(connected_geojson),
            "support_geojson": str(support_geojson),
            "package_geojson": str(package_geojson),
            "connected_shp": str(connected_geojson.with_suffix(".shp")),
            "support_shp": str(support_geojson.with_suffix(".shp")),
        },
    }
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
        station = dx * self.ux + dy * self.uy
        offset = dx * self.nx + dy * self.ny
        return station, offset

    def point_at(self, station: float, offset_m: float = 0.0) -> tuple[float, float]:
        return (
            self.start[0] + self.ux * station + self.nx * offset_m,
            self.start[1] + self.uy * station + self.ny * offset_m,
        )

    def sample_shifted_line(self, offset_m: float, step_m: float) -> list[tuple[float, float]]:
        step = max(step_m, 1.0)
        count = max(2, int(math.ceil(self.length / step)) + 1)
        return [self.point_at(self.length * index / (count - 1), offset_m=offset_m) for index in range(count)]


def build_semseg_auto_guide(
    features: list[dict[str, Any]],
    *,
    offset_bin_m: float = 0.20,
    cluster_width_m: float = 0.80,
    min_peak_count: int = 1000,
    peak_spacing_m: float = 2.0,
    station_gap_m: float = 40.0,
    neighbor_min_m: float = 3.5,
    neighbor_max_m: float = 6.5,
    neighbor_overlap_m: float = 300.0,
) -> tuple[Guide, dict[str, Any]]:
    points = collect_points(features)
    if len(points) < 2:
        raise ValueError("Not enough semantic-segmentation candidate points to build an automatic mainline guide.")
    initial_theta = estimate_dominant_orientation(features)
    orientation_points = evenly_sample_points(points, max_points=12000)
    theta = refine_orientation_by_offset_histogram(orientation_points, initial_theta, bin_m=offset_bin_m)
    peaks = detect_offset_peaks(
        points,
        theta,
        bin_m=offset_bin_m,
        min_peak_count=min_peak_count,
        peak_spacing_m=peak_spacing_m,
    )
    if not peaks:
        raise ValueError("Could not detect a dominant straight-track offset peak from semantic-segmentation candidates.")
    clusters = summarize_track_peak_clusters(
        points,
        peaks,
        theta,
        cluster_width_m=cluster_width_m,
        station_gap_m=station_gap_m,
    )
    selected_index = choose_mainline_cluster(
        clusters,
        neighbor_min_m=neighbor_min_m,
        neighbor_max_m=neighbor_max_m,
        neighbor_overlap_m=neighbor_overlap_m,
    )
    selected = clusters[selected_index]
    selected_points = list(selected.get("xy_points") or [])
    line_theta = estimate_point_cloud_orientation(selected_points, fallback_theta=theta)
    line_ux, line_uy = math.cos(line_theta), math.sin(line_theta)
    line_nx, line_ny = -line_uy, line_ux
    station_offsets = [(x * line_ux + y * line_uy, x * line_nx + y * line_ny) for x, y in selected_points]
    start_s, end_s = float(selected["station_min_m"]), float(selected["station_max_m"])
    t_center = float(selected["offset_m"])
    if station_offsets:
        stations = [station for station, _offset in station_offsets]
        offsets = [offset for _station, offset in station_offsets]
        start_s, end_s = largest_station_interval(stations, gap_m=station_gap_m)
        t_center = float(median(offsets))
    ux, uy, nx, ny = line_ux, line_uy, line_nx, line_ny
    start = (ux * start_s + nx * t_center, uy * start_s + ny * t_center)
    end = (ux * end_s + nx * t_center, uy * end_s + ny * t_center)
    guide = Guide(start, end)
    report = {
        "initial_orientation_deg": round(math.degrees(initial_theta), 6),
        "refined_orientation_deg": round(math.degrees(theta), 6),
        "selected_line_orientation_deg": round(math.degrees(line_theta), 6),
        "offset_bin_m": offset_bin_m,
        "cluster_width_m": cluster_width_m,
        "selected_cluster_index": selected_index,
        "selected_offset_m": round(t_center, 6),
        "selected_station_min_m": round(start_s, 3),
        "selected_station_max_m": round(end_s, 3),
        "track_peak_count": len(clusters),
        "track_peaks": [
            {
                "index": index,
                "offset_m": round(cluster["offset_m"], 6),
                "point_count": int(cluster["point_count"]),
                "station_span_m": round(cluster["station_max_m"] - cluster["station_min_m"], 3),
                "coverage_m": round(cluster["coverage_m"], 3),
                "two_sided_neighbor": bool(cluster.get("two_sided_neighbor", False)),
            }
            for index, cluster in enumerate(clusters)
        ],
    }
    return guide, report


def collect_points(features: list[dict[str, Any]]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for feature in features:
        points.extend(line_coords(feature))
    return points


def evenly_sample_points(points: list[tuple[float, float]], *, max_points: int) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def estimate_dominant_orientation(features: list[dict[str, Any]]) -> float:
    sum_x = 0.0
    sum_y = 0.0
    for feature in features:
        coords = line_coords(feature)
        for a, b in zip(coords, coords[1:]):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            segment_len = math.hypot(dx, dy)
            if segment_len < 0.5:
                continue
            theta = math.atan2(dy, dx)
            sum_x += segment_len * math.cos(2.0 * theta)
            sum_y += segment_len * math.sin(2.0 * theta)
    if abs(sum_x) <= 1e-12 and abs(sum_y) <= 1e-12:
        raise ValueError("Could not estimate dominant track orientation from candidate segments.")
    theta = 0.5 * math.atan2(sum_y, sum_x)
    if theta < 0:
        theta += math.pi
    return theta


def refine_orientation_by_offset_histogram(
    points: list[tuple[float, float]],
    initial_theta: float,
    *,
    bin_m: float,
    search_deg: float = 0.8,
    coarse_step_deg: float = 0.002,
    fine_step_deg: float = 0.0001,
) -> float:
    best_theta = initial_theta
    best_score = -1.0
    steps = int(round((2.0 * search_deg) / coarse_step_deg))
    for index in range(steps + 1):
        delta_deg = -search_deg + index * coarse_step_deg
        theta = initial_theta + math.radians(delta_deg)
        score = offset_histogram_peak_score(points, theta, bin_m=bin_m)
        if score > best_score:
            best_score = score
            best_theta = theta
    fine_radius_deg = coarse_step_deg * 3.0
    fine_steps = int(round((2.0 * fine_radius_deg) / fine_step_deg))
    refined_theta = best_theta
    for index in range(fine_steps + 1):
        delta_deg = -fine_radius_deg + index * fine_step_deg
        theta = best_theta + math.radians(delta_deg)
        score = offset_histogram_peak_score(points, theta, bin_m=bin_m)
        if score > best_score:
            best_score = score
            refined_theta = theta
    while refined_theta < 0:
        refined_theta += math.pi
    while refined_theta >= math.pi:
        refined_theta -= math.pi
    return refined_theta


def offset_histogram_peak_score(points: list[tuple[float, float]], theta: float, *, bin_m: float) -> float:
    offsets = projected_offsets(points, theta)
    if not offsets:
        return 0.0
    min_offset = min(offsets)
    counts: dict[int, int] = {}
    for offset in offsets:
        bucket = int((offset - min_offset) / bin_m)
        counts[bucket] = counts.get(bucket, 0) + 1
    if not counts:
        return 0.0
    max_bucket = max(counts)
    smoothed = [sum(counts.get(k, 0) for k in range(bucket - 2, bucket + 3)) for bucket in range(max_bucket + 1)]
    top = sorted(smoothed, reverse=True)[:8]
    return float(sum(value * value for value in top))


def detect_offset_peaks(
    points: list[tuple[float, float]],
    theta: float,
    *,
    bin_m: float,
    min_peak_count: int,
    peak_spacing_m: float,
) -> list[dict[str, float]]:
    offsets = projected_offsets(points, theta)
    if not offsets:
        return []
    min_offset = min(offsets)
    max_offset = max(offsets)
    bucket_count = int((max_offset - min_offset) / bin_m) + 1
    counts: dict[int, int] = {}
    for offset in offsets:
        bucket = int((offset - min_offset) / bin_m)
        counts[bucket] = counts.get(bucket, 0) + 1
    smoothed = [sum(counts.get(k, 0) for k in range(bucket - 2, bucket + 3)) for bucket in range(bucket_count)]
    candidates: list[dict[str, float]] = []
    for bucket in range(2, max(2, bucket_count - 2)):
        count = smoothed[bucket]
        if count < min_peak_count:
            continue
        if count >= smoothed[bucket - 1] and count >= smoothed[bucket + 1]:
            candidates.append({"count": float(count), "offset_m": min_offset + (bucket + 0.5) * bin_m})
    selected: list[dict[str, float]] = []
    for candidate in sorted(candidates, key=lambda item: item["count"], reverse=True):
        if all(abs(candidate["offset_m"] - item["offset_m"]) > peak_spacing_m for item in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda item: item["offset_m"])


def summarize_track_peak_clusters(
    points: list[tuple[float, float]],
    peaks: list[dict[str, float]],
    theta: float,
    *,
    cluster_width_m: float,
    station_gap_m: float,
) -> list[dict[str, Any]]:
    ux, uy = math.cos(theta), math.sin(theta)
    nx, ny = -uy, ux
    clusters: list[dict[str, Any]] = []
    for peak in peaks:
        peak_offset = float(peak["offset_m"])
        xy_points = [
            (x, y)
            for x, y in points
            if abs((x * nx + y * ny) - peak_offset) <= cluster_width_m
        ]
        station_offsets = [(x * ux + y * uy, x * nx + y * ny) for x, y in xy_points]
        if not station_offsets:
            continue
        stations = [station for station, _offset in station_offsets]
        offsets = [offset for _station, offset in station_offsets]
        interval = largest_station_interval(stations, gap_m=station_gap_m)
        clusters.append(
            {
                "offset_m": float(median(offsets)),
                "peak_offset_m": peak_offset,
                "point_count": len(station_offsets),
                "station_min_m": interval[0],
                "station_max_m": interval[1],
                "coverage_m": interval[1] - interval[0],
                "xy_points": xy_points,
            }
        )
    return clusters


def choose_mainline_cluster(
    clusters: list[dict[str, Any]],
    *,
    neighbor_min_m: float,
    neighbor_max_m: float,
    neighbor_overlap_m: float,
) -> int:
    if not clusters:
        raise ValueError("No semantic-segmentation track clusters were available for mainline selection.")
    for index, cluster in enumerate(clusters):
        has_left = False
        has_right = False
        for other_index, other in enumerate(clusters):
            if other_index == index:
                continue
            delta = float(other["offset_m"]) - float(cluster["offset_m"])
            overlap = max(
                0.0,
                min(float(cluster["station_max_m"]), float(other["station_max_m"]))
                - max(float(cluster["station_min_m"]), float(other["station_min_m"])),
            )
            if neighbor_min_m <= abs(delta) <= neighbor_max_m and overlap >= neighbor_overlap_m:
                has_left = has_left or delta < 0
                has_right = has_right or delta > 0
        cluster["two_sided_neighbor"] = has_left and has_right
    two_sided = [(index, cluster) for index, cluster in enumerate(clusters) if cluster.get("two_sided_neighbor")]
    candidates = two_sided or list(enumerate(clusters))
    return max(candidates, key=lambda item: (float(item[1]["coverage_m"]), int(item[1]["point_count"])))[0]


def estimate_point_cloud_orientation(points: list[tuple[float, float]], *, fallback_theta: float) -> float:
    if len(points) < 2:
        return fallback_theta
    mean_x = sum(x for x, _y in points) / len(points)
    mean_y = sum(y for _x, y in points) / len(points)
    sxx = sum((x - mean_x) * (x - mean_x) for x, _y in points)
    syy = sum((y - mean_y) * (y - mean_y) for _x, y in points)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    if abs(sxy) <= 1e-12 and abs(sxx - syy) <= 1e-12:
        return fallback_theta
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    if theta < 0:
        theta += math.pi
    return theta


def largest_station_interval(stations: list[float], *, gap_m: float) -> tuple[float, float]:
    ordered = sorted(stations)
    if not ordered:
        return (0.0, 0.0)
    intervals: list[tuple[float, float, int]] = []
    start = ordered[0]
    previous = ordered[0]
    count = 1
    for station in ordered[1:]:
        if station - previous > gap_m:
            intervals.append((start, previous, count))
            start = station
            count = 1
        else:
            count += 1
        previous = station
    intervals.append((start, previous, count))
    best = max(intervals, key=lambda item: (item[1] - item[0], item[2]))
    return (best[0], best[1])


def projected_offsets(points: list[tuple[float, float]], theta: float) -> list[float]:
    nx = -math.sin(theta)
    ny = math.cos(theta)
    return [x * nx + y * ny for x, y in points]


def select_support_features(
    features: list[dict[str, Any]],
    *,
    guide: Guide,
    track_id: str,
    corridor_m: float,
    min_near_points: int,
    min_station_span_m: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    selected: list[dict[str, Any]] = []
    all_offsets: list[float] = []
    for feature in features:
        coords = line_coords(feature)
        near_offsets: list[float] = []
        near_stations: list[float] = []
        for coord in coords:
            station, offset = guide.station_offset(coord)
            if -corridor_m <= station <= guide.length + corridor_m and abs(offset) <= corridor_m:
                near_offsets.append(offset)
                near_stations.append(station)
        if len(near_offsets) < min_near_points:
            continue
        station_span = max(near_stations) - min(near_stations)
        if station_span < min_station_span_m:
            continue
        props = dict(feature.get("properties") or {})
        props.update(
            {
                "role": "mainline_support_candidate",
                "track_id": track_id,
                "source": "raw_dom_centerline_candidate",
                "guide_mean_offset_m": round(float(sum(near_offsets) / len(near_offsets)), 4),
                "guide_median_offset_m": round(float(median(near_offsets)), 4),
                "near_point_count": len(near_offsets),
                "near_station_min_m": round(min(near_stations), 3),
                "near_station_max_m": round(max(near_stations), 3),
            }
        )
        selected.append({"type": "Feature", "properties": props, "geometry": feature["geometry"]})
        all_offsets.extend(near_offsets)
    return selected, all_offsets


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


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    write_json(
        path,
        {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": f"EPSG:{DEFAULT_EPSG}"}},
            "features": features,
        },
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pyshp in the active virtual environment.") from exc

    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("role", "C", size=32)
    writer.field("track_id", "C", size=24)
    writer.field("source", "C", size=40)
    writer.field("cand_id", "N", decimal=0)
    writer.field("pts", "N", decimal=0)
    writer.field("len_m", "F", decimal=3)
    writer.field("offset_m", "F", decimal=4)
    writer.field("conf", "F", decimal=4)
    for index, feature in enumerate(features):
        props = feature.get("properties") or {}
        coords = line_coords(feature)
        writer.line([coords])
        writer.record(
            str(props.get("role", ""))[:32],
            str(props.get("track_id", ""))[:24],
            str(props.get("source", ""))[:40],
            safe_int(props.get("candidate_id", index)),
            safe_int(props.get("point_count", len(coords))),
            safe_float(props.get("length_m", polyline_length(coords))),
            safe_float(props.get("offset_m", props.get("guide_median_offset_m", 0.0))),
            safe_float(props.get("mean_confidence", 0.0)),
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    write_projection(output_path.with_suffix(".prj"), epsg)


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(coords, coords[1:]))


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


def write_qgis_line_style(path: Path, *, color: str, width_mm: float) -> None:
    path.write_text(
        f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="singleSymbol" enableorderby="0" forceraster="0" referencescale="-1" symbollevels="0">
    <symbols>
      <symbol name="0" type="line" alpha="1" clip_to_extent="1" force_rhr="0">
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
      </symbol>
    </symbols>
  </renderer-v2>
  <blendMode>0</blendMode>
  <featureBlendMode>0</featureBlendMode>
  <layerGeometryType>1</layerGeometryType>
</qgis>
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
