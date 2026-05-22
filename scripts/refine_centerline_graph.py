from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Axes:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass
class Segment:
    index: int
    coords: list[tuple[float, float]]
    st: np.ndarray
    s_min: float
    s_max: float
    slope: float
    intercept: float
    heading: np.ndarray
    length: float
    confidence: float
    properties: dict[str, Any]

    def t_at(self, s: float) -> float:
        return self.slope * s + self.intercept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refine per-tile centerline candidates into continuous track chains.")
    parser.add_argument("--input", required=True, help="Input centerline candidate GeoJSON.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--max-lateral-gap", type=float, default=1.8, help="Maximum lateral difference in map units.")
    parser.add_argument("--max-longitudinal-gap", type=float, default=8.0, help="Maximum longitudinal gap in map units.")
    parser.add_argument("--max-angle-deg", type=float, default=18.0, help="Maximum direction angle difference.")
    parser.add_argument("--min-segment-length", type=float, default=5.0)
    parser.add_argument("--min-chain-extent", type=float, default=30.0)
    parser.add_argument("--merge-bin-size", type=float, default=0.75, help="Longitudinal bin size for merging overlapping segments.")
    parser.add_argument("--main-stitch-lateral-gap", type=float, default=4.0, help="Looser lateral gap used to stitch mainline chains.")
    parser.add_argument("--main-stitch-longitudinal-gap", type=float, default=180.0, help="Maximum gap bridged while selecting the mainline.")
    parser.add_argument("--main-stitch-angle-deg", type=float, default=35.0)
    parser.add_argument("--diagnostic-width", type=int, default=1200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_features = load_geojson_lines(input_path)
    axes = estimate_axes([coord for feature in raw_features for coord in feature["coords"]])
    segments = [
        build_segment(index, feature["coords"], feature["properties"], axes)
        for index, feature in enumerate(raw_features)
    ]
    segments = [segment for segment in segments if segment.length >= args.min_segment_length]
    components = build_components(
        segments,
        max_lateral_gap=args.max_lateral_gap,
        max_longitudinal_gap=args.max_longitudinal_gap,
        max_angle_deg=args.max_angle_deg,
    )
    chains = build_chains(components, segments, axes, args.merge_bin_size, args.min_chain_extent)
    main_chain = select_main_path(
        chains,
        axes,
        bin_size=args.merge_bin_size,
        max_lateral_gap=args.main_stitch_lateral_gap,
        max_longitudinal_gap=args.main_stitch_longitudinal_gap,
        max_angle_deg=args.main_stitch_angle_deg,
    )

    network_path = output_dir / "refined_centerline_network.geojson"
    main_path = output_dir / "main_centerline.geojson"
    diagnostic_path = output_dir / "refined_centerline_diagnostic.png"
    summary_path = output_dir / "summary.json"
    write_geojson(network_path, chains)
    write_geojson(main_path, [main_chain] if main_chain is not None else [])
    write_diagnostic(diagnostic_path, raw_features, chains, main_chain, args.diagnostic_width)

    global_s_min = min(segment.s_min for segment in segments) if segments else 0.0
    global_s_max = max(segment.s_max for segment in segments) if segments else 0.0
    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "raw_feature_count": len(raw_features),
        "used_segment_count": len(segments),
        "component_count": len(components),
        "chain_count": len(chains),
        "global_s_extent_m": global_s_max - global_s_min,
        "main_chain": main_chain["properties"] if main_chain is not None else None,
        "parameters": {
            "max_lateral_gap": args.max_lateral_gap,
            "max_longitudinal_gap": args.max_longitudinal_gap,
            "max_angle_deg": args.max_angle_deg,
            "min_segment_length": args.min_segment_length,
            "min_chain_extent": args.min_chain_extent,
            "merge_bin_size": args.merge_bin_size,
            "main_stitch_lateral_gap": args.main_stitch_lateral_gap,
            "main_stitch_longitudinal_gap": args.main_stitch_longitudinal_gap,
            "main_stitch_angle_deg": args.main_stitch_angle_deg,
        },
        "outputs": {
            "network_geojson": str(network_path),
            "main_geojson": str(main_path),
            "diagnostic_png": str(diagnostic_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_geojson_lines(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates", [])]
        if len(coords) >= 2:
            rows.append({"coords": coords, "properties": feature.get("properties") or {}})
    return rows


def estimate_axes(coords: list[tuple[float, float]]) -> Axes:
    if len(coords) < 2:
        raise ValueError("At least two coordinates are required.")
    points = np.asarray(coords, dtype=np.float64)
    origin = points.mean(axis=0)
    centered = points - origin
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    u = vectors[:, int(np.argmax(values))]
    if u[1] < 0:
        u = -u
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]], dtype=np.float64)
    return Axes(origin=origin, u=u, v=v)


def to_st(coords: list[tuple[float, float]], axes: Axes) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    centered = points - axes.origin
    return np.column_stack([centered @ axes.u, centered @ axes.v])


def from_st(s: float, t: float, axes: Axes) -> tuple[float, float]:
    point = axes.origin + axes.u * s + axes.v * t
    return float(point[0]), float(point[1])


def build_segment(index: int, coords: list[tuple[float, float]], properties: dict[str, Any], axes: Axes) -> Segment:
    st = to_st(coords, axes)
    ordered_coords = list(coords)
    if st[-1, 0] < st[0, 0]:
        ordered_coords.reverse()
        st = to_st(ordered_coords, axes)
    s_values = st[:, 0]
    t_values = st[:, 1]
    if float(np.ptp(s_values)) < 1e-6:
        slope = 0.0
        intercept = float(np.mean(t_values))
    else:
        slope, intercept = np.polyfit(s_values, t_values, 1)
        slope = float(slope)
        intercept = float(intercept)
    delta = np.asarray(ordered_coords[-1], dtype=np.float64) - np.asarray(ordered_coords[0], dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    heading = delta / norm if norm > 1e-9 else axes.u
    if float(heading @ axes.u) < 0:
        heading = -heading
    length = polyline_length(ordered_coords)
    confidence = float(properties.get("mean_confidence", 0.0) or 0.0)
    return Segment(
        index=index,
        coords=ordered_coords,
        st=st,
        s_min=float(s_values.min()),
        s_max=float(s_values.max()),
        slope=slope,
        intercept=intercept,
        heading=heading,
        length=length,
        confidence=confidence,
        properties=properties,
    )


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return float(sum(math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]) for i in range(len(coords) - 1)))


def are_segments_compatible(
    a: Segment,
    b: Segment,
    max_lateral_gap: float,
    max_longitudinal_gap: float,
    max_angle_deg: float,
) -> bool:
    dot = float(np.clip(abs(a.heading @ b.heading), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    if angle > max_angle_deg:
        return False

    overlap_start = max(a.s_min, b.s_min)
    overlap_end = min(a.s_max, b.s_max)
    if overlap_end >= overlap_start:
        s_ref = 0.5 * (overlap_start + overlap_end)
        return abs(a.t_at(s_ref) - b.t_at(s_ref)) <= max_lateral_gap

    s_gap = max(a.s_min - b.s_max, b.s_min - a.s_max, 0.0)
    if s_gap > max_longitudinal_gap:
        return False
    if a.s_max < b.s_min:
        s_ref = 0.5 * (a.s_max + b.s_min)
    else:
        s_ref = 0.5 * (b.s_max + a.s_min)
    return abs(a.t_at(s_ref) - b.t_at(s_ref)) <= max_lateral_gap


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def build_components(
    segments: list[Segment],
    max_lateral_gap: float,
    max_longitudinal_gap: float,
    max_angle_deg: float,
) -> list[list[int]]:
    uf = UnionFind(len(segments))
    for i, a in enumerate(segments):
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if max(a.s_min, b.s_min) - min(a.s_max, b.s_max) > max_longitudinal_gap:
                continue
            if are_segments_compatible(a, b, max_lateral_gap, max_longitudinal_gap, max_angle_deg):
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for index in range(len(segments)):
        groups.setdefault(uf.find(index), []).append(index)
    return list(groups.values())


def build_chains(
    components: list[list[int]],
    segments: list[Segment],
    axes: Axes,
    bin_size: float,
    min_chain_extent: float,
) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    for component_id, component in enumerate(components):
        component_segments = [segments[index] for index in component]
        s_min = min(segment.s_min for segment in component_segments)
        s_max = max(segment.s_max for segment in component_segments)
        if s_max - s_min < min_chain_extent:
            continue
        bins: dict[int, list[tuple[float, float, float]]] = {}
        for segment in component_segments:
            weight = max(segment.confidence, 0.05)
            for s, t in segment.st:
                key = int(round((float(s) - s_min) / bin_size))
                bins.setdefault(key, []).append((float(s), float(t), weight))
        line: list[tuple[float, float]] = []
        for key in sorted(bins):
            values = bins[key]
            weights = np.asarray([item[2] for item in values], dtype=np.float64)
            s_avg = float(np.average([item[0] for item in values], weights=weights))
            t_avg = float(np.average([item[1] for item in values], weights=weights))
            point = from_st(s_avg, t_avg, axes)
            if not line or math.hypot(point[0] - line[-1][0], point[1] - line[-1][1]) > 0.05:
                line.append(point)
        if len(line) < 2:
            continue
        length = polyline_length(line)
        confidence = float(np.average([segment.confidence for segment in component_segments], weights=[segment.length for segment in component_segments]))
        chains.append(
            {
                "type": "Feature",
                "properties": {
                    "chain_id": len(chains),
                    "role": "candidate",
                    "feature_count": len(component_segments),
                    "source_feature_ids": ",".join(str(segment.index) for segment in sorted(component_segments, key=lambda item: item.s_min)),
                    "length_m": round(length, 3),
                    "s_extent_m": round(s_max - s_min, 3),
                    "mean_confidence": round(confidence, 4),
                    "s_min": round(s_min, 3),
                    "s_max": round(s_max, 3),
                    "source_images": ",".join(sorted({str(segment.properties.get("image_name", "")) for segment in component_segments})[:12]),
                },
                "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in line]},
            }
        )
    chains.sort(key=lambda feature: (feature["properties"]["s_extent_m"], feature["properties"]["length_m"], feature["properties"]["mean_confidence"]), reverse=True)
    for chain_id, feature in enumerate(chains):
        feature["properties"]["chain_id"] = chain_id
    return chains


def select_main_chain(chains: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not chains:
        return None
    main = json.loads(json.dumps(chains[0]))
    main["properties"]["role"] = "main"
    for feature in chains:
        feature["properties"]["role"] = "main" if feature["properties"]["chain_id"] == main["properties"]["chain_id"] else "support"
    return main


def select_main_path(
    chains: list[dict[str, Any]],
    axes: Axes,
    bin_size: float,
    max_lateral_gap: float,
    max_longitudinal_gap: float,
    max_angle_deg: float,
) -> dict[str, Any] | None:
    if not chains:
        return None
    chain_segments = [
        build_segment(
            int(feature["properties"]["chain_id"]),
            [(float(x), float(y)) for x, y in feature["geometry"]["coordinates"]],
            feature["properties"],
            axes,
        )
        for feature in chains
    ]
    order = sorted(range(len(chain_segments)), key=lambda index: (chain_segments[index].s_min, chain_segments[index].s_max))
    scores = [segment.length for segment in chain_segments]
    previous: list[int | None] = [None] * len(chain_segments)
    gap_penalty = 0.25
    for right_position, right_index in enumerate(order):
        right = chain_segments[right_index]
        for left_index in order[:right_position]:
            left = chain_segments[left_index]
            if right.s_max <= left.s_max + 10.0:
                continue
            if not are_segments_compatible(left, right, max_lateral_gap, max_longitudinal_gap, max_angle_deg):
                continue
            longitudinal_gap = max(right.s_min - left.s_max, 0.0)
            candidate_score = scores[left_index] + right.length - longitudinal_gap * gap_penalty
            if candidate_score > scores[right_index]:
                scores[right_index] = candidate_score
                previous[right_index] = left_index

    best_index = max(range(len(chain_segments)), key=lambda index: (chain_segments[index].s_max - trace_s_min(index, previous, chain_segments), scores[index]))
    path_indices: list[int] = []
    cursor: int | None = best_index
    while cursor is not None:
        path_indices.append(cursor)
        cursor = previous[cursor]
    path_indices.reverse()
    selected_chain_ids = {chain_segments[index].index for index in path_indices}
    for feature in chains:
        feature["properties"]["role"] = "main_path" if feature["properties"]["chain_id"] in selected_chain_ids else "support"
    return build_main_feature([chain_segments[index] for index in path_indices], axes, bin_size)


def trace_s_min(index: int, previous: list[int | None], segments: list[Segment]) -> float:
    values = [segments[index].s_min]
    cursor = previous[index]
    while cursor is not None:
        values.append(segments[cursor].s_min)
        cursor = previous[cursor]
    return min(values)


def build_main_feature(path_segments: list[Segment], axes: Axes, bin_size: float) -> dict[str, Any]:
    s_min = min(segment.s_min for segment in path_segments)
    s_max = max(segment.s_max for segment in path_segments)
    bins: dict[int, list[tuple[float, float, float]]] = {}
    for segment in path_segments:
        weight = max(segment.confidence, 0.05)
        for s, t in segment.st:
            key = int(round((float(s) - s_min) / bin_size))
            bins.setdefault(key, []).append((float(s), float(t), weight))
    line: list[tuple[float, float]] = []
    for key in sorted(bins):
        values = bins[key]
        weights = np.asarray([item[2] for item in values], dtype=np.float64)
        s_avg = float(np.average([item[0] for item in values], weights=weights))
        t_avg = float(np.average([item[1] for item in values], weights=weights))
        line.append(from_st(s_avg, t_avg, axes))
    chain_ids = [segment.index for segment in path_segments]
    length = polyline_length(line)
    bridged_gaps = [
        max(path_segments[index + 1].s_min - path_segments[index].s_max, 0.0)
        for index in range(len(path_segments) - 1)
    ]
    confidence = float(np.average([segment.confidence for segment in path_segments], weights=[segment.length for segment in path_segments]))
    return {
        "type": "Feature",
        "properties": {
            "chain_id": "main_path",
            "role": "main",
            "stitched_chain_ids": ",".join(str(value) for value in chain_ids),
            "stitched_chain_count": len(path_segments),
            "length_m": round(length, 3),
            "s_extent_m": round(s_max - s_min, 3),
            "mean_confidence": round(confidence, 4),
            "max_bridged_gap_m": round(max(bridged_gaps), 3) if bridged_gaps else 0.0,
            "total_bridged_gap_m": round(sum(bridged_gaps), 3),
            "s_min": round(s_min, 3),
            "s_max": round(s_max, 3),
        },
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in line]},
    }


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_diagnostic(
    path: Path,
    raw_features: list[dict[str, Any]],
    chains: list[dict[str, Any]],
    main_chain: dict[str, Any] | None,
    width: int,
) -> None:
    coords = [coord for feature in raw_features for coord in feature["coords"]]
    if not coords:
        return
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    height = max(400, int(width * span_y / span_x))
    height = min(height, 2400)
    margin = 30
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)
    image = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(image)

    def project(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = margin + (x - min_x) * scale
        py = height - margin - (y - min_y) * scale
        return int(round(px)), int(round(py))

    for feature in raw_features:
        points = [project(point) for point in feature["coords"]]
        if len(points) >= 2:
            draw.line(points, fill=(180, 180, 180), width=1)

    palette = [(0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (86, 180, 233), (230, 159, 0)]
    for feature in chains[:24]:
        points = [project((float(x), float(y))) for x, y in feature["geometry"]["coordinates"]]
        is_main = str(feature["properties"]["role"]).startswith("main")
        color = (220, 20, 60) if is_main else palette[feature["properties"]["chain_id"] % len(palette)]
        width_px = 4 if is_main else 2
        if len(points) >= 2:
            draw.line(points, fill=color, width=width_px)

    if main_chain is not None:
        draw.text((10, 10), f"main chain {main_chain['properties']['chain_id']} length={main_chain['properties']['length_m']}m", fill=(120, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
