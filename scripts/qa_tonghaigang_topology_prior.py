#!/usr/bin/env python3
"""Create DOM overlay crops and first-pass topology-prior diagnostics.

This script is deliberately diagnostic.  It does not replace the final
centerline builder; it helps inspect whether the current network matches the
Tonghaigang station prior: one through main track, up to two parallel sidings,
and tangent turnout/crossover connectors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window
from shapely.geometry import LineString, Point, box


DEFAULT_CENTERLINE = Path("output/rail_centerline_final_v6_qa_bridge/final_centerline_network.geojson")
DEFAULT_DOM = Path("data/aligned_dom/aligned_dom.tif")
DEFAULT_OUT_DIR = Path("output/tonghaigang_topology_prior_qa")


SOURCE_COLORS = {
    "automatic_refined": (255, 64, 64),
    "track_path_hint": (0, 210, 255),
    "qgis_accepted_centerline": (0, 210, 170),
    "manual_feedback": (255, 80, 255),
    "qa_supervised_gap_bridge": (255, 220, 0),
}


ROLE_COLORS = {
    "main_candidate": (255, 255, 0),
    "siding_candidate": (80, 220, 255),
    "connector_candidate": (255, 120, 0),
    "unknown": (210, 210, 210),
}


class LineFeature(NamedTuple):
    feature_id: str
    properties: dict[str, Any]
    line: LineString


class Axis(NamedTuple):
    origin: np.ndarray
    longitudinal: np.ndarray
    lateral: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DOM overlay QA crops for the Tonghaigang topology prior.")
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--crop-width-m", type=float, default=150.0)
    parser.add_argument("--crop-height-m", type=float, default=260.0)
    parser.add_argument("--max-crops", type=int, default=12)
    parser.add_argument("--node-tolerance-m", type=float, default=0.50)
    parser.add_argument("--overview-height", type=int, default=2800)
    parser.add_argument("--classified-geojson", type=Path, default=None)
    return parser.parse_args()


def load_features(path: Path) -> list[LineFeature]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features: list[LineFeature] = []
    for index, feature in enumerate(payload.get("features", []) or [], start=1):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates", [])]
        if len(coords) < 2:
            continue
        props = dict(feature.get("properties") or {})
        feature_id = str(props.get("final_id") or props.get("chain_id") or props.get("id_text") or f"feature_{index}")
        features.append(LineFeature(feature_id, props, LineString(coords)))
    return features


def estimate_axis(features: list[LineFeature]) -> Axis:
    points: list[tuple[float, float]] = []
    for feature in features:
        coords = list(feature.line.coords)
        step = max(1, len(coords) // 200)
        points.extend((float(x), float(y)) for x, y, *_ in coords[::step])
    if len(points) < 2:
        raise ValueError("Need at least two points to estimate route axis")
    matrix = np.asarray(points, dtype=float)
    origin = matrix.mean(axis=0)
    centered = matrix - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    longitudinal = vh[0]
    if longitudinal[1] < 0:
        longitudinal = -longitudinal
    lateral = np.array([-longitudinal[1], longitudinal[0]])
    return Axis(origin=origin, longitudinal=longitudinal, lateral=lateral)


def project_points(axis: Axis, coords: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(coords, dtype=float)
    centered = matrix - axis.origin
    return centered @ axis.longitudinal, centered @ axis.lateral


def feature_direction_angle_deg(axis: Axis, line: LineString) -> float:
    coords = list(line.coords)
    start = np.asarray(coords[0][:2], dtype=float)
    end = np.asarray(coords[-1][:2], dtype=float)
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return 90.0
    direction = direction / norm
    dot = abs(float(direction @ axis.longitudinal))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def classify_features(features: list[LineFeature], axis: Axis) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    tracklike: list[dict[str, Any]] = []
    for feature in features:
        coords = [(float(x), float(y)) for x, y, *_ in feature.line.coords]
        s_values, t_values = project_points(axis, coords)
        angle = feature_direction_angle_deg(axis, feature.line)
        item = {
            "feature_id": feature.feature_id,
            "source": str(feature.properties.get("network_source") or ""),
            "length_m": float(feature.line.length),
            "s_min": float(np.min(s_values)),
            "s_max": float(np.max(s_values)),
            "s_span": float(np.max(s_values) - np.min(s_values)),
            "t_median": float(np.median(t_values)),
            "t_span": float(np.max(t_values) - np.min(t_values)),
            "angle_to_route_deg": float(angle),
            "role": "unknown",
            "band": None,
        }
        if angle <= 20.0 and item["t_span"] <= 4.0 and item["length_m"] >= 1.0:
            tracklike.append(item)
        elif item["t_span"] > 4.0 or angle > 20.0:
            item["role"] = "connector_candidate"
        diagnostics.append(item)

    bands = cluster_track_bands(tracklike)
    main_band_index = choose_main_band(bands)
    by_feature_id = {item["feature_id"]: item for item in diagnostics}
    for band_index, band in enumerate(bands, start=1):
        role = "main_candidate" if band_index == main_band_index else "siding_candidate"
        for item in band["items"]:
            target = by_feature_id[item["feature_id"]]
            target["role"] = role
            target["band"] = band_index

    summary = {
        "axis_origin": [round(float(axis.origin[0]), 6), round(float(axis.origin[1]), 6)],
        "axis_longitudinal": [round(float(axis.longitudinal[0]), 8), round(float(axis.longitudinal[1]), 8)],
        "axis_lateral": [round(float(axis.lateral[0]), 8), round(float(axis.lateral[1]), 8)],
        "band_count": len(bands),
        "main_band": main_band_index,
        "bands": [
            {
                "band": index,
                "feature_count": len(band["items"]),
                "length_m": round(float(sum(item["length_m"] for item in band["items"])), 3),
                "s_min": round(float(band["s_min"]), 3),
                "s_max": round(float(band["s_max"]), 3),
                "s_span": round(float(band["s_span"]), 3),
                "t_median": round(float(band["t_median"]), 3),
            }
            for index, band in enumerate(bands, start=1)
        ],
        "role_counts": count_values(item["role"] for item in diagnostics),
        "source_counts": count_values(item["source"] for item in diagnostics),
    }
    return diagnostics, summary


def cluster_track_bands(items: list[dict[str, Any]], *, gap_threshold_m: float = 4.0) -> list[dict[str, Any]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: item["t_median"])
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for item in ordered[1:]:
        previous_t = float(np.median([entry["t_median"] for entry in groups[-1]]))
        if abs(item["t_median"] - previous_t) > gap_threshold_m:
            groups.append([item])
        else:
            groups[-1].append(item)

    while len(groups) > 3:
        gaps = [
            abs(float(np.median([entry["t_median"] for entry in groups[index + 1]])) - float(np.median([entry["t_median"] for entry in groups[index]])))
            for index in range(len(groups) - 1)
        ]
        merge_index = int(np.argmin(gaps))
        groups[merge_index].extend(groups.pop(merge_index + 1))

    bands: list[dict[str, Any]] = []
    for group in groups:
        s_min = min(item["s_min"] for item in group)
        s_max = max(item["s_max"] for item in group)
        bands.append(
            {
                "items": group,
                "s_min": s_min,
                "s_max": s_max,
                "s_span": s_max - s_min,
                "t_median": float(np.median([item["t_median"] for item in group])),
            }
        )
    return sorted(bands, key=lambda band: band["t_median"])


def choose_main_band(bands: list[dict[str, Any]]) -> int | None:
    if not bands:
        return None
    best_index = max(range(len(bands)), key=lambda index: (bands[index]["s_span"], -abs(bands[index]["t_median"])))
    return best_index + 1


def count_values(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))


def build_endpoint_nodes(features: list[LineFeature], tolerance_m: float) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for feature in features:
        coords = list(feature.line.coords)
        for endpoint in ("start", "end"):
            coord = coords[0] if endpoint == "start" else coords[-1]
            refs.append({"feature_id": feature.feature_id, "endpoint": endpoint, "x": float(coord[0]), "y": float(coord[1])})
    parent = list(range(len(refs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, ref in enumerate(refs):
        key = (math.floor(ref["x"] / tolerance_m), math.floor(ref["y"] / tolerance_m))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_index in buckets.get((key[0] + dx, key[1] + dy), []):
                    other = refs[other_index]
                    if math.hypot(ref["x"] - other["x"], ref["y"] - other["y"]) <= tolerance_m:
                        union(index, other_index)
        buckets.setdefault(key, []).append(index)

    groups_by_root: dict[int, list[dict[str, Any]]] = {}
    for index, ref in enumerate(refs):
        groups_by_root.setdefault(find(index), []).append(ref)
    nodes: list[dict[str, Any]] = []
    for index, group in enumerate(groups_by_root.values(), start=1):
        nodes.append(
            {
                "node_id": f"node_{index:04d}",
                "degree": len(group),
                "x": sum(item["x"] for item in group) / len(group),
                "y": sum(item["y"] for item in group) / len(group),
                "refs": group,
            }
        )
    return nodes


def choose_crop_centers(features: list[LineFeature], nodes: list[dict[str, Any]], axis: Axis, max_crops: int) -> list[dict[str, Any]]:
    centers: list[dict[str, Any]] = []
    min_separation_m = 180.0

    def projected_s(x: float, y: float) -> float:
        s_values, _ = project_points(axis, [(x, y)])
        return float(s_values[0])

    def far_enough(candidate_s: float) -> bool:
        return all(abs(candidate_s - float(center["s"])) >= min_separation_m for center in centers)

    for node in sorted(nodes, key=lambda item: (-int(item["degree"]), item["y"], item["x"])):
        if int(node["degree"]) >= 3:
            s = projected_s(float(node["x"]), float(node["y"]))
            if not far_enough(s):
                continue
            centers.append({"name": f"{len(centers)+1:02d}_junction_{node['node_id']}", "x": node["x"], "y": node["y"], "s": s, "kind": "junction", "degree": node["degree"]})
        if len(centers) >= max_crops:
            return centers

    # Add dead-end and evenly spaced route samples if junctions are not enough.
    for node in sorted(nodes, key=lambda item: (item["y"], item["x"])):
        if int(node["degree"]) == 1:
            s = projected_s(float(node["x"]), float(node["y"]))
            if not far_enough(s):
                continue
            centers.append({"name": f"{len(centers)+1:02d}_deadend_{node['node_id']}", "x": node["x"], "y": node["y"], "s": s, "kind": "deadend", "degree": node["degree"]})
            if len(centers) >= max_crops:
                return centers

    all_coords = [(float(x), float(y)) for feature in features for x, y, *_ in feature.line.coords]
    s_values, _ = project_points(axis, all_coords)
    s_min = float(np.min(s_values))
    s_max = float(np.max(s_values))
    for s in np.linspace(s_min, s_max, num=max_crops):
        point = axis.origin + axis.longitudinal * s
        if not far_enough(float(s)):
            continue
        centers.append({"name": f"{len(centers)+1:02d}_route_sample", "x": float(point[0]), "y": float(point[1]), "s": float(s), "kind": "sample", "degree": None})
        if len(centers) >= max_crops:
            break
    return centers[:max_crops]


def raster_window_for_bounds(dataset, bounds: tuple[float, float, float, float]) -> Window:
    minx, miny, maxx, maxy = bounds
    inverse = ~dataset.transform
    corners = [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
    pixels = [inverse * corner for corner in corners]
    cols = [pixel[0] for pixel in pixels]
    rows = [pixel[1] for pixel in pixels]
    col_off = max(0, math.floor(min(cols)) - 8)
    row_off = max(0, math.floor(min(rows)) - 8)
    col_max = min(dataset.width, math.ceil(max(cols)) + 8)
    row_max = min(dataset.height, math.ceil(max(rows)) + 8)
    if col_max <= col_off or row_max <= row_off:
        raise ValueError(f"Crop bounds do not intersect raster: {bounds}")
    return Window(col_off=col_off, row_off=row_off, width=col_max - col_off, height=row_max - row_off)


def read_window_image(dataset, window: Window, *, max_dimension: int = 2200) -> tuple[Image.Image, float, float]:
    scale = min(1.0, max_dimension / max(float(window.width), float(window.height)))
    out_width = max(1, int(round(window.width * scale)))
    out_height = max(1, int(round(window.height * scale)))
    data = dataset.read(indexes=list(range(1, min(dataset.count, 3) + 1)), window=window, out_shape=(min(dataset.count, 3), out_height, out_width))
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    rgb = np.moveaxis(data[:3], 0, -1)
    rgb = normalize_uint8(rgb)
    return Image.fromarray(rgb, mode="RGB"), out_width / float(window.width), out_height / float(window.height)


def normalize_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        high = low + 1.0
    scaled = (array.astype(float) - low) * 255.0 / (high - low)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def draw_features(
    image: Image.Image,
    dataset,
    window: Window,
    scale_x: float,
    scale_y: float,
    features: list[LineFeature],
    *,
    diagnostics_by_id: dict[str, dict[str, Any]] | None = None,
    color_mode: str = "source",
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    crop_poly = crop_polygon(dataset, window)
    for feature in features:
        if not feature.line.intersects(crop_poly):
            continue
        if color_mode == "role" and diagnostics_by_id is not None:
            role = diagnostics_by_id.get(feature.feature_id, {}).get("role", "unknown")
            color = ROLE_COLORS.get(str(role), ROLE_COLORS["unknown"])
        else:
            source = str(feature.properties.get("network_source") or "")
            color = SOURCE_COLORS.get(source, (230, 230, 230))
        points = [world_to_image(dataset, window, scale_x, scale_y, (float(x), float(y))) for x, y, *_ in feature.line.coords]
        if len(points) >= 2:
            draw.line(points, fill=(*color, 230), width=3, joint="curve")


def crop_polygon(dataset, window: Window):
    transform = dataset.transform
    corners = [
        transform * (window.col_off, window.row_off),
        transform * (window.col_off + window.width, window.row_off),
        transform * (window.col_off + window.width, window.row_off + window.height),
        transform * (window.col_off, window.row_off + window.height),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return box(min(xs), min(ys), max(xs), max(ys))


def world_to_image(dataset, window: Window, scale_x: float, scale_y: float, coord: tuple[float, float]) -> tuple[float, float]:
    col, row = (~dataset.transform) * coord
    return ((col - window.col_off) * scale_x, (row - window.row_off) * scale_y)


def draw_legend(image: Image.Image, *, color_mode: str) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    items = ROLE_COLORS.items() if color_mode == "role" else SOURCE_COLORS.items()
    x = 12
    y = 12
    line_h = 22
    width = 310
    height = line_h * (len(list(items)) + 1)
    draw.rectangle([x - 6, y - 6, x + width, y + height], fill=(0, 0, 0, 130))
    items = ROLE_COLORS.items() if color_mode == "role" else SOURCE_COLORS.items()
    draw.text((x, y), color_mode, fill=(255, 255, 255, 255))
    for index, (label, color) in enumerate(items, start=1):
        yy = y + index * line_h
        draw.line([(x, yy + 8), (x + 32, yy + 8)], fill=(*color, 255), width=4)
        draw.text((x + 42, yy), label, fill=(255, 255, 255, 255))


def write_overview(dataset, features: list[LineFeature], diagnostics_by_id: dict[str, dict[str, Any]], out_dir: Path, overview_height: int) -> None:
    out_height = min(overview_height, dataset.height)
    out_width = max(1, int(round(dataset.width * out_height / dataset.height)))
    data = dataset.read(indexes=list(range(1, min(dataset.count, 3) + 1)), out_shape=(min(dataset.count, 3), out_height, out_width))
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    image = Image.fromarray(normalize_uint8(np.moveaxis(data[:3], 0, -1)), mode="RGB")
    window = Window(0, 0, dataset.width, dataset.height)
    scale_x = out_width / float(dataset.width)
    scale_y = out_height / float(dataset.height)
    draw_features(image, dataset, window, scale_x, scale_y, features, diagnostics_by_id=diagnostics_by_id, color_mode="source")
    draw_legend(image, color_mode="source")
    image.save(out_dir / "overview_by_source.jpg", quality=92)

    role_image = Image.fromarray(normalize_uint8(np.moveaxis(data[:3], 0, -1)), mode="RGB")
    draw_features(role_image, dataset, window, scale_x, scale_y, features, diagnostics_by_id=diagnostics_by_id, color_mode="role")
    draw_legend(role_image, color_mode="role")
    role_image.save(out_dir / "overview_by_topology_role.jpg", quality=92)


def write_crops(
    dataset,
    features: list[LineFeature],
    diagnostics_by_id: dict[str, dict[str, Any]],
    centers: list[dict[str, Any]],
    out_dir: Path,
    crop_width_m: float,
    crop_height_m: float,
) -> list[dict[str, Any]]:
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_infos: list[dict[str, Any]] = []
    for center in centers:
        bounds = (
            float(center["x"]) - crop_width_m / 2.0,
            float(center["y"]) - crop_height_m / 2.0,
            float(center["x"]) + crop_width_m / 2.0,
            float(center["y"]) + crop_height_m / 2.0,
        )
        try:
            window = raster_window_for_bounds(dataset, bounds)
        except ValueError:
            continue
        image, scale_x, scale_y = read_window_image(dataset, window)
        draw_features(image, dataset, window, scale_x, scale_y, features, diagnostics_by_id=diagnostics_by_id, color_mode="source")
        draw_legend(image, color_mode="source")
        source_path = crop_dir / f"{center['name']}_source.jpg"
        image.save(source_path, quality=92)

        role_image, scale_x, scale_y = read_window_image(dataset, window)
        draw_features(role_image, dataset, window, scale_x, scale_y, features, diagnostics_by_id=diagnostics_by_id, color_mode="role")
        draw_legend(role_image, color_mode="role")
        role_path = crop_dir / f"{center['name']}_role.jpg"
        role_image.save(role_path, quality=92)

        crop_infos.append(
            {
                **center,
                "bounds": [round(value, 3) for value in bounds],
                "source_overlay": str(source_path),
                "role_overlay": str(role_path),
            }
        )
    return crop_infos


def write_classified_geojson(path: Path, features: list[LineFeature], diagnostics_by_id: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": "EPSG:32651"}},
        "features": [],
    }
    for feature in features:
        props = dict(feature.properties)
        diagnostic = diagnostics_by_id.get(feature.feature_id, {})
        props.update(
            {
                "topology_role": diagnostic.get("role", "unknown"),
                "topology_band": diagnostic.get("band"),
                "axis_t_median": round(float(diagnostic.get("t_median", 0.0)), 3),
                "axis_s_min": round(float(diagnostic.get("s_min", 0.0)), 3),
                "axis_s_max": round(float(diagnostic.get("s_max", 0.0)), 3),
                "angle_route": round(float(diagnostic.get("angle_to_route_deg", 0.0)), 2),
            }
        )
        payload["features"].append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "LineString", "coordinates": [[round(float(x), 6), round(float(y), 6)] for x, y, *_ in feature.line.coords]},
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(args.centerline)
    axis = estimate_axis(features)
    diagnostics, topology_summary = classify_features(features, axis)
    diagnostics_by_id = {item["feature_id"]: item for item in diagnostics}
    nodes = build_endpoint_nodes(features, tolerance_m=args.node_tolerance_m)
    centers = choose_crop_centers(features, nodes, axis, max_crops=args.max_crops)
    classified_path = args.classified_geojson or out_dir / "classified_centerline_topology_prior.geojson"
    write_classified_geojson(classified_path, features, diagnostics_by_id)

    with rasterio.open(args.dom) as dataset:
        write_overview(dataset, features, diagnostics_by_id, out_dir, overview_height=args.overview_height)
        crop_infos = write_crops(
            dataset,
            features,
            diagnostics_by_id,
            centers,
            out_dir,
            crop_width_m=args.crop_width_m,
            crop_height_m=args.crop_height_m,
        )

    summary = {
        "centerline": str(args.centerline),
        "dom": str(args.dom),
        "feature_count": len(features),
        "node_count": len(nodes),
        "degree_counts": count_values(node["degree"] for node in nodes),
        "topology_summary": topology_summary,
        "classified_geojson": str(classified_path),
        "overview_by_source": str(out_dir / "overview_by_source.jpg"),
        "overview_by_topology_role": str(out_dir / "overview_by_topology_role.jpg"),
        "crops": crop_infos,
    }
    (out_dir / "topology_prior_qa_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("feature_count", "node_count", "degree_counts", "topology_summary")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
