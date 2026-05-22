#!/usr/bin/env python3
"""Build a first-pass Tonghaigang topology skeleton from rail candidates.

This is intentionally conservative.  It builds the through-main and observed
siding skeleton first, then exports switch model workzones as the places where
future tangent connectors may be constructed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, NamedTuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rail_curve_extractor.cvat_annotations import TileGeoRecord, load_tile_georef


DEFAULT_CANDIDATES = Path("output/rail_centerline_candidates_v7_tonghaigang_chinese/track_centerline_candidates.geojson")
DEFAULT_SWITCH_MASK_DIR = Path("output/rail_seg_semantic_unet_v7_switch_area_chinese/predictions/masks")
DEFAULT_TILE_GEOREF = Path("data/dom_tiles_aligned_annotation/tile_georef.csv")
DEFAULT_OUT_DIR = Path("output/tonghaigang_topology_rebuild_v1")


class LineFeature(NamedTuple):
    feature_id: str
    properties: dict[str, Any]
    coords: list[tuple[float, float]]


class Axis(NamedTuple):
    origin: np.ndarray
    longitudinal: np.ndarray
    lateral: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a first-pass Tonghaigang topology skeleton.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--switch-mask-dir", type=Path, default=DEFAULT_SWITCH_MASK_DIR)
    parser.add_argument("--tile-georef", type=Path, default=DEFAULT_TILE_GEOREF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-bands", type=int, default=3)
    parser.add_argument("--band-gap-threshold-m", type=float, default=3.0)
    parser.add_argument("--band-assign-threshold-m", type=float, default=3.5)
    parser.add_argument("--tracklike-max-angle-deg", type=float, default=18.0)
    parser.add_argument("--tracklike-max-t-span-m", type=float, default=4.5)
    parser.add_argument("--bin-size-m", type=float, default=5.0)
    parser.add_argument("--siding-gap-m", type=float, default=45.0)
    parser.add_argument("--min-output-length-m", type=float, default=25.0)
    parser.add_argument("--switch-min-component-pixels", type=int, default=250)
    parser.add_argument("--switch-bbox-pad-pixels", type=int, default=32)
    parser.add_argument(
        "--no-merge-switch-workzones",
        action="store_true",
        help="Keep raw per-tile switch mask component boxes instead of merging overlapping boxes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_line_features(args.candidates.expanduser().resolve())
    if not candidates:
        raise ValueError(f"No LineString candidates found in {args.candidates}")

    axis = estimate_axis(candidates)
    stats = [feature_stats(feature, axis) for feature in candidates]
    tracklike = [
        item
        for item in stats
        if item["angle_to_axis_deg"] <= args.tracklike_max_angle_deg
        and item["t_span_m"] <= args.tracklike_max_t_span_m
        and item["length_m"] >= 1.0
    ]
    bands = cluster_bands(
        tracklike,
        max_bands=args.max_bands,
        gap_threshold_m=args.band_gap_threshold_m,
    )
    if not bands:
        raise ValueError("Could not identify any parallel track bands.")
    main_band = choose_main_band(bands)
    assigned = assign_features_to_bands(stats, bands, args.band_assign_threshold_m, args.tracklike_max_angle_deg)

    line_features: list[dict[str, Any]] = []
    main_points = binned_band_points(candidates, assigned[main_band], axis, args.bin_size_m)
    main_line, main_fit = fit_continuous_main_line(main_points, axis, args.bin_size_m)
    line_features.append(
        output_line_feature(
            role="main_through_track",
            line_id="main_through_track_001",
            band=main_band,
            coords=main_line,
            extra={
                "rebuild_source": "v7_rail_candidates_band_fit",
                "continuity_policy": "filled_across_internal_gaps",
                "fit_policy": main_fit["fit_policy"],
                "fit_slope_t_per_s": round(float(main_fit["slope"]), 8),
                "fit_intercept_t": round(float(main_fit["intercept"]), 6),
                "fit_kept_point_count": int(main_fit["kept_point_count"]),
            },
        ),
    )

    siding_count = 0
    for band_id in sorted(assigned):
        if band_id == main_band:
            continue
        points = binned_band_points(candidates, assigned[band_id], axis, args.bin_size_m)
        for segment in split_observed_segments(points, axis, max_gap_m=args.siding_gap_m):
            if polyline_length(segment) < args.min_output_length_m:
                continue
            siding_count += 1
            line_features.append(
                output_line_feature(
                    role="siding_or_terminal_track",
                    line_id=f"siding_track_{siding_count:03d}",
                    band=band_id,
                    coords=segment,
                    extra={
                        "rebuild_source": "v7_rail_candidates_observed_band_fit",
                        "continuity_policy": "observed_range_only",
                    },
                ),
            )

    records = load_tile_georef(args.tile_georef.expanduser().resolve())
    raw_workzone_features = vectorize_switch_masks(
        mask_dir=args.switch_mask_dir.expanduser().resolve(),
        records=records,
        min_component_pixels=args.switch_min_component_pixels,
        bbox_pad_pixels=args.switch_bbox_pad_pixels,
    )
    workzone_features = (
        raw_workzone_features
        if args.no_merge_switch_workzones
        else merge_switch_workzones(raw_workzone_features)
    )

    skeleton_path = out_dir / "topology_skeleton.geojson"
    linework_path = out_dir / "topology_skeleton_linework.geojson"
    switch_workzones_path = out_dir / "switch_model_workzones.geojson"
    summary_path = out_dir / "topology_skeleton_summary.json"
    crs = geojson_crs(records)

    write_geojson(skeleton_path, "tonghaigang_topology_skeleton", line_features + workzone_features, crs)
    write_geojson(linework_path, "tonghaigang_topology_skeleton_linework", line_features, crs)
    write_geojson(switch_workzones_path, "switch_model_workzones", workzone_features, crs)

    summary = {
        "candidate_path": str(args.candidates.expanduser().resolve()),
        "switch_mask_dir": str(args.switch_mask_dir.expanduser().resolve()),
        "tile_georef_path": str(args.tile_georef.expanduser().resolve()),
        "output_dir": str(out_dir),
        "candidate_feature_count": len(candidates),
        "tracklike_feature_count": len(tracklike),
        "assigned_feature_count": sum(len(items) for items in assigned.values()),
        "band_count": len(bands),
        "main_band": main_band,
        "bands": [
            {
                "band": band["band"],
                "feature_count": len(band["items"]),
                "assigned_feature_count": len(assigned.get(band["band"], [])),
                "t_median_m": round(float(band["t_median"]), 3),
                "s_min_m": round(float(band["s_min"]), 3),
                "s_max_m": round(float(band["s_max"]), 3),
                "s_span_m": round(float(band["s_span"]), 3),
                "role": "main_through_track" if band["band"] == main_band else "siding_or_terminal_track",
            }
            for band in bands
        ],
        "main_length_m": round(polyline_length(main_line), 3),
        "main_fit": main_fit,
        "siding_output_count": siding_count,
        "raw_switch_workzone_count": len(raw_workzone_features),
        "switch_workzone_count": len(workzone_features),
        "axis_origin": [round(float(axis.origin[0]), 6), round(float(axis.origin[1]), 6)],
        "axis_longitudinal": [round(float(axis.longitudinal[0]), 8), round(float(axis.longitudinal[1]), 8)],
        "axis_lateral": [round(float(axis.lateral[0]), 8), round(float(axis.lateral[1]), 8)],
        "topology_skeleton": str(skeleton_path),
        "topology_skeleton_linework": str(linework_path),
        "switch_model_workzones": str(switch_workzones_path),
        "connector_policy": (
            "v1 exports switch workzones only; tangent turnout connectors should be generated "
            "inside these workzones in the next builder pass."
        ),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_line_features(path: Path) -> list[LineFeature]:
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
        feature_id = str(props.get("candidate_id") or props.get("id") or f"candidate_{index:04d}")
        features.append(LineFeature(feature_id=feature_id, properties=props, coords=coords))
    return features


def estimate_axis(features: list[LineFeature]) -> Axis:
    points: list[tuple[float, float]] = []
    for feature in features:
        step = max(1, len(feature.coords) // 120)
        points.extend(feature.coords[::step])
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


def unproject_point(axis: Axis, s: float, t: float) -> tuple[float, float]:
    xy = axis.origin + axis.longitudinal * s + axis.lateral * t
    return float(xy[0]), float(xy[1])


def feature_stats(feature: LineFeature, axis: Axis) -> dict[str, Any]:
    s_values, t_values = project_points(axis, feature.coords)
    direction = np.asarray(feature.coords[-1], dtype=float) - np.asarray(feature.coords[0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        angle = 90.0
    else:
        direction = direction / norm
        dot = abs(float(direction @ axis.longitudinal))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    return {
        "feature_id": feature.feature_id,
        "s_min": float(np.min(s_values)),
        "s_max": float(np.max(s_values)),
        "s_span_m": float(np.max(s_values) - np.min(s_values)),
        "t_median": float(np.median(t_values)),
        "t_span_m": float(np.percentile(t_values, 95) - np.percentile(t_values, 5)),
        "angle_to_axis_deg": float(angle),
        "length_m": polyline_length(feature.coords),
    }


def cluster_bands(items: list[dict[str, Any]], *, max_bands: int, gap_threshold_m: float) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: item["t_median"])
    groups: list[list[dict[str, Any]]] = []
    for item in ordered:
        if not groups:
            groups.append([item])
            continue
        previous = float(np.median([entry["t_median"] for entry in groups[-1]]))
        if abs(item["t_median"] - previous) > gap_threshold_m:
            groups.append([item])
        else:
            groups[-1].append(item)
    while len(groups) > max_bands:
        gaps = [
            abs(float(np.median([entry["t_median"] for entry in groups[index + 1]])) - float(np.median([entry["t_median"] for entry in groups[index]])))
            for index in range(len(groups) - 1)
        ]
        merge_index = int(np.argmin(gaps))
        groups[merge_index].extend(groups.pop(merge_index + 1))

    bands: list[dict[str, Any]] = []
    for band_id, group in enumerate(groups, start=1):
        s_min = min(item["s_min"] for item in group)
        s_max = max(item["s_max"] for item in group)
        bands.append(
            {
                "band": band_id,
                "items": group,
                "s_min": s_min,
                "s_max": s_max,
                "s_span": s_max - s_min,
                "t_median": float(np.median([item["t_median"] for item in group])),
            },
        )
    return sorted(bands, key=lambda band: band["t_median"])


def choose_main_band(bands: list[dict[str, Any]]) -> int:
    max_span = max(float(band["s_span"]) for band in bands)
    if len(bands) >= 3:
        middle_band = sorted(bands, key=lambda band: band["t_median"])[len(bands) // 2]
        if float(middle_band["s_span"]) >= max_span * 0.70:
            return int(middle_band["band"])
    best = max(bands, key=lambda band: (float(band["s_span"]), -abs(float(band["t_median"]))))
    return int(best["band"])


def assign_features_to_bands(
    stats: list[dict[str, Any]],
    bands: list[dict[str, Any]],
    assign_threshold_m: float,
    max_angle_deg: float,
) -> dict[int, list[str]]:
    assigned: dict[int, list[str]] = {int(band["band"]): [] for band in bands}
    for item in stats:
        if item["angle_to_axis_deg"] > max_angle_deg:
            continue
        nearest = min(bands, key=lambda band: abs(float(item["t_median"]) - float(band["t_median"])))
        if abs(float(item["t_median"]) - float(nearest["t_median"])) <= assign_threshold_m:
            assigned[int(nearest["band"])].append(str(item["feature_id"]))
    return assigned


def binned_band_points(
    candidates: list[LineFeature],
    feature_ids: list[str],
    axis: Axis,
    bin_size_m: float,
) -> list[tuple[float, float]]:
    wanted = set(feature_ids)
    samples: list[tuple[float, float]] = []
    for feature in candidates:
        if feature.feature_id not in wanted:
            continue
        s_values, t_values = project_points(axis, feature.coords)
        samples.extend((float(s), float(t)) for s, t in zip(s_values, t_values))
    if not samples:
        return []
    s_min = min(s for s, _ in samples)
    bins: dict[int, list[tuple[float, float]]] = {}
    for s, t in samples:
        bin_id = int(math.floor((s - s_min) / bin_size_m))
        bins.setdefault(bin_id, []).append((s, t))
    points: list[tuple[float, float]] = []
    for bin_id in sorted(bins):
        bucket = bins[bin_id]
        points.append(
            (
                float(np.median([s for s, _ in bucket])),
                float(np.median([t for _, t in bucket])),
            ),
        )
    return smooth_st(points, window=5)


def smooth_st(points: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    if len(points) <= 2 or window <= 1:
        return points
    radius = window // 2
    smoothed: list[tuple[float, float]] = []
    for index, (s, _) in enumerate(points):
        left = max(0, index - radius)
        right = min(len(points), index + radius + 1)
        smoothed.append((s, float(np.median([t for _, t in points[left:right]]))))
    return smoothed


def interpolate_continuous_line(points: list[tuple[float, float]], axis: Axis, step_m: float) -> list[tuple[float, float]]:
    if len(points) < 2:
        raise ValueError("Need at least two binned points to create the through main line.")
    ordered = sorted(points)
    s_values = np.asarray([s for s, _ in ordered], dtype=float)
    t_values = np.asarray([t for _, t in ordered], dtype=float)
    s_grid = np.arange(float(s_values[0]), float(s_values[-1]) + step_m, step_m)
    t_grid = np.interp(s_grid, s_values, t_values)
    return [unproject_point(axis, float(s), float(t)) for s, t in zip(s_grid, t_grid)]


def fit_continuous_main_line(
    points: list[tuple[float, float]],
    axis: Axis,
    step_m: float,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    if len(points) < 2:
        raise ValueError("Need at least two binned points to create the through main line.")
    ordered = sorted(points)
    s_values = np.asarray([s for s, _ in ordered], dtype=float)
    t_values = np.asarray([t for _, t in ordered], dtype=float)
    keep = np.ones_like(s_values, dtype=bool)
    slope = 0.0
    intercept = float(np.median(t_values))
    for _ in range(4):
        if int(np.count_nonzero(keep)) >= 2:
            slope, intercept = np.polyfit(s_values[keep], t_values[keep], 1)
        residuals = np.abs(t_values - (slope * s_values + intercept))
        cutoff = max(1.25, float(np.percentile(residuals[keep], 75)) * 2.5)
        next_keep = residuals <= cutoff
        if int(np.count_nonzero(next_keep)) < max(2, int(len(s_values) * 0.45)):
            break
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    s_grid = np.arange(float(s_values[0]), float(s_values[-1]) + step_m, step_m)
    t_grid = slope * s_grid + intercept
    fit = {
        "fit_policy": "robust_linear_through_main",
        "slope": float(slope),
        "intercept": float(intercept),
        "input_point_count": int(len(points)),
        "kept_point_count": int(np.count_nonzero(keep)),
        "residual_median_m": float(np.median(np.abs(t_values[keep] - (slope * s_values[keep] + intercept)))) if np.any(keep) else None,
    }
    return [unproject_point(axis, float(s), float(t)) for s, t in zip(s_grid, t_grid)], fit


def split_observed_segments(points: list[tuple[float, float]], axis: Axis, *, max_gap_m: float) -> list[list[tuple[float, float]]]:
    if len(points) < 2:
        return []
    ordered = sorted(points)
    groups: list[list[tuple[float, float]]] = [[ordered[0]]]
    for point in ordered[1:]:
        if point[0] - groups[-1][-1][0] > max_gap_m:
            groups.append([point])
        else:
            groups[-1].append(point)
    return [[unproject_point(axis, s, t) for s, t in group] for group in groups if len(group) >= 2]


def vectorize_switch_masks(
    *,
    mask_dir: Path,
    records: dict[str, TileGeoRecord],
    min_component_pixels: int,
    bbox_pad_pixels: int,
) -> list[dict[str, Any]]:
    if not mask_dir.exists():
        return []
    lookup = build_tile_lookup(records)
    features: list[dict[str, Any]] = []
    try:
        from scipy import ndimage
    except Exception:
        ndimage = None
    for mask_path in sorted(mask_dir.glob("*.png")):
        record = match_tile_record(mask_path.name, lookup)
        if record is None:
            continue
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        if not np.any(mask):
            continue
        components: list[tuple[np.ndarray, int]]
        if ndimage is not None:
            labeled, count = ndimage.label(mask)
            components = [(labeled == component_id, component_id) for component_id in range(1, count + 1)]
        else:
            components = [(mask, 1)]
        kept_index = 0
        for component_mask, component_id in components:
            ys, xs = np.nonzero(component_mask)
            pixel_count = int(xs.size)
            if pixel_count < min_component_pixels:
                continue
            kept_index += 1
            x0 = max(0, int(xs.min()) - bbox_pad_pixels)
            x1 = min(mask.shape[1], int(xs.max()) + bbox_pad_pixels + 1)
            y0 = max(0, int(ys.min()) - bbox_pad_pixels)
            y1 = min(mask.shape[0], int(ys.max()) + bbox_pad_pixels + 1)
            ring_pixels = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            ring_map = [pixel_to_map(record.tile_transform, point) for point in ring_pixels]
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "image_name": mask_path.name,
                        "tile_name": record.tile_name,
                        "component_id": int(component_id),
                        "kept_component_index": kept_index,
                        "positive_pixels": pixel_count,
                        "bbox_pixels": [x0, y0, x1, y1],
                        "topology_role": "switch_workzone",
                        "network_source": "switch_area_model_v7",
                        "geometry_policy": "component_bbox_with_padding",
                        "epsg": record.epsg,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[round_coord(x), round_coord(y)] for x, y in ring_map]],
                    },
                },
            )
    return features


def merge_switch_workzones(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not features:
        return []
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except Exception:
        return features

    polygons = []
    for feature in features:
        coords = ((feature.get("geometry") or {}).get("coordinates") or [[]])[0]
        if len(coords) < 4:
            continue
        polygons.append(Polygon([(float(x), float(y)) for x, y, *_ in coords]))
    if not polygons:
        return []
    merged = unary_union(polygons)
    geoms = list(getattr(merged, "geoms", [merged]))
    epsg = None
    for feature in features:
        epsg = (feature.get("properties") or {}).get("epsg")
        if epsg is not None:
            break

    merged_features: list[dict[str, Any]] = []
    for index, polygon in enumerate(geoms, start=1):
        if polygon.is_empty or polygon.area <= 0:
            continue
        exterior = [(round_coord(x), round_coord(y)) for x, y in polygon.exterior.coords]
        merged_features.append(
            {
                "type": "Feature",
                "properties": {
                    "workzone_id": f"switch_workzone_{index:03d}",
                    "topology_role": "switch_workzone",
                    "network_source": "switch_area_model_v7",
                    "geometry_policy": "merged_component_bboxes",
                    "source_component_count": len(features),
                    "area_m2": round(float(polygon.area), 3),
                    "epsg": epsg,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [exterior],
                },
            },
        )
    return merged_features


def output_line_feature(
    *,
    role: str,
    line_id: str,
    band: int,
    coords: list[tuple[float, float]],
    extra: dict[str, Any],
) -> dict[str, Any]:
    props = {
        "line_id": line_id,
        "topology_role": role,
        "network_source": "tonghaigang_topology_skeleton_v1",
        "band": band,
        "length_m": round(polyline_length(coords), 3),
        "point_count": len(coords),
    }
    props.update(extra)
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {
            "type": "LineString",
            "coordinates": [[round_coord(x), round_coord(y)] for x, y in coords],
        },
    }


def build_tile_lookup(records: dict[str, TileGeoRecord]) -> dict[str, TileGeoRecord]:
    lookup: dict[str, TileGeoRecord] = {}
    for record in records.values():
        keys = {
            norm_key(record.tile_name),
            norm_key(Path(record.tile_name).name),
            norm_key(record.image_path),
            norm_key(Path(record.image_path).name),
        }
        for key in keys:
            if key:
                lookup.setdefault(key, record)
    return lookup


def match_tile_record(image_name: str, lookup: dict[str, TileGeoRecord]) -> TileGeoRecord | None:
    keys = (
        norm_key(image_name),
        norm_key(Path(image_name.replace("\\", "/")).name),
    )
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def norm_key(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name.lower()


def pixel_to_map(
    transform: tuple[float, float, float, float, float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    col, row = point
    return a * col + b * row + c, d * col + e * row + f


def polyline_length(coords: list[tuple[float, float]]) -> float:
    total = 0.0
    for left, right in zip(coords, coords[1:]):
        total += math.hypot(right[0] - left[0], right[1] - left[1])
    return total


def geojson_crs(records: dict[str, TileGeoRecord]) -> dict[str, Any] | None:
    epsg_values = sorted({record.epsg for record in records.values() if record.epsg is not None})
    if len(epsg_values) == 1:
        return {"type": "name", "properties": {"name": f"EPSG:{epsg_values[0]}"}}
    return None


def write_geojson(path: Path, name: str, features: list[dict[str, Any]], crs: dict[str, Any] | None) -> None:
    payload: dict[str, Any] = {
        "type": "FeatureCollection",
        "name": name,
        "features": features,
    }
    if crs is not None:
        payload["crs"] = crs
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def round_coord(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
