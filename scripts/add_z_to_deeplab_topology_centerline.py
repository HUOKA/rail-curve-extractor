from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = (
    Path("output")
    / "dom_centerline_strict_auto_v1"
    / "global_centerline_review_tangent_occlusion"
    / "global_centerline_2d.geojson"
)
DEFAULT_OUTPUT_DIR = (
    Path("output")
    / "dom_centerline_strict_auto_v1"
    / "global_centerline_review_tangent_occlusion_z"
)
DEFAULT_DSM = Path("D:/") / "\u6b63\u5c04" / "lidars" / "terra_dsm" / "dsm.tif"
DEFAULT_LAS_DIR = Path("D:/") / "\u6b63\u5c04" / "lidars" / "terra_las"
DEFAULT_EPSG = 32651


@dataclass(slots=True)
class LineFeature:
    index: int
    properties: dict[str, Any]
    coords: np.ndarray


@dataclass(slots=True)
class DenseLine:
    source: LineFeature
    coords: np.ndarray
    stations: np.ndarray
    tangents: np.ndarray


@dataclass(slots=True)
class ZLine:
    dense: DenseLine
    dsm_z: np.ndarray
    las_z: np.ndarray
    las_counts: np.ndarray
    raw_z: np.ndarray
    smooth_z: np.ndarray
    source: str
    fallback_count: int
    outlier_count: int
    bridge_z_mode: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add smoothed LAS/DSM Z values to a 2D centerline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dsm", type=Path, default=DEFAULT_DSM)
    parser.add_argument("--las-dir", type=Path, default=DEFAULT_LAS_DIR)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    parser.add_argument("--spacing-m", type=float, default=1.0)
    parser.add_argument("--rail-offset-m", type=float, default=0.7175)
    parser.add_argument("--las-radius-m", type=float, default=0.35)
    parser.add_argument("--las-quantile", type=float, default=0.85)
    parser.add_argument("--las-min-points", type=int, default=3)
    parser.add_argument("--las-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--smooth-window-m", type=float, default=51.0)
    parser.add_argument("--despike-window-m", type=float, default=201.0)
    parser.add_argument("--despike-threshold-m", type=float, default=0.45)
    parser.add_argument("--smooth-polyorder", type=int, default=2)
    parser.add_argument("--endpoint-tolerance-m", type=float, default=0.25)
    parser.add_argument("--endpoint-taper-m", type=float, default=15.0)
    parser.add_argument("--bridge-replace-threshold-m", type=float, default=0.50)
    parser.add_argument("--source", choices=["auto", "las", "dsm"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect and compare Z sources; do not write shp outputs.")
    return parser.parse_args()


def read_features(path: Path) -> tuple[list[LineFeature], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features: list[LineFeature] = []
    for index, feature in enumerate(payload.get("features") or []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = np.asarray([[float(c[0]), float(c[1])] for c in geometry.get("coordinates") or []], dtype=float)
        if coords.shape[0] < 2:
            continue
        features.append(LineFeature(index=index, properties=dict(feature.get("properties") or {}), coords=coords))
    if not features:
        raise ValueError(f"No LineString features found in {path}")
    return features, payload


def line_length(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return 0.0
    delta = np.diff(coords[:, :2], axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def densify_line(feature: LineFeature, spacing_m: float) -> DenseLine:
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive.")
    segments: list[tuple[float, float, float, float, float, float, float, float, float]] = []
    total = 0.0
    coords = feature.coords
    for index in range(1, coords.shape[0]):
        ax, ay = coords[index - 1]
        bx, by = coords[index]
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        segments.append((total, total + length, ax, ay, bx, by, dx / length, dy / length, length))
        total += length
    if not segments:
        raise ValueError(f"Feature {feature.properties.get('line_id', feature.index)} has no usable segment.")
    sample_count = max(2, int(math.ceil(total / spacing_m)) + 1)
    stations = np.linspace(0.0, total, sample_count)
    out_coords = np.empty((sample_count, 2), dtype=float)
    tangents = np.empty((sample_count, 2), dtype=float)
    seg_index = 0
    for sample_index, station in enumerate(stations):
        while seg_index < len(segments) - 1 and station > segments[seg_index][1] + 1e-9:
            seg_index += 1
        s0, _s1, ax, ay, bx, by, tx, ty, length = segments[seg_index]
        t = min(1.0, max(0.0, (float(station) - s0) / length))
        out_coords[sample_index] = [ax + (bx - ax) * t, ay + (by - ay) * t]
        tangents[sample_index] = [tx, ty]
    out_coords[0] = coords[0]
    out_coords[-1] = coords[-1]
    return DenseLine(source=feature, coords=out_coords, stations=stations, tangents=tangents)


def normals_for(dense: DenseLine) -> np.ndarray:
    tangents = dense.tangents
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def valid_z(values: np.ndarray, nodata: float | int | None) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if nodata is not None:
        out = np.where(out == float(nodata), np.nan, out)
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def sample_dsm_points(dsm_path: Path, points: np.ndarray) -> np.ndarray:
    import rasterio

    with rasterio.open(dsm_path) as dataset:
        values = np.asarray([float(item[0]) for item in dataset.sample([(float(x), float(y)) for x, y in points])], dtype=float)
        return valid_z(values, dataset.nodata)


def sample_dsm_lines(dsm_path: Path, dense_lines: list[DenseLine], rail_offset_m: float) -> list[np.ndarray]:
    dsm_values: list[np.ndarray] = []
    for dense in dense_lines:
        normals = normals_for(dense)
        center = sample_dsm_points(dsm_path, dense.coords)
        left = sample_dsm_points(dsm_path, dense.coords + normals * rail_offset_m)
        right = sample_dsm_points(dsm_path, dense.coords - normals * rail_offset_m)
        pair = np.nanmean(np.vstack([left, right]), axis=0)
        z = np.where(np.isfinite(pair), pair, center)
        dsm_values.append(z)
    return dsm_values


def las_files_from_dir(las_dir: Path) -> list[Path]:
    if las_dir.is_file():
        return [las_dir]
    files = sorted(
        [path for path in las_dir.glob("cloud*.las") if path.name.lower() != "cloud_merged.las"],
        key=lambda path: int(path.stem.replace("cloud", "")) if path.stem.replace("cloud", "").isdigit() else 10**9,
    )
    if not files:
        files = sorted(list(las_dir.glob("*.las")) + list(las_dir.glob("*.laz")))
    if not files:
        raise FileNotFoundError(f"No LAS/LAZ files found under {las_dir}")
    return files


def build_las_queries(dense_lines: list[DenseLine], rail_offset_m: float) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    query_points: list[np.ndarray] = []
    meta: list[tuple[int, int, int]] = []
    for line_index, dense in enumerate(dense_lines):
        normals = normals_for(dense)
        for side_index, sign in enumerate((1.0, -1.0)):
            points = dense.coords + normals * (sign * rail_offset_m)
            for sample_index, point in enumerate(points):
                query_points.append(point)
                meta.append((line_index, sample_index, side_index))
    return np.asarray(query_points, dtype=float), meta


def transformed_header_bounds(header: Any, target_epsg: int) -> tuple[float, float, float, float]:
    src_crs = header.parse_crs()
    xs = np.asarray([header.mins[0], header.maxs[0], header.maxs[0], header.mins[0]], dtype=float)
    ys = np.asarray([header.mins[1], header.mins[1], header.maxs[1], header.maxs[1]], dtype=float)
    if src_crs is not None and src_crs.to_epsg() != target_epsg:
        from pyproj import CRS, Transformer

        transformer = Transformer.from_crs(src_crs, CRS.from_epsg(target_epsg), always_xy=True)
        xs, ys = transformer.transform(xs, ys)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def collect_las_values(
    las_files: list[Path],
    query_points: np.ndarray,
    *,
    radius_m: float,
    chunk_size: int,
    target_epsg: int,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    import laspy
    from scipy.spatial import cKDTree

    tree = cKDTree(query_points)
    values: list[list[float]] = [[] for _ in range(query_points.shape[0])]
    min_x, min_y = query_points.min(axis=0) - radius_m
    max_x, max_y = query_points.max(axis=0) + radius_m
    las_reports: list[dict[str, Any]] = []
    for path in las_files:
        with laspy.open(path) as reader:
            bx0, by0, bx1, by1 = transformed_header_bounds(reader.header, target_epsg)
            src_crs = reader.header.parse_crs()
            transformer = None
            if src_crs is not None and src_crs.to_epsg() != target_epsg:
                from pyproj import CRS, Transformer

                transformer = Transformer.from_crs(src_crs, CRS.from_epsg(target_epsg), always_xy=True)
            report = {
                "path": str(path),
                "point_count": int(reader.header.point_count),
                "source_epsg": src_crs.to_epsg() if src_crs is not None else None,
                "transformed_bounds": [bx0, by0, bx1, by1],
                "read_points": 0,
                "assigned_points": 0,
                "skipped_by_bounds": False,
            }
            if bx1 < min_x or bx0 > max_x or by1 < min_y or by0 > max_y:
                report["skipped_by_bounds"] = True
                las_reports.append(report)
                continue
            for points in reader.chunk_iterator(chunk_size):
                if len(points) == 0:
                    continue
                x = np.asarray(points.x, dtype=float)
                y = np.asarray(points.y, dtype=float)
                z = np.asarray(points.z, dtype=float)
                report["read_points"] += int(x.size)
                if transformer is not None:
                    x, y = transformer.transform(x, y)
                    x = np.asarray(x, dtype=float)
                    y = np.asarray(y, dtype=float)
                mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y) & (z > 5.0) & (z < 80.0)
                if not np.any(mask):
                    continue
                xy = np.column_stack([x[mask], y[mask]])
                distances, indices = tree.query(xy, k=1, distance_upper_bound=radius_m, workers=-1)
                ok = np.isfinite(distances) & (indices < query_points.shape[0])
                if not np.any(ok):
                    continue
                assigned_indices = indices[ok]
                assigned_z = z[mask][ok]
                order = np.argsort(assigned_indices)
                sorted_indices = assigned_indices[order]
                sorted_z = assigned_z[order]
                cuts = np.flatnonzero(np.diff(sorted_indices)) + 1
                starts = np.r_[0, cuts]
                ends = np.r_[cuts, sorted_indices.size]
                for start, end in zip(starts, ends):
                    query_index = int(sorted_indices[start])
                    chunk_z = sorted_z[start:end]
                    values[query_index].extend(chunk_z.astype(float).tolist())
                    report["assigned_points"] += int(chunk_z.size)
            las_reports.append(report)
    return values, las_reports


def las_z_for_lines(
    dense_lines: list[DenseLine],
    values: list[list[float]],
    meta: list[tuple[int, int, int]],
    *,
    quantile: float,
    min_points: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    side_values: list[list[list[float]]] = []
    for dense in dense_lines:
        side_values.append([[math.nan, math.nan] for _ in range(dense.coords.shape[0])])
    counts: list[np.ndarray] = [np.zeros(dense.coords.shape[0], dtype=np.int32) for dense in dense_lines]
    for query_index, query_values in enumerate(values):
        line_index, sample_index, side_index = meta[query_index]
        counts[line_index][sample_index] += len(query_values)
        if len(query_values) < min_points:
            continue
        side_values[line_index][sample_index][side_index] = float(np.quantile(np.asarray(query_values, dtype=float), quantile))
    line_z: list[np.ndarray] = []
    for line_index, dense in enumerate(dense_lines):
        z = np.full(dense.coords.shape[0], np.nan, dtype=float)
        for sample_index, pair in enumerate(side_values[line_index]):
            valid = [value for value in pair if math.isfinite(value)]
            if valid:
                z[sample_index] = float(np.mean(valid))
        line_z.append(z)
    return line_z, counts


def interpolate_missing(z: np.ndarray, fallback: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    out = np.asarray(z, dtype=float).copy()
    fallback_count = 0
    if fallback is not None:
        use_fallback = ~np.isfinite(out) & np.isfinite(fallback)
        fallback_count = int(np.count_nonzero(use_fallback))
        out[use_fallback] = fallback[use_fallback]
    finite = np.isfinite(out)
    if finite.all():
        return out, fallback_count
    if not np.any(finite):
        return out, fallback_count
    x = np.arange(out.size, dtype=float)
    out[~finite] = np.interp(x[~finite], x[finite], out[finite])
    return out, fallback_count


def rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    radius = window // 2
    out = np.empty(values.shape, dtype=float)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        out[index] = float(np.nanmedian(values[start:end]))
    return out


def choose_odd_window(sample_count: int, spacing_m: float, window_m: float, minimum: int = 5) -> int:
    if sample_count < 3:
        return sample_count
    window = max(minimum, int(round(window_m / max(spacing_m, 1e-6))))
    if window % 2 == 0:
        window += 1
    window = min(window, sample_count if sample_count % 2 == 1 else sample_count - 1)
    return max(3, window)


def smooth_z_profile(
    z: np.ndarray,
    *,
    spacing_m: float,
    window_m: float,
    despike_window_m: float,
    despike_threshold_m: float,
    polyorder: int,
) -> tuple[np.ndarray, int]:
    clean, _ = interpolate_missing(z)
    if not np.any(np.isfinite(clean)):
        return clean, 0
    if clean.size < 5:
        return clean, 0
    median_window = choose_odd_window(clean.size, spacing_m, window_m, minimum=5)
    median = rolling_nanmedian(clean, median_window)
    residual = clean - median
    mad = float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
    threshold = max(0.18, 6.0 * 1.4826 * mad)
    outliers = np.abs(residual) > threshold
    cleaned = clean.copy()
    cleaned[outliers] = median[outliers]
    despike_window = choose_odd_window(cleaned.size, spacing_m, despike_window_m, minimum=9)
    long_median = rolling_nanmedian(cleaned, despike_window)
    long_residual = cleaned - long_median
    long_mad = float(np.nanmedian(np.abs(long_residual - np.nanmedian(long_residual))))
    long_threshold = max(despike_threshold_m, 8.0 * 1.4826 * long_mad)
    broad_outliers = np.abs(long_residual) > long_threshold
    cleaned[broad_outliers] = long_median[broad_outliers]
    try:
        from scipy.signal import savgol_filter

        smooth_window = choose_odd_window(cleaned.size, spacing_m, window_m, minimum=7)
        effective_polyorder = min(polyorder, smooth_window - 1)
        smoothed = savgol_filter(cleaned, smooth_window, effective_polyorder, mode="interp")
    except Exception:
        smooth_window = choose_odd_window(cleaned.size, spacing_m, max(5.0, window_m * 0.5), minimum=3)
        smoothed = rolling_nanmedian(cleaned, smooth_window)
    return np.asarray(smoothed, dtype=float), int(np.count_nonzero(outliers | broad_outliers))


def is_bridge_line(feature: LineFeature) -> bool:
    props = feature.properties
    text = " ".join(str(props.get(key, "")) for key in ("line_id", "network_role", "source_layer", "qa_status"))
    return "bridge" in text.lower()


def is_topology_gap_bridge(feature: LineFeature) -> bool:
    props = feature.properties
    return str(props.get("source_layer", "")).lower() == "topology_gap_bridge" or str(props.get("network_role", "")).lower() == "straight_gap_bridge"


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def endpoint_targets(z_lines: list[ZLine], tolerance_m: float) -> dict[tuple[int, str], float]:
    endpoints: list[tuple[int, str, np.ndarray]] = []
    for line_index, z_line in enumerate(z_lines):
        endpoints.append((line_index, "start", z_line.dense.coords[0]))
        endpoints.append((line_index, "end", z_line.dense.coords[-1]))
    uf = UnionFind(len(endpoints))
    for i, (_, _, a) in enumerate(endpoints):
        for j in range(i + 1, len(endpoints)):
            _, _, b = endpoints[j]
            if float(np.hypot(*(a - b))) <= tolerance_m:
                uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for index in range(len(endpoints)):
        groups.setdefault(uf.find(index), []).append(index)
    targets: dict[tuple[int, str], float] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        non_bridge_values: list[float] = []
        all_values: list[float] = []
        for endpoint_index in members:
            line_index, side, _point = endpoints[endpoint_index]
            z_line = z_lines[line_index]
            value = z_line.smooth_z[0] if side == "start" else z_line.smooth_z[-1]
            if not math.isfinite(float(value)):
                continue
            all_values.append(float(value))
            if not is_bridge_line(z_line.dense.source):
                non_bridge_values.append(float(value))
        source_values = non_bridge_values if non_bridge_values else all_values
        if not source_values:
            continue
        target = float(np.median(source_values))
        for endpoint_index in members:
            line_index, side, _point = endpoints[endpoint_index]
            targets[(line_index, side)] = target
    return targets


def apply_endpoint_taper(z: np.ndarray, stations: np.ndarray, start_target: float | None, end_target: float | None, taper_m: float) -> np.ndarray:
    out = z.copy()
    if start_target is not None and math.isfinite(start_target) and out.size:
        delta = float(start_target) - float(out[0])
        weights = np.clip(1.0 - stations / max(taper_m, 1e-6), 0.0, 1.0)
        out += delta * weights
        out[0] = float(start_target)
    if end_target is not None and math.isfinite(end_target) and out.size:
        delta = float(end_target) - float(out[-1])
        weights = np.clip(1.0 - (stations[-1] - stations) / max(taper_m, 1e-6), 0.0, 1.0)
        out += delta * weights
        out[-1] = float(end_target)
    return out


def apply_topology_z_constraints(z_lines: list[ZLine], *, endpoint_tolerance_m: float, endpoint_taper_m: float, bridge_replace_threshold_m: float) -> None:
    targets = endpoint_targets(z_lines, endpoint_tolerance_m)
    for line_index, z_line in enumerate(z_lines):
        start_target = targets.get((line_index, "start"))
        end_target = targets.get((line_index, "end"))
        if start_target is None and end_target is None:
            continue
        if is_bridge_line(z_line.dense.source) and start_target is not None and end_target is not None:
            interp = np.interp(z_line.dense.stations, [z_line.dense.stations[0], z_line.dense.stations[-1]], [start_target, end_target])
            median_delta = float(np.nanmedian(np.abs(z_line.smooth_z - interp)))
            if is_topology_gap_bridge(z_line.dense.source) or median_delta > bridge_replace_threshold_m:
                z_line.smooth_z = interp
                z_line.bridge_z_mode = "endpoint_interpolated"
                continue
        z_line.smooth_z = apply_endpoint_taper(z_line.smooth_z, z_line.dense.stations, start_target, end_target, endpoint_taper_m)
        if is_bridge_line(z_line.dense.source):
            z_line.bridge_z_mode = "endpoint_tapered"


def roughness(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"median": math.nan, "std": math.nan, "p95_abs_dz": math.nan, "max_abs_dz": math.nan, "p99": math.nan}
    dz = np.abs(np.diff(finite))
    return {
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p95_abs_dz": float(np.percentile(dz, 95)) if dz.size else 0.0,
        "max_abs_dz": float(np.max(dz)) if dz.size else 0.0,
        "p99": float(np.percentile(finite, 99)),
    }


def aggregate_roughness(lines: list[np.ndarray]) -> dict[str, float]:
    if not lines:
        return {"median": math.nan, "std": math.nan, "p95_abs_dz": math.nan, "max_abs_dz": math.nan, "p99": math.nan}
    finite_values = [np.asarray(line, dtype=float)[np.isfinite(line)] for line in lines]
    finite_values = [line for line in finite_values if line.size]
    if not finite_values:
        return {"median": math.nan, "std": math.nan, "p95_abs_dz": math.nan, "max_abs_dz": math.nan, "p99": math.nan}
    all_values = np.concatenate(finite_values)
    diffs = [np.abs(np.diff(line)) for line in finite_values if line.size > 1]
    all_diffs = np.concatenate(diffs) if diffs else np.asarray([], dtype=float)
    return {
        "median": float(np.median(all_values)),
        "std": float(np.std(all_values)),
        "p95_abs_dz": float(np.percentile(all_diffs, 95)) if all_diffs.size else 0.0,
        "max_abs_dz": float(np.max(all_diffs)) if all_diffs.size else 0.0,
        "p99": float(np.percentile(all_values, 99)),
    }


def z_stats(z: np.ndarray) -> dict[str, float]:
    finite = np.asarray(z, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"z_min": math.nan, "z_max": math.nan, "z_med": math.nan, "z_std": math.nan}
    return {
        "z_min": float(np.min(finite)),
        "z_max": float(np.max(finite)),
        "z_med": float(np.median(finite)),
        "z_std": float(np.std(finite)),
    }


def choose_source(args: argparse.Namespace, dsm_lines: list[np.ndarray], las_lines: list[np.ndarray]) -> str:
    if args.source != "auto":
        return args.source
    las_all = np.concatenate(las_lines)
    dsm_all = np.concatenate(dsm_lines)
    las_valid = float(np.isfinite(las_all).mean())
    dsm_valid = float(np.isfinite(dsm_all).mean())
    if las_valid >= 0.90:
        return "las"
    if dsm_valid >= las_valid:
        return "dsm"
    return "las"


def build_z_lines(
    dense_lines: list[DenseLine],
    dsm_lines: list[np.ndarray],
    las_lines: list[np.ndarray],
    las_counts: list[np.ndarray],
    *,
    source: str,
    spacing_m: float,
    smooth_window_m: float,
    despike_window_m: float,
    despike_threshold_m: float,
    polyorder: int,
) -> list[ZLine]:
    z_lines: list[ZLine] = []
    for dense, dsm_z, las_z, counts in zip(dense_lines, dsm_lines, las_lines, las_counts):
        if source == "las":
            raw, fallback_count = interpolate_missing(las_z, dsm_z)
            source_name = "las_rail_pair"
        else:
            raw, fallback_count = interpolate_missing(dsm_z)
            source_name = "dsm_rail_pair"
        smooth, outlier_count = smooth_z_profile(
            raw,
            spacing_m=spacing_m,
            window_m=smooth_window_m,
            despike_window_m=despike_window_m,
            despike_threshold_m=despike_threshold_m,
            polyorder=polyorder,
        )
        z_lines.append(
            ZLine(
                dense=dense,
                dsm_z=dsm_z,
                las_z=las_z,
                las_counts=counts,
                raw_z=raw,
                smooth_z=smooth,
                source=source_name,
                fallback_count=fallback_count,
                outlier_count=outlier_count,
            )
        )
    return z_lines


def make_3d_feature(z_line: ZLine) -> dict[str, Any]:
    source_props = z_line.dense.source.properties
    coords = [
        [round(float(x), 6), round(float(y), 6), round(float(z), 3)]
        for (x, y), z in zip(z_line.dense.coords, z_line.smooth_z)
    ]
    stats = z_stats(z_line.smooth_z)
    props = dict(source_props)
    props.update(
        {
            "z_source": z_line.source,
            "z_spacing_m": round(float(np.median(np.diff(z_line.dense.stations))) if z_line.dense.stations.size > 1 else 0.0, 3),
            "z_valid_ratio": round(float(np.isfinite(z_line.raw_z).mean()), 4),
            "z_fallback_n": int(z_line.fallback_count),
            "z_outlier_n": int(z_line.outlier_count),
            "z_bridge_mode": z_line.bridge_z_mode,
            "z_min": round(stats["z_min"], 3),
            "z_max": round(stats["z_max"], 3),
            "z_med": round(stats["z_med"], 3),
            "z_std": round(stats["z_std"], 3),
        }
    )
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def write_geojson(path: Path, features: list[dict[str, Any]], epsg: int) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def write_polylinez_shapefile(path: Path, z_lines: list[ZLine], epsg: int) -> None:
    import rasterio.crs
    import shapefile

    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYLINEZ, encoding="utf-8")
    writer.field("line_id", "C", size=96)
    writer.field("net_role", "C", size=32)
    writer.field("src_layer", "C", size=32)
    writer.field("branch_id", "C", size=24)
    writer.field("qa_status", "C", size=36)
    writer.field("len_m", "F", decimal=3)
    writer.field("z_src", "C", size=24)
    writer.field("z_valid", "F", decimal=4)
    writer.field("z_fb_n", "N", size=8)
    writer.field("z_out_n", "N", size=8)
    writer.field("z_min", "F", decimal=3)
    writer.field("z_max", "F", decimal=3)
    writer.field("z_med", "F", decimal=3)
    writer.field("z_std", "F", decimal=3)
    writer.field("z_mode", "C", size=32)
    writer.field("note", "C", size=160)
    for z_line in z_lines:
        props = z_line.dense.source.properties
        stats = z_stats(z_line.smooth_z)
        coords3 = [
            [round(float(x), 6), round(float(y), 6), round(float(z), 3)]
            for (x, y), z in zip(z_line.dense.coords, z_line.smooth_z)
        ]
        writer.linez([coords3])
        writer.record(
            str(props.get("line_id", ""))[:96],
            str(props.get("network_role", ""))[:32],
            str(props.get("source_layer", ""))[:32],
            str(props.get("branch_id", props.get("anchor_id", "")))[:24],
            str(props.get("qa_status", ""))[:36],
            safe_float(props.get("length_m", line_length(z_line.dense.source.coords))),
            z_line.source[:24],
            safe_float(np.isfinite(z_line.raw_z).mean()),
            int(z_line.fallback_count),
            int(z_line.outlier_count),
            safe_float(stats["z_min"]),
            safe_float(stats["z_max"]),
            safe_float(stats["z_med"]),
            safe_float(stats["z_std"]),
            z_line.bridge_z_mode[:32],
            str(props.get("self_note", props.get("review_note", "")))[:160],
        )
    writer.close()
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    path.with_suffix(".prj").write_text(rasterio.crs.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")


def write_review(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DeepLab Topology Centerline Z Review",
        "",
        f"- Base 2D centerline: `{summary['input']}`",
        f"- Output 3D shp: `{summary['outputs'].get('network_z_shp', '')}`",
        f"- Selected source: `{summary['selected_source']}`",
        "- XY policy: input 2D geometry is only densified along existing segments; topology and plan-view alignment are not rebuilt.",
        f"- LAS coverage: {summary['source_comparison']['las']['valid_ratio']:.4f}",
        f"- DSM coverage: {summary['source_comparison']['dsm']['valid_ratio']:.4f}",
        "",
        "## Notes",
        "",
        "- LAS Z uses left/right rail-offset query points at standard-gauge half offset and an 85th percentile height statistic.",
        "- DSM Z is kept as a fallback and comparison source.",
        "- Topology gap bridges use endpoint interpolation when local samples disagree with connected track endpoints.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(args: argparse.Namespace, z_lines: list[ZLine], summary: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = [make_3d_feature(z_line) for z_line in z_lines]
    geojson_path = args.output_dir / "deeplab_topology_centerline_network_z.geojson"
    shp_path = args.output_dir / "deeplab_topology_centerline_network_z.shp"
    write_geojson(geojson_path, features, args.epsg)
    write_polylinez_shapefile(shp_path, z_lines, args.epsg)
    source_qml = args.input.with_suffix(".qml")
    if source_qml.exists():
        shutil.copy2(source_qml, shp_path.with_suffix(".qml"))
    summary["outputs"] = {
        "network_z_geojson": str(geojson_path),
        "network_z_shp": str(shp_path),
        "summary_json": str(args.output_dir / "summary.json"),
        "review_md": str(args.output_dir / "REVIEW.md"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(args.output_dir / "REVIEW.md", summary)


def build_summary(
    args: argparse.Namespace,
    features: list[LineFeature],
    dense_lines: list[DenseLine],
    dsm_lines: list[np.ndarray],
    las_lines: list[np.ndarray],
    las_counts: list[np.ndarray],
    z_lines: list[ZLine] | None,
    las_reports: list[dict[str, Any]],
    selected_source: str,
) -> dict[str, Any]:
    dsm_all = np.concatenate(dsm_lines)
    las_all = np.concatenate(las_lines)
    raw_lines = [z_line.raw_z for z_line in z_lines] if z_lines else []
    smooth_lines = [z_line.smooth_z for z_line in z_lines] if z_lines else []
    smooth_all = np.concatenate(smooth_lines) if smooth_lines else np.asarray([], dtype=float)
    counts_all = np.concatenate(las_counts)
    line_reports: list[dict[str, Any]] = []
    for line_index, dense in enumerate(dense_lines):
        props = dense.source.properties
        report = {
            "line_id": props.get("line_id", f"line_{line_index}"),
            "network_role": props.get("network_role", ""),
            "source_layer": props.get("source_layer", ""),
            "sample_count": int(dense.coords.shape[0]),
            "length_m": round(line_length(dense.source.coords), 3),
            "dsm_valid_ratio": round(float(np.isfinite(dsm_lines[line_index]).mean()), 4),
            "las_valid_ratio": round(float(np.isfinite(las_lines[line_index]).mean()), 4),
            "las_point_count_median": float(np.median(las_counts[line_index])),
            "dsm_roughness": roughness(dsm_lines[line_index]),
            "las_roughness": roughness(las_lines[line_index]),
        }
        if z_lines:
            report.update(
                {
                    "z_source": z_lines[line_index].source,
                    "z_fallback_count": int(z_lines[line_index].fallback_count),
                    "z_outlier_count": int(z_lines[line_index].outlier_count),
                    "z_bridge_mode": z_lines[line_index].bridge_z_mode,
                    "z_stats": z_stats(z_lines[line_index].smooth_z),
                    "smooth_roughness": roughness(z_lines[line_index].smooth_z),
                }
            )
        line_reports.append(report)
    summary = {
        "mode": "deeplab_topology_centerline_z_v1",
        "input": str(args.input),
        "dsm": str(args.dsm),
        "las_dir": str(args.las_dir),
        "feature_count": len(features),
        "dense_vertex_count": int(sum(line.coords.shape[0] for line in dense_lines)),
        "xy_policy": "Input 2D XY is preserved as geometry; output is densified along source segments only.",
        "spacing_m": args.spacing_m,
        "rail_offset_m": args.rail_offset_m,
        "las_radius_m": args.las_radius_m,
        "las_quantile": args.las_quantile,
        "despike_window_m": args.despike_window_m,
        "despike_threshold_m": args.despike_threshold_m,
        "selected_source": selected_source,
        "source_comparison": {
            "dsm": {
                "valid_ratio": float(np.isfinite(dsm_all).mean()),
                **aggregate_roughness(dsm_lines),
            },
            "las": {
                "valid_ratio": float(np.isfinite(las_all).mean()),
                "point_count_median": float(np.median(counts_all)),
                "point_count_p10": float(np.percentile(counts_all, 10)),
                "point_count_p90": float(np.percentile(counts_all, 90)),
                **aggregate_roughness(las_lines),
            },
        },
        "final_profile": {
            "raw": aggregate_roughness(raw_lines) if raw_lines else {},
            "smooth": aggregate_roughness(smooth_lines) if smooth_lines else {},
            "z_stats": z_stats(smooth_all) if smooth_all.size else {},
        },
        "las_reports": las_reports,
        "line_reports": line_reports,
        "outputs": {},
    }
    return summary


def main() -> None:
    args = parse_args()
    features, _payload = read_features(args.input)
    dense_lines = [densify_line(feature, args.spacing_m) for feature in features]
    dsm_lines = sample_dsm_lines(args.dsm, dense_lines, args.rail_offset_m)
    las_query_points, las_meta = build_las_queries(dense_lines, args.rail_offset_m)
    las_files = las_files_from_dir(args.las_dir)
    las_values, las_reports = collect_las_values(
        las_files,
        las_query_points,
        radius_m=args.las_radius_m,
        chunk_size=args.las_chunk_size,
        target_epsg=args.epsg,
    )
    las_lines, las_counts = las_z_for_lines(
        dense_lines,
        las_values,
        las_meta,
        quantile=args.las_quantile,
        min_points=args.las_min_points,
    )
    selected_source = choose_source(args, dsm_lines, las_lines)
    z_lines = build_z_lines(
        dense_lines,
        dsm_lines,
        las_lines,
        las_counts,
        source=selected_source,
        spacing_m=args.spacing_m,
        smooth_window_m=args.smooth_window_m,
        despike_window_m=args.despike_window_m,
        despike_threshold_m=args.despike_threshold_m,
        polyorder=args.smooth_polyorder,
    )
    apply_topology_z_constraints(
        z_lines,
        endpoint_tolerance_m=args.endpoint_tolerance_m,
        endpoint_taper_m=args.endpoint_taper_m,
        bridge_replace_threshold_m=args.bridge_replace_threshold_m,
    )
    summary = build_summary(args, features, dense_lines, dsm_lines, las_lines, las_counts, z_lines, las_reports, selected_source)
    if args.dry_run:
        print(json.dumps(summary["source_comparison"], ensure_ascii=False, indent=2))
        return
    write_outputs(args, z_lines, summary)
    print(json.dumps(summary["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
