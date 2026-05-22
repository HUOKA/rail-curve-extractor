#!/usr/bin/env python3
"""Validate paired crossover centerlines against handheld LAS rail evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import laspy
import numpy as np
from pyproj import CRS, Transformer

import build_turnout_connector_candidates as btc


DEFAULT_CENTERLINES = Path("output/raw_dom_roi_fullpass_v1/turnout_crossover_connectors/turnout_crossover_connector_proposals.geojson")
DEFAULT_LAS_DIR = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u8f68\u9053" / "Las"
DEFAULT_GAUGE_SUMMARY = Path("output/handheld_las_constraints_fullpass_switch_excluded/summary.json")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/turnout_crossover_las_validation")
DEFAULT_EPSG = 32651


class LasChunk(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


class SampleSet(NamedTuple):
    connector_id: str
    props: dict[str, Any]
    station: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    tangent_x: np.ndarray
    tangent_y: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    chord_x: float
    chord_y: float
    step_m: float
    bounds: tuple[float, float, float, float]


class SampleAssignment(NamedTuple):
    index: np.ndarray
    along: np.ndarray
    lateral: np.ndarray
    z: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate crossover centerlines by matching predicted rail coordinates to handheld LAS points.")
    parser.add_argument("--centerlines", type=Path, default=DEFAULT_CENTERLINES)
    parser.add_argument("--las-dir", type=Path, default=DEFAULT_LAS_DIR)
    parser.add_argument("--gauge-summary", type=Path, default=DEFAULT_GAUGE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-epsg", type=int, default=DEFAULT_EPSG)
    parser.add_argument("--gauge-m", type=float, default=0.0, help="Rail gauge in meters. 0 means read previous LAS summary median.")
    parser.add_argument("--sample-step-m", type=float, default=0.75)
    parser.add_argument("--chunk-size", type=int, default=750_000)
    parser.add_argument("--assign-neighbors", type=int, default=3)
    parser.add_argument("--along-half-window-m", type=float, default=0.45)
    parser.add_argument("--corridor-m", type=float, default=1.35)
    parser.add_argument("--rail-search-m", type=float, default=0.32)
    parser.add_argument("--residual-bin-m", type=float, default=0.02)
    parser.add_argument("--z-bin-size-m", type=float, default=0.02)
    parser.add_argument("--ground-percentile", type=float, default=20.0)
    parser.add_argument("--min-height-above-local-m", type=float, default=0.08)
    parser.add_argument("--max-height-above-local-m", type=float, default=0.35)
    parser.add_argument("--min-side-points", type=int, default=18)
    parser.add_argument("--min-correction-samples", type=int, default=8)
    parser.add_argument("--max-correction-m", type=float, default=0.35)
    parser.add_argument("--endpoint-lock-m", type=float, default=8.0, help="Keep this distance from both crossover endpoints unchanged.")
    parser.add_argument("--endpoint-taper-m", type=float, default=12.0, help="Smoothly fade LAS correction in after the locked endpoint distance.")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    centerline_path = args.centerlines.expanduser().resolve()
    centerline_features = btc.load_line_features(centerline_path)
    if not centerline_features:
        raise ValueError(f"No centerline LineString features found: {centerline_path}")

    gauge_m = resolve_gauge(args.gauge_m, args.gauge_summary.expanduser().resolve())
    sample_sets = [build_samples(feature, gauge_m=gauge_m, sample_step_m=args.sample_step_m) for feature in centerline_features]
    las_files = sorted(args.las_dir.expanduser().resolve().glob("*.las")) + sorted(args.las_dir.expanduser().resolve().glob("*.laz"))
    if not las_files:
        raise FileNotFoundError(f"No LAS/LAZ files found under: {args.las_dir}")

    z_min, z_max, file_summaries = inspect_las_headers(las_files, target_epsg=args.target_epsg)
    z_edges = make_edges(z_min, z_max, args.z_bin_size_m, pad=0.5)
    residual_edges = make_edges(-args.rail_search_m, args.rail_search_m, args.residual_bin_m, pad=0.0)

    z_hists = {samples.connector_id: np.zeros((samples.station.size, z_edges.size - 1), dtype=np.int64) for samples in sample_sets}
    for chunk in iter_las_chunks(las_files, target_epsg=args.target_epsg, chunk_size=args.chunk_size):
        for samples in sample_sets:
            assignment = assign_chunk_to_samples(chunk, samples, args=args)
            if assignment.index.size == 0:
                continue
            z_index = bin_indices(assignment.z, z_edges)
            valid = z_index >= 0
            if np.any(valid):
                np.add.at(z_hists[samples.connector_id], (assignment.index[valid], z_index[valid]), 1)

    ground_by_connector = {
        connector_id: percentile_by_histogram(hist, z_edges, args.ground_percentile)
        for connector_id, hist in z_hists.items()
    }
    residual_hists = {
        samples.connector_id: np.zeros((samples.station.size, 2, residual_edges.size - 1), dtype=np.int64)
        for samples in sample_sets
    }
    corridor_counts = {samples.connector_id: np.zeros(samples.station.size, dtype=np.int64) for samples in sample_sets}
    rail_like_counts = {samples.connector_id: np.zeros(samples.station.size, dtype=np.int64) for samples in sample_sets}

    rail_offsets = np.asarray([-gauge_m / 2.0, gauge_m / 2.0], dtype=float)
    for chunk in iter_las_chunks(las_files, target_epsg=args.target_epsg, chunk_size=args.chunk_size):
        for samples in sample_sets:
            assignment = assign_chunk_to_samples(chunk, samples, args=args)
            if assignment.index.size == 0:
                continue
            np.add.at(corridor_counts[samples.connector_id], assignment.index, 1)
            ground = ground_by_connector[samples.connector_id][assignment.index]
            z_above = assignment.z - ground
            rail_like = (
                np.isfinite(z_above)
                & (z_above >= args.min_height_above_local_m)
                & (z_above <= args.max_height_above_local_m)
            )
            if not np.any(rail_like):
                continue
            idx = assignment.index[rail_like]
            lat = assignment.lateral[rail_like]
            np.add.at(rail_like_counts[samples.connector_id], idx, 1)
            for side_index, rail_offset in enumerate(rail_offsets):
                residual = lat - rail_offset
                residual_index = bin_indices(residual, residual_edges)
                valid = residual_index >= 0
                if np.any(valid):
                    np.add.at(residual_hists[samples.connector_id], (idx[valid], side_index, residual_index[valid]), 1)

    analyses: list[dict[str, Any]] = []
    refined_features: list[dict[str, Any]] = []
    endpoint_locked_features: list[dict[str, Any]] = []
    predicted_rail_features: list[dict[str, Any]] = []
    observed_rail_features: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    sample_point_features: list[dict[str, Any]] = []

    for samples in sample_sets:
        analysis = analyze_connector(
            samples,
            residual_hist=residual_hists[samples.connector_id],
            residual_edges=residual_edges,
            corridor_count=corridor_counts[samples.connector_id],
            rail_like_count=rail_like_counts[samples.connector_id],
            gauge_m=gauge_m,
            min_side_points=args.min_side_points,
            min_correction_samples=args.min_correction_samples,
            max_correction_m=args.max_correction_m,
        )
        analyses.append(analysis["summary"])
        locked_correction = endpoint_locked_correction(
            samples.station,
            analysis["correction"],
            endpoint_lock_m=args.endpoint_lock_m,
            endpoint_taper_m=args.endpoint_taper_m,
        )
        locked_summary = {
            **analysis["summary"],
            "correction_source": "las_middle_endpoint_locked",
            "endpoint_lock_m": args.endpoint_lock_m,
            "endpoint_taper_m": args.endpoint_taper_m,
            "max_abs_smoothed_correction_m": round(float(np.max(np.abs(locked_correction))), 4) if locked_correction.size else 0.0,
        }
        refined_features.append(centerline_feature(samples, analysis["correction"], kind="las_refined_centerline", gauge_m=gauge_m, summary=analysis["summary"]))
        endpoint_locked_features.append(
            centerline_feature(
                samples,
                locked_correction,
                kind="las_refined_endpoint_locked_centerline",
                gauge_m=gauge_m,
                summary=locked_summary,
            )
        )
        predicted_rail_features.extend(rail_line_features(samples, correction=np.zeros(samples.station.size), gauge_m=gauge_m, source="predicted_from_current_centerline"))
        observed_rail_features.extend(rail_line_features(samples, correction=analysis["correction"], gauge_m=gauge_m, source="las_refined_from_observed_rail_peaks"))
        rows, point_features = sample_diagnostics(samples, analysis, gauge_m=gauge_m)
        sample_rows.extend(rows)
        sample_point_features.extend(point_features)

    refined_geojson = out_dir / "turnout_crossover_las_refined_centerlines.geojson"
    endpoint_locked_geojson = out_dir / "turnout_crossover_las_endpoint_locked_centerlines.geojson"
    predicted_rails_geojson = out_dir / "turnout_crossover_predicted_rails.geojson"
    observed_rails_geojson = out_dir / "turnout_crossover_las_observed_rails.geojson"
    sample_geojson = out_dir / "turnout_crossover_las_sample_diagnostics.geojson"
    btc.write_geojson(refined_geojson, refined_features, epsg=args.epsg)
    btc.write_geojson(endpoint_locked_geojson, endpoint_locked_features, epsg=args.epsg)
    btc.write_geojson(predicted_rails_geojson, predicted_rail_features, epsg=args.epsg)
    btc.write_geojson(observed_rails_geojson, observed_rail_features, epsg=args.epsg)
    write_point_geojson(sample_geojson, sample_point_features, epsg=args.epsg)
    write_line_shapefile(refined_features, refined_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_line_shapefile(endpoint_locked_features, endpoint_locked_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_line_shapefile(predicted_rail_features, predicted_rails_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_line_shapefile(observed_rail_features, observed_rails_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_point_shapefile(sample_point_features, sample_geojson.with_suffix(".shp"), epsg=args.epsg)
    sample_csv = out_dir / "turnout_crossover_las_sample_diagnostics.csv"
    write_csv(sample_csv, sample_rows)

    summary = {
        "mode": "centerline_to_predicted_rails_las_validation",
        "centerlines": str(centerline_path),
        "las_dir": str(args.las_dir.expanduser().resolve()),
        "target_epsg": args.target_epsg,
        "gauge_m": gauge_m,
        "chunk_size": args.chunk_size,
        "endpoint_lock_m": args.endpoint_lock_m,
        "endpoint_taper_m": args.endpoint_taper_m,
        "files": file_summaries,
        "connectors": analyses,
        "outputs": {
            "refined_centerlines_shp": str(refined_geojson.with_suffix(".shp")),
            "endpoint_locked_centerlines_shp": str(endpoint_locked_geojson.with_suffix(".shp")),
            "predicted_rails_shp": str(predicted_rails_geojson.with_suffix(".shp")),
            "observed_rails_shp": str(observed_rails_geojson.with_suffix(".shp")),
            "sample_diagnostics_shp": str(sample_geojson.with_suffix(".shp")),
            "sample_diagnostics_csv": str(sample_csv),
            "summary_json": str(out_dir / "summary.json"),
        },
        "interpretation": "Predicted rail lines are offsets from the reviewed centerline. LAS observed rail lines shift those offsets by per-sample rail-like point ridges; use the refined centerline only where support_samples are sufficient.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_visual_qa(out_dir / "VISUAL_QA.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def resolve_gauge(gauge_m: float, summary_path: Path) -> float:
    if gauge_m > 0:
        return float(gauge_m)
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        median = ((payload.get("estimated_gauge_m") or {}).get("median"))
        if median:
            return float(median)
    return 1.55


def build_samples(feature: dict[str, Any], *, gauge_m: float, sample_step_m: float) -> SampleSet:
    coords = btc.line_coords(feature)
    station, sampled = densify_line(coords, step_m=sample_step_m)
    tangent_x, tangent_y = sample_tangents(sampled)
    normal_x = -tangent_y
    normal_y = tangent_x
    chord_dx = sampled[-1][0] - sampled[0][0]
    chord_dy = sampled[-1][1] - sampled[0][1]
    chord_len = max(math.hypot(chord_dx, chord_dy), 1e-6)
    props = feature.get("properties") or {}
    pad = gauge_m / 2.0 + 1.5
    xs = [point[0] for point in sampled]
    ys = [point[1] for point in sampled]
    return SampleSet(
        connector_id=str(props.get("connector_id", props.get("conn_id", "connector"))),
        props=dict(props),
        station=np.asarray(station, dtype=float),
        center_x=np.asarray(xs, dtype=float),
        center_y=np.asarray(ys, dtype=float),
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        normal_x=normal_x,
        normal_y=normal_y,
        chord_x=chord_dx / chord_len,
        chord_y=chord_dy / chord_len,
        step_m=sample_step_m,
        bounds=(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad),
    )


def densify_line(coords: list[tuple[float, float]], *, step_m: float) -> tuple[list[float], list[tuple[float, float]]]:
    if len(coords) < 2:
        raise ValueError("Cannot densify a line with fewer than two coordinates.")
    distances = [0.0]
    for a, b in zip(coords, coords[1:]):
        distances.append(distances[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = distances[-1]
    count = max(2, int(math.ceil(total / max(step_m, 0.05))) + 1)
    target_s = [total * index / (count - 1) for index in range(count)]
    sampled: list[tuple[float, float]] = []
    segment_index = 0
    for s_value in target_s:
        while segment_index < len(distances) - 2 and distances[segment_index + 1] < s_value:
            segment_index += 1
        s0 = distances[segment_index]
        s1 = distances[segment_index + 1]
        a = coords[segment_index]
        b = coords[segment_index + 1]
        u = 0.0 if s1 <= s0 else (s_value - s0) / (s1 - s0)
        sampled.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
    return target_s, sampled


def sample_tangents(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    tx = np.zeros(len(points), dtype=float)
    ty = np.zeros(len(points), dtype=float)
    for index in range(len(points)):
        if index == 0:
            a, b = points[0], points[1]
        elif index == len(points) - 1:
            a, b = points[-2], points[-1]
        else:
            a, b = points[index - 1], points[index + 1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = max(math.hypot(dx, dy), 1e-6)
        tx[index] = dx / length
        ty[index] = dy / length
    return tx, ty


def inspect_las_headers(las_files: list[Path], *, target_epsg: int) -> tuple[float, float, list[dict[str, Any]]]:
    target_crs = CRS.from_epsg(target_epsg)
    z_min = math.inf
    z_max = -math.inf
    summaries: list[dict[str, Any]] = []
    for path in las_files:
        with laspy.open(path) as reader:
            header = reader.header
            src_crs = header.parse_crs()
            transformer = None if src_crs is None else Transformer.from_crs(src_crs, target_crs, always_xy=True)
            xs = np.asarray([header.mins[0], header.maxs[0], header.maxs[0], header.mins[0]], dtype=float)
            ys = np.asarray([header.mins[1], header.mins[1], header.maxs[1], header.maxs[1]], dtype=float)
            if transformer is not None:
                xs, ys = transformer.transform(xs, ys)
                xs = np.asarray(xs, dtype=float)
                ys = np.asarray(ys, dtype=float)
            z_min = min(z_min, float(header.mins[2]))
            z_max = max(z_max, float(header.maxs[2]))
            summaries.append(
                {
                    "path": str(path),
                    "point_count": int(header.point_count),
                    "source_epsg": src_crs.to_epsg() if src_crs is not None else None,
                    "transformed_bounds": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                }
            )
    if not math.isfinite(z_min) or not math.isfinite(z_max):
        raise ValueError("Could not inspect LAS z range.")
    return z_min, z_max, summaries


def iter_las_chunks(las_files: list[Path], *, target_epsg: int, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    target_crs = CRS.from_epsg(target_epsg)
    for path in las_files:
        with laspy.open(path) as reader:
            src_crs = reader.header.parse_crs()
            transformer = None if src_crs is None else Transformer.from_crs(src_crs, target_crs, always_xy=True)
            for points in reader.chunk_iterator(chunk_size):
                if len(points) == 0:
                    continue
                x = np.asarray(points.x, dtype=float)
                y = np.asarray(points.y, dtype=float)
                z = np.asarray(points.z, dtype=float)
                if transformer is not None:
                    x, y = transformer.transform(x, y)
                    x = np.asarray(x, dtype=float)
                    y = np.asarray(y, dtype=float)
                yield LasChunk(x=x, y=y, z=z)


def assign_chunk_to_samples(chunk: LasChunk, samples: SampleSet, *, args: argparse.Namespace) -> SampleAssignment:
    min_x, min_y, max_x, max_y = samples.bounds
    mask = (chunk.x >= min_x) & (chunk.x <= max_x) & (chunk.y >= min_y) & (chunk.y <= max_y)
    if not np.any(mask):
        empty = np.asarray([], dtype=float)
        return SampleAssignment(index=np.asarray([], dtype=np.int64), along=empty, lateral=empty, z=empty)
    x = chunk.x[mask]
    y = chunk.y[mask]
    z = chunk.z[mask]
    approx = np.rint(((x - samples.center_x[0]) * samples.chord_x + (y - samples.center_y[0]) * samples.chord_y) / max(samples.step_m, 1e-6)).astype(np.int64)
    best_index = np.clip(approx, 0, samples.station.size - 1)
    best_d2 = np.full(x.shape, np.inf, dtype=float)
    for delta in range(-args.assign_neighbors, args.assign_neighbors + 1):
        candidate = np.clip(approx + delta, 0, samples.station.size - 1)
        dx = x - samples.center_x[candidate]
        dy = y - samples.center_y[candidate]
        d2 = dx * dx + dy * dy
        better = d2 < best_d2
        best_index[better] = candidate[better]
        best_d2[better] = d2[better]
    dx = x - samples.center_x[best_index]
    dy = y - samples.center_y[best_index]
    along = dx * samples.tangent_x[best_index] + dy * samples.tangent_y[best_index]
    lateral = dx * samples.normal_x[best_index] + dy * samples.normal_y[best_index]
    valid = (np.abs(along) <= args.along_half_window_m) & (np.abs(lateral) <= args.corridor_m)
    return SampleAssignment(index=best_index[valid], along=along[valid], lateral=lateral[valid], z=z[valid])


def make_edges(low: float, high: float, bin_size: float, *, pad: float) -> np.ndarray:
    left = math.floor((low - pad) / bin_size) * bin_size
    right = math.ceil((high + pad) / bin_size) * bin_size
    if right <= left:
        right = left + bin_size
    return np.arange(left, right + bin_size, bin_size, dtype=float)


def bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges, values, side="right") - 1
    invalid = (indices < 0) | (indices >= len(edges) - 1) | ~np.isfinite(values)
    indices = indices.astype(np.int64, copy=False)
    indices[invalid] = -1
    return indices


def percentile_by_histogram(hist: np.ndarray, edges: np.ndarray, percentile: float) -> np.ndarray:
    centers = (edges[:-1] + edges[1:]) / 2.0
    result = np.full(hist.shape[0], np.nan, dtype=float)
    for row_index, counts in enumerate(hist):
        total = int(counts.sum())
        if total <= 0:
            continue
        cumulative = np.cumsum(counts)
        bin_index = int(np.searchsorted(cumulative, total * percentile / 100.0, side="left"))
        bin_index = max(0, min(bin_index, centers.size - 1))
        result[row_index] = centers[bin_index]
    return result


def analyze_connector(
    samples: SampleSet,
    *,
    residual_hist: np.ndarray,
    residual_edges: np.ndarray,
    corridor_count: np.ndarray,
    rail_like_count: np.ndarray,
    gauge_m: float,
    min_side_points: int,
    min_correction_samples: int,
    max_correction_m: float,
) -> dict[str, Any]:
    centers = (residual_edges[:-1] + residual_edges[1:]) / 2.0
    left_residual = np.full(samples.station.size, np.nan, dtype=float)
    right_residual = np.full(samples.station.size, np.nan, dtype=float)
    left_count = residual_hist[:, 0, :].sum(axis=1)
    right_count = residual_hist[:, 1, :].sum(axis=1)
    for sample_index in range(samples.station.size):
        for side_index, target in ((0, left_residual), (1, right_residual)):
            counts = residual_hist[sample_index, side_index]
            if int(counts.sum()) < min_side_points:
                continue
            smooth = moving_average(counts.astype(float), window=5)
            peak_index = stable_peak_index(counts, smooth)
            target[sample_index] = centers[peak_index]
    both = np.isfinite(left_residual) & np.isfinite(right_residual)
    one = np.isfinite(left_residual) ^ np.isfinite(right_residual)
    raw_correction = np.full(samples.station.size, np.nan, dtype=float)
    raw_correction[both] = (left_residual[both] + right_residual[both]) / 2.0
    raw_correction[one & np.isfinite(left_residual)] = left_residual[one & np.isfinite(left_residual)]
    raw_correction[one & np.isfinite(right_residual)] = right_residual[one & np.isfinite(right_residual)]
    valid = np.isfinite(raw_correction)
    correction = np.zeros(samples.station.size, dtype=float)
    correction_source = "insufficient_las_support_no_shift"
    if int(np.count_nonzero(valid)) >= min_correction_samples:
        correction = np.interp(samples.station, samples.station[valid], raw_correction[valid])
        correction = moving_average(correction, window=7)
        correction = np.clip(correction, -max_correction_m, max_correction_m)
        correction_source = "las_observed_rail_peak_shift"
    valid_values = raw_correction[valid]
    summary = {
        "connector_id": samples.connector_id,
        "sample_count": int(samples.station.size),
        "corridor_point_count": int(corridor_count.sum()),
        "rail_like_point_count": int(rail_like_count.sum()),
        "support_samples_any_rail": int(np.count_nonzero(valid)),
        "support_samples_both_rails": int(np.count_nonzero(both)),
        "support_fraction_any_rail": round(float(np.count_nonzero(valid) / max(samples.station.size, 1)), 4),
        "support_fraction_both_rails": round(float(np.count_nonzero(both) / max(samples.station.size, 1)), 4),
        "median_raw_correction_m": round(float(np.nanmedian(valid_values)), 4) if valid_values.size else None,
        "p10_raw_correction_m": round(float(np.nanpercentile(valid_values, 10)), 4) if valid_values.size else None,
        "p90_raw_correction_m": round(float(np.nanpercentile(valid_values, 90)), 4) if valid_values.size else None,
        "max_abs_smoothed_correction_m": round(float(np.max(np.abs(correction))), 4) if correction.size else 0.0,
        "correction_source": correction_source,
        "gauge_m": gauge_m,
    }
    return {
        "summary": summary,
        "correction": correction,
        "raw_correction": raw_correction,
        "left_residual": left_residual,
        "right_residual": right_residual,
        "left_count": left_count,
        "right_count": right_count,
        "corridor_count": corridor_count,
        "rail_like_count": rail_like_count,
    }


def stable_peak_index(counts: np.ndarray, smooth: np.ndarray) -> int:
    if counts.size == 0:
        return 0
    max_smooth = float(np.max(smooth)) if smooth.size else 0.0
    candidates = np.flatnonzero(np.isclose(smooth, max_smooth))
    if candidates.size == 0:
        return int(np.argmax(counts))
    candidate_counts = counts[candidates]
    best_count = np.max(candidate_counts)
    best = candidates[candidate_counts == best_count]
    return int(best[len(best) // 2])


def endpoint_locked_correction(
    station: np.ndarray,
    correction: np.ndarray,
    *,
    endpoint_lock_m: float,
    endpoint_taper_m: float,
) -> np.ndarray:
    if station.size == 0:
        return correction.copy()
    total = float(station[-1] - station[0])
    distance_from_endpoint = np.minimum(station - station[0], station[-1] - station)
    if endpoint_taper_m <= 0:
        weights = (distance_from_endpoint > endpoint_lock_m).astype(float)
    else:
        u = np.clip((distance_from_endpoint - endpoint_lock_m) / endpoint_taper_m, 0.0, 1.0)
        weights = smoothstep(u)
    if total <= endpoint_lock_m * 2.0:
        weights[:] = 0.0
    return correction * weights


def smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def moving_average(values: np.ndarray, *, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.copy()
    window = min(window, values.size)
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def centerline_feature(samples: SampleSet, correction: np.ndarray, *, kind: str, gauge_m: float, summary: dict[str, Any]) -> dict[str, Any]:
    coords = [
        [
            round(float(x + nx * corr), 6),
            round(float(y + ny * corr), 6),
        ]
        for x, y, nx, ny, corr in zip(samples.center_x, samples.center_y, samples.normal_x, samples.normal_y, correction)
    ]
    props = {
        **samples.props,
        "geom_kind": kind,
        "source": "handheld_las_rail_fit",
        "review_status": "candidate_needs_qgis_review",
        "gauge_m": round(gauge_m, 4),
        "las_src": summary.get("correction_source", ""),
        "las_any": summary.get("support_samples_any_rail", 0),
        "las_both": summary.get("support_samples_both_rails", 0),
        "med_shift": summary.get("median_raw_correction_m"),
        "max_shift": summary.get("max_abs_smoothed_correction_m"),
        "end_lock_m": summary.get("endpoint_lock_m"),
        "end_taper": summary.get("endpoint_taper_m"),
    }
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def rail_line_features(samples: SampleSet, *, correction: np.ndarray, gauge_m: float, source: str) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for rail_side, rail_offset in (("left", -gauge_m / 2.0), ("right", gauge_m / 2.0)):
        coords = []
        for x, y, nx, ny, corr in zip(samples.center_x, samples.center_y, samples.normal_x, samples.normal_y, correction):
            offset = corr + rail_offset
            coords.append([round(float(x + nx * offset), 6), round(float(y + ny * offset), 6)])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "connector_id": samples.connector_id,
                    "geom_kind": "rail_line",
                    "rail_side": rail_side,
                    "source": source,
                    "gauge_m": round(gauge_m, 4),
                },
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )
    return features


def sample_diagnostics(samples: SampleSet, analysis: dict[str, Any], *, gauge_m: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for index, station in enumerate(samples.station):
        raw_corr = analysis["raw_correction"][index]
        corr = analysis["correction"][index]
        left_res = analysis["left_residual"][index]
        right_res = analysis["right_residual"][index]
        props = {
            "connector_id": samples.connector_id,
            "sample_i": int(index),
            "station_m": round(float(station), 3),
            "corr_m": round(float(corr), 4),
            "raw_corr_m": round(float(raw_corr), 4) if np.isfinite(raw_corr) else None,
            "left_res_m": round(float(left_res), 4) if np.isfinite(left_res) else None,
            "right_res_m": round(float(right_res), 4) if np.isfinite(right_res) else None,
            "left_count": int(analysis["left_count"][index]),
            "right_count": int(analysis["right_count"][index]),
            "rail_like": int(analysis["rail_like_count"][index]),
            "corridor": int(analysis["corridor_count"][index]),
            "gauge_m": round(gauge_m, 4),
        }
        rows.append(props)
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [round(float(samples.center_x[index]), 6), round(float(samples.center_y[index]), 6)]},
            }
        )
    return rows, features


def write_line_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("conn_id", "C", size=16)
    writer.field("kind", "C", size=40)
    writer.field("source", "C", size=48)
    writer.field("side", "C", size=12)
    writer.field("status", "C", size=32)
    writer.field("gauge_m", "F", decimal=4)
    writer.field("las_any", "N", size=8)
    writer.field("las_both", "N", size=8)
    writer.field("med_shift", "F", decimal=4)
    writer.field("max_shift", "F", decimal=4)
    writer.field("end_lock", "F", decimal=2)
    writer.field("end_taper", "F", decimal=2)
    for feature in features:
        props = feature.get("properties") or {}
        writer.line([btc.line_coords(feature)])
        writer.record(
            str(props.get("connector_id", props.get("conn_id", "")))[:16],
            str(props.get("geom_kind", ""))[:40],
            str(props.get("source", ""))[:48],
            str(props.get("rail_side", ""))[:12],
            str(props.get("review_status", ""))[:32],
            btc.safe_float(props.get("gauge_m", 0.0)),
            int(btc.safe_float(props.get("las_any", 0))),
            int(btc.safe_float(props.get("las_both", 0))),
            btc.safe_float(props.get("med_shift", 0.0)),
            btc.safe_float(props.get("max_shift", 0.0)),
            btc.safe_float(props.get("end_lock_m", 0.0)),
            btc.safe_float(props.get("end_taper", 0.0)),
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    btc.write_projection(output_path.with_suffix(".prj"), epsg)


def write_point_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}}, "features": features}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_point_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POINT, encoding="utf-8")
    writer.field("conn_id", "C", size=16)
    writer.field("sample_i", "N", size=8)
    writer.field("station_m", "F", decimal=3)
    writer.field("corr_m", "F", decimal=4)
    writer.field("raw_corr", "F", decimal=4)
    writer.field("left_res", "F", decimal=4)
    writer.field("right_res", "F", decimal=4)
    writer.field("left_cnt", "N", size=8)
    writer.field("right_cnt", "N", size=8)
    writer.field("rail_like", "N", size=8)
    writer.field("corridor", "N", size=8)
    for feature in features:
        props = feature.get("properties") or {}
        x, y = feature["geometry"]["coordinates"][:2]
        writer.point(float(x), float(y))
        writer.record(
            str(props.get("connector_id", ""))[:16],
            int(props.get("sample_i", 0)),
            btc.safe_float(props.get("station_m", 0.0)),
            btc.safe_float(props.get("corr_m", 0.0)),
            btc.safe_float(props.get("raw_corr_m", 0.0)),
            btc.safe_float(props.get("left_res_m", 0.0)),
            btc.safe_float(props.get("right_res_m", 0.0)),
            int(props.get("left_count", 0)),
            int(props.get("right_count", 0)),
            int(props.get("rail_like", 0)),
            int(props.get("corridor", 0)),
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    btc.write_projection(output_path.with_suffix(".prj"), epsg)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_visual_qa(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 手扫 LAS 渡线钢轨反推 QA",
        "",
        "本层用于验证当前成对渡线中心线：先按轨距把中心线反推出两根钢轨坐标，再在手扫 LAS 中寻找预测钢轨附近的轨头高度点云脊线。",
        "",
        "## 输出",
        "",
        f"- `{Path(summary['outputs']['endpoint_locked_centerlines_shp']).name}`：推荐优先验收的中心线，端点保持原线，中段使用 LAS 修正。",
        f"- `{Path(summary['outputs']['refined_centerlines_shp']).name}`：按 LAS 钢轨脊线平移后的候选中心线。",
        f"- `{Path(summary['outputs']['predicted_rails_shp']).name}`：由当前中心线和轨距直接反推的两根钢轨线。",
        f"- `{Path(summary['outputs']['observed_rails_shp']).name}`：按 LAS 峰值修正后的两根钢轨线。",
        f"- `{Path(summary['outputs']['sample_diagnostics_shp']).name}`：逐采样点的修正量和支撑点数量。",
        "",
        "## 渡线摘要",
        "",
        "| connector | 有任一钢轨支撑采样点 | 双钢轨支撑采样点 | 原始中位平移 m | 平滑最大平移 m | 来源 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.get("connectors", []):
        lines.append(
            f"| `{row['connector_id']}` | {row['support_samples_any_rail']} | {row['support_samples_both_rails']} | "
            f"{row.get('median_raw_correction_m')} | {row.get('max_abs_smoothed_correction_m')} | {row.get('correction_source')} |"
        )
    lines.extend(
        [
            "",
            "## 注意",
            "",
            "- 这仍是候选 QA 层，不是最终验收中心线。",
            f"- `endpoint_locked` 版本锁定两端 `{summary.get('endpoint_lock_m')} m`，再用 `{summary.get('endpoint_taper_m')} m` 平滑过渡到 LAS 修正，避免转辙机/护轨区域污染端点。",
            "- 正平移表示 LAS 推断出的钢轨对中心在当前中心线的局部法线正方向一侧。",
            "- QGIS 里应同时打开 `predicted_rails`、`las_observed_rails` 和 `sample_diagnostics`，检查局部修正是否由双侧钢轨共同支撑。",
            "- 道岔区域存在护轨、转辙机和交叉结构，LAS 轨头高度筛选可能被局部构件污染；不能只看分数自动替换。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
