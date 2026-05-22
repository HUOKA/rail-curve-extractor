#!/usr/bin/env python3
"""Estimate rail geometry constraints from handheld LAS point clouds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import laspy
import numpy as np
from PIL import Image, ImageDraw
from pyproj import CRS, Transformer


DEFAULT_LAS_DIR = Path("data/生产数据/轨道/Las")
DEFAULT_CENTERLINE = Path("output/tonghaigang_topology_rebuild_v1/topology_skeleton_linework.geojson")
DEFAULT_OUT_DIR = Path("output/handheld_las_constraints")
DEFAULT_FULLPASS_OUT_DIR = Path("output/handheld_las_constraints_fullpass")


class Axis(NamedTuple):
    origin: np.ndarray
    longitudinal: np.ndarray
    lateral: np.ndarray


class TrackLine(NamedTuple):
    line_id: str
    role: str
    s: np.ndarray
    t: np.ndarray
    length_m: float


class ExclusionWindow(NamedTuple):
    source_id: str
    s_min: float
    s_max: float
    t_min: float
    t_max: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample handheld LAS files and estimate rail gauge / track-spacing constraints.",
    )
    parser.add_argument("--las-dir", type=Path, default=DEFAULT_LAS_DIR)
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument(
        "--exclude-workzones",
        type=Path,
        default=None,
        help="Optional GeoJSON polygons to exclude from gauge calibration, typically switch workzones.",
    )
    parser.add_argument(
        "--exclude-workzone-buffer-m",
        type=float,
        default=2.0,
        help="Extra s/t bbox margin around excluded workzone polygons.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-epsg", type=int, default=32651)
    parser.add_argument("--full-pass", action="store_true", help="Use every LAS point via chunked streaming.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Points to read per LAS chunk in full-pass mode.")
    parser.add_argument("--z-bin-size-m", type=float, default=0.02, help="Z histogram bin size for full-pass ground estimation.")
    parser.add_argument("--ground-percentile", type=float, default=20.0, help="Local Z percentile used as ground in each s bin.")
    parser.add_argument("--max-points-per-file", type=int, default=250_000)
    parser.add_argument("--windows-per-file", type=int, default=24)
    parser.add_argument("--window-read-size", type=int, default=80_000)
    parser.add_argument("--s-bin-size-m", type=float, default=2.0)
    parser.add_argument("--t-bin-size-m", type=float, default=0.05)
    parser.add_argument("--offset-bin-size-m", type=float, default=0.025)
    parser.add_argument("--min-height-above-local-m", type=float, default=0.08)
    parser.add_argument("--max-height-above-local-m", type=float, default=0.35)
    parser.add_argument("--near-track-window-m", type=float, default=2.4)
    parser.add_argument("--gauge-min-m", type=float, default=1.25)
    parser.add_argument("--gauge-max-m", type=float, default=1.75)
    parser.add_argument("--spacing-min-m", type=float, default=3.5)
    parser.add_argument("--spacing-max-m", type=float, default=7.0)
    parser.add_argument("--max-track-lines", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir_arg = DEFAULT_FULLPASS_OUT_DIR if args.full_pass and args.out_dir == DEFAULT_OUT_DIR else args.out_dir
    out_dir = out_dir_arg.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    centerline_path = args.centerline.expanduser().resolve()
    axis, track_lines = load_axis_and_tracks(centerline_path)
    exclusion_windows = load_exclusion_windows(args.exclude_workzones, axis, buffer_m=args.exclude_workzone_buffer_m)
    las_files = sorted(args.las_dir.expanduser().resolve().glob("*.las")) + sorted(args.las_dir.expanduser().resolve().glob("*.laz"))
    if not las_files:
        raise FileNotFoundError(f"No LAS/LAZ files found under: {args.las_dir}")

    target_crs = CRS.from_epsg(args.target_epsg)
    if args.full_pass:
        summary = run_full_pass_analysis(
            args=args,
            las_files=las_files,
            target_crs=target_crs,
            axis=axis,
            track_lines=track_lines,
            exclusion_windows=exclusion_windows,
            centerline_path=centerline_path,
            out_dir=out_dir,
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    sampled, file_summaries = sample_las_points(
        las_files=las_files,
        target_crs=target_crs,
        max_points_per_file=args.max_points_per_file,
        windows_per_file=args.windows_per_file,
        window_read_size=args.window_read_size,
    )
    if sampled["x"].size < 100:
        raise ValueError("Too few sampled LAS points for constraint analysis.")

    s_values, t_values = project_points(axis, sampled["x"], sampled["y"])
    z_values = sampled["z"]
    z_local = local_ground_percentile(
        s_values=s_values,
        z_values=z_values,
        s_bin_size=args.s_bin_size_m,
        percentile=20.0,
    )
    z_above = z_values - z_local
    rail_like_mask = (
        np.isfinite(z_above)
        & (z_above >= args.min_height_above_local_m)
        & (z_above <= args.max_height_above_local_m)
    )
    excluded_mask = rail_like_mask & points_in_exclusion_windows(s_values, t_values, exclusion_windows)
    calibration_mask = rail_like_mask & ~excluded_mask

    lateral_profile = build_histogram_profile(
        values=t_values[calibration_mask],
        bin_size=args.t_bin_size_m,
        value_range=percentile_range(t_values[calibration_mask], low=0.5, high=99.5, pad=1.0),
    )
    lateral_peaks = find_profile_peaks(
        lateral_profile,
        min_distance_m=0.55,
        min_prominence_fraction=0.08,
    )
    lateral_pairs = pair_peaks_by_distance(
        lateral_peaks,
        min_distance=args.gauge_min_m,
        max_distance=args.gauge_max_m,
        distance_name="gauge_m",
    )
    center_spacing_candidates = spacing_from_gauge_pairs(
        lateral_pairs,
        min_spacing=args.spacing_min_m,
        max_spacing=args.spacing_max_m,
    )

    track_profiles: list[dict[str, Any]] = []
    gauge_candidates: list[dict[str, Any]] = []
    selected_tracks = track_lines[: args.max_track_lines]
    for track in selected_tracks:
        t_line = interpolate_track_t(track, s_values)
        valid = np.isfinite(t_line) & calibration_mask
        offsets = t_values[valid] - t_line[valid]
        offsets = offsets[np.abs(offsets) <= args.near_track_window_m]
        if offsets.size < 100:
            continue
        profile = build_histogram_profile(
            values=offsets,
            bin_size=args.offset_bin_size_m,
            value_range=(-args.near_track_window_m, args.near_track_window_m),
        )
        peaks = find_profile_peaks(
            profile,
            min_distance_m=0.45,
            min_prominence_fraction=0.07,
        )
        pairs = pair_peaks_by_distance(
            peaks,
            min_distance=args.gauge_min_m,
            max_distance=args.gauge_max_m,
            distance_name="gauge_m",
        )
        track_profiles.extend(profile_rows(profile, track.line_id, track.role))
        for pair in pairs:
            gauge_candidates.append(
                {
                    "line_id": track.line_id,
                    "topology_role": track.role,
                    "left_peak_offset_m": round(pair["left_value_m"], 4),
                    "right_peak_offset_m": round(pair["right_value_m"], 4),
                    "gauge_m": round(pair["gauge_m"], 4),
                    "center_offset_m": round((pair["left_value_m"] + pair["right_value_m"]) / 2.0, 4),
                    "score": round(pair["score"], 3),
                    "left_count": int(pair["left_count"]),
                    "right_count": int(pair["right_count"]),
                },
            )

    output_paths = write_outputs(
        out_dir=out_dir,
        lateral_profile=lateral_profile,
        lateral_peaks=lateral_peaks,
        lateral_pairs=lateral_pairs,
        center_spacing_candidates=center_spacing_candidates,
        track_profiles=track_profiles,
        gauge_candidates=gauge_candidates,
    )
    write_profile_png(
        out_dir / "lateral_profile.png",
        title="handheld LAS rail-like lateral profile",
        profile=lateral_profile,
        peaks=lateral_peaks,
        pairs=lateral_pairs,
    )

    gauge_values = [float(row["gauge_m"]) for row in gauge_candidates]
    spacing_values = [float(row["spacing_m"]) for row in center_spacing_candidates]
    summary = {
        "las_dir": str(args.las_dir.expanduser().resolve()),
        "centerline": str(centerline_path),
        "target_epsg": args.target_epsg,
        "file_count": len(las_files),
        "files": file_summaries,
        "sampled_point_count": int(sampled["x"].size),
        "rail_like_point_count": int(np.count_nonzero(rail_like_mask)),
        "rail_like_fraction": float(np.count_nonzero(rail_like_mask) / max(sampled["x"].size, 1)),
        "calibration_point_count": int(np.count_nonzero(calibration_mask)),
        "workzone_excluded_rail_like_point_count": int(np.count_nonzero(excluded_mask)),
        "exclusion_workzones": exclusion_summary(args, exclusion_windows),
        "height_above_local_range_m": [
            args.min_height_above_local_m,
            args.max_height_above_local_m,
        ],
        "axis_origin": [round(float(axis.origin[0]), 6), round(float(axis.origin[1]), 6)],
        "axis_longitudinal": [round(float(axis.longitudinal[0]), 8), round(float(axis.longitudinal[1]), 8)],
        "axis_lateral": [round(float(axis.lateral[0]), 8), round(float(axis.lateral[1]), 8)],
        "lateral_peak_count": len(lateral_peaks),
        "global_gauge_pair_count": len(lateral_pairs),
        "track_gauge_candidate_count": len(gauge_candidates),
        "estimated_gauge_m": robust_value_summary(gauge_values),
        "center_spacing_candidate_count": len(center_spacing_candidates),
        "estimated_center_spacing_m": robust_value_summary(spacing_values),
        "outputs": output_paths,
        "interpretation": (
            "Use estimated_gauge_m to constrain which two semantic rail candidates may form one track. "
            "Use estimated_center_spacing_m only as a local prior because handheld LAS coverage is partial."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def run_full_pass_analysis(
    *,
    args: argparse.Namespace,
    las_files: list[Path],
    target_crs: CRS,
    axis: Axis,
    track_lines: list[TrackLine],
    exclusion_windows: list[ExclusionWindow],
    centerline_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    ranges, file_summaries = inspect_las_ranges(las_files=las_files, target_crs=target_crs, axis=axis)
    s_edges = make_edges(ranges["s_min"], ranges["s_max"], args.s_bin_size_m, pad=args.s_bin_size_m)
    t_edges = make_edges(ranges["t_min"], ranges["t_max"], args.t_bin_size_m, pad=1.0)
    z_edges = make_edges(ranges["z_min"], ranges["z_max"], args.z_bin_size_m, pad=args.z_bin_size_m)
    offset_edges = make_edges(-args.near_track_window_m, args.near_track_window_m, args.offset_bin_size_m, pad=0.0)

    ground_hist = np.zeros((len(s_edges) - 1, len(z_edges) - 1), dtype=np.uint64)
    processed_first_pass = 0
    for chunk in iter_projected_las_chunks(las_files, target_crs, axis, args.chunk_size):
        processed_first_pass += int(chunk["s"].size)
        s_bin = bin_indices(chunk["s"], s_edges)
        z_bin = bin_indices(chunk["z"], z_edges)
        valid = (s_bin >= 0) & (z_bin >= 0)
        if np.any(valid):
            np.add.at(ground_hist, (s_bin[valid], z_bin[valid]), 1)

    ground_z = percentile_by_histogram(ground_hist, z_edges, args.ground_percentile)
    fallback_ground = global_percentile_from_histogram(ground_hist, z_edges, args.ground_percentile)
    ground_z[~np.isfinite(ground_z)] = fallback_ground

    lateral_counts = np.zeros(len(t_edges) - 1, dtype=np.uint64)
    selected_tracks = track_lines[: args.max_track_lines]
    track_offset_counts: dict[str, np.ndarray] = {
        track.line_id: np.zeros(len(offset_edges) - 1, dtype=np.uint64)
        for track in selected_tracks
    }
    track_roles = {track.line_id: track.role for track in selected_tracks}
    processed_second_pass = 0
    rail_like_count = 0
    calibration_point_count = 0
    excluded_rail_like_count = 0
    for chunk in iter_projected_las_chunks(las_files, target_crs, axis, args.chunk_size):
        processed_second_pass += int(chunk["s"].size)
        s_bin = bin_indices(chunk["s"], s_edges)
        valid_s = s_bin >= 0
        z_local = np.full(chunk["z"].shape, np.nan, dtype=float)
        z_local[valid_s] = ground_z[s_bin[valid_s]]
        z_above = chunk["z"] - z_local
        rail_like = (
            np.isfinite(z_above)
            & (z_above >= args.min_height_above_local_m)
            & (z_above <= args.max_height_above_local_m)
        )
        rail_like_count += int(np.count_nonzero(rail_like))
        excluded = rail_like & points_in_exclusion_windows(chunk["s"], chunk["t"], exclusion_windows)
        calibration = rail_like & ~excluded
        excluded_rail_like_count += int(np.count_nonzero(excluded))
        calibration_point_count += int(np.count_nonzero(calibration))

        t_bin = bin_indices(chunk["t"][calibration], t_edges)
        valid_t = t_bin >= 0
        if np.any(valid_t):
            np.add.at(lateral_counts, t_bin[valid_t], 1)

        s_rail = chunk["s"][calibration]
        t_rail = chunk["t"][calibration]
        for track in selected_tracks:
            t_line = interpolate_track_t(track, s_rail)
            valid_track = np.isfinite(t_line)
            if not np.any(valid_track):
                continue
            offsets = t_rail[valid_track] - t_line[valid_track]
            offset_bin = bin_indices(offsets, offset_edges)
            valid_offset = offset_bin >= 0
            if np.any(valid_offset):
                np.add.at(track_offset_counts[track.line_id], offset_bin[valid_offset], 1)

    lateral_profile = profile_from_counts(t_edges, lateral_counts)
    lateral_peaks = find_profile_peaks(
        lateral_profile,
        min_distance_m=0.55,
        min_prominence_fraction=0.08,
    )
    lateral_pairs = pair_peaks_by_distance(
        lateral_peaks,
        min_distance=args.gauge_min_m,
        max_distance=args.gauge_max_m,
        distance_name="gauge_m",
    )
    center_spacing_candidates = spacing_from_gauge_pairs(
        lateral_pairs,
        min_spacing=args.spacing_min_m,
        max_spacing=args.spacing_max_m,
    )

    track_profiles: list[dict[str, Any]] = []
    gauge_candidates: list[dict[str, Any]] = []
    for track_id, counts in track_offset_counts.items():
        profile = profile_from_counts(offset_edges, counts)
        peaks = find_profile_peaks(
            profile,
            min_distance_m=0.45,
            min_prominence_fraction=0.07,
        )
        pairs = pair_peaks_by_distance(
            peaks,
            min_distance=args.gauge_min_m,
            max_distance=args.gauge_max_m,
            distance_name="gauge_m",
        )
        track_profiles.extend(profile_rows(profile, track_id, track_roles.get(track_id, "")))
        for pair in pairs:
            gauge_candidates.append(
                {
                    "line_id": track_id,
                    "topology_role": track_roles.get(track_id, ""),
                    "left_peak_offset_m": round(pair["left_value_m"], 4),
                    "right_peak_offset_m": round(pair["right_value_m"], 4),
                    "gauge_m": round(pair["gauge_m"], 4),
                    "center_offset_m": round((pair["left_value_m"] + pair["right_value_m"]) / 2.0, 4),
                    "score": round(pair["score"], 3),
                    "left_count": int(pair["left_count"]),
                    "right_count": int(pair["right_count"]),
                },
            )

    output_paths = write_outputs(
        out_dir=out_dir,
        lateral_profile=lateral_profile,
        lateral_peaks=lateral_peaks,
        lateral_pairs=lateral_pairs,
        center_spacing_candidates=center_spacing_candidates,
        track_profiles=track_profiles,
        gauge_candidates=gauge_candidates,
    )
    write_profile_png(
        out_dir / "lateral_profile.png",
        title="handheld LAS full-pass rail-like lateral profile",
        profile=lateral_profile,
        peaks=lateral_peaks,
        pairs=lateral_pairs,
    )

    gauge_values = [float(row["gauge_m"]) for row in gauge_candidates]
    spacing_values = [float(row["spacing_m"]) for row in center_spacing_candidates]
    return {
        "mode": "full_pass_streaming",
        "las_dir": str(args.las_dir.expanduser().resolve()),
        "centerline": str(centerline_path),
        "target_epsg": args.target_epsg,
        "file_count": len(las_files),
        "files": file_summaries,
        "chunk_size": args.chunk_size,
        "processed_point_count": int(processed_second_pass),
        "first_pass_point_count": int(processed_first_pass),
        "rail_like_point_count": int(rail_like_count),
        "rail_like_fraction": float(rail_like_count / max(processed_second_pass, 1)),
        "calibration_point_count": int(calibration_point_count),
        "workzone_excluded_rail_like_point_count": int(excluded_rail_like_count),
        "exclusion_workzones": exclusion_summary(args, exclusion_windows),
        "height_above_local_range_m": [
            args.min_height_above_local_m,
            args.max_height_above_local_m,
        ],
        "ground_percentile": args.ground_percentile,
        "s_bin_size_m": args.s_bin_size_m,
        "z_bin_size_m": args.z_bin_size_m,
        "axis_origin": [round(float(axis.origin[0]), 6), round(float(axis.origin[1]), 6)],
        "axis_longitudinal": [round(float(axis.longitudinal[0]), 8), round(float(axis.longitudinal[1]), 8)],
        "axis_lateral": [round(float(axis.lateral[0]), 8), round(float(axis.lateral[1]), 8)],
        "analysis_ranges": {key: round(float(value), 6) for key, value in ranges.items()},
        "lateral_peak_count": len(lateral_peaks),
        "global_gauge_pair_count": len(lateral_pairs),
        "track_gauge_candidate_count": len(gauge_candidates),
        "estimated_gauge_m": robust_value_summary(gauge_values),
        "center_spacing_candidate_count": len(center_spacing_candidates),
        "estimated_center_spacing_m": robust_value_summary(spacing_values),
        "outputs": output_paths,
        "interpretation": (
            "All LAS points participated through chunked two-pass streaming. "
            "Use estimated_gauge_m as a map-space rail-pairing constraint; keep spacing local because coverage is partial."
        ),
    }


def inspect_las_ranges(las_files: list[Path], target_crs: CRS, axis: Axis) -> tuple[dict[str, float], list[dict[str, Any]]]:
    s_min = math.inf
    s_max = -math.inf
    t_min = math.inf
    t_max = -math.inf
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
            s_values, t_values = project_points(axis, xs, ys)
            s_min = min(s_min, float(np.min(s_values)))
            s_max = max(s_max, float(np.max(s_values)))
            t_min = min(t_min, float(np.min(t_values)))
            t_max = max(t_max, float(np.max(t_values)))
            z_min = min(z_min, float(header.mins[2]))
            z_max = max(z_max, float(header.maxs[2]))
            summaries.append(
                {
                    "path": str(path),
                    "point_count": int(header.point_count),
                    "source_crs": src_crs.to_string() if src_crs is not None else None,
                    "source_epsg": src_crs.to_epsg() if src_crs is not None else None,
                    "source_bounds": [
                        float(header.mins[0]),
                        float(header.mins[1]),
                        float(header.maxs[0]),
                        float(header.maxs[1]),
                    ],
                    "transformed_bounds": [
                        float(np.min(xs)),
                        float(np.min(ys)),
                        float(np.max(xs)),
                        float(np.max(ys)),
                    ],
                },
            )
    if not np.isfinite([s_min, s_max, t_min, t_max, z_min, z_max]).all():
        raise ValueError("Could not derive LAS analysis ranges from headers.")
    return {
        "s_min": s_min,
        "s_max": s_max,
        "t_min": t_min,
        "t_max": t_max,
        "z_min": z_min,
        "z_max": z_max,
    }, summaries


def iter_projected_las_chunks(
    las_files: list[Path],
    target_crs: CRS,
    axis: Axis,
    chunk_size: int,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
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
                s, t = project_points(axis, x, y)
                yield {"s": s, "t": t, "z": z}


def make_edges(low: float, high: float, bin_size: float, *, pad: float) -> np.ndarray:
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")
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
    result = np.full(hist.shape[0], np.nan, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    for row_index in range(hist.shape[0]):
        counts = hist[row_index]
        total = int(counts.sum())
        if total <= 0:
            continue
        target = total * percentile / 100.0
        cumulative = np.cumsum(counts)
        bin_index = int(np.searchsorted(cumulative, target, side="left"))
        bin_index = max(0, min(bin_index, centers.size - 1))
        result[row_index] = centers[bin_index]
    return result


def global_percentile_from_histogram(hist: np.ndarray, edges: np.ndarray, percentile: float) -> float:
    counts = hist.sum(axis=0)
    total = int(counts.sum())
    if total <= 0:
        return float((edges[0] + edges[-1]) / 2.0)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cumulative = np.cumsum(counts)
    bin_index = int(np.searchsorted(cumulative, total * percentile / 100.0, side="left"))
    bin_index = max(0, min(bin_index, centers.size - 1))
    return float(centers[bin_index])


def profile_from_counts(edges: np.ndarray, counts: np.ndarray) -> list[dict[str, float]]:
    centers = (edges[:-1] + edges[1:]) / 2.0
    smooth = moving_average(counts.astype(float), window=7)
    return [
        {"value_m": float(center), "count": int(count), "smooth_count": float(score)}
        for center, count, score in zip(centers, counts, smooth)
    ]


def sample_las_points(
    *,
    las_files: list[Path],
    target_crs: CRS,
    max_points_per_file: int,
    windows_per_file: int,
    window_read_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for path in las_files:
        with laspy.open(path) as reader:
            header = reader.header
            src_crs = header.parse_crs()
            transformer = None if src_crs is None else Transformer.from_crs(src_crs, target_crs, always_xy=True)
            point_count = int(header.point_count)
            offsets = sample_offsets(point_count, windows_per_file, window_read_size)
            target_per_window = max(1, math.ceil(max_points_per_file / max(len(offsets), 1)))
            file_x: list[np.ndarray] = []
            file_y: list[np.ndarray] = []
            file_z: list[np.ndarray] = []
            for offset in offsets:
                reader.seek(int(offset))
                count = min(window_read_size, point_count - int(offset))
                if count <= 0:
                    continue
                points = reader.read_points(count)
                n = len(points)
                if n == 0:
                    continue
                stride = max(1, math.ceil(n / target_per_window))
                index = np.arange(0, n, stride, dtype=int)
                x = np.asarray(points.x)[index]
                y = np.asarray(points.y)[index]
                z = np.asarray(points.z)[index]
                if transformer is not None:
                    x, y = transformer.transform(x, y)
                    x = np.asarray(x, dtype=float)
                    y = np.asarray(y, dtype=float)
                file_x.append(np.asarray(x, dtype=float))
                file_y.append(np.asarray(y, dtype=float))
                file_z.append(np.asarray(z, dtype=float))
            if file_x:
                x_arr = np.concatenate(file_x)[:max_points_per_file]
                y_arr = np.concatenate(file_y)[:max_points_per_file]
                z_arr = np.concatenate(file_z)[:max_points_per_file]
            else:
                x_arr = np.asarray([], dtype=float)
                y_arr = np.asarray([], dtype=float)
                z_arr = np.asarray([], dtype=float)
            all_x.append(x_arr)
            all_y.append(y_arr)
            all_z.append(z_arr)
            summaries.append(
                {
                    "path": str(path),
                    "point_count": point_count,
                    "sampled_point_count": int(x_arr.size),
                    "source_crs": src_crs.to_string() if src_crs is not None else None,
                    "source_epsg": src_crs.to_epsg() if src_crs is not None else None,
                    "source_bounds": [
                        float(header.mins[0]),
                        float(header.mins[1]),
                        float(header.maxs[0]),
                        float(header.maxs[1]),
                    ],
                    "transformed_bounds": [
                        float(np.min(x_arr)) if x_arr.size else None,
                        float(np.min(y_arr)) if y_arr.size else None,
                        float(np.max(x_arr)) if x_arr.size else None,
                        float(np.max(y_arr)) if y_arr.size else None,
                    ],
                },
            )
    return {
        "x": np.concatenate(all_x) if all_x else np.asarray([], dtype=float),
        "y": np.concatenate(all_y) if all_y else np.asarray([], dtype=float),
        "z": np.concatenate(all_z) if all_z else np.asarray([], dtype=float),
    }, summaries


def sample_offsets(point_count: int, windows_per_file: int, window_read_size: int) -> list[int]:
    if point_count <= 0:
        return []
    if point_count <= window_read_size or windows_per_file <= 1:
        return [0]
    max_offset = max(0, point_count - window_read_size)
    return sorted({int(round(value)) for value in np.linspace(0, max_offset, windows_per_file)})


def load_axis_and_tracks(path: Path) -> tuple[Axis, list[TrackLine]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    main_coords: list[tuple[float, float]] | None = None
    raw_tracks: list[dict[str, Any]] = []
    for index, feature in enumerate(payload.get("features", []) or [], start=1):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates", [])]
        if len(coords) < 2:
            continue
        props = dict(feature.get("properties") or {})
        line_id = str(props.get("line_id") or props.get("id") or f"line_{index:03d}")
        role = str(props.get("topology_role") or "")
        raw_tracks.append({"line_id": line_id, "role": role, "coords": coords})
        if role == "main_through_track" and main_coords is None:
            main_coords = coords
    if main_coords is None:
        all_points = [point for track in raw_tracks for point in track["coords"]]
        axis = estimate_axis_from_points(all_points)
    else:
        axis = axis_from_line(main_coords)
    tracks: list[TrackLine] = []
    for track in raw_tracks:
        s, t = project_array(axis, np.asarray(track["coords"], dtype=float))
        order = np.argsort(s)
        tracks.append(
            TrackLine(
                line_id=track["line_id"],
                role=track["role"],
                s=s[order],
                t=t[order],
                length_m=polyline_length(track["coords"]),
            ),
        )
    return axis, sorted(tracks, key=lambda item: (item.role != "main_through_track", item.line_id))


def axis_from_line(coords: list[tuple[float, float]]) -> Axis:
    start = np.asarray(coords[0], dtype=float)
    end = np.asarray(coords[-1], dtype=float)
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return estimate_axis_from_points(coords)
    longitudinal = direction / norm
    if longitudinal[1] < 0:
        longitudinal = -longitudinal
    origin = np.asarray(coords, dtype=float).mean(axis=0)
    lateral = np.array([-longitudinal[1], longitudinal[0]])
    return Axis(origin=origin, longitudinal=longitudinal, lateral=lateral)


def estimate_axis_from_points(coords: list[tuple[float, float]]) -> Axis:
    matrix = np.asarray(coords, dtype=float)
    origin = matrix.mean(axis=0)
    centered = matrix - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    longitudinal = vh[0]
    if longitudinal[1] < 0:
        longitudinal = -longitudinal
    lateral = np.array([-longitudinal[1], longitudinal[0]])
    return Axis(origin=origin, longitudinal=longitudinal, lateral=lateral)


def project_points(axis: Axis, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack([x, y])
    return project_array(axis, matrix)


def project_array(axis: Axis, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = matrix - axis.origin
    return centered @ axis.longitudinal, centered @ axis.lateral


def load_exclusion_windows(path: Path | None, axis: Axis, *, buffer_m: float) -> list[ExclusionWindow]:
    if path is None:
        return []
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Exclusion workzone GeoJSON not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    windows: list[ExclusionWindow] = []
    for index, feature in enumerate(payload.get("features", []) or [], start=1):
        geometry = feature.get("geometry") or {}
        coords = collect_xy_coordinates(geometry.get("coordinates"))
        if len(coords) < 3:
            continue
        matrix = np.asarray(coords, dtype=float)
        s_values, t_values = project_array(axis, matrix)
        props = dict(feature.get("properties") or {})
        source_id = str(props.get("workzone_id") or props.get("id") or f"workzone_{index:03d}")
        windows.append(
            ExclusionWindow(
                source_id=source_id,
                s_min=float(np.min(s_values)) - buffer_m,
                s_max=float(np.max(s_values)) + buffer_m,
                t_min=float(np.min(t_values)) - buffer_m,
                t_max=float(np.max(t_values)) + buffer_m,
            ),
        )
    return windows


def collect_xy_coordinates(value: Any) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    if not isinstance(value, list):
        return coords
    if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
        return [(float(value[0]), float(value[1]))]
    for item in value:
        coords.extend(collect_xy_coordinates(item))
    return coords


def points_in_exclusion_windows(
    s_values: np.ndarray,
    t_values: np.ndarray,
    windows: list[ExclusionWindow],
) -> np.ndarray:
    mask = np.zeros(s_values.shape, dtype=bool)
    for window in windows:
        mask |= (
            (s_values >= window.s_min)
            & (s_values <= window.s_max)
            & (t_values >= window.t_min)
            & (t_values <= window.t_max)
        )
    return mask


def exclusion_summary(args: argparse.Namespace, windows: list[ExclusionWindow]) -> dict[str, Any]:
    path = args.exclude_workzones
    return {
        "path": str(path.expanduser().resolve()) if path is not None else None,
        "window_count": len(windows),
        "buffer_m": args.exclude_workzone_buffer_m,
        "geometry_policy": "projected_s_t_bbox",
    }


def local_ground_percentile(
    *,
    s_values: np.ndarray,
    z_values: np.ndarray,
    s_bin_size: float,
    percentile: float,
) -> np.ndarray:
    s_min = float(np.min(s_values))
    bin_ids = np.floor((s_values - s_min) / s_bin_size).astype(np.int64)
    local = np.full_like(z_values, np.nan, dtype=float)
    for bin_id in np.unique(bin_ids):
        mask = bin_ids == bin_id
        if np.count_nonzero(mask) < 8:
            continue
        local[mask] = float(np.percentile(z_values[mask], percentile))
    fallback = float(np.percentile(z_values, percentile))
    local[~np.isfinite(local)] = fallback
    return local


def build_histogram_profile(
    *,
    values: np.ndarray,
    bin_size: float,
    value_range: tuple[float, float],
) -> list[dict[str, float]]:
    if values.size == 0:
        return []
    low, high = value_range
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(values)), float(np.max(values))
    bins = np.arange(low, high + bin_size, bin_size)
    if bins.size < 3:
        bins = np.asarray([low, low + bin_size, low + 2 * bin_size])
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    smooth = moving_average(counts.astype(float), window=7)
    return [
        {"value_m": float(center), "count": int(count), "smooth_count": float(score)}
        for center, count, score in zip(centers, counts, smooth)
    ]


def percentile_range(values: np.ndarray, *, low: float, high: float, pad: float) -> tuple[float, float]:
    if values.size == 0:
        return (-1.0, 1.0)
    left = float(np.percentile(values, low)) - pad
    right = float(np.percentile(values, high)) + pad
    return left, right


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def find_profile_peaks(
    profile: list[dict[str, float]],
    *,
    min_distance_m: float,
    min_prominence_fraction: float,
) -> list[dict[str, float]]:
    if len(profile) < 3:
        return []
    values = np.asarray([row["value_m"] for row in profile], dtype=float)
    scores = np.asarray([row["smooth_count"] for row in profile], dtype=float)
    raw_counts = np.asarray([row["count"] for row in profile], dtype=float)
    max_score = float(np.max(scores))
    if not np.isfinite(max_score) or max_score <= 0.0:
        return []
    bin_size = float(np.median(np.diff(values))) if values.size > 1 else min_distance_m
    min_index_distance = max(1, int(round(min_distance_m / max(abs(bin_size), 1e-9))))
    threshold = max(max_score * min_prominence_fraction, float(np.percentile(scores, 70)))
    candidates: list[int] = []
    for index in range(1, len(scores) - 1):
        if (
            scores[index] > 0.0
            and scores[index] >= scores[index - 1]
            and scores[index] >= scores[index + 1]
            and scores[index] >= threshold
        ):
            candidates.append(index)
    candidates.sort(key=lambda index: scores[index], reverse=True)
    selected: list[int] = []
    for index in candidates:
        if all(abs(index - existing) >= min_index_distance for existing in selected):
            selected.append(index)
    selected.sort()
    return [
        {
            "value_m": float(values[index]),
            "count": int(raw_counts[index]),
            "smooth_count": float(scores[index]),
        }
        for index in selected
    ]


def pair_peaks_by_distance(
    peaks: list[dict[str, float]],
    *,
    min_distance: float,
    max_distance: float,
    distance_name: str,
) -> list[dict[str, float]]:
    pairs: list[dict[str, float]] = []
    for left_index, left in enumerate(peaks):
        for right in peaks[left_index + 1 :]:
            distance = float(right["value_m"] - left["value_m"])
            if min_distance <= distance <= max_distance:
                pairs.append(
                    {
                        "left_value_m": float(left["value_m"]),
                        "right_value_m": float(right["value_m"]),
                        distance_name: distance,
                        "score": float(min(left["smooth_count"], right["smooth_count"])),
                        "left_count": int(left["count"]),
                        "right_count": int(right["count"]),
                    },
                )
    pairs.sort(key=lambda item: item["score"], reverse=True)
    return pairs


def spacing_from_gauge_pairs(
    pairs: list[dict[str, float]],
    *,
    min_spacing: float,
    max_spacing: float,
) -> list[dict[str, float]]:
    centers = sorted(
        {
            round((float(pair["left_value_m"]) + float(pair["right_value_m"])) / 2.0, 4)
            for pair in pairs
        },
    )
    result: list[dict[str, float]] = []
    for left, right in zip(centers, centers[1:]):
        spacing = right - left
        if min_spacing <= spacing <= max_spacing:
            result.append({"left_center_t_m": left, "right_center_t_m": right, "spacing_m": round(spacing, 4)})
    return result


def interpolate_track_t(track: TrackLine, s_values: np.ndarray) -> np.ndarray:
    result = np.full_like(s_values, np.nan, dtype=float)
    if track.s.size < 2:
        return result
    mask = (s_values >= float(track.s[0])) & (s_values <= float(track.s[-1]))
    result[mask] = np.interp(s_values[mask], track.s, track.t)
    return result


def write_outputs(
    *,
    out_dir: Path,
    lateral_profile: list[dict[str, float]],
    lateral_peaks: list[dict[str, float]],
    lateral_pairs: list[dict[str, float]],
    center_spacing_candidates: list[dict[str, float]],
    track_profiles: list[dict[str, Any]],
    gauge_candidates: list[dict[str, Any]],
) -> dict[str, str]:
    paths = {
        "lateral_profile_csv": str(out_dir / "lateral_profile.csv"),
        "lateral_peaks_csv": str(out_dir / "lateral_peaks.csv"),
        "global_gauge_pairs_csv": str(out_dir / "global_gauge_pairs.csv"),
        "center_spacing_candidates_csv": str(out_dir / "center_spacing_candidates.csv"),
        "track_offset_profiles_csv": str(out_dir / "track_offset_profiles.csv"),
        "track_gauge_candidates_csv": str(out_dir / "track_gauge_candidates.csv"),
        "lateral_profile_png": str(out_dir / "lateral_profile.png"),
        "summary_json": str(out_dir / "summary.json"),
    }
    write_csv(Path(paths["lateral_profile_csv"]), lateral_profile)
    write_csv(Path(paths["lateral_peaks_csv"]), lateral_peaks)
    write_csv(Path(paths["global_gauge_pairs_csv"]), lateral_pairs)
    write_csv(Path(paths["center_spacing_candidates_csv"]), center_spacing_candidates)
    write_csv(Path(paths["track_offset_profiles_csv"]), track_profiles)
    write_csv(Path(paths["track_gauge_candidates_csv"]), gauge_candidates)
    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def profile_rows(profile: list[dict[str, float]], line_id: str, role: str) -> list[dict[str, Any]]:
    return [
        {
            "line_id": line_id,
            "topology_role": role,
            "offset_m": round(row["value_m"], 4),
            "count": int(row["count"]),
            "smooth_count": round(float(row["smooth_count"]), 3),
        }
        for row in profile
    ]


def write_profile_png(
    path: Path,
    *,
    title: str,
    profile: list[dict[str, float]],
    peaks: list[dict[str, float]],
    pairs: list[dict[str, float]],
) -> None:
    width = 1400
    height = 520
    margin_left = 70
    margin_right = 30
    margin_top = 48
    margin_bottom = 55
    image = Image.new("RGB", (width, height), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    draw.text((margin_left, 16), title, fill=(35, 35, 35))
    if not profile:
        image.save(path)
        return
    xs = np.asarray([row["value_m"] for row in profile], dtype=float)
    ys = np.asarray([row["smooth_count"] for row in profile], dtype=float)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_max = max(float(ys.max()), 1.0)

    def px(value: float) -> float:
        return margin_left + (value - x_min) / max(x_max - x_min, 1e-9) * (width - margin_left - margin_right)

    def py(value: float) -> float:
        return height - margin_bottom - value / y_max * (height - margin_top - margin_bottom)

    draw.line((margin_left, height - margin_bottom, width - margin_right, height - margin_bottom), fill=(80, 80, 80))
    draw.line((margin_left, margin_top, margin_left, height - margin_bottom), fill=(80, 80, 80))
    points = [(px(float(x)), py(float(y))) for x, y in zip(xs, ys)]
    if len(points) >= 2:
        draw.line(points, fill=(30, 110, 170), width=2)
    for peak in peaks:
        x = px(float(peak["value_m"]))
        draw.line((x, margin_top, x, height - margin_bottom), fill=(230, 120, 30), width=1)
        draw.text((x + 3, margin_top + 3), f"{peak['value_m']:.2f}", fill=(110, 70, 20))
    for pair in pairs[:8]:
        x1 = px(float(pair["left_value_m"]))
        x2 = px(float(pair["right_value_m"]))
        y = margin_top + 25 + pairs.index(pair) * 16
        draw.line((x1, y, x2, y), fill=(70, 150, 80), width=2)
        draw.text(((x1 + x2) / 2, y + 2), f"{pair.get('gauge_m', 0):.3f}m", fill=(40, 100, 50))
    draw.text((margin_left, height - 35), f"lateral t / offset in meters, range {x_min:.2f}..{x_max:.2f}", fill=(60, 60, 60))
    image.save(path, quality=92)


def robust_value_summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "median": round(float(np.median(array)), 4),
        "mean": round(float(np.mean(array)), 4),
        "p10": round(float(np.percentile(array, 10)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
    }


def polyline_length(coords: list[tuple[float, float]]) -> float:
    total = 0.0
    for left, right in zip(coords, coords[1:]):
        total += math.hypot(right[0] - left[0], right[1] - left[1])
    return total


if __name__ == "__main__":
    raise SystemExit(main())
