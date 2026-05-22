#!/usr/bin/env python3
"""Prototype turnout centerline correction from the outside rail only.

This is an experimental strict-auto helper. It uses the current automatically
detected turnout connector only as a corridor/tangent guide, then samples the
DeepLab rail probability tiles on the outside side of the route. The resulting
outside rail is shifted inward by half the gauge to recover the centerline.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter1d, median_filter


DEFAULT_TURNOUTS = Path("output/dom_centerline_strict_auto_v1/08_auto_turnout_crossover_evidence/all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson")
DEFAULT_MAINLINE = Path("output/dom_centerline_strict_auto_v1/06_mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_TILE_INDEX = Path("output/dom_centerline_strict_auto_v1/01_dom_tiles/selected_tile_index.csv")
DEFAULT_PROB_DIR = Path("output/dom_centerline_strict_auto_v1/02_deeplab_segmentation/probabilities")
DEFAULT_DOM_GLOB = "data/**/dom.tif"
DEFAULT_OUT_DIR = Path("output/dom_centerline_strict_auto_v1/experiments/pilot_AUTO_007")
DEFAULT_EPSG = 32651


@dataclass(frozen=True)
class Sample:
    station_m: float
    center: tuple[float, float]
    tangent: tuple[float, float]
    normal: tuple[float, float]
    outer_offset_m: float | None
    inner_offset_m: float | None
    outer_score: float
    inner_score: float
    support_offset_m: float | None
    support_score: float
    support_kind: str
    correction_m: float
    valid: bool


@dataclass(frozen=True)
class PeakCandidate:
    outer_offset_m: float
    outer_score: float
    inner_offset_m: float
    inner_score: float
    support_offset_m: float
    support_score: float
    support_kind: str
    correction_m: float
    local_score: float


class Polyline:
    def __init__(self, coords: list[tuple[float, float]]) -> None:
        if len(coords) < 2:
            raise ValueError("Polyline requires at least two points.")
        self.coords = coords
        lengths = [0.0]
        total = 0.0
        for a, b in zip(coords[:-1], coords[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1])
            lengths.append(total)
        if total <= 0.0:
            raise ValueError("Polyline length must be positive.")
        self.lengths = lengths
        self.length = total

    def point_tangent_at(self, station: float, *, tangent_window_m: float = 2.0) -> tuple[tuple[float, float], tuple[float, float]]:
        station = min(max(station, 0.0), self.length)
        index = 0
        while index < len(self.lengths) - 2 and self.lengths[index + 1] < station:
            index += 1
        a = self.coords[index]
        b = self.coords[index + 1]
        seg_len = max(self.lengths[index + 1] - self.lengths[index], 1e-9)
        t = (station - self.lengths[index]) / seg_len
        point = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

        p0 = self.point_at(max(0.0, station - tangent_window_m))
        p1 = self.point_at(min(self.length, station + tangent_window_m))
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = math.hypot(dx, dy)
        if norm <= 0.0:
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            norm = math.hypot(dx, dy)
        return point, (dx / norm, dy / norm)

    def point_at(self, station: float) -> tuple[float, float]:
        station = min(max(station, 0.0), self.length)
        index = 0
        while index < len(self.lengths) - 2 and self.lengths[index + 1] < station:
            index += 1
        a = self.coords[index]
        b = self.coords[index + 1]
        seg_len = max(self.lengths[index + 1] - self.lengths[index], 1e-9)
        t = (station - self.lengths[index]) / seg_len
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


class ProbabilityCrop:
    def __init__(self, array: np.ndarray, left: float, top: float, pixel_x: float, pixel_y: float) -> None:
        self.array = array
        self.left = left
        self.top = top
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.height, self.width = array.shape

    def sample(self, x: float, y: float) -> float:
        fx = (x - self.left) / self.pixel_x
        fy = (y - self.top) / self.pixel_y
        if fx < 0.0 or fy < 0.0 or fx >= self.width - 1 or fy >= self.height - 1:
            return 0.0
        x0 = int(fx)
        y0 = int(fy)
        dx = fx - x0
        dy = fy - y0
        a = self.array[y0, x0]
        b = self.array[y0, x0 + 1]
        c = self.array[y0 + 1, x0]
        d = self.array[y0 + 1, x0 + 1]
        return float(a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prototype outside-rail turnout centerline correction.")
    parser.add_argument("--turnouts", type=Path, default=DEFAULT_TURNOUTS)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument("--branch-id", default="AUTO_007")
    parser.add_argument("--tile-index", type=Path, default=DEFAULT_TILE_INDEX)
    parser.add_argument("--probabilities-dir", type=Path, default=DEFAULT_PROB_DIR)
    parser.add_argument("--dom", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gauge-m", type=float, default=1.5)
    parser.add_argument("--sample-step-m", type=float, default=0.5)
    parser.add_argument("--profile-step-m", type=float, default=0.025)
    parser.add_argument("--outer-search-min-m", type=float, default=0.12)
    parser.add_argument("--outer-search-max-m", type=float, default=2.1)
    parser.add_argument("--expected-rail-offset-tolerance-m", type=float, default=0.55)
    parser.add_argument("--expected-rail-offset-max-deviation-m", type=float, default=1.35)
    parser.add_argument("--inner-partner-tolerance-m", type=float, default=0.28)
    parser.add_argument("--single-rail-offset-tolerance-m", type=float, default=0.28)
    parser.add_argument("--min-inner-rail-probability", type=float, default=0.28)
    parser.add_argument("--rail-continuity-penalty", type=float, default=1.2)
    parser.add_argument("--min-peak-probability", type=float, default=0.28)
    parser.add_argument("--max-correction-m", type=float, default=1.15)
    parser.add_argument("--smooth-sigma-m", type=float, default=3.0)
    parser.add_argument("--median-window-m", type=float, default=2.5)
    parser.add_argument("--unsupported-gap-max-delta-m", type=float, default=0.55)
    parser.add_argument("--mainline-anchor-taper-m", type=float, default=12.0)
    parser.add_argument("--geometry-smooth-sigma-m", type=float, default=5.0)
    parser.add_argument("--geometry-smooth-end-taper-m", type=float, default=8.0)
    parser.add_argument("--max-turn-smooth-deg", type=float, default=0.75)
    parser.add_argument("--max-turn-smooth-iterations", type=int, default=20)
    parser.add_argument("--max-turn-smooth-alpha", type=float, default=0.35)
    parser.add_argument("--crop-margin-m", type=float, default=6.0)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    turnout = find_branch(load_line_features(args.turnouts.expanduser().resolve()), args.branch_id)
    props = turnout.get("properties") or {}
    line = Polyline(line_coords(turnout))
    mainline = Polyline(line_coords(load_line_features(args.mainline.expanduser().resolve())[0]))
    main_normal = mainline_reference_normal(mainline)
    side_sign = outside_side_sign(props)
    sample_stations = np.arange(0.0, line.length + args.sample_step_m * 0.5, args.sample_step_m)
    rough_samples = [line.point_tangent_at(float(station)) for station in sample_stations]
    bounds = expanded_bounds([point for point, _ in rough_samples], args.crop_margin_m + args.outer_search_max_m + args.gauge_m)
    probability = read_probability_crop(
        bounds,
        tile_index=args.tile_index.expanduser().resolve(),
        probabilities_dir=args.probabilities_dir.expanduser().resolve(),
        pixel_size=0.0326,
    )

    samples = extract_outside_rail_samples(
        sample_stations,
        rough_samples,
        probability=probability,
        main_normal=main_normal,
        side_sign=side_sign,
        gauge_m=args.gauge_m,
        profile_step_m=args.profile_step_m,
        search_min_m=args.outer_search_min_m,
        search_max_m=args.outer_search_max_m,
        expected_offset_tolerance_m=args.expected_rail_offset_tolerance_m,
        expected_offset_max_deviation_m=args.expected_rail_offset_max_deviation_m,
        inner_partner_tolerance_m=args.inner_partner_tolerance_m,
        single_rail_offset_tolerance_m=args.single_rail_offset_tolerance_m,
        min_inner_rail_probability=args.min_inner_rail_probability,
        continuity_penalty=args.rail_continuity_penalty,
        min_peak_probability=args.min_peak_probability,
        max_correction_m=args.max_correction_m,
    )
    smoothed = smooth_samples(
        samples,
        step_m=args.sample_step_m,
        median_window_m=args.median_window_m,
        smooth_sigma_m=args.smooth_sigma_m,
        unsupported_gap_max_delta_m=args.unsupported_gap_max_delta_m,
    )
    smoothed = apply_mainline_anchor_taper(
        smoothed,
        start_band=str(props.get("start_band", "")),
        end_band=str(props.get("end_band", "")),
        taper_m=args.mainline_anchor_taper_m,
    )

    centerline_coords = [
        (
            sample.center[0] + sample.normal[0] * sample.correction_m,
            sample.center[1] + sample.normal[1] * sample.correction_m,
        )
        for sample in smoothed
    ]
    centerline_coords = smooth_centerline_geometry(
        centerline_coords,
        step_m=args.sample_step_m,
        smooth_sigma_m=args.geometry_smooth_sigma_m,
        end_taper_m=args.geometry_smooth_end_taper_m,
    )
    centerline_coords = limit_local_turn_angles(
        centerline_coords,
        max_turn_deg=args.max_turn_smooth_deg,
        iterations=args.max_turn_smooth_iterations,
        alpha=args.max_turn_smooth_alpha,
    )
    center_feature = build_line_feature(centerline_coords, props, args.branch_id, kind="outside_rail_offset_centerline")
    rail_features = build_outer_rail_evidence_features(samples, props, args.branch_id)
    outer_rail_coords = [
        (float(x), float(y))
        for feature in rail_features
        for x, y, *_ in feature.get("geometry", {}).get("coordinates", [])
    ]
    center_geojson = out_dir / f"{args.branch_id}_outer_rail_centerline.geojson"
    rail_geojson = out_dir / f"{args.branch_id}_outer_rail_evidence.geojson"
    write_geojson(center_geojson, [center_feature], epsg=args.epsg)
    write_geojson(rail_geojson, rail_features, epsg=args.epsg)
    write_line_shapefile([center_feature], center_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_line_shapefile(rail_features, rail_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_samples_csv(out_dir / f"{args.branch_id}_outer_rail_samples.csv", samples, smoothed)

    dom_path = args.dom.expanduser().resolve() if args.dom else find_dom_path()
    overlay_path = out_dir / f"{args.branch_id}_outer_rail_centerline_overlay.png"
    write_overlay(
        dom_path,
        bounds=expanded_bounds(centerline_coords + outer_rail_coords, 8.0),
        rough_coords=line.coords,
        centerline_coords=centerline_coords,
        outer_rail_coords=outer_rail_coords,
        output_path=overlay_path,
    )

    valid_count = sum(1 for sample in samples if sample.valid)
    corrections = np.array([sample.correction_m for sample in smoothed], dtype=np.float64)
    summary = {
        "mode": "prototype_outside_rail_turnout_centerline",
        "policy": "Uses current strict-auto turnout connector and DeepLab probability tiles only. QA points and accepted old centerlines are not inputs.",
        "branch_id": args.branch_id,
        "input_turnouts": str(args.turnouts.expanduser().resolve()),
        "input_mainline": str(args.mainline.expanduser().resolve()),
        "tile_index": str(args.tile_index.expanduser().resolve()),
        "probabilities_dir": str(args.probabilities_dir.expanduser().resolve()),
        "outside_side_sign": side_sign,
        "gauge_m": args.gauge_m,
        "sample_count": len(samples),
        "valid_outer_rail_sample_count": valid_count,
        "valid_outer_rail_ratio": valid_count / max(len(samples), 1),
        "expected_rail_offset_tolerance_m": args.expected_rail_offset_tolerance_m,
        "expected_rail_offset_max_deviation_m": args.expected_rail_offset_max_deviation_m,
        "inner_partner_tolerance_m": args.inner_partner_tolerance_m,
        "single_rail_offset_tolerance_m": args.single_rail_offset_tolerance_m,
        "min_inner_rail_probability": args.min_inner_rail_probability,
        "rail_continuity_penalty": args.rail_continuity_penalty,
        "unsupported_gap_max_delta_m": args.unsupported_gap_max_delta_m,
        "geometry_smooth_sigma_m": args.geometry_smooth_sigma_m,
        "geometry_smooth_end_taper_m": args.geometry_smooth_end_taper_m,
        "max_turn_smooth_deg": args.max_turn_smooth_deg,
        "max_turn_smooth_iterations": args.max_turn_smooth_iterations,
        "max_turn_smooth_alpha": args.max_turn_smooth_alpha,
        "mainline_anchor_taper_m": args.mainline_anchor_taper_m,
        "correction_m": {
            "min": float(np.min(corrections)) if corrections.size else 0.0,
            "median": float(np.median(corrections)) if corrections.size else 0.0,
            "max": float(np.max(corrections)) if corrections.size else 0.0,
            "p95_abs": float(np.percentile(np.abs(corrections), 95)) if corrections.size else 0.0,
        },
        "outputs": {
            "centerline_geojson": str(center_geojson),
            "centerline_shp": str(center_geojson.with_suffix(".shp")),
            "outer_rail_geojson": str(rail_geojson),
            "outer_rail_shp": str(rail_geojson.with_suffix(".shp")),
            "samples_csv": str(out_dir / f"{args.branch_id}_outer_rail_samples.csv"),
            "overlay_png": str(overlay_path),
            "summary_json": str(out_dir / f"{args.branch_id}_outer_rail_centerline_summary.json"),
        },
    }
    (out_dir / f"{args.branch_id}_outer_rail_centerline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def extract_outside_rail_samples(
    stations: np.ndarray,
    rough_samples: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    probability: ProbabilityCrop,
    main_normal: tuple[float, float],
    side_sign: float,
    gauge_m: float,
    profile_step_m: float,
    search_min_m: float,
    search_max_m: float,
    expected_offset_tolerance_m: float,
    expected_offset_max_deviation_m: float,
    inner_partner_tolerance_m: float,
    single_rail_offset_tolerance_m: float,
    min_inner_rail_probability: float,
    continuity_penalty: float,
    min_peak_probability: float,
    max_correction_m: float,
) -> list[Sample]:
    outside_offsets = np.arange(search_min_m, search_max_m + profile_step_m * 0.5, profile_step_m)
    paired_offsets = np.arange(-search_max_m, search_max_m + profile_step_m * 0.5, profile_step_m)
    expected_outer_offset = gauge_m * 0.5
    profile_candidates: list[list[PeakCandidate]] = []
    sample_frames: list[tuple[float, tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for station, (center, tangent) in zip(stations, rough_samples):
        normal = oriented_normal(tangent, main_normal=main_normal, side_sign=side_sign)
        outside_values = np.array(
            [probability.sample(center[0] + normal[0] * offset, center[1] + normal[1] * offset) for offset in outside_offsets],
            dtype=np.float64,
        )
        paired_values = np.array(
            [probability.sample(center[0] + normal[0] * offset, center[1] + normal[1] * offset) for offset in paired_offsets],
            dtype=np.float64,
        )
        candidates = profile_peak_candidates(
            outside_offsets,
            outside_values,
            paired_offsets=paired_offsets,
            paired_values=paired_values,
            gauge_m=gauge_m,
            expected_offset_m=expected_outer_offset,
            expected_tolerance_m=expected_offset_tolerance_m,
            expected_max_deviation_m=expected_offset_max_deviation_m,
            inner_partner_tolerance_m=inner_partner_tolerance_m,
            single_rail_offset_tolerance_m=single_rail_offset_tolerance_m,
            min_peak_probability=min_peak_probability,
            min_inner_rail_probability=min_inner_rail_probability,
        )
        profile_candidates.append(candidates)
        sample_frames.append((float(station), center, tangent, normal))
    selected_indices = select_smooth_peak_sequence(profile_candidates, continuity_penalty=continuity_penalty)
    samples: list[Sample] = []
    for (station, center, tangent, normal), candidates, selected_index in zip(sample_frames, profile_candidates, selected_indices):
        peak = candidates[selected_index] if selected_index is not None else None
        if peak is None:
            samples.append(Sample(station, center, tangent, normal, None, None, 0.0, 0.0, None, 0.0, "", 0.0, False))
            continue
        valid = abs(peak.correction_m) <= max_correction_m and peak.local_score > -0.25
        if not valid:
            samples.append(
                Sample(
                    station,
                    center,
                    tangent,
                    normal,
                    float(peak.outer_offset_m),
                    float(peak.inner_offset_m),
                    float(peak.outer_score),
                    float(peak.inner_score),
                    float(peak.support_offset_m),
                    float(peak.support_score),
                    peak.support_kind,
                    0.0,
                    False,
                )
            )
            continue
        samples.append(
            Sample(
                station,
                center,
                tangent,
                normal,
                float(peak.outer_offset_m),
                float(peak.inner_offset_m),
                float(peak.outer_score),
                float(peak.inner_score),
                float(peak.support_offset_m),
                float(peak.support_score),
                peak.support_kind,
                float(peak.correction_m),
                True,
            )
        )
    return samples


def profile_peak_candidates(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    paired_offsets: np.ndarray,
    paired_values: np.ndarray,
    gauge_m: float,
    expected_offset_m: float,
    expected_tolerance_m: float,
    expected_max_deviation_m: float,
    inner_partner_tolerance_m: float,
    single_rail_offset_tolerance_m: float,
    min_peak_probability: float,
    min_inner_rail_probability: float,
) -> list[PeakCandidate]:
    if values.size == 0:
        return []
    smooth = gaussian_filter1d(values, sigma=max(1.0, 0.06 / max(offsets[1] - offsets[0], 1e-6)))
    paired_smooth = gaussian_filter1d(paired_values, sigma=max(1.0, 0.06 / max(paired_offsets[1] - paired_offsets[0], 1e-6)))
    candidates: list[PeakCandidate] = []

    def add_candidate(index: int) -> None:
        offset = float(offsets[index])
        if abs(offset - expected_offset_m) > expected_max_deviation_m:
            return
        probability_score = float(smooth[index])
        inner = find_inner_partner(
            paired_offsets,
            paired_smooth,
            expected_inner_offset_m=offset - gauge_m,
            tolerance_m=inner_partner_tolerance_m,
            min_probability=min_inner_rail_probability,
        )
        if inner is None:
            return
        inner_offset, inner_score = inner
        gauge_error = abs((offset - inner_offset) - gauge_m)
        correction = (offset + inner_offset) * 0.5
        local_score = 0.5 * probability_score + 0.5 * inner_score
        outer_prior_error = abs(offset - expected_offset_m) / max(expected_tolerance_m, 1e-6)
        local_score -= 0.03 * outer_prior_error * outer_prior_error
        local_score -= 0.35 * (gauge_error / max(inner_partner_tolerance_m, 1e-6)) ** 2
        candidates.append(
            PeakCandidate(
                outer_offset_m=offset,
                outer_score=probability_score,
                inner_offset_m=float(inner_offset),
                inner_score=float(inner_score),
                support_offset_m=offset,
                support_score=probability_score,
                support_kind="paired_outer",
                correction_m=float(correction),
                local_score=float(local_score),
            )
        )

    def add_single_rail_candidate(expected_rail_offset: float, *, kind: str) -> None:
        rail = find_single_rail_peak(
            paired_offsets,
            paired_smooth,
            expected_offset_m=expected_rail_offset,
            tolerance_m=single_rail_offset_tolerance_m,
            min_probability=min_inner_rail_probability,
        )
        if rail is None:
            return
        rail_offset, rail_score = rail
        if kind == "single_left":
            correction = rail_offset + gauge_m * 0.5
            outer_offset = rail_offset + gauge_m
            inner_offset = rail_offset
            outer_score = 0.0
            inner_score = rail_score
        else:
            correction = rail_offset - gauge_m * 0.5
            outer_offset = rail_offset
            inner_offset = rail_offset - gauge_m
            outer_score = rail_score
            inner_score = 0.0
        prior_error = abs(rail_offset - expected_rail_offset) / max(single_rail_offset_tolerance_m, 1e-6)
        local_score = rail_score - 0.12 - 0.04 * prior_error * prior_error
        candidates.append(
            PeakCandidate(
                outer_offset_m=float(outer_offset),
                outer_score=float(outer_score),
                inner_offset_m=float(inner_offset),
                inner_score=float(inner_score),
                support_offset_m=float(rail_offset),
                support_score=float(rail_score),
                support_kind=kind,
                correction_m=float(correction),
                local_score=float(local_score),
            )
        )

    for index in range(1, len(smooth) - 1):
        if smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1] and smooth[index] >= min_peak_probability:
            add_candidate(index)
    index = int(np.argmax(smooth))
    if smooth[index] >= min_peak_probability:
        offset = float(offsets[index])
        if all(abs(offset - item.outer_offset_m) > 0.08 for item in candidates):
            add_candidate(index)
    add_single_rail_candidate(-gauge_m * 0.5, kind="single_left")
    add_single_rail_candidate(gauge_m * 0.5, kind="single_right")
    candidates.sort(key=lambda item: item.local_score, reverse=True)
    return candidates[:6]


def find_inner_partner(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    expected_inner_offset_m: float,
    tolerance_m: float,
    min_probability: float,
) -> tuple[float, float] | None:
    if offsets.size == 0 or values.size == 0:
        return None
    mask = np.abs(offsets - expected_inner_offset_m) <= tolerance_m
    if not bool(mask.any()):
        return None
    candidate_indices = np.flatnonzero(mask)
    best_index = int(candidate_indices[np.argmax(values[candidate_indices])])
    if values[best_index] < min_probability:
        return None
    return float(offsets[best_index]), float(values[best_index])


def find_single_rail_peak(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    expected_offset_m: float,
    tolerance_m: float,
    min_probability: float,
) -> tuple[float, float] | None:
    if offsets.size == 0 or values.size == 0:
        return None
    mask = np.abs(offsets - expected_offset_m) <= tolerance_m
    if not bool(mask.any()):
        return None
    candidate_indices = np.flatnonzero(mask)
    local_maxima = [
        index
        for index in candidate_indices
        if 0 < index < values.size - 1 and values[index] >= values[index - 1] and values[index] >= values[index + 1]
    ]
    search_indices = np.array(local_maxima if local_maxima else candidate_indices, dtype=np.int64)
    best_index = int(search_indices[np.argmax(values[search_indices])])
    if values[best_index] < min_probability:
        return None
    return float(offsets[best_index]), float(values[best_index])


def peak_local_score(
    offset_m: float,
    probability_score: float,
    *,
    expected_offset_m: float,
    expected_tolerance_m: float,
) -> float:
    normalized_error = (offset_m - expected_offset_m) / max(expected_tolerance_m, 1e-6)
    return float(probability_score - 0.22 * normalized_error * normalized_error)


def select_smooth_peak_sequence(
    candidates_by_station: list[list[PeakCandidate]],
    *,
    continuity_penalty: float,
) -> list[int | None]:
    if not candidates_by_station:
        return []
    states_by_station: list[list[PeakCandidate | None]] = [candidates + [None] for candidates in candidates_by_station]
    dp: list[list[float]] = []
    parents: list[list[int | None]] = []
    for station_index, states in enumerate(states_by_station):
        station_scores: list[float] = []
        station_parents: list[int | None] = []
        for state_index, state in enumerate(states):
            local_score = state.local_score if state is not None else -0.45
            if station_index == 0:
                station_scores.append(local_score)
                station_parents.append(None)
                continue
            best_score = -1e18
            best_parent: int | None = None
            for prev_index, prev_state in enumerate(states_by_station[station_index - 1]):
                score = dp[station_index - 1][prev_index] + local_score
                if state is not None and prev_state is not None:
                    score -= continuity_penalty * abs(state.correction_m - prev_state.correction_m)
                elif (state is None) != (prev_state is None):
                    score -= 0.2
                if score > best_score:
                    best_score = score
                    best_parent = prev_index
            station_scores.append(best_score)
            station_parents.append(best_parent)
        dp.append(station_scores)
        parents.append(station_parents)
    last_index = max(range(len(dp[-1])), key=lambda index: dp[-1][index])
    selected: list[int | None] = [None] * len(states_by_station)
    for station_index in range(len(states_by_station) - 1, -1, -1):
        candidates = candidates_by_station[station_index]
        selected[station_index] = last_index if last_index < len(candidates) else None
        parent = parents[station_index][last_index]
        if parent is None:
            break
        last_index = parent
    return selected


def smooth_samples(
    samples: list[Sample],
    *,
    step_m: float,
    median_window_m: float,
    smooth_sigma_m: float,
    unsupported_gap_max_delta_m: float,
) -> list[Sample]:
    if not samples:
        return []
    stations = np.array([sample.station_m for sample in samples], dtype=np.float64)
    raw = np.array([sample.correction_m if sample.valid else np.nan for sample in samples], dtype=np.float64)
    valid = np.isfinite(raw)
    if not valid.any():
        return samples
    filled = np.interp(stations, stations[valid], raw[valid])
    median_size = max(1, int(round(median_window_m / max(step_m, 1e-6))))
    if median_size % 2 == 0:
        median_size += 1
    filtered = median_filter(filled, size=median_size, mode="nearest")
    sigma = max(0.0, smooth_sigma_m / max(step_m, 1e-6))
    smoothed = gaussian_filter1d(filtered, sigma=sigma, mode="nearest") if sigma > 0.0 else filtered
    smoothed = protect_unsupported_gap_jumps(smoothed, raw, valid, max_delta_m=unsupported_gap_max_delta_m)
    return [
        Sample(
            sample.station_m,
            sample.center,
            sample.tangent,
            sample.normal,
            sample.outer_offset_m,
            sample.inner_offset_m,
            sample.outer_score,
            sample.inner_score,
            sample.support_offset_m,
            sample.support_score,
            sample.support_kind,
            float(smoothed[index]),
            sample.valid,
        )
        for index, sample in enumerate(samples)
    ]


def protect_unsupported_gap_jumps(
    smoothed: np.ndarray,
    raw: np.ndarray,
    valid: np.ndarray,
    *,
    max_delta_m: float,
) -> np.ndarray:
    if smoothed.size == 0 or max_delta_m <= 0.0:
        return smoothed
    protected = smoothed.copy()
    index = 0
    while index < valid.size:
        if valid[index]:
            index += 1
            continue
        start = index
        while index < valid.size and not valid[index]:
            index += 1
        end = index - 1
        left = start - 1
        right = index
        if left < 0 or right >= valid.size or not valid[left] or not valid[right]:
            continue
        left_corr = float(raw[left])
        right_corr = float(raw[right])
        if abs(left_corr - right_corr) <= max_delta_m:
            continue
        fallback = left_corr if abs(left_corr) <= abs(right_corr) else right_corr
        protected[start : end + 1] = fallback
    return protected


def apply_mainline_anchor_taper(
    samples: list[Sample],
    *,
    start_band: str,
    end_band: str,
    taper_m: float,
) -> list[Sample]:
    if not samples or taper_m <= 0.0:
        return samples
    total_length = samples[-1].station_m
    anchored_start = start_band == "mainline_2_track"
    anchored_end = end_band == "mainline_2_track"
    if not anchored_start and not anchored_end:
        return samples
    tapered: list[Sample] = []
    for sample in samples:
        weight = 1.0
        if anchored_start:
            weight = min(weight, smoothstep(min(max(sample.station_m / taper_m, 0.0), 1.0)))
        if anchored_end:
            distance_to_end = max(total_length - sample.station_m, 0.0)
            weight = min(weight, smoothstep(min(max(distance_to_end / taper_m, 0.0), 1.0)))
        tapered.append(
            Sample(
                sample.station_m,
                sample.center,
                sample.tangent,
                sample.normal,
                sample.outer_offset_m,
                sample.inner_offset_m,
                sample.outer_score,
                sample.inner_score,
                sample.support_offset_m,
                sample.support_score,
                sample.support_kind,
                sample.correction_m * weight,
                sample.valid,
            )
        )
    return tapered


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def smooth_centerline_geometry(
    coords: list[tuple[float, float]],
    *,
    step_m: float,
    smooth_sigma_m: float,
    end_taper_m: float,
    preserve_mask: list[bool] | None = None,
) -> list[tuple[float, float]]:
    if len(coords) < 3 or smooth_sigma_m <= 0.0:
        return coords
    sigma = smooth_sigma_m / max(step_m, 1e-6)
    x = np.array([point[0] for point in coords], dtype=np.float64)
    y = np.array([point[1] for point in coords], dtype=np.float64)
    smooth_x = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    smooth_y = gaussian_filter1d(y, sigma=sigma, mode="nearest")
    taper_count = max(1, int(round(end_taper_m / max(step_m, 1e-6))))
    last = len(coords) - 1
    for index in range(len(coords)):
        endpoint_weight = min(index / taper_count, (last - index) / taper_count, 1.0)
        endpoint_weight = max(0.0, min(1.0, endpoint_weight))
        smooth_x[index] = x[index] * (1.0 - endpoint_weight) + smooth_x[index] * endpoint_weight
        smooth_y[index] = y[index] * (1.0 - endpoint_weight) + smooth_y[index] * endpoint_weight
    if preserve_mask is not None:
        for index, preserve in enumerate(preserve_mask[: len(coords)]):
            if preserve:
                smooth_x[index] = x[index]
                smooth_y[index] = y[index]
    return [(float(px), float(py)) for px, py in zip(smooth_x, smooth_y)]


def limit_local_turn_angles(
    coords: list[tuple[float, float]],
    *,
    max_turn_deg: float,
    iterations: int,
    alpha: float,
) -> list[tuple[float, float]]:
    if len(coords) < 3 or max_turn_deg <= 0.0 or iterations <= 0 or alpha <= 0.0:
        return coords
    alpha = min(max(alpha, 0.0), 1.0)
    current = list(coords)
    for _ in range(iterations):
        changed = False
        next_coords = list(current)
        for index in range(1, len(current) - 1):
            turn = turn_angle_deg(current[index - 1], current[index], current[index + 1])
            if turn <= max_turn_deg:
                continue
            prev_point = current[index - 1]
            point = current[index]
            next_point = current[index + 1]
            midpoint = ((prev_point[0] + next_point[0]) * 0.5, (prev_point[1] + next_point[1]) * 0.5)
            excess = min(1.0, (turn - max_turn_deg) / max(max_turn_deg, 1e-6))
            weight = alpha * excess
            next_coords[index] = (
                point[0] * (1.0 - weight) + midpoint[0] * weight,
                point[1] * (1.0 - weight) + midpoint[1] * weight,
            )
            changed = True
        current = next_coords
        if not changed:
            break
    return current


def turn_angle_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def read_probability_crop(
    bounds: tuple[float, float, float, float],
    *,
    tile_index: Path,
    probabilities_dir: Path,
    pixel_size: float,
) -> ProbabilityCrop:
    xmin, ymin, xmax, ymax = bounds
    width = max(1, int(math.ceil((xmax - xmin) / pixel_size)))
    height = max(1, int(math.ceil((ymax - ymin) / pixel_size)))
    xmax = xmin + width * pixel_size
    ymin = ymax - height * pixel_size
    probability = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    with tile_index.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            txmin = float(row["x_min"])
            tymin = float(row["y_min"])
            txmax = float(row["x_max"])
            tymax = float(row["y_max"])
            if txmax < xmin or txmin > xmax or tymax < ymin or tymin > ymax:
                continue
            image_path = probabilities_dir / row["image_name"]
            if not image_path.exists():
                continue
            tile = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
            ox0 = max(xmin, txmin)
            ox1 = min(xmax, txmax)
            oy0 = max(ymin, tymin)
            oy1 = min(ymax, tymax)
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            out_x0 = max(0, int(math.floor((ox0 - xmin) / (xmax - xmin) * width)))
            out_x1 = min(width, int(math.ceil((ox1 - xmin) / (xmax - xmin) * width)))
            out_y0 = max(0, int(math.floor((ymax - oy1) / (ymax - ymin) * height)))
            out_y1 = min(height, int(math.ceil((ymax - oy0) / (ymax - ymin) * height)))
            tile_x0 = max(0, int(math.floor((ox0 - txmin) / (txmax - txmin) * tile.shape[1])))
            tile_x1 = min(tile.shape[1], int(math.ceil((ox1 - txmin) / (txmax - txmin) * tile.shape[1])))
            tile_y0 = max(0, int(math.floor((tymax - oy1) / (tymax - tymin) * tile.shape[0])))
            tile_y1 = min(tile.shape[0], int(math.ceil((tymax - oy0) / (tymax - tymin) * tile.shape[0])))
            if out_x1 <= out_x0 or out_y1 <= out_y0 or tile_x1 <= tile_x0 or tile_y1 <= tile_y0:
                continue
            patch = Image.fromarray((tile[tile_y0:tile_y1, tile_x0:tile_x1] * 255).astype(np.uint8), mode="L").resize(
                (out_x1 - out_x0, out_y1 - out_y0),
                resample=Image.Resampling.BILINEAR,
            )
            patch_arr = np.asarray(patch, dtype=np.float32) / 255.0
            probability[out_y0:out_y1, out_x0:out_x1] += patch_arr
            weight[out_y0:out_y1, out_x0:out_x1] += 1.0
    return ProbabilityCrop(probability / np.maximum(weight, 1.0), xmin, ymax, pixel_size, -pixel_size)


def load_line_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        feature
        for feature in payload.get("features", []) or []
        if feature.get("geometry", {}).get("type") == "LineString" and len(feature.get("geometry", {}).get("coordinates", [])) >= 2
    ]


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, *_ in feature["geometry"]["coordinates"]]


def find_branch(features: list[dict[str, Any]], branch_id: str) -> dict[str, Any]:
    for feature in features:
        props = feature.get("properties") or {}
        if props.get("branch_id") == branch_id or props.get("line_id") == f"TURNOUT_{branch_id}":
            return feature
    raise ValueError(f"Branch not found: {branch_id}")


def mainline_reference_normal(mainline: Polyline) -> tuple[float, float]:
    start = mainline.coords[0]
    end = mainline.coords[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    return (-dy / length, dx / length)


def outside_side_sign(props: dict[str, Any]) -> float:
    offsets: list[float] = []
    for key in ("offset_start_m", "offset_end_m"):
        try:
            offsets.append(float(props.get(key)))
        except (TypeError, ValueError):
            pass
    if offsets:
        far = max(offsets, key=lambda value: abs(value))
        if abs(far) > 1e-6:
            return 1.0 if far > 0.0 else -1.0
    for key in ("start_band", "end_band"):
        value = str(props.get(key, ""))
        if "plus" in value:
            return 1.0
        if "minus" in value:
            return -1.0
    return 1.0


def oriented_normal(
    tangent: tuple[float, float],
    *,
    main_normal: tuple[float, float],
    side_sign: float,
) -> tuple[float, float]:
    normal = (-tangent[1], tangent[0])
    if dot(normal, main_normal) * side_sign < 0.0:
        normal = (-normal[0], -normal[1])
    return normal


def dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def expanded_bounds(points: list[tuple[float, float]], margin_m: float) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs) - margin_m, min(ys) - margin_m, max(xs) + margin_m, max(ys) + margin_m)


def build_line_feature(coords: list[tuple[float, float]], source_props: dict[str, Any], branch_id: str, *, kind: str) -> dict[str, Any]:
    props = {
        "line_id": f"{branch_id}_{kind}",
        "branch_id": branch_id,
        "source": "deeplab_outside_single_rail",
        "geom_kind": kind,
        "network_role": "turnout_connector",
        "start_band": source_props.get("start_band", ""),
        "end_band": source_props.get("end_band", ""),
    }
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
    }


def build_outer_rail_evidence_features(samples: list[Sample], source_props: dict[str, Any], branch_id: str) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    current: list[tuple[float, float]] = []
    current_kind = ""

    def flush() -> None:
        nonlocal current, current_kind
        if len(current) >= 2:
            feature = build_line_feature(current, source_props, branch_id, kind="outside_rail_evidence")
            feature["properties"]["line_id"] = f"{branch_id}_outside_rail_evidence_{len(features) + 1:02d}"
            feature["properties"]["support_kind"] = current_kind
            features.append(feature)
        current = []
        current_kind = ""

    for sample in samples:
        if not sample.valid or sample.support_offset_m is None:
            flush()
            continue
        if current and sample.support_kind != current_kind:
            flush()
        current_kind = sample.support_kind
        current.append(
            (
                sample.center[0] + sample.normal[0] * sample.support_offset_m,
                sample.center[1] + sample.normal[1] * sample.support_offset_m,
            )
        )
    flush()
    return features


def write_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_line_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    import shapefile
    import rasterio.crs

    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("line_id", "C", size=80)
    writer.field("branch_id", "C", size=32)
    writer.field("geom_kind", "C", size=64)
    writer.field("support", "C", size=32)
    writer.field("start_band", "C", size=64)
    writer.field("end_band", "C", size=64)
    for feature in features:
        props = feature.get("properties") or {}
        coords = feature.get("geometry", {}).get("coordinates", [])
        writer.line([[[float(x), float(y)] for x, y, *_ in coords]])
        writer.record(
            str(props.get("line_id", "")),
            str(props.get("branch_id", "")),
            str(props.get("geom_kind", "")),
            str(props.get("support_kind", "")),
            str(props.get("start_band", "")),
            str(props.get("end_band", "")),
        )
    writer.close()
    output_path.with_suffix(".prj").write_text(rasterio.crs.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def write_samples_csv(path: Path, raw_samples: list[Sample], smoothed_samples: list[Sample]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "station_m",
                "x",
                "y",
                "outer_offset_m",
                "inner_offset_m",
                "pair_gauge_m",
                "outer_score",
                "inner_score",
                "support_offset_m",
                "support_score",
                "support_kind",
                "raw_correction_m",
                "smoothed_correction_m",
                "valid",
            ],
        )
        writer.writeheader()
        for raw, smooth in zip(raw_samples, smoothed_samples):
            writer.writerow(
                {
                    "station_m": round(raw.station_m, 3),
                    "x": round(raw.center[0], 6),
                    "y": round(raw.center[1], 6),
                    "outer_offset_m": "" if raw.outer_offset_m is None else round(raw.outer_offset_m, 4),
                    "inner_offset_m": "" if raw.inner_offset_m is None else round(raw.inner_offset_m, 4),
                    "pair_gauge_m": "" if raw.outer_offset_m is None or raw.inner_offset_m is None else round(raw.outer_offset_m - raw.inner_offset_m, 4),
                    "outer_score": round(raw.outer_score, 4),
                    "inner_score": round(raw.inner_score, 4),
                    "support_offset_m": "" if raw.support_offset_m is None else round(raw.support_offset_m, 4),
                    "support_score": round(raw.support_score, 4),
                    "support_kind": raw.support_kind,
                    "raw_correction_m": round(raw.correction_m, 4),
                    "smoothed_correction_m": round(smooth.correction_m, 4),
                    "valid": int(raw.valid),
                }
            )


def find_dom_path() -> Path:
    matches = sorted(Path(".").glob(DEFAULT_DOM_GLOB))
    if not matches:
        raise FileNotFoundError(f"No DOM found with glob {DEFAULT_DOM_GLOB}")
    return matches[0]


def write_overlay(
    dom_path: Path,
    *,
    bounds: tuple[float, float, float, float],
    rough_coords: list[tuple[float, float]],
    centerline_coords: list[tuple[float, float]],
    outer_rail_coords: list[tuple[float, float]],
    output_path: Path,
) -> None:
    with rasterio.open(dom_path) as dataset:
        window = from_bounds(*bounds, transform=dataset.transform).round_offsets().round_lengths()
        data = dataset.read([1, 2, 3], window=window, boundless=True, fill_value=0)
        transform = dataset.window_transform(window)
    rgb = np.moveaxis(data, 0, -1)
    if rgb.dtype != np.uint8:
        mask = rgb > 0
        low = np.percentile(rgb[mask], 1) if np.any(mask) else 0.0
        high = np.percentile(rgb[mask], 99) if np.any(mask) else 255.0
        rgb = np.clip((rgb - low) / (high - low + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    left = transform.c
    top = transform.f
    pixel_x = transform.a
    pixel_y = transform.e

    def world_to_pixel(x: float, y: float) -> tuple[int, int]:
        return (int(round((x - left) / pixel_x)), int(round((y - top) / pixel_y)))

    draw_polyline(draw, rough_coords, world_to_pixel, (255, 0, 0, 220), width=4)
    draw_polyline(draw, centerline_coords, world_to_pixel, (0, 255, 255, 255), width=3)
    draw_polyline(draw, outer_rail_coords, world_to_pixel, (0, 255, 0, 230), width=2)
    draw.rectangle((8, 8, 300, 70), fill=(0, 0, 0, 120))
    for idx, (text, color) in enumerate(
        [
            ("red rough auto turnout", (255, 0, 0, 255)),
            ("cyan outside-rail centerline", (0, 255, 255, 255)),
            ("green detected outside rail", (0, 255, 0, 255)),
        ]
    ):
        y = 16 + idx * 16
        draw.rectangle((16, y + 3, 30, y + 12), fill=color)
        draw.text((36, y), text, fill=(255, 255, 255, 255))
    image.save(output_path, quality=95)


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    coords: list[tuple[float, float]],
    world_to_pixel: Any,
    color: tuple[int, int, int, int],
    *,
    width: int,
) -> None:
    if len(coords) < 2:
        return
    draw.line([world_to_pixel(x, y) for x, y in coords], fill=color, width=width, joint="curve")


if __name__ == "__main__":
    raise SystemExit(main())
