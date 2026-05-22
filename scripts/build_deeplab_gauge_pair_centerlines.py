#!/usr/bin/env python3
"""Build gauge-paired centerline candidates from DeepLab single-rail evidence."""
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
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds


DEFAULT_PROBABILITY = Path("output/raw_dom_roi_fullpass_v1/segmentation_evidence_overlay_ta08_deeplab_v1/ta08_segmentation_probability_u8.tif")
DEFAULT_MAINLINE = Path("output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_DOM = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u65e0\u4eba\u673a\u6570\u636e" / "\u6b63\u5c04" / "dom.tif"
DEFAULT_DSM = Path("D:/") / "\u6b63\u5c04" / "lidars" / "terra_dsm" / "dsm.tif"
DEFAULT_GAUGE_SUMMARY = Path("output/handheld_las_constraints_fullpass_switch_excluded/summary.json")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_ta08_v1")
DEFAULT_EPSG = 32651

REVIEW_POINTS = [
    ("ta08_user_coord", 315334.923, 3520755.899),
    ("coord01_user_ok", 315349.015, 3520808.310),
    ("coord02_user_deviates", 315341.558, 3520778.002),
    ("coord03_user_intersects_rail", 315333.927, 3520749.822),
]


@dataclass(frozen=True)
class Guide:
    start: tuple[float, float]
    end: tuple[float, float]
    ux: float
    uy: float
    nx: float
    ny: float
    length: float

    @classmethod
    def from_coords(cls, coords: list[tuple[float, float]]) -> "Guide":
        if len(coords) < 2:
            raise ValueError("Guide requires at least two coordinates.")
        start = coords[0]
        end = coords[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            raise ValueError("Guide endpoints must be different.")
        ux = dx / length
        uy = dy / length
        return cls(start=start, end=end, ux=ux, uy=uy, nx=-uy, ny=ux, length=length)

    def station_offset(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dx = x - self.start[0]
        dy = y - self.start[1]
        return dx * self.ux + dy * self.uy, dx * self.nx + dy * self.ny

    def station_offset_one(self, point: tuple[float, float]) -> tuple[float, float]:
        dx = point[0] - self.start[0]
        dy = point[1] - self.start[1]
        return dx * self.ux + dy * self.uy, dx * self.nx + dy * self.ny

    def point_at(self, station_m: float, offset_m: float) -> tuple[float, float]:
        return (
            self.start[0] + self.ux * station_m + self.nx * offset_m,
            self.start[1] + self.uy * station_m + self.ny * offset_m,
        )


@dataclass(frozen=True)
class LateralPeak:
    station_index: int
    station_m: float
    offset_m: float
    score: float
    count: int
    mean_probability: float
    dsm_contrast_m: float | None = None


@dataclass(frozen=True)
class PairSample:
    sample_id: int
    station_index: int
    station_m: float
    center_offset_m: float
    left_offset_m: float
    right_offset_m: float
    gauge_m: float
    score: float
    probability_score: float
    dsm_contrast_m: float | None
    left_count: int
    right_count: int


@dataclass(frozen=True)
class PairSequence:
    sequence_id: str
    samples: list[PairSample]


@dataclass(frozen=True)
class DsmEvidence:
    path: Path
    transform: Affine
    array: np.ndarray
    nodata: float | None
    station_ground_m: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TA08 rail-pair centerline candidates from DeepLab probability evidence.")
    parser.add_argument("--probability", type=Path, default=DEFAULT_PROBABILITY)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--dsm", type=Path, default=DEFAULT_DSM)
    parser.add_argument("--gauge-summary", type=Path, default=DEFAULT_GAUGE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gauge-m", type=float, default=0.0)
    parser.add_argument("--gauge-tolerance-m", type=float, default=0.22)
    parser.add_argument("--prob-threshold", type=float, default=0.70)
    parser.add_argument("--strong-threshold", type=float, default=0.90)
    parser.add_argument("--station-bin-m", type=float, default=0.75)
    parser.add_argument("--offset-bin-m", type=float, default=0.06)
    parser.add_argument("--offset-min-m", type=float, default=-3.0)
    parser.add_argument("--offset-max-m", type=float, default=8.2)
    parser.add_argument("--min-peak-score", type=float, default=5.0)
    parser.add_argument("--max-pairs-per-station", type=int, default=4)
    parser.add_argument("--max-link-gap-m", type=float, default=4.0)
    parser.add_argument("--max-link-offset-m", type=float, default=0.55)
    parser.add_argument("--max-link-slope", type=float, default=0.18)
    parser.add_argument("--min-sequence-samples", type=int, default=10)
    parser.add_argument("--min-sequence-length-m", type=float, default=8.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--evidence-every-n", type=int, default=2)
    parser.add_argument("--qa-crop-m", type=float, default=78.0)
    parser.add_argument("--no-dsm", action="store_true")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    probability_path = args.probability.expanduser().resolve()
    guide = Guide.from_coords(load_first_line(args.mainline.expanduser().resolve()))
    gauge_m = resolve_gauge(args.gauge_m, args.gauge_summary.expanduser().resolve())

    with rasterio.open(probability_path) as prob_ds:
        probability = prob_ds.read(1)
        prob_transform = prob_ds.transform
        prob_bounds = prob_ds.bounds
        crs = prob_ds.crs

    station_edges, offset_edges = build_edges(probability, prob_transform, guide, args=args)
    dsm_evidence = None
    dsm_path = args.dsm.expanduser().resolve()
    if not args.no_dsm and dsm_path.exists():
        dsm_evidence = load_dsm_evidence(dsm_path, prob_bounds=prob_bounds, guide=guide, station_edges=station_edges, args=args)

    samples, extraction_stats = extract_pair_samples(
        probability,
        prob_transform,
        guide,
        gauge_m=gauge_m,
        station_edges=station_edges,
        offset_edges=offset_edges,
        dsm_evidence=dsm_evidence,
        args=args,
    )
    sequences = link_pair_samples(
        samples,
        max_gap_m=args.max_link_gap_m,
        max_offset_m=args.max_link_offset_m,
        max_slope=args.max_link_slope,
        min_samples=args.min_sequence_samples,
        min_length_m=args.min_sequence_length_m,
    )
    centerline_features = build_centerline_features(sequences, guide=guide, gauge_m=gauge_m, smooth_window=args.smooth_window)
    evidence_features = build_pair_evidence_features(sequences, guide=guide, every_n=max(1, args.evidence_every_n))

    centerline_geojson = out_dir / "deeplab_gauge_pair_centerlines.geojson"
    evidence_geojson = out_dir / "deeplab_gauge_pair_evidence.geojson"
    write_geojson(centerline_geojson, centerline_features, epsg=args.epsg)
    write_geojson(evidence_geojson, evidence_features, epsg=args.epsg)
    write_line_shapefile(centerline_features, centerline_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_line_shapefile(evidence_features, evidence_geojson.with_suffix(".shp"), epsg=args.epsg)
    write_centerline_qml(centerline_geojson.with_suffix(".qml"))
    write_evidence_qml(evidence_geojson.with_suffix(".qml"))

    qa_summary = write_qa_crops(
        args.dom.expanduser().resolve(),
        probability_path=probability_path,
        centerline_features=centerline_features,
        evidence_features=evidence_features,
        out_dir=out_dir / "qa_crops",
        crop_m=args.qa_crop_m,
        threshold=args.prob_threshold,
        strong_threshold=args.strong_threshold,
    )

    sample_rows = pair_sample_rows(sequences)
    write_csv(out_dir / "deeplab_gauge_pair_samples.csv", sample_rows)
    summary = {
        "mode": "deeplab_probability_gauge_pair_filtering",
        "probability": str(probability_path),
        "mainline": str(args.mainline.expanduser().resolve()),
        "dom": str(args.dom.expanduser().resolve()),
        "dsm": str(dsm_path) if dsm_evidence is not None else None,
        "crs": str(crs),
        "gauge_m": gauge_m,
        "parameters": {
            "prob_threshold": args.prob_threshold,
            "strong_threshold": args.strong_threshold,
            "station_bin_m": args.station_bin_m,
            "offset_bin_m": args.offset_bin_m,
            "offset_min_m": args.offset_min_m,
            "offset_max_m": args.offset_max_m,
            "gauge_tolerance_m": args.gauge_tolerance_m,
            "min_peak_score": args.min_peak_score,
            "max_link_gap_m": args.max_link_gap_m,
            "max_link_offset_m": args.max_link_offset_m,
            "max_link_slope": args.max_link_slope,
        },
        "extraction": extraction_stats,
        "sequence_count": len(sequences),
        "centerline_feature_count": len(centerline_features),
        "evidence_feature_count": len(evidence_features),
        "sequences": [sequence_summary(seq) for seq in sequences],
        "outputs": {
            "centerlines_geojson": str(centerline_geojson),
            "centerlines_shp": str(centerline_geojson.with_suffix(".shp")),
            "evidence_geojson": str(evidence_geojson),
            "evidence_shp": str(evidence_geojson.with_suffix(".shp")),
            "samples_csv": str(out_dir / "deeplab_gauge_pair_samples.csv"),
            "qa_crops": qa_summary,
            "summary_json": str(out_dir / "summary.json"),
        },
        "interpretation": "Candidate centerlines require paired DeepLab single-rail peaks separated by the LAS-derived gauge. DSM contrast is recorded as supporting evidence but is not a hard veto in this prototype.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
    return 1.6


def load_first_line(path: Path) -> list[tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates") or []]
        if len(coords) >= 2:
            return coords
    raise ValueError(f"No LineString found in {path}")


def build_edges(probability: np.ndarray, transform: Affine, guide: Guide, *, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    height, width = probability.shape
    corners = np.asarray(
        [
            pixel_to_world(transform, 0, 0),
            pixel_to_world(transform, width, 0),
            pixel_to_world(transform, 0, height),
            pixel_to_world(transform, width, height),
        ],
        dtype=float,
    )
    stations, _ = guide.station_offset(corners[:, 0], corners[:, 1])
    s_min = math.floor(float(stations.min()) / args.station_bin_m) * args.station_bin_m
    s_max = math.ceil(float(stations.max()) / args.station_bin_m) * args.station_bin_m
    station_edges = np.arange(s_min, s_max + args.station_bin_m * 1.5, args.station_bin_m, dtype=float)
    offset_edges = np.arange(args.offset_min_m, args.offset_max_m + args.offset_bin_m * 1.5, args.offset_bin_m, dtype=float)
    return station_edges, offset_edges


def load_dsm_evidence(path: Path, *, prob_bounds: Any, guide: Guide, station_edges: np.ndarray, args: argparse.Namespace) -> DsmEvidence:
    with rasterio.open(path) as dsm_ds:
        window = from_bounds(prob_bounds.left, prob_bounds.bottom, prob_bounds.right, prob_bounds.top, transform=dsm_ds.transform)
        window = clamp_window(dsm_ds, window, pad=2)
        array = dsm_ds.read(1, window=window).astype("float32")
        transform = dsm_ds.window_transform(window)
        nodata = dsm_ds.nodata
    ground = dsm_station_ground(array, transform, nodata=nodata, guide=guide, station_edges=station_edges, args=args)
    return DsmEvidence(path=path, transform=transform, array=array, nodata=nodata, station_ground_m=ground)


def dsm_station_ground(
    array: np.ndarray,
    transform: Affine,
    *,
    nodata: float | None,
    guide: Guide,
    station_edges: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    rows, cols = np.indices(array.shape, dtype="float32")
    xs, ys = pixel_arrays_to_world(transform, cols + 0.5, rows + 0.5)
    stations, offsets = guide.station_offset(xs, ys)
    values = array.astype("float32", copy=False)
    valid = np.isfinite(values) & (offsets >= args.offset_min_m) & (offsets <= args.offset_max_m)
    if nodata is not None:
        valid &= values != nodata
    station_index = np.floor((stations[valid] - station_edges[0]) / np.diff(station_edges[:2])[0]).astype(np.int64)
    z_values = values[valid]
    inside = (station_index >= 0) & (station_index < station_edges.size - 1)
    station_index = station_index[inside]
    z_values = z_values[inside]
    ground = np.full(station_edges.size - 1, np.nan, dtype="float32")
    for index in range(ground.size):
        local = z_values[station_index == index]
        if local.size:
            ground[index] = float(np.nanpercentile(local, 20))
    return ground


def extract_pair_samples(
    probability: np.ndarray,
    transform: Affine,
    guide: Guide,
    *,
    gauge_m: float,
    station_edges: np.ndarray,
    offset_edges: np.ndarray,
    dsm_evidence: DsmEvidence | None,
    args: argparse.Namespace,
) -> tuple[list[PairSample], dict[str, Any]]:
    threshold_u8 = threshold_to_u8(args.prob_threshold)
    rows, cols = np.nonzero(probability >= threshold_u8)
    if rows.size == 0:
        return [], {"probability_pixel_count": 0, "peak_count": 0, "raw_pair_count": 0}
    xs, ys = pixel_arrays_to_world(transform, cols.astype("float64") + 0.5, rows.astype("float64") + 0.5)
    stations, offsets = guide.station_offset(xs, ys)
    station_step = float(station_edges[1] - station_edges[0])
    offset_step = float(offset_edges[1] - offset_edges[0])
    station_index = np.floor((stations - station_edges[0]) / station_step).astype(np.int64)
    offset_index = np.floor((offsets - offset_edges[0]) / offset_step).astype(np.int64)
    valid = (
        (station_index >= 0)
        & (station_index < station_edges.size - 1)
        & (offset_index >= 0)
        & (offset_index < offset_edges.size - 1)
    )
    station_index = station_index[valid]
    offset_index = offset_index[valid]
    prob_values = probability[rows[valid], cols[valid]].astype("float32")
    weights = prob_values / 255.0

    hist = np.zeros((station_edges.size - 1, offset_edges.size - 1), dtype="float32")
    counts = np.zeros_like(hist, dtype="int32")
    prob_sum = np.zeros_like(hist, dtype="float32")
    np.add.at(hist, (station_index, offset_index), weights)
    np.add.at(counts, (station_index, offset_index), 1)
    np.add.at(prob_sum, (station_index, offset_index), prob_values)

    z_sum = None
    z_count = None
    if dsm_evidence is not None:
        z = sample_dsm(dsm_evidence, xs[valid], ys[valid])
        z_valid = np.isfinite(z)
        z_sum = np.zeros_like(hist, dtype="float32")
        z_count = np.zeros_like(hist, dtype="int32")
        np.add.at(z_sum, (station_index[z_valid], offset_index[z_valid]), z[z_valid])
        np.add.at(z_count, (station_index[z_valid], offset_index[z_valid]), 1)

    all_samples: list[PairSample] = []
    peak_count = 0
    raw_pair_count = 0
    sample_id = 1
    for s_index in range(hist.shape[0]):
        station_m = float((station_edges[s_index] + station_edges[s_index + 1]) / 2.0)
        peaks = find_lateral_peaks(
            hist[s_index],
            counts[s_index],
            prob_sum[s_index],
            offset_edges,
            station_index=s_index,
            station_m=station_m,
            min_peak_score=args.min_peak_score,
            z_sum=z_sum[s_index] if z_sum is not None else None,
            z_count=z_count[s_index] if z_count is not None else None,
            station_ground_m=(
                float(dsm_evidence.station_ground_m[s_index])
                if dsm_evidence is not None and np.isfinite(dsm_evidence.station_ground_m[s_index])
                else None
            ),
        )
        peak_count += len(peaks)
        pairs = pair_lateral_peaks(
            peaks,
            gauge_m=gauge_m,
            tolerance_m=args.gauge_tolerance_m,
            max_pairs=args.max_pairs_per_station,
        )
        raw_pair_count += len(pairs)
        for pair in pairs:
            all_samples.append(
                PairSample(
                    sample_id=sample_id,
                    station_index=s_index,
                    station_m=pair.station_m,
                    center_offset_m=pair.center_offset_m,
                    left_offset_m=pair.left_offset_m,
                    right_offset_m=pair.right_offset_m,
                    gauge_m=pair.gauge_m,
                    score=pair.score,
                    probability_score=pair.probability_score,
                    dsm_contrast_m=pair.dsm_contrast_m,
                    left_count=pair.left_count,
                    right_count=pair.right_count,
                )
            )
            sample_id += 1
    stats = {
        "probability_pixel_count": int(rows.size),
        "valid_corridor_pixel_count": int(valid.sum()),
        "peak_count": int(peak_count),
        "raw_pair_count": int(raw_pair_count),
        "station_bin_count": int(station_edges.size - 1),
        "offset_bin_count": int(offset_edges.size - 1),
    }
    return all_samples, stats


@dataclass(frozen=True)
class RawPair:
    station_m: float
    center_offset_m: float
    left_offset_m: float
    right_offset_m: float
    gauge_m: float
    score: float
    probability_score: float
    dsm_contrast_m: float | None
    left_count: int
    right_count: int


def find_lateral_peaks(
    values: np.ndarray,
    counts: np.ndarray,
    prob_sum: np.ndarray,
    offset_edges: np.ndarray,
    *,
    station_index: int,
    station_m: float,
    min_peak_score: float,
    z_sum: np.ndarray | None = None,
    z_count: np.ndarray | None = None,
    station_ground_m: float | None = None,
) -> list[LateralPeak]:
    if values.size < 3 or float(values.max()) < min_peak_score:
        return []
    smoothed = smooth_profile(values)
    peaks: list[LateralPeak] = []
    for idx in range(1, values.size - 1):
        if smoothed[idx] < min_peak_score:
            continue
        if smoothed[idx] < smoothed[idx - 1] or smoothed[idx] < smoothed[idx + 1]:
            continue
        lo = max(0, idx - 1)
        hi = min(values.size, idx + 2)
        local_weights = values[lo:hi]
        local_counts = counts[lo:hi]
        if int(local_counts.sum()) <= 0:
            continue
        centers = (offset_edges[lo:hi] + offset_edges[lo + 1 : hi + 1]) / 2.0
        offset_m = float(np.average(centers, weights=np.maximum(local_weights, 1e-6)))
        mean_probability = float(prob_sum[lo:hi].sum() / max(local_counts.sum(), 1) / 255.0)
        dsm_contrast = None
        if z_sum is not None and z_count is not None and station_ground_m is not None:
            local_z_count = int(z_count[lo:hi].sum())
            if local_z_count > 0:
                mean_z = float(z_sum[lo:hi].sum() / local_z_count)
                dsm_contrast = mean_z - station_ground_m
        peaks.append(
            LateralPeak(
                station_index=station_index,
                station_m=station_m,
                offset_m=offset_m,
                score=float(smoothed[idx]),
                count=int(local_counts.sum()),
                mean_probability=mean_probability,
                dsm_contrast_m=dsm_contrast,
            )
        )
    return dedupe_nearby_peaks(peaks, min_separation_m=0.24)


def smooth_profile(values: np.ndarray) -> np.ndarray:
    if values.size < 5:
        return values.astype("float32", copy=True)
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype="float32") / 9.0
    return np.convolve(values, kernel, mode="same")


def dedupe_nearby_peaks(peaks: list[LateralPeak], *, min_separation_m: float) -> list[LateralPeak]:
    selected: list[LateralPeak] = []
    for peak in sorted(peaks, key=lambda item: item.score, reverse=True):
        if all(abs(peak.offset_m - existing.offset_m) >= min_separation_m for existing in selected):
            selected.append(peak)
    return sorted(selected, key=lambda item: item.offset_m)


def pair_lateral_peaks(
    peaks: list[LateralPeak],
    *,
    gauge_m: float,
    tolerance_m: float,
    max_pairs: int,
) -> list[RawPair]:
    candidates: list[tuple[float, int, int, RawPair]] = []
    for left_index, left in enumerate(peaks):
        for right_index in range(left_index + 1, len(peaks)):
            right = peaks[right_index]
            gauge = right.offset_m - left.offset_m
            gauge_error = abs(gauge - gauge_m)
            if gauge_error > tolerance_m:
                continue
            probability_score = min(left.score, right.score) * max(0.05, 1.0 - gauge_error / max(tolerance_m, 1e-6))
            dsm_values = [value for value in (left.dsm_contrast_m, right.dsm_contrast_m) if value is not None and np.isfinite(value)]
            dsm_contrast = float(np.mean(dsm_values)) if dsm_values else None
            dsm_bonus = 1.0
            if dsm_contrast is not None:
                dsm_bonus += max(-0.1, min(0.2, dsm_contrast * 0.2))
            score = probability_score * dsm_bonus
            pair = RawPair(
                station_m=left.station_m,
                center_offset_m=(left.offset_m + right.offset_m) / 2.0,
                left_offset_m=left.offset_m,
                right_offset_m=right.offset_m,
                gauge_m=gauge,
                score=float(score),
                probability_score=float(probability_score),
                dsm_contrast_m=dsm_contrast,
                left_count=left.count,
                right_count=right.count,
            )
            candidates.append((score, left_index, right_index, pair))
    selected: list[RawPair] = []
    used_peaks: set[int] = set()
    for _, left_index, right_index, pair in sorted(candidates, key=lambda item: item[0], reverse=True):
        if left_index in used_peaks or right_index in used_peaks:
            continue
        selected.append(pair)
        used_peaks.add(left_index)
        used_peaks.add(right_index)
        if len(selected) >= max_pairs:
            break
    return sorted(selected, key=lambda item: item.center_offset_m)


def link_pair_samples(
    samples: list[PairSample],
    *,
    max_gap_m: float,
    max_offset_m: float,
    max_slope: float,
    min_samples: int,
    min_length_m: float,
) -> list[PairSequence]:
    by_station: dict[int, list[PairSample]] = {}
    for sample in samples:
        by_station.setdefault(sample.station_index, []).append(sample)

    active: list[list[PairSample]] = []
    finished: list[list[PairSample]] = []
    for station_index in sorted(by_station):
        station_samples = sorted(by_station[station_index], key=lambda item: item.score, reverse=True)
        still_active: list[list[PairSample]] = []
        for seq in active:
            if station_samples and station_samples[0].station_m - seq[-1].station_m <= max_gap_m:
                still_active.append(seq)
            else:
                finished.append(seq)
        active = still_active

        assigned_sequences: set[int] = set()
        assigned_samples: set[int] = set()
        for sample in station_samples:
            best_index = None
            best_cost = float("inf")
            for seq_index, seq in enumerate(active):
                if seq_index in assigned_sequences:
                    continue
                ds = sample.station_m - seq[-1].station_m
                if ds < 0 or ds > max_gap_m:
                    continue
                predicted = predict_next_offset(seq, sample.station_m)
                allowed = max_offset_m + max_slope * ds
                cost = abs(sample.center_offset_m - predicted)
                if cost <= allowed and cost < best_cost:
                    best_index = seq_index
                    best_cost = cost
            if best_index is None:
                active.append([sample])
                assigned_sequences.add(len(active) - 1)
            else:
                active[best_index].append(sample)
                assigned_sequences.add(best_index)
            assigned_samples.add(sample.sample_id)

        for sample in station_samples:
            if sample.sample_id not in assigned_samples:
                active.append([sample])
    finished.extend(active)

    filtered = [
        seq
        for seq in finished
        if len(seq) >= min_samples and seq[-1].station_m - seq[0].station_m >= min_length_m
    ]
    filtered.sort(key=lambda seq: (-len(seq), seq[0].station_m, float(np.mean([sample.center_offset_m for sample in seq]))))
    return [PairSequence(sequence_id=f"GP{index:02d}", samples=seq) for index, seq in enumerate(filtered, start=1)]


def predict_next_offset(seq: list[PairSample], station_m: float) -> float:
    if len(seq) < 2:
        return seq[-1].center_offset_m
    a = seq[-2]
    b = seq[-1]
    ds = max(b.station_m - a.station_m, 1e-6)
    slope = (b.center_offset_m - a.center_offset_m) / ds
    return b.center_offset_m + slope * (station_m - b.station_m)


def build_centerline_features(sequences: list[PairSequence], *, guide: Guide, gauge_m: float, smooth_window: int) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for sequence in sequences:
        stations = np.asarray([sample.station_m for sample in sequence.samples], dtype=float)
        offsets = smooth_offsets(np.asarray([sample.center_offset_m for sample in sequence.samples], dtype=float), smooth_window)
        coords = [guide.point_at(float(s), float(t)) for s, t in zip(stations, offsets)]
        gauges = np.asarray([sample.gauge_m for sample in sequence.samples], dtype=float)
        scores = np.asarray([sample.score for sample in sequence.samples], dtype=float)
        dsm_values = np.asarray(
            [sample.dsm_contrast_m for sample in sequence.samples if sample.dsm_contrast_m is not None and np.isfinite(sample.dsm_contrast_m)],
            dtype=float,
        )
        props = {
            "role": "deeplab_gauge_pair_centerline",
            "geom_kind": "gauge_paired_centerline",
            "source": "deeplab_v1_probability_gauge_pair",
            "seq_id": sequence.sequence_id,
            "sample_count": len(sequence.samples),
            "length_m": round(polyline_length(coords), 3),
            "s_min_m": round(float(stations.min()), 3),
            "s_max_m": round(float(stations.max()), 3),
            "t_min_m": round(float(offsets.min()), 3),
            "t_max_m": round(float(offsets.max()), 3),
            "mean_score": round(float(scores.mean()), 4),
            "mean_gauge_m": round(float(gauges.mean()), 4),
            "mean_gauge_error_m": round(float(np.mean(np.abs(gauges - gauge_m))), 4),
            "mean_dsm_contrast_m": round(float(dsm_values.mean()), 4) if dsm_values.size else 0.0,
            "dsm_sample_count": int(dsm_values.size),
        }
        features.append(line_feature(coords, props))
    return features


def build_pair_evidence_features(sequences: list[PairSequence], *, guide: Guide, every_n: int) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for sequence in sequences:
        for index, sample in enumerate(sequence.samples):
            if index % every_n != 0 and index != len(sequence.samples) - 1:
                continue
            left = guide.point_at(sample.station_m, sample.left_offset_m)
            right = guide.point_at(sample.station_m, sample.right_offset_m)
            props = {
                "role": "deeplab_gauge_pair_evidence",
                "geom_kind": "paired_rail_cross_section",
                "source": "deeplab_v1_probability_gauge_pair",
                "seq_id": sequence.sequence_id,
                "sample_id": sample.sample_id,
                "station_m": round(sample.station_m, 3),
                "center_t_m": round(sample.center_offset_m, 3),
                "gauge_m": round(sample.gauge_m, 4),
                "score": round(sample.score, 4),
                "prob_score": round(sample.probability_score, 4),
                "dsm_contrast_m": round(float(sample.dsm_contrast_m), 4) if sample.dsm_contrast_m is not None else 0.0,
                "left_count": sample.left_count,
                "right_count": sample.right_count,
            }
            features.append(line_feature([left, right], props))
    return features


def smooth_offsets(offsets: np.ndarray, window: int) -> np.ndarray:
    if offsets.size < 3 or window <= 1:
        return offsets
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed = np.empty_like(offsets)
    for index in range(offsets.size):
        lo = max(0, index - half)
        hi = min(offsets.size, index + half + 1)
        smoothed[index] = float(np.mean(offsets[lo:hi]))
    return smoothed


def write_qa_crops(
    dom_path: Path,
    *,
    probability_path: Path,
    centerline_features: list[dict[str, Any]],
    evidence_features: list[dict[str, Any]],
    out_dir: Path,
    crop_m: float,
    threshold: float,
    strong_threshold: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in out_dir.glob("*.png"):
        old_path.unlink()

    with rasterio.open(probability_path) as prob_ds:
        prob = prob_ds.read(1)
        prob_bounds = prob_ds.bounds
    with rasterio.open(dom_path) as dom_ds:
        full_window = clamp_window(dom_ds, from_bounds(prob_bounds.left, prob_bounds.bottom, prob_bounds.right, prob_bounds.top, dom_ds.transform), pad=0)
        rgb = read_rgb_window(dom_ds, full_window)
        transform = dom_ds.window_transform(full_window)
        prob_on_dom = resample_probability_to_window(prob, probability_path, transform, rgb.shape[:2])
        full_path = out_dir / "ta08_full_dom_deeplab_gauge_pair_overlay.png"
        save_overlay(full_path, rgb, transform, prob_on_dom, centerline_features, evidence_features, threshold=threshold, strong_threshold=strong_threshold, label="TA08 full gauge-pair overlay")

        crop_paths = [str(full_path)]
        targets = qa_targets(centerline_features)
        for name, x, y in REVIEW_POINTS:
            targets.insert(0, {"name": name, "label": name, "point": (x, y)})
        for target in targets[:12]:
            x, y = target["point"]
            window = square_window(dom_ds, x, y, crop_m)
            if window.width <= 1 or window.height <= 1:
                continue
            crop_rgb = read_rgb_window(dom_ds, window)
            crop_transform = dom_ds.window_transform(window)
            crop_prob = resample_probability_to_window(prob, probability_path, crop_transform, crop_rgb.shape[:2])
            path = out_dir / f"{sanitize_filename(target['name'])}_overlay.png"
            save_overlay(path, crop_rgb, crop_transform, crop_prob, centerline_features, evidence_features, threshold=threshold, strong_threshold=strong_threshold, label=str(target["label"]))
            crop_paths.append(str(path))
    index = {"count": len(crop_paths), "paths": crop_paths}
    (out_dir / "qa_crops_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def resample_probability_to_window(probability: np.ndarray, probability_path: Path, dst_transform: Affine, dst_shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(probability_path) as prob_ds:
        dst = np.zeros(dst_shape, dtype="uint8")
        reproject(
            source=probability,
            destination=dst,
            src_transform=prob_ds.transform,
            src_crs=prob_ds.crs,
            dst_transform=dst_transform,
            dst_crs=prob_ds.crs,
            resampling=Resampling.nearest,
        )
        return dst


def save_overlay(
    path: Path,
    rgb: np.ndarray,
    transform: Affine,
    probability: np.ndarray,
    centerline_features: list[dict[str, Any]],
    evidence_features: list[dict[str, Any]],
    *,
    threshold: float,
    strong_threshold: float,
    label: str,
) -> None:
    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype="uint8")
    weak = probability >= threshold_to_u8(threshold)
    strong = probability >= threshold_to_u8(strong_threshold)
    overlay[weak] = np.asarray([210, 40, 40, 95], dtype="uint8")
    overlay[strong] = np.asarray([255, 210, 0, 140], dtype="uint8")
    base.alpha_composite(Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(base, "RGBA")
    draw_features(draw, evidence_features, transform, width_px=2, fallback_color=(255, 255, 255, 190))
    draw_features(draw, centerline_features, transform, width_px=7, fallback_color=(0, 210, 255, 245))
    draw_review_points(draw, transform)
    draw_label(draw, label)
    base.convert("RGB").save(path)


def qa_targets(centerline_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for feature in centerline_features:
        coords = line_coords(feature)
        if len(coords) < 2:
            continue
        props = feature.get("properties") or {}
        mid = coords[len(coords) // 2]
        targets.append({"name": f"{props.get('seq_id', 'seq')}_mid", "label": f"{props.get('seq_id', '')} {props.get('t_min_m', '')}..{props.get('t_max_m', '')}", "point": mid})
    return targets


def sample_dsm(dsm: DsmEvidence, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    cols, rows = ~dsm.transform * (xs, ys)
    cols_i = np.rint(cols).astype(np.int64)
    rows_i = np.rint(rows).astype(np.int64)
    valid = (rows_i >= 0) & (rows_i < dsm.array.shape[0]) & (cols_i >= 0) & (cols_i < dsm.array.shape[1])
    out = np.full(xs.shape, np.nan, dtype="float32")
    values = dsm.array[rows_i[valid], cols_i[valid]]
    valid_values = np.isfinite(values)
    if dsm.nodata is not None:
        valid_values &= values != dsm.nodata
    tmp = out[valid]
    tmp[valid_values] = values[valid_values]
    out[valid] = tmp
    return out


def pixel_to_world(transform: Affine, col: float, row: float) -> tuple[float, float]:
    x, y = transform * (col, row)
    return float(x), float(y)


def pixel_arrays_to_world(transform: Affine, cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = transform.a * cols + transform.b * rows + transform.c
    ys = transform.d * cols + transform.e * rows + transform.f
    return xs, ys


def clamp_window(dataset: Any, window: Window, *, pad: int) -> Window:
    col_off = max(0, int(math.floor(window.col_off)) - pad)
    row_off = max(0, int(math.floor(window.row_off)) - pad)
    col_max = min(dataset.width, int(math.ceil(window.col_off + window.width)) + pad)
    row_max = min(dataset.height, int(math.ceil(window.row_off + window.height)) + pad)
    return Window(col_off, row_off, max(0, col_max - col_off), max(0, row_max - row_off))


def square_window(dataset: Any, x: float, y: float, crop_m: float) -> Window:
    pixel_width = max(abs(float(dataset.transform.a)), 1e-6)
    pixel_height = max(abs(float(dataset.transform.e)), 1e-6)
    width_px = max(64, int(math.ceil(crop_m / pixel_width)))
    height_px = max(64, int(math.ceil(crop_m / pixel_height)))
    row, col = dataset.index(x, y)
    col_off = max(0, min(dataset.width - 1, col - width_px // 2))
    row_off = max(0, min(dataset.height - 1, row - height_px // 2))
    return Window(col_off, row_off, min(width_px, dataset.width - col_off), min(height_px, dataset.height - row_off))


def read_rgb_window(dataset: Any, window: Window) -> np.ndarray:
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


def draw_features(draw: ImageDraw.ImageDraw, features: list[dict[str, Any]], transform: Affine, *, width_px: int, fallback_color: tuple[int, int, int, int]) -> None:
    for feature in features:
        coords = []
        for x, y in line_coords(feature):
            col, row = ~transform * (x, y)
            coords.append((col, row))
        if len(coords) < 2:
            continue
        props = feature.get("properties") or {}
        color = color_for_sequence(str(props.get("seq_id", "")), fallback_color)
        draw.line(coords, fill=color, width=width_px, joint="curve")


def color_for_sequence(seq_id: str, fallback: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    palette = [
        (0, 210, 255, 245),
        (0, 190, 120, 245),
        (255, 120, 30, 245),
        (190, 80, 255, 245),
        (255, 60, 120, 245),
    ]
    if seq_id.startswith("GP"):
        try:
            return palette[(int(seq_id[2:]) - 1) % len(palette)]
        except ValueError:
            return fallback
    return fallback


def draw_review_points(draw: ImageDraw.ImageDraw, transform: Affine) -> None:
    for index, (_, x, y) in enumerate(REVIEW_POINTS, start=1):
        col, row = ~transform * (x, y)
        radius = 8
        draw.ellipse([col - radius, row - radius, col + radius, row + radius], fill=(35, 85, 255, 230), outline=(255, 255, 255, 255), width=2)
        draw.text((col + radius + 3, row - radius - 2), str(index), fill=(255, 255, 255, 255))


def draw_label(draw: ImageDraw.ImageDraw, label: str) -> None:
    draw.rectangle([8, 8, 760, 42], fill=(0, 0, 0, 190))
    draw.text((18, 17), label[:96], fill=(255, 255, 255, 255))


def line_feature(coords: list[tuple[float, float]], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "LineString", "coordinates": [[round(float(x), 6), round(float(y), 6)] for x, y in coords]},
    }


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, *_ in feature["geometry"]["coordinates"]]


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(coords, coords[1:]))


def sequence_summary(sequence: PairSequence) -> dict[str, Any]:
    gauges = np.asarray([sample.gauge_m for sample in sequence.samples], dtype=float)
    scores = np.asarray([sample.score for sample in sequence.samples], dtype=float)
    offsets = np.asarray([sample.center_offset_m for sample in sequence.samples], dtype=float)
    dsm_values = np.asarray([sample.dsm_contrast_m for sample in sequence.samples if sample.dsm_contrast_m is not None], dtype=float)
    return {
        "seq_id": sequence.sequence_id,
        "sample_count": len(sequence.samples),
        "s_min_m": round(sequence.samples[0].station_m, 3),
        "s_max_m": round(sequence.samples[-1].station_m, 3),
        "t_min_m": round(float(offsets.min()), 3),
        "t_max_m": round(float(offsets.max()), 3),
        "mean_score": round(float(scores.mean()), 4),
        "mean_gauge_m": round(float(gauges.mean()), 4),
        "mean_dsm_contrast_m": round(float(dsm_values.mean()), 4) if dsm_values.size else 0.0,
        "dsm_sample_count": int(dsm_values.size),
    }


def pair_sample_rows(sequences: list[PairSequence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        for sample in sequence.samples:
            rows.append(
                {
                    "seq_id": sequence.sequence_id,
                    "sample_id": sample.sample_id,
                    "station_m": round(sample.station_m, 3),
                    "center_offset_m": round(sample.center_offset_m, 4),
                    "left_offset_m": round(sample.left_offset_m, 4),
                    "right_offset_m": round(sample.right_offset_m, 4),
                    "gauge_m": round(sample.gauge_m, 4),
                    "score": round(sample.score, 4),
                    "probability_score": round(sample.probability_score, 4),
                    "dsm_contrast_m": round(float(sample.dsm_contrast_m), 4) if sample.dsm_contrast_m is not None else "",
                    "left_count": sample.left_count,
                    "right_count": sample.right_count,
                }
            )
    return rows


def write_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}}, "features": features}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_line_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("role", "C", size=32)
    writer.field("kind", "C", size=32)
    writer.field("seq_id", "C", size=12)
    writer.field("source", "C", size=48)
    writer.field("samples", "N", size=8)
    writer.field("len_m", "F", decimal=3)
    writer.field("gauge_m", "F", decimal=4)
    writer.field("score", "F", decimal=4)
    writer.field("dsm_m", "F", decimal=4)
    writer.field("s_min", "F", decimal=3)
    writer.field("s_max", "F", decimal=3)
    writer.field("t_min", "F", decimal=3)
    writer.field("t_max", "F", decimal=3)
    for feature in features:
        props = feature.get("properties") or {}
        coords = line_coords(feature)
        writer.line([coords])
        writer.record(
            str(props.get("role", ""))[:32],
            str(props.get("geom_kind", ""))[:32],
            str(props.get("seq_id", ""))[:12],
            str(props.get("source", ""))[:48],
            safe_int(props.get("sample_count", props.get("sample_id", 0))),
            safe_float(props.get("length_m", polyline_length(coords))),
            safe_float(props.get("mean_gauge_m", props.get("gauge_m", 0.0))),
            safe_float(props.get("mean_score", props.get("score", 0.0))),
            safe_float(props.get("mean_dsm_contrast_m", props.get("dsm_contrast_m", 0.0))),
            safe_float(props.get("s_min_m", props.get("station_m", 0.0))),
            safe_float(props.get("s_max_m", props.get("station_m", 0.0))),
            safe_float(props.get("t_min_m", props.get("center_t_m", 0.0))),
            safe_float(props.get("t_max_m", props.get("center_t_m", 0.0))),
        )
    writer.close()
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    write_projection(output_path.with_suffix(".prj"), epsg)


def write_projection(path: Path, epsg: int) -> None:
    path.write_text(rasterio.crs.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")


def write_centerline_qml(path: Path) -> None:
    write_simple_qml(path, color="0,210,255,255", width_mm=0.85)


def write_evidence_qml(path: Path) -> None:
    write_simple_qml(path, color="255,255,255,210", width_mm=0.35)


def write_simple_qml(path: Path, *, color: str, width_mm: float) -> None:
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def threshold_to_u8(value: float) -> int:
    return int(math.ceil(max(0.0, min(1.0, value)) * 255.0))


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def sanitize_filename(name: str) -> str:
    chars = []
    for char in name:
        if char.isalnum() or char in ("-", "_"):
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "crop"


if __name__ == "__main__":
    raise SystemExit(main())
