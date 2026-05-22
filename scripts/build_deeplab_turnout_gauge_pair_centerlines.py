#!/usr/bin/env python3
"""Build DeepLab gauge-pair centerline evidence for all turnout windows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.windows import Window, from_bounds

import build_deeplab_gauge_pair_centerlines as gauge_pair
import export_segmentation_evidence_overlay as overlay


DEFAULT_TILE_INDEX = Path("output/raw_dom_roi_fullpass_v1/raw_dom_roi_tiles/selected_tile_index.csv")
DEFAULT_PROB_DIR = Path("output/raw_dom_roi_fullpass_v1/rail_predictions_deeplab_v1_thr050/probabilities")
DEFAULT_TURNOUTS = Path("output/raw_dom_roi_fullpass_v1/turnout_template_connectors/turnout_template_connector_proposals.geojson")
DEFAULT_MAINLINE = Path("output/raw_dom_roi_fullpass_v1/mainline_prior/mainline_2_track_connected.geojson")
DEFAULT_DOM = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u65e0\u4eba\u673a\u6570\u636e" / "\u6b63\u5c04" / "dom.tif"
DEFAULT_DSM = Path("D:/") / "\u6b63\u5c04" / "lidars" / "terra_dsm" / "dsm.tif"
DEFAULT_GAUGE_SUMMARY = Path("output/handheld_las_constraints_fullpass_switch_excluded/summary.json")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_turnouts_v1")
DEFAULT_EPSG = 32651


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build reusable DeepLab paired-rail centerline evidence around every turnout candidate. "
            "The output is consumed by topology post-processing; it is not tied to one review point."
        )
    )
    parser.add_argument("--tile-index", type=Path, default=DEFAULT_TILE_INDEX)
    parser.add_argument("--prob-dir", type=Path, default=DEFAULT_PROB_DIR)
    parser.add_argument("--turnouts", type=Path, default=DEFAULT_TURNOUTS)
    parser.add_argument("--mainline", type=Path, default=DEFAULT_MAINLINE)
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--dsm", type=Path, default=DEFAULT_DSM)
    parser.add_argument("--gauge-summary", type=Path, default=DEFAULT_GAUGE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--branch-ids", default="", help="Optional comma-separated anchor/branch ids to process.")
    parser.add_argument("--padding-m", type=float, default=55.0)
    parser.add_argument("--offset-padding-m", type=float, default=2.5)
    parser.add_argument("--min-offset-span-m", type=float, default=7.0)
    parser.add_argument("--min-covered-fraction", type=float, default=0.25)
    parser.add_argument("--max-window-pixels", type=int, default=85_000_000)
    parser.add_argument("--gauge-m", type=float, default=0.0)
    parser.add_argument("--gauge-tolerance-m", type=float, default=0.22)
    parser.add_argument("--prob-threshold", type=float, default=0.70)
    parser.add_argument("--station-bin-m", type=float, default=0.75)
    parser.add_argument("--offset-bin-m", type=float, default=0.06)
    parser.add_argument("--min-peak-score", type=float, default=5.0)
    parser.add_argument("--max-pairs-per-station", type=int, default=4)
    parser.add_argument("--max-link-gap-m", type=float, default=4.0)
    parser.add_argument("--max-link-offset-m", type=float, default=0.55)
    parser.add_argument("--max-link-slope", type=float, default=0.18)
    parser.add_argument("--min-sequence-samples", type=int, default=10)
    parser.add_argument("--min-sequence-length-m", type=float, default=8.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--evidence-every-n", type=int, default=2)
    parser.add_argument("--no-dsm", action="store_true")
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    turnouts_path = args.turnouts.expanduser().resolve()
    tile_index_path = args.tile_index.expanduser().resolve()
    prob_dir = args.prob_dir.expanduser().resolve()
    dom_path = args.dom.expanduser().resolve()
    dsm_path = args.dsm.expanduser().resolve()
    guide = gauge_pair.Guide.from_coords(gauge_pair.load_first_line(args.mainline.expanduser().resolve()))
    gauge_m = gauge_pair.resolve_gauge(args.gauge_m, args.gauge_summary.expanduser().resolve())
    selected_ids = parse_branch_ids(args.branch_ids)
    turnout_features = load_turnout_features(turnouts_path, selected_ids=selected_ids)
    tile_rows = overlay.load_tile_index(tile_index_path)

    centerline_features: list[dict[str, Any]] = []
    evidence_features: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    branch_summaries: list[dict[str, Any]] = []

    with rasterio.open(dom_path) as dom_ds:
        for feature in turnout_features:
            result = extract_branch_gauge_pairs(
                feature,
                dom_ds=dom_ds,
                tile_rows=tile_rows,
                prob_dir=prob_dir,
                dsm_path=dsm_path,
                guide=guide,
                gauge_m=gauge_m,
                args=args,
            )
            branch_summaries.append(result["summary"])
            centerline_features.extend(result["centerline_features"])
            evidence_features.extend(result["evidence_features"])
            sample_rows.extend(result["sample_rows"])
        crs = dom_ds.crs

    centerline_geojson = out_dir / "deeplab_gauge_pair_centerlines.geojson"
    evidence_geojson = out_dir / "deeplab_gauge_pair_evidence.geojson"
    gauge_pair.write_geojson(centerline_geojson, centerline_features, epsg=args.epsg)
    gauge_pair.write_geojson(evidence_geojson, evidence_features, epsg=args.epsg)
    gauge_pair.write_line_shapefile(centerline_features, centerline_geojson.with_suffix(".shp"), epsg=args.epsg)
    gauge_pair.write_line_shapefile(evidence_features, evidence_geojson.with_suffix(".shp"), epsg=args.epsg)
    gauge_pair.write_centerline_qml(centerline_geojson.with_suffix(".qml"))
    gauge_pair.write_evidence_qml(evidence_geojson.with_suffix(".qml"))
    gauge_pair.write_csv(out_dir / "deeplab_gauge_pair_samples.csv", sample_rows)

    summary = {
        "mode": "deeplab_turnout_window_gauge_pair_filtering",
        "purpose": "Reusable paired-rail DeepLab evidence for turnout topology refinement across all turnout windows.",
        "inputs": {
            "tile_index": str(tile_index_path),
            "probability_dir": str(prob_dir),
            "turnouts": str(turnouts_path),
            "mainline": str(args.mainline.expanduser().resolve()),
            "dom": str(dom_path),
            "dsm": str(dsm_path) if not args.no_dsm and dsm_path.exists() else None,
        },
        "crs": str(crs),
        "gauge_m": gauge_m,
        "parameters": {
            "padding_m": args.padding_m,
            "offset_padding_m": args.offset_padding_m,
            "min_offset_span_m": args.min_offset_span_m,
            "prob_threshold": args.prob_threshold,
            "gauge_tolerance_m": args.gauge_tolerance_m,
            "station_bin_m": args.station_bin_m,
            "offset_bin_m": args.offset_bin_m,
            "min_peak_score": args.min_peak_score,
            "max_link_gap_m": args.max_link_gap_m,
            "max_link_offset_m": args.max_link_offset_m,
            "max_link_slope": args.max_link_slope,
            "min_covered_fraction": args.min_covered_fraction,
        },
        "branch_count": len(turnout_features),
        "centerline_feature_count": len(centerline_features),
        "evidence_feature_count": len(evidence_features),
        "branches": branch_summaries,
        "outputs": {
            "centerlines_geojson": str(centerline_geojson),
            "centerlines_shp": str(centerline_geojson.with_suffix(".shp")),
            "evidence_geojson": str(evidence_geojson),
            "evidence_shp": str(evidence_geojson.with_suffix(".shp")),
            "samples_csv": str(out_dir / "deeplab_gauge_pair_samples.csv"),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_branch_ids(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def load_turnout_features(path: Path, *, selected_ids: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, feature in enumerate(payload.get("features", []), start=1):
        coords = gauge_pair.line_coords(feature)
        if len(coords) < 2:
            continue
        branch_id = branch_id_for_feature(feature) or f"BR{index:02d}"
        if selected_ids and branch_id not in selected_ids:
            continue
        if branch_id in seen:
            continue
        seen.add(branch_id)
        cloned = json.loads(json.dumps(feature))
        cloned.setdefault("properties", {})["branch_id"] = branch_id
        features.append(cloned)
    return features


def branch_id_for_feature(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    for key in ("branch_id", "anchor_id", "connector_id", "candidate_id", "id"):
        value = str(props.get(key, "")).strip()
        if not value:
            continue
        if key in {"connector_id", "candidate_id"} and "_" in value:
            return value.split("_", 1)[0]
        return value
    return ""


def extract_branch_gauge_pairs(
    feature: dict[str, Any],
    *,
    dom_ds: Any,
    tile_rows: list[Any],
    prob_dir: Path,
    dsm_path: Path,
    guide: gauge_pair.Guide,
    gauge_m: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    branch_id = branch_id_for_feature(feature)
    coords = gauge_pair.line_coords(feature)
    window = build_feature_window(dom_ds, coords, padding_m=args.padding_m)
    window_pixels = int(window.width) * int(window.height)
    if window_pixels > args.max_window_pixels:
        return empty_branch_result(branch_id, "window_too_large", window=window, window_pixels=window_pixels)

    prob_u8, coverage_count, used_tiles = overlay.mosaic_probability_tiles(window, tile_rows, prob_dir)
    covered_fraction = float(np.count_nonzero(coverage_count) / coverage_count.size) if coverage_count.size else 0.0
    if covered_fraction < args.min_covered_fraction:
        return empty_branch_result(
            branch_id,
            "insufficient_prediction_tile_coverage",
            window=window,
            window_pixels=window_pixels,
            used_tiles=used_tiles,
            covered_fraction=covered_fraction,
        )

    transform = dom_ds.window_transform(window)
    extraction_args = extraction_args_for_feature(args, feature, guide=guide)
    station_edges, offset_edges = gauge_pair.build_edges(prob_u8, transform, guide, args=extraction_args)
    bounds = bounds_object(prob_u8, transform)
    dsm_evidence = None
    if not args.no_dsm and dsm_path.exists():
        dsm_evidence = gauge_pair.load_dsm_evidence(dsm_path, prob_bounds=bounds, guide=guide, station_edges=station_edges, args=extraction_args)
    samples, extraction_stats = gauge_pair.extract_pair_samples(
        prob_u8,
        transform,
        guide,
        gauge_m=gauge_m,
        station_edges=station_edges,
        offset_edges=offset_edges,
        dsm_evidence=dsm_evidence,
        args=extraction_args,
    )
    sequences = gauge_pair.link_pair_samples(
        samples,
        max_gap_m=args.max_link_gap_m,
        max_offset_m=args.max_link_offset_m,
        max_slope=args.max_link_slope,
        min_samples=args.min_sequence_samples,
        min_length_m=args.min_sequence_length_m,
    )
    centerline_features = gauge_pair.build_centerline_features(sequences, guide=guide, gauge_m=gauge_m, smooth_window=args.smooth_window)
    evidence_features = gauge_pair.build_pair_evidence_features(sequences, guide=guide, every_n=max(1, args.evidence_every_n))
    tag_branch_features(centerline_features, branch_id=branch_id)
    tag_branch_features(evidence_features, branch_id=branch_id)
    rows = []
    for row in gauge_pair.pair_sample_rows(sequences):
        rows.append({"branch_id": branch_id, **row, "seq_id": f"{branch_id}_{row['seq_id']}"})
    summary = {
        "branch_id": branch_id,
        "status": "processed",
        "window": window_summary(window),
        "covered_fraction": covered_fraction,
        "used_tile_count": len(used_tiles),
        "offset_min_m": extraction_args.offset_min_m,
        "offset_max_m": extraction_args.offset_max_m,
        "extraction": extraction_stats,
        "sequence_count": len(sequences),
        "centerline_feature_count": len(centerline_features),
        "evidence_feature_count": len(evidence_features),
        "sequences": [gauge_pair.sequence_summary(sequence) for sequence in sequences],
    }
    return {
        "centerline_features": centerline_features,
        "evidence_features": evidence_features,
        "sample_rows": rows,
        "summary": summary,
    }


def extraction_args_for_feature(args: argparse.Namespace, feature: dict[str, Any], *, guide: gauge_pair.Guide) -> SimpleNamespace:
    coords = gauge_pair.line_coords(feature)
    offsets = [guide.station_offset_one(point)[1] for point in coords]
    offset_min = min(offsets) - args.offset_padding_m
    offset_max = max(offsets) + args.offset_padding_m
    span = offset_max - offset_min
    if span < args.min_offset_span_m:
        pad = (args.min_offset_span_m - span) / 2.0
        offset_min -= pad
        offset_max += pad
    return SimpleNamespace(
        prob_threshold=args.prob_threshold,
        station_bin_m=args.station_bin_m,
        offset_bin_m=args.offset_bin_m,
        offset_min_m=float(offset_min),
        offset_max_m=float(offset_max),
        min_peak_score=args.min_peak_score,
        gauge_tolerance_m=args.gauge_tolerance_m,
        max_pairs_per_station=args.max_pairs_per_station,
    )


def tag_branch_features(features: list[dict[str, Any]], *, branch_id: str) -> None:
    for feature in features:
        props = feature.setdefault("properties", {})
        old_seq = str(props.get("seq_id", "GP"))
        props["branch_id"] = branch_id
        props["source_branch_id"] = branch_id
        props["source"] = "deeplab_turnout_window_gauge_pair"
        props["seq_id"] = f"{branch_id}_{old_seq}"


def build_feature_window(dataset: Any, coords: list[tuple[float, float]], *, padding_m: float) -> Window:
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    if not xs or not ys:
        raise ValueError("No feature coordinates are available.")
    window = from_bounds(
        min(xs) - padding_m,
        min(ys) - padding_m,
        max(xs) + padding_m,
        max(ys) + padding_m,
        transform=dataset.transform,
    )
    return overlay.clamp_window(dataset, window)


def bounds_object(array: np.ndarray, transform: Any) -> SimpleNamespace:
    west, south, east, north = array_bounds(array.shape[0], array.shape[1], transform)
    return SimpleNamespace(left=west, bottom=south, right=east, top=north)


def empty_branch_result(
    branch_id: str,
    status: str,
    *,
    window: Window,
    window_pixels: int,
    used_tiles: list[str] | None = None,
    covered_fraction: float | None = None,
) -> dict[str, Any]:
    return {
        "centerline_features": [],
        "evidence_features": [],
        "sample_rows": [],
        "summary": {
            "branch_id": branch_id,
            "status": status,
            "window": window_summary(window),
            "window_pixels": window_pixels,
            "covered_fraction": covered_fraction,
            "used_tile_count": len(used_tiles or []),
            "sequence_count": 0,
            "centerline_feature_count": 0,
            "evidence_feature_count": 0,
        },
    }


def window_summary(window: Window) -> dict[str, int]:
    return {
        "row_off": int(math.floor(window.row_off)),
        "col_off": int(math.floor(window.col_off)),
        "height": int(math.ceil(window.height)),
        "width": int(math.ceil(window.width)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
