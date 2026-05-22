from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.windows import Window, from_bounds


DEFAULT_DOM = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u65e0\u4eba\u673a\u6570\u636e" / "\u6b63\u5c04" / "dom.tif"
DEFAULT_TILE_INDEX = Path("output/raw_dom_roi_fullpass_v1/raw_dom_roi_tiles/selected_tile_index.csv")
DEFAULT_PROB_DIR = Path("output/raw_dom_roi_fullpass_v1/rail_predictions/probabilities")
DEFAULT_BRANCHES = Path("output/raw_dom_roi_fullpass_v1/all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/segmentation_evidence_overlay_ta08")

TA08_REVIEW_POINTS = [
    ("coord01_user_ok", 315349.015, 3520808.310),
    ("coord02_user_deviates", 315341.558, 3520778.002),
    ("coord03_user_intersects_rail", 315333.927, 3520749.822),
]


@dataclass(frozen=True)
class TileRow:
    image_name: str
    row_off: int
    col_off: int
    height: int
    width: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export georeferenced segmentation evidence around a turnout from existing raw-DOM prediction tiles."
    )
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--tile-index", type=Path, default=DEFAULT_TILE_INDEX)
    parser.add_argument("--prob-dir", type=Path, default=DEFAULT_PROB_DIR)
    parser.add_argument("--branches", type=Path, default=DEFAULT_BRANCHES)
    parser.add_argument("--branch-id", default="TA08")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--padding-m", type=float, default=55.0)
    parser.add_argument("--point-crop-m", type=float, default=54.0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--weak-threshold", type=float, default=0.50)
    parser.add_argument("--stats-radius-m", type=float, default=5.0)
    parser.add_argument("--line-stats-radius-m", type=float, default=2.2)
    parser.add_argument("--line-samples", type=int, default=18)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dom_path = args.dom.expanduser().resolve()
    tile_index_path = args.tile_index.expanduser().resolve()
    prob_dir = args.prob_dir.expanduser().resolve()
    branches_path = args.branches.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = out_dir / "qa_crops"
    qa_dir.mkdir(exist_ok=True)

    line_coords = load_branch_line(branches_path, args.branch_id)
    review_points = list(TA08_REVIEW_POINTS)

    with rasterio.open(dom_path) as dataset:
        window = build_review_window(dataset, review_points, line_coords, padding_m=args.padding_m)
        transform = dataset.window_transform(window)
        rgb = read_rgb_window(dataset, window)
        tile_rows = load_tile_index(tile_index_path)
        prob_u8, coverage_count, used_tiles = mosaic_probability_tiles(window, tile_rows, prob_dir)

        outputs = write_geotiff_outputs(
            out_dir=out_dir,
            rgb=rgb,
            prob_u8=prob_u8,
            coverage_count=coverage_count,
            transform=transform,
            crs=dataset.crs,
            threshold=args.threshold,
            weak_threshold=args.weak_threshold,
        )

        qa_outputs = write_qa_crops(
            qa_dir=qa_dir,
            rgb=rgb,
            prob_u8=prob_u8,
            transform=transform,
            review_points=review_points,
            line_coords=line_coords,
            point_crop_m=args.point_crop_m,
            threshold=args.threshold,
            weak_threshold=args.weak_threshold,
        )

        point_stats = [
            point_probability_stats(
                name=name,
                x=x,
                y=y,
                prob_u8=prob_u8,
                coverage_count=coverage_count,
                transform=transform,
                radius_m=args.stats_radius_m,
                thresholds=(args.weak_threshold, args.threshold),
            )
            for name, x, y in review_points
        ]
        line_stats = line_probability_samples(
            line_coords=line_coords,
            prob_u8=prob_u8,
            coverage_count=coverage_count,
            transform=transform,
            radius_m=args.line_stats_radius_m,
            sample_count=args.line_samples,
            thresholds=(args.weak_threshold, args.threshold),
        )

        summary = {
            "mode": "segmentation_evidence_overlay",
            "branch_id": args.branch_id,
            "purpose": "Georeferenced evidence layer showing what the semantic segmentation model recognized before centerline post-processing.",
            "inputs": {
                "dom": str(dom_path),
                "tile_index": str(tile_index_path),
                "probability_dir": str(prob_dir),
                "branch_geojson": str(branches_path),
            },
            "crop": {
                "row_off": int(window.row_off),
                "col_off": int(window.col_off),
                "height": int(window.height),
                "width": int(window.width),
                "bounds": raster_bounds(transform, int(window.width), int(window.height)),
                "padding_m": args.padding_m,
                "pixel_size_m": [abs(float(transform.a)), abs(float(transform.e))],
            },
            "thresholds": {
                "weak_threshold": args.weak_threshold,
                "strong_threshold": args.threshold,
                "strong_threshold_u8": threshold_to_u8(args.threshold),
                "weak_threshold_u8": threshold_to_u8(args.weak_threshold),
            },
            "tile_mosaic": {
                "used_tile_count": len(used_tiles),
                "used_tiles": used_tiles,
                "covered_pixel_fraction": float(np.count_nonzero(coverage_count) / coverage_count.size),
                "max_overlap_count": int(coverage_count.max()) if coverage_count.size else 0,
            },
            "outputs": outputs | {"qa_crops": qa_outputs, "summary_json": str(out_dir / "summary.json")},
            "review_points": point_stats,
            "line_samples": line_stats,
        }

    write_csv(out_dir / "ta08_review_point_segmentation_stats.csv", point_stats)
    write_csv(out_dir / "ta08_line_segmentation_samples.csv", line_stats)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_branch_line(path: Path, branch_id: str) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        if str(props.get("branch_id", props.get("anchor_id", ""))) != branch_id and str(props.get("anchor_id", "")) != branch_id:
            continue
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") == "LineString":
            return [(float(x), float(y)) for x, y, *_ in coords]
        if geom.get("type") == "MultiLineString" and coords:
            return [(float(x), float(y)) for x, y, *_ in coords[0]]
    return []


def build_review_window(
    dataset: Any,
    review_points: list[tuple[str, float, float]],
    line_coords: list[tuple[float, float]],
    *,
    padding_m: float,
) -> Window:
    xs = [x for _, x, _ in review_points] + [x for x, _ in line_coords]
    ys = [y for _, _, y in review_points] + [y for _, y in line_coords]
    if not xs or not ys:
        raise ValueError("No review coordinates or branch line coordinates are available.")
    window = from_bounds(
        min(xs) - padding_m,
        min(ys) - padding_m,
        max(xs) + padding_m,
        max(ys) + padding_m,
        transform=dataset.transform,
    )
    return clamp_window(dataset, window)


def clamp_window(dataset: Any, window: Window) -> Window:
    col_off = max(0, int(math.floor(window.col_off)))
    row_off = max(0, int(math.floor(window.row_off)))
    col_max = min(dataset.width, int(math.ceil(window.col_off + window.width)))
    row_max = min(dataset.height, int(math.ceil(window.row_off + window.height)))
    width = col_max - col_off
    height = row_max - row_off
    if width <= 1 or height <= 1:
        raise ValueError(f"Invalid crop window: {window}")
    return Window(col_off, row_off, width, height)


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


def load_tile_index(path: Path) -> list[TileRow]:
    rows: list[TileRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                TileRow(
                    image_name=str(row["image_name"]),
                    row_off=int(float(row["row_off"])),
                    col_off=int(float(row["col_off"])),
                    width=int(float(row["width"])),
                    height=int(float(row["height"])),
                )
            )
    return rows


def mosaic_probability_tiles(window: Window, tile_rows: list[TileRow], prob_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    crop_row0 = int(window.row_off)
    crop_col0 = int(window.col_off)
    crop_row1 = crop_row0 + int(window.height)
    crop_col1 = crop_col0 + int(window.width)
    probability_sum = np.zeros((int(window.height), int(window.width)), dtype=np.uint32)
    coverage_count = np.zeros((int(window.height), int(window.width)), dtype=np.uint16)
    used_tiles: list[str] = []

    for tile in tile_rows:
        tile_row1 = tile.row_off + tile.height
        tile_col1 = tile.col_off + tile.width
        row0 = max(crop_row0, tile.row_off)
        row1 = min(crop_row1, tile_row1)
        col0 = max(crop_col0, tile.col_off)
        col1 = min(crop_col1, tile_col1)
        if row1 <= row0 or col1 <= col0:
            continue
        prob_path = prob_dir / Path(tile.image_name).with_suffix(".png").name
        if not prob_path.exists():
            continue
        with Image.open(prob_path) as image:
            prob = np.asarray(image.convert("L"), dtype=np.uint16)
        actual_h, actual_w = prob.shape
        local_row0 = row0 - tile.row_off
        local_row1 = row1 - tile.row_off
        local_col0 = col0 - tile.col_off
        local_col1 = col1 - tile.col_off
        if actual_h != tile.height or actual_w != tile.width:
            local_row0 = int(round(local_row0 * actual_h / tile.height))
            local_row1 = int(round(local_row1 * actual_h / tile.height))
            local_col0 = int(round(local_col0 * actual_w / tile.width))
            local_col1 = int(round(local_col1 * actual_w / tile.width))
        target_rows = slice(row0 - crop_row0, row1 - crop_row0)
        target_cols = slice(col0 - crop_col0, col1 - crop_col0)
        probability_sum[target_rows, target_cols] += prob[local_row0:local_row1, local_col0:local_col1]
        coverage_count[target_rows, target_cols] += 1
        used_tiles.append(tile.image_name)

    prob_u8 = np.zeros_like(coverage_count, dtype=np.uint8)
    covered = coverage_count > 0
    averaged = (probability_sum[covered].astype(np.float32) / coverage_count[covered].astype(np.float32)).round()
    prob_u8[covered] = np.clip(averaged, 0, 255).astype(np.uint8)
    return prob_u8, coverage_count, sorted(set(used_tiles))


def write_geotiff_outputs(
    *,
    out_dir: Path,
    rgb: np.ndarray,
    prob_u8: np.ndarray,
    coverage_count: np.ndarray,
    transform: Affine,
    crs: Any,
    threshold: float,
    weak_threshold: float,
) -> dict[str, str]:
    strong_u8 = threshold_to_u8(threshold)
    weak_u8 = threshold_to_u8(weak_threshold)
    strong = prob_u8 >= strong_u8
    weak = (prob_u8 >= weak_u8) & ~strong
    mask_strong = (strong.astype(np.uint8) * 255)
    mask_weak = ((prob_u8 >= weak_u8).astype(np.uint8) * 255)
    coverage = np.clip(coverage_count, 0, 255).astype(np.uint8)
    overlay_rgb = blend_segmentation(rgb, strong=strong, weak=weak)
    rgba = build_transparent_overlay(strong=strong, weak=weak)

    outputs = {
        "probability_u8_tif": str(out_dir / "ta08_segmentation_probability_u8.tif"),
        "mask_thr090_tif": str(out_dir / "ta08_segmentation_mask_thr090.tif"),
        "mask_thr050_tif": str(out_dir / "ta08_segmentation_mask_thr050.tif"),
        "transparent_overlay_tif": str(out_dir / "ta08_segmentation_overlay_rgba_thr050_090.tif"),
        "dom_overlay_tif": str(out_dir / "ta08_dom_segmentation_overlay.tif"),
        "tile_coverage_tif": str(out_dir / "ta08_prediction_tile_coverage_count.tif"),
    }
    write_single_band_tif(Path(outputs["probability_u8_tif"]), prob_u8, transform, crs)
    write_single_band_tif(Path(outputs["mask_thr090_tif"]), mask_strong, transform, crs)
    write_single_band_tif(Path(outputs["mask_thr050_tif"]), mask_weak, transform, crs)
    write_single_band_tif(Path(outputs["tile_coverage_tif"]), coverage, transform, crs)
    write_multiband_tif(Path(outputs["transparent_overlay_tif"]), rgba, transform, crs)
    write_multiband_tif(Path(outputs["dom_overlay_tif"]), overlay_rgb, transform, crs)
    return outputs


def write_single_band_tif(path: Path, arr: np.ndarray, transform: Affine, crs: Any) -> None:
    profile = base_tif_profile(arr.shape[1], arr.shape[0], 1, transform, crs)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def write_multiband_tif(path: Path, arr: np.ndarray, transform: Affine, crs: Any) -> None:
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC array, got shape {arr.shape}")
    profile = base_tif_profile(arr.shape[1], arr.shape[0], arr.shape[2], transform, crs)
    if arr.shape[2] in {3, 4}:
        profile["photometric"] = "RGB"
        profile["interleave"] = "pixel"
    if arr.shape[2] == 4:
        profile["ALPHA"] = "YES"
    with rasterio.open(path, "w", **profile) as dst:
        if arr.shape[2] == 3:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        elif arr.shape[2] == 4:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
        dst.write(np.moveaxis(arr, -1, 0))


def base_tif_profile(width: int, height: int, count: int, transform: Affine, crs: Any) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "IF_SAFER",
    }


def blend_segmentation(rgb: np.ndarray, *, strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    weak_color = np.array([230, 57, 70], dtype=np.float32)
    strong_color = np.array([255, 183, 3], dtype=np.float32)
    out[weak] = out[weak] * 0.58 + weak_color * 0.42
    out[strong] = out[strong] * 0.40 + strong_color * 0.60
    return np.clip(out, 0, 255).astype(np.uint8)


def build_transparent_overlay(*, strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    rgba = np.zeros((strong.shape[0], strong.shape[1], 4), dtype=np.uint8)
    rgba[weak] = np.array([230, 57, 70, 110], dtype=np.uint8)
    rgba[strong] = np.array([255, 183, 3, 180], dtype=np.uint8)
    return rgba


def write_qa_crops(
    *,
    qa_dir: Path,
    rgb: np.ndarray,
    prob_u8: np.ndarray,
    transform: Affine,
    review_points: list[tuple[str, float, float]],
    line_coords: list[tuple[float, float]],
    point_crop_m: float,
    threshold: float,
    weak_threshold: float,
) -> list[str]:
    outputs: list[str] = []
    full_path = qa_dir / "ta08_full_segment_dom_seg_line.png"
    save_overlay_png(
        full_path,
        rgb,
        prob_u8,
        transform,
        review_points=review_points,
        line_coords=line_coords,
        threshold=threshold,
        weak_threshold=weak_threshold,
        label="TA08 segmentation evidence: full local crop",
    )
    outputs.append(str(full_path))

    pixel_size = max(abs(float(transform.a)), abs(float(transform.e)), 1e-6)
    half_px = max(16, int(math.ceil((point_crop_m / 2.0) / pixel_size)))
    for index, (name, x, y) in enumerate(review_points, start=1):
        col, row = world_to_pixel(transform, x, y)
        row0 = max(0, int(round(row)) - half_px)
        row1 = min(rgb.shape[0], int(round(row)) + half_px)
        col0 = max(0, int(round(col)) - half_px)
        col1 = min(rgb.shape[1], int(round(col)) + half_px)
        sub_transform = transform * Affine.translation(col0, row0)
        sub_path = qa_dir / f"ta08_point{index:02d}_{name}_dom_seg_line.png"
        save_overlay_png(
            sub_path,
            rgb[row0:row1, col0:col1],
            prob_u8[row0:row1, col0:col1],
            sub_transform,
            review_points=review_points,
            line_coords=line_coords,
            threshold=threshold,
            weak_threshold=weak_threshold,
            label=f"TA08 {name}",
        )
        outputs.append(str(sub_path))
    return outputs


def save_overlay_png(
    path: Path,
    rgb: np.ndarray,
    prob_u8: np.ndarray,
    transform: Affine,
    *,
    review_points: list[tuple[str, float, float]],
    line_coords: list[tuple[float, float]],
    threshold: float,
    weak_threshold: float,
    label: str,
) -> None:
    strong = prob_u8 >= threshold_to_u8(threshold)
    weak = (prob_u8 >= threshold_to_u8(weak_threshold)) & ~strong
    overlay = Image.fromarray(blend_segmentation(rgb, strong=strong, weak=weak), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw_line(draw, line_coords, transform)
    draw_points(draw, review_points, transform)
    draw_label(draw, label)
    overlay.convert("RGB").save(path)


def draw_line(draw: ImageDraw.ImageDraw, line_coords: list[tuple[float, float]], transform: Affine) -> None:
    pixels = [world_to_pixel(transform, x, y) for x, y in line_coords]
    if len(pixels) >= 2:
        draw.line(pixels, fill=(0, 210, 255, 245), width=4, joint="curve")


def draw_points(draw: ImageDraw.ImageDraw, review_points: list[tuple[str, float, float]], transform: Affine) -> None:
    for index, (_, x, y) in enumerate(review_points, start=1):
        col, row = world_to_pixel(transform, x, y)
        radius = 9
        draw.ellipse([col - radius, row - radius, col + radius, row + radius], fill=(30, 80, 255, 235), outline=(255, 255, 255, 255), width=2)
        draw.text((col + radius + 3, row - radius - 2), str(index), fill=(255, 255, 255, 255))


def draw_label(draw: ImageDraw.ImageDraw, label: str) -> None:
    draw.rectangle([8, 8, 620, 40], fill=(0, 0, 0, 190))
    draw.text((18, 16), label[:82], fill=(255, 255, 255, 255))


def point_probability_stats(
    *,
    name: str,
    x: float,
    y: float,
    prob_u8: np.ndarray,
    coverage_count: np.ndarray,
    transform: Affine,
    radius_m: float,
    thresholds: tuple[float, float],
) -> dict[str, Any]:
    col, row = world_to_pixel(transform, x, y)
    values, covered = sample_square(prob_u8, coverage_count, transform, col, row, radius_m)
    return probability_stats_row(
        name=name,
        x=x,
        y=y,
        local_col=float(col),
        local_row=float(row),
        values=values,
        covered=covered,
        radius_m=radius_m,
        thresholds=thresholds,
    )


def line_probability_samples(
    *,
    line_coords: list[tuple[float, float]],
    prob_u8: np.ndarray,
    coverage_count: np.ndarray,
    transform: Affine,
    radius_m: float,
    sample_count: int,
    thresholds: tuple[float, float],
) -> list[dict[str, Any]]:
    samples = sample_polyline(line_coords, max(2, sample_count))
    rows: list[dict[str, Any]] = []
    for index, (s_m, x, y) in enumerate(samples):
        col, row = world_to_pixel(transform, x, y)
        values, covered = sample_square(prob_u8, coverage_count, transform, col, row, radius_m)
        stats = probability_stats_row(
            name=f"sample_{index:02d}",
            x=x,
            y=y,
            local_col=float(col),
            local_row=float(row),
            values=values,
            covered=covered,
            radius_m=radius_m,
            thresholds=thresholds,
        )
        stats["s_m"] = round(float(s_m), 3)
        rows.append(stats)
    return rows


def sample_square(
    prob_u8: np.ndarray,
    coverage_count: np.ndarray,
    transform: Affine,
    col: float,
    row: float,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_size = max(abs(float(transform.a)), abs(float(transform.e)), 1e-6)
    radius_px = max(1, int(math.ceil(radius_m / pixel_size)))
    r0 = max(0, int(round(row)) - radius_px)
    r1 = min(prob_u8.shape[0], int(round(row)) + radius_px + 1)
    c0 = max(0, int(round(col)) - radius_px)
    c1 = min(prob_u8.shape[1], int(round(col)) + radius_px + 1)
    return prob_u8[r0:r1, c0:c1], coverage_count[r0:r1, c0:c1] > 0


def probability_stats_row(
    *,
    name: str,
    x: float,
    y: float,
    local_col: float,
    local_row: float,
    values: np.ndarray,
    covered: np.ndarray,
    radius_m: float,
    thresholds: tuple[float, float],
) -> dict[str, Any]:
    weak_thr, strong_thr = thresholds
    weak_u8 = threshold_to_u8(weak_thr)
    strong_u8 = threshold_to_u8(strong_thr)
    covered_values = values[covered]
    if covered_values.size == 0:
        max_prob = mean_prob = weak_fraction = strong_fraction = 0.0
        weak_count = strong_count = 0
    else:
        max_prob = float(covered_values.max() / 255.0)
        mean_prob = float(covered_values.mean() / 255.0)
        weak_count = int(np.count_nonzero(covered_values >= weak_u8))
        strong_count = int(np.count_nonzero(covered_values >= strong_u8))
        weak_fraction = float(weak_count / covered_values.size)
        strong_fraction = float(strong_count / covered_values.size)
    return {
        "name": name,
        "x": round(float(x), 3),
        "y": round(float(y), 3),
        "local_col": round(local_col, 2),
        "local_row": round(local_row, 2),
        "radius_m": radius_m,
        "covered_fraction": float(np.count_nonzero(covered) / covered.size) if covered.size else 0.0,
        "max_probability": round(max_prob, 4),
        "mean_probability": round(mean_prob, 4),
        "weak_pixel_count": weak_count,
        "strong_pixel_count": strong_count,
        "weak_fraction": round(weak_fraction, 6),
        "strong_fraction": round(strong_fraction, 6),
    }


def sample_polyline(coords: list[tuple[float, float]], sample_count: int) -> list[tuple[float, float, float]]:
    if len(coords) < 2:
        return []
    distances = [0.0]
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        distances.append(distances[-1] + math.hypot(x1 - x0, y1 - y0))
    total = distances[-1]
    if total <= 0:
        x, y = coords[0]
        return [(0.0, x, y)]
    targets = np.linspace(0.0, total, sample_count)
    samples: list[tuple[float, float, float]] = []
    seg_index = 0
    for target in targets:
        while seg_index + 1 < len(distances) and distances[seg_index + 1] < target:
            seg_index += 1
        if seg_index + 1 >= len(coords):
            x, y = coords[-1]
        else:
            s0 = distances[seg_index]
            s1 = distances[seg_index + 1]
            ratio = 0.0 if s1 <= s0 else float((target - s0) / (s1 - s0))
            x0, y0 = coords[seg_index]
            x1, y1 = coords[seg_index + 1]
            x = x0 + (x1 - x0) * ratio
            y = y0 + (y1 - y0) * ratio
        samples.append((float(target), float(x), float(y)))
    return samples


def world_to_pixel(transform: Affine, x: float, y: float) -> tuple[float, float]:
    col, row = ~transform * (x, y)
    return float(col), float(row)


def threshold_to_u8(value: float) -> int:
    return int(math.ceil(max(0.0, min(1.0, value)) * 255.0))


def raster_bounds(transform: Affine, width: int, height: int) -> dict[str, float]:
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (width, height)
    return {
        "x_min": round(min(x0, x1), 3),
        "y_min": round(min(y0, y1), 3),
        "x_max": round(max(x0, x1), 3),
        "y_max": round(max(y0, y1), 3),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
