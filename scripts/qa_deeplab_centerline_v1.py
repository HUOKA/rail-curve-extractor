#!/usr/bin/env python3
"""Write lossless original-resolution QA crops for DeepLab centerline outputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window


DEFAULT_DOM = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u65e0\u4eba\u673a\u6570\u636e" / "\u6b63\u5c04" / "dom.tif"
DEFAULT_REFINED = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_refined_deeplab_v1_thr050/refined_centerline_network.geojson")
DEFAULT_MAIN = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_refined_deeplab_v1_thr050/main_centerline.geojson")
DEFAULT_CANDIDATES = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_candidates_deeplab_v1_thr050/track_centerline_candidates.geojson")
DEFAULT_OUT = Path("output/raw_dom_roi_fullpass_v1/deeplab_centerline_v1_qa")


REVIEW_POINTS: list[tuple[str, float, float]] = [
    ("ta08_user_coord", 315334.923, 3520755.899),
    ("ta08_coord01_user_ok", 315349.015, 3520808.310),
    ("ta08_coord02_user_deviates", 315341.558, 3520778.002),
    ("ta08_coord03_user_intersects_rail", 315333.927, 3520749.822),
]


@dataclass(frozen=True)
class Target:
    name: str
    label: str
    point: tuple[float, float]
    source: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate original-resolution DOM QA crops for DeepLab centerline V1.")
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--refined", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--crop-m", type=float, default=80.0)
    parser.add_argument("--max-crops", type=int, default=24)
    parser.add_argument("--candidate-limit", type=int, default=0, help="0 draws all candidate features that intersect each crop.")
    parser.add_argument("--with-raw", action="store_true", help="Also save the unannotated DOM crop.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_path in list(out_dir.glob("*.png")) + [out_dir / "qa_crops_index.json"]:
        if old_path.exists():
            old_path.unlink()

    refined = load_features(args.refined)
    mainline = select_mainline_features(load_features(args.main))
    candidates = load_features(args.candidates)
    if args.candidate_limit > 0:
        candidates = candidates[: args.candidate_limit]

    targets = dedupe_targets(build_targets(mainline, refined), max_count=args.max_crops)
    entries: list[dict[str, Any]] = []

    with rasterio.open(args.dom) as dataset:
        for index, target in enumerate(targets, start=1):
            window = centered_window(dataset, target.point[0], target.point[1], args.crop_m)
            if window.width <= 1 or window.height <= 1:
                continue
            rgb = read_rgb_window(dataset, window)
            transform = dataset.window_transform(window)
            bbox = window_bounds(window, transform)
            image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
            draw = ImageDraw.Draw(image, "RGBA")

            draw_feature_set(draw, candidates, transform, bbox, color=(0, 185, 255, 190), width_px=2)
            draw_feature_set(draw, refined, transform, bbox, color=(20, 220, 90, 245), width_px=5, role_colors={"main_path": (255, 70, 50, 255), "main": (255, 70, 50, 255)})
            draw_feature_set(draw, mainline, transform, bbox, color=(255, 40, 220, 255), width_px=7)
            draw_target_marker(draw, transform, target.point)
            draw_label(draw, f"{index:02d} {target.label}")

            stem = sanitize_filename(f"{index:02d}_{target.name}")
            overlay_path = out_dir / f"{stem}_overlay.png"
            image.convert("RGB").save(overlay_path, compress_level=0)
            raw_path = None
            if args.with_raw:
                raw_path = out_dir / f"{stem}_raw.png"
                Image.fromarray(rgb, mode="RGB").save(raw_path, compress_level=0)

            entries.append(
                {
                    "index": index,
                    "name": target.name,
                    "label": target.label,
                    "source": target.source,
                    "point": [round(target.point[0], 3), round(target.point[1], 3)],
                    "crop_m": args.crop_m,
                    "pixel_size": [int(window.width), int(window.height)],
                    "overlay": str(overlay_path),
                    "raw": str(raw_path) if raw_path else None,
                }
            )

    index = {
        "dom": str(args.dom.resolve()),
        "refined": str(args.refined.resolve()),
        "main": str(args.main.resolve()),
        "candidates": str(args.candidates.resolve()),
        "policy": "native DOM resolution, no resizing, PNG compress_level=0, no JPEG",
        "crop_count": len(entries),
        "crops": entries,
    }
    (out_dir / "qa_crops_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


def load_features(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [feature for feature in data.get("features", []) if line_coords(feature)]


def select_mainline_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not features:
        return []
    preferred_roles = {"mainline", "main", "main_path"}
    selected = [
        feature
        for feature in features
        if str((feature.get("properties") or {}).get("role", "")) in preferred_roles
    ]
    return selected[:1] if selected else features[:1]


def build_targets(mainline: list[dict[str, Any]], refined: list[dict[str, Any]]) -> list[Target]:
    targets: list[Target] = []
    for name, x, y in REVIEW_POINTS:
        targets.append(Target(name=name, label=name, point=(x, y), source="manual_review_point"))

    if mainline:
        coords = line_coords(mainline[0])
        for fraction in (0.08, 0.22, 0.38, 0.54, 0.70, 0.86):
            point = point_at_fraction(coords, fraction)
            targets.append(Target(name=f"mainline_{fraction:.2f}", label=f"main centerline {fraction:.0%}", point=point, source="mainline"))

    sorted_refined = sorted(refined, key=lambda feature: float(feature.get("properties", {}).get("length_m", 0.0)), reverse=True)
    for feature in sorted_refined[:12]:
        coords = line_coords(feature)
        props = feature.get("properties") or {}
        chain_id = str(props.get("chain_id", "chain"))
        role = str(props.get("role", "support"))
        for fraction in (0.20, 0.50, 0.80):
            point = point_at_fraction(coords, fraction)
            targets.append(
                Target(
                    name=f"chain_{sanitize_filename(chain_id)}_{fraction:.2f}",
                    label=f"chain {chain_id} {role} {fraction:.0%}",
                    point=point,
                    source="refined_network",
                )
            )
    return targets


def dedupe_targets(targets: list[Target], *, max_count: int) -> list[Target]:
    accepted: list[Target] = []
    for target in targets:
        if any(math.hypot(target.point[0] - old.point[0], target.point[1] - old.point[1]) < 20.0 for old in accepted):
            continue
        accepted.append(target)
        if len(accepted) >= max_count:
            break
    return accepted


def centered_window(dataset: Any, x: float, y: float, crop_m: float) -> Window:
    pixel_width = max(abs(float(dataset.transform.a)), 1e-9)
    pixel_height = max(abs(float(dataset.transform.e)), 1e-9)
    width_px = max(64, int(math.ceil(crop_m / pixel_width)))
    height_px = max(64, int(math.ceil(crop_m / pixel_height)))
    row, col = dataset.index(x, y)
    col_off = max(0, min(dataset.width - 1, col - width_px // 2))
    row_off = max(0, min(dataset.height - 1, row - height_px // 2))
    width = min(width_px, dataset.width - col_off)
    height = min(height_px, dataset.height - row_off)
    return Window(col_off, row_off, width, height)


def read_rgb_window(dataset: Any, window: Window) -> Any:
    import numpy as np

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


def draw_feature_set(
    draw: ImageDraw.ImageDraw,
    features: Iterable[dict[str, Any]],
    transform: Any,
    bbox: tuple[float, float, float, float],
    *,
    color: tuple[int, int, int, int],
    width_px: int,
    role_colors: dict[str, tuple[int, int, int, int]] | None = None,
) -> None:
    for feature in features:
        coords = line_coords(feature)
        if not coords or not intersects_bbox(coords, bbox):
            continue
        role = str((feature.get("properties") or {}).get("role", ""))
        line_color = role_colors.get(role, color) if role_colors else color
        pixel_coords = [world_to_pixel(transform, x, y) for x, y in coords]
        if len(pixel_coords) < 2:
            continue
        halo_width = max(width_px + 3, width_px)
        draw.line(pixel_coords, fill=(0, 0, 0, 180), width=halo_width, joint="curve")
        draw.line(pixel_coords, fill=line_color, width=width_px, joint="curve")


def draw_target_marker(draw: ImageDraw.ImageDraw, transform: Any, point: tuple[float, float]) -> None:
    col, row = world_to_pixel(transform, point[0], point[1])
    radius = 9
    draw.ellipse([col - radius, row - radius, col + radius, row + radius], outline=(255, 255, 255, 255), width=3)
    draw.line([(col - 18, row), (col + 18, row)], fill=(255, 0, 255, 235), width=3)
    draw.line([(col, row - 18), (col, row + 18)], fill=(255, 0, 255, 235), width=3)


def draw_label(draw: ImageDraw.ImageDraw, label: str) -> None:
    text = label[:120]
    draw.rectangle([8, 8, 980, 44], fill=(0, 0, 0, 190))
    draw.text((18, 18), text, fill=(255, 255, 255, 255))


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "LineString":
        return []
    return [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates", [])]


def point_at_fraction(coords: list[tuple[float, float]], fraction: float) -> tuple[float, float]:
    if not coords:
        raise ValueError("Cannot sample an empty polyline.")
    if len(coords) == 1:
        return coords[0]
    total = polyline_length(coords)
    target = max(0.0, min(1.0, fraction)) * total
    covered = 0.0
    for (ax, ay), (bx, by) in zip(coords, coords[1:]):
        segment = math.hypot(bx - ax, by - ay)
        if segment <= 0:
            continue
        if covered + segment >= target:
            t = (target - covered) / segment
            return ax + (bx - ax) * t, ay + (by - ay) * t
        covered += segment
    return coords[-1]


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(coords, coords[1:]))


def world_to_pixel(transform: Any, x: float, y: float) -> tuple[float, float]:
    col, row = ~transform * (x, y)
    return float(col), float(row)


def window_bounds(window: Window, transform: Any) -> tuple[float, float, float, float]:
    left, top = transform * (0, 0)
    right, bottom = transform * (window.width, window.height)
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def intersects_bbox(coords: list[tuple[float, float]], bbox: tuple[float, float, float, float]) -> bool:
    left, bottom, right, top = bbox
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return max(xs) >= left and min(xs) <= right and max(ys) >= bottom and min(ys) <= top


def sanitize_filename(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    cleaned = "".join(ch if ch in allowed else "_" for ch in value)
    return cleaned.strip("_") or "item"


if __name__ == "__main__":
    raise SystemExit(main())
