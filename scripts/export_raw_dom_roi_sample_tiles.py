#!/usr/bin/env python3
"""Export representative raw-DOM ROI tiles for semantic-segmentation QA."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry


DEFAULT_TILE_INDEX = Path("output/raw_dom_corridor_roi/raw_dom_tile_index.csv")
DEFAULT_FOCUS_GEOJSON = Path("output/tonghaigang_topology_rebuild_v1/switch_model_workzones.geojson")
DEFAULT_OUT_DIR = Path("output/raw_dom_segmentation_sample")


class TileRow(NamedTuple):
    tile_id: int
    tile_name: str
    row_off: int
    col_off: int
    width: int
    height: int
    source_path: Path
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    roi_intersection_area_m2: float
    focus_intersection_area_m2: float = 0.0
    selection_reason: str = ""
    image_name: str = ""
    tile_transform: str = ""
    source_transform: str = ""
    crs: str = ""
    epsg: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export sample images from a raw DOM ROI tile index.")
    parser.add_argument("--tile-index", type=Path, default=DEFAULT_TILE_INDEX)
    parser.add_argument(
        "--source-dom",
        type=Path,
        default=None,
        help="Override source_path from the tile index. Useful when an older index has mojibake in a Windows path.",
    )
    parser.add_argument("--focus-geojson", type=Path, default=DEFAULT_FOCUS_GEOJSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-tiles", type=int, default=36)
    parser.add_argument("--focus-quota", type=int, default=14)
    parser.add_argument("--row-bins", type=int, default=18)
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--quality", type=int, default=94)
    parser.add_argument("--contact-sheet-max", type=int, default=24)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = load_tile_rows(args.tile_index)
    if args.source_dom is not None:
        source_dom = args.source_dom.expanduser().resolve()
        rows = [row._replace(source_path=source_dom) for row in rows]
    focus_geometries = load_focus_geometries(args.focus_geojson)
    rows = score_focus_intersections(rows, focus_geometries)
    selected = select_tiles(rows, max_tiles=args.max_tiles, focus_quota=args.focus_quota, row_bins=args.row_bins)
    if not selected:
        raise ValueError("No sample tiles were selected.")

    source_paths = {row.source_path.resolve() for row in selected}
    if len(source_paths) != 1:
        raise ValueError(f"Expected all selected tiles to share one source DOM, got {len(source_paths)}.")
    source_path = next(iter(source_paths))

    exported: list[TileRow] = []
    with rasterio.open(source_path) as dataset:
        for row in selected:
            image_name = f"{Path(row.tile_name).stem}.{args.format}"
            image_path = images_dir / image_name
            export_tile(dataset, row, image_path, image_format=args.format, quality=args.quality)
            exported.append(row._replace(image_name=image_name))
        contact_sheet_path = out_dir / "sample_contact_sheet.jpg"
        write_contact_sheet(contact_sheet_path, dataset, exported, max_items=args.contact_sheet_max)

    selected_csv = out_dir / "selected_tile_index.csv"
    selected_json = out_dir / "selected_tile_index.json"
    write_selected_csv(selected_csv, exported)
    summary = {
        "tile_index": str(args.tile_index.expanduser().resolve()),
        "focus_geojson": str(args.focus_geojson.expanduser().resolve()) if args.focus_geojson else None,
        "out_dir": str(out_dir),
        "images_dir": str(images_dir),
        "image_count": len(exported),
        "source_path": str(source_path),
        "max_tiles": args.max_tiles,
        "focus_quota": args.focus_quota,
        "row_bins": args.row_bins,
        "format": args.format,
        "focus_selected_count": sum(1 for row in exported if row.selection_reason == "focus_workzone"),
        "even_selected_count": sum(1 for row in exported if row.selection_reason == "row_bin"),
        "contact_sheet": str(contact_sheet_path),
        "selected_tile_index_csv": str(selected_csv),
        "selected_tile_index_json": str(selected_json),
        "tile_georef_path": str(selected_csv),
    }
    selected_json.write_text(json.dumps({"summary": summary, "tiles": [row._asdict() for row in exported]}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_tile_rows(path: Path) -> list[TileRow]:
    rows: list[TileRow] = []
    with path.expanduser().resolve().open("r", encoding="utf-8", newline="") as fh:
        for item in csv.DictReader(fh):
            rows.append(
                TileRow(
                    tile_id=int(item["tile_id"]),
                    tile_name=item["tile_name"],
                    row_off=int(item["row_off"]),
                    col_off=int(item["col_off"]),
                    width=int(item["width"]),
                    height=int(item["height"]),
                    source_path=Path(item["source_path"]),
                    x_min=float(item["x_min"]),
                    y_min=float(item["y_min"]),
                    x_max=float(item["x_max"]),
                    y_max=float(item["y_max"]),
                    roi_intersection_area_m2=float(item["roi_intersection_area_m2"]),
                    tile_transform=item.get("tile_transform", ""),
                    source_transform=item.get("source_transform", ""),
                    crs=item.get("crs", ""),
                    epsg=item.get("epsg", ""),
                ),
            )
    return rows


def load_focus_geometries(path: Path | None) -> list[BaseGeometry]:
    if path is None:
        return []
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return []
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    geometries: list[BaseGeometry] = []
    for feature in payload.get("features", []) or []:
        geometry_payload = feature.get("geometry")
        if geometry_payload:
            geometry = shape(geometry_payload)
            if not geometry.is_empty:
                geometries.append(geometry)
    return geometries


def score_focus_intersections(rows: list[TileRow], focus_geometries: list[BaseGeometry]) -> list[TileRow]:
    if not focus_geometries:
        return rows
    scored: list[TileRow] = []
    for row in rows:
        tile_polygon = box(row.x_min, row.y_min, row.x_max, row.y_max)
        focus_area = sum(float(tile_polygon.intersection(geometry).area) for geometry in focus_geometries if tile_polygon.intersects(geometry))
        scored.append(row._replace(focus_intersection_area_m2=focus_area))
    return scored


def select_tiles(rows: list[TileRow], *, max_tiles: int, focus_quota: int, row_bins: int) -> list[TileRow]:
    if max_tiles <= 0 or len(rows) <= max_tiles:
        return [row._replace(selection_reason="all") for row in rows]
    selected: dict[int, TileRow] = {}

    focus_candidates = sorted(
        (row for row in rows if row.focus_intersection_area_m2 > 0.0),
        key=lambda row: (row.focus_intersection_area_m2, row.roi_intersection_area_m2),
        reverse=True,
    )
    for row in focus_candidates[: max(0, min(focus_quota, max_tiles))]:
        selected[row.tile_id] = row._replace(selection_reason="focus_workzone")

    remaining_count = max_tiles - len(selected)
    if remaining_count <= 0:
        return sort_selected(selected.values())

    for row in best_rows_by_row_bin(rows, row_bins=row_bins):
        if row.tile_id in selected:
            continue
        selected[row.tile_id] = row._replace(selection_reason="row_bin")
        if len(selected) >= max_tiles:
            break

    if len(selected) < max_tiles:
        for row in sorted(rows, key=lambda item: item.roi_intersection_area_m2, reverse=True):
            if row.tile_id not in selected:
                selected[row.tile_id] = row._replace(selection_reason="top_roi_area")
            if len(selected) >= max_tiles:
                break
    return sort_selected(selected.values())


def best_rows_by_row_bin(rows: list[TileRow], *, row_bins: int) -> list[TileRow]:
    if not rows:
        return []
    row_bins = max(1, row_bins)
    row_values = np.asarray([row.row_off for row in rows], dtype=float)
    edges = np.linspace(float(row_values.min()), float(row_values.max()) + 1.0, row_bins + 1)
    best: list[TileRow] = []
    for index in range(row_bins):
        left, right = edges[index], edges[index + 1]
        bucket = [row for row in rows if left <= row.row_off < right]
        if bucket:
            best.append(max(bucket, key=lambda row: row.roi_intersection_area_m2))
    return best


def sort_selected(rows: Any) -> list[TileRow]:
    return sorted(rows, key=lambda row: (row.row_off, row.col_off, row.tile_id))


def export_tile(
    dataset: rasterio.io.DatasetReader,
    row: TileRow,
    path: Path,
    *,
    image_format: str,
    quality: int,
) -> None:
    window = Window(row.col_off, row.row_off, row.width, row.height)
    indexes = [1] if dataset.count == 1 else ([1, 2, 3] if dataset.count >= 3 else list(range(1, dataset.count + 1)))
    data = dataset.read(indexes, window=window)
    image_array = to_uint8(data)
    if image_array.shape[0] == 1:
        image = Image.fromarray(image_array[0], mode="L").convert("RGB")
    else:
        image = Image.fromarray(np.moveaxis(image_array[:3], 0, -1), mode="RGB")
    save_format = "JPEG" if image_format == "jpg" else "PNG"
    save_kwargs = {"quality": quality} if save_format == "JPEG" else {}
    image.save(path, format=save_format, **save_kwargs)


def write_contact_sheet(
    path: Path,
    dataset: rasterio.io.DatasetReader,
    rows: list[TileRow],
    *,
    max_items: int,
) -> None:
    selected = rows[:max_items]
    if not selected:
        return
    columns = min(4, len(selected))
    cell_width = 310
    cell_height = 260
    label_height = 24
    rows_count = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows_count * cell_height), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        col = index % columns
        out_row = index // columns
        x0 = col * cell_width
        y0 = out_row * cell_height
        draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 1), outline=(180, 180, 174))
        label = f"{Path(row.image_name).stem[-28:]} | {row.selection_reason}"
        draw.text((x0 + 5, y0 + 5), label, fill=(35, 35, 35))
        thumb = read_tile_thumbnail(dataset, row, cell_width - 12, cell_height - label_height - 12)
        sheet.paste(thumb, (x0 + 6, y0 + label_height + 6))
    sheet.save(path, quality=92)


def read_tile_thumbnail(dataset: rasterio.io.DatasetReader, row: TileRow, width: int, height: int) -> Image.Image:
    window = Window(row.col_off, row.row_off, row.width, row.height)
    indexes = [1] if dataset.count == 1 else ([1, 2, 3] if dataset.count >= 3 else list(range(1, dataset.count + 1)))
    data = dataset.read(indexes, window=window, out_shape=(len(indexes), height, width), resampling=Resampling.bilinear)
    image_array = to_uint8(data)
    if image_array.shape[0] == 1:
        return Image.fromarray(image_array[0], mode="L").convert("RGB")
    return Image.fromarray(np.moveaxis(image_array[:3], 0, -1), mode="RGB")


def write_selected_csv(path: Path, rows: list[TileRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TileRow._fields))
        writer.writeheader()
        for row in rows:
            data = row._asdict()
            data["source_path"] = str(data["source_path"])
            writer.writerow(data)


def to_uint8(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return data
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        if info.max <= 255 and info.min >= 0:
            return data.astype(np.uint8)
        scaled = (data.astype(np.float32) - info.min) / max(info.max - info.min, 1)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    finite = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if finite.max(initial=0.0) <= 1.0 and finite.min(initial=0.0) >= 0.0:
        return np.clip(finite * 255.0, 0, 255).astype(np.uint8)
    return np.clip(finite, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
