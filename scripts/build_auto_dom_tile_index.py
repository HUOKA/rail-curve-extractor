#!/usr/bin/env python3
"""Build a DOM-derived tile index without manual corridor geometry."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import rasterio
from rasterio.windows import Window


DEFAULT_DOM = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u65e0\u4eba\u673a\u6570\u636e" / "\u6b63\u5c04" / "dom.tif"
DEFAULT_OUT_DIR = Path("output/dom_centerline_strict_auto_v1/00_auto_dom_tile_index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a full-DOM tile index from raster metadata only.")
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tile-width", type=int, default=3072)
    parser.add_argument("--tile-height", type=int, default=3072)
    parser.add_argument("--tile-overlap", type=float, default=0.0)
    parser.add_argument("--prefix", default="auto_dom_tile")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dom_path = args.dom.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dom_path) as dataset:
        rows = build_tile_rows(
            dataset,
            dom_path=dom_path,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            tile_overlap=args.tile_overlap,
            prefix=args.prefix,
        )
        if not rows:
            raise RuntimeError("DOM tile index is empty.")
        csv_path = out_dir / "raw_dom_tile_index.csv"
        geojson_path = out_dir / "raw_dom_tile_index.geojson"
        selected_csv_path = out_dir / "selected_tile_index.csv"
        write_tile_csv(csv_path, rows)
        write_tile_csv(selected_csv_path, rows)
        write_tile_geojson(geojson_path, rows, dataset)
        summary = {
            "mode": "strict_auto_full_dom_tile_index",
            "dom": str(dom_path),
            "out_dir": str(out_dir),
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "tile_overlap": args.tile_overlap,
            "tile_count": len(rows),
            "dom_width": int(dataset.width),
            "dom_height": int(dataset.height),
            "dom_crs": dataset.crs.to_string() if dataset.crs else None,
            "dom_epsg": dataset.crs.to_epsg() if dataset.crs else None,
            "outputs": {
                "raw_dom_tile_index_csv": str(csv_path),
                "selected_tile_index_csv": str(selected_csv_path),
                "raw_dom_tile_index_geojson": str(geojson_path),
                "summary_json": str(out_dir / "summary.json"),
            },
            "policy": "No manual corridor, review geometry, or retained ROI index is used.",
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_tile_rows(
    dataset: Any,
    *,
    dom_path: Path,
    tile_width: int,
    tile_height: int,
    tile_overlap: float,
    prefix: str,
) -> list[dict[str, Any]]:
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Tile width and height must be positive.")
    if not 0.0 <= tile_overlap < 1.0:
        raise ValueError("Tile overlap must be in [0, 1).")
    stride_x = max(1, int(round(tile_width * (1.0 - tile_overlap))))
    stride_y = max(1, int(round(tile_height * (1.0 - tile_overlap))))
    col_offsets = offsets_for_extent(dataset.width, tile_width, stride_x)
    row_offsets = offsets_for_extent(dataset.height, tile_height, stride_y)
    rows: list[dict[str, Any]] = []
    tile_id = 0
    for row_off in row_offsets:
        for col_off in col_offsets:
            width = min(tile_width, dataset.width - col_off)
            height = min(tile_height, dataset.height - row_off)
            if width <= 0 or height <= 0:
                continue
            window = Window(col_off, row_off, width, height)
            transform = dataset.window_transform(window)
            x0, y0 = transform * (0.0, 0.0)
            x1, y1 = transform * (float(width), float(height))
            x_min, x_max = sorted((float(x0), float(x1)))
            y_min, y_max = sorted((float(y0), float(y1)))
            area = abs((x_max - x_min) * (y_max - y_min))
            tile_name = f"{prefix}_r{row_off:06d}_c{col_off:06d}.png"
            rows.append(
                {
                    "tile_id": tile_id,
                    "tile_name": tile_name,
                    "row_off": row_off,
                    "col_off": col_off,
                    "width": width,
                    "height": height,
                    "source_path": str(dom_path),
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "roi_intersection_area_m2": area,
                    "focus_intersection_area_m2": 0.0,
                    "selection_reason": "strict_auto_full_dom",
                    "image_name": tile_name,
                    "tile_transform": json.dumps(affine_to_list(transform)),
                    "source_transform": json.dumps(affine_to_list(dataset.transform)),
                    "crs": dataset.crs.to_string() if dataset.crs else "",
                    "epsg": dataset.crs.to_epsg() if dataset.crs else "",
                }
            )
            tile_id += 1
    return rows


def offsets_for_extent(size: int, tile_size: int, stride: int) -> list[int]:
    if size <= tile_size:
        return [0]
    offsets = list(range(0, max(1, size - tile_size + 1), stride))
    last = size - tile_size
    if offsets[-1] != last:
        offsets.append(last)
    return sorted(set(offsets))


def affine_to_list(transform: Any) -> list[float]:
    return [float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)]


def write_tile_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "tile_id",
        "tile_name",
        "row_off",
        "col_off",
        "width",
        "height",
        "source_path",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "roi_intersection_area_m2",
        "focus_intersection_area_m2",
        "selection_reason",
        "image_name",
        "tile_transform",
        "source_transform",
        "crs",
        "epsg",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_tile_geojson(path: Path, rows: list[dict[str, Any]], dataset: Any) -> None:
    features = []
    for row in rows:
        x_min = float(row["x_min"])
        y_min = float(row["y_min"])
        x_max = float(row["x_max"])
        y_max = float(row["y_max"])
        props = {key: value for key, value in row.items() if key not in {"tile_transform", "source_transform"}}
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [x_min, y_min],
                            [x_max, y_min],
                            [x_max, y_max],
                            [x_min, y_max],
                            [x_min, y_min],
                        ]
                    ],
                },
            }
        )
    payload = {
        "type": "FeatureCollection",
        "name": "strict_auto_dom_tile_index",
        "crs": crs_payload(dataset),
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def crs_payload(dataset: Any) -> dict[str, Any] | None:
    if dataset.crs is None:
        return None
    epsg = dataset.crs.to_epsg()
    if epsg is None:
        return {"type": "name", "properties": {"name": dataset.crs.to_string()}}
    return {"type": "name", "properties": {"name": f"EPSG:{epsg}"}}


if __name__ == "__main__":
    raise SystemExit(main())
