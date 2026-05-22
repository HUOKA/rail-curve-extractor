#!/usr/bin/env python3
"""Build a raw-DOM rail corridor ROI and tile index from existing map geometry."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, NamedTuple
import warnings

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NodataShadowWarning
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import LineString, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry


DEFAULT_RAW_DOM = Path("data/生产数据/无人机数据/正射/dom.tif")
DEFAULT_LINEWORK = Path("output/tonghaigang_topology_rebuild_v1/topology_skeleton_linework.geojson")
DEFAULT_MANUAL_REFERENCE = Path("output/tonghaigang_topology_rebuild_v1/manual_reference_centerlines.geojson")
DEFAULT_SWITCH_WORKZONES = Path("output/tonghaigang_topology_rebuild_v1/switch_model_workzones.geojson")
DEFAULT_ALIGNED_DOM = Path("data/aligned_dom/aligned_dom.tif")
DEFAULT_OUT_DIR = Path("output/raw_dom_corridor_roi")


class Axis(NamedTuple):
    origin: np.ndarray
    longitudinal: np.ndarray
    lateral: np.ndarray


class EvidenceGeometry(NamedTuple):
    source_name: str
    feature_id: str
    geometry: BaseGeometry
    properties: dict[str, Any]


class TileRecord(NamedTuple):
    tile_id: int
    tile_name: str
    row_off: int
    col_off: int
    width: int
    height: int
    source_width: int
    source_height: int
    source_path: str
    tile_transform: str
    source_transform: str
    crs: str | None
    epsg: int | None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    roi_intersection_area_m2: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build raw DOM rail corridor ROI and segmentation tile index.")
    parser.add_argument("--dom", type=Path, default=DEFAULT_RAW_DOM)
    parser.add_argument("--linework", type=Path, default=DEFAULT_LINEWORK)
    parser.add_argument("--manual-reference", type=Path, default=DEFAULT_MANUAL_REFERENCE)
    parser.add_argument("--switch-workzones", type=Path, default=DEFAULT_SWITCH_WORKZONES)
    parser.add_argument("--aligned-dom", type=Path, default=DEFAULT_ALIGNED_DOM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--longitudinal-buffer-m", type=float, default=100.0)
    parser.add_argument("--lateral-buffer-m", type=float, default=50.0)
    parser.add_argument("--tile-width", type=int, default=3072)
    parser.add_argument("--tile-height", type=int, default=3072)
    parser.add_argument("--tile-overlap", type=float, default=0.5)
    parser.add_argument("--overview-max-width", type=int, default=1600)
    parser.add_argument("--overview-max-height", type=int, default=5200)
    parser.add_argument("--contact-sheet-max", type=int, default=18)
    parser.add_argument("--contact-cell-width", type=int, default=360)
    parser.add_argument("--contact-cell-height", type=int, default=300)
    parser.add_argument("--prefix", default="raw_dom_roi")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dom_path = args.dom.expanduser().resolve()
    evidence = load_evidence(args)
    if not evidence:
        raise ValueError("No ROI evidence geometries were loaded.")

    with rasterio.open(dom_path) as dataset:
        if dataset.crs is None:
            raise ValueError("Raw DOM must have CRS metadata.")
        axis = estimate_axis(evidence)
        roi_nominal = build_oriented_roi(
            evidence=evidence,
            axis=axis,
            longitudinal_buffer_m=args.longitudinal_buffer_m,
            lateral_buffer_m=args.lateral_buffer_m,
        )
        dom_extent = box(*dataset.bounds)
        roi = roi_nominal.intersection(dom_extent)
        if roi.is_empty:
            raise ValueError("Computed rail corridor ROI does not intersect the raw DOM bounds.")

        tile_records, tile_features = build_tile_index(
            dataset=dataset,
            roi=roi,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            tile_overlap=args.tile_overlap,
            prefix=args.prefix,
        )
        if not tile_records:
            raise ValueError("ROI produced no raw DOM tiles.")

        roi_path = out_dir / "rail_corridor_roi.geojson"
        tile_geojson_path = out_dir / "raw_dom_tile_index.geojson"
        tile_csv_path = out_dir / "raw_dom_tile_index.csv"
        overview_path = out_dir / "overview_corridor_on_raw_dom.jpg"
        contact_sheet_path = out_dir / "tile_contact_sheet.jpg"
        summary_path = out_dir / "summary.json"

        write_roi_geojson(
            roi_path,
            roi=roi,
            roi_nominal=roi_nominal,
            axis=axis,
            args=args,
            dataset=dataset,
            evidence=evidence,
        )
        write_geojson(tile_geojson_path, "raw_dom_tile_index", tile_features, dataset)
        write_tile_csv(tile_csv_path, tile_records)
        write_overview_image(
            overview_path,
            dataset=dataset,
            roi=roi,
            tile_features=tile_features,
            max_width=args.overview_max_width,
            max_height=args.overview_max_height,
        )
        write_contact_sheet(
            contact_sheet_path,
            dataset=dataset,
            roi=roi,
            tile_records=tile_records,
            max_items=args.contact_sheet_max,
            cell_width=args.contact_cell_width,
            cell_height=args.contact_cell_height,
        )

        roi_pixel_bounds = pixel_bounds_for_geometry(dataset, roi)
        summary = {
            "raw_dom": str(dom_path),
            "output_dir": str(out_dir),
            "dom_width": int(dataset.width),
            "dom_height": int(dataset.height),
            "dom_crs": dataset.crs.to_string() if dataset.crs else None,
            "dom_epsg": dataset.crs.to_epsg() if dataset.crs else None,
            "dom_transform": affine_to_list(dataset.transform),
            "evidence_sources": evidence_summary(evidence),
            "axis_origin": [round(float(axis.origin[0]), 6), round(float(axis.origin[1]), 6)],
            "axis_longitudinal": [round(float(axis.longitudinal[0]), 8), round(float(axis.longitudinal[1]), 8)],
            "axis_lateral": [round(float(axis.lateral[0]), 8), round(float(axis.lateral[1]), 8)],
            "longitudinal_buffer_m": args.longitudinal_buffer_m,
            "lateral_buffer_m": args.lateral_buffer_m,
            "roi_area_m2": float(roi.area),
            "roi_bounds": [float(value) for value in roi.bounds],
            "roi_pixel_bounds": roi_pixel_bounds,
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "tile_overlap": args.tile_overlap,
            "tile_count": len(tile_records),
            "outputs": {
                "rail_corridor_roi_geojson": str(roi_path),
                "raw_dom_tile_index_geojson": str(tile_geojson_path),
                "raw_dom_tile_index_csv": str(tile_csv_path),
                "overview_corridor_on_raw_dom": str(overview_path),
                "tile_contact_sheet": str(contact_sheet_path),
                "summary_json": str(summary_path),
            },
            "interpretation": (
                "Use this ROI as the first raw-DOM semantic segmentation target. "
                "It is derived from existing EPSG:32651 geometry evidence, not from raw-DOM predictions."
            ),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_evidence(args: argparse.Namespace) -> list[EvidenceGeometry]:
    evidence: list[EvidenceGeometry] = []
    evidence.extend(load_geojson_geometries(args.linework, "topology_skeleton_linework"))
    evidence.extend(load_geojson_geometries(args.manual_reference, "manual_reference_centerlines"))
    evidence.extend(load_geojson_geometries(args.switch_workzones, "switch_model_workzones"))
    aligned_dom = args.aligned_dom.expanduser().resolve()
    if aligned_dom.exists():
        with rasterio.open(aligned_dom) as dataset:
            evidence.append(
                EvidenceGeometry(
                    source_name="aligned_dom_extent",
                    feature_id="aligned_dom_bounds",
                    geometry=raster_footprint_polygon(dataset),
                    properties={
                        "path": str(aligned_dom),
                        "width": int(dataset.width),
                        "height": int(dataset.height),
                    },
                ),
            )
    return [item for item in evidence if not item.geometry.is_empty]


def raster_footprint_polygon(dataset: rasterio.io.DatasetReader) -> Polygon:
    corners = [
        dataset.transform * (0.0, 0.0),
        dataset.transform * (float(dataset.width), 0.0),
        dataset.transform * (float(dataset.width), float(dataset.height)),
        dataset.transform * (0.0, float(dataset.height)),
        dataset.transform * (0.0, 0.0),
    ]
    return Polygon([(float(x), float(y)) for x, y in corners])


def load_geojson_geometries(path: Path, source_name: str) -> list[EvidenceGeometry]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return []
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    result: list[EvidenceGeometry] = []
    for index, feature in enumerate(payload.get("features", []) or [], start=1):
        geometry_payload = feature.get("geometry")
        if not geometry_payload:
            continue
        geometry = shape(geometry_payload)
        if geometry.is_empty:
            continue
        props = dict(feature.get("properties") or {})
        feature_id = str(
            props.get("line_id")
            or props.get("workzone_id")
            or props.get("id")
            or props.get("name")
            or f"{source_name}_{index:03d}"
        )
        result.append(EvidenceGeometry(source_name, feature_id, geometry, props))
    return result


def estimate_axis(evidence: list[EvidenceGeometry]) -> Axis:
    for item in evidence:
        if item.properties.get("topology_role") == "main_through_track" and isinstance(item.geometry, LineString):
            coords = np.asarray(item.geometry.coords, dtype=float)[:, :2]
            return axis_from_line(coords)
    coords = geometry_coordinate_matrix([item.geometry for item in evidence])
    if coords.shape[0] < 2:
        raise ValueError("Need at least two evidence coordinates to estimate axis.")
    origin = coords.mean(axis=0)
    centered = coords - origin
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    longitudinal = vh[0]
    if longitudinal[1] < 0:
        longitudinal = -longitudinal
    lateral = np.array([-longitudinal[1], longitudinal[0]])
    return Axis(origin=origin, longitudinal=longitudinal, lateral=lateral)


def axis_from_line(coords: np.ndarray) -> Axis:
    start = coords[0]
    end = coords[-1]
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise ValueError("Main track line is too short to define an axis.")
    longitudinal = direction / norm
    if longitudinal[1] < 0:
        longitudinal = -longitudinal
    origin = coords.mean(axis=0)
    lateral = np.array([-longitudinal[1], longitudinal[0]])
    return Axis(origin=origin, longitudinal=longitudinal, lateral=lateral)


def build_oriented_roi(
    *,
    evidence: list[EvidenceGeometry],
    axis: Axis,
    longitudinal_buffer_m: float,
    lateral_buffer_m: float,
) -> Polygon:
    coords = geometry_coordinate_matrix([item.geometry for item in evidence])
    s_values, t_values = project_coords(axis, coords)
    s_min = float(np.min(s_values)) - longitudinal_buffer_m
    s_max = float(np.max(s_values)) + longitudinal_buffer_m
    t_min = float(np.min(t_values)) - lateral_buffer_m
    t_max = float(np.max(t_values)) + lateral_buffer_m
    corners_st = [(s_min, t_min), (s_max, t_min), (s_max, t_max), (s_min, t_max), (s_min, t_min)]
    return Polygon([unproject_point(axis, s, t) for s, t in corners_st])


def geometry_coordinate_matrix(geometries: list[BaseGeometry]) -> np.ndarray:
    coords: list[tuple[float, float]] = []
    for geometry in geometries:
        coords.extend(collect_geometry_coords(geometry))
    if not coords:
        return np.empty((0, 2), dtype=float)
    return np.asarray(coords, dtype=float)


def collect_geometry_coords(geometry: BaseGeometry) -> list[tuple[float, float]]:
    if geometry.geom_type == "Point":
        return [(float(geometry.x), float(geometry.y))]
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [(float(x), float(y)) for x, y, *_ in geometry.coords]
    if geometry.geom_type == "Polygon":
        coords = [(float(x), float(y)) for x, y, *_ in geometry.exterior.coords]
        for interior in geometry.interiors:
            coords.extend((float(x), float(y)) for x, y, *_ in interior.coords)
        return coords
    if hasattr(geometry, "geoms"):
        coords: list[tuple[float, float]] = []
        for part in geometry.geoms:
            coords.extend(collect_geometry_coords(part))
        return coords
    return []


def project_coords(axis: Axis, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = coords - axis.origin
    return centered @ axis.longitudinal, centered @ axis.lateral


def unproject_point(axis: Axis, s_value: float, t_value: float) -> tuple[float, float]:
    point = axis.origin + s_value * axis.longitudinal + t_value * axis.lateral
    return float(point[0]), float(point[1])


def build_tile_index(
    *,
    dataset: rasterio.io.DatasetReader,
    roi: BaseGeometry,
    tile_width: int,
    tile_height: int,
    tile_overlap: float,
    prefix: str,
) -> tuple[list[TileRecord], list[dict[str, Any]]]:
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Tile width and height must be positive.")
    if not 0.0 <= tile_overlap < 1.0:
        raise ValueError("Tile overlap must be in [0, 1).")
    stride_x = max(1, int(round(tile_width * (1.0 - tile_overlap))))
    stride_y = max(1, int(round(tile_height * (1.0 - tile_overlap))))
    col_min, row_min, col_max, row_max = pixel_bounds_for_geometry(dataset, roi)
    col_offsets = axis_offsets_for_interval(col_min, col_max, dataset.width, tile_width, stride_x)
    row_offsets = axis_offsets_for_interval(row_min, row_max, dataset.height, tile_height, stride_y)
    records: list[TileRecord] = []
    features: list[dict[str, Any]] = []
    source_transform = json.dumps(affine_to_list(dataset.transform), ensure_ascii=False)
    crs = dataset.crs.to_string() if dataset.crs else None
    epsg = dataset.crs.to_epsg() if dataset.crs else None
    source_path = str(Path(dataset.name).resolve())
    tile_id = 0
    for row_off in row_offsets:
        for col_off in col_offsets:
            width = min(tile_width, dataset.width - col_off)
            height = min(tile_height, dataset.height - row_off)
            if width <= 0 or height <= 0:
                continue
            window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
            tile_polygon = window_polygon(dataset.transform, window)
            intersection_area = float(tile_polygon.intersection(roi).area)
            if intersection_area <= 0.0:
                continue
            x_min, y_min, x_max, y_max = tile_polygon.bounds
            tile_name = f"{prefix}_r{row_off:06d}_c{col_off:06d}.png"
            tile_transform = json.dumps(affine_to_list(dataset.window_transform(window)), ensure_ascii=False)
            record = TileRecord(
                tile_id=tile_id,
                tile_name=tile_name,
                row_off=int(row_off),
                col_off=int(col_off),
                width=int(width),
                height=int(height),
                source_width=int(dataset.width),
                source_height=int(dataset.height),
                source_path=source_path,
                tile_transform=tile_transform,
                source_transform=source_transform,
                crs=crs,
                epsg=epsg,
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max),
                roi_intersection_area_m2=intersection_area,
            )
            records.append(record)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "tile_id": tile_id,
                        "tile_name": tile_name,
                        "row_off": int(row_off),
                        "col_off": int(col_off),
                        "width": int(width),
                        "height": int(height),
                        "roi_intersection_area_m2": round(intersection_area, 3),
                        "epsg": epsg,
                    },
                    "geometry": mapping(tile_polygon),
                },
            )
            tile_id += 1
    return records, features


def axis_offsets_for_interval(start: int, stop: int, max_length: int, tile_size: int, stride: int) -> list[int]:
    start = max(0, min(start, max_length))
    stop = max(start + 1, min(stop, max_length))
    max_offset = max(0, max_length - tile_size)
    if stop - start <= tile_size:
        center_offset = int(round((start + stop - tile_size) / 2.0))
        return [max(0, min(center_offset, max_offset))]
    offsets = list(range(start, max(start + 1, stop - tile_size + 1), stride))
    final_offset = min(stop - tile_size, max_offset)
    if not offsets or offsets[-1] != final_offset:
        offsets.append(final_offset)
    return sorted({max(0, min(offset, max_offset)) for offset in offsets})


def pixel_bounds_for_geometry(dataset: rasterio.io.DatasetReader, geometry: BaseGeometry) -> dict[str, int] | tuple[int, int, int, int]:
    inverse = ~dataset.transform
    cols: list[float] = []
    rows: list[float] = []
    for x, y in collect_geometry_coords(geometry):
        col, row = inverse * (x, y)
        cols.append(float(col))
        rows.append(float(row))
    if not cols:
        raise ValueError("Geometry has no coordinates for pixel bounds.")
    col_min = max(0, int(math.floor(min(cols))))
    col_max = min(dataset.width, int(math.ceil(max(cols))))
    row_min = max(0, int(math.floor(min(rows))))
    row_max = min(dataset.height, int(math.ceil(max(rows))))
    if col_max <= col_min or row_max <= row_min:
        raise ValueError("Geometry pixel bounds are empty.")
    return col_min, row_min, col_max, row_max


def window_polygon(transform: Affine, window: Window) -> Polygon:
    col0 = float(window.col_off)
    row0 = float(window.row_off)
    col1 = col0 + float(window.width)
    row1 = row0 + float(window.height)
    return Polygon([transform * point for point in ((col0, row0), (col1, row0), (col1, row1), (col0, row1), (col0, row0))])


def write_roi_geojson(
    path: Path,
    *,
    roi: BaseGeometry,
    roi_nominal: BaseGeometry,
    axis: Axis,
    args: argparse.Namespace,
    dataset: rasterio.io.DatasetReader,
    evidence: list[EvidenceGeometry],
) -> None:
    pixel_bounds = pixel_bounds_for_geometry(dataset, roi)
    feature = {
        "type": "Feature",
        "properties": {
            "name": "raw_dom_rail_corridor_roi",
            "source": "existing_epsg32651_geometry_evidence",
            "longitudinal_buffer_m": args.longitudinal_buffer_m,
            "lateral_buffer_m": args.lateral_buffer_m,
            "nominal_area_m2": round(float(roi_nominal.area), 3),
            "clipped_area_m2": round(float(roi.area), 3),
            "pixel_bounds": list(pixel_bounds),
            "axis_origin": [round(float(axis.origin[0]), 6), round(float(axis.origin[1]), 6)],
            "axis_longitudinal": [round(float(axis.longitudinal[0]), 8), round(float(axis.longitudinal[1]), 8)],
            "axis_lateral": [round(float(axis.lateral[0]), 8), round(float(axis.lateral[1]), 8)],
            "evidence_sources": evidence_summary(evidence),
        },
        "geometry": mapping(roi),
    }
    write_geojson(path, "raw_dom_rail_corridor_roi", [feature], dataset)


def write_geojson(path: Path, name: str, features: list[dict[str, Any]], dataset: rasterio.io.DatasetReader) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": name,
        "crs": {
            "type": "name",
            "properties": {"name": dataset.crs.to_string() if dataset.crs else None},
        },
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_tile_csv(path: Path, records: list[TileRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TileRecord._fields))
        writer.writeheader()
        for record in records:
            writer.writerow(record._asdict())


def write_overview_image(
    path: Path,
    *,
    dataset: rasterio.io.DatasetReader,
    roi: BaseGeometry,
    tile_features: list[dict[str, Any]],
    max_width: int,
    max_height: int,
) -> None:
    preview_width, preview_height = preview_shape(dataset.width, dataset.height, max_width, max_height)
    rgb = read_rgb_preview(dataset, preview_width, preview_height)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw_geometry(draw, dataset, roi, preview_width, preview_height, outline=(255, 230, 0), width=4)
    for feature in tile_features:
        draw_geometry(
            draw,
            dataset,
            shape(feature["geometry"]),
            preview_width,
            preview_height,
            outline=(60, 180, 255),
            width=1,
        )
    draw.text((16, 16), f"raw DOM rail corridor ROI | tiles: {len(tile_features)}", fill=(255, 255, 255))
    image.save(path, quality=92)


def write_contact_sheet(
    path: Path,
    *,
    dataset: rasterio.io.DatasetReader,
    roi: BaseGeometry,
    tile_records: list[TileRecord],
    max_items: int,
    cell_width: int,
    cell_height: int,
) -> None:
    selected = evenly_sample(tile_records, max_items)
    if not selected:
        return
    columns = min(3, len(selected))
    rows = (len(selected) + columns - 1) // columns
    label_height = 28
    margin = 8
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(selected):
        col = index % columns
        row = index // columns
        cell_x = col * cell_width
        cell_y = row * cell_height
        draw.rectangle((cell_x, cell_y, cell_x + cell_width - 1, cell_y + cell_height - 1), outline=(180, 180, 174))
        draw.text((cell_x + 6, cell_y + 6), record.tile_name.replace(".png", ""), fill=(30, 30, 30))
        image_box = (cell_x + margin, cell_y + label_height, cell_x + cell_width - margin, cell_y + cell_height - margin)
        thumb = read_tile_thumbnail(dataset, record, image_box[2] - image_box[0], image_box[3] - image_box[1])
        sheet.paste(thumb, (image_box[0], image_box[1]))
        draw_roi_in_tile(draw, dataset, roi, record, image_box, outline=(255, 230, 0), width=3)
    sheet.save(path, quality=92)


def read_rgb_preview(dataset: rasterio.io.DatasetReader, width: int, height: int) -> np.ndarray:
    band_indexes = [1] if dataset.count == 1 else ([1, 2, 3] if dataset.count >= 3 else list(range(1, dataset.count + 1)))
    data = dataset.read(band_indexes, out_shape=(len(band_indexes), height, width), resampling=Resampling.bilinear)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NodataShadowWarning)
        mask = dataset.dataset_mask(out_shape=(height, width), resampling=Resampling.nearest)
    rgb = to_uint8(data)
    if rgb.shape[0] == 1:
        rgb = np.repeat(rgb, 3, axis=0)
    rgb = np.moveaxis(rgb[:3], 0, -1).copy()
    rgb[mask == 0] = 0
    return rgb


def read_tile_thumbnail(dataset: rasterio.io.DatasetReader, record: TileRecord, width: int, height: int) -> Image.Image:
    window = Window(record.col_off, record.row_off, record.width, record.height)
    band_indexes = [1] if dataset.count == 1 else ([1, 2, 3] if dataset.count >= 3 else list(range(1, dataset.count + 1)))
    data = dataset.read(band_indexes, window=window, out_shape=(len(band_indexes), height, width), resampling=Resampling.bilinear)
    rgb = to_uint8(data)
    if rgb.shape[0] == 1:
        rgb = np.repeat(rgb, 3, axis=0)
    return Image.fromarray(np.moveaxis(rgb[:3], 0, -1), mode="RGB")


def draw_geometry(
    draw: ImageDraw.ImageDraw,
    dataset: rasterio.io.DatasetReader,
    geometry: BaseGeometry,
    preview_width: int,
    preview_height: int,
    *,
    outline: tuple[int, int, int],
    width: int,
) -> None:
    for coords in geometry_line_coords(geometry):
        points = [map_to_preview_pixel(dataset, x, y, preview_width, preview_height) for x, y in coords]
        if len(points) >= 2:
            draw.line(points, fill=outline, width=width, joint="curve")


def draw_roi_in_tile(
    draw: ImageDraw.ImageDraw,
    dataset: rasterio.io.DatasetReader,
    roi: BaseGeometry,
    record: TileRecord,
    image_box: tuple[int, int, int, int],
    *,
    outline: tuple[int, int, int],
    width: int,
) -> None:
    tile_poly = window_polygon(dataset.transform, Window(record.col_off, record.row_off, record.width, record.height))
    intersection = roi.intersection(tile_poly)
    if intersection.is_empty:
        return
    left, top, right, bottom = image_box
    scale_x = (right - left) / max(record.width, 1)
    scale_y = (bottom - top) / max(record.height, 1)
    inverse = ~dataset.transform
    for coords in geometry_line_coords(intersection):
        points = []
        for x, y in coords:
            col, row = inverse * (x, y)
            points.append((left + (col - record.col_off) * scale_x, top + (row - record.row_off) * scale_y))
        if len(points) >= 2:
            draw.line(points, fill=outline, width=width, joint="curve")


def geometry_line_coords(geometry: BaseGeometry) -> list[list[tuple[float, float]]]:
    if geometry.geom_type == "Polygon":
        return [[(float(x), float(y)) for x, y, *_ in geometry.exterior.coords]]
    if geometry.geom_type == "LineString":
        return [[(float(x), float(y)) for x, y, *_ in geometry.coords]]
    if hasattr(geometry, "geoms"):
        lines: list[list[tuple[float, float]]] = []
        for part in geometry.geoms:
            lines.extend(geometry_line_coords(part))
        return lines
    return []


def map_to_preview_pixel(
    dataset: rasterio.io.DatasetReader,
    x: float,
    y: float,
    preview_width: int,
    preview_height: int,
) -> tuple[float, float]:
    col, row = (~dataset.transform) * (x, y)
    return float(col * preview_width / dataset.width), float(row * preview_height / dataset.height)


def preview_shape(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


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


def evenly_sample(records: list[TileRecord], max_items: int) -> list[TileRecord]:
    if max_items <= 0 or len(records) <= max_items:
        return records
    indexes = np.linspace(0, len(records) - 1, max_items).round().astype(int)
    return [records[int(index)] for index in sorted(set(indexes))]


def affine_to_list(transform: Affine) -> list[float]:
    return [float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f)]


def evidence_summary(evidence: list[EvidenceGeometry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence:
        counts[item.source_name] = counts.get(item.source_name, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
