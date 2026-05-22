#!/usr/bin/env python3
"""Export non-segmentation CVAT reference geometries into map-space GeoJSON."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rail_curve_extractor.cvat_annotations import TileGeoRecord, load_tile_georef


DEFAULT_ANNOTATIONS = Path("data/cvat_exports/task_1_annotations_20260518_165544_chinese.zip")
DEFAULT_TILE_GEOREF = Path("data/dom_tiles_aligned_annotation/tile_georef.csv")
DEFAULT_OUT_DIR = Path("output/tonghaigang_topology_rebuild_v1")
DEFAULT_LINE_LABEL = "手动绘制的可用于参考的中心线"
DEFAULT_POINT_LABEL = "switch_center"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export CVAT polylines/points as georeferenced reference geometry. "
            "This is meant for topology hints, not for semantic segmentation masks."
        ),
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS, help="CVAT XML/zip export.")
    parser.add_argument("--tile-georef", type=Path, default=DEFAULT_TILE_GEOREF, help="tile_georef.csv/json.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument(
        "--line-label",
        default=DEFAULT_LINE_LABEL,
        help="Polyline label to export. Empty exports all polylines.",
    )
    parser.add_argument(
        "--point-label",
        default=DEFAULT_POINT_LABEL,
        help="Point label to export. Empty exports all point shapes.",
    )
    parser.add_argument("--allow-unmatched", action="store_true", help="Skip CVAT images missing from tile georef.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    annotation_path = args.annotations.expanduser().resolve()
    tile_georef_path = args.tile_georef.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root = load_cvat_xml(annotation_path)
    records = load_tile_georef(tile_georef_path)
    lookup = build_tile_lookup(records)
    crs = geojson_crs(records)

    line_features: list[dict[str, Any]] = []
    point_features: list[dict[str, Any]] = []
    skipped_unmatched: list[str] = []
    image_count = 0
    polyline_shape_count = 0
    point_shape_count = 0
    exported_line_shape_count = 0
    exported_point_count = 0

    for image_el in root.findall("image"):
        image_count += 1
        image_name = image_el.attrib.get("name", "")
        image_width = parse_int(image_el.attrib.get("width"))
        image_height = parse_int(image_el.attrib.get("height"))
        record = match_tile_record(image_name, lookup)
        if record is None:
            skipped_unmatched.append(image_name)
            if not args.allow_unmatched:
                continue
        for shape_index, shape_el in enumerate(image_el, start=1):
            label = shape_el.attrib.get("label", "").strip()
            if shape_el.tag == "polyline":
                polyline_shape_count += 1
                if args.line_label and label != args.line_label:
                    continue
                if record is None:
                    continue
                points = normalize_points(parse_points(shape_el.attrib.get("points", "")), image_width, image_height)
                if len(points) < 2:
                    continue
                line_features.append(
                    line_feature(
                        shape_el=shape_el,
                        image_name=image_name,
                        shape_index=shape_index,
                        record=record,
                        pixel_points=points,
                    ),
                )
                exported_line_shape_count += 1
            elif shape_el.tag == "points":
                point_shape_count += 1
                if args.point_label and label != args.point_label:
                    continue
                if record is None:
                    continue
                points = normalize_points(parse_points(shape_el.attrib.get("points", "")), image_width, image_height)
                for point_index, point in enumerate(points, start=1):
                    point_features.append(
                        point_feature(
                            shape_el=shape_el,
                            image_name=image_name,
                            shape_index=shape_index,
                            point_index=point_index,
                            record=record,
                            pixel_point=point,
                        ),
                    )
                    exported_point_count += 1

    if skipped_unmatched and not args.allow_unmatched:
        examples = ", ".join(skipped_unmatched[:5])
        raise ValueError(f"CVAT images are missing from tile_georef metadata: {examples}")

    line_geojson_path = out_dir / "manual_reference_centerlines.geojson"
    point_geojson_path = out_dir / "switch_centers.geojson"
    all_geojson_path = out_dir / "reference_geometry.geojson"
    summary_path = out_dir / "reference_geometry_summary.json"

    write_geojson(line_geojson_path, "manual_reference_centerlines", line_features, crs)
    write_geojson(point_geojson_path, "switch_centers", point_features, crs)
    write_geojson(all_geojson_path, "cvat_reference_geometry", line_features + point_features, crs)
    summary = {
        "annotation_path": str(annotation_path),
        "tile_georef_path": str(tile_georef_path),
        "output_dir": str(out_dir),
        "image_count": image_count,
        "polyline_shape_count": polyline_shape_count,
        "point_shape_count": point_shape_count,
        "exported_line_shape_count": exported_line_shape_count,
        "exported_point_count": exported_point_count,
        "line_label": args.line_label,
        "point_label": args.point_label,
        "skipped_unmatched_images": skipped_unmatched,
        "manual_reference_centerlines": str(line_geojson_path),
        "switch_centers": str(point_geojson_path),
        "reference_geometry": str(all_geojson_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_cvat_xml(path: Path) -> ET.Element:
    if path.is_dir():
        annotation = path / "annotations.xml"
        if annotation.exists():
            text = annotation.read_text(encoding="utf-8-sig")
        else:
            candidates = sorted(path.glob("*.xml"))
            if not candidates:
                raise FileNotFoundError(f"No CVAT XML file found under: {path}")
            text = candidates[0].read_text(encoding="utf-8-sig")
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist() if name.lower().endswith(".xml"))
            if not names:
                raise ValueError(f"No XML annotation file found inside: {path}")
            names.sort(key=lambda name: (Path(name).name.lower() != "annotations.xml", name))
            text = archive.read(names[0]).decode("utf-8-sig")
    else:
        text = path.read_text(encoding="utf-8-sig")
    root = ET.fromstring(text)
    if root.tag != "annotations":
        raise ValueError("Expected a CVAT for images XML file with <annotations> root.")
    return root


def build_tile_lookup(records: dict[str, TileGeoRecord]) -> dict[str, TileGeoRecord]:
    lookup: dict[str, TileGeoRecord] = {}
    for record in records.values():
        keys = {
            norm_key(record.tile_name),
            norm_key(Path(record.tile_name).name),
            norm_key(record.image_path),
            norm_key(Path(record.image_path).name),
        }
        for key in keys:
            if key:
                lookup.setdefault(key, record)
    return lookup


def match_tile_record(image_name: str, lookup: dict[str, TileGeoRecord]) -> TileGeoRecord | None:
    keys = (
        norm_key(image_name),
        norm_key(Path(image_name.replace("\\", "/")).name),
    )
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def norm_key(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name.lower()


def parse_int(value: str | None) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def parse_optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def parse_points(points_text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for raw_point in points_text.split(";"):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        raw_x, raw_y = raw_point.split(",", maxsplit=1)
        points.append((float(raw_x), float(raw_y)))
    return points


def normalize_points(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    if width <= 0 or height <= 0:
        return points
    return [(min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))) for x, y in points]


def parse_attributes(shape_el: ET.Element) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for attribute_el in shape_el.findall("attribute"):
        name = attribute_el.attrib.get("name", "").strip()
        if name:
            attributes[name] = (attribute_el.text or "").strip()
    return attributes


def pixel_to_map(
    transform: tuple[float, float, float, float, float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    col, row = point
    return a * col + b * row + c, d * col + e * row + f


def line_feature(
    *,
    shape_el: ET.Element,
    image_name: str,
    shape_index: int,
    record: TileGeoRecord,
    pixel_points: list[tuple[float, float]],
) -> dict[str, Any]:
    map_points = [pixel_to_map(record.tile_transform, point) for point in pixel_points]
    return {
        "type": "Feature",
        "properties": common_properties(
            shape_el=shape_el,
            image_name=image_name,
            shape_index=shape_index,
            record=record,
            shape_type="polyline",
            point_count=len(pixel_points),
            extra={"length_m": round(polyline_length(map_points), 3)},
            pixel_points=pixel_points,
        ),
        "geometry": {
            "type": "LineString",
            "coordinates": [[round_coord(x), round_coord(y)] for x, y in map_points],
        },
    }


def point_feature(
    *,
    shape_el: ET.Element,
    image_name: str,
    shape_index: int,
    point_index: int,
    record: TileGeoRecord,
    pixel_point: tuple[float, float],
) -> dict[str, Any]:
    x, y = pixel_to_map(record.tile_transform, pixel_point)
    return {
        "type": "Feature",
        "properties": common_properties(
            shape_el=shape_el,
            image_name=image_name,
            shape_index=shape_index,
            record=record,
            shape_type="points",
            point_count=1,
            extra={"point_index": point_index},
            pixel_points=[pixel_point],
        ),
        "geometry": {
            "type": "Point",
            "coordinates": [round_coord(x), round_coord(y)],
        },
    }


def common_properties(
    *,
    shape_el: ET.Element,
    image_name: str,
    shape_index: int,
    record: TileGeoRecord,
    shape_type: str,
    point_count: int,
    extra: dict[str, Any],
    pixel_points: list[tuple[float, float]],
) -> dict[str, Any]:
    props: dict[str, Any] = {
        "image_name": image_name,
        "tile_name": record.tile_name,
        "label": shape_el.attrib.get("label", "").strip(),
        "shape_type": shape_type,
        "shape_index": shape_index,
        "point_count": point_count,
        "occluded": shape_el.attrib.get("occluded") == "1",
        "z_order": parse_optional_int(shape_el.attrib.get("z_order")),
        "epsg": record.epsg,
        "pixel_points": [[round_coord(x), round_coord(y)] for x, y in pixel_points],
        "attributes": parse_attributes(shape_el),
    }
    props.update(extra)
    return props


def polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for left, right in zip(points, points[1:]):
        total += math.hypot(right[0] - left[0], right[1] - left[1])
    return total


def geojson_crs(records: dict[str, TileGeoRecord]) -> dict[str, Any] | None:
    epsg_values = sorted({record.epsg for record in records.values() if record.epsg is not None})
    if len(epsg_values) == 1:
        return {"type": "name", "properties": {"name": f"EPSG:{epsg_values[0]}"}}
    return None


def write_geojson(path: Path, name: str, features: list[dict[str, Any]], crs: dict[str, Any] | None) -> None:
    payload: dict[str, Any] = {
        "type": "FeatureCollection",
        "name": name,
        "features": features,
    }
    if crs is not None:
        payload["crs"] = crs
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def round_coord(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
