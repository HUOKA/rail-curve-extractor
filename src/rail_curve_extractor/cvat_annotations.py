from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
from pathlib import Path
import shutil
import zipfile
import xml.etree.ElementTree as ET


@dataclass(frozen=True, slots=True)
class TileGeoRecord:
    tile_name: str
    image_path: str
    row_off: int
    col_off: int
    width: int
    height: int
    tile_transform: tuple[float, float, float, float, float, float]
    crs: str | None
    epsg: int | None


@dataclass(frozen=True, slots=True)
class CvatShape:
    image_name: str
    image_width: int
    image_height: int
    label: str
    shape_type: str
    points: tuple[tuple[float, float], ...]
    occluded: bool = False
    z_order: int | None = None
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CvatImage:
    name: str
    width: int
    height: int
    shapes: tuple[CvatShape, ...]


@dataclass(frozen=True, slots=True)
class CvatAnnotationSet:
    source_path: str
    version: str | None
    labels: tuple[str, ...]
    images: tuple[CvatImage, ...]


@dataclass(frozen=True, slots=True)
class CvatConversionOptions:
    class_names: tuple[str, ...] = ()
    copy_images: bool = True
    make_overlays: bool = False
    include_empty_labels: bool = True
    allow_unmatched: bool = False
    skip_unknown_labels: bool = False


@dataclass(frozen=True, slots=True)
class CvatConversionResult:
    annotation_path: str
    tile_georef_path: str
    output_dir: str
    images_dir: str
    labels_dir: str
    overlays_dir: str | None
    classes_path: str
    geojson_path: str
    manifest_csv_path: str
    manifest_json_path: str
    summary_path: str
    image_count: int
    matched_image_count: int
    copied_image_count: int
    shape_count: int
    skipped_shape_count: int
    unmatched_images: list[str]
    class_names: list[str]


def convert_cvat_annotations(
    annotation_path: Path,
    tile_georef_path: Path,
    output_dir: Path,
    options: CvatConversionOptions | None = None,
) -> CvatConversionResult:
    options = options or CvatConversionOptions()
    annotation_set = load_cvat_annotations(annotation_path)
    tile_records = load_tile_georef(tile_georef_path)
    class_names = _resolve_class_names(annotation_set, options.class_names)
    class_index = {name: index for index, name in enumerate(class_names)}
    output_dir = output_dir.expanduser().resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    overlays_dir = output_dir / "overlays" if options.make_overlays else None
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if options.copy_images:
        images_dir.mkdir(parents=True, exist_ok=True)
    if overlays_dir is not None:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    tile_lookup = _build_tile_lookup(tile_records)
    unmatched_images: list[str] = []
    matched_images: list[tuple[CvatImage, TileGeoRecord]] = []
    for image in annotation_set.images:
        record = _match_tile_record(image.name, tile_lookup)
        if record is None:
            unmatched_images.append(image.name)
            continue
        matched_images.append((image, record))
    if unmatched_images and not options.allow_unmatched:
        examples = ", ".join(unmatched_images[:5])
        raise ValueError(f"CVAT images are missing from tile_georef metadata: {examples}")

    features: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    copied_image_count = 0
    shape_count = 0
    skipped_shape_count = 0
    for image, record in matched_images:
        label_path = labels_dir / f"{Path(record.tile_name).stem}.txt"
        label_lines: list[str] = []
        image_shapes: list[CvatShape] = []
        for shape in image.shapes:
            if shape.label not in class_index:
                if options.skip_unknown_labels:
                    skipped_shape_count += 1
                    continue
                raise ValueError(f"CVAT label is not listed in classes: {shape.label}")
            polygon = _clipped_polygon(shape.points, image.width, image.height)
            if len(polygon) < 3 or abs(_polygon_area(polygon)) <= 1e-6:
                skipped_shape_count += 1
                continue
            class_id = class_index[shape.label]
            map_polygon = tuple(_pixel_to_map(record.tile_transform, point) for point in polygon)
            label_lines.append(_yolo_segmentation_line(class_id, polygon, image.width, image.height))
            features.append(_geojson_feature(shape, record, polygon, map_polygon, class_id))
            manifest_rows.append(
                _manifest_row(
                    shape=shape,
                    record=record,
                    class_id=class_id,
                    polygon=polygon,
                    map_polygon=map_polygon,
                    label_path=label_path,
                )
            )
            image_shapes.append(shape)
            shape_count += 1
        if label_lines or options.include_empty_labels:
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        if options.copy_images:
            copied_image_count += _copy_tile_image(record, images_dir)
        if overlays_dir is not None:
            _write_overlay(record, image_shapes, class_index, overlays_dir)

    classes_path = output_dir / "classes.txt"
    classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    geojson_path = output_dir / "annotations_map.geojson"
    _write_json(
        geojson_path,
        {
            "type": "FeatureCollection",
            "name": "cvat_annotations_map",
            "crs": _geojson_crs(tile_records),
            "features": features,
        },
    )
    manifest_csv_path = output_dir / "manifest.csv"
    manifest_json_path = output_dir / "manifest.json"
    _write_manifest_csv(manifest_csv_path, manifest_rows)
    _write_json(manifest_json_path, manifest_rows)
    summary_path = output_dir / "summary.json"
    result = CvatConversionResult(
        annotation_path=str(annotation_path.expanduser().resolve()),
        tile_georef_path=str(tile_georef_path.expanduser().resolve()),
        output_dir=str(output_dir),
        images_dir=str(images_dir),
        labels_dir=str(labels_dir),
        overlays_dir=str(overlays_dir) if overlays_dir is not None else None,
        classes_path=str(classes_path),
        geojson_path=str(geojson_path),
        manifest_csv_path=str(manifest_csv_path),
        manifest_json_path=str(manifest_json_path),
        summary_path=str(summary_path),
        image_count=len(annotation_set.images),
        matched_image_count=len(matched_images),
        copied_image_count=copied_image_count,
        shape_count=shape_count,
        skipped_shape_count=skipped_shape_count,
        unmatched_images=unmatched_images,
        class_names=list(class_names),
    )
    _write_json(summary_path, asdict(result))
    if unmatched_images:
        _write_json(output_dir / "unmatched_images.json", unmatched_images)
    return result


def load_tile_georef(path: Path) -> dict[str, TileGeoRecord]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"tile_georef metadata does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("records", payload if isinstance(payload, list) else [])
    else:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    records: dict[str, TileGeoRecord] = {}
    for row in rows:
        record = _tile_record_from_row(row)
        records[record.tile_name] = record
    if not records:
        raise ValueError(f"No tile records found in: {path}")
    return records


def load_cvat_annotations(path: Path) -> CvatAnnotationSet:
    xml_text = _read_cvat_xml_text(path)
    root = ET.fromstring(xml_text)
    if root.tag != "annotations":
        raise ValueError("Expected a CVAT for images XML file with <annotations> as the root element.")
    labels = _parse_cvat_labels(root)
    images: list[CvatImage] = []
    for image_el in root.findall("image"):
        name = image_el.attrib.get("name", "")
        width = _int_attr(image_el, "width")
        height = _int_attr(image_el, "height")
        shapes = tuple(_parse_image_shapes(image_el, name, width, height))
        images.append(CvatImage(name=name, width=width, height=height, shapes=shapes))
    return CvatAnnotationSet(
        source_path=str(path.expanduser().resolve()),
        version=root.attrib.get("version"),
        labels=tuple(labels),
        images=tuple(images),
    )


def _read_cvat_xml_text(path: Path) -> str:
    path = path.expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.xml"))
        annotations = path / "annotations.xml"
        if annotations.exists():
            return annotations.read_text(encoding="utf-8-sig")
        if candidates:
            return candidates[0].read_text(encoding="utf-8-sig")
        raise FileNotFoundError(f"No CVAT XML file found under: {path}")
    if not path.exists():
        raise FileNotFoundError(f"CVAT annotation input does not exist: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if not xml_names:
                raise ValueError(f"No XML annotation file found inside: {path}")
            xml_names.sort(key=lambda name: (Path(name).name.lower() != "annotations.xml", name))
            return archive.read(xml_names[0]).decode("utf-8-sig")
    return path.read_text(encoding="utf-8-sig")


def _parse_cvat_labels(root: ET.Element) -> list[str]:
    labels: list[str] = []
    for name_el in root.findall(".//labels/label/name"):
        name = (name_el.text or "").strip()
        if name and name not in labels:
            labels.append(name)
    return labels


def _parse_image_shapes(image_el: ET.Element, image_name: str, width: int, height: int) -> list[CvatShape]:
    shapes: list[CvatShape] = []
    for shape_el in image_el:
        if shape_el.tag == "polygon":
            points = _parse_points(shape_el.attrib.get("points", ""))
            shape_type = "polygon"
        elif shape_el.tag == "box":
            points = _box_points(shape_el)
            shape_type = "box"
        else:
            continue
        points = _remove_duplicate_closing_point(points)
        if len(points) < 3:
            continue
        shapes.append(
            CvatShape(
                image_name=image_name,
                image_width=width,
                image_height=height,
                label=shape_el.attrib.get("label", "").strip(),
                shape_type=shape_type,
                points=tuple(points),
                occluded=shape_el.attrib.get("occluded") == "1",
                z_order=_optional_int(shape_el.attrib.get("z_order")),
                attributes=_parse_attributes(shape_el),
            )
        )
    return shapes


def _tile_record_from_row(row: dict[str, object]) -> TileGeoRecord:
    transform_value = row.get("tile_transform")
    if isinstance(transform_value, str):
        transform = tuple(float(value) for value in json.loads(transform_value))
    else:
        transform = tuple(float(value) for value in transform_value or ())
    if len(transform) != 6:
        raise ValueError(f"Tile transform must have 6 values for tile: {row.get('tile_name')}")
    epsg_value = row.get("epsg")
    epsg = int(epsg_value) if epsg_value not in (None, "") else None
    return TileGeoRecord(
        tile_name=str(row.get("tile_name", "")),
        image_path=str(row.get("image_path", "")),
        row_off=int(float(row.get("row_off", 0))),
        col_off=int(float(row.get("col_off", 0))),
        width=int(float(row.get("width", 0))),
        height=int(float(row.get("height", 0))),
        tile_transform=transform,  # type: ignore[arg-type]
        crs=str(row.get("crs")) if row.get("crs") not in (None, "") else None,
        epsg=epsg,
    )


def _resolve_class_names(annotation_set: CvatAnnotationSet, explicit_class_names: tuple[str, ...]) -> tuple[str, ...]:
    if explicit_class_names:
        return tuple(_unique_names(explicit_class_names))
    names = list(annotation_set.labels)
    for image in annotation_set.images:
        for shape in image.shapes:
            if shape.label and shape.label not in names:
                names.append(shape.label)
    if not names:
        raise ValueError("No labels found in the CVAT annotations.")
    return tuple(names)


def _build_tile_lookup(records: dict[str, TileGeoRecord]) -> dict[str, TileGeoRecord]:
    lookup: dict[str, TileGeoRecord] = {}
    for record in records.values():
        keys = {
            _norm_key(record.tile_name),
            _norm_key(Path(record.tile_name).name),
            _norm_key(record.image_path),
            _norm_key(Path(record.image_path).name),
        }
        for key in keys:
            if key:
                lookup.setdefault(key, record)
    return lookup


def _match_tile_record(image_name: str, lookup: dict[str, TileGeoRecord]) -> TileGeoRecord | None:
    keys = (
        _norm_key(image_name),
        _norm_key(Path(image_name.replace("\\", "/")).name),
    )
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def _parse_points(points_text: str) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for raw_point in points_text.split(";"):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        raw_x, raw_y = raw_point.split(",", maxsplit=1)
        points.append((float(raw_x), float(raw_y)))
    return tuple(points)


def _box_points(shape_el: ET.Element) -> tuple[tuple[float, float], ...]:
    xtl = float(shape_el.attrib["xtl"])
    ytl = float(shape_el.attrib["ytl"])
    xbr = float(shape_el.attrib["xbr"])
    ybr = float(shape_el.attrib["ybr"])
    return ((xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr))


def _parse_attributes(shape_el: ET.Element) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for attribute_el in shape_el.findall("attribute"):
        name = attribute_el.attrib.get("name", "").strip()
        if name:
            attributes[name] = (attribute_el.text or "").strip()
    return attributes


def _clipped_polygon(
    points: tuple[tuple[float, float], ...],
    width: int,
    height: int,
) -> tuple[tuple[float, float], ...]:
    return tuple((min(max(x, 0.0), float(width)), min(max(y, 0.0), float(height))) for x, y in points)


def _remove_duplicate_closing_point(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    if len(points) > 1 and math.isclose(points[0][0], points[-1][0]) and math.isclose(points[0][1], points[-1][1]):
        return points[:-1]
    return points


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _pixel_to_map(
    transform: tuple[float, float, float, float, float, float],
    point: tuple[float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    col, row = point
    return a * col + b * row + c, d * col + e * row + f


def _yolo_segmentation_line(
    class_id: int,
    points: tuple[tuple[float, float], ...],
    width: int,
    height: int,
) -> str:
    values: list[str] = [str(class_id)]
    for x, y in points:
        values.append(_format_float(x / max(width, 1)))
        values.append(_format_float(y / max(height, 1)))
    return " ".join(values)


def _geojson_feature(
    shape: CvatShape,
    record: TileGeoRecord,
    pixel_polygon: tuple[tuple[float, float], ...],
    map_polygon: tuple[tuple[float, float], ...],
    class_id: int,
) -> dict[str, object]:
    ring = list(map_polygon)
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return {
        "type": "Feature",
        "properties": {
            "image_name": shape.image_name,
            "tile_name": record.tile_name,
            "label": shape.label,
            "class_id": class_id,
            "shape_type": shape.shape_type,
            "occluded": shape.occluded,
            "z_order": shape.z_order,
            "epsg": record.epsg,
            "pixel_points": [[_round_coord(x), _round_coord(y)] for x, y in pixel_polygon],
            "attributes": shape.attributes,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[_round_coord(x), _round_coord(y)] for x, y in ring]],
        },
    }


def _manifest_row(
    shape: CvatShape,
    record: TileGeoRecord,
    class_id: int,
    polygon: tuple[tuple[float, float], ...],
    map_polygon: tuple[tuple[float, float], ...],
    label_path: Path,
) -> dict[str, object]:
    xs = [point[0] for point in map_polygon]
    ys = [point[1] for point in map_polygon]
    return {
        "image_name": shape.image_name,
        "tile_name": record.tile_name,
        "label": shape.label,
        "class_id": class_id,
        "shape_type": shape.shape_type,
        "point_count": len(polygon),
        "label_path": str(label_path),
        "image_path": record.image_path,
        "tile_row_off": record.row_off,
        "tile_col_off": record.col_off,
        "epsg": record.epsg or "",
        "x_min": _round_coord(min(xs)),
        "y_min": _round_coord(min(ys)),
        "x_max": _round_coord(max(xs)),
        "y_max": _round_coord(max(ys)),
    }


def _copy_tile_image(record: TileGeoRecord, images_dir: Path) -> int:
    source = Path(record.image_path)
    if not source.exists():
        return 0
    destination = images_dir / Path(record.tile_name).name
    if source.resolve() == destination.resolve():
        return 0
    shutil.copy2(source, destination)
    return 1


def _write_overlay(
    record: TileGeoRecord,
    shapes: list[CvatShape],
    class_index: dict[str, int],
    overlays_dir: Path,
) -> None:
    source = Path(record.image_path)
    if not source.exists():
        return
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    with Image.open(source) as image:
        base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for shape in shapes:
        if shape.label not in class_index:
            continue
        color = _class_color(class_index[shape.label])
        polygon = _clipped_polygon(shape.points, shape.image_width, shape.image_height)
        draw.polygon(polygon, fill=(*color, 70), outline=(*color, 230))
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    composed.save(overlays_dir / Path(record.tile_name).name)


def _class_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (230, 57, 70),
        (42, 157, 143),
        (69, 123, 157),
        (244, 162, 97),
        (131, 56, 236),
        (255, 183, 3),
    )
    return palette[class_id % len(palette)]


def _write_manifest_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image_name",
        "tile_name",
        "label",
        "class_id",
        "shape_type",
        "point_count",
        "label_path",
        "image_path",
        "tile_row_off",
        "tile_col_off",
        "epsg",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _geojson_crs(records: dict[str, TileGeoRecord]) -> dict[str, object] | None:
    epsgs = {record.epsg for record in records.values() if record.epsg is not None}
    if len(epsgs) == 1:
        return {"type": "name", "properties": {"name": f"EPSG:{next(iter(epsgs))}"}}
    return None


def _unique_names(names: tuple[str, ...]) -> list[str]:
    unique: list[str] = []
    for name in names:
        clean_name = name.strip()
        if clean_name and clean_name not in unique:
            unique.append(clean_name)
    return unique


def _int_attr(element: ET.Element, name: str) -> int:
    return int(float(element.attrib[name]))


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _norm_key(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _round_coord(value: float) -> float:
    return round(float(value), 6)
