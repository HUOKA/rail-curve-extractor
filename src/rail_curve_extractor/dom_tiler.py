from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class DomTileOptions:
    tile_width: int | None = None
    tile_height: int | None = None
    stride_x: int | None = None
    stride_y: int | None = None
    tile_size: int = 3072
    stride: int = 1536
    image_format: str = "png"
    prefix: str = "dom"
    skip_empty: bool = True
    min_valid_ratio: float = 0.01
    blank_threshold: int = 0
    include_alpha: bool = False
    max_tiles: int | None = None


@dataclass(frozen=True, slots=True)
class TileRecord:
    tile_id: int
    tile_name: str
    image_path: str
    source_path: str
    row_off: int
    col_off: int
    width: int
    height: int
    source_width: int
    source_height: int
    tile_transform: list[float]
    source_transform: list[float]
    crs: str | None
    epsg: int | None
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    valid_ratio: float


@dataclass(frozen=True, slots=True)
class DomTilingResult:
    source_path: str
    output_dir: str
    images_dir: str
    csv_path: str
    json_path: str
    tile_count: int
    source_width: int
    source_height: int
    source_count: int
    crs: str | None
    epsg: int | None
    transform: list[float]
    options: dict[str, Any]
    records: list[TileRecord]


@dataclass(frozen=True, slots=True)
class DomPreviewResult:
    source_path: str
    preview_path: str
    source_width: int
    source_height: int
    preview_width: int
    preview_height: int
    scale_x: float
    scale_y: float
    preview_to_source_transform: list[float]
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class DomAlignmentOptions:
    target_axis: str = "vertical"
    padding_pixels: int = 32
    crop_to_valid_data: bool = True
    mask_sample_max_width: int = 1200
    mask_sample_max_height: int = 6000
    resampling: str = "bilinear"
    compress: str = "deflate"
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class DomAlignmentResult:
    source_path: str
    output_path: str
    metadata_path: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_count: int
    crs: str | None
    epsg: int | None
    source_transform: list[float]
    output_transform: list[float]
    point1_pixel: tuple[float, float]
    point2_pixel: tuple[float, float]
    point1_map: tuple[float, float]
    point2_map: tuple[float, float]
    source_axis_angle_degrees: float
    target_axis: str
    pixel_size: float
    padding_pixels: int
    crop_to_valid_data: bool


@dataclass(frozen=True, slots=True)
class AnnotationTileSuggestion:
    tile_width: int
    tile_height: int
    stride_x: int
    stride_y: int
    overlap_ratio: float


@dataclass(frozen=True, slots=True)
class DomAutoCorridorResult:
    source_path: str
    source_width: int
    source_height: int
    sample_width: int
    sample_height: int
    points_pixel: list[tuple[float, float]]
    points_map: list[tuple[float, float]]
    valid_ratio: float
    angle_degrees: float


def discover_dom_file(path: Path) -> Path:
    """Return the preferred DJI Terra DOM GeoTIFF from a file or project directory."""
    path = path.expanduser()
    if path.is_file():
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"DOM input must be a GeoTIFF: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"DOM input does not exist: {path}")

    preferred = [
        path / "lidars" / "terra_dom" / "dom.tif",
        path / "lidars" / "terra_dom" / "dom.tiff",
        path / "terra_dom" / "dom.tif",
        path / "terra_dom" / "dom.tiff",
        path / "dom.tif",
        path / "dom.tiff",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    candidates = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in {".tif", ".tiff"}
        and "dom" in candidate.name.lower()
        and "dsm" not in candidate.name.lower()
        and ".temp" not in {part.lower() for part in candidate.relative_to(path).parts}
    ]
    if not candidates:
        raise FileNotFoundError(f"No DOM GeoTIFF found under: {path}")
    return max(candidates, key=lambda candidate: candidate.stat().st_size)


def _discover_dom_overview_file(input_path: Path, source_path: Path) -> Path | None:
    roots: list[Path] = []
    expanded_input = input_path.expanduser()
    if expanded_input.is_dir():
        roots.append(expanded_input)
    for parent in source_path.parents:
        if parent.name.lower() == "lidars":
            roots.append(parent.parent)

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "lidars" / ".temp" / "Reconstruction2d" / "dom_overview.tif",
                root / "lidars" / ".temp" / "Reconstruction2d" / "dom_overview.tiff",
                root / ".temp" / "Reconstruction2d" / "dom_overview.tif",
                root / ".temp" / "Reconstruction2d" / "dom_overview.tiff",
            ]
        )
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != source_path:
            return candidate
    return None


def create_dom_preview(input_path: Path, max_width: int = 900, max_height: int = 6000) -> DomPreviewResult:
    if max_width <= 0:
        raise ValueError("max_width must be positive.")
    if max_height <= 0:
        raise ValueError("max_height must be positive.")
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.errors import NodataShadowWarning
        from rasterio.transform import Affine
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("DOM preview requires rasterio. Install project dependencies first.") from exc

    source_path = discover_dom_file(input_path).resolve()
    preview_path = _discover_dom_overview_file(input_path, source_path) or source_path
    with rasterio.open(source_path) as source_dataset, rasterio.open(preview_path) as preview_dataset:
        preview_width, preview_height = _preview_shape(
            preview_dataset.width,
            preview_dataset.height,
            max_width,
            max_height,
        )
        band_indexes = _output_band_indexes(preview_dataset.count, include_alpha=False)
        data = preview_dataset.read(
            band_indexes,
            out_shape=(len(band_indexes), preview_height, preview_width),
            resampling=Resampling.bilinear,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NodataShadowWarning)
            mask = preview_dataset.dataset_mask(
                out_shape=(preview_height, preview_width),
                resampling=Resampling.nearest,
            )
        preview_scale_x = preview_dataset.width / preview_width
        preview_scale_y = preview_dataset.height / preview_height
        preview_to_map = preview_dataset.transform * Affine.scale(preview_scale_x, preview_scale_y)
        preview_to_source = (~source_dataset.transform) * preview_to_map

        image_array = _to_uint8(data)
        if image_array.shape[0] == 1:
            rgb = np.repeat(image_array, 3, axis=0)
        else:
            rgb = image_array[:3]
        image = np.moveaxis(rgb, 0, -1).copy()
        image[mask == 0] = 0
        return DomPreviewResult(
            source_path=str(source_path),
            preview_path=str(preview_path.resolve()),
            source_width=int(source_dataset.width),
            source_height=int(source_dataset.height),
            preview_width=int(preview_width),
            preview_height=int(preview_height),
            scale_x=float(math.hypot(preview_to_source.a, preview_to_source.d)),
            scale_y=float(math.hypot(preview_to_source.b, preview_to_source.e)),
            preview_to_source_transform=_affine_to_list(preview_to_source),
            image=image,
        )


def align_dom_to_axis(
    input_path: Path,
    output_path: Path,
    point1_pixel: tuple[float, float],
    point2_pixel: tuple[float, float],
    options: DomAlignmentOptions | None = None,
) -> DomAlignmentResult:
    options = options or DomAlignmentOptions()
    _validate_alignment_options(options)
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("DOM alignment requires rasterio. Install project dependencies first.") from exc

    source_path = discover_dom_file(input_path).resolve()
    output_path = output_path.expanduser().resolve()
    if output_path == source_path:
        raise ValueError("Aligned DOM output path must be different from the source DOM path.")
    if output_path.exists() and not options.overwrite:
        raise FileExistsError(f"Aligned DOM already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_path) as dataset:
        if dataset.crs is None:
            raise ValueError("Source DOM must have a CRS to create a georeferenced aligned GeoTIFF.")
        _validate_alignment_points(dataset.width, dataset.height, point1_pixel, point2_pixel)
        point1_map = _pixel_to_map(dataset.transform, point1_pixel)
        point2_map = _pixel_to_map(dataset.transform, point2_pixel)
        direction = _unit_vector(point2_map[0] - point1_map[0], point2_map[1] - point1_map[1])
        col_unit, row_unit = _alignment_units(direction, options.target_axis)
        pixel_size = _mean_pixel_size(dataset.transform)
        extent = _alignment_extent(
            dataset,
            col_unit,
            row_unit,
            pixel_size,
            options,
            extra_points=(point1_map, point2_map),
        )
        col_min_px, row_min_px, col_max_px, row_max_px = extent
        output_width = max(1, int(col_max_px - col_min_px))
        output_height = max(1, int(row_max_px - row_min_px))
        origin_x = (col_unit[0] * col_min_px + row_unit[0] * row_min_px) * pixel_size
        origin_y = (col_unit[1] * col_min_px + row_unit[1] * row_min_px) * pixel_size
        output_transform = Affine(
            col_unit[0] * pixel_size,
            row_unit[0] * pixel_size,
            origin_x,
            col_unit[1] * pixel_size,
            row_unit[1] * pixel_size,
            origin_y,
        )
        if options.crop_to_valid_data:
            crop_window = _exact_crop_window_from_source_signal(
                dataset,
                output_transform,
                output_width,
                output_height,
                options.padding_pixels,
                Resampling,
                reproject,
                rasterio,
            )
            if crop_window is not None:
                col_start, row_start, col_stop, row_stop = crop_window
                output_transform = output_transform * Affine.translation(col_start, row_start)
                output_width = int(col_stop - col_start)
                output_height = int(row_stop - row_start)

        dst_nodata = dataset.nodata if dataset.nodata is not None else 0
        profile = dataset.profile.copy()
        profile.update(
            driver="GTiff",
            width=output_width,
            height=output_height,
            transform=output_transform,
            crs=dataset.crs,
            nodata=dst_nodata,
            compress=options.compress,
            BIGTIFF="IF_SAFER",
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        if output_width >= 256 and output_height >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        else:
            profile.update(tiled=False)

        resampling = _resampling_from_name(options.resampling, Resampling)
        with rasterio.open(output_path, "w", **profile) as destination:
            for band_index in range(1, dataset.count + 1):
                band_resampling = Resampling.nearest if dataset.count >= 4 and band_index == dataset.count else resampling
                reproject(
                    source=rasterio.band(dataset, band_index),
                    destination=rasterio.band(destination, band_index),
                    src_transform=dataset.transform,
                    src_crs=dataset.crs,
                    src_nodata=dataset.nodata,
                    dst_transform=output_transform,
                    dst_crs=dataset.crs,
                    dst_nodata=dst_nodata,
                    resampling=band_resampling,
                )

        result = DomAlignmentResult(
            source_path=str(source_path),
            output_path=str(output_path),
            metadata_path=str(output_path.with_suffix(".alignment.json")),
            source_width=int(dataset.width),
            source_height=int(dataset.height),
            output_width=int(output_width),
            output_height=int(output_height),
            source_count=int(dataset.count),
            crs=dataset.crs.to_string() if dataset.crs else None,
            epsg=dataset.crs.to_epsg() if dataset.crs else None,
            source_transform=_affine_to_list(dataset.transform),
            output_transform=_affine_to_list(output_transform),
            point1_pixel=(float(point1_pixel[0]), float(point1_pixel[1])),
            point2_pixel=(float(point2_pixel[0]), float(point2_pixel[1])),
            point1_map=(float(point1_map[0]), float(point1_map[1])),
            point2_map=(float(point2_map[0]), float(point2_map[1])),
            source_axis_angle_degrees=float(math.degrees(math.atan2(direction[1], direction[0]))),
            target_axis=options.target_axis,
            pixel_size=float(pixel_size),
            padding_pixels=int(options.padding_pixels),
            crop_to_valid_data=bool(options.crop_to_valid_data),
        )

    _write_alignment_metadata(Path(result.metadata_path), result)
    return result


def align_dom_to_axis_from_map_points(
    input_path: Path,
    output_path: Path,
    point1_map: tuple[float, float],
    point2_map: tuple[float, float],
    options: DomAlignmentOptions | None = None,
) -> DomAlignmentResult:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("DOM alignment requires rasterio. Install project dependencies first.") from exc

    source_path = discover_dom_file(input_path).resolve()
    with rasterio.open(source_path) as dataset:
        point1_pixel = (~dataset.transform) * point1_map
        point2_pixel = (~dataset.transform) * point2_map
    return align_dom_to_axis(source_path, output_path, point1_pixel, point2_pixel, options)


def align_dom_to_corridor_from_map_points(
    input_path: Path,
    output_path: Path,
    corridor_points_map: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    options: DomAlignmentOptions | None = None,
) -> DomAlignmentResult:
    options = options or DomAlignmentOptions(crop_to_valid_data=False)
    _validate_alignment_options(options)
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import Affine
        from rasterio.warp import reproject
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("DOM corridor alignment requires rasterio. Install project dependencies first.") from exc

    source_path = discover_dom_file(input_path).resolve()
    output_path = output_path.expanduser().resolve()
    if output_path == source_path:
        raise ValueError("Aligned DOM output path must be different from the source DOM path.")
    if output_path.exists() and not options.overwrite:
        raise FileExistsError(f"Aligned DOM already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    points_map = _validate_corridor_points(corridor_points_map)
    with rasterio.open(source_path) as dataset:
        if dataset.crs is None:
            raise ValueError("Source DOM must have a CRS to create a georeferenced aligned GeoTIFF.")

        pixel_size = _mean_pixel_size(dataset.transform)
        col_unit, row_unit, point1_map, point2_map, bounds_px = _corridor_alignment_geometry(
            points_map,
            pixel_size,
            options.target_axis,
            options.padding_pixels,
        )
        col_min_px, row_min_px, col_max_px, row_max_px = bounds_px
        output_width = max(1, int(col_max_px - col_min_px))
        output_height = max(1, int(row_max_px - row_min_px))
        origin_x = (col_unit[0] * col_min_px + row_unit[0] * row_min_px) * pixel_size
        origin_y = (col_unit[1] * col_min_px + row_unit[1] * row_min_px) * pixel_size
        output_transform = Affine(
            col_unit[0] * pixel_size,
            row_unit[0] * pixel_size,
            origin_x,
            col_unit[1] * pixel_size,
            row_unit[1] * pixel_size,
            origin_y,
        )

        dst_nodata = dataset.nodata if dataset.nodata is not None else 0
        profile = dataset.profile.copy()
        profile.update(
            driver="GTiff",
            width=output_width,
            height=output_height,
            transform=output_transform,
            crs=dataset.crs,
            nodata=dst_nodata,
            compress=options.compress,
            BIGTIFF="IF_SAFER",
        )
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        if output_width >= 256 and output_height >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        else:
            profile.update(tiled=False)

        resampling = _resampling_from_name(options.resampling, Resampling)
        with rasterio.open(output_path, "w", **profile) as destination:
            for band_index in range(1, dataset.count + 1):
                band_resampling = Resampling.nearest if dataset.count >= 4 and band_index == dataset.count else resampling
                reproject(
                    source=rasterio.band(dataset, band_index),
                    destination=rasterio.band(destination, band_index),
                    src_transform=dataset.transform,
                    src_crs=dataset.crs,
                    src_nodata=dataset.nodata,
                    dst_transform=output_transform,
                    dst_crs=dataset.crs,
                    dst_nodata=dst_nodata,
                    resampling=band_resampling,
                )

        point1_pixel = (~dataset.transform) * point1_map
        point2_pixel = (~dataset.transform) * point2_map
        direction = _unit_vector(point2_map[0] - point1_map[0], point2_map[1] - point1_map[1])
        result = DomAlignmentResult(
            source_path=str(source_path),
            output_path=str(output_path),
            metadata_path=str(output_path.with_suffix(".alignment.json")),
            source_width=int(dataset.width),
            source_height=int(dataset.height),
            output_width=int(output_width),
            output_height=int(output_height),
            source_count=int(dataset.count),
            crs=dataset.crs.to_string() if dataset.crs else None,
            epsg=dataset.crs.to_epsg() if dataset.crs else None,
            source_transform=_affine_to_list(dataset.transform),
            output_transform=_affine_to_list(output_transform),
            point1_pixel=(float(point1_pixel[0]), float(point1_pixel[1])),
            point2_pixel=(float(point2_pixel[0]), float(point2_pixel[1])),
            point1_map=(float(point1_map[0]), float(point1_map[1])),
            point2_map=(float(point2_map[0]), float(point2_map[1])),
            source_axis_angle_degrees=float(math.degrees(math.atan2(direction[1], direction[0]))),
            target_axis=options.target_axis,
            pixel_size=float(pixel_size),
            padding_pixels=int(options.padding_pixels),
            crop_to_valid_data=False,
        )

    _write_alignment_metadata(Path(result.metadata_path), result)
    return result


def auto_detect_dom_corridor_points(
    input_path: Path,
    *,
    blank_threshold: int = 5,
    max_width: int = 1200,
    max_height: int = 6000,
    quantile_margin: float = 0.001,
    morphology_iterations: int = 2,
) -> DomAutoCorridorResult:
    if blank_threshold < 0:
        raise ValueError("blank_threshold must be greater than or equal to 0.")
    if max_width <= 0:
        raise ValueError("max_width must be positive.")
    if max_height <= 0:
        raise ValueError("max_height must be positive.")
    if not 0.0 <= quantile_margin < 0.1:
        raise ValueError("quantile_margin must be between 0 and 0.1.")
    if morphology_iterations < 0:
        raise ValueError("morphology_iterations must be greater than or equal to 0.")
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.errors import NodataShadowWarning
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("DOM auto corridor detection requires rasterio. Install project dependencies first.") from exc
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - project dependency missing
        raise RuntimeError("DOM auto corridor detection requires scipy. Install project dependencies first.") from exc

    source_path = discover_dom_file(input_path).resolve()
    with rasterio.open(source_path) as dataset:
        if dataset.crs is None:
            raise ValueError("Source DOM must have a CRS to auto-detect map-coordinate corridor points.")
        sample_width, sample_height = _preview_shape(dataset.width, dataset.height, max_width, max_height)
        band_indexes = _output_band_indexes(dataset.count, include_alpha=False)
        data = dataset.read(
            band_indexes,
            out_shape=(len(band_indexes), sample_height, sample_width),
            resampling=Resampling.nearest,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NodataShadowWarning)
            dataset_mask = dataset.dataset_mask(
                out_shape=(sample_height, sample_width),
                resampling=Resampling.nearest,
            )

        signal_mask = _rgb_signal_mask(data, blank_threshold)
        valid_mask = signal_mask & (dataset_mask > 0)
        if not np.any(valid_mask):
            valid_mask = signal_mask
        if not np.any(valid_mask):
            raise ValueError("Could not detect non-black DOM pixels. Try lowering the blank threshold.")

        cleaned_mask = _largest_connected_mask(valid_mask, ndimage, morphology_iterations)
        rows, cols = np.nonzero(cleaned_mask)
        if rows.size < 8:
            raise ValueError("Detected valid DOM area is too small to estimate a corridor.")

        scale_x = dataset.width / sample_width
        scale_y = dataset.height / sample_height
        source_cols = (cols.astype(np.float64) + 0.5) * scale_x
        source_rows = (rows.astype(np.float64) + 0.5) * scale_y
        points = np.column_stack([source_cols, source_rows])
        corner_pixels, angle_degrees = _oriented_pixel_box(
            points,
            dataset.width,
            dataset.height,
            quantile_margin,
        )
        corner_maps = [_pixel_to_map(dataset.transform, point) for point in corner_pixels]

        return DomAutoCorridorResult(
            source_path=str(source_path),
            source_width=int(dataset.width),
            source_height=int(dataset.height),
            sample_width=int(sample_width),
            sample_height=int(sample_height),
            points_pixel=[(float(col), float(row)) for col, row in corner_pixels],
            points_map=[(float(x), float(y)) for x, y in corner_maps],
            valid_ratio=float(np.count_nonzero(cleaned_mask) / cleaned_mask.size),
            angle_degrees=float(angle_degrees),
        )


def suggest_annotation_tile_size(
    image_width: int,
    image_height: int,
    preferred_height: int = 3072,
    overlap_ratio: float = 0.5,
    divisor: int = 32,
) -> AnnotationTileSuggestion:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")
    if preferred_height <= 0:
        raise ValueError("preferred_height must be positive.")
    if divisor <= 0:
        raise ValueError("divisor must be positive.")
    tile_width = _floor_to_multiple(min(image_width, 3072), divisor, minimum=min(128, image_width))
    tile_height = _floor_to_multiple(min(image_height, preferred_height), divisor, minimum=min(128, image_height))
    return AnnotationTileSuggestion(
        tile_width=tile_width,
        tile_height=tile_height,
        stride_x=tile_width,
        stride_y=stride_from_overlap(tile_height, overlap_ratio),
        overlap_ratio=float(overlap_ratio),
    )


def tile_dom(input_path: Path, output_dir: Path, options: DomTileOptions | None = None) -> DomTilingResult:
    options = options or DomTileOptions()
    _validate_options(options)
    tile_width, tile_height, stride_x, stride_y = _resolved_window_options(options)

    try:
        import rasterio
        from rasterio.errors import NodataShadowWarning
        from rasterio.windows import Window
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError("DOM tiling requires rasterio. Install project dependencies first.") from exc

    source_path = discover_dom_file(input_path).resolve()
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[TileRecord] = []
    image_format = options.image_format.lower().lstrip(".")
    suffix = "jpg" if image_format == "jpeg" else image_format

    with rasterio.open(source_path) as dataset:
        source_transform = _affine_to_list(dataset.transform)
        crs = dataset.crs.to_string() if dataset.crs else None
        epsg = dataset.crs.to_epsg() if dataset.crs else None
        band_indexes = _output_band_indexes(dataset.count, include_alpha=options.include_alpha)
        tile_id = 0

        for row_off, col_off in _iter_offsets(dataset.height, dataset.width, tile_height, tile_width, stride_y, stride_x):
            height = min(tile_height, dataset.height - row_off)
            width = min(tile_width, dataset.width - col_off)
            window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
            data = dataset.read(band_indexes, window=window)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NodataShadowWarning)
                dataset_mask = dataset.dataset_mask(window=window)
            valid_ratio = _valid_ratio(data, dataset_mask, options.blank_threshold)
            if options.skip_empty and valid_ratio < options.min_valid_ratio:
                continue

            tile_name = f"{options.prefix}_r{row_off:06d}_c{col_off:06d}.{suffix}"
            image_path = images_dir / tile_name
            _save_tile_image(data, image_path, image_format)

            x_min, y_min, x_max, y_max = _window_bounds(window, dataset.transform)
            tile_record = TileRecord(
                tile_id=tile_id,
                tile_name=tile_name,
                image_path=str(image_path),
                source_path=str(source_path),
                row_off=int(row_off),
                col_off=int(col_off),
                width=int(width),
                height=int(height),
                source_width=int(dataset.width),
                source_height=int(dataset.height),
                tile_transform=_affine_to_list(dataset.window_transform(window)),
                source_transform=source_transform,
                crs=crs,
                epsg=epsg,
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max),
                valid_ratio=float(valid_ratio),
            )
            records.append(tile_record)
            tile_id += 1
            if options.max_tiles is not None and tile_id >= options.max_tiles:
                break

        result = DomTilingResult(
            source_path=str(source_path),
            output_dir=str(output_dir),
            images_dir=str(images_dir),
            csv_path=str(output_dir / "tile_georef.csv"),
            json_path=str(output_dir / "tile_georef.json"),
            tile_count=len(records),
            source_width=int(dataset.width),
            source_height=int(dataset.height),
            source_count=int(dataset.count),
            crs=crs,
            epsg=epsg,
            transform=source_transform,
            options=asdict(options),
            records=records,
        )

    _write_csv(Path(result.csv_path), records)
    _write_json(Path(result.json_path), result)
    return result


def _validate_options(options: DomTileOptions) -> None:
    tile_width, tile_height, stride_x, stride_y = _resolved_window_options(options)
    if tile_width <= 0:
        raise ValueError("tile_width must be positive.")
    if tile_height <= 0:
        raise ValueError("tile_height must be positive.")
    if stride_x <= 0:
        raise ValueError("stride_x must be positive.")
    if stride_y <= 0:
        raise ValueError("stride_y must be positive.")
    if stride_x > tile_width:
        raise ValueError("stride_x must be less than or equal to tile_width.")
    if stride_y > tile_height:
        raise ValueError("stride_y must be less than or equal to tile_height.")
    if not 0.0 <= options.min_valid_ratio <= 1.0:
        raise ValueError("min_valid_ratio must be between 0 and 1.")
    if options.image_format.lower().lstrip(".") not in {"png", "jpg", "jpeg"}:
        raise ValueError("image_format must be png, jpg, or jpeg.")


def options_from_overlap(
    tile_width: int,
    tile_height: int,
    overlap_ratio: float,
    **kwargs: Any,
) -> DomTileOptions:
    return DomTileOptions(
        tile_width=tile_width,
        tile_height=tile_height,
        stride_x=stride_from_overlap(tile_width, overlap_ratio),
        stride_y=stride_from_overlap(tile_height, overlap_ratio),
        **kwargs,
    )


def stride_from_overlap(tile_length: int, overlap_ratio: float) -> int:
    if tile_length <= 0:
        raise ValueError("tile_length must be positive.")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be greater than or equal to 0 and less than 1.")
    return max(1, int(round(tile_length * (1.0 - overlap_ratio))))


def _validate_alignment_options(options: DomAlignmentOptions) -> None:
    if options.target_axis not in {"vertical", "horizontal"}:
        raise ValueError("target_axis must be vertical or horizontal.")
    if options.padding_pixels < 0:
        raise ValueError("padding_pixels must be greater than or equal to 0.")
    if options.mask_sample_max_width <= 0:
        raise ValueError("mask_sample_max_width must be positive.")
    if options.mask_sample_max_height <= 0:
        raise ValueError("mask_sample_max_height must be positive.")
    if options.resampling not in {"nearest", "bilinear", "cubic"}:
        raise ValueError("resampling must be nearest, bilinear, or cubic.")
    if not options.compress:
        raise ValueError("compress must not be empty.")


def _validate_alignment_points(
    width: int,
    height: int,
    point1_pixel: tuple[float, float],
    point2_pixel: tuple[float, float],
) -> None:
    for point in (point1_pixel, point2_pixel):
        if len(point) != 2:
            raise ValueError("Alignment points must be (x, y) pixel coordinates.")
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Alignment points must be finite pixel coordinates.")
        col, row = point
        if col < 0 or col > width or row < 0 or row > height:
            raise ValueError("Alignment points must be inside the source DOM image.")
    distance = math.hypot(point2_pixel[0] - point1_pixel[0], point2_pixel[1] - point1_pixel[1])
    if distance < 16.0:
        raise ValueError("Alignment points are too close together.")


def _validate_corridor_points(
    corridor_points_map: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    if len(corridor_points_map) != 4:
        raise ValueError("Corridor alignment requires four map points: top-left, top-right, bottom-left, bottom-right.")
    points: list[tuple[float, float]] = []
    for point in corridor_points_map:
        if len(point) != 2:
            raise ValueError("Each corridor point must be an (x, y) map coordinate.")
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Corridor points must be finite map coordinates.")
        points.append((float(point[0]), float(point[1])))
    top_left, top_right, bottom_left, bottom_right = points
    top_width = math.dist(top_left, top_right)
    bottom_width = math.dist(bottom_left, bottom_right)
    left_length = math.dist(top_left, bottom_left)
    right_length = math.dist(top_right, bottom_right)
    if min(top_width, bottom_width) <= 0.0:
        raise ValueError("Corridor top and bottom edges must have positive width.")
    if min(left_length, right_length) <= 0.0:
        raise ValueError("Corridor side edges must have positive length.")
    if (left_length + right_length) <= (top_width + bottom_width):
        raise ValueError("Corridor points should describe a long rail corridor, not a short cross-section.")
    return top_left, top_right, bottom_left, bottom_right


def _corridor_alignment_geometry(
    points_map: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
    pixel_size: float,
    target_axis: str,
    padding_pixels: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], tuple[int, int, int, int]]:
    top_left, top_right, bottom_left, bottom_right = points_map
    top_center = ((top_left[0] + top_right[0]) / 2.0, (top_left[1] + top_right[1]) / 2.0)
    bottom_center = ((bottom_left[0] + bottom_right[0]) / 2.0, (bottom_left[1] + bottom_right[1]) / 2.0)
    long_unit = _unit_vector(bottom_center[0] - top_center[0], bottom_center[1] - top_center[1])
    width_vector = (
        ((top_right[0] - top_left[0]) + (bottom_right[0] - bottom_left[0])) / 2.0,
        ((top_right[1] - top_left[1]) + (bottom_right[1] - bottom_left[1])) / 2.0,
    )

    if target_axis == "vertical":
        row_unit = long_unit
        col_unit = _projected_unit(width_vector, row_unit)
    else:
        col_unit = long_unit
        row_unit = _projected_unit(width_vector, col_unit)

    xs = np.array([point[0] for point in points_map], dtype=np.float64)
    ys = np.array([point[1] for point in points_map], dtype=np.float64)
    local_cols = (xs * col_unit[0] + ys * col_unit[1]) / pixel_size
    local_rows = (xs * row_unit[0] + ys * row_unit[1]) / pixel_size
    padding = float(padding_pixels)
    bounds = (
        math.floor(float(np.min(local_cols)) - padding),
        math.floor(float(np.min(local_rows)) - padding),
        math.ceil(float(np.max(local_cols)) + padding),
        math.ceil(float(np.max(local_rows)) + padding),
    )
    return col_unit, row_unit, top_center, bottom_center, bounds


def _projected_unit(vector: tuple[float, float], axis_unit: tuple[float, float]) -> tuple[float, float]:
    dot = vector[0] * axis_unit[0] + vector[1] * axis_unit[1]
    projected = (vector[0] - dot * axis_unit[0], vector[1] - dot * axis_unit[1])
    return _unit_vector(projected[0], projected[1])


def _rgb_signal_mask(data: np.ndarray, blank_threshold: int) -> np.ndarray:
    image_array = _to_uint8(data[: min(3, data.shape[0])])
    if image_array.shape[0] == 1:
        return image_array[0] > blank_threshold
    return np.any(image_array > blank_threshold, axis=0)


def _largest_connected_mask(valid_mask: np.ndarray, ndimage_module: Any, iterations: int) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool)
    if iterations:
        structure = np.ones((3, 3), dtype=bool)
        mask = ndimage_module.binary_closing(mask, structure=structure, iterations=iterations)
        mask = ndimage_module.binary_opening(mask, structure=structure, iterations=1)
    mask = ndimage_module.binary_fill_holes(mask)
    labels, label_count = ndimage_module.label(mask)
    if label_count == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _oriented_pixel_box(
    points: np.ndarray,
    image_width: int,
    image_height: int,
    quantile_margin: float,
) -> tuple[list[tuple[float, float]], float]:
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    long_axis = eigenvectors[:, order[0]]
    short_axis = eigenvectors[:, order[1]]
    if abs(long_axis[1]) >= abs(long_axis[0]):
        if long_axis[1] < 0:
            long_axis = -long_axis
            short_axis = -short_axis
    elif long_axis[0] < 0:
        long_axis = -long_axis
        short_axis = -short_axis

    long_values = centered @ long_axis
    short_values = centered @ short_axis
    if quantile_margin > 0:
        long_min, long_max = np.quantile(long_values, [quantile_margin, 1.0 - quantile_margin])
        short_min, short_max = np.quantile(short_values, [quantile_margin, 1.0 - quantile_margin])
    else:
        long_min, long_max = float(np.min(long_values)), float(np.max(long_values))
        short_min, short_max = float(np.min(short_values)), float(np.max(short_values))

    corners = [
        center + long_min * long_axis + short_min * short_axis,
        center + long_min * long_axis + short_max * short_axis,
        center + long_max * long_axis + short_min * short_axis,
        center + long_max * long_axis + short_max * short_axis,
    ]
    clipped = [
        (
            float(np.clip(corner[0], 0.0, float(image_width))),
            float(np.clip(corner[1], 0.0, float(image_height))),
        )
        for corner in corners
    ]
    top_pair = sorted(clipped, key=lambda point: point[1])[:2]
    bottom_pair = sorted(clipped, key=lambda point: point[1])[2:]
    top_left, top_right = sorted(top_pair, key=lambda point: point[0])
    bottom_left, bottom_right = sorted(bottom_pair, key=lambda point: point[0])
    angle_degrees = math.degrees(math.atan2(float(long_axis[1]), float(long_axis[0])))
    return [top_left, top_right, bottom_left, bottom_right], angle_degrees


def _preview_shape(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / width, max_height / height, 1.0)
    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    return preview_width, preview_height


def _pixel_to_map(transform: Any, point_pixel: tuple[float, float]) -> tuple[float, float]:
    x = transform.a * point_pixel[0] + transform.b * point_pixel[1] + transform.c
    y = transform.d * point_pixel[0] + transform.e * point_pixel[1] + transform.f
    return float(x), float(y)


def _unit_vector(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError("Alignment points must not map to the same coordinate.")
    return dx / length, dy / length


def _alignment_units(direction: tuple[float, float], target_axis: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if target_axis == "vertical":
        row_unit = direction
        col_unit = (-direction[1], direction[0])
    else:
        col_unit = direction
        row_unit = (direction[1], -direction[0])
    return col_unit, row_unit


def _mean_pixel_size(transform: Any) -> float:
    col_size = math.hypot(float(transform.a), float(transform.d))
    row_size = math.hypot(float(transform.b), float(transform.e))
    pixel_size = (col_size + row_size) / 2.0
    if pixel_size <= 0:
        raise ValueError("Source DOM has an invalid pixel size.")
    return pixel_size


def _alignment_extent(
    dataset: Any,
    col_unit: tuple[float, float],
    row_unit: tuple[float, float],
    pixel_size: float,
    options: DomAlignmentOptions,
    extra_points: Iterable[tuple[float, float]] = (),
) -> tuple[int, int, int, int]:
    xs, ys, sample_margin_pixels = _sample_extent_points(dataset, options)
    extra_points = tuple(extra_points)
    if extra_points:
        extra_xs = np.array([point[0] for point in extra_points], dtype=np.float64)
        extra_ys = np.array([point[1] for point in extra_points], dtype=np.float64)
        xs = np.concatenate([xs, extra_xs])
        ys = np.concatenate([ys, extra_ys])
    local_cols = (xs * col_unit[0] + ys * col_unit[1]) / pixel_size
    local_rows = (xs * row_unit[0] + ys * row_unit[1]) / pixel_size
    margin = sample_margin_pixels + float(options.padding_pixels)
    col_min = math.floor(float(np.min(local_cols)) - margin)
    col_max = math.ceil(float(np.max(local_cols)) + margin)
    row_min = math.floor(float(np.min(local_rows)) - margin)
    row_max = math.ceil(float(np.max(local_rows)) + margin)
    if col_max <= col_min or row_max <= row_min:
        raise ValueError("Could not compute a valid aligned DOM extent.")
    return col_min, row_min, col_max, row_max


def _sample_extent_points(dataset: Any, options: DomAlignmentOptions) -> tuple[np.ndarray, np.ndarray, float]:
    if not options.crop_to_valid_data:
        cols = np.array([0, dataset.width, dataset.width, 0], dtype=np.float64)
        rows = np.array([0, 0, dataset.height, dataset.height], dtype=np.float64)
        return _pixels_to_map_arrays(dataset.transform, cols, rows) + (0.0,)

    try:
        from rasterio.enums import Resampling
        from rasterio.errors import NodataShadowWarning

        preview_width, preview_height = _preview_shape(
            dataset.width,
            dataset.height,
            options.mask_sample_max_width,
            options.mask_sample_max_height,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NodataShadowWarning)
            mask = dataset.dataset_mask(
                out_shape=(preview_height, preview_width),
                resampling=Resampling.nearest,
            )
        valid_rows, valid_cols = np.nonzero(mask > 0)
    except Exception:
        valid_rows = np.array([], dtype=np.int64)
        valid_cols = np.array([], dtype=np.int64)
        preview_width = dataset.width
        preview_height = dataset.height

    if valid_cols.size == 0:
        cols = np.array([0, dataset.width, dataset.width, 0], dtype=np.float64)
        rows = np.array([0, 0, dataset.height, dataset.height], dtype=np.float64)
        return _pixels_to_map_arrays(dataset.transform, cols, rows) + (0.0,)

    scale_x = dataset.width / preview_width
    scale_y = dataset.height / preview_height
    cols = (valid_cols.astype(np.float64) + 0.5) * scale_x
    rows = (valid_rows.astype(np.float64) + 0.5) * scale_y
    sample_margin_pixels = max(scale_x, scale_y)
    return _pixels_to_map_arrays(dataset.transform, cols, rows) + (float(sample_margin_pixels),)


def _pixels_to_map_arrays(transform: Any, cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = transform.a * cols + transform.b * rows + transform.c
    ys = transform.d * cols + transform.e * rows + transform.f
    return xs.astype(np.float64), ys.astype(np.float64)


def _resampling_from_name(name: str, resampling_enum: Any) -> Any:
    return {
        "nearest": resampling_enum.nearest,
        "bilinear": resampling_enum.bilinear,
        "cubic": resampling_enum.cubic,
    }[name]


def _floor_to_multiple(value: int, divisor: int, minimum: int) -> int:
    if value <= minimum:
        return max(1, int(value))
    floored = (int(value) // divisor) * divisor
    return max(minimum, floored)


def _exact_crop_window_from_source_signal(
    dataset: Any,
    output_transform: Any,
    output_width: int,
    output_height: int,
    padding_pixels: int,
    resampling_enum: Any,
    reproject_fn: Any,
    rasterio_module: Any,
) -> tuple[int, int, int, int] | None:
    if output_width <= 0 or output_height <= 0:
        return None
    if output_width * output_height > 300_000_000:
        return None

    mask = np.zeros((output_height, output_width), dtype=np.uint8)
    signal_band = dataset.count if dataset.count >= 4 else 1
    reproject_fn(
        source=rasterio_module.band(dataset, signal_band),
        destination=mask,
        src_transform=dataset.transform,
        src_crs=dataset.crs,
        src_nodata=0,
        dst_transform=output_transform,
        dst_crs=dataset.crs,
        dst_nodata=0,
        resampling=resampling_enum.nearest,
    )
    rows, cols = np.nonzero(mask > 0)
    if rows.size == 0:
        return None

    col_start = max(0, int(cols.min()) - padding_pixels)
    row_start = max(0, int(rows.min()) - padding_pixels)
    col_stop = min(output_width, int(cols.max()) + padding_pixels + 1)
    row_stop = min(output_height, int(rows.max()) + padding_pixels + 1)
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return col_start, row_start, col_stop, row_stop


def _resolved_window_options(options: DomTileOptions) -> tuple[int, int, int, int]:
    tile_width = int(options.tile_width if options.tile_width is not None else options.tile_size)
    tile_height = int(options.tile_height if options.tile_height is not None else options.tile_size)
    stride_x = int(options.stride_x if options.stride_x is not None else options.stride)
    stride_y = int(options.stride_y if options.stride_y is not None else options.stride)
    return tile_width, tile_height, stride_x, stride_y


def _iter_offsets(
    height: int,
    width: int,
    tile_height: int,
    tile_width: int,
    stride_y: int,
    stride_x: int,
) -> Iterable[tuple[int, int]]:
    for row_off in _axis_offsets(height, tile_height, stride_y):
        for col_off in _axis_offsets(width, tile_width, stride_x):
            yield row_off, col_off


def _axis_offsets(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    offsets = list(range(0, length - tile_size + 1, stride))
    final_offset = length - tile_size
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    return offsets


def _output_band_indexes(source_count: int, include_alpha: bool) -> list[int]:
    if source_count <= 0:
        raise ValueError("DOM GeoTIFF has no raster bands.")
    if source_count == 1:
        return [1]
    if include_alpha and source_count >= 4:
        return [1, 2, 3, 4]
    return [1, 2, 3] if source_count >= 3 else list(range(1, source_count + 1))


def _save_tile_image(data: np.ndarray, path: Path, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_array = _to_uint8(data)
    if image_array.shape[0] == 1:
        image = Image.fromarray(image_array[0], mode="L")
    else:
        image = Image.fromarray(np.moveaxis(image_array, 0, -1))
    save_format = "JPEG" if image_format.lower() in {"jpg", "jpeg"} else "PNG"
    save_kwargs: dict[str, Any] = {}
    if save_format == "JPEG":
        save_kwargs["quality"] = 95
    image.save(path, format=save_format, **save_kwargs)


def _to_uint8(data: np.ndarray) -> np.ndarray:
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


def _valid_ratio(data: np.ndarray, dataset_mask: np.ndarray, blank_threshold: int) -> float:
    if dataset_mask.size == 0:
        return 0.0
    valid_mask = dataset_mask > 0
    if data.size:
        signal_mask = np.any(data[: min(3, data.shape[0])] > blank_threshold, axis=0)
        valid_mask = valid_mask & signal_mask
    return float(np.count_nonzero(valid_mask) / valid_mask.size)


def _window_bounds(window: Any, transform: Any) -> tuple[float, float, float, float]:
    col_min = float(window.col_off)
    row_min = float(window.row_off)
    col_max = col_min + float(window.width)
    row_max = row_min + float(window.height)
    xs: list[float] = []
    ys: list[float] = []
    for col, row in ((col_min, row_min), (col_max, row_min), (col_max, row_max), (col_min, row_max)):
        x, y = _pixel_to_map(transform, (col, row))
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _affine_to_list(transform: Any) -> list[float]:
    return [
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    ]


def _write_csv(path: Path, records: list[TileRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TileRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["tile_transform"] = json.dumps(row["tile_transform"], ensure_ascii=False)
            row["source_transform"] = json.dumps(row["source_transform"], ensure_ascii=False)
            writer.writerow(row)


def _write_json(path: Path, result: DomTilingResult) -> None:
    payload = asdict(result)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _write_alignment_metadata(path: Path, result: DomAlignmentResult) -> None:
    payload = asdict(result)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
