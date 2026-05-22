from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np

SUPPORTED_POINT_CLOUD_SUFFIXES = {".las", ".laz", ".csv", ".txt", ".xyz", ".npy"}
LAS_SUFFIXES = {".las", ".laz"}
DIRECTORY_DEFAULT_MAX_POINTS = 5_000_000
XYBounds = tuple[float, float, float, float]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class PointCloudData:
    points: np.ndarray
    intensity: np.ndarray | None = None
    classification: np.ndarray | None = None
    rgb: np.ndarray | None = None
    source_format: str = ""
    source_paths: list[Path] = field(default_factory=list)
    source_point_count: int | None = None


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_point_cloud(path: Path) -> np.ndarray:
    return load_point_cloud_data(path).points


def load_point_cloud_data(
    path: Path,
    max_points: int | None = None,
    xy_bounds: XYBounds | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PointCloudData:
    path = path.expanduser()
    if path.is_dir():
        return _load_point_cloud_directory(path, max_points=max_points, xy_bounds=xy_bounds, progress_callback=progress_callback)

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".xyz"}:
        _emit_progress(progress_callback, phase="reading_file", current_file=str(path), file_index=1, file_count=1)
        points = _load_text_points(path)
        source_point_count = len(points)
        points = _sample_points_array(_filter_points_by_xy_bounds(points, xy_bounds), max_points)
        _emit_progress(
            progress_callback,
            phase="file_loaded",
            current_file=str(path),
            file_index=1,
            file_count=1,
            file_loaded_points=source_point_count,
            file_total_points=source_point_count,
            loaded_points=source_point_count,
            total_points=source_point_count,
        )
        return PointCloudData(points=points, source_format=suffix, source_paths=[path], source_point_count=source_point_count)
    if suffix == ".npy":
        _emit_progress(progress_callback, phase="reading_file", current_file=str(path), file_index=1, file_count=1)
        points = np.load(path)
        validated = _validate_points(points, path)
        source_point_count = len(validated)
        validated = _sample_points_array(_filter_points_by_xy_bounds(validated, xy_bounds), max_points)
        _emit_progress(
            progress_callback,
            phase="file_loaded",
            current_file=str(path),
            file_index=1,
            file_count=1,
            file_loaded_points=source_point_count,
            file_total_points=source_point_count,
            loaded_points=source_point_count,
            total_points=source_point_count,
        )
        return PointCloudData(points=validated, source_format=suffix, source_paths=[path], source_point_count=source_point_count)
    if suffix in {".las", ".laz"}:
        return _load_las_points(path, max_points=max_points, xy_bounds=xy_bounds, progress_callback=progress_callback)
    raise ValueError(f"Unsupported point cloud format: {path.suffix}")


def discover_point_cloud_files(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_POINT_CLOUD_SUFFIXES:
            raise ValueError(f"Unsupported point cloud format: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Point cloud input does not exist: {path}")

    priority_dirs = [
        path / "lidars" / "terra_las",
        path / "terra_las",
        path / "lidars" / "terra_ply",
        path / "terra_ply",
    ]
    for candidate_dir in priority_dirs:
        if candidate_dir.exists():
            files = _files_with_suffixes(candidate_dir, LAS_SUFFIXES if "las" in candidate_dir.name.lower() else SUPPORTED_POINT_CLOUD_SUFFIXES)
            if files:
                return files

    las_files = [
        file_path
        for file_path in _files_with_suffixes(path, LAS_SUFFIXES)
        if ".temp" not in {part.lower() for part in file_path.relative_to(path).parts}
    ]
    if las_files:
        return las_files

    files = [
        file_path
        for file_path in _files_with_suffixes(path, SUPPORTED_POINT_CLOUD_SUFFIXES)
        if ".temp" not in {part.lower() for part in file_path.relative_to(path).parts}
    ]
    if files:
        return files

    raise FileNotFoundError(f"No supported point cloud files found in directory: {path}")


def summarize_point_cloud_input(path: Path) -> dict[str, Any]:
    files = discover_point_cloud_files(path)
    total_points = 0
    minimums: list[np.ndarray] = []
    maximums: list[np.ndarray] = []
    has_rgb = False
    has_intensity = False
    source_formats = sorted({file_path.suffix.lower() for file_path in files})

    las_files = [file_path for file_path in files if file_path.suffix.lower() in LAS_SUFFIXES]
    if len(las_files) == len(files):
        try:
            import laspy
        except ImportError as exc:  # pragma: no cover - exercised only without dependency
            raise RuntimeError("Reading LAS/LAZ requires laspy. Install project dependencies first.") from exc
        for file_path in las_files:
            with laspy.open(file_path) as las_file:
                header = las_file.header
                total_points += int(header.point_count)
                minimums.append(np.asarray(header.mins, dtype=float)[:3])
                maximums.append(np.asarray(header.maxs, dtype=float)[:3])
                dims = set(header.point_format.dimension_names)
                has_rgb = has_rgb or {"red", "green", "blue"} <= dims
                has_intensity = has_intensity or "intensity" in dims
    else:
        for file_path in files:
            point_cloud = load_point_cloud_data(file_path)
            total_points += int(point_cloud.source_point_count or len(point_cloud.points))
            minimums.append(point_cloud.points.min(axis=0))
            maximums.append(point_cloud.points.max(axis=0))
            has_rgb = has_rgb or point_cloud.rgb is not None
            has_intensity = has_intensity or point_cloud.intensity is not None

    source_format = "+".join(source_formats) if len(source_formats) > 1 else source_formats[0]
    if path.is_dir():
        source_format = f"directory:{source_format}"

    return {
        "input_path": str(path),
        "source_paths": [str(file_path) for file_path in files],
        "source_file_count": len(files),
        "points": total_points,
        "has_intensity": has_intensity,
        "has_rgb": has_rgb,
        "source_format": source_format,
        "bounds": {
            "minimum": np.vstack(minimums).min(axis=0).astype(float).tolist(),
            "maximum": np.vstack(maximums).max(axis=0).astype(float).tolist(),
        },
    }


def save_xyz(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points, fmt="%.6f", delimiter=" ")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _load_text_points(path: Path) -> np.ndarray:
    points = np.loadtxt(path, delimiter=_guess_delimiter(path), dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return _validate_points(points, path)


def _guess_delimiter(path: Path) -> str | None:
    if path.suffix.lower() == ".csv":
        return ","
    return None


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _load_las_points(
    path: Path,
    max_points: int | None = None,
    xy_bounds: XYBounds | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PointCloudData:
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover - exercised only without dependency
        raise RuntimeError(
            "Reading LAS/LAZ requires laspy. Install project dependencies first."
        ) from exc

    with laspy.open(path) as las_file:
        source_point_count = int(las_file.header.point_count)
        dims = set(las_file.header.point_format.dimension_names)
        _emit_progress(
            progress_callback,
            phase="reading_file",
            current_file=str(path),
            file_index=1,
            file_count=1,
            file_loaded_points=0,
            file_total_points=source_point_count,
            loaded_points=0,
            total_points=source_point_count,
        )
        if xy_bounds is not None:
            points, intensity, classification, rgb = _read_las_bounded_arrays(
                las_file,
                xy_bounds,
                dims,
                max_points,
                progress_callback=progress_callback,
                file_path=path,
                source_point_count=source_point_count,
            )
        else:
            sample_indices = _sample_indices_for_limit(source_point_count, max_points)
            if len(sample_indices) >= source_point_count:
                if progress_callback is None:
                    las = las_file.read()
                    points, intensity, classification, rgb = _las_record_to_arrays(las, dims)
                else:
                    points, intensity, classification, rgb = _read_las_all_arrays(
                        las_file,
                        dims,
                        progress_callback=progress_callback,
                        file_path=path,
                        source_point_count=source_point_count,
                    )
            else:
                points, intensity, classification, rgb = _read_las_sampled_arrays(
                    las_file,
                    sample_indices,
                    dims,
                    progress_callback=progress_callback,
                    file_path=path,
                    source_point_count=source_point_count,
                )
        _emit_progress(
            progress_callback,
            phase="file_loaded",
            current_file=str(path),
            file_index=1,
            file_count=1,
            file_loaded_points=source_point_count,
            file_total_points=source_point_count,
            loaded_points=source_point_count,
            total_points=source_point_count,
        )
    return PointCloudData(
        points=_validate_points(points, path),
        intensity=intensity,
        classification=classification,
        rgb=rgb,
        source_format=path.suffix.lower(),
        source_paths=[path],
        source_point_count=source_point_count,
    )


def _read_las_bounded_arrays(
    las_file: Any,
    xy_bounds: XYBounds,
    dims: set[str],
    max_points: int | None,
    progress_callback: ProgressCallback | None = None,
    file_path: Path | None = None,
    source_point_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    point_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray | None] = []
    classification_parts: list[np.ndarray | None] = []
    rgb_parts: list[np.ndarray | None] = []
    chunk_size = 1_000_000

    chunk_start = 0
    for las_chunk in las_file.chunk_iterator(chunk_size):
        x_values = np.asarray(las_chunk.x)
        y_values = np.asarray(las_chunk.y)
        chunk_count = len(las_chunk)
        chunk_start += chunk_count
        mask = _xy_bounds_mask_from_arrays(x_values, y_values, xy_bounds)
        if bool(mask.any()):
            indices = np.flatnonzero(mask)
            points, intensity, classification, rgb = _las_record_to_arrays(las_chunk, dims, indices)
            point_parts.append(points)
            intensity_parts.append(intensity)
            classification_parts.append(classification)
            rgb_parts.append(rgb)
        _emit_progress(
            progress_callback,
            phase="reading_file",
            current_file=str(file_path) if file_path is not None else None,
            file_loaded_points=chunk_start,
            file_total_points=source_point_count,
            loaded_points=chunk_start,
            total_points=source_point_count,
        )

    points = np.vstack(point_parts) if point_parts else np.empty((0, 3), dtype=float)
    intensity = _concat_optional(intensity_parts)
    classification = _concat_optional(classification_parts)
    rgb = _concat_optional(rgb_parts)
    return _limit_arrays(points, intensity, classification, rgb, max_points)


def _limit_arrays(
    points: np.ndarray,
    intensity: np.ndarray | None,
    classification: np.ndarray | None,
    rgb: np.ndarray | None,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    indices = _sample_indices_for_limit(len(points), max_points)
    if len(indices) >= len(points):
        return points, intensity, classification, rgb
    return (
        points[indices],
        intensity[indices] if intensity is not None and len(intensity) == len(points) else intensity,
        classification[indices] if classification is not None and len(classification) == len(points) else classification,
        rgb[indices] if rgb is not None and len(rgb) == len(points) else rgb,
    )


def _sample_points_array(points: np.ndarray, max_points: int | None) -> np.ndarray:
    indices = _sample_indices_for_limit(len(points), max_points)
    return points[indices] if len(indices) < len(points) else points


def _filter_points_by_xy_bounds(points: np.ndarray, xy_bounds: XYBounds | None) -> np.ndarray:
    if xy_bounds is None or len(points) == 0:
        return points
    mask = _xy_bounds_mask_from_arrays(points[:, 0], points[:, 1], xy_bounds)
    return points[mask]


def _xy_bounds_mask_from_arrays(x_values: np.ndarray, y_values: np.ndarray, xy_bounds: XYBounds) -> np.ndarray:
    x_min, x_max, y_min, y_max = xy_bounds
    return (x_values >= x_min) & (x_values <= x_max) & (y_values >= y_min) & (y_values <= y_max)


def _file_intersects_xy_bounds(path: Path, xy_bounds: XYBounds | None) -> bool:
    if xy_bounds is None:
        return True
    if path.suffix.lower() not in LAS_SUFFIXES:
        return True
    try:
        import laspy
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Reading LAS/LAZ requires laspy. Install project dependencies first.") from exc
    with laspy.open(path) as las_file:
        mins = np.asarray(las_file.header.mins, dtype=float)
        maxs = np.asarray(las_file.header.maxs, dtype=float)
    x_min, x_max, y_min, y_max = xy_bounds
    return not (maxs[0] < x_min or mins[0] > x_max or maxs[1] < y_min or mins[1] > y_max)


def _empty_point_cloud_directory(path: Path, files: list[Path], total_points: int) -> PointCloudData:
    source_formats = sorted({file_path.suffix.lower() for file_path in files})
    return PointCloudData(
        points=np.empty((0, 3), dtype=float),
        source_format=f"directory:{'+'.join(source_formats)}" if source_formats else "directory",
        source_paths=[],
        source_point_count=total_points,
    )


def _read_las_all_arrays(
    las_file: Any,
    dims: set[str],
    progress_callback: ProgressCallback | None = None,
    file_path: Path | None = None,
    source_point_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    point_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray | None] = []
    classification_parts: list[np.ndarray | None] = []
    rgb_parts: list[np.ndarray | None] = []
    loaded_points = 0
    chunk_size = 1_000_000

    for las_chunk in las_file.chunk_iterator(chunk_size):
        loaded_points += len(las_chunk)
        points, intensity, classification, rgb = _las_record_to_arrays(las_chunk, dims)
        point_parts.append(points)
        intensity_parts.append(intensity)
        classification_parts.append(classification)
        rgb_parts.append(rgb)
        _emit_progress(
            progress_callback,
            phase="reading_file",
            current_file=str(file_path) if file_path is not None else None,
            file_loaded_points=loaded_points,
            file_total_points=source_point_count,
            loaded_points=loaded_points,
            total_points=source_point_count,
        )

    points = np.vstack(point_parts) if point_parts else np.empty((0, 3), dtype=float)
    return (
        points,
        _concat_optional(intensity_parts),
        _concat_optional(classification_parts),
        _concat_optional(rgb_parts),
    )


def _read_las_sampled_arrays(
    las_file: Any,
    sample_indices: np.ndarray,
    dims: set[str],
    progress_callback: ProgressCallback | None = None,
    file_path: Path | None = None,
    source_point_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    point_parts: list[np.ndarray] = []
    intensity_parts: list[np.ndarray | None] = []
    classification_parts: list[np.ndarray | None] = []
    rgb_parts: list[np.ndarray | None] = []
    chunk_start = 0
    chunk_size = 1_000_000

    for las_chunk in las_file.chunk_iterator(chunk_size):
        chunk_count = len(las_chunk)
        chunk_end = chunk_start + chunk_count
        first_index = int(np.searchsorted(sample_indices, chunk_start, side="left"))
        last_index = int(np.searchsorted(sample_indices, chunk_end, side="left"))
        if last_index > first_index:
            local_indices = sample_indices[first_index:last_index] - chunk_start
            points, intensity, classification, rgb = _las_record_to_arrays(las_chunk, dims, local_indices)
            point_parts.append(points)
            intensity_parts.append(intensity)
            classification_parts.append(classification)
            rgb_parts.append(rgb)
        chunk_start = chunk_end
        _emit_progress(
            progress_callback,
            phase="reading_file",
            current_file=str(file_path) if file_path is not None else None,
            file_loaded_points=chunk_start,
            file_total_points=source_point_count,
            loaded_points=chunk_start,
            total_points=source_point_count,
        )

    points = np.vstack(point_parts) if point_parts else np.empty((0, 3), dtype=float)
    return (
        points,
        _concat_optional(intensity_parts),
        _concat_optional(classification_parts),
        _concat_optional(rgb_parts),
    )


def _las_record_to_arrays(
    las_record: Any,
    dims: set[str],
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    x_values = _take_las_dimension(las_record.x, indices)
    y_values = _take_las_dimension(las_record.y, indices)
    z_values = _take_las_dimension(las_record.z, indices)
    points = np.column_stack([x_values, y_values, z_values]).astype(float)
    intensity = (
        np.asarray(_take_las_dimension(las_record.intensity, indices), dtype=np.float32)
        if "intensity" in dims
        else None
    )
    classification = (
        np.asarray(_take_las_dimension(las_record.classification, indices), dtype=np.uint8)
        if "classification" in dims
        else None
    )
    rgb = None
    if {"red", "green", "blue"} <= dims:
        rgb = np.column_stack(
            [
                _take_las_dimension(las_record.red, indices),
                _take_las_dimension(las_record.green, indices),
                _take_las_dimension(las_record.blue, indices),
            ]
        ).astype(np.uint16)
    return points, intensity, classification, rgb


def _take_las_dimension(values: Any, indices: np.ndarray | None) -> np.ndarray:
    array = np.asarray(values)
    if indices is None:
        return array
    return array[indices]


def _validate_points(points: np.ndarray, path: Path) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud in {path} must be an Nx3-compatible array.")
    return np.asarray(points[:, :3], dtype=float)


def _load_point_cloud_directory(
    path: Path,
    max_points: int | None = None,
    xy_bounds: XYBounds | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PointCloudData:
    _emit_progress(progress_callback, phase="discovering_files", current_file=str(path))
    files = discover_point_cloud_files(path)
    total_counts = [_point_count_for_file(file_path) for file_path in files]
    total_points = int(sum(total_counts))
    _emit_progress(
        progress_callback,
        phase="files_discovered",
        current_file=str(path),
        file_count=len(files),
        loaded_points=0,
        total_points=total_points,
        source_total_points=total_points,
    )
    effective_max = DIRECTORY_DEFAULT_MAX_POINTS if max_points is None else int(max_points)
    candidate_files = [file_path for file_path in files if _file_intersects_xy_bounds(file_path, xy_bounds)]
    if not candidate_files:
        return _empty_point_cloud_directory(path, files, total_points)

    candidate_counts = [_point_count_for_file(file_path) for file_path in candidate_files]
    candidate_total_points = int(sum(candidate_counts))
    per_file_limits = _allocate_directory_sample_limits(candidate_counts, effective_max)

    loaded_parts: list[PointCloudData] = []
    completed_points = 0
    for file_index, (file_path, per_file_limit, candidate_count) in enumerate(
        zip(candidate_files, per_file_limits, candidate_counts),
        start=1,
    ):
        if per_file_limit is not None and per_file_limit <= 0:
            completed_points += candidate_count
            _emit_progress(
                progress_callback,
                phase="file_skipped",
                current_file=str(file_path),
                file_index=file_index,
                file_count=len(candidate_files),
                file_loaded_points=candidate_count,
                file_total_points=candidate_count,
                loaded_points=completed_points,
                total_points=candidate_total_points,
                source_total_points=total_points,
            )
            continue

        points_before_file = completed_points

        def file_progress(event: dict[str, Any]) -> None:
            file_loaded_points = int(event.get("file_loaded_points") or event.get("loaded_points") or 0)
            payload = {
                **event,
                "current_file": str(file_path),
                "file_index": file_index,
                "file_count": len(candidate_files),
                "file_loaded_points": file_loaded_points,
                "file_total_points": candidate_count,
                "loaded_points": min(points_before_file + file_loaded_points, candidate_total_points),
                "total_points": candidate_total_points,
                "source_total_points": total_points,
            }
            _emit_progress(progress_callback, **payload)

        loaded_parts.append(
            load_point_cloud_data(
                file_path,
                max_points=per_file_limit,
                xy_bounds=xy_bounds,
                progress_callback=file_progress,
            )
        )
        completed_points += candidate_count
        _emit_progress(
            progress_callback,
            phase="file_loaded",
            current_file=str(file_path),
            file_index=file_index,
            file_count=len(candidate_files),
            file_loaded_points=candidate_count,
            file_total_points=candidate_count,
            loaded_points=completed_points,
            total_points=candidate_total_points,
            source_total_points=total_points,
        )
    nonempty_points = [part.points for part in loaded_parts if len(part.points) > 0]
    _emit_progress(
        progress_callback,
        phase="combining_arrays",
        file_count=len(candidate_files),
        loaded_points=candidate_total_points,
        total_points=candidate_total_points,
        source_total_points=total_points,
    )
    points = np.vstack(nonempty_points) if nonempty_points else np.empty((0, 3), dtype=float)
    intensity = _concat_optional([part.intensity for part in loaded_parts])
    classification = _concat_optional([part.classification for part in loaded_parts])
    rgb = _concat_optional([part.rgb for part in loaded_parts])
    points, intensity, classification, rgb = _limit_arrays(points, intensity, classification, rgb, effective_max)
    source_formats = sorted({file_path.suffix.lower() for file_path in files})
    return PointCloudData(
        points=points,
        intensity=intensity,
        classification=classification,
        rgb=rgb,
        source_format=f"directory:{'+'.join(source_formats)}",
        source_paths=candidate_files,
        source_point_count=total_points,
    )


def _point_count_for_file(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in LAS_SUFFIXES:
        try:
            import laspy
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Reading LAS/LAZ requires laspy. Install project dependencies first.") from exc
        with laspy.open(path) as las_file:
            return int(las_file.header.point_count)
    if suffix == ".npy":
        points = np.load(path, mmap_mode="r")
        return int(points.shape[0])
    points = np.loadtxt(path, delimiter=_guess_delimiter(path), dtype=float)
    return int(1 if points.ndim == 1 else points.shape[0])


def _allocate_directory_sample_limits(total_counts: list[int], effective_max: int) -> list[int | None]:
    total_points = int(sum(total_counts))
    if effective_max <= 0 or total_points <= effective_max:
        return [None for _ in total_counts]

    limits = [0 for _ in total_counts]
    nonempty_indices = [index for index, count in enumerate(total_counts) if count > 0]
    if not nonempty_indices:
        return limits

    if effective_max < len(nonempty_indices):
        largest_indices = sorted(nonempty_indices, key=lambda index: total_counts[index], reverse=True)[:effective_max]
        for index in largest_indices:
            limits[index] = 1
        return limits

    raw_limits = np.array(
        [effective_max * count / max(total_points, 1) for count in total_counts],
        dtype=float,
    )
    for index in nonempty_indices:
        limits[index] = max(1, int(np.floor(raw_limits[index])))

    remaining = effective_max - sum(limits)
    if remaining > 0:
        remainders = sorted(
            nonempty_indices,
            key=lambda index: raw_limits[index] - np.floor(raw_limits[index]),
            reverse=True,
        )
        for index in remainders[:remaining]:
            limits[index] += 1
    elif remaining < 0:
        reducible = sorted(
            (index for index in nonempty_indices if limits[index] > 1),
            key=lambda index: raw_limits[index] - np.floor(raw_limits[index]),
        )
        for index in reducible[: abs(remaining)]:
            limits[index] -= 1

    return limits


def _sample_indices_for_limit(length: int, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=max_points, dtype=np.int64)


def _concat_optional(parts: list[np.ndarray | None]) -> np.ndarray | None:
    if any(part is None for part in parts):
        return None
    valid_parts = [part for part in parts if part is not None and len(part) > 0]
    if not valid_parts:
        return None
    return np.concatenate(valid_parts)


def _files_with_suffixes(path: Path, suffixes: set[str]) -> list[Path]:
    files = [
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in suffixes
    ]
    return sorted(files, key=_natural_path_key)


def _natural_path_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]
