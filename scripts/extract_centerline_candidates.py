from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract first-pass track centerline candidates by pairing rail masks row by row.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory with images/ and summary.json.")
    parser.add_argument("--mask-dir", required=True, help="Semantic prediction mask directory.")
    parser.add_argument("--out", required=True, help="Output directory for CSV, GeoJSON, overlays, and summary.")
    parser.add_argument("--row-step", type=int, default=16)
    parser.add_argument("--column-threshold", type=float, default=0.08)
    parser.add_argument("--min-run-pixels", type=int, default=2)
    parser.add_argument("--min-pair-gap", type=float, default=20.0)
    parser.add_argument("--max-pair-gap", type=float, default=130.0)
    parser.add_argument("--target-gap", type=float, default=0.0, help="0 means estimate from masks.")
    parser.add_argument("--gap-tolerance", type=float, default=0.50, help="Relative tolerance around target gap.")
    parser.add_argument(
        "--target-gauge-m",
        type=float,
        default=0.0,
        help="Optional map-space rail-pair target gauge in meters; 0 disables map-space gauge filtering.",
    )
    parser.add_argument(
        "--gauge-tolerance-m",
        type=float,
        default=0.15,
        help="Absolute tolerance around --target-gauge-m when map-space gauge filtering is enabled.",
    )
    parser.add_argument("--min-track-points", type=int, default=4, help="Drop grouped centerline tracks shorter than this many sampled rows.")
    parser.add_argument(
        "--max-track-x-jump",
        type=float,
        default=0.0,
        help="Maximum centerline x jump between adjacent sampled rows; 0 derives it from target gap.",
    )
    parser.add_argument("--max-track-row-gap", type=int, default=2, help="Maximum missing sampled rows allowed while grouping tracks.")
    parser.add_argument(
        "--ignore-labels",
        default="",
        help="Comma-separated dataset label names or class ids to remove from masks before centerline extraction; empty disables.",
    )
    parser.add_argument("--contact-sheet-max", type=int, default=16, help="Maximum overlay images to include in contact_sheet.jpg; 0 disables.")
    parser.add_argument("--max-images", type=int, default=0, help="Debug limit; 0 means all images.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    mask_dir = Path(args.mask_dir).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dataset_dir / "images"
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    tile_lookup = load_tile_lookup(dataset_dir)
    ignore_ids = resolve_ignore_ids(dataset_dir, args.ignore_labels)

    all_rail_rows: list[dict[str, Any]] = []
    per_image_rails: dict[str, list[dict[str, float]]] = {}
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        mask = load_mask(mask_path)
        if ignore_ids:
            mask = apply_yolo_ignore_labels(mask, dataset_dir / "labels" / f"{image_path.stem}.txt", ignore_ids)
        rail_rows = extract_rail_centers(
            mask=mask,
            row_step=args.row_step,
            column_threshold=args.column_threshold,
            min_run_pixels=args.min_run_pixels,
        )
        per_image_rails[image_path.name] = rail_rows
        for row in rail_rows:
            all_rail_rows.append({"image_name": image_path.name, **row})

    target_gap = args.target_gap if args.target_gap > 0 else estimate_target_gap(
        per_image_rails,
        args.min_pair_gap,
        args.max_pair_gap,
    )
    pair_min = max(args.min_pair_gap, target_gap * (1.0 - args.gap_tolerance))
    pair_max = min(args.max_pair_gap, target_gap * (1.0 + args.gap_tolerance))

    candidate_rows: list[dict[str, Any]] = []
    geo_features: list[dict[str, Any]] = []
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    raw_candidate_count = 0
    grouped_track_count = 0
    max_track_x_jump = args.max_track_x_jump if args.max_track_x_jump > 0 else max(24.0, target_gap * 0.8)
    overlay_paths: list[Path] = []
    for image_path in image_paths:
        rail_rows = per_image_rails.get(image_path.name, [])
        record = match_tile_record(tile_lookup, image_path.name)
        raw_candidates = extract_track_candidates(
            rail_rows,
            pair_min,
            pair_max,
            target_gap,
            map_transform=record["tile_transform"] if record is not None else None,
            target_gauge_m=args.target_gauge_m,
            gauge_tolerance_m=args.gauge_tolerance_m,
        )
        raw_candidate_count += len(raw_candidates)
        candidates = group_candidate_tracks(
            raw_candidates,
            row_step=args.row_step,
            max_x_jump=max_track_x_jump,
            max_row_gap=args.max_track_row_gap,
            min_track_points=args.min_track_points,
        )
        grouped_track_count += len({int(row["candidate_id"]) for row in candidates})
        record = match_tile_record(tile_lookup, image_path.name)
        for row in candidates:
            pixel_x = row["x"]
            pixel_y = row["y"]
            map_xy = pixel_to_map(record["tile_transform"], pixel_x, pixel_y) if record is not None else (None, None)
            payload = {
                "image_name": image_path.name,
                "candidate_id": row["candidate_id"],
                "pixel_x": round(pixel_x, 3),
                "pixel_y": round(pixel_y, 3),
                "left_rail_x": round(row["left_x"], 3),
                "right_rail_x": round(row["right_x"], 3),
                "gap_px": round(row["gap_px"], 3),
                "gap_m": round(row["gap_m"], 4) if row.get("gap_m") is not None else "",
                "target_gap_px": round(target_gap, 3),
                "confidence": round(row["confidence"], 4),
                "map_x": round(map_xy[0], 6) if map_xy[0] is not None else "",
                "map_y": round(map_xy[1], 6) if map_xy[1] is not None else "",
            }
            candidate_rows.append(payload)
        if candidates:
            geo_features.extend(features_for_image(image_path.name, candidates, record))
        overlay_path = overlay_dir / f"{image_path.stem}.png"
        write_overlay(image_path, rail_rows, candidates, overlay_path)
        overlay_paths.append(overlay_path)

    write_csv(output_dir / "rail_centers.csv", all_rail_rows)
    write_csv(output_dir / "track_centerline_candidates.csv", candidate_rows)
    write_geojson(output_dir / "track_centerline_candidates.geojson", geo_features)
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    if args.contact_sheet_max > 0:
        make_contact_sheet(overlay_paths, contact_sheet_path, max_items=args.contact_sheet_max)
    summary = {
        "dataset_dir": str(dataset_dir),
        "mask_dir": str(mask_dir),
        "image_count": len(image_paths),
        "images_with_masks": len(per_image_rails),
        "rail_center_count": len(all_rail_rows),
        "raw_centerline_candidate_count": raw_candidate_count,
        "centerline_candidate_count": len(candidate_rows),
        "grouped_track_count": grouped_track_count,
        "target_gap_px": target_gap,
        "pair_min_gap_px": pair_min,
        "pair_max_gap_px": pair_max,
        "max_track_x_jump": max_track_x_jump,
        "max_track_row_gap": args.max_track_row_gap,
        "min_track_points": args.min_track_points,
        "ignore_labels": args.ignore_labels,
        "ignore_ids": sorted(ignore_ids),
        "target_gauge_m": args.target_gauge_m,
        "gauge_tolerance_m": args.gauge_tolerance_m,
        "map_gauge_filter_enabled": args.target_gauge_m > 0,
        "contact_sheet_path": str(contact_sheet_path) if args.contact_sheet_max > 0 else "",
        "row_step": args.row_step,
        "column_threshold": args.column_threshold,
        "min_run_pixels": args.min_run_pixels,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 0


def resolve_ignore_ids(dataset_dir: Path, ignore_labels: str) -> set[int]:
    tokens = [token.strip() for token in ignore_labels.split(",") if token.strip()]
    if not tokens:
        return set()
    class_names_path = dataset_dir / "classes.txt"
    class_names = class_names_path.read_text(encoding="utf-8").splitlines() if class_names_path.exists() else []
    ids: set[int] = set()
    unknown: list[str] = []
    for token in tokens:
        if token.isdigit():
            ids.add(int(token))
        elif token in class_names:
            ids.add(class_names.index(token))
        else:
            unknown.append(token)
    if unknown:
        raise ValueError(f"Unknown ignore label(s): {', '.join(unknown)}")
    return ids


def apply_yolo_ignore_labels(mask: np.ndarray, label_path: Path, ignore_ids: set[int]) -> np.ndarray:
    if not label_path.exists():
        return mask
    height, width = mask.shape
    ignore_image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(ignore_image)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if class_id not in ignore_ids:
            continue
        points = [
            (values[index] * width, values[index + 1] * height)
            for index in range(0, len(values) - 1, 2)
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    ignore_mask = np.asarray(ignore_image, dtype=bool)
    return np.logical_and(mask, np.logical_not(ignore_mask))


def extract_rail_centers(mask: np.ndarray, row_step: int, column_threshold: float, min_run_pixels: int) -> list[dict[str, float]]:
    if row_step <= 0:
        raise ValueError("row_step must be positive.")
    rows: list[dict[str, float]] = []
    height, _ = mask.shape
    half_window = max(1, row_step // 2)
    for y in range(half_window, height, row_step):
        y0 = max(0, y - half_window)
        y1 = min(height, y + half_window)
        if y1 <= y0:
            continue
        scores = mask[y0:y1, :].mean(axis=0)
        runs = runs_from_scores(scores, column_threshold, min_run_pixels)
        for run_index, run in enumerate(runs):
            rows.append(
                {
                    "y": float(y),
                    "rail_index": float(run_index),
                    "x": float(run["center"]),
                    "run_start": float(run["start"]),
                    "run_end": float(run["end"]),
                    "run_width": float(run["width"]),
                    "score": float(run["score"]),
                }
            )
    return rows


def runs_from_scores(scores: np.ndarray, threshold: float, min_run_pixels: int) -> list[dict[str, float]]:
    active = scores >= threshold
    runs: list[dict[str, float]] = []
    index = 0
    while index < active.size:
        if not active[index]:
            index += 1
            continue
        start = index
        while index < active.size and active[index]:
            index += 1
        end = index
        width = end - start
        if width >= min_run_pixels:
            segment = scores[start:end]
            xs = np.arange(start, end, dtype=np.float32)
            weight = float(segment.sum())
            center = float((xs * segment).sum() / weight) if weight > 1e-6 else float((start + end - 1) / 2.0)
            runs.append({"start": float(start), "end": float(end - 1), "width": float(width), "center": center, "score": float(segment.mean())})
    return runs


def estimate_target_gap(per_image_rails: dict[str, list[dict[str, float]]], min_gap: float, max_gap: float) -> float:
    gaps: list[float] = []
    for rows in per_image_rails.values():
        for _, centers in rows_by_y(rows).items():
            xs = sorted(row["x"] for row in centers)
            for left, right in zip(xs, xs[1:]):
                gap = right - left
                if min_gap <= gap <= max_gap:
                    gaps.append(gap)
    if not gaps:
        return (min_gap + max_gap) / 2.0
    gaps = sorted(gaps)
    percentile_index = min(len(gaps) - 1, max(0, int(len(gaps) * 0.4)))
    return float(gaps[percentile_index])


def extract_track_candidates(
    rail_rows: list[dict[str, float]],
    min_gap: float,
    max_gap: float,
    target_gap: float,
    map_transform: tuple[float, float, float, float, float, float] | None = None,
    target_gauge_m: float = 0.0,
    gauge_tolerance_m: float = 0.15,
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for y, rows in rows_by_y(rail_rows).items():
        sorted_rows = sorted(rows, key=lambda row: row["x"])
        for pair_index, (left, right, confidence, gap_m) in enumerate(
            select_non_overlapping_pairs(
                sorted_rows,
                min_gap,
                max_gap,
                target_gap,
                y=y,
                map_transform=map_transform,
                target_gauge_m=target_gauge_m,
                gauge_tolerance_m=gauge_tolerance_m,
            )
        ):
            gap = float(right["x"] - left["x"])
            candidates.append(
                {
                    "candidate_id": float(pair_index),
                    "row_pair_index": float(pair_index),
                    "y": float(y),
                    "x": float((left["x"] + right["x"]) / 2.0),
                    "left_x": float(left["x"]),
                    "right_x": float(right["x"]),
                    "gap_px": gap,
                    "gap_m": gap_m,
                    "confidence": confidence,
                }
            )
            pair_index += 1
    return candidates


def select_non_overlapping_pairs(
    sorted_rows: list[dict[str, float]],
    min_gap: float,
    max_gap: float,
    target_gap: float,
    y: float | None = None,
    map_transform: tuple[float, float, float, float, float, float] | None = None,
    target_gauge_m: float = 0.0,
    gauge_tolerance_m: float = 0.15,
) -> list[tuple[dict[str, float], dict[str, float], float, float | None]]:
    pair_scores: list[float | None] = []
    pair_gaps_m: list[float | None] = []
    for left, right in zip(sorted_rows, sorted_rows[1:]):
        gap = float(right["x"] - left["x"])
        if gap < min_gap or gap > max_gap:
            pair_scores.append(None)
            pair_gaps_m.append(None)
            continue
        gap_score = max(0.0, 1.0 - abs(gap - target_gap) / max(target_gap, 1e-6))
        gap_m = map_gap_m(map_transform, y, left["x"], right["x"]) if map_transform is not None and y is not None else None
        if target_gauge_m > 0 and gap_m is not None:
            gauge_delta = abs(gap_m - target_gauge_m)
            if gauge_delta > gauge_tolerance_m:
                pair_scores.append(None)
                pair_gaps_m.append(gap_m)
                continue
            gap_score *= max(0.0, 1.0 - gauge_delta / max(gauge_tolerance_m, 1e-6))
        confidence = 0.5 * (float(left["score"]) + float(right["score"])) * gap_score
        pair_scores.append(confidence)
        pair_gaps_m.append(gap_m)

    count = len(sorted_rows)
    best_score = [0.0] * (count + 2)
    take_pair = [False] * count
    for index in range(count - 2, -1, -1):
        skip_score = best_score[index + 1]
        confidence = pair_scores[index]
        pair_score = (confidence if confidence is not None else -1.0) + best_score[index + 2]
        if confidence is not None and pair_score > skip_score:
            best_score[index] = pair_score
            take_pair[index] = True
        else:
            best_score[index] = skip_score

    pairs: list[tuple[dict[str, float], dict[str, float], float, float | None]] = []
    index = 0
    while index < count - 1:
        if take_pair[index] and pair_scores[index] is not None:
            pairs.append((sorted_rows[index], sorted_rows[index + 1], float(pair_scores[index]), pair_gaps_m[index]))
            index += 2
        else:
            index += 1
    return pairs


def map_gap_m(
    transform: tuple[float, float, float, float, float, float] | None,
    y: float | None,
    left_x: float,
    right_x: float,
) -> float | None:
    if transform is None or y is None:
        return None
    left = pixel_to_map(transform, float(left_x), float(y))
    right = pixel_to_map(transform, float(right_x), float(y))
    return float(np.hypot(right[0] - left[0], right[1] - left[1]))


def group_candidate_tracks(
    candidates: list[dict[str, float]],
    row_step: int,
    max_x_jump: float,
    max_row_gap: int,
    min_track_points: int,
) -> list[dict[str, float]]:
    if not candidates:
        return []
    if row_step <= 0:
        raise ValueError("row_step must be positive.")
    if min_track_points <= 0:
        raise ValueError("min_track_points must be positive.")

    rows = rows_by_y(candidates)
    active: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    next_track_id = 0
    max_y_gap = row_step * (max_row_gap + 1.25)

    for y in sorted(rows):
        row_candidates = sorted(rows[y], key=lambda row: (-row["confidence"], row["x"]))
        stale = [track for track in active if y - float(track["last_y"]) > max_y_gap]
        if stale:
            completed.extend(stale)
            active = [track for track in active if y - float(track["last_y"]) <= max_y_gap]

        used_tracks: set[int] = set()
        for candidate in row_candidates:
            best_index = None
            best_cost = float("inf")
            for index, track in enumerate(active):
                if index in used_tracks:
                    continue
                dy = y - float(track["last_y"])
                if dy <= 0 or dy > max_y_gap:
                    continue
                dx = abs(float(candidate["x"]) - float(track["last_x"]))
                if dx > max_x_jump:
                    continue
                gap_delta = abs(float(candidate["gap_px"]) - float(track["mean_gap"]))
                cost = dx / max(max_x_jump, 1e-6) + gap_delta / max(float(track["mean_gap"]), 1e-6) + dy / max_y_gap
                if cost < best_cost:
                    best_cost = cost
                    best_index = index
            if best_index is None:
                active.append(
                    {
                        "track_id": next_track_id,
                        "last_y": y,
                        "last_x": float(candidate["x"]),
                        "mean_gap": float(candidate["gap_px"]),
                        "rows": [candidate],
                    }
                )
                next_track_id += 1
                continue

            track = active[best_index]
            used_tracks.add(best_index)
            track_rows = track["rows"]
            track_rows.append(candidate)
            count = len(track_rows)
            track["last_y"] = y
            track["last_x"] = float(candidate["x"])
            track["mean_gap"] = float(track["mean_gap"]) + (float(candidate["gap_px"]) - float(track["mean_gap"])) / count

    completed.extend(active)
    kept = [track for track in completed if len(track["rows"]) >= min_track_points]
    kept = sorted(kept, key=lambda track: (min(row["x"] for row in track["rows"]), min(row["y"] for row in track["rows"])))

    grouped: list[dict[str, float]] = []
    for new_id, track in enumerate(kept):
        for row in sorted(track["rows"], key=lambda item: item["y"]):
            grouped.append({**row, "candidate_id": float(new_id)})
    return grouped


def rows_by_y(rows: list[dict[str, float]]) -> dict[float, list[dict[str, float]]]:
    grouped: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(float(row["y"]), []).append(row)
    return grouped


def load_tile_lookup(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = dataset_dir / "summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tile_georef_value = summary.get("tile_georef_path") or summary.get("selected_tile_index_csv") or ""
    if not tile_georef_value:
        return {}
    tile_georef_path = Path(tile_georef_value)
    if not tile_georef_path.is_absolute():
        tile_georef_path = dataset_dir / tile_georef_path
    if not tile_georef_path.exists() or not tile_georef_path.is_file():
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    with tile_georef_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            transform = row.get("tile_transform", "")
            if not transform:
                continue
            parsed = json.loads(transform) if transform else None
            payload = {**row, "tile_transform": tuple(float(value) for value in parsed)}
            keys = {
                str(row.get("tile_name", "")).replace("\\", "/").lower(),
                Path(str(row.get("tile_name", ""))).name.lower(),
                Path(str(row.get("image_path", ""))).name.lower(),
                Path(str(row.get("image_name", ""))).name.lower(),
            }
            for key in keys:
                if key:
                    lookup[key] = payload
    return lookup


def match_tile_record(lookup: dict[str, dict[str, Any]], image_name: str) -> dict[str, Any] | None:
    keys = [image_name.replace("\\", "/").lower(), Path(image_name).name.lower()]
    for key in keys:
        if key in lookup:
            return lookup[key]
    return None


def pixel_to_map(transform: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    return a * x + b * y + c, d * x + e * y + f


def features_for_image(image_name: str, candidates: list[dict[str, float]], record: dict[str, Any] | None) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in candidates:
        grouped.setdefault(int(row["candidate_id"]), []).append(row)
    for candidate_id, rows in grouped.items():
        rows = sorted(rows, key=lambda row: row["y"])
        coords = []
        pixel_coords = []
        for row in rows:
            pixel_coords.append([round(row["x"], 3), round(row["y"], 3)])
            if record is not None:
                coords.append([round(value, 6) for value in pixel_to_map(record["tile_transform"], row["x"], row["y"])])
        geometry = {"type": "LineString", "coordinates": coords if coords else pixel_coords}
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "image_name": image_name,
                    "candidate_id": candidate_id,
                    "point_count": len(rows),
                    "coordinate_space": "map" if coords else "pixel",
                    "mean_gap_px": round(float(np.mean([row["gap_px"] for row in rows])), 3),
                    "mean_gap_m": round(float(np.mean([row["gap_m"] for row in rows if row.get("gap_m") is not None])), 4)
                    if any(row.get("gap_m") is not None for row in rows)
                    else None,
                    "mean_confidence": round(float(np.mean([row["confidence"] for row in rows])), 4),
                    "pixel_points": pixel_coords,
                },
                "geometry": geometry,
            }
        )
    return features


def write_overlay(image_path: Path, rail_rows: list[dict[str, float]], candidates: list[dict[str, float]], output_path: Path) -> None:
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    draw = ImageDraw.Draw(base)
    for row in rail_rows:
        x = float(row["x"])
        y = float(row["y"])
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(42, 157, 143))
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in candidates:
        grouped.setdefault(int(row["candidate_id"]), []).append(row)
    palette = [(255, 183, 3), (230, 57, 70), (131, 56, 236), (0, 114, 178), (213, 94, 0)]
    for candidate_id, rows in grouped.items():
        points = [(float(row["x"]), float(row["y"])) for row in sorted(rows, key=lambda row: row["y"])]
        color = palette[candidate_id % len(palette)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 2.0, y - 2.0, x + 2.0, y + 2.0), fill=color)
    base.save(output_path)


def make_contact_sheet(overlay_paths: list[Path], output_path: Path, max_items: int) -> None:
    selected = [path for path in overlay_paths if path.exists()][:max_items]
    if not selected:
        return
    columns = min(8, len(selected))
    rows = (len(selected) + columns - 1) // columns
    cell_width = 170
    cell_height = 520
    label_height = 22
    margin = 8
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (238, 238, 232))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(selected):
        col = index % columns
        row = index // columns
        x0 = col * cell_width
        y0 = row * cell_height
        draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 1), outline=(190, 190, 184))
        label = path.stem.replace("aligned_", "")
        draw.text((x0 + 4, y0 + 3), label[:24], fill=(35, 35, 35))
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((cell_width - 2 * margin, cell_height - label_height - 2 * margin), Image.Resampling.LANCZOS)
        paste_x = x0 + (cell_width - thumb.width) // 2
        paste_y = y0 + label_height + margin
        sheet.paste(thumb, (paste_x, paste_y))
    sheet.save(output_path, quality=92)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    write_json(path, {"type": "FeatureCollection", "features": features})


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
