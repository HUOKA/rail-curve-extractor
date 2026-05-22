from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class BaselineTrainOptions:
    class_id: int = 0
    val_every: int = 5
    seed: int = 42
    max_positive_per_image: int = 20000
    max_negative_per_image: int = 20000
    iterations: int = 350
    learning_rate: float = 0.25
    l2: float = 0.001
    threshold_steps: int = 101
    min_component_pixels: int = 512
    predict_all: bool = True


@dataclass(frozen=True, slots=True)
class BaselineResult:
    dataset_dir: str
    output_dir: str
    class_id: int
    class_name: str
    train_count: int
    val_count: int
    unlabeled_count: int
    threshold: float
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    model_path: str
    split_csv_path: str
    metrics_json_path: str
    report_path: str
    val_overlay_dir: str
    prediction_dir: str | None


@dataclass(frozen=True, slots=True)
class ImageItem:
    name: str
    image_path: Path
    label_path: Path
    is_labeled: bool


def train_rail_segmentation_baseline(
    dataset_dir: Path,
    output_dir: Path,
    options: BaselineTrainOptions | None = None,
) -> BaselineResult:
    options = options or BaselineTrainOptions()
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Dataset images directory does not exist: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Dataset labels directory does not exist: {labels_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = _load_class_names(dataset_dir / "classes.txt")
    if options.class_id < 0 or options.class_id >= len(class_names):
        raise ValueError(f"class_id {options.class_id} is outside classes.txt.")
    items = _discover_items(images_dir, labels_dir)
    labeled_items = [item for item in items if item.is_labeled]
    if len(labeled_items) < 5:
        raise ValueError("At least 5 labeled images are required for a train/validation split.")
    train_items, val_items = _split_labeled_items(labeled_items, options.val_every)
    if not train_items or not val_items:
        raise ValueError("The train/validation split produced an empty subset.")

    rng = np.random.default_rng(options.seed)
    train_features, train_targets = _sample_training_pixels(train_items, options, rng)
    weights, bias, mean, scale, losses = _fit_logistic_regression(train_features, train_targets, options)
    train_probabilities = _sigmoid(((train_features - mean) / scale) @ weights + bias)
    threshold = _best_threshold(train_targets, train_probabilities, options.threshold_steps)

    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "rail_pixel_baseline.npz"
    np.savez_compressed(
        model_path,
        weights=weights,
        bias=np.array([bias], dtype=np.float32),
        mean=mean,
        scale=scale,
        threshold=np.array([threshold], dtype=np.float32),
        class_id=np.array([options.class_id], dtype=np.int32),
        losses=np.asarray(losses, dtype=np.float32),
    )

    split_csv_path = output_dir / "split.csv"
    _write_split_csv(split_csv_path, train_items, val_items, [item for item in items if not item.is_labeled])

    val_overlay_dir = output_dir / "val_overlays"
    val_prediction_dir = output_dir / "val_masks"
    val_overlay_dir.mkdir(exist_ok=True)
    val_prediction_dir.mkdir(exist_ok=True)
    train_metrics = _evaluate_items(
        train_items,
        weights,
        bias,
        mean,
        scale,
        threshold,
        options,
        overlay_dir=None,
        mask_dir=None,
    )
    val_metrics = _evaluate_items(
        val_items,
        weights,
        bias,
        mean,
        scale,
        threshold,
        options,
        overlay_dir=val_overlay_dir,
        mask_dir=val_prediction_dir,
    )

    prediction_dir: Path | None = None
    if options.predict_all:
        prediction_dir = output_dir / "all_predictions"
        mask_dir = prediction_dir / "masks"
        overlay_dir = prediction_dir / "overlays"
        mask_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        _predict_items(
            items,
            weights,
            bias,
            mean,
            scale,
            threshold,
            options,
            mask_dir=mask_dir,
            overlay_dir=overlay_dir,
        )

    metrics_payload = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "class_id": options.class_id,
        "class_name": class_names[options.class_id],
        "train_count": len(train_items),
        "val_count": len(val_items),
        "unlabeled_count": len([item for item in items if not item.is_labeled]),
        "threshold": threshold,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "options": asdict(options),
    }
    metrics_json_path = output_dir / "metrics.json"
    _write_json(metrics_json_path, metrics_payload)
    report_path = output_dir / "REPORT.md"
    _write_report(report_path, metrics_payload)

    return BaselineResult(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        class_id=options.class_id,
        class_name=class_names[options.class_id],
        train_count=len(train_items),
        val_count=len(val_items),
        unlabeled_count=len([item for item in items if not item.is_labeled]),
        threshold=threshold,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        model_path=str(model_path),
        split_csv_path=str(split_csv_path),
        metrics_json_path=str(metrics_json_path),
        report_path=str(report_path),
        val_overlay_dir=str(val_overlay_dir),
        prediction_dir=str(prediction_dir) if prediction_dir is not None else None,
    )


def _discover_items(images_dir: Path, labels_dir: Path) -> list[ImageItem]:
    items: list[ImageItem] = []
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        is_labeled = label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())
        items.append(ImageItem(name=image_path.name, image_path=image_path, label_path=label_path, is_labeled=is_labeled))
    if not items:
        raise ValueError(f"No images found under: {images_dir}")
    return items


def _split_labeled_items(items: list[ImageItem], val_every: int) -> tuple[list[ImageItem], list[ImageItem]]:
    if val_every <= 1:
        raise ValueError("val_every must be greater than 1.")
    train: list[ImageItem] = []
    val: list[ImageItem] = []
    for index, item in enumerate(items):
        if (index + 1) % val_every == 0:
            val.append(item)
        else:
            train.append(item)
    return train, val


def _sample_training_pixels(
    items: list[ImageItem],
    options: BaselineTrainOptions,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    feature_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    for item in items:
        image = _load_rgb(item.image_path)
        mask = _load_mask(item.label_path, image.shape[1], image.shape[0], options.class_id)
        features = _pixel_features(image)
        flat_features = features.reshape(-1, features.shape[-1])
        flat_mask = mask.reshape(-1)
        pos_indices = np.flatnonzero(flat_mask)
        neg_indices = np.flatnonzero(~flat_mask)
        if pos_indices.size == 0 or neg_indices.size == 0:
            continue
        pos_take = min(options.max_positive_per_image, int(pos_indices.size))
        neg_take = min(options.max_negative_per_image, int(neg_indices.size))
        sampled_pos = rng.choice(pos_indices, size=pos_take, replace=False)
        sampled_neg = rng.choice(neg_indices, size=neg_take, replace=False)
        sampled = np.concatenate([sampled_pos, sampled_neg])
        targets = np.concatenate([np.ones(pos_take, dtype=np.float32), np.zeros(neg_take, dtype=np.float32)])
        order = rng.permutation(sampled.size)
        feature_chunks.append(flat_features[sampled[order]].astype(np.float32, copy=False))
        target_chunks.append(targets[order])
    if not feature_chunks:
        raise ValueError("No trainable foreground/background pixels were sampled.")
    return np.vstack(feature_chunks), np.concatenate(target_chunks)


def _fit_logistic_regression(
    features: np.ndarray,
    targets: np.ndarray,
    options: BaselineTrainOptions,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[float]]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    x = (features - mean) / scale
    weights = np.zeros(x.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    losses: list[float] = []
    for iteration in range(options.iterations):
        logits = x @ weights + bias
        probabilities = _sigmoid(logits)
        error = probabilities - targets
        grad_w = (x.T @ error) / targets.size + options.l2 * weights
        grad_b = float(error.mean())
        weights -= np.float32(options.learning_rate) * grad_w.astype(np.float32)
        bias -= np.float32(options.learning_rate * grad_b)
        if iteration % 25 == 0 or iteration == options.iterations - 1:
            loss = _binary_cross_entropy(targets, probabilities) + 0.5 * options.l2 * float(np.sum(weights * weights))
            losses.append(float(loss))
    return weights.astype(np.float32), float(bias), mean, scale, losses


def _evaluate_items(
    items: list[ImageItem],
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    scale: np.ndarray,
    threshold: float,
    options: BaselineTrainOptions,
    overlay_dir: Path | None,
    mask_dir: Path | None,
) -> dict[str, float]:
    totals = _empty_metric_counts()
    per_image_rows: list[dict[str, float | str]] = []
    for item in items:
        image = _load_rgb(item.image_path)
        truth = _load_mask(item.label_path, image.shape[1], image.shape[0], options.class_id)
        prediction = _predict_mask(image, weights, bias, mean, scale, threshold, options)
        counts = _metric_counts(truth, prediction)
        for key, value in counts.items():
            totals[key] += value
        metrics = _metrics_from_counts(counts)
        per_image_rows.append({"image": item.name, **metrics})
        if mask_dir is not None:
            _save_mask(mask_dir / f"{item.image_path.stem}.png", prediction)
        if overlay_dir is not None:
            _save_eval_overlay(overlay_dir / item.image_path.name, image, truth, prediction)
    metrics = _metrics_from_counts(totals)
    metrics["image_count"] = float(len(items))
    _ = per_image_rows
    return metrics


def _predict_items(
    items: list[ImageItem],
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    scale: np.ndarray,
    threshold: float,
    options: BaselineTrainOptions,
    mask_dir: Path,
    overlay_dir: Path,
) -> None:
    for item in items:
        image = _load_rgb(item.image_path)
        prediction = _predict_mask(image, weights, bias, mean, scale, threshold, options)
        truth = (
            _load_mask(item.label_path, image.shape[1], image.shape[0], options.class_id)
            if item.label_path.exists()
            else np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        )
        _save_mask(mask_dir / f"{item.image_path.stem}.png", prediction)
        _save_eval_overlay(overlay_dir / item.image_path.name, image, truth, prediction)


def _predict_mask(
    image: np.ndarray,
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    scale: np.ndarray,
    threshold: float,
    options: BaselineTrainOptions,
) -> np.ndarray:
    features = _pixel_features(image)
    height, width, feature_count = features.shape
    x = ((features.reshape(-1, feature_count) - mean) / scale).astype(np.float32, copy=False)
    probabilities = _sigmoid(x @ weights + bias)
    mask = probabilities.reshape(height, width) >= threshold
    return _postprocess_mask(mask, options.min_component_pixels)


def _postprocess_mask(mask: np.ndarray, min_component_pixels: int) -> np.ndarray:
    cleaned = ndimage.binary_opening(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    cleaned = ndimage.binary_closing(cleaned, structure=np.ones((5, 5), dtype=bool), iterations=1)
    if min_component_pixels <= 0:
        return cleaned
    labels, label_count = ndimage.label(cleaned)
    if label_count == 0:
        return cleaned
    counts = np.bincount(labels.ravel())
    keep = counts >= min_component_pixels
    keep[0] = False
    return keep[labels]


def _pixel_features(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    value = np.max(rgb, axis=2)
    min_value = np.min(rgb, axis=2)
    saturation = (value - min_value) / np.maximum(value, 1e-6)
    gray = (red + green + blue) / 3.0
    height, width = red.shape
    x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)
    x_norm = np.broadcast_to(x_coords[None, :], (height, width))
    center_distance = np.abs(x_norm - 0.5)
    gray_for_filter = gray.astype(np.float32, copy=False)
    sobel_x = np.abs(ndimage.sobel(gray_for_filter, axis=1, mode="nearest")) / 4.0
    sobel_y = np.abs(ndimage.sobel(gray_for_filter, axis=0, mode="nearest")) / 4.0
    local_mean = ndimage.uniform_filter(gray_for_filter, size=9, mode="nearest")
    local_mean_sq = ndimage.uniform_filter(gray_for_filter * gray_for_filter, size=9, mode="nearest")
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0.0))
    horizontal_wide = ndimage.uniform_filter(gray_for_filter, size=(1, 17), mode="nearest")
    horizontal_narrow = ndimage.uniform_filter(gray_for_filter, size=(1, 3), mode="nearest")
    dark_vertical_line = horizontal_wide - horizontal_narrow
    bright_vertical_line = horizontal_narrow - horizontal_wide
    return np.stack(
        [
            red,
            green,
            blue,
            gray,
            value,
            saturation,
            red - green,
            green - blue,
            x_norm,
            center_distance,
            sobel_x,
            sobel_y,
            local_std,
            dark_vertical_line,
            bright_vertical_line,
        ],
        axis=2,
    ).astype(np.float32, copy=False)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_mask(label_path: Path, width: int, height: int, class_id: int) -> np.ndarray:
    mask_image = Image.new("L", (width, height), 0)
    if not label_path.exists():
        return np.zeros((height, width), dtype=bool)
    draw = ImageDraw.Draw(mask_image)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        if int(float(parts[0])) != class_id:
            continue
        values = [float(value) for value in parts[1:]]
        points = [
            (
                float(np.clip(values[index] * width, 0.0, width - 1.0)),
                float(np.clip(values[index + 1] * height, 0.0, height - 1.0)),
            )
            for index in range(0, len(values) - 1, 2)
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _save_eval_overlay(path: Path, image: np.ndarray, truth: np.ndarray, prediction: np.ndarray) -> None:
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_array = np.asarray(overlay).copy()
    truth_only = truth & ~prediction
    pred_only = prediction & ~truth
    overlap = truth & prediction
    overlay_array[truth_only] = np.array([42, 157, 143, 120], dtype=np.uint8)
    overlay_array[pred_only] = np.array([230, 57, 70, 120], dtype=np.uint8)
    overlay_array[overlap] = np.array([255, 183, 3, 145], dtype=np.uint8)
    composed = Image.alpha_composite(base, Image.fromarray(overlay_array, mode="RGBA")).convert("RGB")
    composed.save(path)


def _empty_metric_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _metric_counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, int]:
    truth = truth.astype(bool, copy=False)
    prediction = prediction.astype(bool, copy=False)
    return {
        "tp": int(np.count_nonzero(truth & prediction)),
        "fp": int(np.count_nonzero(~truth & prediction)),
        "fn": int(np.count_nonzero(truth & ~prediction)),
        "tn": int(np.count_nonzero(~truth & ~prediction)),
    }


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    tn = float(counts["tn"])
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    iou = tp / max(tp + fp + fn, 1.0)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _best_threshold(targets: np.ndarray, probabilities: np.ndarray, steps: int) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, max(steps, 3)):
        prediction = probabilities >= threshold
        metrics = _metrics_from_counts(_metric_counts(targets.astype(bool), prediction))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold


def _binary_cross_entropy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"classes.txt does not exist: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"classes.txt is empty: {path}")
    return names


def _write_split_csv(
    path: Path,
    train_items: Iterable[ImageItem],
    val_items: Iterable[ImageItem],
    unlabeled_items: Iterable[ImageItem],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["split", "image", "label"])
        writer.writeheader()
        for split, items in (("train", train_items), ("val", val_items), ("unlabeled", unlabeled_items)):
            for item in items:
                writer.writerow({"split": split, "image": str(item.image_path), "label": str(item.label_path)})


def _write_report(path: Path, payload: dict[str, object]) -> None:
    val_metrics = payload["val_metrics"]
    train_metrics = payload["train_metrics"]
    assert isinstance(val_metrics, dict)
    assert isinstance(train_metrics, dict)
    text = f"""# Rail Segmentation Baseline Report

## Conclusion

This is a lightweight baseline validation, not a production model. It uses only the currently labeled CVAT tiles and existing Python dependencies.

## Data

- Dataset: `{payload["dataset_dir"]}`
- Class: `{payload["class_name"]}`
- Train images: {payload["train_count"]}
- Validation images: {payload["val_count"]}
- Unlabeled images kept for prediction only: {payload["unlabeled_count"]}
- Decision threshold: {payload["threshold"]:.4f}

## Metrics

| Split | Precision | Recall | F1 | IoU | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | {train_metrics["precision"]:.4f} | {train_metrics["recall"]:.4f} | {train_metrics["f1"]:.4f} | {train_metrics["iou"]:.4f} | {train_metrics["accuracy"]:.4f} |
| Validation | {val_metrics["precision"]:.4f} | {val_metrics["recall"]:.4f} | {val_metrics["f1"]:.4f} | {val_metrics["iou"]:.4f} | {val_metrics["accuracy"]:.4f} |

## Overlay Legend

- Green: ground truth only
- Red: prediction only
- Yellow: overlap between ground truth and prediction

## Notes

- Empty CVAT label files are treated as unlabeled, not as true negative samples.
- The baseline includes horizontal position features, so it is mainly a sanity check for the aligned corridor tiles.
- A real model should be trained later with more labels and original DOM direction slices.
"""
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
