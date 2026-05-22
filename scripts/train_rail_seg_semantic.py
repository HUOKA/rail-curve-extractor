from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight semantic rail-body segmentation model from local YOLO polygon labels.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory with images/, labels/, and classes.txt.")
    parser.add_argument("--out", required=True, help="Output directory for model, metrics, masks, and overlays.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--workers", type=int, default=0, help="Windows-safe default is 0.")
    parser.add_argument("--val-every", type=int, default=5, help="Use every Nth labeled image as validation.")
    parser.add_argument("--tile-height", type=int, default=1024)
    parser.add_argument("--tile-stride", type=int, default=512)
    parser.add_argument("--input-width", type=int, default=384)
    parser.add_argument("--input-height", type=int, default=1024)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold used for full-image masks.")
    parser.add_argument("--foreground-label", default="track_area", help="Class name to train as rail foreground.")
    parser.add_argument(
        "--ignore-labels",
        default="switch_area,ignore_area",
        help="Comma-separated class names to exclude from loss and metrics.",
    )
    parser.add_argument("--no-predict", action="store_true", help="Skip full-image prediction after validation.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("Semantic segmentation training requires torch. Run this script with .yolo-venv Python.") from exc

    set_deterministic(args.seed)
    device = resolve_device(args.device)
    class_names = load_class_names(dataset_dir / "classes.txt")
    foreground_ids = resolve_label_ids(class_names, [args.foreground_label], "--foreground-label")
    ignore_names = [name.strip() for name in args.ignore_labels.split(",") if name.strip()]
    ignore_ids = resolve_label_ids(class_names, ignore_names, "--ignore-labels", allow_missing=True)
    items = discover_items(dataset_dir)
    labeled = [item for item in items if item_has_foreground(item["label_path"], foreground_ids)]
    if len(labeled) < 5:
        raise ValueError("At least 5 foreground-labeled images are required for train/validation.")
    train_items, val_items = split_labeled(labeled, args.val_every)
    train_samples = build_crop_samples(train_items, args.tile_height, args.tile_stride)
    val_samples = build_crop_samples(val_items, args.tile_height, args.tile_stride)
    if not train_samples or not val_samples:
        raise ValueError("No train or validation crop samples were created.")

    train_dataset = RailSemanticDataset(
        train_samples,
        (args.input_width, args.input_height),
        foreground_ids=foreground_ids,
        ignore_ids=ignore_ids,
        augment=True,
    )
    val_dataset = RailSemanticDataset(
        val_samples,
        (args.input_width, args.input_height),
        foreground_ids=foreground_ids,
        ignore_ids=ignore_ids,
        augment=False,
    )
    positive_fraction = estimate_positive_fraction(train_dataset)
    pos_weight = float(np.clip((1.0 - positive_fraction) / max(positive_fraction, 1e-6), 1.0, 30.0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=str(device).startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=str(device).startswith("cuda"),
    )

    model = SmallUNet(in_channels=3, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    best_state: dict[str, Any] | None = None
    best_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, bce_loss, scaler, device)
        val_probs, val_targets = collect_probabilities(model, val_loader, device)
        threshold, threshold_metrics = choose_threshold(val_probs, val_targets)
        fixed_metrics = metrics_from_arrays(val_targets, val_probs >= args.threshold)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "best_threshold": float(threshold),
            **{f"val_best_{key}": value for key, value in threshold_metrics.items()},
            **{f"val_fixed_{key}": value for key, value in fixed_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"loss={train_loss:.4f} "
            f"thr={threshold:.2f} "
            f"f1={threshold_metrics['f1']:.4f} "
            f"iou={threshold_metrics['iou']:.4f} "
            f"p={threshold_metrics['precision']:.4f} "
            f"r={threshold_metrics['recall']:.4f}"
        )
        if threshold_metrics["f1"] > best_f1:
            best_f1 = threshold_metrics["f1"]
            best_state = {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "threshold": threshold,
                "metrics": threshold_metrics,
            }

    if best_state is None:
        raise RuntimeError("Training produced no model state.")
    model.load_state_dict(best_state["model"])
    model_path = output_dir / "rail_semantic_unet.pt"
    torch.save(
        {
            "model_state": best_state["model"],
            "base_channels": args.base_channels,
            "input_width": args.input_width,
            "input_height": args.input_height,
            "threshold": best_state["threshold"],
            "epoch": best_state["epoch"],
        },
        model_path,
    )

    val_probs, val_targets = collect_probabilities(model, val_loader, device)
    best_threshold, best_metrics = choose_threshold(val_probs, val_targets)
    fixed_metrics = metrics_from_arrays(val_targets, val_probs >= args.threshold)
    metrics_payload = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "source_train_count": len(train_items),
        "source_val_count": len(val_items),
        "unlabeled_count": len(items) - len(labeled),
        "class_names": class_names,
        "foreground_label": args.foreground_label,
        "foreground_ids": sorted(foreground_ids),
        "ignore_labels": ignore_names,
        "ignore_ids": sorted(ignore_ids),
        "train_tile_count": len(train_samples),
        "val_tile_count": len(val_samples),
        "positive_fraction": positive_fraction,
        "pos_weight": pos_weight,
        "best_epoch": best_state["epoch"],
        "best_threshold": best_threshold,
        "fixed_threshold": args.threshold,
        "best_threshold_metrics": best_metrics,
        "fixed_threshold_metrics": fixed_metrics,
        "history": history,
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    write_history_csv(output_dir / "history.csv", history)

    prediction_dir: Path | None = None
    if not args.no_predict:
        prediction_dir = output_dir / "predictions"
        predict_full_images(
            model=model,
            items=items,
            output_dir=prediction_dir,
            input_size=(args.input_width, args.input_height),
            tile_height=args.tile_height,
            tile_stride=args.tile_stride,
            threshold=best_threshold,
            device=device,
            foreground_ids=foreground_ids,
            ignore_ids=ignore_ids,
        )
        make_contact_sheet(
            items=items,
            prediction_dir=prediction_dir,
            output_path=output_dir / "eval_contact_sheet.jpg",
            max_items=10,
        )
        metrics_payload["full_image_metrics"] = evaluate_full_image_predictions(
            prediction_dir / "masks",
            {
                "train": train_items,
                "val": val_items,
                "labeled": labeled,
            },
            foreground_ids=foreground_ids,
            ignore_ids=ignore_ids,
        )
        write_json(output_dir / "metrics.json", metrics_payload)

    summary = {
        **{key: value for key, value in metrics_payload.items() if key != "history"},
        "prediction_dir": str(prediction_dir) if prediction_dir is not None else None,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Model: {model_path}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    if prediction_dir is not None:
        print(f"Predictions: {prediction_dir}")
        print(f"Contact sheet: {output_dir / 'eval_contact_sheet.jpg'}")
    print(
        "best-threshold metrics: "
        f"precision={best_metrics['precision']:.4f}, "
        f"recall={best_metrics['recall']:.4f}, "
        f"f1={best_metrics['f1']:.4f}, "
        f"iou={best_metrics['iou']:.4f}, "
        f"threshold={best_threshold:.2f}"
    )
    return 0


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(requested: str):
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"classes.txt does not exist: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"classes.txt is empty: {path}")
    return names


def resolve_label_ids(class_names: list[str], names: list[str], option_name: str, allow_missing: bool = False) -> set[int]:
    ids: set[int] = set()
    for name in names:
        if name not in class_names:
            if allow_missing:
                continue
            raise ValueError(f"{option_name} references unknown class {name!r}; classes are {class_names!r}")
        ids.add(class_names.index(name))
    if not ids and not allow_missing:
        raise ValueError(f"{option_name} did not resolve to any class ids.")
    return ids


def item_has_foreground(label_path: Path, foreground_ids: set[int]) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        if int(float(parts[0])) in foreground_ids:
            return True
    return False


def discover_items(dataset_dir: Path) -> list[dict[str, Any]]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("Dataset must contain images/ and labels/ directories.")
    items: list[dict[str, Any]] = []
    for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}):
        label_path = labels_dir / f"{image_path.stem}.txt"
        is_labeled = label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())
        items.append({"name": image_path.name, "image_path": image_path, "label_path": label_path, "is_labeled": is_labeled})
    if not items:
        raise ValueError(f"No images found under: {images_dir}")
    return items


def split_labeled(items: list[dict[str, Any]], val_every: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_every <= 1:
        raise ValueError("val_every must be greater than 1.")
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if (index + 1) % val_every == 0:
            val.append(item)
        else:
            train.append(item)
    if not train or not val:
        raise ValueError("Split produced an empty train or validation set.")
    return train, val


def build_crop_samples(items: list[dict[str, Any]], tile_height: int, tile_stride: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for item in items:
        with Image.open(item["image_path"]) as image:
            width, height = image.size
        for top in tile_offsets(height, tile_height, tile_stride):
            samples.append({**item, "crop_left": 0, "crop_top": top, "crop_width": width, "crop_height": min(tile_height, height - top)})
    return samples


def tile_offsets(length: int, tile_length: int, stride: int) -> list[int]:
    if tile_length <= 0 or stride <= 0:
        raise ValueError("tile_length and stride must be positive.")
    if length <= tile_length:
        return [0]
    offsets = list(range(0, length - tile_length + 1, stride))
    final_offset = length - tile_length
    if offsets[-1] != final_offset:
        offsets.append(final_offset)
    return offsets


class RailSemanticDataset:
    def __init__(
        self,
        samples: list[dict[str, Any]],
        input_size: tuple[int, int],
        foreground_ids: set[int] | None = None,
        ignore_ids: set[int] | None = None,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.input_size = input_size
        self.foreground_ids = foreground_ids or {0}
        self.ignore_ids = ignore_ids or set()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        image = load_rgb(sample["image_path"])
        mask, valid = load_label_masks(sample["label_path"], image.size[0], image.size[1], self.foreground_ids, self.ignore_ids)
        box = (
            sample["crop_left"],
            sample["crop_top"],
            sample["crop_left"] + sample["crop_width"],
            sample["crop_top"] + sample["crop_height"],
        )
        image = image.crop(box)
        mask = mask.crop(box)
        valid = valid.crop(box)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            valid = valid.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.resize(self.input_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.input_size, Image.Resampling.NEAREST)
        valid = valid.resize(self.input_size, Image.Resampling.NEAREST)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        valid_arr = (np.asarray(valid, dtype=np.uint8) > 0).astype(np.float32)
        image_arr = (image_arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (
            torch.from_numpy(image_arr.transpose(2, 0, 1)).float(),
            torch.from_numpy(mask_arr[None, :, :]).float(),
            torch.from_numpy(valid_arr[None, :, :]).float(),
        )


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_mask(label_path: Path, width: int, height: int) -> Image.Image:
    mask, _ = load_label_masks(label_path, width, height, {0}, set())
    return mask


def load_label_masks(
    label_path: Path,
    width: int,
    height: int,
    foreground_ids: set[int],
    ignore_ids: set[int],
) -> tuple[Image.Image, Image.Image]:
    mask = Image.new("L", (width, height), 0)
    valid = Image.new("L", (width, height), 255)
    if not label_path.exists():
        return mask, valid
    mask_draw = ImageDraw.Draw(mask)
    valid_draw = ImageDraw.Draw(valid)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
        points = []
        for value_index in range(0, len(values) - 1, 2):
            x = float(np.clip(values[value_index] * width, 0, width - 1))
            y = float(np.clip(values[value_index + 1] * height, 0, height - 1))
            points.append((x, y))
        if len(points) >= 3:
            if class_id in ignore_ids:
                valid_draw.polygon(points, fill=0)
            elif class_id in foreground_ids:
                mask_draw.polygon(points, fill=255)
    return mask, valid


def estimate_positive_fraction(dataset: RailSemanticDataset, max_samples: int = 64) -> float:
    if len(dataset) == 0:
        return 0.0
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[: min(max_samples, len(indices))]
    positive = 0.0
    total = 0.0
    for index in indices:
        _, mask, valid = dataset[index]
        positive += float((mask * valid).sum().item())
        total += float(valid.sum().item())
    return positive / max(total, 1.0)


def conv_block(in_channels: int, out_channels: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.SiLU(inplace=True),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.SiLU(inplace=True),
    )


try:
    import torch

    _SMALL_UNET_BASE = torch.nn.Module
except ImportError:
    _SMALL_UNET_BASE = object


class SmallUNet(_SMALL_UNET_BASE):
    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        import torch

        b = base_channels
        self.enc1 = conv_block(in_channels, b)
        self.enc2 = conv_block(b, b * 2)
        self.enc3 = conv_block(b * 2, b * 4)
        self.enc4 = conv_block(b * 4, b * 8)
        self.pool = torch.nn.MaxPool2d(2)
        self.bottleneck = conv_block(b * 8, b * 16)
        self.up4 = torch.nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(b * 16, b * 8)
        self.up3 = torch.nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(b * 8, b * 4)
        self.up2 = torch.nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(b * 4, b * 2)
        self.up1 = torch.nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = conv_block(b * 2, b)
        self.out = torch.nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(cat_same(self.up4(b), e4))
        d3 = self.dec3(cat_same(self.up3(d4), e3))
        d2 = self.dec2(cat_same(self.up2(d3), e2))
        d1 = self.dec1(cat_same(self.up1(d2), e1))
        return self.out(d1)


def cat_same(a, b):
    import torch

    if a.shape[-2:] != b.shape[-2:]:
        a = torch.nn.functional.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([a, b], dim=1)


def train_one_epoch(model, loader, optimizer, bce_loss, scaler, device) -> float:
    import torch

    model.train()
    losses: list[float] = []
    for images, masks, valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            logits = model(images)
            pixel_loss = bce_loss(logits, masks)
            loss = (pixel_loss * valid).sum() / torch.clamp(valid.sum(), min=1.0)
            loss = loss + dice_loss(logits, masks, valid)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def dice_loss(logits, targets, valid, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    probs = probs * valid
    targets = targets * valid
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs + targets, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def collect_probabilities(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for images, masks, valid in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            valid_arr = valid.numpy().reshape(-1) > 0
            probs.append(probabilities.reshape(-1)[valid_arr])
            targets.append(masks.numpy().reshape(-1)[valid_arr])
    return np.concatenate(probs), np.concatenate(targets)


def choose_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.9, 18):
        metrics = metrics_from_arrays(targets, probabilities >= threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    return best_threshold, best_metrics


def metrics_from_arrays(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    truth = targets.astype(bool, copy=False)
    pred = predictions.astype(bool, copy=False)
    tp = float(np.count_nonzero(truth & pred))
    fp = float(np.count_nonzero(~truth & pred))
    fn = float(np.count_nonzero(truth & ~pred))
    tn = float(np.count_nonzero(~truth & ~pred))
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


def predict_full_images(
    model,
    items: list[dict[str, Any]],
    output_dir: Path,
    input_size: tuple[int, int],
    tile_height: int,
    tile_stride: int,
    threshold: float,
    device,
    foreground_ids: set[int],
    ignore_ids: set[int],
) -> None:
    import torch

    prob_dir = output_dir / "probabilities"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    prob_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for item in items:
        image = load_rgb(item["image_path"])
        width, height = image.size
        probability_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        for top in tile_offsets(height, tile_height, tile_stride):
            bottom = min(top + tile_height, height)
            crop = image.crop((0, top, width, bottom))
            input_image = crop.resize(input_size, Image.Resampling.BILINEAR)
            array = np.asarray(input_image, dtype=np.float32) / 255.0
            array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = torch.from_numpy(array.transpose(2, 0, 1)[None, :, :, :]).float().to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
            prob_img = Image.fromarray(np.clip(probs * 255.0, 0, 255).astype(np.uint8), mode="L").resize((width, bottom - top), Image.Resampling.BILINEAR)
            prob_arr = np.asarray(prob_img, dtype=np.float32) / 255.0
            probability_sum[top:bottom, :] += prob_arr
            weight_sum[top:bottom, :] += 1.0
        probability = probability_sum / np.maximum(weight_sum, 1e-6)
        prediction = probability >= threshold
        truth_img, valid_img = load_label_masks(item["label_path"], width, height, foreground_ids, ignore_ids)
        truth = np.asarray(truth_img, dtype=np.uint8) > 0
        valid = np.asarray(valid_img, dtype=np.uint8) > 0
        stem = Path(item["image_path"]).stem
        Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8), mode="L").save(prob_dir / f"{stem}.png")
        Image.fromarray((prediction.astype(np.uint8) * 255), mode="L").save(mask_dir / f"{stem}.png")
        save_overlay(overlay_dir / f"{stem}.jpg", image, truth, prediction, valid)


def save_overlay(path: Path, image: Image.Image, truth: np.ndarray, prediction: np.ndarray, valid: np.ndarray | None = None) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_array = np.asarray(overlay).copy()
    if valid is None:
        valid = np.ones_like(truth, dtype=bool)
    ignored = ~valid
    truth_only = truth & ~prediction & valid
    pred_only = prediction & ~truth & valid
    overlap = truth & prediction & valid
    overlay_array[ignored] = np.array([69, 123, 157, 85], dtype=np.uint8)
    overlay_array[truth_only] = np.array([42, 157, 143, 130], dtype=np.uint8)
    overlay_array[pred_only] = np.array([230, 57, 70, 120], dtype=np.uint8)
    overlay_array[overlap] = np.array([255, 183, 3, 150], dtype=np.uint8)
    Image.alpha_composite(base, Image.fromarray(overlay_array, mode="RGBA")).convert("RGB").save(path, quality=92)


def make_contact_sheet(items: list[dict[str, Any]], prediction_dir: Path, output_path: Path, max_items: int) -> None:
    labeled = [item for item in items if item["is_labeled"]]
    selected = labeled[: max_items // 2] + labeled[-max_items // 2 :]
    if not selected:
        return
    cell_w = 160
    cell_h = 520
    sheet = Image.new("RGB", (cell_w * len(selected), cell_h), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(selected):
        stem = Path(item["image_path"]).stem
        overlay_path = prediction_dir / "overlays" / f"{stem}.jpg"
        image = Image.open(overlay_path).convert("RGB") if overlay_path.exists() else load_rgb(item["image_path"])
        image.thumbnail((cell_w - 12, cell_h - 42), Image.Resampling.LANCZOS)
        x = index * cell_w
        draw.rectangle((x, 0, x + cell_w - 1, cell_h - 1), outline=(200, 200, 200))
        draw.text((x + 6, 6), stem.replace("aligned_", ""), fill=(30, 30, 30))
        sheet.paste(image, (x + (cell_w - image.width) // 2, 30))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def evaluate_full_image_predictions(
    mask_dir: Path,
    subsets: dict[str, list[dict[str, Any]]],
    foreground_ids: set[int],
    ignore_ids: set[int],
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for subset_name, items in subsets.items():
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for item in items:
            image = load_rgb(item["image_path"])
            width, height = image.size
            truth_img, valid_img = load_label_masks(item["label_path"], width, height, foreground_ids, ignore_ids)
            truth = np.asarray(truth_img, dtype=np.uint8) > 0
            valid = np.asarray(valid_img, dtype=np.uint8) > 0
            pred_path = mask_dir / f"{Path(item['image_path']).stem}.png"
            prediction = np.asarray(Image.open(pred_path).convert("L"), dtype=np.uint8) > 0
            counts["tp"] += int(np.count_nonzero(truth & prediction & valid))
            counts["fp"] += int(np.count_nonzero(~truth & prediction & valid))
            counts["fn"] += int(np.count_nonzero(truth & ~prediction & valid))
            counts["tn"] += int(np.count_nonzero(~truth & ~prediction & valid))
        results[subset_name] = metrics_from_counts(counts)
    return results


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
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


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_history_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
