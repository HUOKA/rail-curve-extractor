from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_rail_seg_semantic import (  # noqa: E402
    discover_items,
    item_has_foreground,
    load_class_names,
    load_label_masks,
    load_rgb,
    metrics_from_counts,
    resolve_device,
    resolve_label_ids,
    save_overlay,
    set_deterministic,
    split_labeled,
    tile_offsets,
    write_history_csv,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a DeepLabV3-ResNet50 rail segmentation model on native-resolution "
            "image patches. Patches are cropped, not resized, so thin rails keep their source pixels."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory with images/, labels/, and classes.txt.")
    parser.add_argument("--out", required=True, help="Output directory for model, metrics, masks, and overlays.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=768, help="Native-pixel square patch size; no resize is applied.")
    parser.add_argument("--stride", type=int, default=512, help="Native-pixel patch stride.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold also reported beside best threshold.")
    parser.add_argument("--foreground-label", default="单根铁轨")
    parser.add_argument("--ignore-labels", default="从转辙机开始到岔心后面一点结束的道岔区域,ignore_area")
    parser.add_argument("--min-positive-pixels", type=int, default=24, help="Keep all patches with at least this many rail pixels.")
    parser.add_argument("--negative-keep-ratio", type=float, default=0.08, help="Fraction of empty/negative patches to keep.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means no cap; useful for smoke runs.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="0 means no cap; useful for smoke runs.")
    parser.add_argument("--max-predict-images", type=int, default=0, help="0 predicts all dataset images after training.")
    parser.add_argument("--no-predict", action="store_true")
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-random-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aux-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--freeze-bn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep BatchNorm layers in eval mode; recommended for native-patch training with batch=1.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader

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

    train_samples = build_patch_samples(
        train_items,
        crop_size=args.crop_size,
        stride=args.stride,
        foreground_ids=foreground_ids,
        ignore_ids=ignore_ids,
        min_positive_pixels=args.min_positive_pixels,
        negative_keep_ratio=args.negative_keep_ratio,
        seed=args.seed,
    )
    val_samples = build_patch_samples(
        val_items,
        crop_size=args.crop_size,
        stride=args.stride,
        foreground_ids=foreground_ids,
        ignore_ids=ignore_ids,
        min_positive_pixels=1,
        negative_keep_ratio=0.25,
        seed=args.seed + 1,
    )
    if args.max_train_samples > 0:
        train_samples = balanced_sample_cap(train_samples, args.max_train_samples, args.seed)
    if args.max_val_samples > 0:
        val_samples = balanced_sample_cap(val_samples, args.max_val_samples, args.seed + 2)
    if not train_samples or not val_samples:
        raise ValueError("No train or validation patches were created.")

    train_dataset = RailPatchDataset(train_samples, crop_size=args.crop_size, foreground_ids=foreground_ids, ignore_ids=ignore_ids, augment=True)
    val_dataset = RailPatchDataset(val_samples, crop_size=args.crop_size, foreground_ids=foreground_ids, ignore_ids=ignore_ids, augment=False)
    positive_fraction = estimate_positive_fraction(train_dataset)
    pos_weight = float(np.clip((1.0 - positive_fraction) / max(positive_fraction, 1e-6), 1.0, 80.0))

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

    model, pretrained_note = build_deeplab_model(
        pretrained_backbone=args.pretrained_backbone,
        allow_random_init=args.allow_random_init,
        aux_loss=args.aux_loss,
    )
    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    thresholds = [float(v) for v in np.linspace(0.05, 0.95, 19)]

    best_state: dict[str, Any] | None = None
    best_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            bce_loss,
            scaler,
            device,
            channels_last=args.channels_last,
            freeze_bn=args.freeze_bn,
        )
        threshold, best_metrics, fixed_metrics = evaluate_thresholds(model, val_loader, device, thresholds, args.threshold, channels_last=args.channels_last)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "best_threshold": threshold,
            **{f"val_best_{key}": value for key, value in best_metrics.items()},
            **{f"val_fixed_{key}": value for key, value in fixed_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={train_loss:.4f} "
            f"thr={threshold:.2f} f1={best_metrics['f1']:.4f} iou={best_metrics['iou']:.4f} "
            f"p={best_metrics['precision']:.4f} r={best_metrics['recall']:.4f}"
        )
        if best_metrics["f1"] > best_f1:
            best_f1 = float(best_metrics["f1"])
            best_state = {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "threshold": threshold,
                "metrics": best_metrics,
            }

    if best_state is None:
        raise RuntimeError("Training produced no model state.")
    model.load_state_dict(best_state["model"])
    model_path = output_dir / "rail_semantic_deeplab_resnet50.pt"
    checkpoint = {
        "architecture": "deeplabv3_resnet50_binary_native_patch",
        "model_state": best_state["model"],
        "crop_size": args.crop_size,
        "stride": args.stride,
        "threshold": best_state["threshold"],
        "epoch": best_state["epoch"],
        "class_names": class_names,
        "foreground_label": args.foreground_label,
        "foreground_ids": sorted(foreground_ids),
        "ignore_labels": ignore_names,
        "ignore_ids": sorted(ignore_ids),
        "pretrained_backbone": args.pretrained_backbone,
        "pretrained_note": pretrained_note,
        "aux_loss": args.aux_loss,
        "freeze_bn": args.freeze_bn,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    }
    torch.save(checkpoint, model_path)

    best_threshold, best_metrics, fixed_metrics = evaluate_thresholds(model, val_loader, device, thresholds, args.threshold, channels_last=args.channels_last)
    metrics_payload = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "architecture": checkpoint["architecture"],
        "pretrained_note": pretrained_note,
        "source_train_count": len(train_items),
        "source_val_count": len(val_items),
        "unlabeled_count": len(items) - len(labeled),
        "class_names": class_names,
        "foreground_label": args.foreground_label,
        "foreground_ids": sorted(foreground_ids),
        "ignore_labels": ignore_names,
        "ignore_ids": sorted(ignore_ids),
        "crop_size": args.crop_size,
        "stride": args.stride,
        "train_patch_count": len(train_samples),
        "val_patch_count": len(val_samples),
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
        predict_items(
            model=model,
            items=items,
            output_dir=prediction_dir,
            crop_size=args.crop_size,
            stride=args.stride,
            threshold=best_threshold,
            device=device,
            foreground_ids=foreground_ids,
            ignore_ids=ignore_ids,
            max_images=args.max_predict_images,
            channels_last=args.channels_last,
        )
        metrics_payload["prediction_dir"] = str(prediction_dir)
        write_json(output_dir / "metrics.json", metrics_payload)

    summary = {
        **{key: value for key, value in metrics_payload.items() if key != "history"},
        "prediction_dir": str(prediction_dir) if prediction_dir else None,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Model: {model_path}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    if prediction_dir:
        print(f"Predictions: {prediction_dir}")
    print(
        "best-threshold metrics: "
        f"precision={best_metrics['precision']:.4f}, "
        f"recall={best_metrics['recall']:.4f}, "
        f"f1={best_metrics['f1']:.4f}, "
        f"iou={best_metrics['iou']:.4f}, "
        f"threshold={best_threshold:.2f}"
    )
    return 0


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_deeplab_model(*, pretrained_backbone: bool, allow_random_init: bool, aux_loss: bool):
    import torch
    from torchvision.models import ResNet50_Weights
    from torchvision.models.segmentation import deeplabv3_resnet50

    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    note = "imagenet_backbone" if pretrained_backbone else "random_init"
    try:
        model = deeplabv3_resnet50(weights=None, weights_backbone=weights_backbone, aux_loss=aux_loss, num_classes=1)
    except Exception as exc:
        if not pretrained_backbone or not allow_random_init:
            raise
        print(f"warning: pretrained backbone load failed ({exc}); falling back to random init")
        model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=aux_loss, num_classes=1)
        note = f"random_init_after_pretrained_failure: {type(exc).__name__}"
    if hasattr(model, "aux_classifier") and model.aux_classifier is None:
        model.aux_classifier = torch.nn.Identity()
    return model, note


def build_patch_samples(
    items: list[dict[str, Any]],
    *,
    crop_size: int,
    stride: int,
    foreground_ids: set[int],
    ignore_ids: set[int],
    min_positive_pixels: int,
    negative_keep_ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    for item in items:
        image = load_rgb(item["image_path"])
        width, height = image.size
        mask_img, valid_img = load_label_masks(item["label_path"], width, height, foreground_ids, ignore_ids)
        mask = np.asarray(mask_img, dtype=np.uint8) > 0
        valid = np.asarray(valid_img, dtype=np.uint8) > 0
        for top in tile_offsets(height, crop_size, stride):
            crop_h = min(crop_size, height - top)
            for left in tile_offsets(width, crop_size, stride):
                crop_w = min(crop_size, width - left)
                area_mask = mask[top : top + crop_h, left : left + crop_w] & valid[top : top + crop_h, left : left + crop_w]
                positive_pixels = int(np.count_nonzero(area_mask))
                if positive_pixels >= min_positive_pixels or rng.random() < negative_keep_ratio:
                    samples.append(
                        {
                            **item,
                            "crop_left": left,
                            "crop_top": top,
                            "crop_width": crop_w,
                            "crop_height": crop_h,
                            "positive_pixels": positive_pixels,
                        }
                    )
    samples.sort(key=lambda row: (row["name"], -int(row["positive_pixels"]), int(row["crop_top"]), int(row["crop_left"])))
    return samples


def balanced_sample_cap(samples: list[dict[str, Any]], max_count: int, seed: int) -> list[dict[str, Any]]:
    if max_count <= 0 or len(samples) <= max_count:
        return samples
    positives = [sample for sample in samples if int(sample.get("positive_pixels", 0)) > 0]
    negatives = [sample for sample in samples if int(sample.get("positive_pixels", 0)) <= 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    keep_positive = min(len(positives), max(1, int(max_count * 0.8)))
    selected = positives[:keep_positive] + negatives[: max_count - keep_positive]
    if len(selected) < max_count:
        selected.extend(positives[keep_positive : keep_positive + (max_count - len(selected))])
    selected.sort(key=lambda row: (row["name"], int(row["crop_top"]), int(row["crop_left"])))
    return selected


class RailPatchDataset:
    def __init__(
        self,
        samples: list[dict[str, Any]],
        *,
        crop_size: int,
        foreground_ids: set[int],
        ignore_ids: set[int],
        augment: bool,
    ) -> None:
        self.samples = samples
        self.crop_size = crop_size
        self.foreground_ids = foreground_ids
        self.ignore_ids = ignore_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        image = load_rgb(sample["image_path"])
        mask, valid = load_label_masks(sample["label_path"], image.size[0], image.size[1], self.foreground_ids, self.ignore_ids)
        box = (
            int(sample["crop_left"]),
            int(sample["crop_top"]),
            int(sample["crop_left"]) + int(sample["crop_width"]),
            int(sample["crop_top"]) + int(sample["crop_height"]),
        )
        image = image.crop(box)
        mask = mask.crop(box)
        valid = valid.crop(box)
        image, mask, valid = pad_to_square(image, mask, valid, self.crop_size)
        if self.augment:
            image, mask, valid = augment_patch(image, mask, valid)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        image_arr = (image_arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        valid_arr = (np.asarray(valid, dtype=np.uint8) > 0).astype(np.float32)
        return (
            torch.from_numpy(image_arr.transpose(2, 0, 1)).float(),
            torch.from_numpy(mask_arr[None, :, :]).float(),
            torch.from_numpy(valid_arr[None, :, :]).float(),
        )


def pad_to_square(image: Image.Image, mask: Image.Image, valid: Image.Image, crop_size: int) -> tuple[Image.Image, Image.Image, Image.Image]:
    if image.size == (crop_size, crop_size):
        return image, mask, valid
    padded_image = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))
    padded_mask = Image.new("L", (crop_size, crop_size), 0)
    padded_valid = Image.new("L", (crop_size, crop_size), 0)
    padded_image.paste(image, (0, 0))
    padded_mask.paste(mask, (0, 0))
    padded_valid.paste(valid, (0, 0))
    return padded_image, padded_mask, padded_valid


def augment_patch(image: Image.Image, mask: Image.Image, valid: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        valid = valid.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.25:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        valid = valid.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if random.random() < 0.6:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.82, 1.18))
    if random.random() < 0.6:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.82, 1.25))
    if random.random() < 0.35:
        image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.12))
    return image, mask, valid


def estimate_positive_fraction(dataset: RailPatchDataset, max_samples: int = 80) -> float:
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


def train_one_epoch(model, loader, optimizer, bce_loss, scaler, device, *, channels_last: bool, freeze_bn: bool) -> float:
    import torch

    model.train()
    if freeze_bn:
        set_batchnorm_eval(model)
    losses: list[float] = []
    for images, masks, valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        if channels_last:
            images = images.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            outputs = model(images)
            logits = outputs["out"]
            loss = masked_bce_dice_loss(logits, masks, valid, bce_loss)
            aux_logits = outputs.get("aux")
            if aux_logits is not None and getattr(aux_logits, "shape", None) == logits.shape:
                loss = loss + 0.35 * masked_bce_dice_loss(aux_logits, masks, valid, bce_loss)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def set_batchnorm_eval(model) -> None:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def masked_bce_dice_loss(logits, masks, valid, bce_loss):
    pixel_loss = bce_loss(logits, masks)
    bce = (pixel_loss * valid).sum() / torch_clamp(valid.sum(), min_value=1.0)
    return bce + dice_loss(logits, masks, valid)


def torch_clamp(value, *, min_value: float):
    import torch

    return torch.clamp(value, min=min_value)


def dice_loss(logits, targets, valid, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits) * valid
    targets = targets * valid
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    union = torch.sum(probs + targets, dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def evaluate_thresholds(model, loader, device, thresholds: list[float], fixed_threshold: float, *, channels_last: bool):
    import torch

    model.eval()
    threshold_counts = {threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for threshold in thresholds}
    fixed_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    with torch.no_grad():
        for images, masks, valid in loader:
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.to(memory_format=torch.channels_last)
            probs = torch.sigmoid(model(images)["out"]).detach().cpu().numpy()
            truth = masks.numpy() > 0.5
            valid_arr = valid.numpy() > 0.5
            for threshold in thresholds:
                update_counts(threshold_counts[threshold], truth, probs >= threshold, valid_arr)
            update_counts(fixed_counts, truth, probs >= fixed_threshold, valid_arr)
    best_threshold = thresholds[0]
    best_metrics = metrics_from_counts(threshold_counts[best_threshold])
    for threshold in thresholds[1:]:
        metrics = metrics_from_counts(threshold_counts[threshold])
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics
    fixed_metrics = metrics_from_counts(fixed_counts)
    return float(best_threshold), best_metrics, fixed_metrics


def update_counts(counts: dict[str, int], truth: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> None:
    truth = truth & valid
    pred = pred & valid
    counts["tp"] += int(np.count_nonzero(truth & pred))
    counts["fp"] += int(np.count_nonzero(~truth & pred & valid))
    counts["fn"] += int(np.count_nonzero(truth & ~pred))
    counts["tn"] += int(np.count_nonzero(~truth & ~pred & valid))


def predict_items(
    *,
    model,
    items: list[dict[str, Any]],
    output_dir: Path,
    crop_size: int,
    stride: int,
    threshold: float,
    device,
    foreground_ids: set[int],
    ignore_ids: set[int],
    max_images: int,
    channels_last: bool,
) -> None:
    prob_dir = output_dir / "probabilities"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    prob_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    selected = items[:max_images] if max_images > 0 else items
    for item in selected:
        image = load_rgb(item["image_path"])
        probability = predict_image_probability(model, image, crop_size=crop_size, stride=stride, device=device, channels_last=channels_last)
        prediction = probability >= threshold
        truth_img, valid_img = load_label_masks(item["label_path"], image.size[0], image.size[1], foreground_ids, ignore_ids)
        truth = np.asarray(truth_img, dtype=np.uint8) > 0
        valid = np.asarray(valid_img, dtype=np.uint8) > 0
        stem = Path(item["image_path"]).stem
        Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8), mode="L").save(prob_dir / f"{stem}.png")
        Image.fromarray((prediction.astype(np.uint8) * 255), mode="L").save(mask_dir / f"{stem}.png")
        save_overlay(overlay_dir / f"{stem}.jpg", image, truth, prediction, valid)


def predict_image_probability(model, image: Image.Image, *, crop_size: int, stride: int, device, channels_last: bool) -> np.ndarray:
    import torch

    model.eval()
    width, height = image.size
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    x_offsets = tile_offsets(width, crop_size, stride)
    y_offsets = tile_offsets(height, crop_size, stride)
    with torch.no_grad():
        for top in y_offsets:
            crop_h = min(crop_size, height - top)
            for left in x_offsets:
                crop_w = min(crop_size, width - left)
                crop = image.crop((left, top, left + crop_w, top + crop_h))
                crop, _, _ = pad_to_square(crop, Image.new("L", crop.size, 0), Image.new("L", crop.size, 255), crop_size)
                array = np.asarray(crop, dtype=np.float32) / 255.0
                array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
                tensor = torch.from_numpy(array.transpose(2, 0, 1)[None, :, :, :]).float().to(device)
                if channels_last:
                    tensor = tensor.to(memory_format=torch.channels_last)
                probs = torch.sigmoid(model(tensor)["out"])[0, 0].detach().cpu().numpy()
                probability_sum[top : top + crop_h, left : left + crop_w] += probs[:crop_h, :crop_w]
                weight_sum[top : top + crop_h, left : left + crop_w] += 1.0
    return probability_sum / np.maximum(weight_sum, 1e-6)


if __name__ == "__main__":
    raise SystemExit(main())
