#!/usr/bin/env python3
"""Run DeepLab over the full production DOM and burn the result into a GeoTIFF."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_rail_seg_deeplab import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_deeplab_model,
    pad_to_square,
    resolve_device,
    tile_offsets,
)


DEFAULT_DOM = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u65e0\u4eba\u673a\u6570\u636e" / "\u6b63\u5c04" / "dom.tif"
DEFAULT_MODEL = Path("output/rail_seg_deeplab_resnet50_native_v1/rail_semantic_deeplab_resnet50.pt")
DEFAULT_OUT_DIR = Path("output/full_dom_deeplab_v1_overlay")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Burn DeepLab v1 rail segmentation into the full georeferenced DOM.")
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-name", default="deeplab_v1_full_dom_overlay_weak050_strong090.tif")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-size", type=int, default=0, help="0 uses checkpoint crop_size.")
    parser.add_argument("--stride", type=int, default=0, help="0 uses checkpoint stride.")
    parser.add_argument("--window-size", type=int, default=4096, help="Core DOM window size written per iteration.")
    parser.add_argument("--padding", type=int, default=256, help="Context pixels around each core window.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--weak-threshold", type=float, default=0.50)
    parser.add_argument("--strong-threshold", type=float, default=0.90)
    parser.add_argument("--weak-alpha", type=float, default=0.35)
    parser.add_argument("--strong-alpha", type=float, default=0.55)
    parser.add_argument("--max-windows", type=int, default=0, help="Smoke-test limit. 0 means all windows.")
    parser.add_argument("--start-window", type=int, default=0, help="Skip windows before this zero-based index.")
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dom_path = args.dom.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.output_name
    summary_path = out_dir / "summary.json"
    progress_path = out_dir / "progress.jsonl"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    device = resolve_device(args.device)
    model, checkpoint = load_model(model_path, device, channels_last=args.channels_last)
    crop_size = args.crop_size if args.crop_size > 0 else int(checkpoint.get("crop_size", 768))
    stride = args.stride if args.stride > 0 else int(checkpoint.get("stride", max(1, crop_size // 2)))

    with rasterio.open(dom_path) as src:
        windows = list(iter_core_windows(src.width, src.height, args.window_size))
        if args.start_window > 0:
            windows = windows[args.start_window :]
        if args.max_windows > 0:
            windows = windows[: args.max_windows]
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=3,
            dtype="uint8",
            nodata=None,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            BIGTIFF="YES",
            interleave="pixel",
        )
        start_time = time.time()
        total_positive = 0
        total_strong = 0
        total_pixels = 0
        processed_windows = 0
        if progress_path.exists():
            progress_path.unlink()
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            for absolute_index, core_window in enumerate(windows, start=args.start_window):
                read_window = padded_window(src.width, src.height, core_window, args.padding)
                rgb_read = read_rgb_window(src, read_window)
                probability = predict_array_probability(
                    model,
                    rgb_read,
                    crop_size=crop_size,
                    stride=stride,
                    batch_size=args.batch_size,
                    device=device,
                    channels_last=args.channels_last,
                )
                row_delta = int(core_window.row_off - read_window.row_off)
                col_delta = int(core_window.col_off - read_window.col_off)
                core_h = int(core_window.height)
                core_w = int(core_window.width)
                rgb_core = rgb_read[row_delta : row_delta + core_h, col_delta : col_delta + core_w]
                prob_core = probability[row_delta : row_delta + core_h, col_delta : col_delta + core_w]
                burned = burn_segmentation(
                    rgb_core,
                    prob_core,
                    weak_threshold=args.weak_threshold,
                    strong_threshold=args.strong_threshold,
                    weak_alpha=args.weak_alpha,
                    strong_alpha=args.strong_alpha,
                )
                dst.write(np.moveaxis(burned, -1, 0), window=core_window)

                positive = int(np.count_nonzero(prob_core >= args.weak_threshold))
                strong = int(np.count_nonzero(prob_core >= args.strong_threshold))
                total_positive += positive
                total_strong += strong
                total_pixels += int(prob_core.size)
                processed_windows += 1
                event = {
                    "window_index": absolute_index,
                    "processed_windows": processed_windows,
                    "total_windows": len(windows),
                    "row_off": int(core_window.row_off),
                    "col_off": int(core_window.col_off),
                    "height": int(core_window.height),
                    "width": int(core_window.width),
                    "positive_fraction": positive / max(prob_core.size, 1),
                    "strong_fraction": strong / max(prob_core.size, 1),
                    "elapsed_s": round(time.time() - start_time, 2),
                }
                append_jsonl(progress_path, event)
                print(json.dumps(event, ensure_ascii=False), flush=True)

    summary = {
        "mode": "full_dom_deeplab_v1_burned_overlay",
        "dom": str(dom_path),
        "model": str(model_path),
        "output": str(output_path),
        "device": str(device),
        "architecture": checkpoint.get("architecture", ""),
        "crop_size": crop_size,
        "stride": stride,
        "window_size": args.window_size,
        "padding": args.padding,
        "batch_size": args.batch_size,
        "weak_threshold": args.weak_threshold,
        "strong_threshold": args.strong_threshold,
        "processed_windows": processed_windows,
        "positive_fraction": total_positive / max(total_pixels, 1),
        "strong_fraction": total_strong / max(total_pixels, 1),
        "progress_jsonl": str(progress_path),
        "summary_json": str(summary_path),
        "interpretation": "RGB GeoTIFF with segmentation burned into the DOM: red is weak rail probability, yellow is strong rail probability. This is a visual QA layer, not a final centerline.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_model(model_path: Path, device: Any, *, channels_last: bool):
    import torch

    checkpoint = torch.load(model_path, map_location=device)
    model, _ = build_deeplab_model(pretrained_backbone=False, allow_random_init=True, aux_loss=bool(checkpoint.get("aux_loss", True)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, checkpoint


def iter_core_windows(width: int, height: int, window_size: int) -> list[Window]:
    windows: list[Window] = []
    for row_off in range(0, height, window_size):
        h = min(window_size, height - row_off)
        for col_off in range(0, width, window_size):
            w = min(window_size, width - col_off)
            windows.append(Window(col_off, row_off, w, h))
    return windows


def padded_window(width: int, height: int, window: Window, padding: int) -> Window:
    col0 = max(0, int(window.col_off) - padding)
    row0 = max(0, int(window.row_off) - padding)
    col1 = min(width, int(window.col_off + window.width) + padding)
    row1 = min(height, int(window.row_off + window.height) + padding)
    return Window(col0, row0, col1 - col0, row1 - row0)


def read_rgb_window(src: Any, window: Window) -> np.ndarray:
    if src.count >= 3:
        arr = src.read([1, 2, 3], window=window)
    else:
        one = src.read(1, window=window)
        arr = np.stack([one, one, one], axis=0)
    arr = np.moveaxis(arr, 0, -1)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype("float32")
    finite = arr[np.isfinite(arr)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        if hi > lo:
            arr = (arr - lo) * (255.0 / (hi - lo))
    return np.clip(arr, 0, 255).astype("uint8")


def predict_array_probability(
    model: Any,
    rgb: np.ndarray,
    *,
    crop_size: int,
    stride: int,
    batch_size: int,
    device: Any,
    channels_last: bool,
) -> np.ndarray:
    import torch

    height, width = rgb.shape[:2]
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    x_offsets = tile_offsets(width, crop_size, stride)
    y_offsets = tile_offsets(height, crop_size, stride)
    batch_images: list[np.ndarray] = []
    batch_meta: list[tuple[int, int, int, int]] = []
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)

    def flush_batch() -> None:
        nonlocal batch_images, batch_meta
        if not batch_images:
            return
        tensor_np = np.stack(batch_images, axis=0)
        tensor = torch.from_numpy(tensor_np).float().to(device)
        if channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
            probs = torch.sigmoid(model(tensor)["out"])[:, 0].detach().float().cpu().numpy()
        for prob, (top, left, crop_h, crop_w) in zip(probs, batch_meta):
            probability_sum[top : top + crop_h, left : left + crop_w] += prob[:crop_h, :crop_w]
            weight_sum[top : top + crop_h, left : left + crop_w] += 1.0
        batch_images = []
        batch_meta = []

    for top in y_offsets:
        crop_h = min(crop_size, height - top)
        for left in x_offsets:
            crop_w = min(crop_size, width - left)
            crop_rgb = rgb[top : top + crop_h, left : left + crop_w]
            image = Image.fromarray(crop_rgb, mode="RGB")
            image, _, _ = pad_to_square(image, Image.new("L", image.size, 0), Image.new("L", image.size, 255), crop_size)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - mean) / std
            batch_images.append(array.transpose(2, 0, 1))
            batch_meta.append((top, left, crop_h, crop_w))
            if len(batch_images) >= max(1, batch_size):
                flush_batch()
    flush_batch()
    return probability_sum / np.maximum(weight_sum, 1e-6)


def burn_segmentation(
    rgb: np.ndarray,
    probability: np.ndarray,
    *,
    weak_threshold: float,
    strong_threshold: float,
    weak_alpha: float,
    strong_alpha: float,
) -> np.ndarray:
    out = rgb.astype(np.float32, copy=True)
    weak = probability >= weak_threshold
    strong = probability >= strong_threshold
    weak_only = weak & ~strong
    if np.any(weak_only):
        out[weak_only] = blend(out[weak_only], np.asarray([230, 57, 70], dtype=np.float32), weak_alpha)
    if np.any(strong):
        out[strong] = blend(out[strong], np.asarray([255, 183, 3], dtype=np.float32), strong_alpha)
    return np.clip(out, 0, 255).astype("uint8")


def blend(base: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    return base * (1.0 - alpha) + color * alpha


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
