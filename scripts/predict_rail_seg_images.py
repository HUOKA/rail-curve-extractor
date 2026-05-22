from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from train_rail_seg_semantic import SmallUNet, resolve_device, tile_offsets


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a trained rail semantic segmentation model on arbitrary images.")
    parser.add_argument("--input-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--model", required=True, help="Path to rail_semantic_unet.pt.")
    parser.add_argument("--out", required=True, help="Output directory for masks, probabilities, overlays, and contact sheet.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--threshold", type=float, default=0.0, help="0 uses the threshold stored in the model.")
    parser.add_argument("--tile-width", type=int, default=768)
    parser.add_argument("--tile-height", type=int, default=1024)
    parser.add_argument("--tile-stride-x", type=int, default=512)
    parser.add_argument("--tile-stride-y", type=int, default=512)
    parser.add_argument("--sample-every", type=int, default=1, help="Use every Nth image after sorting.")
    parser.add_argument("--max-images", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--contact-sheet-max", type=int, default=16)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(input_dir)
    if args.sample_every > 1:
        image_paths = image_paths[:: args.sample_every]
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"No supported images found in {input_dir}")

    device = resolve_device(args.device)
    model, model_threshold, input_size = load_model(model_path, device)
    threshold = args.threshold if args.threshold > 0 else model_threshold

    prob_dir = output_dir / "probabilities"
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    prob_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    overlay_dir.mkdir(exist_ok=True)

    summaries: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        probability = predict_image(
            model=model,
            image=image,
            input_size=input_size,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            stride_x=args.tile_stride_x,
            stride_y=args.tile_stride_y,
            device=device,
        )
        prediction = probability >= threshold
        stem = image_path.stem
        Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8), mode="L").save(prob_dir / f"{stem}.png")
        Image.fromarray((prediction.astype(np.uint8) * 255), mode="L").save(mask_dir / f"{stem}.png")
        overlay_path = overlay_dir / f"{stem}.png"
        save_prediction_overlay(overlay_path, image, probability, prediction)
        overlay_paths.append(overlay_path)
        positive_fraction = float(np.count_nonzero(prediction) / prediction.size)
        summaries.append(
            {
                "image_name": image_path.name,
                "width": image.width,
                "height": image.height,
                "positive_fraction": positive_fraction,
                "mean_probability": float(probability.mean()),
                "max_probability": float(probability.max()),
            }
        )

    contact_sheet_path = output_dir / "contact_sheet.jpg"
    make_contact_sheet(overlay_paths, contact_sheet_path, args.contact_sheet_max)
    summary = {
        "input_dir": str(input_dir),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "image_count": len(image_paths),
        "threshold": threshold,
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "tile_stride_x": args.tile_stride_x,
        "tile_stride_y": args.tile_stride_y,
        "contact_sheet_path": str(contact_sheet_path),
        "images": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def load_model(model_path: Path, device):
    import torch

    checkpoint = torch.load(model_path, map_location=device)
    base_channels = int(checkpoint.get("base_channels", 16))
    input_width = int(checkpoint.get("input_width", 384))
    input_height = int(checkpoint.get("input_height", 1024))
    threshold = float(checkpoint.get("threshold", 0.5))
    model = SmallUNet(in_channels=3, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, threshold, (input_width, input_height)


def predict_image(
    model,
    image: Image.Image,
    input_size: tuple[int, int],
    tile_width: int,
    tile_height: int,
    stride_x: int,
    stride_y: int,
    device,
) -> np.ndarray:
    import torch

    width, height = image.size
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    x_offsets = tile_offsets(width, tile_width, stride_x)
    y_offsets = tile_offsets(height, tile_height, stride_y)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    with torch.no_grad():
        for top in y_offsets:
            bottom = min(top + tile_height, height)
            for left in x_offsets:
                right = min(left + tile_width, width)
                crop = image.crop((left, top, right, bottom))
                input_image = crop.resize(input_size, Image.Resampling.BILINEAR)
                array = np.asarray(input_image, dtype=np.float32) / 255.0
                array = (array - mean) / std
                tensor = torch.from_numpy(array.transpose(2, 0, 1)[None, :, :, :]).float().to(device)
                probs = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
                prob_img = Image.fromarray(np.clip(probs * 255.0, 0, 255).astype(np.uint8), mode="L").resize(
                    (right - left, bottom - top),
                    Image.Resampling.BILINEAR,
                )
                prob_arr = np.asarray(prob_img, dtype=np.float32) / 255.0
                probability_sum[top:bottom, left:right] += prob_arr
                weight_sum[top:bottom, left:right] += 1.0
    return probability_sum / np.maximum(weight_sum, 1e-6)


def save_prediction_overlay(path: Path, image: Image.Image, probability: np.ndarray, prediction: np.ndarray) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_arr = np.asarray(overlay).copy()
    strong = prediction & (probability >= 0.95)
    weak = prediction & ~strong
    overlay_arr[weak] = np.array([230, 57, 70, 95], dtype=np.uint8)
    overlay_arr[strong] = np.array([255, 183, 3, 135], dtype=np.uint8)
    Image.alpha_composite(base, Image.fromarray(overlay_arr, mode="RGBA")).convert("RGB").save(path)


def make_contact_sheet(overlay_paths: list[Path], output_path: Path, max_items: int) -> None:
    selected = overlay_paths[:max_items]
    if not selected:
        return
    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    cell_width = 290
    cell_height = 245
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
        draw.text((x0 + 5, y0 + 4), path.stem[-28:], fill=(35, 35, 35))
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((cell_width - 2 * margin, cell_height - label_height - 2 * margin), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x0 + (cell_width - thumb.width) // 2, y0 + label_height + margin))
    sheet.save(output_path, quality=92)


if __name__ == "__main__":
    raise SystemExit(main())
