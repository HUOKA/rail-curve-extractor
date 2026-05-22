from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_rail_seg_deeplab import build_deeplab_model, predict_image_probability, resolve_device  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a native-patch DeepLab rail segmentation model on arbitrary images.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model", required=True, help="Path to rail_semantic_deeplab_resnet50.pt.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.0, help="0 uses checkpoint threshold.")
    parser.add_argument("--crop-size", type=int, default=0, help="0 uses checkpoint crop_size.")
    parser.add_argument("--stride", type=int, default=0, help="0 uses checkpoint stride.")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--contact-sheet-max", type=int, default=16)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, checkpoint = load_model(model_path, device)
    threshold = args.threshold if args.threshold > 0 else float(checkpoint.get("threshold", 0.5))
    crop_size = args.crop_size if args.crop_size > 0 else int(checkpoint.get("crop_size", 768))
    stride = args.stride if args.stride > 0 else int(checkpoint.get("stride", max(1, crop_size // 2)))

    image_paths = discover_images(input_dir)
    if args.sample_every > 1:
        image_paths = image_paths[:: args.sample_every]
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"No supported images found in {input_dir}")

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
        probability = predict_image_probability(
            model,
            image,
            crop_size=crop_size,
            stride=stride,
            device=device,
            channels_last=args.channels_last,
        )
        prediction = probability >= threshold
        stem = image_path.stem
        Image.fromarray(np.clip(probability * 255.0, 0, 255).astype(np.uint8), mode="L").save(prob_dir / f"{stem}.png")
        Image.fromarray((prediction.astype(np.uint8) * 255), mode="L").save(mask_dir / f"{stem}.png")
        overlay_path = overlay_dir / f"{stem}.png"
        save_prediction_overlay(overlay_path, image, probability, prediction)
        overlay_paths.append(overlay_path)
        summaries.append(
            {
                "image_name": image_path.name,
                "width": image.width,
                "height": image.height,
                "positive_fraction": float(np.count_nonzero(prediction) / prediction.size),
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
        "architecture": checkpoint.get("architecture", ""),
        "image_count": len(image_paths),
        "threshold": threshold,
        "crop_size": crop_size,
        "stride": stride,
        "contact_sheet_path": str(contact_sheet_path),
        "images": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_model(model_path: Path, device):
    import torch

    checkpoint = torch.load(model_path, map_location=device)
    model, _ = build_deeplab_model(pretrained_backbone=False, allow_random_init=True, aux_loss=bool(checkpoint.get("aux_loss", True)))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def discover_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def save_prediction_overlay(path: Path, image: Image.Image, probability: np.ndarray, prediction: np.ndarray) -> None:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_arr = np.asarray(overlay).copy()
    strong = prediction & (probability >= 0.9)
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
