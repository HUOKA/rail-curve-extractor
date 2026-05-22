from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rail_curve_extractor.rail_segmentation_baseline import BaselineTrainOptions, train_rail_segmentation_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a lightweight rail-area segmentation baseline from local CVAT labels.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset directory generated from CVAT annotations.")
    parser.add_argument("--out", required=True, help="Output directory for model, metrics, masks, and overlays.")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO segmentation class id to train. Default: 0.")
    parser.add_argument("--val-every", type=int, default=5, help="Use every Nth labeled image as validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pixel sampling.")
    parser.add_argument("--max-positive-per-image", type=int, default=20000)
    parser.add_argument("--max-negative-per-image", type=int, default=20000)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--min-component-pixels", type=int, default=512)
    parser.add_argument("--no-predict-all", action="store_true", help="Only evaluate train/validation images.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = train_rail_segmentation_baseline(
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.out),
        options=BaselineTrainOptions(
            class_id=args.class_id,
            val_every=args.val_every,
            seed=args.seed,
            max_positive_per_image=args.max_positive_per_image,
            max_negative_per_image=args.max_negative_per_image,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            l2=args.l2,
            min_component_pixels=args.min_component_pixels,
            predict_all=not args.no_predict_all,
        ),
    )
    print(f"Dataset: {result.dataset_dir}")
    print(f"Class: {result.class_name} ({result.class_id})")
    print(f"Train/val/unlabeled: {result.train_count}/{result.val_count}/{result.unlabeled_count}")
    print(f"Threshold: {result.threshold:.4f}")
    print(
        "Validation: "
        f"precision={result.val_metrics['precision']:.4f}, "
        f"recall={result.val_metrics['recall']:.4f}, "
        f"f1={result.val_metrics['f1']:.4f}, "
        f"iou={result.val_metrics['iou']:.4f}"
    )
    print(f"Model: {result.model_path}")
    print(f"Metrics: {result.metrics_json_path}")
    print(f"Report: {result.report_path}")
    print(f"Validation overlays: {result.val_overlay_dir}")
    if result.prediction_dir:
        print(f"All predictions: {result.prediction_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
