from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rail_curve_extractor.cvat_annotations import CvatConversionOptions, convert_cvat_annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert CVAT image annotations into georeferenced segmentation labels.",
    )
    parser.add_argument("--annotations", required=True, help="CVAT annotations XML, zip, or directory.")
    parser.add_argument("--tile-georef", required=True, help="tile_georef.csv or tile_georef.json.")
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated class order. Defaults to CVAT labels if omitted.",
    )
    parser.add_argument("--no-copy-images", action="store_true", help="Do not copy tiles into the output dataset.")
    parser.add_argument("--overlay", action="store_true", help="Write overlay previews for quick spot checks.")
    parser.add_argument("--no-empty-labels", action="store_true", help="Skip empty label files for unlabeled images.")
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Keep running when some CVAT image names are not present in tile_georef metadata.",
    )
    parser.add_argument(
        "--skip-unknown-labels",
        action="store_true",
        help="Skip labels that are not listed in --classes or the CVAT XML metadata.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    class_names = tuple(
        name.strip()
        for name in args.classes.split(",")
        if name.strip()
    )
    result = convert_cvat_annotations(
        annotation_path=Path(args.annotations),
        tile_georef_path=Path(args.tile_georef),
        output_dir=Path(args.out),
        options=CvatConversionOptions(
            class_names=class_names,
            copy_images=not args.no_copy_images,
            make_overlays=args.overlay,
            include_empty_labels=not args.no_empty_labels,
            allow_unmatched=args.allow_unmatched,
            skip_unknown_labels=args.skip_unknown_labels,
        ),
    )
    print(f"Annotations: {result.annotation_path}")
    print(f"Tiles: {result.matched_image_count}/{result.image_count}")
    print(f"Shapes: {result.shape_count}")
    print(f"Skipped shapes: {result.skipped_shape_count}")
    print(f"Output: {result.output_dir}")
    print(f"Labels: {result.labels_dir}")
    if result.overlays_dir:
        print(f"Overlays: {result.overlays_dir}")
    print(f"GeoJSON: {result.geojson_path}")
    print(f"Manifest: {result.manifest_csv_path}")
    print(f"Summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
