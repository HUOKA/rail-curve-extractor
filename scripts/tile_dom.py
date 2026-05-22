from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rail_curve_extractor.dom_tiler import DomTileOptions, stride_from_overlap, tile_dom


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tile a georeferenced DOM GeoTIFF into CVAT-ready images with georeference metadata.",
    )
    parser.add_argument("--dom", required=True, help="DOM GeoTIFF path or DJI Terra output directory.")
    parser.add_argument("--out", required=True, help="Output directory for images and metadata.")
    parser.add_argument("--tile-size", type=int, default=3072, help="Legacy square tile size in pixels.")
    parser.add_argument("--tile-width", type=int, default=None, help="Tile width in pixels. Defaults to tile-size.")
    parser.add_argument("--tile-height", type=int, default=None, help="Tile height in pixels. Defaults to tile-size.")
    parser.add_argument("--stride", type=int, default=None, help="Legacy square stride in pixels.")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio, e.g. 0.5 means 50%% overlap. Used when --stride is not set.",
    )
    parser.add_argument("--format", default="png", choices=["png", "jpg", "jpeg"], help="Output image format.")
    parser.add_argument("--prefix", default="dom", help="Output tile filename prefix.")
    parser.add_argument("--max-tiles", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--no-skip-empty", action="store_true", help="Keep visually empty / nodata tiles.")
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.01,
        help="Minimum valid pixel ratio when skipping empty tiles.",
    )
    parser.add_argument("--blank-threshold", type=int, default=0, help="Pixels at or below this value count as blank.")
    parser.add_argument("--include-alpha", action="store_true", help="Write RGBA PNG tiles when the source has alpha.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tile_width = args.tile_width or args.tile_size
    tile_height = args.tile_height or args.tile_size
    if args.stride is None:
        stride_x = stride_from_overlap(tile_width, args.overlap)
        stride_y = stride_from_overlap(tile_height, args.overlap)
    else:
        stride_x = args.stride
        stride_y = args.stride
    result = tile_dom(
        input_path=Path(args.dom),
        output_dir=Path(args.out),
        options=DomTileOptions(
            tile_width=tile_width,
            tile_height=tile_height,
            stride_x=stride_x,
            stride_y=stride_y,
            image_format=args.format,
            prefix=args.prefix,
            skip_empty=not args.no_skip_empty,
            min_valid_ratio=args.min_valid_ratio,
            blank_threshold=args.blank_threshold,
            include_alpha=args.include_alpha,
            max_tiles=args.max_tiles,
        ),
    )
    print(f"Source: {result.source_path}")
    print(f"Tile: {tile_width}x{tile_height}px")
    print(f"Stride: {stride_x}x{stride_y}px")
    print(f"Tiles: {result.tile_count}")
    print(f"Images: {result.images_dir}")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
