from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rail_curve_extractor",
        description="Extract rail centerline from a selected lidar point cloud and export USD BasisCurves.",
    )
    parser.add_argument("--input", required=True, help="Input point cloud path.")
    parser.add_argument("--output-dir", required=True, help="Directory for extracted outputs.")
    parser.add_argument("--config", help="Optional JSON config file.", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config) if args.config else None,
    )
    return 0
