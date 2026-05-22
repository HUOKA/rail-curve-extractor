#!/usr/bin/env python3
"""Run the raw-DOM ROI rail-mask full pass and map-space candidate extraction."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TILE_INDEX = Path("output/raw_dom_corridor_roi/raw_dom_tile_index.csv")
DEFAULT_RAIL_MODEL = Path("output/rail_seg_semantic_unet_v7_tonghaigang_chinese/rail_semantic_unet.pt")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw production DOM ROI tiles, run the rail segmentation model, "
            "and extract EPSG:32651 rail/centerline candidates."
        ),
    )
    parser.add_argument("--tile-index", type=Path, default=DEFAULT_TILE_INDEX)
    parser.add_argument("--rail-model", type=Path, default=DEFAULT_RAIL_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda", help="Use cuda/cuda:0 for GPU inference, or cpu for debugging.")
    parser.add_argument("--threshold", type=float, default=0.0, help="0 uses the model checkpoint threshold.")
    parser.add_argument("--max-tiles", type=int, default=0, help="0 means all ROI tiles; use a small value for smoke runs.")
    parser.add_argument("--focus-quota", type=int, default=14)
    parser.add_argument("--row-bins", type=int, default=22)
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--quality", type=int, default=94)
    parser.add_argument("--contact-sheet-max", type=int, default=24)
    parser.add_argument("--row-step", type=int, default=16)
    parser.add_argument("--column-threshold", type=float, default=0.08)
    parser.add_argument("--min-run-pixels", type=int, default=2)
    parser.add_argument("--min-pair-gap", type=float, default=20.0)
    parser.add_argument("--max-pair-gap", type=float, default=130.0)
    parser.add_argument("--gap-tolerance", type=float, default=0.50)
    parser.add_argument("--target-gauge-m", type=float, default=1.50)
    parser.add_argument("--gauge-tolerance-m", type=float, default=0.15)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-candidates", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write and print the command plan without running it.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args, repo_root=repo_root, python_exe=Path(sys.executable))
    plan_path = out_dir / "fullpass_plan.json"
    write_json(plan_path, plan)

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if not args.skip_export:
        require_file(args.tile_index.expanduser().resolve(), "raw DOM ROI tile index")
    if not args.skip_predict:
        require_file(args.rail_model.expanduser().resolve(), "rail segmentation model")

    for step in plan["steps"]:
        if step["skipped"]:
            continue
        run_command(step["command"], cwd=repo_root)

    summary = {
        "out_dir": str(out_dir),
        "plan_path": str(plan_path),
        "dataset_dir": plan["paths"]["dataset_dir"],
        "images_dir": plan["paths"]["images_dir"],
        "rail_prediction_dir": plan["paths"]["rail_prediction_dir"],
        "candidate_dir": plan["paths"]["candidate_dir"],
        "dataset_summary": read_json_if_exists(Path(plan["paths"]["dataset_dir"]) / "summary.json"),
        "prediction_summary": read_json_if_exists(Path(plan["paths"]["rail_prediction_dir"]) / "summary.json"),
        "candidate_summary": read_json_if_exists(Path(plan["paths"]["candidate_dir"]) / "summary.json"),
    }
    summary_path = out_dir / "fullpass_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_plan(args: argparse.Namespace, *, repo_root: Path, python_exe: Path) -> dict[str, Any]:
    script_dir = repo_root / "scripts"
    out_dir = args.out_dir.expanduser().resolve()
    dataset_dir = out_dir / "raw_dom_roi_tiles"
    images_dir = dataset_dir / "images"
    rail_prediction_dir = out_dir / "rail_predictions"
    candidate_dir = out_dir / "rail_centerline_candidates"

    export_command = [
        str(python_exe),
        str(script_dir / "export_raw_dom_roi_sample_tiles.py"),
        "--tile-index",
        str(args.tile_index.expanduser().resolve()),
        "--out-dir",
        str(dataset_dir),
        "--max-tiles",
        str(args.max_tiles),
        "--focus-quota",
        str(args.focus_quota),
        "--row-bins",
        str(args.row_bins),
        "--format",
        args.image_format,
        "--quality",
        str(args.quality),
        "--contact-sheet-max",
        str(args.contact_sheet_max),
    ]
    predict_command = [
        str(python_exe),
        str(script_dir / "predict_rail_seg_images.py"),
        "--input-dir",
        str(images_dir),
        "--model",
        str(args.rail_model.expanduser().resolve()),
        "--out",
        str(rail_prediction_dir),
        "--device",
        str(args.device),
        "--threshold",
        str(args.threshold),
        "--contact-sheet-max",
        str(args.contact_sheet_max),
    ]
    candidate_command = [
        str(python_exe),
        str(script_dir / "extract_centerline_candidates.py"),
        "--dataset",
        str(dataset_dir),
        "--mask-dir",
        str(rail_prediction_dir / "masks"),
        "--out",
        str(candidate_dir),
        "--row-step",
        str(args.row_step),
        "--column-threshold",
        str(args.column_threshold),
        "--min-run-pixels",
        str(args.min_run_pixels),
        "--min-pair-gap",
        str(args.min_pair_gap),
        "--max-pair-gap",
        str(args.max_pair_gap),
        "--gap-tolerance",
        str(args.gap_tolerance),
        "--ignore-labels",
        "",
        "--target-gauge-m",
        str(args.target_gauge_m),
        "--gauge-tolerance-m",
        str(args.gauge_tolerance_m),
        "--contact-sheet-max",
        str(args.contact_sheet_max),
    ]

    return {
        "raw_dom_first": True,
        "max_tiles": args.max_tiles,
        "device": args.device,
        "paths": {
            "tile_index": str(args.tile_index.expanduser().resolve()),
            "rail_model": str(args.rail_model.expanduser().resolve()),
            "out_dir": str(out_dir),
            "dataset_dir": str(dataset_dir),
            "images_dir": str(images_dir),
            "rail_prediction_dir": str(rail_prediction_dir),
            "candidate_dir": str(candidate_dir),
        },
        "steps": [
            command_step("export_raw_dom_roi_tiles", export_command, args.skip_export),
            command_step("predict_rail_masks", predict_command, args.skip_predict),
            command_step("extract_map_candidates", candidate_command, args.skip_candidates),
        ],
    }


def command_step(name: str, command: list[str], skipped: bool) -> dict[str, Any]:
    return {
        "name": name,
        "skipped": bool(skipped),
        "command": command,
        "command_text": subprocess.list2cmdline(command),
    }


def run_command(command: list[str], *, cwd: Path) -> None:
    print(f"> {subprocess.list2cmdline(command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def require_file(path: Path, description: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def read_json_if_exists(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
