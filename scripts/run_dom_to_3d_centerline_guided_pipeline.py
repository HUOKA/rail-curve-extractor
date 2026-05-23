#!/usr/bin/env python3
"""Run DOM-to-2D/3D rail centerline workflows.

The default `strict-auto` profile rejects retained review evidence, user-picked
geometry, and named-turnout special cases. `dom-full` is kept as an explicit
compatibility profile for reproducing the latest accepted project output.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROFILE_STRICT_AUTO = "strict-auto"
PROFILE_DOM_FULL = "dom-full"
PROFILE_ACCEPTED_BASELINE = "accepted-baseline"

DEFAULT_DOM = Path("data") / "\u751f\u4ea7\u6570\u636e" / "\u65e0\u4eba\u673a\u6570\u636e" / "\u6b63\u5c04" / "dom.tif"
DEFAULT_DSM = Path("D:/") / "\u6b63\u5c04" / "lidars" / "terra_dsm" / "dsm.tif"
DEFAULT_LAS_DIR = Path("D:/") / "\u6b63\u5c04" / "lidars" / "terra_las"
DEFAULT_RAW_ROOT = Path("output/raw_dom_roi_fullpass_v1")
DEFAULT_STRICT_OUT_DIR = Path("output/dom_centerline_strict_auto_v1")
DEFAULT_OUT_DIR = Path("output/dom_centerline_pipeline_v1")
DEFAULT_ACCEPTED_OUT_DIR = Path("output/dom_to_3d_centerline_accepted_v1")
DEFAULT_TILE_INDEX = DEFAULT_RAW_ROOT / "raw_dom_roi_tiles" / "selected_tile_index.csv"
DEFAULT_EPSG = 32651

ARCHIVE_ROOT = Path("output/_archive_superseded_20260521_v20z_baseline/output_root")
DEFAULT_DEEPLAB_MODEL_CANDIDATES = [
    Path("models/rail_seg_deeplab_resnet50_native_v1/rail_semantic_deeplab_resnet50.pt"),
    Path("output/rail_seg_deeplab_resnet50_native_v1/rail_semantic_deeplab_resnet50.pt"),
    ARCHIVE_ROOT / "rail_seg_deeplab_resnet50_native_v1" / "rail_semantic_deeplab_resnet50.pt",
]

ACCEPTED_V19_DIR = Path("deeplab_topology_centerline_review_v19_crossover_tangent")
ACCEPTED_V20_Z_DIR = Path("deeplab_topology_centerline_review_v20_z")
NETWORK_STEM = "deeplab_topology_centerline_network"
NETWORK_Z_STEM = "deeplab_topology_centerline_network_z"


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    outputs: list[Path]
    command: list[str] = field(default_factory=list)
    action: str = "command"
    guided_dependency: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DOM -> 2D centerline SHP -> 3D centerline SHP.")
    parser.add_argument(
        "--profile",
        choices=[PROFILE_STRICT_AUTO, PROFILE_DOM_FULL, PROFILE_ACCEPTED_BASELINE],
        default=PROFILE_STRICT_AUTO,
        help="strict-auto is the clean production path. dom-full reproduces the accepted retained-evidence result. accepted-baseline is reference-only.",
    )
    parser.add_argument("--dom", type=Path, default=DEFAULT_DOM)
    parser.add_argument("--dsm", type=Path, default=DEFAULT_DSM)
    parser.add_argument("--las-dir", type=Path, default=DEFAULT_LAS_DIR)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--tile-index", type=Path, default=DEFAULT_TILE_INDEX)
    parser.add_argument("--deeplab-model", type=Path, default=default_deeplab_model())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--max-tiles", type=int, default=0, help="0 means all tiles from the tile index.")
    parser.add_argument("--force", action="store_true", help="Run every stage even if its declared outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Write and print the plan without running stages.")
    parser.add_argument("--progress-file", type=Path, default=None, help="Optional JSON file updated after each pipeline stage.")
    parser.add_argument("--stop-after", default="", help="Optional stage name; stop after this stage finishes.")
    parser.add_argument("--start-at", default="", help="Optional stage name; skip earlier stages and start here.")
    parser.add_argument("--skip-qa-crops", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-turnout-exclusions", action=argparse.BooleanOptionalAction, default=False)
    return parser


def default_deeplab_model() -> Path:
    for path in DEFAULT_DEEPLAB_MODEL_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_DEEPLAB_MODEL_CANDIDATES[0]


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = resolve_out_dir(args).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = args.progress_file.expanduser().resolve() if args.progress_file is not None else None

    stages = build_stages(args, repo_root=repo_root, out_dir=out_dir)
    plan = build_plan(args, repo_root=repo_root, out_dir=out_dir, stages=stages)
    write_json(out_dir / "pipeline_plan.json", plan)
    write_progress(
        progress_file,
        state="planned" if args.dry_run else "starting",
        message="Pipeline plan is ready.",
        out_dir=out_dir,
        stage_count=len(stages),
        stage_index=0,
        percent=0.0,
        plan=plan,
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        write_progress(
            progress_file,
            state="completed",
            message="Dry run plan written.",
            out_dir=out_dir,
            stage_count=len(stages),
            stage_index=len(stages),
            percent=100.0,
            plan=plan,
        )
        return 0

    events: list[dict[str, Any]] = []
    current_stage_name = ""
    try:
        preflight(args, repo_root=repo_root)
        started = not bool(args.start_at)
        for stage_index, stage in enumerate(stages, start=1):
            current_stage_name = stage.name
            if not started:
                if args.start_at == stage.name:
                    started = True
                else:
                    event = {
                        "stage": stage.name,
                        "description": stage.description,
                        "action": stage.action,
                        "command": subprocess.list2cmdline(stage.command) if stage.command else "",
                        "outputs": [str(path) for path in stage.outputs],
                        "status": "skipped_before_start",
                        "elapsed_s": 0.0,
                    }
                    events.append(event)
                    write_progress(
                        progress_file,
                        state="running",
                        message=f"Skipped before requested start stage: {stage.name}",
                        out_dir=out_dir,
                        stage_count=len(stages),
                        stage_index=stage_index,
                        stage_name=stage.name,
                        stage_description=stage.description,
                        stage_status=event["status"],
                        percent=stage_index / max(len(stages), 1) * 100.0,
                        latest_event=event,
                    )
                    continue
            write_progress(
                progress_file,
                state="running",
                message=stage.description,
                out_dir=out_dir,
                stage_count=len(stages),
                stage_index=stage_index,
                stage_name=stage.name,
                stage_description=stage.description,
                stage_status="running",
                percent=(stage_index - 1) / max(len(stages), 1) * 100.0,
                latest_event={
                    "stage": stage.name,
                    "description": stage.description,
                    "action": stage.action,
                    "command": subprocess.list2cmdline(stage.command) if stage.command else "",
                    "outputs": [str(path) for path in stage.outputs],
                    "status": "running",
                    "elapsed_s": 0.0,
                },
            )
            event = run_stage(stage, args=args, repo_root=repo_root, out_dir=out_dir)
            events.append(event)
            write_progress(
                progress_file,
                state="running",
                message=f"Stage finished: {stage.name}",
                out_dir=out_dir,
                stage_count=len(stages),
                stage_index=stage_index,
                stage_name=stage.name,
                stage_description=stage.description,
                stage_status=event["status"],
                percent=stage_index / max(len(stages), 1) * 100.0,
                latest_event=event,
            )
            if args.stop_after and args.stop_after == stage.name:
                break
        if args.start_at and not started:
            raise ValueError(f"Unknown --start-at stage: {args.start_at}")

        summary = build_summary(args, repo_root=repo_root, out_dir=out_dir, stages=stages, events=events)
        write_json(out_dir / "pipeline_summary.json", summary)
        copy_delivery_package(args.profile, out_dir)
        if args.profile == PROFILE_STRICT_AUTO:
            run_strict_auto_qa(args, repo_root=repo_root, out_dir=out_dir)
        write_progress(
            progress_file,
            state="completed",
            message="DOM to 3D centerline pipeline completed.",
            out_dir=out_dir,
            stage_count=len(stages),
            stage_index=len(stages),
            percent=100.0,
            outputs=summary.get("outputs", {}),
            summary_path=str(out_dir / "pipeline_summary.json"),
            latest_event=events[-1] if events else None,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        write_progress(
            progress_file,
            state="failed",
            message=str(exc),
            out_dir=out_dir,
            stage_count=len(stages),
            stage_index=len(events) + 1,
            stage_name=current_stage_name,
            percent=min(len(events) / max(len(stages), 1) * 100.0, 99.0),
            error=str(exc),
            latest_event=events[-1] if events else None,
        )
        raise


def resolve_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir.expanduser()
    if args.profile == PROFILE_ACCEPTED_BASELINE:
        return DEFAULT_ACCEPTED_OUT_DIR
    if args.profile == PROFILE_STRICT_AUTO:
        return DEFAULT_STRICT_OUT_DIR
    return DEFAULT_OUT_DIR


def build_stages(args: argparse.Namespace, *, repo_root: Path, out_dir: Path) -> list[Stage]:
    if args.profile == PROFILE_ACCEPTED_BASELINE:
        return build_accepted_baseline_stages(args, repo_root=repo_root, out_dir=out_dir)
    return build_dom_full_stages(args, repo_root=repo_root, out_dir=out_dir, strict_auto=args.profile == PROFILE_STRICT_AUTO)


def build_dom_full_stages(args: argparse.Namespace, *, repo_root: Path, out_dir: Path, strict_auto: bool = False) -> list[Stage]:
    script_dir = repo_root / "scripts"
    py = str(args.python.expanduser().resolve())
    raw_root = args.raw_root.expanduser()

    auto_tile_dir = out_dir / "00_auto_dom_tile_index"
    tiles_dir = out_dir / "01_dom_tiles"
    seg_dir = out_dir / "02_deeplab_segmentation"
    candidates_dir = out_dir / "03_rail_candidates"
    refined_dir = out_dir / "04_refined_centerline"
    deeplab_network_dir = out_dir / "05_deeplab_network"
    mainline_dir = out_dir / "06_mainline_prior"
    track_band_dir = out_dir / "07_track_band_priors"
    auto_evidence_dir = out_dir / "08_auto_turnout_crossover_evidence"
    topology_dir = out_dir / "08_topology_centerline"
    radius_dir = out_dir / "09_semseg_radius"
    centerline_2d_dir = out_dir / "10_centerline_2d"
    z_dir = out_dir / "11_centerline_3d"

    tile_index = auto_tile_dir / "raw_dom_tile_index.csv" if strict_auto else args.tile_index.expanduser()
    turnout_evidence = (
        auto_evidence_dir / "all_turnout_branch_centerlines" / "all_turnout_branch_centerlines.geojson"
        if strict_auto
        else raw_root / "all_turnout_branch_centerlines" / "all_turnout_branch_centerlines.geojson"
    )
    turnout_gauge_evidence = (
        auto_evidence_dir / "deeplab_gauge_pair_turnouts_v1" / "deeplab_gauge_pair_centerlines.geojson"
        if strict_auto
        else raw_root / "deeplab_gauge_pair_turnouts_v1" / "deeplab_gauge_pair_centerlines.geojson"
    )
    crossover_gauge_evidence = (
        auto_evidence_dir / "deeplab_gauge_pair_crossovers_v1" / "deeplab_gauge_pair_centerlines.geojson"
        if strict_auto
        else raw_root / "deeplab_gauge_pair_crossovers_v1" / "deeplab_gauge_pair_centerlines.geojson"
    )

    track_band_command = [
        py,
        str(script_dir / "build_track_band_priors.py"),
        "--candidates",
        str(candidates_dir / "track_centerline_candidates.geojson"),
        "--mainline",
        str(mainline_dir / "mainline_2_track_connected.geojson"),
        "--out-dir",
        str(track_band_dir),
        "--dom",
        str(args.dom.expanduser()),
        "--epsg",
        str(args.epsg),
    ]
    if not strict_auto:
        track_band_command.append("--allow-reviewed-bridges")
    if args.use_turnout_exclusions and not strict_auto:
        track_band_command.extend(["--turnout-exclusions", str(turnout_evidence)])
    if args.skip_qa_crops:
        track_band_command.append("--skip-qa-crops")

    stages: list[Stage] = []
    if strict_auto:
        stages.append(
            Stage(
                name="build_auto_dom_tile_index",
                description="Build a full-DOM tile index from raster metadata only.",
                guided_dependency="Strict-auto does not read retained ROI tile indexes or corridor geometry.",
                command=[
                    py,
                    str(script_dir / "build_auto_dom_tile_index.py"),
                    "--dom",
                    str(args.dom.expanduser()),
                    "--out-dir",
                    str(auto_tile_dir),
                ],
                outputs=[auto_tile_dir / "raw_dom_tile_index.csv", auto_tile_dir / "selected_tile_index.csv"],
            )
        )

    stages.extend(
        [
        Stage(
            name="export_dom_tiles",
            description="Export tiles directly from the supplied DOM.",
            guided_dependency=(
                "Strict-auto uses the tile index generated from the current DOM."
                if strict_auto
                else "Compatibility path uses the retained project ROI tile index."
            ),
            command=[
                py,
                str(script_dir / "export_raw_dom_roi_sample_tiles.py"),
                "--tile-index",
                str(tile_index),
                "--source-dom",
                str(args.dom.expanduser()),
                "--out-dir",
                str(tiles_dir),
                "--max-tiles",
                str(args.max_tiles),
                "--focus-quota",
                "14",
                "--row-bins",
                "22",
                "--format",
                "png",
                "--contact-sheet-max",
                "24",
            ],
            outputs=[tiles_dir / "selected_tile_index.csv", tiles_dir / "images"],
        ),
        Stage(
            name="predict_deeplab_segmentation",
            description="Run DeepLabV3+ style native-patch rail semantic segmentation on exported DOM tiles.",
            command=[
                py,
                str(script_dir / "predict_rail_seg_deeplab_images.py"),
                "--input-dir",
                str(tiles_dir / "images"),
                "--model",
                str(args.deeplab_model.expanduser()),
                "--out",
                str(seg_dir),
                "--device",
                str(args.device),
                "--threshold",
                str(args.threshold),
                "--contact-sheet-max",
                "24",
            ],
            outputs=[seg_dir / "summary.json", seg_dir / "masks", seg_dir / "probabilities"],
        ),
        Stage(
            name="extract_rail_candidates",
            description="Convert segmentation masks to map-space paired-rail centerline candidates.",
            command=[
                py,
                str(script_dir / "extract_centerline_candidates.py"),
                "--dataset",
                str(tiles_dir),
                "--mask-dir",
                str(seg_dir / "masks"),
                "--out",
                str(candidates_dir),
                "--row-step",
                "16",
                "--column-threshold",
                "0.08",
                "--min-run-pixels",
                "2",
                "--min-pair-gap",
                "20.0",
                "--max-pair-gap",
                "130.0",
                "--gap-tolerance",
                "0.50",
                "--ignore-labels",
                "",
                "--target-gauge-m",
                "1.50",
                "--gauge-tolerance-m",
                "0.15",
                "--contact-sheet-max",
                "24",
            ],
            outputs=[candidates_dir / "track_centerline_candidates.geojson"],
        ),
        Stage(
            name="refine_centerline_candidates",
            description="Refine per-tile candidates into continuous DeepLab support chains.",
            command=[
                py,
                str(script_dir / "refine_centerline_graph.py"),
                "--input",
                str(candidates_dir / "track_centerline_candidates.geojson"),
                "--out",
                str(refined_dir),
            ],
            outputs=[refined_dir / "refined_centerline_network.geojson", refined_dir / "main_centerline.geojson"],
        ),
        Stage(
            name="package_deeplab_network",
            description="Package refined DeepLab centerline chains for topology processing.",
            command=[
                py,
                str(script_dir / "package_deeplab_centerline_network_v1.py"),
                "--refined",
                str(refined_dir / "refined_centerline_network.geojson"),
                "--main",
                str(refined_dir / "main_centerline.geojson"),
                "--out-dir",
                str(deeplab_network_dir),
                "--epsg",
                str(args.epsg),
            ],
            outputs=[deeplab_network_dir / "deeplab_centerline_network_v1.geojson"],
        ),
        Stage(
            name="build_mainline_prior",
            description="Build the 2-track mainline prior automatically from DOM-derived DeepLab candidates.",
            guided_dependency="Default mode is semseg-auto; user-picked endpoints are only available via build_mainline_prior.py --mode manual.",
            command=[
                py,
                str(script_dir / "build_mainline_prior.py"),
                "--candidates",
                str(candidates_dir / "track_centerline_candidates.geojson"),
                "--out-dir",
                str(mainline_dir),
                "--epsg",
                str(args.epsg),
            ],
            outputs=[mainline_dir / "mainline_2_track_connected.geojson"],
        ),
        Stage(
            name="build_track_band_priors",
            description="Build main and parallel straight-track priors in the guided station/offset frame.",
            guided_dependency=(
                "Strict-auto uses support-bounded straight-track evidence without reviewed bridge zones."
                if strict_auto
                else "Compatibility path allows reviewed straight-gap bridge zones."
            ),
            command=track_band_command,
            outputs=[track_band_dir / "track_band_centerline_priors.geojson"],
        ),
    ]
    )
    if strict_auto:
        stages.append(
            Stage(
                name="build_auto_turnout_crossover_evidence",
                description="Build turnout/crossover transition evidence from current DeepLab candidates.",
                guided_dependency="Strict-auto transition evidence is derived from current semantic candidates only.",
                command=[
                    py,
                    str(script_dir / "build_auto_turnout_crossover_evidence.py"),
                    "--candidates",
                    str(candidates_dir / "track_centerline_candidates.geojson"),
                    "--mainline",
                    str(mainline_dir / "mainline_2_track_connected.geojson"),
                    "--track-bands",
                    str(track_band_dir / "track_band_centerline_priors.geojson"),
                    "--out-dir",
                    str(auto_evidence_dir),
                    "--epsg",
                    str(args.epsg),
                ],
                outputs=[
                    turnout_evidence,
                    turnout_gauge_evidence,
                    crossover_gauge_evidence,
                ],
            )
        )

    topology_command = [
        py,
        str(script_dir / "build_deeplab_topology_centerline_network.py"),
        "--deeplab-network",
        str(deeplab_network_dir / "deeplab_centerline_network_v1.geojson"),
        "--track-bands",
        str(track_band_dir / "track_band_centerline_priors.geojson"),
        "--turnouts",
        str(turnout_evidence),
        "--gauge-pair",
        str(turnout_gauge_evidence),
        "--out-dir",
        str(topology_dir),
        "--epsg",
        str(args.epsg),
    ]
    if strict_auto:
        topology_command.extend(["--weak-gap-bridge-max-gap-m", "60.0"])
    else:
        topology_command.append("--allow-specialized-turnout-rebuilds")

    stages.append(
        Stage(
            name="build_topology_centerline",
            description="Build topology-aware 2D centerline from DeepLab support, track bands, and transition evidence.",
            guided_dependency=(
                "Strict-auto uses automatically generated transition evidence and disables named-turnout rebuilds."
                if strict_auto
                else "Compatibility path uses retained turnout/crossover evidence and legacy named-turnout rebuilds."
            ),
            command=topology_command,
            outputs=[topology_dir / f"{NETWORK_STEM}.geojson"],
        )
    )

    polish_base_dir = topology_dir
    if not strict_auto:
        stages.append(
            Stage(
            name="build_semseg_radius_package",
            description="Compatibility stage: add TA08 semantic-segmentation curvature/radius diagnostics.",
            guided_dependency="Compatibility/debug only; strict-auto skips TA-specific diagnostic packages.",
            command=[
                py,
                str(script_dir / "build_ta08_semseg_radius_package.py"),
                "--base-network",
                str(topology_dir / f"{NETWORK_STEM}.geojson"),
                "--base-evidence",
                str(topology_dir / "deeplab_topology_evidence.geojson"),
                "--out-dir",
                str(radius_dir),
                "--epsg",
                str(args.epsg),
            ],
            outputs=[radius_dir / f"{NETWORK_STEM}.geojson"],
            )
        )
        polish_base_dir = radius_dir

    stages.extend(
        [
        Stage(
            name="polish_centerline_2d",
            description="Apply guarded smoothing, trimming, crossover tangency, and final 2D Shapefile export.",
            command=[
                py,
                str(script_dir / "build_semseg_smooth_review_package.py"),
                "--base-network",
                str(polish_base_dir / f"{NETWORK_STEM}.geojson"),
                "--base-evidence",
                str(polish_base_dir / "deeplab_topology_evidence.geojson"),
                "--crossover-evidence",
                str(crossover_gauge_evidence),
                "--out-dir",
                str(centerline_2d_dir),
                "--epsg",
                str(args.epsg),
            ],
            outputs=[centerline_2d_dir / f"{NETWORK_STEM}.shp", centerline_2d_dir / f"{NETWORK_STEM}.geojson"],
        ),
        Stage(
            name="add_las_z",
            description="Add smoothed LAS/DSM-derived Z to the 2D centerline and write PolyLineZ output.",
            command=[
                py,
                str(script_dir / "add_z_to_deeplab_topology_centerline.py"),
                "--input",
                str(centerline_2d_dir / f"{NETWORK_STEM}.geojson"),
                "--output-dir",
                str(z_dir),
                "--dsm",
                str(args.dsm.expanduser()),
                "--las-dir",
                str(args.las_dir.expanduser()),
                "--epsg",
                str(args.epsg),
            ],
            outputs=[z_dir / f"{NETWORK_Z_STEM}.shp"],
        ),
        ]
    )
    return stages


def build_accepted_baseline_stages(args: argparse.Namespace, *, repo_root: Path, out_dir: Path) -> list[Stage]:
    script_dir = repo_root / "scripts"
    py = str(args.python.expanduser().resolve())
    centerline_2d_dir = out_dir / "01_accepted_v19_centerline_2d"
    z_dir = out_dir / "02_centerline_3d"
    return [
        Stage(
            name="package_accepted_v19_2d",
            description="Reference wrapper only: copy the accepted v19 2D centerline.",
            action="copy_accepted_v19_2d",
            guided_dependency="This profile intentionally depends on old review output and is not the production path.",
            outputs=[centerline_2d_dir / f"{NETWORK_STEM}.geojson", centerline_2d_dir / f"{NETWORK_STEM}.shp"],
        ),
        Stage(
            name="add_las_z",
            description="Rebuild Z for the copied reference 2D centerline.",
            command=[
                py,
                str(script_dir / "add_z_to_deeplab_topology_centerline.py"),
                "--input",
                str(centerline_2d_dir / f"{NETWORK_STEM}.geojson"),
                "--output-dir",
                str(z_dir),
                "--dsm",
                str(args.dsm.expanduser()),
                "--las-dir",
                str(args.las_dir.expanduser()),
                "--epsg",
                str(args.epsg),
            ],
            outputs=[z_dir / f"{NETWORK_Z_STEM}.shp"],
        ),
    ]


def run_stage(stage: Stage, *, args: argparse.Namespace, repo_root: Path, out_dir: Path) -> dict[str, Any]:
    should_run = args.force or not all(path.exists() for path in stage.outputs)
    event: dict[str, Any] = {
        "stage": stage.name,
        "description": stage.description,
        "action": stage.action,
        "command": subprocess.list2cmdline(stage.command) if stage.command else "",
        "outputs": [str(path) for path in stage.outputs],
        "status": "skipped_existing",
        "elapsed_s": 0.0,
    }
    if not should_run:
        return event
    start = time.time()
    if stage.action == "command":
        print(f"> {subprocess.list2cmdline(stage.command)}", flush=True)
        subprocess.run(stage.command, cwd=str(repo_root), check=True)
    elif stage.action == "copy_accepted_v19_2d":
        copy_accepted_v19_2d(args, repo_root=repo_root, out_dir=out_dir)
    else:
        raise ValueError(f"Unknown stage action: {stage.action}")
    event["status"] = "completed"
    event["elapsed_s"] = round(time.time() - start, 2)
    return event


def run_strict_auto_qa(args: argparse.Namespace, *, repo_root: Path, out_dir: Path) -> None:
    subprocess.run(
        [
            str(args.python.expanduser().resolve()),
            str(repo_root / "scripts" / "qa_strict_auto_centerline_delivery.py"),
            "--out-dir",
            str(out_dir),
            "--epsg",
            str(args.epsg),
        ],
        cwd=str(repo_root),
        check=True,
    )


def copy_accepted_v19_2d(args: argparse.Namespace, *, repo_root: Path, out_dir: Path) -> None:
    source_dir = resolve_repo_path(repo_root, args.raw_root.expanduser() / ACCEPTED_V19_DIR)
    target_dir = out_dir / "01_accepted_v19_centerline_2d"
    target_dir.mkdir(parents=True, exist_ok=True)
    copy_stemmed_files(source_dir, target_dir, NETWORK_STEM, (".geojson", ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qml"))


def copy_stemmed_files(source_dir: Path, target_dir: Path, stem: str, suffixes: tuple[str, ...]) -> None:
    missing_required: list[Path] = []
    for suffix in suffixes:
        source = source_dir / f"{stem}{suffix}"
        if source.exists():
            shutil.copy2(source, target_dir / source.name)
        elif suffix in {".geojson", ".shp", ".shx", ".dbf", ".prj"}:
            missing_required.append(source)
    if missing_required:
        missing = "\n".join(str(path) for path in missing_required)
        raise FileNotFoundError(f"Accepted v19 package is incomplete:\n{missing}")


def build_plan(args: argparse.Namespace, *, repo_root: Path, out_dir: Path, stages: list[Stage]) -> dict[str, Any]:
    plan_tile_index = out_dir / "00_auto_dom_tile_index" / "raw_dom_tile_index.csv" if args.profile == PROFILE_STRICT_AUTO else args.tile_index.expanduser()
    return {
        "mode": "dom_to_centerline_2d_3d_pipeline_v1",
        "profile": args.profile,
        "repo_root": str(repo_root),
        "dom": str(args.dom.expanduser()),
        "dsm": str(args.dsm.expanduser()),
        "las_dir": str(args.las_dir.expanduser()),
        "tile_index": str(plan_tile_index),
        "deeplab_model": str(args.deeplab_model.expanduser()),
        "raw_root": None if args.profile == PROFILE_STRICT_AUTO else str(args.raw_root.expanduser()),
        "out_dir": str(out_dir),
        "epsg": args.epsg,
        "force": bool(args.force),
        "known_boundary": profile_boundary(args.profile),
        "forbidden_as_default_inputs": [
            "deeplab_topology_centerline_review_v15*",
            "deeplab_topology_centerline_review_v19*",
            "deeplab_topology_centerline_review_v20*",
            "data/manual_feedback/*",
            "output/raw_dom_roi_fullpass_v1/all_turnout_branch_centerlines/*",
            "output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_turnouts_v1/*",
            "output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_crossovers_v1/*",
            "scripts/build_ta08_*",
        ],
        "stages": [
            {
                "name": stage.name,
                "description": stage.description,
                "guided_dependency": stage.guided_dependency,
                "action": stage.action,
                "command": stage.command,
                "command_text": subprocess.list2cmdline(stage.command) if stage.command else "",
                "outputs": [str(path) for path in stage.outputs],
            }
            for stage in stages
        ],
    }


def profile_boundary(profile: str) -> str:
    if profile == PROFILE_ACCEPTED_BASELINE:
        return "Reference-only profile. It depends on old accepted v19/v20 artifacts and is not the production DOM pipeline."
    if profile == PROFILE_STRICT_AUTO:
        return (
            "Strict-auto pipeline: DOM-derived tile index -> DeepLab segmentation -> automatic transition evidence -> "
            "2D topology centerline -> LAS/DSM Z. It must not read retained ROI indexes, manual feedback, "
            "review-version outputs, retained turnout evidence, or named-turnout special rebuilds."
        )
    return (
        "Compatibility pipeline: DOM tiles -> DeepLab segmentation -> 2D topology centerline -> LAS/DSM Z. "
        "It reproduces the accepted project result and still uses a retained ROI tile index, retained turnout evidence, "
        "reviewed straight-gap bridges, and legacy named-turnout rebuilds."
    )


def preflight(args: argparse.Namespace, *, repo_root: Path) -> None:
    required = [
        (args.dom.expanduser(), "DOM"),
        (args.dsm.expanduser(), "DSM"),
        (args.las_dir.expanduser(), "LAS directory"),
    ]
    if args.profile == PROFILE_STRICT_AUTO:
        required.append((args.deeplab_model.expanduser(), "DeepLab model"))
    elif args.profile == PROFILE_DOM_FULL:
        raw_root = args.raw_root.expanduser()
        required.extend(
            [
                (args.tile_index.expanduser(), "DOM ROI tile index"),
                (args.deeplab_model.expanduser(), "DeepLab model"),
                (raw_root / "all_turnout_branch_centerlines" / "all_turnout_branch_centerlines.geojson", "retained turnout branch evidence"),
                (raw_root / "deeplab_gauge_pair_turnouts_v1" / "deeplab_gauge_pair_centerlines.geojson", "retained turnout gauge-pair evidence"),
                (raw_root / "deeplab_gauge_pair_crossovers_v1" / "deeplab_gauge_pair_centerlines.geojson", "retained crossover gauge-pair evidence"),
            ]
        )
    else:
        raw_root = args.raw_root.expanduser()
        required.append((raw_root / ACCEPTED_V19_DIR / f"{NETWORK_STEM}.geojson", "accepted v19 2D reference"))
    for path, label in required:
        full = resolve_repo_path(repo_root, path)
        if not full.exists():
            raise FileNotFoundError(f"Missing {label}: {full}")


def build_summary(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    out_dir: Path,
    stages: list[Stage],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    centerline_2d_dir = final_2d_dir(args.profile, out_dir)
    centerline_3d_dir = final_3d_dir(args.profile, out_dir)
    shp_2d = centerline_2d_dir / f"{NETWORK_STEM}.shp"
    shp_3d = centerline_3d_dir / f"{NETWORK_Z_STEM}.shp"
    shape_2d = inspect_shapefile(shp_2d) if shp_2d.exists() else None
    shape_3d = inspect_shapefile(shp_3d) if shp_3d.exists() else None
    baseline_info = None if args.profile == PROFILE_STRICT_AUTO else accepted_v20_info(args, repo_root=repo_root)
    summary_tile_index = out_dir / "00_auto_dom_tile_index" / "raw_dom_tile_index.csv" if args.profile == PROFILE_STRICT_AUTO else args.tile_index.expanduser()
    return {
        "mode": "dom_to_centerline_2d_3d_pipeline_v1",
        "profile": args.profile,
        "status": "completed" if shp_2d.exists() and shp_3d.exists() else "incomplete",
        "repo_root": str(repo_root),
        "known_boundary": profile_boundary(args.profile),
        "known_guided_dependencies": guided_dependencies(args.profile),
        "inputs": {
            "dom": str(args.dom.expanduser()),
            "dsm": str(args.dsm.expanduser()),
            "las_dir": str(args.las_dir.expanduser()),
            "tile_index": str(summary_tile_index),
            "deeplab_model": str(args.deeplab_model.expanduser()),
            "raw_root": None if args.profile == PROFILE_STRICT_AUTO else str(args.raw_root.expanduser()),
        },
        "stages": [
            {
                "name": stage.name,
                "description": stage.description,
                "guided_dependency": stage.guided_dependency,
                "action": stage.action,
                "outputs": [str(path) for path in stage.outputs],
            }
            for stage in stages
        ],
        "events": events,
        "outputs": {
            "centerline_2d_shp": str(shp_2d),
            "centerline_2d_geojson": str(centerline_2d_dir / f"{NETWORK_STEM}.geojson"),
            "centerline_3d_shp": str(shp_3d),
            "centerline_3d_geojson": str(centerline_3d_dir / f"{NETWORK_Z_STEM}.geojson"),
            "pipeline_summary": str(out_dir / "pipeline_summary.json"),
            "pipeline_plan": str(out_dir / "pipeline_plan.json"),
            "final_delivery_dir": str(out_dir / "final_delivery"),
        },
        "shape_2d": shape_2d,
        "shape_3d": shape_3d,
        "reference_z_baseline": baseline_info,
        "baseline_comparison": compare_shape_info(shape_3d, baseline_info),
        "z_summary": read_json_if_exists(centerline_3d_dir / "summary.json"),
    }


def guided_dependencies(profile: str) -> list[str]:
    if profile == PROFILE_ACCEPTED_BASELINE:
        return ["This reference profile copies v19 2D and is intentionally not the production pipeline."]
    if profile == PROFILE_STRICT_AUTO:
        return [
            "Tile index is generated from the current DOM by build_auto_dom_tile_index.py.",
            "build_mainline_prior.py defaults to semseg-auto; manual mainline endpoints are not used by this pipeline.",
            "Turnout and crossover evidence is generated from current DeepLab candidates by build_auto_turnout_crossover_evidence.py.",
            "Named-turnout special rebuilds and reviewed straight-gap bridge zones are disabled.",
        ]
    return [
        "ROI tile index is retained for accepted-output reproduction.",
        "build_mainline_prior.py defaults to semseg-auto; manual mainline endpoints are retained only as an explicit debug mode.",
        "Turnout and crossover evidence layers are retained compatibility artifacts, not v15/v19/v20 centerline outputs.",
        "Reviewed straight-gap bridges and legacy named-turnout rebuilds are allowed only in this compatibility profile.",
    ]


def final_2d_dir(profile: str, out_dir: Path) -> Path:
    if profile == PROFILE_ACCEPTED_BASELINE:
        return out_dir / "01_accepted_v19_centerline_2d"
    return out_dir / "10_centerline_2d"


def final_3d_dir(profile: str, out_dir: Path) -> Path:
    if profile == PROFILE_ACCEPTED_BASELINE:
        return out_dir / "02_centerline_3d"
    return out_dir / "11_centerline_3d"


def accepted_v20_info(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any] | None:
    baseline_shp = resolve_repo_path(repo_root, args.raw_root.expanduser() / ACCEPTED_V20_Z_DIR / f"{NETWORK_Z_STEM}.shp")
    if not baseline_shp.exists():
        return None
    return inspect_shapefile(baseline_shp)


def compare_shape_info(current: dict[str, Any] | None, baseline: dict[str, Any] | None) -> dict[str, Any]:
    if current is None or baseline is None:
        return {"available": False}
    current_ids = current.get("line_ids") or []
    baseline_ids = baseline.get("line_ids") or []
    return {
        "available": True,
        "record_count_match": current.get("record_count") == baseline.get("record_count"),
        "shape_type_match": current.get("shape_type") == baseline.get("shape_type"),
        "line_id_set_match": set(current_ids) == set(baseline_ids),
        "missing_from_current": sorted(set(baseline_ids) - set(current_ids)),
        "extra_in_current": sorted(set(current_ids) - set(baseline_ids)),
    }


def inspect_shapefile(path: Path) -> dict[str, Any]:
    import shapefile

    reader = shapefile.Reader(str(path))
    fields = [field[0] for field in reader.fields[1:]]
    rows = [dict(zip(fields, list(record))) for record in reader.records()]
    line_key = "line_id" if "line_id" in fields else ""
    role_key = "net_role" if "net_role" in fields else ("network_role" if "network_role" in fields else "role")
    line_ids = [str(row.get(line_key, "")) for row in rows] if line_key else []
    roles = [str(row.get(role_key, "")) for row in rows]
    return {
        "path": str(path),
        "shape_type": reader.shapeTypeName,
        "record_count": len(reader),
        "fields": fields,
        "line_ids": line_ids,
        "duplicate_line_ids": sorted([line_id for line_id, count in Counter(line_ids).items() if line_id and count > 1]),
        "role_counts": dict(sorted(Counter(roles).items())),
    }


def copy_delivery_package(profile: str, out_dir: Path) -> None:
    delivery_dir = out_dir / "final_delivery"
    delivery_dir.mkdir(parents=True, exist_ok=True)
    copy_stemmed_if_exists(final_2d_dir(profile, out_dir), delivery_dir, NETWORK_STEM, "centerline_2d")
    copy_stemmed_if_exists(final_3d_dir(profile, out_dir), delivery_dir, NETWORK_Z_STEM, "centerline_3d")


def copy_stemmed_if_exists(source_dir: Path, delivery_dir: Path, stem: str, delivery_stem: str) -> None:
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qml", ".geojson"):
        source = source_dir / f"{stem}{suffix}"
        if source.exists():
            shutil.copy2(source, delivery_dir / f"{delivery_stem}{suffix}")


def resolve_repo_path(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (repo_root / expanded).resolve()


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_progress(path: Path | None, **payload: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "updated_at": time.time(),
        **payload,
    }
    temporary_path = path.with_name(f"{path.name}.{time.time_ns()}.tmp")
    temporary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
