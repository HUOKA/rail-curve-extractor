from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_dom_to_3d_centerline_guided_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_dom_to_3d_centerline_guided_pipeline", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_profile_is_strict_auto_and_avoids_retained_review_inputs(tmp_path: Path) -> None:
    module = _load_module()
    args = module.build_parser().parse_args(["--out-dir", str(tmp_path)])
    repo_root = Path(__file__).resolve().parents[1]

    stages = module.build_stages(args, repo_root=repo_root, out_dir=tmp_path)
    plan = module.build_plan(args, repo_root=repo_root, out_dir=tmp_path, stages=stages)

    assert plan["profile"] == module.PROFILE_STRICT_AUTO
    assert [stage.name for stage in stages][:5] == [
        "build_auto_dom_tile_index",
        "export_dom_tiles",
        "predict_deeplab_segmentation",
        "extract_rail_candidates",
        "refine_centerline_candidates",
    ]
    assert "build_auto_turnout_crossover_evidence" in [stage.name for stage in stages]
    assert "build_semseg_radius_package" not in [stage.name for stage in stages]
    assert stages[-2].name == "polish_centerline_2d"
    assert stages[-1].name == "add_las_z"
    command_text = "\n".join(stage["command_text"] for stage in plan["stages"])
    assert "deeplab_topology_centerline_review_v15" not in command_text
    assert "deeplab_topology_centerline_review_v19" not in command_text
    assert "deeplab_topology_centerline_review_v20" not in command_text
    assert "data/manual_feedback" not in command_text
    normalized_command_text = command_text.replace("\\", "/")
    assert "all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson" in normalized_command_text
    assert "output/raw_dom_roi_fullpass_v1/all_turnout_branch_centerlines" not in normalized_command_text
    assert "build_ta08_semseg_radius_package.py" not in command_text
    assert "--allow-reviewed-bridges" not in command_text
    assert "--allow-specialized-turnout-rebuilds" not in command_text
    assert "--weak-gap-bridge-max-gap-m 60.0" in command_text
    assert "--turnout-exclusions" not in command_text
    assert args.use_turnout_exclusions is False


def test_dom_full_profile_is_compatibility_reproduction_path(tmp_path: Path) -> None:
    module = _load_module()
    args = module.build_parser().parse_args(["--profile", module.PROFILE_DOM_FULL, "--out-dir", str(tmp_path)])
    repo_root = Path(__file__).resolve().parents[1]

    stages = module.build_stages(args, repo_root=repo_root, out_dir=tmp_path)
    plan = module.build_plan(args, repo_root=repo_root, out_dir=tmp_path, stages=stages)
    command_text = "\n".join(stage["command_text"] for stage in plan["stages"])

    assert plan["profile"] == module.PROFILE_DOM_FULL
    assert "Compatibility pipeline" in plan["known_boundary"]
    assert "build_auto_dom_tile_index" not in [stage.name for stage in stages]
    assert "build_auto_turnout_crossover_evidence" not in [stage.name for stage in stages]
    assert "build_semseg_radius_package" in [stage.name for stage in stages]
    assert "output/raw_dom_roi_fullpass_v1/all_turnout_branch_centerlines" in command_text.replace("\\", "/")
    assert "--allow-reviewed-bridges" in command_text
    assert "--allow-specialized-turnout-rebuilds" in command_text


def test_accepted_baseline_profile_is_explicitly_reference_only(tmp_path: Path) -> None:
    module = _load_module()
    args = module.build_parser().parse_args(["--profile", module.PROFILE_ACCEPTED_BASELINE, "--out-dir", str(tmp_path)])
    repo_root = Path(__file__).resolve().parents[1]

    stages = module.build_stages(args, repo_root=repo_root, out_dir=tmp_path)
    plan = module.build_plan(args, repo_root=repo_root, out_dir=tmp_path, stages=stages)

    assert plan["profile"] == module.PROFILE_ACCEPTED_BASELINE
    assert [stage.name for stage in stages] == ["package_accepted_v19_2d", "add_las_z"]
    assert stages[-1].name == "add_las_z"
    assert "Reference-only" in plan["known_boundary"]


def test_compare_shape_info_reports_line_id_differences() -> None:
    module = _load_module()
    current = {
        "shape_type": "POLYLINEZ",
        "record_count": 3,
        "line_ids": ["A", "B", "B"],
    }
    baseline = {
        "shape_type": "POLYLINEZ",
        "record_count": 3,
        "line_ids": ["A", "B", "C"],
    }

    comparison = module.compare_shape_info(current, baseline)

    assert comparison["available"] is True
    assert comparison["record_count_match"] is True
    assert comparison["line_id_set_match"] is False
    assert comparison["missing_from_current"] == ["C"]


def test_start_at_keeps_late_stage_reruns_explicit(tmp_path: Path) -> None:
    module = _load_module()

    args = module.build_parser().parse_args(["--out-dir", str(tmp_path), "--start-at", "build_track_band_priors"])

    assert args.start_at == "build_track_band_priors"


def test_dry_run_writes_progress_file(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    progress_path = tmp_path / "progress.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_dom_to_3d_centerline_guided_pipeline.py",
            "--out-dir",
            str(tmp_path / "out"),
            "--progress-file",
            str(progress_path),
            "--dry-run",
        ],
    )

    assert module.main() == 0
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["state"] == "completed"
    assert progress["percent"] == 100.0
    assert progress["stage_count"] > 0
