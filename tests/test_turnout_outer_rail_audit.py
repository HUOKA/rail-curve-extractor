from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_turnout_outer_rail_generalization.py"
    spec = importlib.util.spec_from_file_location("audit_turnout_outer_rail_generalization", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_branch_passes_explicit_current_inputs(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    captured: dict[str, list[str]] = {}

    def fake_run(command, *, check, text, capture_output):
        captured["command"] = [str(item) for item in command]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    module.run_branch(
        "python",
        Path("scripts/prototype_turnout_outer_rail_centerline.py"),
        "AUTO_001",
        tmp_path / "AUTO_001",
        logs_dir,
        turnouts=Path("run/08_auto_turnout_crossover_evidence/all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson"),
        mainline=Path("run/06_mainline_prior/mainline_2_track_connected.geojson"),
        tile_index=Path("run/01_dom_tiles/selected_tile_index.csv"),
        probabilities_dir=Path("run/02_deeplab_segmentation/probabilities"),
        dom=Path("data/production/dom.tif"),
    )

    command = captured["command"]
    assert "--turnouts" in command
    assert "--mainline" in command
    assert "--tile-index" in command
    assert "--probabilities-dir" in command
    assert "--dom" in command
    normalized_command = [item.replace("\\", "/") for item in command]
    assert "run/02_deeplab_segmentation/probabilities" in normalized_command
