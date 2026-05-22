from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prototype_turnout_outer_rail_centerline.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("prototype_turnout_outer_rail_centerline", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _max_turn(coords: list[tuple[float, float]]) -> float:
    module = _load_module()
    return max(module.turn_angle_deg(a, b, c) for a, b, c in zip(coords, coords[1:], coords[2:]))


def test_local_turn_limiter_preserves_endpoints_and_reduces_isolated_kink() -> None:
    module = _load_module()
    coords = [
        (0.0, 0.0),
        (0.07, 0.61),
        (0.15, 1.18),
        (0.24, 1.73),
        (0.33, 2.25),
    ]

    smoothed = module.limit_local_turn_angles(coords, max_turn_deg=0.75, iterations=20, alpha=0.35)

    assert smoothed[0] == coords[0]
    assert smoothed[-1] == coords[-1]
    assert _max_turn(smoothed) <= 0.80
    assert _max_turn(smoothed) < _max_turn(coords)


def test_local_turn_limiter_leaves_gentle_line_unchanged() -> None:
    module = _load_module()
    coords = [(float(index), math.sin(index * 0.03)) for index in range(12)]

    smoothed = module.limit_local_turn_angles(coords, max_turn_deg=0.75, iterations=20, alpha=0.35)

    assert smoothed == coords
