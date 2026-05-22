from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_turnout_crossover_connectors.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("build_turnout_crossover_connectors", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pair(module, *, start_offset: float = -5.0, end_offset: float = 0.0):
    return module.CrossoverPair(
        crossover_id="CX_TEST",
        south_anchor_id="TA_S",
        north_anchor_id="TA_N",
        south_role="parallel_minus_5m",
        north_role="mainline_2_track",
        south_point=(0.0, start_offset),
        north_point=(100.0, end_offset),
        south_station=0.0,
        south_offset=start_offset,
        north_station=100.0,
        north_offset=end_offset,
        note="",
    )


def test_select_transition_evidence_keeps_only_between_track_slope_direction() -> None:
    module = _load_module()
    pair = _pair(module, start_offset=-5.0, end_offset=0.0)
    good = module.RawEvidence(
        evidence_id="good",
        points=[
            module.RawPoint(30.0, -3.6, 0.7),
            module.RawPoint(45.0, -2.4, 0.7),
            module.RawPoint(60.0, -1.4, 0.7),
        ],
        station_min=30.0,
        station_max=60.0,
        offset_min=-3.6,
        offset_max=-1.4,
        station_span=30.0,
        offset_span=2.2,
        slope_dt_ds=0.073,
        mean_confidence=0.7,
    )
    parallel = module.RawEvidence(
        evidence_id="parallel",
        points=[module.RawPoint(20.0, -5.0, 0.8), module.RawPoint(80.0, -5.0, 0.8)],
        station_min=20.0,
        station_max=80.0,
        offset_min=-5.0,
        offset_max=-5.0,
        station_span=60.0,
        offset_span=0.0,
        slope_dt_ds=0.0,
        mean_confidence=0.8,
    )
    wrong_side = module.RawEvidence(
        evidence_id="wrong_side",
        points=[module.RawPoint(30.0, 1.2, 0.7), module.RawPoint(60.0, 3.0, 0.7)],
        station_min=30.0,
        station_max=60.0,
        offset_min=1.2,
        offset_max=3.0,
        station_span=30.0,
        offset_span=1.8,
        slope_dt_ds=0.06,
        mean_confidence=0.7,
    )

    selected = module.select_transition_evidence(
        pair,
        raw_evidence=[parallel, wrong_side, good],
        station_pad_m=5.0,
        offset_pad_m=0.75,
        min_station_span_m=4.0,
        min_offset_span_m=0.35,
        min_abs_slope=0.025,
        max_abs_slope=0.25,
    )

    assert [item.evidence_id for item in selected] == ["good"]


def test_fit_crossover_points_keeps_fixed_endpoints_and_monotonic_offsets() -> None:
    module = _load_module()
    pair = _pair(module, start_offset=-5.0, end_offset=0.0)
    transition_points = [
        module.RawPoint(40.0, -3.0, 0.9),
        module.RawPoint(50.0, -2.5, 0.9),
        module.RawPoint(60.0, -2.0, 0.9),
    ]

    points = module.fit_crossover_points(
        pair,
        transition_points=transition_points,
        sample_step_m=10.0,
        evidence_window_m=6.0,
        evidence_corridor_m=1.0,
    )

    assert points[0] == (0.0, -5.0)
    assert points[-1] == (100.0, 0.0)
    offsets = [offset for _, offset in points]
    assert all(a <= b + 0.03 for a, b in zip(offsets, offsets[1:]))
    middle = min(points, key=lambda item: abs(item[0] - 50.0))
    assert abs(middle[1] - -2.5) < 0.35


def test_curve_straight_curve_baseline_has_tangent_like_ends_and_linear_middle() -> None:
    module = _load_module()

    values = [module.curve_straight_curve_baseline(-5.0, 0.0, u) for u in (0.0, 0.1, 0.5, 0.9, 1.0)]

    assert values[0] == -5.0
    assert values[-1] == 0.0
    assert abs(values[2] - -2.5) < 1e-6
    assert abs(values[1] - values[0]) < abs(values[2] - values[1])
    assert abs(values[-1] - values[-2]) < abs(values[-2] - values[2])
