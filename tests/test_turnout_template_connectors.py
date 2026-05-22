from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_turnout_template_connectors.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("build_turnout_template_connectors", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _template(module):
    return module.TemplateSpec(
        template_id="P003",
        length_m=70.0,
        samples=[
            module.TemplateSample(distance_from_main_m=0.0, offset_fraction=0.0),
            module.TemplateSample(distance_from_main_m=35.0, offset_fraction=0.45),
            module.TemplateSample(distance_from_main_m=70.0, offset_fraction=1.0),
        ],
        source_station_min_m=30.0,
        source_station_max_m=100.0,
        source_offset_start_m=-5.0,
        source_offset_end_m=0.0,
    )


def test_project_template_from_main_anchor_side_before_main() -> None:
    module = _load_module()
    points = module.project_template(_template(module), anchor_station=100.0, side_offset=-5.0, orientation=-1, endpoint_role="main")

    assert points[0] == (30.0, -5.0)
    assert points[-1] == (100.0, -0.0)


def test_project_template_from_side_anchor_side_after_main() -> None:
    module = _load_module()
    points = module.project_template(_template(module), anchor_station=170.0, side_offset=-5.0, orientation=1, endpoint_role="side")

    assert points[0] == (100.0, -0.0)
    assert points[-1] == (170.0, -5.0)


def test_endpoint_roles_classifies_main_and_side_offsets() -> None:
    module = _load_module()

    assert module.endpoint_roles(0.1, 5.0, 1.5, 2.0) == [("main", [-5.0, 5.0])]
    assert module.endpoint_roles(-4.9, 5.0, 1.5, 2.0) == [("side", [-5.0])]
    assert module.endpoint_roles(5.2, 5.0, 1.5, 2.0) == [("side", [5.0])]


def test_score_candidate_rewards_nearby_raw_support() -> None:
    module = _load_module()
    raw_index = {
        0: [
            module.RawPoint(station=0.0, offset=0.0, confidence=0.8),
            module.RawPoint(station=2.0, offset=0.1, confidence=0.8),
        ],
        1: [module.RawPoint(station=4.0, offset=0.1, confidence=0.8)],
    }

    score = module.score_candidate(
        [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)],
        raw_index=raw_index,
        support_distance_m=0.5,
        support_station_window_m=3.0,
    )

    assert score.support_coverage == 1.0
    assert score.score > 0.9


def test_score_candidate_rewards_transition_support_over_endpoint_only_support() -> None:
    module = _load_module()
    raw_index = {
        0: [
            module.RawPoint(station=0.0, offset=0.0, confidence=0.8),
            module.RawPoint(station=2.0, offset=0.0, confidence=0.8),
        ],
        2: [module.RawPoint(station=8.0, offset=5.0, confidence=0.8)],
        3: [module.RawPoint(station=10.0, offset=5.0, confidence=0.8)],
    }

    endpoint_only = module.score_candidate(
        [(0.0, 0.0), (2.0, 1.0), (4.0, 2.0), (6.0, 3.0), (8.0, 4.0), (10.0, 5.0)],
        raw_index=raw_index,
        support_distance_m=0.5,
        support_station_window_m=3.0,
    )
    transition_supported = module.score_candidate(
        [(0.0, 0.0), (2.0, 1.0), (4.0, 2.0), (6.0, 3.0), (8.0, 4.0), (10.0, 5.0)],
        raw_index={
            **raw_index,
            1: [module.RawPoint(station=4.0, offset=2.0, confidence=0.8)],
            2: [
                module.RawPoint(station=6.0, offset=3.0, confidence=0.8),
                module.RawPoint(station=8.0, offset=5.0, confidence=0.8),
            ],
        },
        support_distance_m=0.5,
        support_station_window_m=3.0,
    )

    assert endpoint_only.transition_coverage == 0.0
    assert transition_supported.transition_coverage == 1.0
    assert transition_supported.score > endpoint_only.score


def test_select_best_candidate_respects_branch_direction_feedback() -> None:
    module = _load_module()
    north = {"properties": {"template_score": 0.9, "branch_dir": "north", "shape_model": "p003_template"}}
    south = {"properties": {"template_score": 0.4, "branch_dir": "south", "shape_model": "p003_template"}}
    feedback = module.TurnoutFeedback(anchor_id="TA01", branch_direction="south", shape_model="p003_template", note="")

    selected = module.select_best_candidate([north, south], feedback)

    assert selected is south


def test_select_best_candidate_respects_shape_feedback() -> None:
    module = _load_module()
    template = {"properties": {"template_score": 0.9, "branch_dir": "south", "shape_model": "p003_template"}}
    piecewise = {"properties": {"template_score": 0.2, "branch_dir": "south", "shape_model": "curve_straight_reverse"}}
    feedback = module.TurnoutFeedback(anchor_id="TA08", branch_direction="south", shape_model="curve_straight_reverse", note="")

    selected = module.select_best_candidate([template, piecewise], feedback)

    assert selected is piecewise


def test_curve_straight_reverse_has_middle_straight_like_section() -> None:
    module = _load_module()

    points = module.curve_straight_reverse_points(
        anchor_station=100.0,
        side_offset=5.0,
        direction_sign=-1.0,
        total_length_m=72.0,
        sample_step_m=1.0,
    )

    assert points[0][0] == 28.0
    assert points[0][1] == 5.0
    assert points[-1][0] == 100.0
    assert points[-1][1] == 0.0
    assert module.branch_direction_for_points(points, anchor_station=100.0) == "south"
    offsets = [offset for _, offset in points]
    middle = offsets[len(offsets) // 2 - 3 : len(offsets) // 2 + 3]
    diffs = [round(a - b, 3) for a, b in zip(middle, middle[1:])]
    assert max(diffs) - min(diffs) < 0.02
