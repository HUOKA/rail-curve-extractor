from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_all_turnout_branch_centerlines.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("build_all_turnout_branch_centerlines", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _line(connector_id: str, **props):
    properties = {"connector_id": connector_id, **props}
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]},
    }


def _line_with_coords(coords, connector_id: str, **props):
    feature = _line(connector_id, **props)
    feature["geometry"]["coordinates"] = [[float(x), float(y)] for x, y in coords]
    return feature


def test_build_all_turnout_features_replaces_paired_anchors_with_crossovers() -> None:
    module = _load_module()
    templates = [
        _line("TA01_BEST", anchor_id="TA01"),
        _line("TA02_BEST", anchor_id="TA02"),
        _line("TA03_BEST", anchor_id="TA03"),
        _line("TA04_BEST", anchor_id="TA04"),
        _line("TA05_BEST", anchor_id="TA05"),
        _line("TA06_BEST", anchor_id="TA06"),
        _line("TA09_BEST", anchor_id="TA09"),
    ]
    crossovers = [
        _line("CX01", south_anchor="TA02", north_anchor="TA01"),
        _line("CX02", south_anchor="TA05", north_anchor="TA04"),
    ]

    selected = module.build_all_turnout_features(templates, crossovers)
    branch_ids = [feature["properties"]["branch_id"] for feature in selected]

    assert branch_ids == ["CX01", "CX02", "TA03", "TA06", "TA09"]
    assert "TA01" not in branch_ids
    assert "TA04" not in branch_ids


def test_default_gauge_pair_input_is_all_turnout_evidence() -> None:
    module = _load_module()

    assert "deeplab_gauge_pair_turnouts_v1" in str(module.DEFAULT_GAUGE_PAIR)
    assert "ta08" not in str(module.DEFAULT_GAUGE_PAIR).lower()


def test_normalize_feature_marks_crossover_as_self_reviewed() -> None:
    module = _load_module()
    feature = module.normalize_feature(_line("CX01", south_anchor="TA02", north_anchor="TA01"), source_type="crossover_las_endpoint_locked")

    props = feature["properties"]
    assert props["branch_id"] == "CX01"
    assert props["anchors"] == "TA02,TA01"
    assert props["review_status"] == "preferred_candidate_self_reviewed"
    assert props["qa_status"] == "self_review_pass_visual"


def test_low_support_template_uses_nearby_gauge_pair_evidence() -> None:
    module = _load_module()
    guide = module.btc.Guide((0.0, 0.0), (100.0, 0.0))
    template = _line_with_coords(
        [(10.0, 5.0), (40.0, 3.0), (80.0, 0.0)],
        "TA08_BEST",
        anchor_id="TA08",
        support_cov=0.1,
        trans_cov=0.2,
    )
    gauge_pair = _line_with_coords(
        [(8.0, 7.0), (30.0, 5.0), (45.0, 2.5)],
        "GP02",
        seq_id="GP02",
        role="deeplab_gauge_pair_centerline",
    )

    selected = module.build_all_turnout_features([template], [], gauge_pair_features=[gauge_pair], guide=guide)

    assert len(selected) == 1
    props = selected[0]["properties"]
    coords = selected[0]["geometry"]["coordinates"]
    assert props["source_type"] == "gauge_pair_evidence_constrained"
    assert props["gauge_seq"] == "GP02"
    assert props["qa_status"] == "self_review_needs_visual_check"
    assert props["completion_mode"] == "evidence_limited_tangent_endpoint"
    assert coords[0] == [8.0, 7.0]
    assert coords[-1][0] == 80.0
    assert coords[-1][1] == 0.0
    mid = min(coords, key=lambda point: abs(point[0] - 52.0))
    assert mid[1] < 1.5


def test_gauge_pair_refinement_uses_template_endpoint_when_offset_shift_is_small() -> None:
    module = _load_module()
    guide = module.btc.Guide((0.0, 0.0), (100.0, 0.0))
    template = _line_with_coords(
        [(10.0, 5.0), (40.0, 3.0), (80.0, 2.0)],
        "TA08_BEST",
        anchor_id="TA08",
        support_cov=0.1,
        trans_cov=0.2,
    )
    gauge_pair = _line_with_coords(
        [(8.0, 7.0), (30.0, 5.0), (45.0, 2.5)],
        "GP02",
        seq_id="GP02",
        role="deeplab_gauge_pair_centerline",
    )

    selected = module.build_all_turnout_features([template], [], gauge_pair_features=[gauge_pair], guide=guide)

    props = selected[0]["properties"]
    coords = selected[0]["geometry"]["coordinates"]
    assert props["completion_mode"] == "template_endpoint"
    assert coords[-1][0] == 80.0
    assert coords[-1][1] == 2.0


def test_station_offset_smoothing_limits_suffix_kink() -> None:
    module = _load_module()
    points = [(0.0, 5.0), (1.0, 4.0), (2.0, 3.0), (3.0, 2.0), (4.0, 1.9), (5.0, 0.2), (6.0, 0.0)]

    smoothed = module.smooth_station_offset_curve(points, step_m=1.0, window_size=5, passes=4)
    slopes = [
        abs((b[1] - a[1]) / (b[0] - a[0]))
        for a, b in zip(smoothed, smoothed[1:])
        if abs(b[0] - a[0]) > 1e-9
    ]

    assert smoothed[0] == points[0]
    assert smoothed[-1] == points[-1]
    assert max(slopes) < 1.6


def test_gauge_pair_refinement_ignores_mainline_like_evidence() -> None:
    module = _load_module()
    guide = module.btc.Guide((0.0, 0.0), (100.0, 0.0))
    template = _line_with_coords(
        [(10.0, 5.0), (40.0, 3.0), (80.0, 0.0)],
        "TA08_BEST",
        anchor_id="TA08",
        support_cov=0.1,
        trans_cov=0.2,
    )
    mainline_like = _line_with_coords(
        [(8.0, 0.1), (45.0, 0.0)],
        "GP01",
        seq_id="GP01",
        role="deeplab_gauge_pair_centerline",
    )

    selected = module.build_all_turnout_features([template], [], gauge_pair_features=[mainline_like], guide=guide)

    assert selected[0]["properties"]["source_type"] == "remaining_template_or_special"
