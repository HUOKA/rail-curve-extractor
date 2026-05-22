from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_deeplab_turnout_gauge_pair_centerlines.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("build_deeplab_turnout_gauge_pair_centerlines", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _line(coords, **props):
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in coords]},
    }


def test_branch_id_for_feature_prefers_anchor_id_and_normalizes_connector_id() -> None:
    module = _load_module()

    assert module.branch_id_for_feature(_line([(0, 0), (1, 1)], anchor_id="TA08", connector_id="TA08_BEST")) == "TA08"
    assert module.branch_id_for_feature(_line([(0, 0), (1, 1)], connector_id="TA09_BEST")) == "TA09"


def test_extraction_args_uses_feature_offset_window_not_ta08_range() -> None:
    module = _load_module()
    args = module.build_parser().parse_args([])
    args.offset_padding_m = 2.0
    args.min_offset_span_m = 7.0
    guide = module.gauge_pair.Guide.from_coords([(0.0, 0.0), (100.0, 0.0)])
    feature = _line([(10.0, -5.0), (40.0, 0.0)], anchor_id="TA01")

    extraction_args = module.extraction_args_for_feature(args, feature, guide=guide)

    assert extraction_args.offset_min_m == -7.0
    assert extraction_args.offset_max_m == 2.0


def test_tag_branch_features_keeps_sequence_ids_unique_per_turnout_window() -> None:
    module = _load_module()
    features = [
        _line([(0, 0), (1, 1)], seq_id="GP01", source="old"),
        _line([(2, 2), (3, 3)], seq_id="GP02", source="old"),
    ]

    module.tag_branch_features(features, branch_id="TA08")

    assert [feature["properties"]["seq_id"] for feature in features] == ["TA08_GP01", "TA08_GP02"]
    assert all(feature["properties"]["branch_id"] == "TA08" for feature in features)
    assert all(feature["properties"]["source"] == "deeplab_turnout_window_gauge_pair" for feature in features)
