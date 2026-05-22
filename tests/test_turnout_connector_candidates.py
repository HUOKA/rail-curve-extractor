from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_turnout_connector_candidates.py"
    spec = importlib.util.spec_from_file_location("build_turnout_connector_candidates", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _feature(coords, *, candidate_id: int = 1, confidence: float = 0.7):
    return {
        "type": "Feature",
        "properties": {
            "candidate_id": candidate_id,
            "point_count": len(coords),
            "mean_confidence": confidence,
            "image_name": "synthetic.png",
        },
        "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in coords]},
    }


def test_find_transition_evidence_keeps_diagonal_and_rejects_parallel() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (220.0, 0.0))
    features = [
        _feature([(100.0, -5.0), (130.0, -3.0), (160.0, -1.0)], candidate_id=1),
        _feature([(100.0, -5.0), (160.0, -5.0)], candidate_id=2),
    ]

    evidence = module.find_transition_evidence(
        features,
        guide=guide,
        min_station_span_m=8.0,
        min_offset_span_m=1.2,
        min_abs_slope=0.015,
        max_abs_slope=0.25,
        offset_pad_m=0.8,
        anchor_tolerance_m=1.8,
        min_confidence=0.28,
        min_points=2,
    )

    assert len(evidence) == 1
    assert evidence[0]["properties"]["pair_id"] == "minus_to_main"
    assert evidence[0]["properties"]["geom_kind"] == "raw_transition_evidence"


def test_dedupe_transition_evidence_keeps_best_overlapping_candidate() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (220.0, 0.0))
    features = [
        _feature([(100.0, -5.0), (130.0, -3.0), (160.0, -1.0)], candidate_id=1, confidence=0.7),
        _feature([(105.0, -5.0), (130.0, -3.0), (155.0, -1.2)], candidate_id=2, confidence=0.4),
    ]
    evidence = module.find_transition_evidence(
        features,
        guide=guide,
        min_station_span_m=8.0,
        min_offset_span_m=1.2,
        min_abs_slope=0.015,
        max_abs_slope=0.25,
        offset_pad_m=0.8,
        anchor_tolerance_m=1.8,
        min_confidence=0.28,
        min_points=2,
    )

    deduped = module.dedupe_transition_evidence(evidence)

    assert len(deduped) == 1
    assert deduped[0]["properties"]["candidate_id"] == 1
    assert deduped[0]["properties"]["connector_id"] == "E001"


def test_evidence_curve_proposal_does_not_force_full_band_without_anchor() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (220.0, 0.0))
    raw = _feature([(100.0, -5.0), (130.0, -3.0), (160.0, -1.0)], candidate_id=1)
    evidence = module.dedupe_transition_evidence(
        module.find_transition_evidence(
            [raw],
            guide=guide,
            min_station_span_m=8.0,
            min_offset_span_m=1.2,
            min_abs_slope=0.015,
            max_abs_slope=0.25,
            offset_pad_m=0.8,
            anchor_tolerance_m=1.8,
            min_confidence=0.28,
            min_points=2,
        )
    )[0]

    proposal = module.build_evidence_curve_proposal(
        evidence,
        pair=module.pair_by_id("minus_to_main"),
        guide=guide,
        switch_anchors=[],
        connector_splits=[],
        max_extrapolate_m=80.0,
        max_switch_anchor_distance_m=45.0,
        endpoint_snap_tolerance_m=0.35,
        max_proposal_span_m=220.0,
        sample_step_m=10.0,
    )

    assert proposal is not None
    coords = proposal["geometry"]["coordinates"]
    assert coords[0] == [100.0, -5.0]
    assert coords[-1] == [160.0, -1.0]
    assert proposal["properties"]["direction"] == "parallel_minus_5m->mainline_2_track"
    assert proposal["properties"]["geom_kind"] == "evidence_curve_proposal"
    assert proposal["properties"]["completion_m"] == 0.0
    assert proposal["properties"]["qa_note"] == "partial_raw_evidence_no_forced_smoothstep"


def test_evidence_curve_proposal_uses_switch_anchor_for_short_completion() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (220.0, 0.0))
    raw = _feature([(100.0, -5.0), (130.0, -3.0), (160.0, -1.0)], candidate_id=1)
    evidence = module.dedupe_transition_evidence(
        module.find_transition_evidence(
            [raw],
            guide=guide,
            min_station_span_m=8.0,
            min_offset_span_m=1.2,
            min_abs_slope=0.015,
            max_abs_slope=0.25,
            offset_pad_m=0.8,
            anchor_tolerance_m=1.8,
            min_confidence=0.28,
            min_points=2,
        )
    )[0]
    anchor = module.SwitchAnchor(
        anchor_id="switch_001",
        point=(175.0, 0.0),
        station=175.0,
        offset=0.0,
        pair_id="minus_to_main",
        target_band="mainline_2_track",
        target_offset=0.0,
        source="test",
    )

    proposal = module.build_evidence_curve_proposal(
        evidence,
        pair=module.pair_by_id("minus_to_main"),
        guide=guide,
        switch_anchors=[anchor],
        connector_splits=[],
        max_extrapolate_m=80.0,
        max_switch_anchor_distance_m=45.0,
        endpoint_snap_tolerance_m=0.35,
        max_proposal_span_m=220.0,
        sample_step_m=5.0,
    )

    assert proposal is not None
    coords = proposal["geometry"]["coordinates"]
    assert coords[0] == [100.0, -5.0]
    assert coords[-1] == [175.0, 0.0]
    assert [130.0, -3.0] in coords
    assert proposal["properties"]["completion_m"] == 15.0
    assert proposal["properties"]["anchor_id"] == "switch_001"
    assert proposal["properties"]["qa_note"] == "raw_evidence_plus_switch_anchor_completion"


def test_evidence_curve_proposal_can_trim_straight_tail_at_split() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (220.0, 0.0))
    raw = _feature([(100.0, -5.0), (130.0, -3.0), (160.0, -1.0)], candidate_id=1)
    evidence = module.dedupe_transition_evidence(
        module.find_transition_evidence(
            [raw],
            guide=guide,
            min_station_span_m=8.0,
            min_offset_span_m=1.2,
            min_abs_slope=0.015,
            max_abs_slope=0.25,
            offset_pad_m=0.8,
            anchor_tolerance_m=1.8,
            min_confidence=0.28,
            min_points=2,
        )
    )[0]
    split = module.ConnectorSplit(
        split_id="split_001",
        point=(120.0, -4.0),
        station=120.0,
        offset=-4.0,
        pair_id="minus_to_main",
        evidence_id="E001",
        keep_connector_side="north",
        straight_side="south",
        straight_band="parallel_minus_5m",
        source="test",
    )

    proposal = module.build_evidence_curve_proposal(
        evidence,
        pair=module.pair_by_id("minus_to_main"),
        guide=guide,
        switch_anchors=[],
        connector_splits=[split],
        max_extrapolate_m=80.0,
        max_switch_anchor_distance_m=45.0,
        endpoint_snap_tolerance_m=0.35,
        max_proposal_span_m=220.0,
        sample_step_m=5.0,
    )

    assert proposal is not None
    coords = proposal["geometry"]["coordinates"]
    assert coords[0] == [120.0, -4.0]
    assert coords[-1] == [160.0, -1.0]
    assert proposal["properties"]["split_id"] == "split_001"
    assert proposal["properties"]["straight_tail_m"] == 20.0
    assert proposal["properties"]["qa_note"] == "trimmed_straight_tail_at_user_split"


def test_smoothstep_curve_has_flat_offsets_at_ends() -> None:
    module = _load_module()
    guide = module.Guide((0.0, 0.0), (100.0, 0.0))

    coords = module.sample_smoothstep_curve(
        guide,
        start_s=0.0,
        end_s=100.0,
        start_offset=-5.0,
        end_offset=0.0,
        step_m=25.0,
    )

    assert coords[0] == (0.0, -5.0)
    assert coords[-1] == (100.0, 0.0)
    assert coords[2] == (50.0, -2.5)
