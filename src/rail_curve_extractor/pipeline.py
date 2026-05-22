from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks, savgol_filter

from .exporters import export_usda_basis_curves, export_usda_multi_basis_curves
from .geometry import (
    LocalFrame,
    cumulative_arc_length,
    ensure_odd,
    estimate_local_frame,
    heading_from_points,
    local_to_world,
    world_to_local,
)
from .io import load_config, load_point_cloud_data, save_json, save_xyz


DEFAULT_CONFIG: dict[str, Any] = {
    "roi": {
        "x_min": None,
        "x_max": None,
        "y_min": None,
        "y_max": None,
        "z_min": None,
        "z_max": None,
    },
    "oriented_roi": {
        "enabled": False,
        "origin": None,
        "axis_s": None,
        "axis_t": None,
        "s_min": None,
        "s_max": None,
        "t_min": None,
        "t_max": None,
        "z_min": None,
        "z_max": None,
    },
    "height_filter": {
        "enabled": True,
        "keep_top_percent": 0.55,
    },
    "slice_length": 0.5,
    "min_points_per_slice": 30,
    "rail_pair_spacing_min": 1.2,
    "rail_pair_spacing_max": 1.8,
    "rail_pair_spacing_target": 1.5,
    "peak_min_prominence_ratio": 0.25,
    "peak_search_bins": 80,
    "peak_window_radius": 0.08,
    "savgol_window": 9,
    "savgol_polyorder": 2,
    "xy_constraint": {
        "mode": "free",
        "smooth_window": 21,
        "straight_trim_ratio": 0.10,
    },
    "manual_anchor": {
        "enabled": False,
        "points": [],
        "snap_distance": 2.5,
        "score_weight": 14.0,
    },
    "guided_paths": {
        "enabled": False,
        "default_corridor_width": 5.0,
        "longitudinal_margin": 3.0,
        "anchor_snap_distance": 1.4,
        "anchor_score_weight": 24.0,
        "tracks": [],
        "turnouts": [],
    },
    "rail_network": {
        "enabled": False,
        "export_confirmed_only": True,
        "candidate_policy": "manual_confirmed_only",
        "nodes": [],
        "segments": [],
        "switches": [],
    },
    "curve_width": 0.05,
    "auto_track_split": {
        "enabled": False,
        "count": 1,
        "roi": {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "z_min": None,
            "z_max": None,
        },
        "oriented_roi": {
            "enabled": False,
            "origin": None,
            "axis_s": None,
            "axis_t": None,
            "s_min": None,
            "s_max": None,
            "t_min": None,
            "t_max": None,
            "z_min": None,
            "z_max": None,
        },
        "band_overlap_ratio": 0.0,
    },
    "tracks": [],
    "turnout": {
        "enabled": False,
        "roi": {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
            "z_min": None,
            "z_max": None,
        },
        "branch_min_separation": 0.45,
        "trace_max_candidates_per_slice": 8,
        "trace_max_gap_s": 3.0,
        "trace_max_lateral_jump_per_m": 0.55,
        "trace_min_path_points": 6,
        "trace_min_branch_length_m": 8.0,
        "trace_branch_start_max_separation": 1.75,
        "trace_branch_anchor_max_separation": 1.0,
        "trace_score_weight": 1.0,
        "trace_length_weight": 2.0,
        "trace_lateral_jump_weight": 8.0,
        "trace_gap_weight": 0.9,
        "trace_z_jump_weight": 1.5,
    },
    "advanced_las": {
        "enabled": True,
        "sample_max_points": 1000000,
        "ground_cell_size": 1.0,
        "ground_percentile": 0.10,
        "rail_height_min": 0.05,
        "rail_height_max": 0.22,
        "intensity_quantile_max": 0.50,
        "occupancy_threshold": 6,
        "min_component_points": 5000,
        "component_cell_size": 1.0,
        "corridor_quantile_low": 0.02,
        "corridor_quantile_high": 0.98,
        "corridor_margin": 1.5,
    },
}


@dataclass(slots=True)
class SliceDetection:
    center_local: np.ndarray
    rail_points_local: np.ndarray
    left_peak_t: float
    right_peak_t: float
    pair_mode: str = "dual"


@dataclass(slots=True)
class TrackResult:
    track_id: int
    frame: LocalFrame
    filtered_points_world: np.ndarray
    rail_points_world: np.ndarray
    centerline_world: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(slots=True)
class TurnoutResult:
    turnout_id: int
    frame: LocalFrame
    filtered_points_world: np.ndarray
    rail_points_world: np.ndarray
    main_centerline_world: np.ndarray
    branch_centerline_world: np.ndarray
    switch_point_world: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(slots=True)
class _CandidatePath:
    detections: list[SliceDetection]
    slice_indices: list[int]
    score: float = 0.0


@dataclass(slots=True)
class PipelineResult:
    config: dict[str, Any]
    raw_points_world: np.ndarray
    frame: LocalFrame
    filtered_points_world: np.ndarray
    rail_points_world: np.ndarray
    centerline_world: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)
    track_results: list[TrackResult] = field(default_factory=list)
    turnout_results: list[TurnoutResult] = field(default_factory=list)


def run_pipeline(input_path: Path, output_dir: Path, config_path: Path | None = None) -> PipelineResult:
    result = analyze_input(input_path=input_path, config_path=config_path)
    export_pipeline_result(
        result=result,
        output_dir=output_dir,
        input_path=input_path,
        config_path=config_path,
    )
    return result


def prepare_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = _merge_config(DEFAULT_CONFIG, load_config(config_path))
    if overrides:
        merged = _merge_config(merged, overrides)
    return merged


def analyze_input(
    input_path: Path,
    config_path: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> PipelineResult:
    point_cloud = load_point_cloud_data(input_path)
    config = prepare_config(config_path=config_path, overrides=config_overrides)
    return analyze_point_cloud(
        raw_points=point_cloud.points,
        config=config,
        intensity=point_cloud.intensity,
        source_format=point_cloud.source_format,
    )


def analyze_point_cloud(
    raw_points: np.ndarray,
    config: dict[str, Any],
    intensity: np.ndarray | None = None,
    source_format: str = "",
) -> PipelineResult:
    if _guided_paths_config_enabled(config):
        return _analyze_guided_paths_point_cloud(raw_points, config, intensity, source_format)

    if _turnout_config_enabled(config):
        return _analyze_turnout_point_cloud(raw_points, config, intensity, source_format)

    track_configs = _enabled_track_configs(config)
    if track_configs:
        return _analyze_multi_track_point_cloud(raw_points, config, track_configs, intensity, source_format)

    track_result = _analyze_single_track(
        raw_points=raw_points,
        config=config,
        track_id=1,
        roi=config["roi"],
        oriented_roi=config.get("oriented_roi"),
        intensity=intensity,
        source_format=source_format,
        input_point_count=len(raw_points),
    )

    return PipelineResult(
        config=config,
        raw_points_world=raw_points,
        frame=track_result.frame,
        filtered_points_world=track_result.filtered_points_world,
        rail_points_world=track_result.rail_points_world,
        centerline_world=track_result.centerline_world,
        summary=track_result.summary,
        track_results=[track_result],
    )


def _analyze_guided_paths_point_cloud(
    raw_points: np.ndarray,
    config: dict[str, Any],
    intensity: np.ndarray | None,
    source_format: str,
) -> PipelineResult:
    guided_config = config.get("guided_paths", {})
    guided_tracks = guided_config.get("tracks", []) if isinstance(guided_config, dict) else []
    guided_turnouts = guided_config.get("turnouts", []) if isinstance(guided_config, dict) else []
    if not isinstance(guided_tracks, list):
        guided_tracks = []
    if not isinstance(guided_turnouts, list):
        guided_turnouts = []

    track_results: list[TrackResult] = []
    turnout_results: list[TurnoutResult] = []
    failures: list[dict[str, Any]] = []
    next_track_id = 1

    for index, guided_track in enumerate(guided_tracks, start=1):
        if not isinstance(guided_track, dict) or not bool(guided_track.get("enabled", True)):
            continue
        track_id = int(guided_track.get("id") or next_track_id)
        next_track_id = max(next_track_id, track_id + 1)
        try:
            track_points = _guided_path_points(guided_track.get("points"), f"guided track {track_id}")
            oriented_roi = _guided_oriented_roi_from_config(guided_track, guided_config, track_points)
            track_runtime_config = _guided_track_runtime_config(config, guided_config, guided_track, track_points)
            roi = _merge_config(config["roi"], guided_track.get("roi", {}))
            track_result = _analyze_single_track(
                raw_points=raw_points,
                config=track_runtime_config,
                track_id=track_id,
                roi=roi,
                oriented_roi=oriented_roi,
                intensity=intensity,
                source_format=source_format,
                input_point_count=len(raw_points),
            )
            track_result.summary["source"] = "guided_path"
            track_result.summary["guided_path_points"] = len(track_points)
            track_results.append(track_result)
        except Exception as exc:
            failures.append({"type": "track", "id": track_id, "error": str(exc)})

    for index, guided_turnout in enumerate(guided_turnouts, start=1):
        if not isinstance(guided_turnout, dict) or not bool(guided_turnout.get("enabled", True)):
            continue
        turnout_id = int(guided_turnout.get("id") or index)
        try:
            main_points = _guided_path_points(guided_turnout.get("main_points"), f"guided turnout {turnout_id} main")
            branch_points = _guided_path_points(guided_turnout.get("branch_points"), f"guided turnout {turnout_id} branch")
            oriented_roi = _guided_turnout_oriented_roi_from_config(
                guided_turnout,
                guided_config,
                main_points,
                branch_points,
            )
            turnout_runtime_config = _guided_turnout_runtime_config(
                config,
                guided_config,
                guided_turnout,
                main_points,
                branch_points,
                oriented_roi,
            )
            roi = _merge_config(config["roi"], guided_turnout.get("roi", {}))
            turnout_result = _analyze_turnout(
                raw_points=raw_points,
                config=turnout_runtime_config,
                turnout_id=turnout_id,
                roi=roi,
                oriented_roi=oriented_roi,
                intensity=intensity,
                source_format=source_format,
            )
            turnout_result.summary["source"] = "guided_path"
            turnout_result.summary["guided_main_points"] = len(main_points)
            turnout_result.summary["guided_branch_points"] = len(branch_points)
            turnout_results.append(turnout_result)

            main_track_id = int(guided_turnout.get("main_track_id") or next_track_id)
            next_track_id = max(next_track_id, main_track_id + 1)
            branch_track_id = int(guided_turnout.get("branch_track_id") or next_track_id)
            next_track_id = max(next_track_id, branch_track_id + 1)
            track_results.append(_track_result_from_turnout(turnout_result, track_id=main_track_id, branch=False))
            track_results.append(_track_result_from_turnout(turnout_result, track_id=branch_track_id, branch=True))
        except Exception as exc:
            failures.append({"type": "turnout", "id": turnout_id, "error": str(exc)})

    if not track_results:
        details = "; ".join(f"{item['type']} {item['id']}: {item['error']}" for item in failures)
        raise RuntimeError(f"No guided path produced a centerline. {details}")

    filtered_sources = [item.filtered_points_world for item in track_results if "turnout_id" not in item.summary]
    filtered_sources.extend(item.filtered_points_world for item in turnout_results)
    rail_sources = [item.rail_points_world for item in track_results if "turnout_id" not in item.summary]
    rail_sources.extend(item.rail_points_world for item in turnout_results)
    filtered_points = _vstack_nonempty(filtered_sources)
    rail_points = _vstack_nonempty(rail_sources)
    centerline = _vstack_nonempty([item.centerline_world for item in track_results])
    primary = track_results[0]
    total_length = float(sum(item.summary.get("curve_length_m", 0.0) for item in track_results))
    summary = {
        "input_points": int(len(raw_points)),
        "filtered_points": int(len(filtered_points)),
        "working_points": int(sum(item.summary.get("working_points", 0) for item in track_results)),
        "rail_points": int(len(rail_points)),
        "centerline_points": int(len(centerline)),
        "curve_length_m": total_length,
        "track_count": len(track_results),
        "turnout_count": len(turnout_results),
        "failed_guided_count": len(failures),
        "guided_failures": failures,
        "tracks": [_public_track_summary(item) for item in track_results],
        "turnouts": [dict(item.summary) for item in turnout_results],
        "preprocessing_mode": "guided_paths",
        "guided_path_mode": True,
    }

    return PipelineResult(
        config=config,
        raw_points_world=raw_points,
        frame=primary.frame,
        filtered_points_world=filtered_points,
        rail_points_world=rail_points,
        centerline_world=centerline,
        summary=summary,
        track_results=track_results,
        turnout_results=turnout_results,
    )


def _analyze_turnout_point_cloud(
    raw_points: np.ndarray,
    config: dict[str, Any],
    intensity: np.ndarray | None,
    source_format: str,
) -> PipelineResult:
    turnout_config = config.get("turnout", {})
    roi = _merge_config(config["roi"], turnout_config.get("roi", {}))
    oriented_roi = turnout_config.get("oriented_roi") or config.get("oriented_roi")
    turnout_result = _analyze_turnout(
        raw_points=raw_points,
        config=config,
        turnout_id=1,
        roi=roi,
        oriented_roi=oriented_roi,
        intensity=intensity,
        source_format=source_format,
    )
    main_track = _track_result_from_turnout(turnout_result, track_id=1, branch=False)
    branch_track = _track_result_from_turnout(turnout_result, track_id=2, branch=True)
    centerline = np.vstack([turnout_result.main_centerline_world, turnout_result.branch_centerline_world])
    summary = {
        "input_points": int(len(raw_points)),
        "filtered_points": int(len(turnout_result.filtered_points_world)),
        "working_points": int(turnout_result.summary.get("working_points", len(turnout_result.filtered_points_world))),
        "rail_points": int(len(turnout_result.rail_points_world)),
        "centerline_points": int(len(centerline)),
        "curve_length_m": float(
            turnout_result.summary.get("main_curve_length_m", 0.0)
            + turnout_result.summary.get("branch_curve_length_m", 0.0)
        ),
        "track_count": 2,
        "turnout_count": 1,
        "turnouts": [dict(turnout_result.summary)],
        "tracks": [_public_track_summary(main_track), _public_track_summary(branch_track)],
        "preprocessing_mode": "turnout_roi",
    }
    return PipelineResult(
        config=config,
        raw_points_world=raw_points,
        frame=turnout_result.frame,
        filtered_points_world=turnout_result.filtered_points_world,
        rail_points_world=turnout_result.rail_points_world,
        centerline_world=centerline,
        summary=summary,
        track_results=[main_track, branch_track],
        turnout_results=[turnout_result],
    )


def _analyze_multi_track_point_cloud(
    raw_points: np.ndarray,
    config: dict[str, Any],
    track_configs: list[dict[str, Any]],
    intensity: np.ndarray | None,
    source_format: str,
) -> PipelineResult:
    track_results: list[TrackResult] = []
    failures: list[dict[str, Any]] = []

    for index, track_config in enumerate(track_configs, start=1):
        track_id = int(track_config.get("id") or index)
        roi = _merge_config(config["roi"], track_config.get("roi", {}))
        oriented_roi = track_config.get("oriented_roi") or config.get("oriented_roi")
        track_runtime_config = config
        if isinstance(track_config.get("manual_anchor"), dict):
            track_runtime_config = _merge_config(
                config,
                {"manual_anchor": track_config.get("manual_anchor", {})},
            )
        try:
            track_results.append(
                _analyze_single_track(
                    raw_points=raw_points,
                    config=track_runtime_config,
                    track_id=track_id,
                    roi=roi,
                    oriented_roi=oriented_roi,
                    intensity=intensity,
                    source_format=source_format,
                    input_point_count=len(raw_points),
                )
            )
        except Exception as exc:
            failures.append({"track_id": track_id, "error": str(exc)})

    if not track_results:
        details = "; ".join(f"track {item['track_id']}: {item['error']}" for item in failures)
        raise RuntimeError(f"No configured track ROI produced a centerline. {details}")

    filtered_points = _vstack_nonempty([item.filtered_points_world for item in track_results])
    rail_points = _vstack_nonempty([item.rail_points_world for item in track_results])
    centerline = _vstack_nonempty([item.centerline_world for item in track_results])
    primary = track_results[0]
    total_length = float(sum(item.summary.get("curve_length_m", 0.0) for item in track_results))
    auto_split_enabled = any(track.get("source") == "auto_track_split" for track in track_configs)
    summary = {
        "input_points": int(len(raw_points)),
        "filtered_points": int(len(filtered_points)),
        "working_points": int(sum(item.summary.get("working_points", 0) for item in track_results)),
        "rail_points": int(len(rail_points)),
        "centerline_points": int(len(centerline)),
        "curve_length_m": total_length,
        "track_count": len(track_results),
        "failed_track_count": len(failures),
        "tracks": [_public_track_summary(item) for item in track_results],
        "track_failures": failures,
        "preprocessing_mode": "auto_track_split" if auto_split_enabled else "multi_track_roi",
        "configured_track_count": len(track_configs),
        "auto_track_split": auto_split_enabled,
    }

    return PipelineResult(
        config=config,
        raw_points_world=raw_points,
        frame=primary.frame,
        filtered_points_world=filtered_points,
        rail_points_world=rail_points,
        centerline_world=centerline,
        summary=summary,
        track_results=track_results,
    )


def _analyze_turnout(
    raw_points: np.ndarray,
    config: dict[str, Any],
    turnout_id: int,
    roi: dict[str, float | None],
    oriented_roi: dict[str, Any] | None,
    intensity: np.ndarray | None,
    source_format: str,
) -> TurnoutResult:
    roi_mask = _build_roi_mask(raw_points, roi, oriented_roi)
    roi_points = raw_points[roi_mask]
    roi_intensity = intensity[roi_mask] if intensity is not None else None

    preprocessing_mode = "basic_height_filter"
    filtered_points = _apply_global_height_filter(roi_points, config["height_filter"])
    preview_points = filtered_points
    if _should_use_advanced_las(config, source_format, roi_intensity):
        advanced_selection = _extract_track_corridor_points(roi_points, roi_intensity, config)
        if advanced_selection is not None:
            corridor_points, component_points = advanced_selection
            filtered_points = corridor_points
            preview_points = component_points
            preprocessing_mode = "advanced_las_corridor"

    if len(filtered_points) < 80:
        raise RuntimeError("Turnout ROI has too few filtered points. Enlarge ROI or relax height filtering.")

    frame, frame_source = _build_analysis_frame(filtered_points, oriented_roi)
    local_points = world_to_local(filtered_points, frame)
    turnout_config = config.get("turnout", {})
    main_anchor_points_local = _manual_anchor_points_to_local(
        {"enabled": bool(turnout_config.get("main_anchor_points")), "points": turnout_config.get("main_anchor_points", [])},
        frame,
    )
    if main_anchor_points_local is None:
        main_anchor_points_local = _manual_anchor_points_to_local(config.get("manual_anchor"), frame)
    branch_anchor_points_local = _manual_anchor_points_to_local(
        {"enabled": bool(turnout_config.get("branch_anchor_points")), "points": turnout_config.get("branch_anchor_points", [])},
        frame,
    )
    main_detections, branch_detections, _switch_local = _detect_turnout_paths(
        local_points,
        config,
        main_anchor_points_local=main_anchor_points_local,
        branch_anchor_points_local=branch_anchor_points_local,
    )
    if len(main_detections) < 4 or len(branch_detections) < 4:
        raise RuntimeError("Too few valid turnout slices. Adjust turnout ROI, slice length, or rail spacing.")

    main_local = _build_centerline_local(main_detections, config)
    branch_local = _build_centerline_local(branch_detections, config)
    branch_local = _orient_branch_from_switch(main_local, branch_local)
    branch_local = _anchor_branch_start_to_main(main_local, branch_local, config)
    switch_local = _estimate_switch_on_main(main_local, branch_local)
    rail_points_local = np.vstack(
        [item.rail_points_local for item in main_detections]
        + [item.rail_points_local for item in branch_detections]
    )
    main_world = local_to_world(main_local, frame)
    branch_world = local_to_world(branch_local, frame)
    rail_points_world = local_to_world(rail_points_local, frame)
    switch_point_world = local_to_world(switch_local.reshape(1, 3), frame)[0]
    main_arc = cumulative_arc_length(main_world)
    branch_arc = cumulative_arc_length(branch_world)
    confidence = _estimate_turnout_confidence(main_detections, branch_detections, main_world, branch_world)
    summary = {
        "turnout_id": int(turnout_id),
        "roi_points": int(len(roi_points)),
        "filtered_points": int(len(preview_points)),
        "working_points": int(len(filtered_points)),
        "rail_points": int(len(rail_points_world)),
        "main_centerline_points": int(len(main_world)),
        "branch_centerline_points": int(len(branch_world)),
        "centerline_points": int(len(main_world) + len(branch_world)),
        "main_curve_length_m": float(main_arc[-1]) if len(main_arc) else 0.0,
        "branch_curve_length_m": float(branch_arc[-1]) if len(branch_arc) else 0.0,
        "switch_point_world": [float(value) for value in switch_point_world],
        "confidence": confidence,
        "preprocessing_mode": preprocessing_mode,
        "local_frame_source": frame_source,
        "xy_constraint_mode": str(config.get("xy_constraint", {}).get("mode", "free")),
        "turnout_trace_mode": "graph_search",
        "manual_main_anchor_points": int(len(main_anchor_points_local)) if main_anchor_points_local is not None else 0,
        "manual_branch_anchor_points": int(len(branch_anchor_points_local)) if branch_anchor_points_local is not None else 0,
    }
    return TurnoutResult(
        turnout_id=int(turnout_id),
        frame=frame,
        filtered_points_world=preview_points,
        rail_points_world=rail_points_world,
        main_centerline_world=main_world,
        branch_centerline_world=branch_world,
        switch_point_world=switch_point_world,
        summary=summary,
        confidence=confidence,
    )


def _track_result_from_turnout(turnout_result: TurnoutResult, track_id: int, branch: bool) -> TrackResult:
    centerline = turnout_result.branch_centerline_world if branch else turnout_result.main_centerline_world
    arc = cumulative_arc_length(centerline)
    label = "branch" if branch else "main"
    summary = {
        "track_id": track_id,
        "turnout_id": turnout_result.turnout_id,
        "turnout_path": label,
        "input_points": int(turnout_result.summary.get("roi_points", 0)),
        "filtered_points": int(len(turnout_result.filtered_points_world)),
        "working_points": int(turnout_result.summary.get("working_points", 0)),
        "rail_points": int(len(turnout_result.rail_points_world)),
        "centerline_points": int(len(centerline)),
        "curve_length_m": float(arc[-1]) if len(arc) else 0.0,
        "confidence": turnout_result.confidence,
        "preprocessing_mode": "turnout_roi",
    }
    return TrackResult(
        track_id=track_id,
        frame=turnout_result.frame,
        filtered_points_world=turnout_result.filtered_points_world,
        rail_points_world=turnout_result.rail_points_world,
        centerline_world=centerline,
        summary=summary,
        confidence=turnout_result.confidence,
    )


def _analyze_single_track(
    raw_points: np.ndarray,
    config: dict[str, Any],
    track_id: int,
    roi: dict[str, float | None],
    oriented_roi: dict[str, Any] | None,
    intensity: np.ndarray | None,
    source_format: str,
    input_point_count: int,
) -> TrackResult:
    roi_mask = _build_roi_mask(raw_points, roi, oriented_roi)
    roi_points = raw_points[roi_mask]
    roi_intensity = intensity[roi_mask] if intensity is not None else None

    preprocessing_mode = "basic_height_filter"
    filtered_points = _apply_global_height_filter(roi_points, config["height_filter"])
    preview_points = filtered_points

    if _should_use_advanced_las(config, source_format, roi_intensity):
        advanced_selection = _extract_track_corridor_points(roi_points, roi_intensity, config)
        if advanced_selection is not None:
            corridor_points, component_points = advanced_selection
            filtered_points = corridor_points
            preview_points = component_points
            preprocessing_mode = "advanced_las_corridor"

    if len(filtered_points) < 50:
        raise RuntimeError("Filtered point count is too small. Relax ROI or height filtering parameters.")

    frame, frame_source = _build_analysis_frame(filtered_points, oriented_roi)
    local_points = world_to_local(filtered_points, frame)
    anchor_points_local = _manual_anchor_points_to_local(config.get("manual_anchor"), frame)
    detections = _detect_slices(local_points, config, anchor_points_local=anchor_points_local)
    if len(detections) < 4 and preprocessing_mode != "basic_height_filter":
        fallback_points = _apply_global_height_filter(roi_points, config["height_filter"])
        if len(fallback_points) >= 50:
            fallback_frame, fallback_frame_source = _build_analysis_frame(fallback_points, oriented_roi)
            fallback_local = world_to_local(fallback_points, fallback_frame)
            fallback_anchor_points_local = _manual_anchor_points_to_local(config.get("manual_anchor"), fallback_frame)
            fallback_detections = _detect_slices(
                fallback_local,
                config,
                anchor_points_local=fallback_anchor_points_local,
            )
            if len(fallback_detections) > len(detections):
                filtered_points = fallback_points
                preview_points = fallback_points
                frame = fallback_frame
                frame_source = fallback_frame_source
                local_points = fallback_local
                anchor_points_local = fallback_anchor_points_local
                detections = fallback_detections
                preprocessing_mode = "basic_height_filter_fallback"

    if len(detections) < 4:
        raise RuntimeError("Too few valid slices to form a stable centerline. Adjust slice length or rail spacing.")

    centerline_local = _build_centerline_local(detections, config)
    rail_points_local = np.vstack([item.rail_points_local for item in detections])
    centerline_world = local_to_world(centerline_local, frame)
    rail_points_world = local_to_world(rail_points_local, frame)

    arc = cumulative_arc_length(centerline_world)
    pair_mode_counts = _count_detection_pair_modes(detections)
    summary = {
        "track_id": int(track_id),
        "input_points": int(input_point_count),
        "roi_points": int(len(roi_points)),
        "filtered_points": int(len(preview_points)),
        "working_points": int(len(filtered_points)),
        "rail_points": int(len(rail_points_world)),
        "centerline_points": int(len(centerline_world)),
        "curve_length_m": float(arc[-1]) if len(arc) else 0.0,
        "heading_rad": float(heading_from_points(centerline_world)),
        "local_frame_origin": [float(v) for v in frame.origin],
        "local_frame_rotation": [[float(v) for v in row] for row in frame.rotation],
        "local_frame_source": frame_source,
        "preprocessing_mode": preprocessing_mode,
        "xy_constraint_mode": str(config.get("xy_constraint", {}).get("mode", "free")),
        "manual_anchor_enabled": anchor_points_local is not None,
        "manual_anchor_points": int(len(anchor_points_local)) if anchor_points_local is not None else 0,
        "slice_pair_modes": pair_mode_counts,
        "single_rail_inferred_slices": int(
            sum(count for mode, count in pair_mode_counts.items() if mode.startswith("single_"))
        ),
        "confidence": _estimate_track_confidence(detections, centerline_world, rail_points_world),
    }

    return TrackResult(
        track_id=int(track_id),
        frame=frame,
        filtered_points_world=preview_points,
        rail_points_world=rail_points_world,
        centerline_world=centerline_world,
        summary=summary,
        confidence=float(summary["confidence"]),
    )


def export_pipeline_result(
    result: PipelineResult,
    output_dir: Path,
    input_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_xyz(output_dir / "filtered_points.xyz", result.filtered_points_world)
    save_xyz(output_dir / "rail_points.xyz", result.rail_points_world)
    save_xyz(output_dir / "centerline_points.xyz", result.centerline_world)
    if result.turnout_results:
        for turnout_result in result.turnout_results:
            prefix = f"turnout_{turnout_result.turnout_id}"
            save_xyz(output_dir / f"{prefix}_filtered_points.xyz", turnout_result.filtered_points_world)
            save_xyz(output_dir / f"{prefix}_rail_points.xyz", turnout_result.rail_points_world)
            save_xyz(output_dir / f"{prefix}_main_centerline_points.xyz", turnout_result.main_centerline_world)
            save_xyz(output_dir / f"{prefix}_branch_centerline_points.xyz", turnout_result.branch_centerline_world)
            save_xyz(output_dir / f"{prefix}_switch_point.xyz", turnout_result.switch_point_world.reshape(1, 3))
            export_usda_multi_basis_curves(
                output_dir / f"{prefix}_centerlines.usda",
                [turnout_result.main_centerline_world, turnout_result.branch_centerline_world],
                width=float(result.config["curve_width"]),
            )
    if _is_multi_track_result(result):
        for track_result in result.track_results:
            prefix = f"track_{track_result.track_id}"
            save_xyz(output_dir / f"{prefix}_filtered_points.xyz", track_result.filtered_points_world)
            save_xyz(output_dir / f"{prefix}_rail_points.xyz", track_result.rail_points_world)
            save_xyz(output_dir / f"{prefix}_centerline_points.xyz", track_result.centerline_world)
        export_usda_multi_basis_curves(
            output_dir / "rail_centerlines.usda",
            [item.centerline_world for item in result.track_results],
            width=float(result.config["curve_width"]),
        )
    else:
        export_usda_basis_curves(
            output_dir / "rail_centerline.usda",
            result.centerline_world,
            width=float(result.config["curve_width"]),
        )
    save_json(output_dir / "used_config.json", result.config)

    summary = dict(result.summary)
    summary.update(
        {
            "input_path": str(input_path) if input_path else None,
            "config_path": str(config_path) if config_path else None,
            "output_dir": str(output_dir),
        }
    )
    save_json(output_dir / "run_summary.json", summary)
    result.summary = summary
    return summary


def _is_multi_track_result(result: PipelineResult) -> bool:
    return len(result.track_results) > 1 or result.summary.get("track_count", 1) > 1


def _merge_config(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key, default_value in defaults.items():
        if isinstance(default_value, dict):
            override_value = overrides.get(key, {})
            if override_value is None:
                override_value = {}
            merged[key] = _merge_config(default_value, override_value)
        else:
            merged[key] = overrides.get(key, default_value)
    for key, value in overrides.items():
        if key not in merged:
            merged[key] = value
    return merged


def _guided_paths_config_enabled(config: dict[str, Any]) -> bool:
    guided_config = config.get("guided_paths") or {}
    if not isinstance(guided_config, dict) or not bool(guided_config.get("enabled", False)):
        return False
    tracks = guided_config.get("tracks", [])
    turnouts = guided_config.get("turnouts", [])
    return (isinstance(tracks, list) and len(tracks) > 0) or (isinstance(turnouts, list) and len(turnouts) > 0)


def _guided_path_points(raw_points: Any, label: str) -> list[list[float]]:
    if not isinstance(raw_points, list):
        raise ValueError(f"{label} points must be a list.")

    parsed_points: list[list[float]] = []
    for point in raw_points:
        if isinstance(point, dict):
            values = [point.get("x"), point.get("y")]
            if point.get("z") is not None:
                values.append(point.get("z"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            values = [point[0], point[1]]
            if len(point) >= 3 and point[2] is not None:
                values.append(point[2])
        else:
            continue
        try:
            parsed_points.append([float(value) for value in values])
        except (TypeError, ValueError):
            continue

    if len(parsed_points) < 2:
        raise ValueError(f"{label} requires at least two valid XY points.")
    return parsed_points


def _guided_oriented_roi_from_config(
    item_config: dict[str, Any],
    guided_config: dict[str, Any],
    path_points: list[list[float]],
) -> dict[str, Any]:
    oriented_roi = item_config.get("oriented_roi")
    if _oriented_roi_enabled(oriented_roi):
        return dict(oriented_roi)
    corridor_width = _guided_corridor_width(item_config, guided_config)
    longitudinal_margin = float(item_config.get("longitudinal_margin", guided_config.get("longitudinal_margin", 3.0)))
    return _oriented_roi_from_guided_points(path_points, corridor_width, longitudinal_margin)


def _guided_turnout_oriented_roi_from_config(
    item_config: dict[str, Any],
    guided_config: dict[str, Any],
    main_points: list[list[float]],
    branch_points: list[list[float]],
) -> dict[str, Any]:
    oriented_roi = item_config.get("oriented_roi")
    if _oriented_roi_enabled(oriented_roi):
        return dict(oriented_roi)
    corridor_width = _guided_corridor_width(item_config, guided_config)
    longitudinal_margin = float(item_config.get("longitudinal_margin", guided_config.get("longitudinal_margin", 3.0)))
    return _oriented_roi_from_guided_points(
        [*main_points, *branch_points],
        corridor_width,
        longitudinal_margin,
        axis_points=main_points,
    )


def _guided_corridor_width(item_config: dict[str, Any], guided_config: dict[str, Any]) -> float:
    width = float(item_config.get("corridor_width", guided_config.get("default_corridor_width", 5.0)))
    return max(width, float(item_config.get("min_corridor_width", 2.4)))


def _oriented_roi_from_guided_points(
    path_points: list[list[float]],
    corridor_width: float,
    longitudinal_margin: float,
    axis_points: list[list[float]] | None = None,
) -> dict[str, Any]:
    xy = np.asarray([[point[0], point[1]] for point in path_points], dtype=float)
    axis_xy = np.asarray([[point[0], point[1]] for point in (axis_points or path_points)], dtype=float)
    if len(xy) < 2 or len(axis_xy) < 2:
        raise ValueError("Guided path requires at least two XY points to build a corridor.")

    origin_xy = axis_xy.mean(axis=0)
    axis_s = axis_xy[-1] - axis_xy[0]
    axis_norm = float(np.linalg.norm(axis_s))
    if axis_norm <= 1e-6:
        centered = axis_xy - origin_xy
        covariance = np.cov(centered.T, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis_s = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
        axis_norm = float(np.linalg.norm(axis_s))
    if axis_norm <= 1e-6:
        raise ValueError("Guided path points are too close to infer a track direction.")

    axis_s = axis_s / axis_norm
    if float(axis_s @ (axis_xy[-1] - axis_xy[0])) < 0.0:
        axis_s = -axis_s
    axis_t = np.array([-axis_s[1], axis_s[0]], dtype=float)

    shifted = xy - origin_xy
    local_s = shifted @ axis_s
    local_t = shifted @ axis_t
    half_width = max(float(corridor_width) / 2.0, 1.2)
    margin = max(float(longitudinal_margin), 0.0)
    return {
        "enabled": True,
        "origin": [float(origin_xy[0]), float(origin_xy[1])],
        "axis_s": [float(axis_s[0]), float(axis_s[1])],
        "axis_t": [float(axis_t[0]), float(axis_t[1])],
        "s_min": float(local_s.min() - margin),
        "s_max": float(local_s.max() + margin),
        "t_min": float(local_t.min() - half_width),
        "t_max": float(local_t.max() + half_width),
        "z_min": None,
        "z_max": None,
    }


def _guided_track_runtime_config(
    config: dict[str, Any],
    guided_config: dict[str, Any],
    item_config: dict[str, Any],
    path_points: list[list[float]],
) -> dict[str, Any]:
    manual_anchor = _guided_anchor_config(config, guided_config, item_config, path_points)
    if isinstance(item_config.get("manual_anchor"), dict):
        manual_anchor = _merge_config(manual_anchor, item_config["manual_anchor"])
        manual_anchor["enabled"] = True
        manual_anchor["points"] = path_points
    return _merge_config(config, {"manual_anchor": manual_anchor})


def _guided_turnout_runtime_config(
    config: dict[str, Any],
    guided_config: dict[str, Any],
    item_config: dict[str, Any],
    main_points: list[list[float]],
    branch_points: list[list[float]],
    oriented_roi: dict[str, Any],
) -> dict[str, Any]:
    manual_anchor = _guided_anchor_config(config, guided_config, item_config, main_points)
    turnout_defaults = config.get("turnout", {})
    turnout_overrides = item_config.get("turnout", {}) if isinstance(item_config.get("turnout"), dict) else {}
    turnout_config = _merge_config(turnout_defaults, turnout_overrides)
    turnout_config.update(
        {
            "enabled": True,
            "main_anchor_points": main_points,
            "branch_anchor_points": branch_points,
            "oriented_roi": oriented_roi,
        }
    )
    return _merge_config(config, {"manual_anchor": manual_anchor, "turnout": turnout_config})


def _guided_anchor_config(
    config: dict[str, Any],
    guided_config: dict[str, Any],
    item_config: dict[str, Any],
    path_points: list[list[float]],
) -> dict[str, Any]:
    existing_anchor = config.get("manual_anchor", {}) if isinstance(config.get("manual_anchor"), dict) else {}
    return {
        "enabled": True,
        "points": path_points,
        "snap_distance": float(
            item_config.get(
                "anchor_snap_distance",
                guided_config.get("anchor_snap_distance", existing_anchor.get("snap_distance", 1.4)),
            )
        ),
        "score_weight": float(
            item_config.get(
                "anchor_score_weight",
                guided_config.get("anchor_score_weight", existing_anchor.get("score_weight", 24.0)),
            )
        ),
    }


def _turnout_config_enabled(config: dict[str, Any]) -> bool:
    turnout = config.get("turnout") or {}
    if not isinstance(turnout, dict) or not bool(turnout.get("enabled", False)):
        return False
    roi = turnout.get("roi", {})
    has_axis_roi = isinstance(roi, dict) and any(value is not None for value in roi.values())
    return has_axis_roi or _oriented_roi_enabled(turnout.get("oriented_roi"))


def _enabled_track_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    tracks = config.get("tracks") or []
    if not isinstance(tracks, list):
        tracks = []
    enabled: list[dict[str, Any]] = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        if not bool(track.get("enabled", True)):
            continue
        roi = track.get("roi", {})
        has_axis_roi = isinstance(roi, dict) and any(value is not None for value in roi.values())
        if has_axis_roi or _oriented_roi_enabled(track.get("oriented_roi")):
            enabled.append(track)
    if enabled:
        return enabled
    return _auto_track_split_configs(config)


def _auto_track_split_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    auto_config = config.get("auto_track_split") or {}
    if not isinstance(auto_config, dict) or not bool(auto_config.get("enabled", False)):
        return []

    try:
        track_count = int(auto_config.get("count", 1))
    except (TypeError, ValueError):
        return []
    if track_count < 1:
        return []

    oriented_roi = auto_config.get("oriented_roi")
    if not _oriented_roi_enabled(oriented_roi):
        oriented_roi = config.get("oriented_roi")
    if not _oriented_roi_enabled(oriented_roi):
        return []

    roi = auto_config.get("roi", {})
    if not isinstance(roi, dict):
        roi = {}

    t_min = float(oriented_roi["t_min"])
    t_max = float(oriented_roi["t_max"])
    if t_max <= t_min:
        return []

    overlap_ratio = max(0.0, float(auto_config.get("band_overlap_ratio", 0.0)))
    band_width = (t_max - t_min) / track_count
    overlap = band_width * overlap_ratio

    generated: list[dict[str, Any]] = []
    for index in range(track_count):
        band_t_min = t_min + band_width * index
        band_t_max = band_t_min + band_width
        track_oriented_roi = dict(oriented_roi)
        track_oriented_roi["enabled"] = True
        track_oriented_roi["t_min"] = max(t_min, band_t_min - overlap)
        track_oriented_roi["t_max"] = min(t_max, band_t_max + overlap)
        generated.append(
            {
                "id": index + 1,
                "enabled": True,
                "roi": dict(roi),
                "oriented_roi": track_oriented_roi,
                "source": "auto_track_split",
            }
        )
    return generated


def _vstack_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    nonempty = [array for array in arrays if len(array) > 0]
    if not nonempty:
        return np.empty((0, 3), dtype=float)
    return np.vstack(nonempty)


def _public_track_summary(track_result: TrackResult) -> dict[str, Any]:
    summary = dict(track_result.summary)
    summary["track_id"] = track_result.track_id
    summary["confidence"] = track_result.confidence
    return summary


def _estimate_track_confidence(
    detections: list[SliceDetection],
    centerline_world: np.ndarray,
    rail_points_world: np.ndarray,
) -> float:
    if len(detections) == 0 or len(centerline_world) < 2 or len(rail_points_world) == 0:
        return 0.0
    spacings = np.array([item.right_peak_t - item.left_peak_t for item in detections], dtype=float)
    spacing_stability = 1.0 / (1.0 + float(np.std(spacings)) * 8.0)
    coverage = min(1.0, len(detections) / 12.0)
    mode_quality = float(np.mean([_pair_mode_quality(item.pair_mode) for item in detections]))
    return float(max(0.0, min(1.0, 0.55 * spacing_stability + 0.25 * coverage + 0.20 * mode_quality)))


def _pair_mode_quality(pair_mode: str) -> float:
    if pair_mode == "dual":
        return 1.0
    if pair_mode == "quantile_fallback":
        return 0.92
    if pair_mode.startswith("single_"):
        return 0.72
    return 0.85


def _count_detection_pair_modes(detections: list[SliceDetection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.pair_mode] = counts.get(detection.pair_mode, 0) + 1
    return counts


def _estimate_turnout_confidence(
    main_detections: list[SliceDetection],
    branch_detections: list[SliceDetection],
    main_world: np.ndarray,
    branch_world: np.ndarray,
) -> float:
    if len(main_detections) < 4 or len(branch_detections) < 4 or len(main_world) < 2 or len(branch_world) < 2:
        return 0.0
    main_confidence = _estimate_track_confidence(main_detections, main_world, np.vstack([d.rail_points_local for d in main_detections]))
    branch_confidence = _estimate_track_confidence(
        branch_detections,
        branch_world,
        np.vstack([d.rail_points_local for d in branch_detections]),
    )
    branch_end_gap = float(np.linalg.norm(main_world[-1, :2] - branch_world[-1, :2]))
    divergence_score = min(1.0, branch_end_gap / 1.5)
    return float(max(0.0, min(1.0, 0.4 * main_confidence + 0.4 * branch_confidence + 0.2 * divergence_score)))


def _orient_branch_from_switch(main_local: np.ndarray, branch_local: np.ndarray) -> np.ndarray:
    if len(main_local) == 0 or len(branch_local) < 2:
        return branch_local
    first_distance = float(np.min(np.linalg.norm(main_local[:, :2] - branch_local[0, :2], axis=1)))
    last_distance = float(np.min(np.linalg.norm(main_local[:, :2] - branch_local[-1, :2], axis=1)))
    if last_distance < first_distance:
        return branch_local[::-1]
    return branch_local


def _should_use_advanced_las(
    config: dict[str, Any],
    source_format: str,
    intensity: np.ndarray | None,
) -> bool:
    advanced = config.get("advanced_las", {})
    source_format_lower = source_format.lower()
    is_las_input = source_format_lower in {".las", ".laz"} or ".las" in source_format_lower or ".laz" in source_format_lower
    return bool(advanced.get("enabled", True)) and intensity is not None and is_las_input


def _build_roi_mask(
    points: np.ndarray,
    roi: dict[str, float | None],
    oriented_roi: dict[str, Any] | None = None,
) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    axis_map = {"x": 0, "y": 1, "z": 2}
    for axis, index in axis_map.items():
        lower = roi.get(f"{axis}_min")
        upper = roi.get(f"{axis}_max")
        if lower is not None:
            mask &= points[:, index] >= float(lower)
        if upper is not None:
            mask &= points[:, index] <= float(upper)
    if _oriented_roi_enabled(oriented_roi):
        mask &= _build_oriented_roi_mask(points, oriented_roi)
    return mask


def _apply_roi(points: np.ndarray, roi: dict[str, float | None]) -> np.ndarray:
    return points[_build_roi_mask(points, roi)]


def _build_analysis_frame(points: np.ndarray, oriented_roi: dict[str, Any] | None) -> tuple[LocalFrame, str]:
    if _oriented_roi_enabled(oriented_roi):
        return _build_frame_from_oriented_roi(points, oriented_roi), "oriented_roi"
    return estimate_local_frame(points), "pca"


def _oriented_roi_enabled(oriented_roi: dict[str, Any] | None) -> bool:
    if not isinstance(oriented_roi, dict) or oriented_roi.get("enabled") is False:
        return False
    if oriented_roi.get("origin") is None or oriented_roi.get("axis_s") is None:
        return False
    return any(oriented_roi.get(key) is not None for key in ("s_min", "s_max", "t_min", "t_max", "z_min", "z_max"))


def _build_oriented_roi_mask(points: np.ndarray, oriented_roi: dict[str, Any]) -> np.ndarray:
    origin = _as_xy_vector(oriented_roi.get("origin"), "oriented_roi.origin")
    axis_s, axis_t = _oriented_roi_axes(oriented_roi)

    xy = points[:, :2] - origin
    local_s = xy @ axis_s
    local_t = xy @ axis_t
    mask = np.ones(len(points), dtype=bool)
    for coordinate, values in (("s", local_s), ("t", local_t), ("z", points[:, 2])):
        lower = oriented_roi.get(f"{coordinate}_min")
        upper = oriented_roi.get(f"{coordinate}_max")
        if lower is not None:
            mask &= values >= float(lower)
        if upper is not None:
            mask &= values <= float(upper)
    return mask


def _build_frame_from_oriented_roi(points: np.ndarray, oriented_roi: dict[str, Any]) -> LocalFrame:
    origin_xy = _as_xy_vector(oriented_roi.get("origin"), "oriented_roi.origin")
    axis_s, axis_t = _oriented_roi_axes(oriented_roi)
    origin = np.array([origin_xy[0], origin_xy[1], float(np.median(points[:, 2]))], dtype=float)
    return LocalFrame(origin=origin, rotation=np.vstack([axis_s, axis_t]))


def _oriented_roi_axes(oriented_roi: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axis_s = _unit_xy_vector(oriented_roi.get("axis_s"), "oriented_roi.axis_s")
    raw_axis_t = oriented_roi.get("axis_t")
    if raw_axis_t is None:
        return axis_s, np.array([-axis_s[1], axis_s[0]], dtype=float)

    axis_t = _as_xy_vector(raw_axis_t, "oriented_roi.axis_t")
    axis_t = axis_t - float(axis_t @ axis_s) * axis_s
    norm = float(np.linalg.norm(axis_t))
    if norm <= 1e-9:
        raise ValueError("oriented_roi.axis_t must not be parallel to oriented_roi.axis_s.")
    return axis_s, axis_t / norm


def _as_xy_vector(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{label} must contain at least x and y values.")
    return array[:2]


def _unit_xy_vector(value: Any, label: str) -> np.ndarray:
    vector = _as_xy_vector(value, label)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{label} must not be a zero vector.")
    return vector / norm


def _extract_track_corridor_points(
    points: np.ndarray,
    intensity: np.ndarray | None,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    if intensity is None or len(points) < int(config["advanced_las"]["min_component_points"]):
        return None

    component = _select_best_candidate_component(points, intensity, config)
    if component is None:
        return None

    _, component_points, frame, component_local, _, _ = component
    low_q = float(config["advanced_las"]["corridor_quantile_low"])
    high_q = float(config["advanced_las"]["corridor_quantile_high"])
    margin = float(config["advanced_las"]["corridor_margin"])
    t_low, t_high = np.quantile(component_local[:, 1], [low_q, high_q])

    full_local = world_to_local(points, frame)
    corridor_mask = (full_local[:, 1] >= float(t_low) - margin) & (full_local[:, 1] <= float(t_high) + margin)
    corridor_points = points[corridor_mask]
    if len(corridor_points) < 50:
        return None
    return corridor_points, component_points


def _select_best_candidate_component(
    points: np.ndarray,
    intensity: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, np.ndarray, LocalFrame, np.ndarray, int, float] | None:
    advanced = config["advanced_las"]
    sample_max = int(advanced["sample_max_points"])
    if len(points) > sample_max:
        sample_indices = np.linspace(0, len(points) - 1, sample_max, dtype=np.int64)
        sampled_points = points[sample_indices]
        sampled_intensity = intensity[sample_indices]
    else:
        sampled_points = points
        sampled_intensity = intensity

    candidate_points = _build_rail_like_candidates(sampled_points, sampled_intensity, advanced)
    if len(candidate_points) < int(advanced["min_component_points"]):
        return None

    components = _split_candidate_components(candidate_points, advanced)
    best: tuple[float, np.ndarray, LocalFrame, np.ndarray, int, float] | None = None
    for component_points in components:
        if len(component_points) < int(advanced["min_component_points"]):
            continue
        frame = estimate_local_frame(component_points)
        local = world_to_local(component_points, frame)
        detections = _detect_slices(local, config)
        if len(detections) == 0:
            continue

        xy = component_points[:, :2]
        centered = xy - xy.mean(axis=0)
        covariance = np.cov(centered.T)
        eigenvalues, _ = np.linalg.eigh(covariance)
        elongation = float(eigenvalues.max() / max(eigenvalues.min(), 1e-6))
        score = float(len(detections) * 10000.0 + elongation + len(component_points) / 100000.0)
        candidate = (score, component_points, frame, local, len(detections), elongation)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def _build_rail_like_candidates(
    points: np.ndarray,
    intensity: np.ndarray,
    advanced: dict[str, Any],
) -> np.ndarray:
    cell_size = float(advanced["ground_cell_size"])
    percentile = float(advanced["ground_percentile"])
    cell_keys = _build_grid_keys(points[:, :2], cell_size)

    z_values = points[:, 2]
    unique_keys = np.unique(cell_keys)
    order = np.argsort(cell_keys, kind="mergesort")
    sorted_keys = cell_keys[order]
    sorted_z = z_values[order]
    boundaries = np.concatenate([[0], np.flatnonzero(np.diff(sorted_keys)) + 1, [len(sorted_keys)]])

    ground = np.empty(len(unique_keys), dtype=np.float32)
    for idx in range(len(unique_keys)):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        ground[idx] = np.quantile(sorted_z[start:end], percentile)

    lookup = np.empty_like(cell_keys, dtype=np.float32)
    lookup_indices = np.searchsorted(unique_keys, cell_keys)
    lookup[:] = ground[lookup_indices]
    relative_z = z_values - lookup

    intensity_limit = float(np.quantile(intensity, float(advanced["intensity_quantile_max"])))
    mask = (
        (relative_z >= float(advanced["rail_height_min"]))
        & (relative_z <= float(advanced["rail_height_max"]))
        & (intensity <= intensity_limit)
    )
    return points[mask]


def _split_candidate_components(points: np.ndarray, advanced: dict[str, Any]) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    cell_size = float(advanced["component_cell_size"])
    occupancy_threshold = int(advanced["occupancy_threshold"])
    grid_x = np.floor((points[:, 0] - points[:, 0].min()) / cell_size).astype(np.int32)
    grid_y = np.floor((points[:, 1] - points[:, 1].min()) / cell_size).astype(np.int32)

    occupancy = np.zeros((int(grid_y.max()) + 1, int(grid_x.max()) + 1), dtype=np.uint16)
    for gx, gy in zip(grid_x, grid_y):
        occupancy[gy, gx] = min(int(occupancy[gy, gx]) + 1, np.iinfo(np.uint16).max)

    labels, component_count = ndimage.label(occupancy >= occupancy_threshold, structure=np.ones((3, 3), dtype=int))
    point_labels = labels[grid_y, grid_x]

    components: list[np.ndarray] = []
    for component_id in range(1, component_count + 1):
        component_points = points[point_labels == component_id]
        if len(component_points) > 0:
            components.append(component_points)
    return components


def _build_grid_keys(xy: np.ndarray, cell_size: float) -> np.ndarray:
    grid_x = np.floor((xy[:, 0] - xy[:, 0].min()) / cell_size).astype(np.int64)
    grid_y = np.floor((xy[:, 1] - xy[:, 1].min()) / cell_size).astype(np.int64)
    return (grid_x << 32) + grid_y


def _apply_global_height_filter(points: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if not config.get("enabled", True):
        return points
    keep_ratio = float(config.get("keep_top_percent", 0.55))
    keep_ratio = min(max(keep_ratio, 0.05), 1.0)
    threshold = np.quantile(points[:, 2], 1.0 - keep_ratio)
    return points[points[:, 2] >= threshold]


def _manual_anchor_points_to_local(anchor_config: Any, frame: LocalFrame) -> np.ndarray | None:
    if not isinstance(anchor_config, dict) or not bool(anchor_config.get("enabled", False)):
        return None
    points = anchor_config.get("points", [])
    if not isinstance(points, list) or len(points) < 2:
        return None

    world_points: list[list[float]] = []
    fallback_z = float(frame.origin[2])
    for point in points:
        if isinstance(point, dict):
            x_value = point.get("x")
            y_value = point.get("y")
            z_value = point.get("z", fallback_z)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x_value = point[0]
            y_value = point[1]
            z_value = point[2] if len(point) >= 3 else fallback_z
        else:
            continue
        try:
            world_points.append([float(x_value), float(y_value), float(z_value)])
        except (TypeError, ValueError):
            continue

    if len(world_points) < 2:
        return None

    local_points = world_to_local(np.asarray(world_points, dtype=float), frame)
    order = np.argsort(local_points[:, 0])
    local_points = local_points[order]
    unique_points = [local_points[0]]
    for point in local_points[1:]:
        if abs(float(point[0] - unique_points[-1][0])) > 1e-6:
            unique_points.append(point)
    if len(unique_points) < 2:
        return None
    return np.vstack(unique_points)


def _expected_anchor_t_at_s(anchor_points_local: np.ndarray | None, s_value: float) -> float | None:
    if anchor_points_local is None or len(anchor_points_local) < 2:
        return None
    s_values = anchor_points_local[:, 0]
    if s_value < float(s_values[0]) or s_value > float(s_values[-1]):
        return None
    return float(np.interp(s_value, s_values, anchor_points_local[:, 1]))


def _anchor_lateral_distance_too_far(
    center_t: float,
    expected_anchor_t: float | None,
    config: dict[str, Any],
) -> bool:
    if expected_anchor_t is None:
        return False
    anchor_config = config.get("manual_anchor", {})
    snap_distance = float(anchor_config.get("snap_distance", 2.5))
    return abs(float(center_t) - expected_anchor_t) > max(snap_distance, 1e-6)


def _anchor_score_bonus(anchor_distance: float, config: dict[str, Any]) -> float:
    anchor_config = config.get("manual_anchor", {})
    snap_distance = max(float(anchor_config.get("snap_distance", 2.5)), 1e-6)
    score_weight = float(anchor_config.get("score_weight", 14.0))
    return max(0.0, 1.0 - anchor_distance / snap_distance) * score_weight


def _anchor_detection_reward(
    detection: SliceDetection,
    anchor_points_local: np.ndarray | None,
    config: dict[str, Any],
) -> float | None:
    expected_anchor_t = _expected_anchor_t_at_s(anchor_points_local, float(detection.center_local[0]))
    if expected_anchor_t is None:
        return 0.0
    center_t = float(detection.center_local[1])
    if _anchor_lateral_distance_too_far(center_t, expected_anchor_t, config):
        return None
    return _anchor_score_bonus(abs(center_t - expected_anchor_t), config)


def _detect_slices(
    local_points: np.ndarray,
    config: dict[str, Any],
    anchor_points_local: np.ndarray | None = None,
) -> list[SliceDetection]:
    s_values = local_points[:, 0]
    s_min = float(s_values.min())
    s_max = float(s_values.max())
    slice_length = float(config["slice_length"])
    min_points = int(config["min_points_per_slice"])
    detections: list[SliceDetection] = []
    previous_center_t: float | None = None

    edges = np.arange(s_min, s_max + slice_length, slice_length)
    if len(edges) < 2:
        return detections

    for start, end in zip(edges[:-1], edges[1:]):
        slice_mask = (s_values >= start) & (s_values < end)
        slice_points = local_points[slice_mask]
        if len(slice_points) < min_points:
            continue

        expected_anchor_t = _expected_anchor_t_at_s(anchor_points_local, float(0.5 * (start + end)))
        detection = _detect_slice(slice_points, start, end, config, previous_center_t, expected_anchor_t)
        if detection is None:
            continue
        detections.append(detection)
        previous_center_t = float(detection.center_local[1])

    return detections


def _detect_turnout_paths(
    local_points: np.ndarray,
    config: dict[str, Any],
    main_anchor_points_local: np.ndarray | None = None,
    branch_anchor_points_local: np.ndarray | None = None,
) -> tuple[list[SliceDetection], list[SliceDetection], np.ndarray]:
    s_values = local_points[:, 0]
    s_min = float(s_values.min())
    s_max = float(s_values.max())
    slice_length = float(config["slice_length"])
    min_points = int(config["min_points_per_slice"])
    turnout_config = config.get("turnout", {})
    branch_min_separation = float(turnout_config.get("branch_min_separation", 0.45))
    max_candidates = int(turnout_config.get("trace_max_candidates_per_slice", 8))
    candidate_groups: list[list[tuple[float, SliceDetection]]] = []

    edges = np.arange(s_min, s_max + slice_length, slice_length)
    if len(edges) < 2:
        raise RuntimeError("Turnout ROI is too short for slice detection.")

    for start, end in zip(edges[:-1], edges[1:]):
        slice_mask = (s_values >= start) & (s_values < end)
        slice_points = local_points[slice_mask]
        if len(slice_points) < min_points:
            continue

        candidates = _detect_slice_candidates(slice_points, start, end, config)
        if not candidates:
            candidate_groups.append([])
            continue
        candidate_groups.append(candidates[:max_candidates])

    main_paths = _trace_candidate_paths(
        candidate_groups,
        config,
        max_paths=16,
        anchor_points_local=main_anchor_points_local,
    )
    if not main_paths:
        raise RuntimeError("Unable to trace a continuous main turnout path.")

    main_path = main_paths[0]
    branch_groups = _build_branch_candidate_groups(candidate_groups, main_path, branch_min_separation)
    branch_paths = _trace_candidate_paths(
        branch_groups,
        config,
        max_paths=96,
        anchor_points_local=branch_anchor_points_local,
    )
    branch_path = _select_branch_path(branch_paths, main_path, config)
    if branch_path is None:
        raise RuntimeError("Unable to trace both main and branch turnout paths.")

    main_detections = main_path.detections
    branch_detections = _prepend_switch_detection(main_path, branch_path)
    switch_local = branch_detections[0].center_local
    return main_detections, branch_detections, switch_local


def _trace_candidate_paths(
    candidate_groups: list[list[tuple[float, SliceDetection]]],
    config: dict[str, Any],
    max_paths: int,
    anchor_points_local: np.ndarray | None = None,
) -> list[_CandidatePath]:
    if not candidate_groups:
        return []

    turnout_config = config.get("turnout", {})
    slice_length = float(config["slice_length"])
    min_path_points = int(turnout_config.get("trace_min_path_points", 6))
    score_weight = float(turnout_config.get("trace_score_weight", 1.0))
    length_weight = float(turnout_config.get("trace_length_weight", 2.0))

    normalized_scores = _normalize_candidate_scores(candidate_groups)
    best_scores: dict[tuple[int, int], float] = {}
    previous_keys: dict[tuple[int, int], tuple[int, int] | None] = {}

    for group_index, candidates in enumerate(candidate_groups):
        for candidate_index, (_raw_score, detection) in enumerate(candidates):
            key = (group_index, candidate_index)
            anchor_reward = _anchor_detection_reward(detection, anchor_points_local, config)
            if anchor_reward is None:
                continue
            node_reward = length_weight + normalized_scores.get(key, 0.0) * score_weight
            node_reward += anchor_reward
            best_score = node_reward
            best_previous: tuple[int, int] | None = None

            for previous_group_index in range(group_index - 1, -1, -1):
                previous_candidates = candidate_groups[previous_group_index]
                if not previous_candidates:
                    continue
                previous_s = float(previous_candidates[0][1].center_local[0])
                current_s = float(detection.center_local[0])
                if current_s - previous_s > float(turnout_config.get("trace_max_gap_s", 3.0)):
                    break
                for previous_candidate_index, (_previous_raw_score, previous_detection) in enumerate(previous_candidates):
                    previous_key = (previous_group_index, previous_candidate_index)
                    if previous_key not in best_scores:
                        continue
                    transition_penalty = _candidate_transition_penalty(
                        previous_detection,
                        detection,
                        slice_length,
                        config,
                    )
                    if transition_penalty is None:
                        continue
                    score = best_scores[previous_key] + node_reward - transition_penalty
                    if score > best_score:
                        best_score = score
                        best_previous = previous_key

            best_scores[key] = best_score
            previous_keys[key] = best_previous

    paths: list[_CandidatePath] = []
    for key in sorted(best_scores, key=lambda item: best_scores[item], reverse=True):
        path = _reconstruct_candidate_path(key, candidate_groups, best_scores, previous_keys)
        if len(path.detections) < min_path_points:
            continue
        if _is_similar_candidate_path(path, paths):
            continue
        paths.append(path)
        if len(paths) >= max_paths:
            break
    return paths


def _normalize_candidate_scores(
    candidate_groups: list[list[tuple[float, SliceDetection]]],
) -> dict[tuple[int, int], float]:
    normalized: dict[tuple[int, int], float] = {}
    for group_index, candidates in enumerate(candidate_groups):
        if not candidates:
            continue
        scores = np.array([float(score) for score, _detection in candidates], dtype=float)
        score_min = float(scores.min())
        score_range = float(scores.max() - score_min)
        for candidate_index, score in enumerate(scores):
            if score_range <= 1e-9:
                normalized[(group_index, candidate_index)] = 0.5
            else:
                normalized[(group_index, candidate_index)] = float((score - score_min) / score_range)
    return normalized


def _candidate_transition_penalty(
    previous_detection: SliceDetection,
    detection: SliceDetection,
    slice_length: float,
    config: dict[str, Any],
) -> float | None:
    turnout_config = config.get("turnout", {})
    previous_center = previous_detection.center_local
    center = detection.center_local
    ds = float(center[0] - previous_center[0])
    if ds <= 1e-6:
        return None
    if ds > float(turnout_config.get("trace_max_gap_s", 3.0)):
        return None

    dt = abs(float(center[1] - previous_center[1]))
    max_dt = max(0.35, float(turnout_config.get("trace_max_lateral_jump_per_m", 0.55)) * ds + 0.10)
    if dt > max_dt:
        return None

    dz = abs(float(center[2] - previous_center[2]))
    gap = max(0.0, ds - slice_length)
    return (
        float(turnout_config.get("trace_lateral_jump_weight", 8.0)) * dt
        + float(turnout_config.get("trace_gap_weight", 0.9)) * gap
        + float(turnout_config.get("trace_z_jump_weight", 1.5)) * dz
    )


def _reconstruct_candidate_path(
    end_key: tuple[int, int],
    candidate_groups: list[list[tuple[float, SliceDetection]]],
    best_scores: dict[tuple[int, int], float],
    previous_keys: dict[tuple[int, int], tuple[int, int] | None],
) -> _CandidatePath:
    keys: list[tuple[int, int]] = []
    key: tuple[int, int] | None = end_key
    while key is not None:
        keys.append(key)
        key = previous_keys.get(key)
    keys.reverse()
    detections = [candidate_groups[group_index][candidate_index][1] for group_index, candidate_index in keys]
    slice_indices = [group_index for group_index, _candidate_index in keys]
    return _CandidatePath(detections=detections, slice_indices=slice_indices, score=float(best_scores[end_key]))


def _is_similar_candidate_path(path: _CandidatePath, accepted_paths: list[_CandidatePath]) -> bool:
    path_by_slice = {
        slice_index: float(detection.center_local[1])
        for slice_index, detection in zip(path.slice_indices, path.detections)
    }
    for accepted in accepted_paths:
        accepted_by_slice = {
            slice_index: float(detection.center_local[1])
            for slice_index, detection in zip(accepted.slice_indices, accepted.detections)
        }
        common_slices = sorted(set(path_by_slice).intersection(accepted_by_slice))
        if len(common_slices) < 3:
            continue
        differences = [abs(path_by_slice[index] - accepted_by_slice[index]) for index in common_slices]
        if float(np.median(differences)) < 0.20:
            return True
    return False


def _build_branch_candidate_groups(
    candidate_groups: list[list[tuple[float, SliceDetection]]],
    main_path: _CandidatePath,
    branch_min_separation: float,
) -> list[list[tuple[float, SliceDetection]]]:
    main_s = np.array([float(detection.center_local[0]) for detection in main_path.detections], dtype=float)
    main_t = np.array([float(detection.center_local[1]) for detection in main_path.detections], dtype=float)
    order = np.argsort(main_s)
    main_s = main_s[order]
    main_t = main_t[order]

    branch_groups: list[list[tuple[float, SliceDetection]]] = []
    for candidates in candidate_groups:
        if not candidates:
            branch_groups.append([])
            continue
        slice_s = float(candidates[0][1].center_local[0])
        reference_t = float(np.interp(slice_s, main_s, main_t))
        branch_candidates = [
            (score, detection)
            for score, detection in candidates
            if abs(float(detection.center_local[1]) - reference_t) >= branch_min_separation
        ]
        branch_groups.append(branch_candidates)
    return branch_groups


def _select_branch_path(
    branch_paths: list[_CandidatePath],
    main_path: _CandidatePath,
    config: dict[str, Any],
) -> _CandidatePath | None:
    if not branch_paths:
        return None

    turnout_config = config.get("turnout", {})
    branch_min_separation = float(turnout_config.get("branch_min_separation", 0.45))
    min_branch_length = float(turnout_config.get("trace_min_branch_length_m", 8.0))
    start_max_separation = float(turnout_config.get("trace_branch_start_max_separation", 1.75))
    best_path: _CandidatePath | None = None
    best_score = -np.inf

    for path in branch_paths:
        length = _candidate_path_length(path)
        if length < min_branch_length:
            continue
        signed_separations = _path_separations_from_main(path, main_path)
        if len(signed_separations) < 3:
            continue
        separations = np.abs(signed_separations)
        max_separation = float(separations.max(initial=0.0))
        if max_separation < branch_min_separation:
            continue

        start_window = max(1, min(8, len(separations) // 5 if len(separations) >= 10 else len(separations)))
        near_start_separation = float(separations[:start_window].min())
        if near_start_separation > start_max_separation:
            continue

        start_separation = float(separations[0])
        separation_growth = max_separation - start_separation
        divergence_bonus = min(max(separation_growth, 0.0), 1.5) * 4.0
        sustained_bonus = min(max_separation, 2.5)
        start_proximity_bonus = max(0.0, start_max_separation - near_start_separation) * 3.0
        already_parallel_penalty = 0.0
        if start_separation > branch_min_separation + 0.75 and separation_growth < 0.35:
            already_parallel_penalty = 12.0
        score = (
            path.score
            + divergence_bonus
            + sustained_bonus
            + start_proximity_bonus
            + min(length, 60.0) * 0.05
            - already_parallel_penalty
        )
        if score > best_score:
            best_score = score
            best_path = path

    return best_path


def _candidate_path_length(path: _CandidatePath) -> float:
    if len(path.detections) < 2:
        return 0.0
    s_values = np.array([float(detection.center_local[0]) for detection in path.detections], dtype=float)
    return float(s_values.max() - s_values.min())


def _path_separations_from_main(path: _CandidatePath, main_path: _CandidatePath) -> np.ndarray:
    main_s = np.array([float(detection.center_local[0]) for detection in main_path.detections], dtype=float)
    main_t = np.array([float(detection.center_local[1]) for detection in main_path.detections], dtype=float)
    order = np.argsort(main_s)
    main_s = main_s[order]
    main_t = main_t[order]
    separations = []
    for detection in path.detections:
        s_value = float(detection.center_local[0])
        t_value = float(detection.center_local[1])
        separations.append(t_value - float(np.interp(s_value, main_s, main_t)))
    return np.array(separations, dtype=float)


def _anchor_branch_start_to_main(
    main_local: np.ndarray,
    branch_local: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    if len(main_local) == 0 or len(branch_local) == 0:
        return branch_local

    anchored = branch_local.copy()
    main_order = np.argsort(main_local[:, 0])
    main_sorted = main_local[main_order]
    switch_s = float(anchored[0, 0])
    if len(anchored) >= 2:
        s_deltas = np.diff(anchored[:, 0])
        nonzero_deltas = s_deltas[np.abs(s_deltas) > 1e-6]
        if len(nonzero_deltas):
            branch_direction = float(np.sign(nonzero_deltas[0]))
            step = float(np.median(np.abs(nonzero_deltas)))
            if abs(float(anchored[1, 0]) - switch_s) < max(0.25, step * 0.5):
                switch_s = float(anchored[1, 0] - branch_direction * step)
                anchored[0, 0] = switch_s

    reference_t, reference_z = _interpolate_main_tz(main_sorted, switch_s)
    first_gap = abs(float(anchored[1, 1] if len(anchored) >= 2 else anchored[0, 1]) - reference_t)
    max_anchor_gap = float(config.get("turnout", {}).get("trace_branch_anchor_max_separation", 1.0))
    if first_gap > max_anchor_gap:
        return branch_local

    anchored[0, 1] = reference_t
    anchored[0, 2] = reference_z
    return anchored


def _estimate_switch_on_main(main_local: np.ndarray, branch_local: np.ndarray) -> np.ndarray:
    if len(main_local) == 0:
        return branch_local[0] if len(branch_local) else np.zeros(3, dtype=float)
    main_order = np.argsort(main_local[:, 0])
    main_sorted = main_local[main_order]
    if len(branch_local) == 0:
        switch_s = float(main_sorted[0, 0])
    else:
        switch_s = float(branch_local[0, 0])
    reference_t, reference_z = _interpolate_main_tz(main_sorted, switch_s)
    return np.array([switch_s, reference_t, reference_z], dtype=float)


def _interpolate_main_tz(main_sorted: np.ndarray, s_value: float) -> tuple[float, float]:
    if s_value < float(main_sorted[0, 0]) or s_value > float(main_sorted[-1, 0]):
        nearest_index = int(np.argmin(np.abs(main_sorted[:, 0] - s_value)))
        return float(main_sorted[nearest_index, 1]), float(main_sorted[nearest_index, 2])

    return (
        float(np.interp(s_value, main_sorted[:, 0], main_sorted[:, 1])),
        float(np.interp(s_value, main_sorted[:, 0], main_sorted[:, 2])),
    )


def _prepend_switch_detection(main_path: _CandidatePath, branch_path: _CandidatePath) -> list[SliceDetection]:
    if not branch_path.detections:
        return []
    branch_start_s = float(branch_path.detections[0].center_local[0])
    before_switch = [
        detection
        for detection in main_path.detections
        if float(detection.center_local[0]) < branch_start_s - 1e-6
    ]
    if before_switch:
        switch_detection = max(before_switch, key=lambda item: float(item.center_local[0]))
    else:
        switch_detection = min(main_path.detections, key=lambda item: abs(float(item.center_local[0]) - branch_start_s))

    if np.allclose(switch_detection.center_local, branch_path.detections[0].center_local):
        return branch_path.detections
    return [switch_detection, *branch_path.detections]


def _choose_continuous_detection(
    candidates: list[tuple[float, SliceDetection]],
    previous_t: float | None,
) -> SliceDetection | None:
    if not candidates:
        return None
    if previous_t is None:
        return candidates[0][1]
    return min(candidates, key=lambda item: abs(float(item[1].center_local[1]) - previous_t))[1]


def _detect_slice_candidates(
    slice_points: np.ndarray,
    start: float,
    end: float,
    config: dict[str, Any],
) -> list[tuple[float, SliceDetection]]:
    top_points = _keep_top_slice_points(slice_points)
    if len(top_points) < int(config["min_points_per_slice"]) // 2:
        return []

    t_values = top_points[:, 1]
    configured_bins = int(config["peak_search_bins"])
    bin_count = max(12, min(configured_bins, int(np.sqrt(len(top_points)) * 3)))
    counts, bin_edges = np.histogram(t_values, bins=bin_count)
    if counts.max(initial=0) == 0:
        return []

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    smoothed = _smooth_histogram(counts.astype(float))
    prominence = max(0.15, float(smoothed.max()) * float(config["peak_min_prominence_ratio"]))
    peak_indices = _find_histogram_peak_indices(smoothed, prominence)
    peak_positions, peak_scores = _merge_dense_histogram_peaks(centers, counts, smoothed, peak_indices)

    candidates: list[tuple[float, SliceDetection]] = []
    if len(peak_positions) < 2:
        fallback_pair = _fallback_pair_from_quantiles(top_points[:, 1], config)
        if fallback_pair is None:
            return []
        detection = _build_detection_from_peak_pair(
            top_points,
            start,
            end,
            fallback_pair[0],
            fallback_pair[1],
            config,
            pair_mode="quantile_fallback",
        )
        return [(0.0, detection)] if detection is not None else []

    spacing_min = float(config["rail_pair_spacing_min"])
    spacing_max = float(config["rail_pair_spacing_max"])
    target_spacing = float(config.get("rail_pair_spacing_target", (spacing_min + spacing_max) / 2.0))
    radius = float(config["peak_window_radius"])
    slice_center_t = float(np.median(top_points[:, 1]))

    for left_index in range(len(peak_positions)):
        for right_index in range(left_index + 1, len(peak_positions)):
            left_t = float(peak_positions[left_index])
            right_t = float(peak_positions[right_index])
            spacing = right_t - left_t
            if spacing < spacing_min or spacing > spacing_max:
                continue
            left_points = top_points[np.abs(top_points[:, 1] - left_t) <= radius]
            right_points = top_points[np.abs(top_points[:, 1] - right_t) <= radius]
            if len(left_points) < 5 or len(right_points) < 5:
                left_points = top_points[np.abs(top_points[:, 1] - left_t) <= radius * 1.5]
                right_points = top_points[np.abs(top_points[:, 1] - right_t) <= radius * 1.5]
            if len(left_points) < 5 or len(right_points) < 5:
                continue
            detection = _build_detection_from_points(
                left_points,
                right_points,
                start,
                end,
                left_t,
                right_t,
                pair_mode="dual",
            )
            score = _score_peak_pair(
                left_t=left_t,
                right_t=right_t,
                left_score=float(peak_scores[left_index]),
                right_score=float(peak_scores[right_index]),
                left_points=left_points,
                right_points=right_points,
                spacing=spacing,
                target_spacing=target_spacing,
                slice_center_t=slice_center_t,
            )
            candidates.append((score, detection))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:6]


def _detect_slice(
    slice_points: np.ndarray,
    start: float,
    end: float,
    config: dict[str, Any],
    previous_center_t: float | None,
    expected_anchor_t: float | None = None,
) -> SliceDetection | None:
    top_points = _keep_top_slice_points(slice_points)
    if len(top_points) < int(config["min_points_per_slice"]) // 2:
        return None

    t_values = top_points[:, 1]
    configured_bins = int(config["peak_search_bins"])
    bin_count = max(12, min(configured_bins, int(np.sqrt(len(top_points)) * 3)))
    counts, bin_edges = np.histogram(t_values, bins=bin_count)
    if counts.max(initial=0) == 0:
        return None

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    smoothed = _smooth_histogram(counts.astype(float))
    prominence = max(0.15, float(smoothed.max()) * float(config["peak_min_prominence_ratio"]))
    peak_indices = _find_histogram_peak_indices(smoothed, prominence)
    peak_positions, peak_scores = _merge_dense_histogram_peaks(centers, counts, smoothed, peak_indices)

    if len(peak_positions) < 2:
        fallback_pair = _fallback_pair_from_quantiles(top_points[:, 1], config)
        if fallback_pair is not None:
            detection = _build_detection_from_peak_pair(
                top_points,
                start,
                end,
                fallback_pair[0],
                fallback_pair[1],
                config,
                pair_mode="quantile_fallback",
            )
            if detection is not None and not _anchor_lateral_distance_too_far(
                float(detection.center_local[1]), expected_anchor_t, config
            ):
                return detection
        single_t = _select_single_peak_t(peak_positions, peak_scores, _reference_center_t(previous_center_t, expected_anchor_t))
        if single_t is None:
            return None
        return _build_detection_from_single_peak(
            top_points,
            start,
            end,
            single_t,
            config,
            previous_center_t=previous_center_t,
            expected_anchor_t=expected_anchor_t,
        )
    else:
        pair = _choose_peak_pair(
            peak_positions,
            peak_scores,
            top_points,
            config,
            previous_center_t,
            expected_anchor_t,
        )
        if pair is None:
            fallback_pair = _fallback_pair_from_quantiles(top_points[:, 1], config)
            if fallback_pair is not None:
                detection = _build_detection_from_peak_pair(
                    top_points,
                    start,
                    end,
                    fallback_pair[0],
                    fallback_pair[1],
                    config,
                    pair_mode="quantile_fallback",
                )
                if detection is not None and not _anchor_lateral_distance_too_far(
                    float(detection.center_local[1]), expected_anchor_t, config
                ):
                    return detection
            single_t = _select_single_peak_t(
                peak_positions,
                peak_scores,
                _reference_center_t(previous_center_t, expected_anchor_t),
            )
            if single_t is None:
                return None
            return _build_detection_from_single_peak(
                top_points,
                start,
                end,
                single_t,
                config,
                previous_center_t=previous_center_t,
                expected_anchor_t=expected_anchor_t,
            )
        else:
            left_t, right_t = pair
    detection = _build_detection_from_peak_pair(top_points, start, end, left_t, right_t, config, pair_mode="dual")
    if detection is not None and not _anchor_lateral_distance_too_far(float(detection.center_local[1]), expected_anchor_t, config):
        return detection

    single_t = _select_single_peak_t(peak_positions, peak_scores, _reference_center_t(previous_center_t, expected_anchor_t))
    if single_t is None:
        return None
    return _build_detection_from_single_peak(
        top_points,
        start,
        end,
        single_t,
        config,
        previous_center_t=previous_center_t,
        expected_anchor_t=expected_anchor_t,
    )


def _reference_center_t(previous_center_t: float | None, expected_anchor_t: float | None) -> float | None:
    if expected_anchor_t is not None:
        return expected_anchor_t
    return previous_center_t


def _select_single_peak_t(
    peak_positions: np.ndarray,
    peak_scores: np.ndarray,
    reference_t: float | None,
) -> float | None:
    if len(peak_positions) == 0:
        return None
    if reference_t is None:
        return None
    scores = peak_scores.astype(float).copy()
    scores -= np.abs(peak_positions.astype(float) - reference_t) * 3.0
    return float(peak_positions[int(np.argmax(scores))])


def _build_detection_from_single_peak(
    top_points: np.ndarray,
    start: float,
    end: float,
    observed_t: float,
    config: dict[str, Any],
    previous_center_t: float | None,
    expected_anchor_t: float | None = None,
) -> SliceDetection | None:
    reference_t = _reference_center_t(previous_center_t, expected_anchor_t)
    if reference_t is None:
        return None

    spacing_min = float(config["rail_pair_spacing_min"])
    spacing_max = float(config["rail_pair_spacing_max"])
    target_spacing = float(config.get("rail_pair_spacing_target", (spacing_min + spacing_max) / 2.0))
    if target_spacing < spacing_min or target_spacing > spacing_max:
        target_spacing = (spacing_min + spacing_max) / 2.0

    offset_from_reference = abs(float(observed_t) - reference_t)
    if offset_from_reference < spacing_min * 0.35 or offset_from_reference > spacing_max * 0.75:
        return None

    if float(observed_t) <= reference_t:
        left_t = float(observed_t)
        right_t = left_t + target_spacing
        pair_mode = "single_left"
    else:
        right_t = float(observed_t)
        left_t = right_t - target_spacing
        pair_mode = "single_right"
    center_t = float((left_t + right_t) / 2.0)

    if previous_center_t is not None:
        max_center_shift = max(0.45, target_spacing * 0.35)
        if abs(center_t - previous_center_t) > max_center_shift:
            return None
    if _anchor_lateral_distance_too_far(center_t, expected_anchor_t, config):
        return None

    radius = float(config["peak_window_radius"])
    observed_points = top_points[np.abs(top_points[:, 1] - float(observed_t)) <= radius]
    if len(observed_points) < 5:
        observed_points = top_points[np.abs(top_points[:, 1] - float(observed_t)) <= radius * 1.5]
    if len(observed_points) < 5:
        return None

    center_s = float(0.5 * (start + end))
    center_z = _top_surface_height(observed_points)
    return SliceDetection(
        center_local=np.array([center_s, center_t, center_z], dtype=float),
        rail_points_local=observed_points,
        left_peak_t=left_t,
        right_peak_t=right_t,
        pair_mode=pair_mode,
    )


def _build_detection_from_peak_pair(
    top_points: np.ndarray,
    start: float,
    end: float,
    left_t: float,
    right_t: float,
    config: dict[str, Any],
    pair_mode: str = "dual",
) -> SliceDetection | None:
    radius = float(config["peak_window_radius"])
    left_points = top_points[np.abs(top_points[:, 1] - left_t) <= radius]
    right_points = top_points[np.abs(top_points[:, 1] - right_t) <= radius]
    if len(left_points) < 5 or len(right_points) < 5:
        left_points = top_points[np.abs(top_points[:, 1] - left_t) <= radius * 1.5]
        right_points = top_points[np.abs(top_points[:, 1] - right_t) <= radius * 1.5]
    if len(left_points) < 5 or len(right_points) < 5:
        return None
    return _build_detection_from_points(left_points, right_points, start, end, left_t, right_t, pair_mode=pair_mode)


def _build_detection_from_points(
    left_points: np.ndarray,
    right_points: np.ndarray,
    start: float,
    end: float,
    left_t: float,
    right_t: float,
    pair_mode: str = "dual",
) -> SliceDetection:
    rail_points = np.vstack([left_points, right_points])
    center_s = float(0.5 * (start + end))
    center_t = float((left_t + right_t) / 2.0)
    center_z = float((_top_surface_height(left_points) + _top_surface_height(right_points)) / 2.0)
    return SliceDetection(
        center_local=np.array([center_s, center_t, center_z], dtype=float),
        rail_points_local=rail_points,
        left_peak_t=float(left_t),
        right_peak_t=float(right_t),
        pair_mode=pair_mode,
    )


def _keep_top_slice_points(slice_points: np.ndarray) -> np.ndarray:
    if len(slice_points) < 8:
        return slice_points
    z_values = slice_points[:, 2]
    target_min = max(12, len(slice_points) // 2)
    for quantile in (0.55, 0.45, 0.35):
        threshold = np.quantile(z_values, quantile)
        top_points = slice_points[z_values >= threshold]
        if len(top_points) >= target_min:
            return top_points
    return slice_points


def _smooth_histogram(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def _find_histogram_peak_indices(values: np.ndarray, prominence: float) -> np.ndarray:
    peak_indices, _ = find_peaks(values, prominence=prominence)
    edge_indices: list[int] = []
    if len(values) >= 2:
        if values[0] >= prominence and values[0] >= values[1]:
            edge_indices.append(0)
        if values[-1] >= prominence and values[-1] >= values[-2]:
            edge_indices.append(len(values) - 1)
    if not edge_indices:
        return peak_indices
    return np.unique(np.concatenate([peak_indices, np.array(edge_indices, dtype=int)]))


def _merge_dense_histogram_peaks(
    centers: np.ndarray,
    counts: np.ndarray,
    smoothed: np.ndarray,
    peak_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dense_threshold = max(2.0, float(counts.max(initial=0)) * 0.35)
    dense_mask = counts.astype(float) >= dense_threshold
    dense_positions: list[float] = []
    dense_scores: list[float] = []
    start: int | None = None
    for index, is_dense in enumerate(dense_mask):
        if is_dense and start is None:
            start = index
        if start is not None and (not is_dense or index == len(dense_mask) - 1):
            end = index + 1 if is_dense and index == len(dense_mask) - 1 else index
            weights = counts[start:end].astype(float)
            if weights.sum() > 0:
                dense_positions.append(float(np.average(centers[start:end], weights=weights)))
                dense_scores.append(float(smoothed[start:end].max(initial=0.0)))
            start = None

    positions = [float(centers[index]) for index in peak_indices] + dense_positions
    scores = [float(smoothed[index]) for index in peak_indices] + dense_scores
    if not positions:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(positions)
    merged_positions: list[float] = []
    merged_scores: list[float] = []
    for index in order:
        position = positions[int(index)]
        score = scores[int(index)]
        if merged_positions and abs(position - merged_positions[-1]) < 0.12:
            if score > merged_scores[-1]:
                merged_positions[-1] = position
                merged_scores[-1] = score
        else:
            merged_positions.append(position)
            merged_scores.append(score)
    return np.array(merged_positions, dtype=float), np.array(merged_scores, dtype=float)


def _choose_peak_pair(
    peak_positions: np.ndarray,
    peak_scores: np.ndarray,
    slice_points: np.ndarray,
    config: dict[str, Any],
    previous_center_t: float | None,
    expected_anchor_t: float | None = None,
) -> tuple[float, float] | None:
    spacing_min = float(config["rail_pair_spacing_min"])
    spacing_max = float(config["rail_pair_spacing_max"])
    target_spacing = float(config.get("rail_pair_spacing_target", (spacing_min + spacing_max) / 2.0))
    radius = float(config["peak_window_radius"])
    slice_center_t = float(np.median(slice_points[:, 1]))
    best_pair: tuple[float, float] | None = None
    best_score = -np.inf

    for left_index in range(len(peak_positions)):
        for right_index in range(left_index + 1, len(peak_positions)):
            left_t = float(peak_positions[left_index])
            right_t = float(peak_positions[right_index])
            spacing = right_t - left_t
            if spacing < spacing_min or spacing > spacing_max:
                continue
            left_points = slice_points[np.abs(slice_points[:, 1] - left_t) <= radius]
            right_points = slice_points[np.abs(slice_points[:, 1] - right_t) <= radius]
            if len(left_points) == 0 or len(right_points) == 0:
                continue

            score = _score_peak_pair(
                left_t=left_t,
                right_t=right_t,
                left_score=float(peak_scores[left_index]),
                right_score=float(peak_scores[right_index]),
                left_points=left_points,
                right_points=right_points,
                spacing=spacing,
                target_spacing=target_spacing,
                slice_center_t=slice_center_t,
            )
            if previous_center_t is not None:
                center_t = (left_t + right_t) / 2.0
                score -= abs(center_t - previous_center_t) * 8.0
            if expected_anchor_t is not None:
                center_t = (left_t + right_t) / 2.0
                anchor_distance = abs(center_t - expected_anchor_t)
                if _anchor_lateral_distance_too_far(center_t, expected_anchor_t, config):
                    continue
                score += _anchor_score_bonus(anchor_distance, config)
            if score > best_score:
                best_score = score
                best_pair = (left_t, right_t)

    return best_pair


def _score_peak_pair(
    left_t: float,
    right_t: float,
    left_score: float,
    right_score: float,
    left_points: np.ndarray,
    right_points: np.ndarray,
    spacing: float,
    target_spacing: float,
    slice_center_t: float,
) -> float:
    pair_center = (left_t + right_t) / 2.0
    spacing_score = max(0.0, 1.0 - abs(spacing - target_spacing) / max(target_spacing, 1e-6))
    count_balance = min(len(left_points), len(right_points)) / max(len(left_points), len(right_points))
    height_delta = abs(_top_surface_height(left_points) - _top_surface_height(right_points))
    height_score = max(0.0, 1.0 - height_delta / 0.25)
    outer_preference = abs(pair_center - slice_center_t)
    inner_guard_penalty = max(0.0, 1.0 - spacing / max(target_spacing, 1e-6)) * 3.0
    return (
        left_score
        + right_score
        + spacing_score * 8.0
        + count_balance * 4.0
        + height_score * 3.0
        + outer_preference * 1.5
        - inner_guard_penalty
    )


def _top_surface_height(points: np.ndarray) -> float:
    return float(np.quantile(points[:, 2], 0.8))


def _fallback_pair_from_quantiles(
    t_values: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, float] | None:
    left_t = float(np.quantile(t_values, 0.2))
    right_t = float(np.quantile(t_values, 0.8))
    spacing = right_t - left_t
    if float(config["rail_pair_spacing_min"]) <= spacing <= float(config["rail_pair_spacing_max"]):
        return left_t, right_t
    return None


def _build_centerline_local(detections: list[SliceDetection], config: dict[str, Any]) -> np.ndarray:
    centers = np.vstack([item.center_local for item in detections])
    centers = centers[np.argsort(centers[:, 0])]
    centers = _drop_centerline_outliers(centers)
    if len(centers) < 4:
        raise RuntimeError("Too few centerline candidates remain after filtering. Unable to smooth.")

    window = min(ensure_odd(int(config["savgol_window"])), len(centers))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return centers

    polyorder = min(int(config["savgol_polyorder"]), window - 1)
    smoothed_t = savgol_filter(centers[:, 1], window_length=window, polyorder=polyorder, mode="interp")
    smoothed_z = savgol_filter(centers[:, 2], window_length=window, polyorder=polyorder, mode="interp")
    centerline = np.column_stack([centers[:, 0], smoothed_t, smoothed_z])
    return _apply_xy_constraint(centerline, config)


def _apply_xy_constraint(centerline: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    xy_config = config.get("xy_constraint", {})
    mode = str(xy_config.get("mode", "free")).lower()
    if mode in {"free", "none", ""} or len(centerline) < 4:
        return centerline

    constrained = centerline.copy()
    if mode == "smooth":
        constrained[:, 1] = _smooth_lateral_t(
            centerline[:, 1],
            window=int(xy_config.get("smooth_window", 21)),
        )
        return constrained

    if mode == "straight":
        constrained[:, 1] = _fit_straight_lateral_t(
            centerline[:, 0],
            centerline[:, 1],
            trim_ratio=float(xy_config.get("straight_trim_ratio", 0.10)),
        )
        return constrained

    return centerline


def _smooth_lateral_t(t_values: np.ndarray, window: int) -> np.ndarray:
    window = min(ensure_odd(window), len(t_values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return t_values
    polyorder = min(2, window - 1)
    return savgol_filter(t_values, window_length=window, polyorder=polyorder, mode="interp")


def _fit_straight_lateral_t(s_values: np.ndarray, t_values: np.ndarray, trim_ratio: float) -> np.ndarray:
    if len(s_values) < 2 or float(np.ptp(s_values)) <= 1e-9:
        return np.full_like(t_values, float(np.median(t_values)))

    coefficients = np.polyfit(s_values, t_values, 1)
    fitted = np.polyval(coefficients, s_values)
    residuals = np.abs(t_values - fitted)
    trim_ratio = min(max(trim_ratio, 0.0), 0.45)
    if trim_ratio > 0.0 and len(t_values) >= 6:
        threshold = np.quantile(residuals, 1.0 - trim_ratio)
        keep_mask = residuals <= threshold
        if int(keep_mask.sum()) >= 2:
            coefficients = np.polyfit(s_values[keep_mask], t_values[keep_mask], 1)
            fitted = np.polyval(coefficients, s_values)
    return fitted.astype(float)


def _drop_centerline_outliers(points: np.ndarray) -> np.ndarray:
    if len(points) < 5:
        return points
    t_values = points[:, 1]
    z_values = points[:, 2]
    t_median = np.median(t_values)
    z_median = np.median(z_values)
    t_mad = np.median(np.abs(t_values - t_median)) + 1e-6
    z_mad = np.median(np.abs(z_values - z_median)) + 1e-6
    mask = (np.abs(t_values - t_median) <= 4.0 * t_mad) & (np.abs(z_values - z_median) <= 4.0 * z_mad)
    filtered = points[mask]
    return filtered if len(filtered) >= 4 else points
