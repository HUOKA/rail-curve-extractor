from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import build_deeplab_topology_centerline_network as topo


DEFAULT_BASE_NETWORK = Path(
    "output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_review_v12_experiment/deeplab_topology_centerline_network.geojson"
)
DEFAULT_BASE_EVIDENCE = Path(
    "output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_review_v12_experiment/deeplab_topology_evidence.geojson"
)
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_review_v14_r350_landmark_probe")
DEFAULT_EPSG = 32651


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a visible TA08 R350 landmark probe for QGIS review.")
    parser.add_argument("--base-network", type=Path, default=DEFAULT_BASE_NETWORK)
    parser.add_argument("--base-evidence", type=Path, default=DEFAULT_BASE_EVIDENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-radius-m", type=float, default=350.0)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_network = load_features(args.base_network.expanduser().resolve())
    evidence_features = load_features(args.base_evidence.expanduser().resolve())

    result = build_probe_features(
        base_network,
        evidence_features=evidence_features,
        target_radius_m=args.target_radius_m,
    )
    final_features = result["features"]
    landmarks = result["landmarks"]

    network_geojson = out_dir / "deeplab_topology_centerline_network.geojson"
    evidence_geojson = out_dir / "deeplab_topology_evidence.geojson"
    landmark_geojson = out_dir / "ta08_landmark_proxies.geojson"
    summary_path = out_dir / "summary.json"
    review_path = out_dir / "REVIEW.md"

    topo.write_geojson(network_geojson, final_features, epsg=args.epsg)
    topo.write_centerline_shapefile(final_features, network_geojson.with_suffix(".shp"), epsg=args.epsg)
    topo.write_centerline_qml(network_geojson.with_suffix(".qml"))
    topo.write_geojson(evidence_geojson, evidence_features, epsg=args.epsg)
    topo.write_evidence_shapefile(evidence_features, evidence_geojson.with_suffix(".shp"), epsg=args.epsg)
    topo.write_evidence_qml(evidence_geojson.with_suffix(".qml"))
    write_point_geojson(landmark_geojson, landmarks, epsg=args.epsg)
    write_landmark_shapefile(landmarks, landmark_geojson.with_suffix(".shp"), epsg=args.epsg)

    summary = {
        "mode": "ta08_r350_landmark_probe_v1",
        "warning": "This is a landmark-visible geometry probe, not a recommended replacement baseline.",
        "policy": (
            "Expose inferred switch/frog proxy points and force a visible R350 arc from the switch-tip proxy "
            "to the frog-front proxy. The proxy points are not hardware labels from a dedicated detector."
        ),
        "base_network": str(args.base_network.expanduser().resolve()),
        "target_radius_m": args.target_radius_m,
        "outputs": {
            "network_geojson": str(network_geojson),
            "network_shp": str(network_geojson.with_suffix(".shp")),
            "network_qml": str(network_geojson.with_suffix(".qml")),
            "evidence_geojson": str(evidence_geojson),
            "evidence_shp": str(evidence_geojson.with_suffix(".shp")),
            "landmark_geojson": str(landmark_geojson),
            "landmark_shp": str(landmark_geojson.with_suffix(".shp")),
            "summary_json": str(summary_path),
            "review_md": str(review_path),
        },
        "feature_count": len(final_features),
        "landmarks": [feature.get("properties", {}) for feature in landmarks],
        "report": result["report"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(review_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("features") or [])


def build_probe_features(
    base_features: list[dict[str, Any]],
    *,
    evidence_features: list[dict[str, Any]],
    target_radius_m: float,
) -> dict[str, Any]:
    mainline = find_feature(base_features, "BAND_mainline_2_track_0")
    ta08 = find_feature(base_features, "TURNOUT_TA08")
    if mainline is None or ta08 is None:
        raise ValueError("Base network must contain BAND_mainline_2_track_0 and TURNOUT_TA08.")

    guide = topo.LinearGuide(topo.line_coords(mainline)[0], topo.line_coords(mainline)[-1])
    old_coords = topo.line_coords(ta08)
    old_st = topo.station_offsets_for_coords(old_coords, guide=guide)
    props = dict(ta08.get("properties") or {})

    switch_s = float(props["switch_curve_s1_m"])
    frog_s = float(props["straight_middle_s0_m"])
    switch_t = topo.interpolated_offset(old_st, switch_s)
    frog_t = topo.interpolated_offset(old_st, frog_s)
    switch_xy = guide.point_at(switch_s, switch_t)
    frog_xy = guide.point_at(frog_s, frog_t)

    arc_xy, arc_report = choose_tangent_r350_arc(
        switch_xy,
        frog_xy,
        guide=guide,
        switch_s=switch_s,
        target_radius_m=target_radius_m,
        samples=130,
    )
    arc_st = topo.station_offsets_for_coords(arc_xy, guide=guide)
    arc_part = topo.resample_station_offsets(arc_st, step_m=0.75)

    prefix_points = [(s, t) for s, t in old_st if s <= frog_s]
    prefix_points.append((frog_s, frog_t))
    merged_st = topo.merge_station_offset_parts([prefix_points, arc_part])
    new_coords = [
        [round(x, 6), round(y, 6)]
        for x, y in (guide.point_at(station, offset) for station, offset in merged_st)
    ]
    old_support = topo.measure_support(
        old_coords,
        evidence_segments=topo.build_segments(evidence_features),
        threshold_m=0.85,
        sample_step_m=5.0,
    )
    new_support = topo.measure_support(
        [(float(x), float(y)) for x, y in new_coords],
        evidence_segments=topo.build_segments(evidence_features),
        threshold_m=0.85,
        sample_step_m=5.0,
    )
    displacement = displacement_summary(old_st, merged_st, guide=guide)
    old_angle = max_local_angle_deg(old_coords)
    new_angle = max_local_angle_deg([(float(x), float(y)) for x, y in new_coords])

    new_props = dict(props)
    stations = [s for s, _ in merged_st]
    offsets = [t for _, t in merged_st]
    new_props.update(
        {
            "geom_kind": "ta08_r350_switch_to_frog_proxy_probe",
            "source_layer": "r350_landmark_probe",
            "source_type": "r350_landmark_probe",
            "postprocess_policy": "r350_landmark_visible_probe",
            "risk_flag": "review_priority_low_support_turnout",
            "qa_status": "self_review_needs_visual_check",
            "length_m": round(topo.polyline_length([(float(x), float(y)) for x, y in new_coords]), 3),
            "station_min_m": round(min(stations), 3),
            "station_max_m": round(max(stations), 3),
            "offset_min_m": round(min(offsets), 3),
            "offset_max_m": round(max(offsets), 3),
            "deeplab_support_ratio": new_support["support_ratio"],
            "deeplab_mean_distance_m": new_support["mean_distance_m"],
            "deeplab_max_unsupported_gap_m": new_support["max_unsupported_gap_m"],
            "deeplab_sample_count": new_support["sample_count"],
            "experiment": "r350_switch_tip_to_frog_front_proxy_v1",
            "r350_target_m": round(target_radius_m, 3),
            "r350_proxy_s0_m": round(frog_s, 3),
            "r350_proxy_s1_m": round(switch_s, 3),
            "self_note": "probe only: R350 forced between inferred switch-tip and frog-front proxies; inspect landmark layer first",
            "review_note": "experimental visible R350 probe; proxy landmarks are not a hardware detector result",
        }
    )
    rebuilt = {"type": "Feature", "properties": new_props, "geometry": {"type": "LineString", "coordinates": new_coords}}

    final_features = [
        feature
        for feature in base_features
        if str((feature.get("properties") or {}).get("line_id", "")) != "TURNOUT_TA08"
    ]
    final_features.append(rebuilt)

    landmarks = build_landmarks(
        guide=guide,
        old_st=old_st,
        props=props,
        target_radius_m=target_radius_m,
        arc_report=arc_report,
    )
    report = {
        "line_id": "TURNOUT_TA08",
        "status": "applied_visible_probe",
        "target_radius_m": round(target_radius_m, 3),
        "proxy_switch_tip_s_m": round(switch_s, 3),
        "proxy_frog_front_s_m": round(frog_s, 3),
        "proxy_switch_tip_xy": [round(switch_xy[0], 3), round(switch_xy[1], 3)],
        "proxy_frog_front_xy": [round(frog_xy[0], 3), round(frog_xy[1], 3)],
        "old_support": old_support,
        "new_support": new_support,
        "old_max_local_angle_deg": round(old_angle, 3),
        "new_max_local_angle_deg": round(new_angle, 3),
        "displacement_from_base_m": displacement,
        "arc_selection": arc_report,
    }
    return {"features": final_features, "landmarks": landmarks, "report": report}


def choose_tangent_r350_arc(
    switch_xy: tuple[float, float],
    frog_xy: tuple[float, float],
    *,
    guide: topo.LinearGuide,
    switch_s: float,
    target_radius_m: float,
    samples: int,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    options = circle_arc_options(switch_xy, frog_xy, target_radius_m, samples=samples)
    if not options:
        raise ValueError("Switch/frog proxy chord is longer than the requested circle diameter.")
    scored: list[tuple[float, dict[str, Any], list[tuple[float, float]]]] = []
    for sign, center, points in options:
        st = topo.station_offsets_for_coords(points, guide=guide)
        slope = local_slope(st, switch_s - 1.0)
        score = abs(slope)
        scored.append(
            (
                score,
                {
                    "circle_side": sign,
                    "circle_center_xy": [round(center[0], 3), round(center[1], 3)],
                    "switch_tangent_slope": round(slope, 6),
                    "selection_rule": "minimum absolute station-offset slope near switch proxy",
                },
                points,
            )
        )
    scored.sort(key=lambda item: item[0])
    _, report, points = scored[0]
    return points, report


def circle_arc_options(
    start: tuple[float, float],
    end: tuple[float, float],
    radius_m: float,
    *,
    samples: int,
) -> list[tuple[int, tuple[float, float], list[tuple[float, float]]]]:
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    chord = math.hypot(dx, dy)
    if chord <= 1e-9 or chord > 2.0 * radius_m:
        return []
    mx = (ax + bx) / 2.0
    my = (ay + by) / 2.0
    h = math.sqrt(max(0.0, radius_m * radius_m - (chord / 2.0) ** 2))
    nx = -dy / chord
    ny = dx / chord
    arcs: list[tuple[int, tuple[float, float], list[tuple[float, float]]]] = []
    for sign in (1, -1):
        center = (mx + sign * h * nx, my + sign * h * ny)
        start_angle = math.atan2(ay - center[1], ax - center[0])
        end_angle = math.atan2(by - center[1], bx - center[0])
        delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
        points = [
            (
                center[0] + radius_m * math.cos(start_angle + delta * index / (samples - 1)),
                center[1] + radius_m * math.sin(start_angle + delta * index / (samples - 1)),
            )
            for index in range(samples)
        ]
        arcs.append((sign, center, points))
    return arcs


def local_slope(points: list[tuple[float, float]], station: float) -> float:
    left_s = max(points[0][0], station - 1.0)
    right_s = min(points[-1][0], station + 1.0)
    if right_s <= left_s:
        return 0.0
    return (topo.interpolated_offset(points, right_s) - topo.interpolated_offset(points, left_s)) / (right_s - left_s)


def displacement_summary(
    old_st: list[tuple[float, float]],
    new_st: list[tuple[float, float]],
    *,
    guide: topo.LinearGuide,
) -> dict[str, float]:
    s0 = max(old_st[0][0], new_st[0][0])
    s1 = min(old_st[-1][0], new_st[-1][0])
    if s1 <= s0:
        return {"sample_count": 0, "mean_m": 0.0, "max_m": 0.0}
    count = max(2, int(math.ceil(s1 - s0)) + 1)
    distances: list[float] = []
    max_distance = 0.0
    max_station = s0
    for index in range(count):
        station = s0 + (s1 - s0) * index / (count - 1)
        old_xy = guide.point_at(station, topo.interpolated_offset(old_st, station))
        new_xy = guide.point_at(station, topo.interpolated_offset(new_st, station))
        dist = math.hypot(new_xy[0] - old_xy[0], new_xy[1] - old_xy[1])
        distances.append(dist)
        if dist > max_distance:
            max_distance = dist
            max_station = station
    return {
        "sample_count": len(distances),
        "mean_m": round(sum(distances) / len(distances), 3),
        "max_m": round(max_distance, 3),
        "max_station_m": round(max_station, 3),
    }


def max_local_angle_deg(coords: list[tuple[float, float]]) -> float:
    values: list[float] = []
    for left, center, right in zip(coords, coords[1:], coords[2:]):
        ax = center[0] - left[0]
        ay = center[1] - left[1]
        bx = right[0] - center[0]
        by = right[1] - center[1]
        la = math.hypot(ax, ay)
        lb = math.hypot(bx, by)
        if la <= 1e-9 or lb <= 1e-9:
            continue
        cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        values.append(math.degrees(math.acos(cosine)))
    return max(values, default=0.0)


def build_landmarks(
    *,
    guide: topo.LinearGuide,
    old_st: list[tuple[float, float]],
    props: dict[str, Any],
    target_radius_m: float,
    arc_report: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        ("SWITCH_TIP_PROXY", "inferred_switch_tip_proxy", float(props["switch_curve_s1_m"])),
        ("FROG_FRONT_PROXY", "inferred_frog_front_proxy", float(props["straight_middle_s0_m"])),
        ("SWITCH_CURVE_START", "geometry_breakpoint_not_hardware_label", float(props["switch_curve_s0_m"])),
        ("OUTER_CURVE_START", "geometry_breakpoint_not_hardware_label", float(props["outer_curve_s0_m"])),
        ("PARALLEL_START", "geometry_breakpoint_not_hardware_label", float(props["parallel_s0_m"])),
    ]
    landmarks: list[dict[str, Any]] = []
    for label, kind, station in rows:
        offset = topo.interpolated_offset(old_st, station)
        x, y = guide.point_at(station, offset)
        landmarks.append(
            {
                "type": "Feature",
                "properties": {
                    "landmark": label,
                    "kind": kind,
                    "station_m": round(station, 3),
                    "offset_m": round(offset, 3),
                    "target_r_m": round(target_radius_m, 3),
                    "circle_side": arc_report.get("circle_side", 0),
                    "confidence": "proxy_from_current_geometry",
                    "note": "not a confirmed hardware semantic label",
                },
                "geometry": {"type": "Point", "coordinates": [round(x, 6), round(y, 6)]},
            }
        )
    return landmarks


def find_feature(features: list[dict[str, Any]], line_id: str) -> dict[str, Any] | None:
    for feature in features:
        if str((feature.get("properties") or {}).get("line_id", "")) == line_id:
            return feature
    return None


def write_point_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_landmark_shapefile(features: list[dict[str, Any]], path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(path), shapeType=shapefile.POINT, encoding="utf-8")
    writer.field("landmark", "C", size=32)
    writer.field("kind", "C", size=48)
    writer.field("station_m", "F", decimal=3)
    writer.field("offset_m", "F", decimal=3)
    writer.field("target_r", "F", decimal=3)
    writer.field("side", "N", size=4)
    writer.field("confidence", "C", size=48)
    writer.field("note", "C", size=96)
    for feature in features:
        props = feature.get("properties") or {}
        x, y = feature["geometry"]["coordinates"]
        writer.point(float(x), float(y))
        writer.record(
            str(props.get("landmark", ""))[:32],
            str(props.get("kind", ""))[:48],
            topo.safe_float(props.get("station_m", 0.0)),
            topo.safe_float(props.get("offset_m", 0.0)),
            topo.safe_float(props.get("target_r_m", 0.0)),
            int(topo.safe_float(props.get("circle_side", 0))),
            str(props.get("confidence", ""))[:48],
            str(props.get("note", ""))[:96],
        )
    writer.close()
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    topo.write_projection(path.with_suffix(".prj"), epsg)


def write_review(path: Path, summary: dict[str, Any]) -> None:
    report = summary["report"]
    lines = [
        "# TA08 R350 Landmark Probe",
        "",
        "This is a visible geometry probe, not a replacement baseline.",
        "",
        "## Outputs",
        "",
        f"- Network Shapefile: `{summary['outputs']['network_shp']}`",
        f"- Landmark proxy Shapefile: `{summary['outputs']['landmark_shp']}`",
        f"- Evidence Shapefile: `{summary['outputs']['evidence_shp']}`",
        "",
        "## Proxy Landmarks",
        "",
        f"- Switch-tip proxy: station `{report['proxy_switch_tip_s_m']}` at `{report['proxy_switch_tip_xy']}`.",
        f"- Frog-front proxy: station `{report['proxy_frog_front_s_m']}` at `{report['proxy_frog_front_xy']}`.",
        "- These are inferred from the current geometry breakpoints, not detected hardware labels.",
        "",
        "## Metrics",
        "",
        f"- Base support: `{report['old_support']}`",
        f"- Probe support: `{report['new_support']}`",
        f"- Base max local angle: `{report['old_max_local_angle_deg']} deg`",
        f"- Probe max local angle: `{report['new_max_local_angle_deg']} deg`",
        f"- Displacement from base: `{report['displacement_from_base_m']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
