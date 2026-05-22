from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import rasterio.crs


DEFAULT_OUT_DIR = Path("output/dom_centerline_strict_auto_v1")
DEFAULT_EPSG = 32651


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build QA artifacts for a strict-auto centerline delivery.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    parser.add_argument("--support-review-threshold", type=float, default=0.70)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    delivery_dir = out_dir / "final_delivery"
    delivery_dir.mkdir(parents=True, exist_ok=True)

    network_path = delivery_dir / "centerline_2d.geojson"
    if not network_path.exists():
        network_path = out_dir / "10_centerline_2d" / "deeplab_topology_centerline_network.geojson"
    if not network_path.exists():
        raise FileNotFoundError(network_path)

    features = load_features(network_path)
    z_summary = load_json(out_dir / "11_centerline_3d" / "summary.json")
    topology_summary = load_json(out_dir / "08_topology_centerline" / "summary.json")
    transition_summary = load_json(out_dir / "08_auto_turnout_crossover_evidence" / "summary.json")
    pipeline_summary = load_json(out_dir / "pipeline_summary.json")
    z_reports = {str(item.get("line_id", "")): item for item in z_summary.get("line_reports", [])}

    targets = build_review_targets(
        features,
        z_reports=z_reports,
        support_review_threshold=args.support_review_threshold,
    )
    target_geojson = delivery_dir / "strict_auto_review_targets.geojson"
    target_shp = delivery_dir / "strict_auto_review_targets.shp"
    report_path = delivery_dir / "strict_auto_QA.md"
    summary_path = delivery_dir / "strict_auto_QA_summary.json"

    write_geojson_points(target_geojson, targets, epsg=args.epsg)
    write_target_shapefile(target_shp, targets, epsg=args.epsg)
    report = build_report(
        features,
        targets=targets,
        z_summary=z_summary,
        topology_summary=topology_summary,
        transition_summary=transition_summary,
        pipeline_summary=pipeline_summary,
        out_dir=out_dir,
    )
    report_path.write_text(report, encoding="utf-8")
    qa_summary = {
        "mode": "strict_auto_centerline_delivery_qa",
        "out_dir": str(out_dir),
        "feature_count": len(features),
        "role_counts": dict(Counter(feature_role(feature) for feature in features)),
        "review_target_count": len(targets),
        "review_target_counts": dict(Counter(target["properties"]["priority"] for target in targets)),
        "outputs": {
            "report_md": str(report_path),
            "review_targets_geojson": str(target_geojson),
            "review_targets_shp": str(target_shp),
        },
    }
    summary_path.write_text(json.dumps(qa_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa_summary, ensure_ascii=False, indent=2))
    return 0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_features(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    return list(payload.get("features") or [])


def build_review_targets(
    features: list[dict[str, Any]],
    *,
    z_reports: dict[str, dict[str, Any]],
    support_review_threshold: float,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for feature in features:
        props = feature.get("properties") or {}
        line_id = str(props.get("line_id", ""))
        role = feature_role(feature)
        support = safe_float(props.get("deeplab_support_ratio", 0.0))
        reasons: list[str] = []
        priority = 0

        if role == "turnout_connector":
            priority = max(priority, 3)
            reasons.append("auto_turnout_unreviewed")
        if role == "promoted_straight_track":
            priority = max(priority, 2)
            reasons.append("promoted_diagnostic_track")
        if "bridge" in role:
            priority = max(priority, 1)
            reasons.append(role)
        if support and support < support_review_threshold:
            priority = max(priority, 2)
            reasons.append("low_deeplab_support")

        z_report = z_reports.get(line_id, {})
        if str(z_report.get("z_bridge_mode", "")):
            priority = max(priority, 1)
            reasons.append(f"z_{z_report['z_bridge_mode']}")
        if safe_float(z_report.get("z_fallback_count", 0.0)) > 0:
            priority = max(priority, 3)
            reasons.append("z_fallback_used")

        if priority <= 0:
            continue
        coords = line_coords(feature)
        midpoint = point_at_fraction(coords, 0.5)
        if midpoint is None:
            continue
        target_props = {
            "target_id": f"QA_{len(targets) + 1:03d}",
            "priority": priority_label(priority),
            "reason": ";".join(dict.fromkeys(reasons)),
            "line_id": line_id,
            "network_role": role,
            "length_m": round(safe_float(props.get("length_m", polyline_length(coords))), 3),
            "station_min_m": props.get("station_min_m", props.get("s_min_m", None)),
            "station_max_m": props.get("station_max_m", props.get("s_max_m", None)),
            "deeplab_support_ratio": support,
            "deeplab_mean_distance_m": safe_float(props.get("deeplab_mean_distance_m", 0.0)),
            "z_source": z_report.get("z_source", ""),
            "z_bridge_mode": z_report.get("z_bridge_mode", ""),
            "z_outlier_count": int(safe_float(z_report.get("z_outlier_count", 0.0))),
        }
        targets.append(
            {
                "type": "Feature",
                "properties": target_props,
                "geometry": {"type": "Point", "coordinates": [midpoint[0], midpoint[1]]},
            }
        )
    return sorted(targets, key=target_sort_key)


def target_sort_key(target: dict[str, Any]) -> tuple[int, float, str]:
    props = target.get("properties") or {}
    rank = {"high": 0, "medium": 1, "low": 2}.get(str(props.get("priority", "")), 3)
    station = safe_float(props.get("station_min_m", 0.0))
    return rank, station, str(props.get("line_id", ""))


def build_report(
    features: list[dict[str, Any]],
    *,
    targets: list[dict[str, Any]],
    z_summary: dict[str, Any],
    topology_summary: dict[str, Any],
    transition_summary: dict[str, Any],
    pipeline_summary: dict[str, Any],
    out_dir: Path,
) -> str:
    role_counts = Counter(feature_role(feature) for feature in features)
    target_counts = Counter(str((target.get("properties") or {}).get("priority", "")) for target in targets)
    lines = [
        "# Strict Auto Centerline QA",
        "",
        "This QA package is derived from the current strict-auto delivery only. It does not add manual acceptance points or retained review constraints to the production result.",
        "",
        "## Delivery",
        "",
        f"- Pipeline status: `{pipeline_summary.get('status', 'unknown')}`",
        f"- 2D Shapefile: `{out_dir / 'final_delivery' / 'centerline_2d.shp'}`",
        f"- 3D Shapefile: `{out_dir / 'final_delivery' / 'centerline_3d.shp'}`",
        f"- Review targets: `{out_dir / 'final_delivery' / 'strict_auto_review_targets.shp'}`",
        "",
        "## Counts",
        "",
        f"- Centerline features: {len(features)}",
        f"- Z source: `{z_summary.get('selected_source', 'unknown')}`",
        f"- LAS coverage: {nested_get(z_summary, ['source_comparison', 'las', 'valid_ratio'], 'unknown')}",
        f"- DSM coverage: {nested_get(z_summary, ['source_comparison', 'dsm', 'valid_ratio'], 'unknown')}",
        f"- Auto transitions: {transition_summary.get('transition_count', 'unknown')}",
        f"- Auto turnout features: {transition_summary.get('turnout_feature_count', 'unknown')}",
        f"- Auto crossover features: {transition_summary.get('crossover_feature_count', 'unknown')}",
        "",
        "## Role Counts",
        "",
    ]
    for role, count in sorted(role_counts.items()):
        lines.append(f"- `{role}`: {count}")
    lines.extend(["", "## Manual Review Targets", ""])
    if not targets:
        lines.append("- No review targets were generated.")
    else:
        lines.append(f"- high: {target_counts.get('high', 0)}")
        lines.append(f"- medium: {target_counts.get('medium', 0)}")
        lines.append(f"- low: {target_counts.get('low', 0)}")
        lines.append("")
        for target in targets:
            props = target.get("properties") or {}
            lines.append(
                f"- `{props.get('target_id')}` `{props.get('priority')}` `{props.get('line_id')}`: "
                f"{props.get('reason')}; support={props.get('deeplab_support_ratio')}; "
                f"station={props.get('station_min_m')}..{props.get('station_max_m')}m"
            )
    lines.extend(["", "## Strict-Auto Safeguards", ""])
    lines.append(f"- Weak topology-only gap bridge max: {topology_summary.get('weak_gap_bridge_max_gap_m', 'unknown')}m")
    lines.append(f"- Bridge evidence support threshold: {topology_summary.get('bridge_evidence_support', 'unknown')}")
    lines.append("- Long weak same-band gaps are not kept in the strict-auto final delivery.")
    return "\n".join(lines) + "\n"


def nested_get(data: dict[str, Any], keys: list[str], default: Any) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def feature_role(feature: dict[str, Any]) -> str:
    props = feature.get("properties") or {}
    return str(props.get("network_role", props.get("net_role", "")))


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    coords = ((feature.get("geometry") or {}).get("coordinates") or [])
    if not coords:
        return []
    if isinstance(coords[0][0], (list, tuple)):
        coords = coords[0]
    return [(float(item[0]), float(item[1])) for item in coords]


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(distance(a, b) for a, b in zip(coords, coords[1:]))


def point_at_fraction(coords: list[tuple[float, float]], fraction: float) -> tuple[float, float] | None:
    if not coords:
        return None
    if len(coords) == 1:
        return coords[0]
    total = polyline_length(coords)
    if total <= 0:
        return coords[0]
    target = total * fraction
    walked = 0.0
    for left, right in zip(coords, coords[1:]):
        segment = distance(left, right)
        if walked + segment >= target:
            ratio = 0.0 if segment == 0 else (target - walked) / segment
            return (left[0] + (right[0] - left[0]) * ratio, left[1] + (right[1] - left[1]) * ratio)
        walked += segment
    return coords[-1]


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def priority_label(priority: int) -> str:
    if priority >= 3:
        return "high"
    if priority == 2:
        return "medium"
    return "low"


def write_geojson_points(path: Path, targets: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        "features": targets,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_target_shapefile(path: Path, targets: list[dict[str, Any]], *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(path), shapeType=shapefile.POINT, encoding="utf-8")
    writer.field("target_id", "C", size=16)
    writer.field("priority", "C", size=12)
    writer.field("reason", "C", size=120)
    writer.field("line_id", "C", size=96)
    writer.field("net_role", "C", size=32)
    writer.field("len_m", "F", decimal=3)
    writer.field("s0_m", "F", decimal=3)
    writer.field("s1_m", "F", decimal=3)
    writer.field("dl_sup", "F", decimal=4)
    writer.field("dl_mean", "F", decimal=4)
    writer.field("z_src", "C", size=24)
    writer.field("z_mode", "C", size=32)
    writer.field("z_out_n", "N", size=8)
    for target in targets:
        props = target.get("properties") or {}
        x, y = (target.get("geometry") or {}).get("coordinates", [0.0, 0.0])[:2]
        writer.point(float(x), float(y))
        writer.record(
            str(props.get("target_id", ""))[:16],
            str(props.get("priority", ""))[:12],
            str(props.get("reason", ""))[:120],
            str(props.get("line_id", ""))[:96],
            str(props.get("network_role", ""))[:32],
            safe_float(props.get("length_m", 0.0)),
            safe_float(props.get("station_min_m", 0.0)),
            safe_float(props.get("station_max_m", 0.0)),
            safe_float(props.get("deeplab_support_ratio", 0.0)),
            safe_float(props.get("deeplab_mean_distance_m", 0.0)),
            str(props.get("z_source", ""))[:24],
            str(props.get("z_bridge_mode", ""))[:32],
            int(safe_float(props.get("z_outlier_count", 0.0))),
        )
    writer.close()
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    path.with_suffix(".prj").write_text(rasterio.crs.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
