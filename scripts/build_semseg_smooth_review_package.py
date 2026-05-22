from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_deeplab_topology_centerline_network as topo


DEFAULT_BASE_NETWORK = Path(
    "output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_review_v15_semseg_radius/deeplab_topology_centerline_network.geojson"
)
DEFAULT_BASE_EVIDENCE = Path(
    "output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_review_v15_semseg_radius/deeplab_topology_evidence.geojson"
)
DEFAULT_CROSSOVER_EVIDENCE = Path(
    "output/raw_dom_roi_fullpass_v1/deeplab_gauge_pair_crossovers_v1/deeplab_gauge_pair_centerlines.geojson"
)
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_review_v19_crossover_tangent")
DEFAULT_EPSG = 32651


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build semseg-first review package with straight-band trimming and turnout smoothing.")
    parser.add_argument("--base-network", type=Path, default=DEFAULT_BASE_NETWORK)
    parser.add_argument("--base-evidence", type=Path, default=DEFAULT_BASE_EVIDENCE)
    parser.add_argument("--crossover-evidence", type=Path, default=DEFAULT_CROSSOVER_EVIDENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--support-threshold-m", type=float, default=0.85)
    parser.add_argument("--sample-step-m", type=float, default=5.0)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_features = load_features(args.base_network.expanduser().resolve())
    evidence_features = load_features(args.base_evidence.expanduser().resolve())
    crossover_evidence_path = args.crossover_evidence.expanduser().resolve()
    crossover_evidence_features = load_features(crossover_evidence_path) if crossover_evidence_path.exists() else []
    evidence_features = evidence_features + crossover_evidence_features
    evidence_segments = topo.build_segments(evidence_features)

    features, report = build_smoothed_features(
        base_features,
        evidence_features=evidence_features,
        evidence_segments=evidence_segments,
        support_threshold_m=args.support_threshold_m,
        sample_step_m=args.sample_step_m,
    )

    network_geojson = out_dir / "deeplab_topology_centerline_network.geojson"
    evidence_geojson = out_dir / "deeplab_topology_evidence.geojson"
    summary_path = out_dir / "summary.json"
    review_path = out_dir / "REVIEW.md"

    topo.write_geojson(network_geojson, features, epsg=args.epsg)
    topo.write_centerline_shapefile(features, network_geojson.with_suffix(".shp"), epsg=args.epsg)
    topo.write_centerline_qml(network_geojson.with_suffix(".qml"))
    topo.write_geojson(evidence_geojson, evidence_features, epsg=args.epsg)
    topo.write_evidence_shapefile(evidence_features, evidence_geojson.with_suffix(".shp"), epsg=args.epsg)
    topo.write_evidence_qml(evidence_geojson.with_suffix(".qml"))

    summary = {
        "mode": "semseg_first_trimmed_smoothed_review_v1",
        "policy": "Start from the accepted evidence-first network; trim straight-band endpoints inside turnout zones, smooth turnout polylines only when DeepLab support is preserved, rebuild crossover connectors from DeepLab gauge-pair boundary evidence with straight-track endpoint tangency when available, and add evidence-supported turnout boundary bridges where trimming exposes a supported gap.",
        "base_network": str(args.base_network.expanduser().resolve()),
        "base_evidence": str(args.base_evidence.expanduser().resolve()),
        "crossover_evidence": str(crossover_evidence_path) if crossover_evidence_features else None,
        "outputs": {
            "network_geojson": str(network_geojson),
            "network_shp": str(network_geojson.with_suffix(".shp")),
            "network_qml": str(network_geojson.with_suffix(".qml")),
            "evidence_geojson": str(evidence_geojson),
            "evidence_shp": str(evidence_geojson.with_suffix(".shp")),
            "summary_json": str(summary_path),
            "review_md": str(review_path),
        },
        "feature_count": len(features),
        "report": report,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(review_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("features") or [])


def build_smoothed_features(
    base_features: list[dict[str, Any]],
    *,
    evidence_features: list[dict[str, Any]],
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    band_features = [
        feature
        for feature in base_features
        if str((feature.get("properties") or {}).get("network_role", "")) in {"main_through_track", "parallel_straight_track"}
    ]
    turnout_features = [
        feature
        for feature in base_features
        if str((feature.get("properties") or {}).get("network_role", "")) == "turnout_connector"
    ]
    other_features = [
        feature
        for feature in base_features
        if str((feature.get("properties") or {}).get("network_role", ""))
        not in {"main_through_track", "parallel_straight_track", "turnout_connector", "turnout_boundary_bridge"}
    ]

    trimmed_bands = topo.trim_straight_band_endpoint_overlaps(band_features, turnout_features=turnout_features)
    trimmed_bands = [
        remeasure_feature(
            feature,
            evidence_segments=evidence_segments,
            support_threshold_m=support_threshold_m,
            sample_step_m=sample_step_m,
        )
        for feature in trimmed_bands
    ]
    merged = trimmed_bands + other_features + turnout_features
    smoothed = topo.smooth_turnout_connectors_with_evidence(
        merged,
        evidence_segments=evidence_segments,
        support_threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    smoothed = topo.rebuild_crossover_connectors_with_evidence(
        smoothed,
        evidence_features=evidence_features,
        evidence_segments=evidence_segments,
        support_threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    bridges = topo.build_turnout_boundary_evidence_bridges(
        smoothed,
        evidence_features=evidence_features,
        evidence_segments=evidence_segments,
        support_threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    output = sorted(smoothed + bridges, key=topo.feature_sort_key)
    return output, build_report(base_features, output)


def remeasure_feature(
    feature: dict[str, Any],
    *,
    evidence_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    support_threshold_m: float,
    sample_step_m: float,
) -> dict[str, Any]:
    coords = topo.line_coords(feature)
    support = topo.measure_support(
        coords,
        evidence_segments=evidence_segments,
        threshold_m=support_threshold_m,
        sample_step_m=sample_step_m,
    )
    props = dict(feature.get("properties") or {})
    props.update(
        {
            "length_m": round(topo.polyline_length(coords), 3),
            "deeplab_support_ratio": support["support_ratio"],
            "deeplab_mean_distance_m": support["mean_distance_m"],
            "deeplab_max_unsupported_gap_m": support["max_unsupported_gap_m"],
            "deeplab_sample_count": support["sample_count"],
        }
    )
    return {"type": "Feature", "properties": props, "geometry": feature["geometry"]}


def build_report(base_features: list[dict[str, Any]], output_features: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_id = {str((feature.get("properties") or {}).get("line_id", "")): feature for feature in base_features}
    output_by_id = {str((feature.get("properties") or {}).get("line_id", "")): feature for feature in output_features}
    trimmed: list[dict[str, Any]] = []
    smoothed: list[dict[str, Any]] = []
    for line_id, output in output_by_id.items():
        props = output.get("properties") or {}
        base = base_by_id.get(line_id)
        if base is not None:
            base_s0, base_s1 = topo.station_range(base)
            new_s0, new_s1 = topo.station_range(output)
            if abs(base_s0 - new_s0) > 1e-3 or abs(base_s1 - new_s1) > 1e-3:
                trimmed.append(
                    {
                        "line_id": line_id,
                        "old_station_min_m": round(base_s0, 3),
                        "old_station_max_m": round(base_s1, 3),
                        "new_station_min_m": round(new_s0, 3),
                        "new_station_max_m": round(new_s1, 3),
                    }
                )
        if props.get("turnout_smooth_status") == "accepted" and props.get("crossover_rebuild_status") != "accepted":
            smoothed.append(
                {
                    "line_id": line_id,
                    "old_angle_deg": props.get("turnout_smooth_old_angle"),
                    "new_angle_deg": props.get("turnout_smooth_new_angle"),
                    "support_ratio": props.get("deeplab_support_ratio"),
                    "mean_distance_m": props.get("deeplab_mean_distance_m"),
                }
            )
    boundary_bridges = [
        {
            "line_id": str((feature.get("properties") or {}).get("line_id", "")),
            "station_min_m": (feature.get("properties") or {}).get("station_min_m"),
            "station_max_m": (feature.get("properties") or {}).get("station_max_m"),
            "gap_m": (feature.get("properties") or {}).get("gap_m"),
            "offset_delta_m": (feature.get("properties") or {}).get("offset_delta_m"),
            "bridge_evidence": (feature.get("properties") or {}).get("bridge_evidence"),
            "support_ratio": (feature.get("properties") or {}).get("deeplab_support_ratio"),
            "mean_distance_m": (feature.get("properties") or {}).get("deeplab_mean_distance_m"),
        }
        for feature in output_features
        if str((feature.get("properties") or {}).get("network_role", "")) == "turnout_boundary_bridge"
    ]
    rebuilt_crossovers = [
        {
            "line_id": str((feature.get("properties") or {}).get("line_id", "")),
            "start_evidence": (feature.get("properties") or {}).get("crossover_start_evidence"),
            "end_evidence": (feature.get("properties") or {}).get("crossover_end_evidence"),
            "start_break_s": (feature.get("properties") or {}).get("crossover_start_break_s"),
            "end_break_s": (feature.get("properties") or {}).get("crossover_end_break_s"),
            "middle_len_m": (feature.get("properties") or {}).get("crossover_middle_len_m"),
            "rebuild_mode": (feature.get("properties") or {}).get("crossover_rebuild_mode"),
            "curve_fraction": (feature.get("properties") or {}).get("crossover_curve_fraction"),
            "endpoint_tangent_slope": (feature.get("properties") or {}).get("crossover_endpoint_tangent_slope"),
            "old_angle_deg": (feature.get("properties") or {}).get("crossover_old_angle_deg"),
            "new_angle_deg": (feature.get("properties") or {}).get("crossover_new_angle_deg"),
            "support_ratio": (feature.get("properties") or {}).get("deeplab_support_ratio"),
            "mean_distance_m": (feature.get("properties") or {}).get("deeplab_mean_distance_m"),
        }
        for feature in output_features
        if (feature.get("properties") or {}).get("crossover_rebuild_status") == "accepted"
    ]
    return {
        "trimmed_bands": trimmed,
        "smoothed_turnouts": smoothed,
        "boundary_bridges": boundary_bridges,
        "rebuilt_crossovers": rebuilt_crossovers,
    }


def write_review(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Semseg Smooth Review V19",
        "",
        "This package starts from the evidence-first network and applies guarded postprocessing: straight-band endpoint trimming inside turnout zones, evidence-constrained smoothing of turnout polylines, semseg evidence crossover rebuilds with straight-track endpoint tangency, and evidence-supported bridges at trimmed turnout boundaries.",
        "",
        "## Outputs",
        "",
        f"- Network Shapefile: `{summary['outputs']['network_shp']}`",
        f"- Evidence Shapefile: `{summary['outputs']['evidence_shp']}`",
        "",
        "## Trimmed Straight Bands",
        "",
    ]
    for row in summary["report"]["trimmed_bands"]:
        lines.append(
            f"- `{row['line_id']}`: {row['old_station_min_m']}..{row['old_station_max_m']} -> "
            f"{row['new_station_min_m']}..{row['new_station_max_m']}"
        )
    lines.extend(["", "## Smoothed Turnouts", ""])
    for row in summary["report"]["smoothed_turnouts"]:
        lines.append(
            f"- `{row['line_id']}`: angle {row['old_angle_deg']} -> {row['new_angle_deg']} deg, "
            f"support={row['support_ratio']}, mean={row['mean_distance_m']}m"
        )
    lines.extend(["", "## Rebuilt Crossovers", ""])
    if not summary["report"]["rebuilt_crossovers"]:
        lines.append("- No crossover connectors were rebuilt.")
    else:
        for row in summary["report"]["rebuilt_crossovers"]:
            lines.append(
                f"- `{row['line_id']}`: evidence `{row['start_evidence']}` -> `{row['end_evidence']}`, "
                f"breaks={row['start_break_s']}..{row['end_break_s']}m, "
                f"mode={row['rebuild_mode']}, middle={row['middle_len_m']}m, curve_fraction={row['curve_fraction']}, "
                f"angle {row['old_angle_deg']} -> {row['new_angle_deg']} deg, "
                f"endpoint_slope={row['endpoint_tangent_slope']}, support={row['support_ratio']}, mean={row['mean_distance_m']}m"
            )
    lines.extend(["", "## Turnout Boundary Bridges", ""])
    if not summary["report"]["boundary_bridges"]:
        lines.append("- No turnout boundary bridges were created.")
    else:
        for row in summary["report"]["boundary_bridges"]:
            lines.append(
                f"- `{row['line_id']}`: station {row['station_min_m']}..{row['station_max_m']}m, "
                f"gap={row['gap_m']}m, offset_delta={row['offset_delta_m']}m, "
                f"evidence=`{row['bridge_evidence']}`, support={row['support_ratio']}, "
                f"mean={row['mean_distance_m']}m"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
