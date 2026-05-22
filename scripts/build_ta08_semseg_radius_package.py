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
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_topology_centerline_review_v15_semseg_radius")
DEFAULT_EPSG = 32651


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package evidence-first TA08 centerline with semantic-segmentation radius diagnostics.")
    parser.add_argument("--base-network", type=Path, default=DEFAULT_BASE_NETWORK)
    parser.add_argument("--base-evidence", type=Path, default=DEFAULT_BASE_EVIDENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-step-m", type=float, default=1.0)
    parser.add_argument("--fit-window-m", type=float, default=8.0)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    network_features = load_features(args.base_network.expanduser().resolve())
    evidence_features = load_features(args.base_evidence.expanduser().resolve())
    normalized_network = mark_evidence_first_ta08(network_features)

    radius_result = build_radius_diagnostics(
        normalized_network,
        evidence_features=evidence_features,
        sample_step_m=args.sample_step_m,
        fit_window_m=args.fit_window_m,
    )

    network_geojson = out_dir / "deeplab_topology_centerline_network.geojson"
    evidence_geojson = out_dir / "deeplab_topology_evidence.geojson"
    samples_geojson = out_dir / "ta08_semseg_radius_samples.geojson"
    sections_geojson = out_dir / "ta08_semseg_radius_sections.geojson"
    summary_path = out_dir / "summary.json"
    review_path = out_dir / "REVIEW.md"

    topo.write_geojson(network_geojson, normalized_network, epsg=args.epsg)
    topo.write_centerline_shapefile(normalized_network, network_geojson.with_suffix(".shp"), epsg=args.epsg)
    topo.write_centerline_qml(network_geojson.with_suffix(".qml"))
    topo.write_geojson(evidence_geojson, evidence_features, epsg=args.epsg)
    topo.write_evidence_shapefile(evidence_features, evidence_geojson.with_suffix(".shp"), epsg=args.epsg)
    topo.write_evidence_qml(evidence_geojson.with_suffix(".qml"))
    write_point_geojson(samples_geojson, radius_result["sample_features"], epsg=args.epsg)
    write_radius_sample_shapefile(radius_result["sample_features"], samples_geojson.with_suffix(".shp"), epsg=args.epsg)
    topo.write_geojson(sections_geojson, radius_result["section_features"], epsg=args.epsg)
    write_radius_section_shapefile(radius_result["section_features"], sections_geojson.with_suffix(".shp"), epsg=args.epsg)

    summary = {
        "mode": "semseg_derived_ta08_radius_package_v1",
        "policy": "No hard R350 prior is applied. Radius is estimated from DeepLab/support-chain and gauge-pair evidence.",
        "base_network": str(args.base_network.expanduser().resolve()),
        "base_evidence": str(args.base_evidence.expanduser().resolve()),
        "sample_step_m": args.sample_step_m,
        "fit_window_m": args.fit_window_m,
        "outputs": {
            "network_geojson": str(network_geojson),
            "network_shp": str(network_geojson.with_suffix(".shp")),
            "network_qml": str(network_geojson.with_suffix(".qml")),
            "evidence_geojson": str(evidence_geojson),
            "evidence_shp": str(evidence_geojson.with_suffix(".shp")),
            "radius_samples_geojson": str(samples_geojson),
            "radius_samples_shp": str(samples_geojson.with_suffix(".shp")),
            "radius_sections_geojson": str(sections_geojson),
            "radius_sections_shp": str(sections_geojson.with_suffix(".shp")),
            "summary_json": str(summary_path),
            "review_md": str(review_path),
        },
        "feature_count": len(normalized_network),
        "radius_sources": radius_result["source_summaries"],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(review_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("features") or [])


def mark_evidence_first_ta08(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for feature in features:
        props = dict(feature.get("properties") or {})
        if str(props.get("line_id", "")) == "TURNOUT_TA08":
            props.update(
                {
                    "experiment": "semseg_radius_diagnostics_no_hard_r350",
                    "postprocess_policy": "evidence_first_centerline_with_semseg_radius_diagnostics",
                    "review_note": "TA08 centerline kept evidence-first; radius diagnostics are derived from segmentation evidence, not forced to 350m.",
                    "self_note": "evidence-first TA08; inspect ta08_semseg_radius_* diagnostic layers for radius estimates",
                }
            )
        marked.append({"type": "Feature", "properties": props, "geometry": feature["geometry"]})
    return marked


def build_radius_diagnostics(
    network_features: list[dict[str, Any]],
    *,
    evidence_features: list[dict[str, Any]],
    sample_step_m: float,
    fit_window_m: float,
) -> dict[str, Any]:
    mainline = find_by_line_id(network_features, "BAND_mainline_2_track_0")
    ta08 = find_by_line_id(network_features, "TURNOUT_TA08")
    if mainline is None or ta08 is None:
        raise ValueError("Network must contain BAND_mainline_2_track_0 and TURNOUT_TA08.")

    guide = topo.LinearGuide(topo.line_coords(mainline)[0], topo.line_coords(mainline)[-1])
    sources: list[dict[str, Any]] = [
        {
            "source_id": "TURNOUT_TA08",
            "source_role": "network_centerline",
            "feature": ta08,
            "note": "evidence-first output centerline",
        }
    ]
    support = find_by_line_id(evidence_features, "DLV1_SUPPORT_08")
    if support is not None:
        sources.append(
            {
                "source_id": "DLV1_SUPPORT_08",
                "source_role": "deeplab_support_chain",
                "feature": support,
                "note": "DeepLab support-chain centerline evidence",
            }
        )
    gp02 = find_by_seq_id(evidence_features, "TA08_GP02")
    if gp02 is not None:
        sources.append(
            {
                "source_id": "TA08_GP02",
                "source_role": "deeplab_gauge_pair_centerline",
                "feature": gp02,
                "note": "paired-rail centerline evidence",
            }
        )

    sample_features: list[dict[str, Any]] = []
    section_features: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    sample_index = 1
    for source in sources:
        coords = topo.line_coords(source["feature"])
        source_samples = radius_samples_for_source(
            source["source_id"],
            source["source_role"],
            source["note"],
            coords,
            guide=guide,
            sample_step_m=sample_step_m,
            fit_window_m=fit_window_m,
            sample_index_start=sample_index,
        )
        sample_index += len(source_samples)
        sample_features.extend(source_samples)
        summary = summarize_radius_samples(source["source_id"], source["source_role"], source_samples)
        source_summaries.append(summary)
        section_features.append(section_feature(source, coords, summary))

    return {
        "sample_features": sample_features,
        "section_features": section_features,
        "source_summaries": source_summaries,
    }


def radius_samples_for_source(
    source_id: str,
    source_role: str,
    note: str,
    coords: list[tuple[float, float]],
    *,
    guide: topo.LinearGuide,
    sample_step_m: float,
    fit_window_m: float,
    sample_index_start: int,
) -> list[dict[str, Any]]:
    station_offsets = topo.station_offsets_for_coords(coords, guide=guide)
    sampled = topo.resample_station_offsets(station_offsets, step_m=sample_step_m)
    features: list[dict[str, Any]] = []
    for offset_index, (station, offset) in enumerate(sampled):
        fit = local_quadratic_radius(sampled, station=station, window_m=fit_window_m)
        x, y = guide.point_at(station, offset)
        radius_m = fit["radius_m"]
        radius_class = classify_radius(radius_m)
        props = {
            "sample_id": f"R{sample_index_start + offset_index:04d}",
            "source_id": source_id,
            "source_role": source_role,
            "station_m": round(station, 3),
            "offset_m": round(offset, 3),
            "radius_m": round(radius_m, 3) if math.isfinite(radius_m) else 99999.0,
            "signed_r_m": round(fit["signed_radius_m"], 3) if math.isfinite(fit["signed_radius_m"]) else 99999.0,
            "curvature": round(fit["curvature"], 8),
            "slope": round(fit["slope"], 6),
            "fit_n": fit["fit_n"],
            "fit_window": round(fit_window_m, 3),
            "radius_cls": radius_class,
            "note": note,
        }
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [round(x, 6), round(y, 6)]},
            }
        )
    return features


def local_quadratic_radius(
    points: list[tuple[float, float]],
    *,
    station: float,
    window_m: float,
) -> dict[str, Any]:
    local = [(s - station, t) for s, t in points if abs(s - station) <= window_m]
    if len(local) < 5:
        return {"radius_m": float("inf"), "signed_radius_m": float("inf"), "curvature": 0.0, "slope": 0.0, "fit_n": len(local)}
    a, b, _ = solve_quadratic(local)
    curvature = (2.0 * a) / ((1.0 + b * b) ** 1.5)
    if abs(curvature) <= 1e-9:
        radius = float("inf")
        signed_radius = float("inf")
    else:
        signed_radius = 1.0 / curvature
        radius = abs(signed_radius)
    return {
        "radius_m": radius,
        "signed_radius_m": signed_radius,
        "curvature": curvature,
        "slope": b,
        "fit_n": len(local),
    }


def solve_quadratic(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    n = float(len(points))
    sx = sum(x for x, _ in points)
    sx2 = sum(x * x for x, _ in points)
    sx3 = sum(x * x * x for x, _ in points)
    sx4 = sum(x * x * x * x for x, _ in points)
    sy = sum(y for _, y in points)
    sxy = sum(x * y for x, y in points)
    sx2y = sum(x * x * y for x, y in points)
    matrix = [
        [sx4, sx3, sx2, sx2y],
        [sx3, sx2, sx, sxy],
        [sx2, sx, n, sy],
    ]
    return gaussian_elimination_3x4(matrix)


def gaussian_elimination_3x4(matrix: list[list[float]]) -> tuple[float, float, float]:
    rows = [row[:] for row in matrix]
    for pivot_index in range(3):
        pivot_row = max(range(pivot_index, 3), key=lambda row_index: abs(rows[row_index][pivot_index]))
        rows[pivot_index], rows[pivot_row] = rows[pivot_row], rows[pivot_index]
        pivot = rows[pivot_index][pivot_index]
        if abs(pivot) <= 1e-12:
            return 0.0, 0.0, rows[2][3] if abs(rows[2][2]) > 1e-12 else 0.0
        for column in range(pivot_index, 4):
            rows[pivot_index][column] /= pivot
        for row_index in range(3):
            if row_index == pivot_index:
                continue
            factor = rows[row_index][pivot_index]
            for column in range(pivot_index, 4):
                rows[row_index][column] -= factor * rows[pivot_index][column]
    return rows[0][3], rows[1][3], rows[2][3]


def classify_radius(radius_m: float) -> str:
    if not math.isfinite(radius_m) or radius_m >= 2500.0:
        return "straight_or_flat"
    if radius_m < 120.0:
        return "sharp_or_noisy"
    if radius_m < 260.0:
        return "tight_curve"
    if radius_m <= 520.0:
        return "turnout_curve_range"
    return "large_radius_curve"


def summarize_radius_samples(source_id: str, source_role: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    radii = [
        float((feature.get("properties") or {}).get("radius_m", 99999.0))
        for feature in samples
        if float((feature.get("properties") or {}).get("radius_m", 99999.0)) < 2500.0
    ]
    stations = [float((feature.get("properties") or {}).get("station_m", 0.0)) for feature in samples]
    radius_classes: dict[str, int] = {}
    for feature in samples:
        radius_class = str((feature.get("properties") or {}).get("radius_cls", ""))
        radius_classes[radius_class] = radius_classes.get(radius_class, 0) + 1
    summary = {
        "source_id": source_id,
        "source_role": source_role,
        "sample_count": len(samples),
        "station_min_m": round(min(stations), 3) if stations else 0.0,
        "station_max_m": round(max(stations), 3) if stations else 0.0,
        "radius_class_counts": radius_classes,
    }
    if radii:
        summary.update(quantile_summary(radii))
    else:
        summary.update({"radius_p25_m": None, "radius_median_m": None, "radius_p75_m": None, "radius_p90_m": None})
    return summary


def quantile_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def pick(p: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return round(ordered[index], 3)

    return {
        "radius_p25_m": pick(0.25),
        "radius_median_m": pick(0.50),
        "radius_p75_m": pick(0.75),
        "radius_p90_m": pick(0.90),
    }


def section_feature(source: dict[str, Any], coords: list[tuple[float, float]], summary: dict[str, Any]) -> dict[str, Any]:
    props = {
        "source_id": source["source_id"],
        "source_role": source["source_role"],
        "sample_n": summary["sample_count"],
        "s0_m": summary["station_min_m"],
        "s1_m": summary["station_max_m"],
        "r_p25_m": summary.get("radius_p25_m") or 0.0,
        "r_med_m": summary.get("radius_median_m") or 0.0,
        "r_p75_m": summary.get("radius_p75_m") or 0.0,
        "r_p90_m": summary.get("radius_p90_m") or 0.0,
        "note": source["note"],
    }
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in coords]},
    }


def find_by_line_id(features: list[dict[str, Any]], line_id: str) -> dict[str, Any] | None:
    for feature in features:
        if str((feature.get("properties") or {}).get("line_id", "")) == line_id:
            return feature
    return None


def find_by_seq_id(features: list[dict[str, Any]], seq_id: str) -> dict[str, Any] | None:
    for feature in features:
        if str((feature.get("properties") or {}).get("seq_id", "")) == seq_id:
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


def write_radius_sample_shapefile(features: list[dict[str, Any]], path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(path), shapeType=shapefile.POINT, encoding="utf-8")
    writer.field("sample_id", "C", size=12)
    writer.field("src_id", "C", size=32)
    writer.field("src_role", "C", size=32)
    writer.field("station_m", "F", decimal=3)
    writer.field("offset_m", "F", decimal=3)
    writer.field("radius_m", "F", decimal=3)
    writer.field("signed_r", "F", decimal=3)
    writer.field("curv", "F", decimal=8)
    writer.field("slope", "F", decimal=6)
    writer.field("fit_n", "N", size=4)
    writer.field("r_class", "C", size=24)
    writer.field("note", "C", size=96)
    for feature in features:
        props = feature.get("properties") or {}
        x, y = feature["geometry"]["coordinates"]
        writer.point(float(x), float(y))
        writer.record(
            str(props.get("sample_id", ""))[:12],
            str(props.get("source_id", ""))[:32],
            str(props.get("source_role", ""))[:32],
            topo.safe_float(props.get("station_m", 0.0)),
            topo.safe_float(props.get("offset_m", 0.0)),
            topo.safe_float(props.get("radius_m", 0.0)),
            topo.safe_float(props.get("signed_r_m", 0.0)),
            topo.safe_float(props.get("curvature", 0.0)),
            topo.safe_float(props.get("slope", 0.0)),
            int(topo.safe_float(props.get("fit_n", 0))),
            str(props.get("radius_cls", ""))[:24],
            str(props.get("note", ""))[:96],
        )
    writer.close()
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    topo.write_projection(path.with_suffix(".prj"), epsg)


def write_radius_section_shapefile(features: list[dict[str, Any]], path: Path, *, epsg: int) -> None:
    import shapefile

    writer = shapefile.Writer(str(path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("src_id", "C", size=32)
    writer.field("src_role", "C", size=32)
    writer.field("sample_n", "N", size=5)
    writer.field("s0_m", "F", decimal=3)
    writer.field("s1_m", "F", decimal=3)
    writer.field("r_p25_m", "F", decimal=3)
    writer.field("r_med_m", "F", decimal=3)
    writer.field("r_p75_m", "F", decimal=3)
    writer.field("r_p90_m", "F", decimal=3)
    writer.field("note", "C", size=96)
    for feature in features:
        props = feature.get("properties") or {}
        writer.line([topo.line_coords(feature)])
        writer.record(
            str(props.get("source_id", ""))[:32],
            str(props.get("source_role", ""))[:32],
            int(topo.safe_float(props.get("sample_n", 0))),
            topo.safe_float(props.get("s0_m", 0.0)),
            topo.safe_float(props.get("s1_m", 0.0)),
            topo.safe_float(props.get("r_p25_m", 0.0)),
            topo.safe_float(props.get("r_med_m", 0.0)),
            topo.safe_float(props.get("r_p75_m", 0.0)),
            topo.safe_float(props.get("r_p90_m", 0.0)),
            str(props.get("note", ""))[:96],
        )
    writer.close()
    path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    topo.write_projection(path.with_suffix(".prj"), epsg)


def write_review(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# TA08 Semantic-Segmentation Radius Package",
        "",
        "No hard R350 prior is applied in this package. The network centerline is the evidence-first baseline, and radius is reported as diagnostics from segmentation-derived centerline evidence.",
        "",
        "## Outputs",
        "",
        f"- Network Shapefile: `{summary['outputs']['network_shp']}`",
        f"- Radius sample points: `{summary['outputs']['radius_samples_shp']}`",
        f"- Radius source sections: `{summary['outputs']['radius_sections_shp']}`",
        f"- Evidence Shapefile: `{summary['outputs']['evidence_shp']}`",
        "",
        "## Radius Sources",
        "",
    ]
    for source in summary["radius_sources"]:
        lines.append(
            f"- `{source['source_id']}`: samples={source['sample_count']}, "
            f"station={source['station_min_m']}..{source['station_max_m']}m, "
            f"radius_median={source.get('radius_median_m')}, p25={source.get('radius_p25_m')}, p75={source.get('radius_p75_m')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
