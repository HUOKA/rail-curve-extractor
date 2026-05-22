#!/usr/bin/env python3
"""Package the preferred DeepLab V1 centerline network for QGIS review."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_REFINED = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_refined_deeplab_v1_thr050/refined_centerline_network.geojson")
DEFAULT_MAIN = Path("output/raw_dom_roi_fullpass_v1/rail_centerline_refined_deeplab_v1_thr050/main_centerline.geojson")
DEFAULT_OUT_DIR = Path("output/raw_dom_roi_fullpass_v1/deeplab_centerline_network_v1")
DEFAULT_EPSG = 32651


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the recommended DeepLab V1 centerline network.")
    parser.add_argument("--refined", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    parser.add_argument("--main-smooth-window", type=int, default=21)
    parser.add_argument("--support-smooth-window", type=int, default=9)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    refined = load_features(args.refined)
    mainline = load_features(args.main)
    if not mainline:
        raise RuntimeError(f"No main centerline feature found: {args.main}")

    features: list[dict[str, Any]] = []
    main_feature = clone_feature(mainline[0])
    main_feature = smooth_feature(main_feature, args.main_smooth_window)
    main_feature["properties"] = {
        **(main_feature.get("properties") or {}),
        "line_id": "DLV1_MAIN_2",
        "role": "mainline",
        "network_source": "deeplab_v1_stitched_main_centerline",
        "review_status": "preferred_for_qgis_review",
        "length_m": round(polyline_length(line_coords(main_feature)), 3),
    }
    features.append(main_feature)

    support_count = 0
    for feature in refined:
        props = feature.get("properties") or {}
        if str(props.get("role", "")) == "main_path":
            continue
        coords = line_coords(feature)
        if len(coords) < 2:
            continue
        support_count += 1
        packaged = clone_feature(feature)
        packaged = smooth_feature(packaged, args.support_smooth_window)
        chain_id = str(props.get("chain_id", support_count))
        packaged["properties"] = {
            **props,
            "line_id": f"DLV1_SUPPORT_{support_count:02d}",
            "orig_chain": chain_id,
            "role": "support",
            "network_source": "deeplab_v1_refined_support_chain",
            "review_status": "auto_candidate_for_qgis_review",
            "length_m": round(polyline_length(coords), 3),
        }
        features.append(packaged)

    geojson_path = out_dir / "deeplab_centerline_network_v1.geojson"
    summary_path = out_dir / "summary.json"
    qml_path = out_dir / "deeplab_centerline_network_v1.qml"
    write_geojson(geojson_path, features, epsg=args.epsg)
    write_qml(qml_path)
    summary = {
        "mode": "deeplab_v1_preferred_review_network",
        "refined_input": str(args.refined.resolve()),
        "main_input": str(args.main.resolve()),
        "output_geojson": str(geojson_path),
        "output_qml": str(qml_path),
        "feature_count": len(features),
        "mainline_count": 1,
        "support_count": support_count,
        "policy": "replace noisy refined main_path chains with stitched main_centerline; keep automatic support chains only",
        "smoothing": {
            "main_smooth_window": args.main_smooth_window,
            "support_smooth_window": args.support_smooth_window,
            "method": "moving average on ordered line coordinates with endpoints preserved",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_features(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [feature for feature in data.get("features", []) if line_coords(feature)]


def clone_feature(feature: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(feature))


def smooth_feature(feature: dict[str, Any], window: int) -> dict[str, Any]:
    coords = line_coords(feature)
    smoothed = smooth_coords(coords, window)
    out = clone_feature(feature)
    out["geometry"]["coordinates"] = [[round(x, 6), round(y, 6)] for x, y in smoothed]
    return out


def smooth_coords(coords: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    if len(coords) < 5 or window <= 1:
        return coords
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if len(coords) <= window:
        window = len(coords) if len(coords) % 2 == 1 else len(coords) - 1
    if window < 3:
        return coords
    half = window // 2
    result: list[tuple[float, float]] = []
    for index, point in enumerate(coords):
        if index < 2 or index >= len(coords) - 2:
            result.append(point)
            continue
        lo = max(0, index - half)
        hi = min(len(coords), index + half + 1)
        xs = [coords[i][0] for i in range(lo, hi)]
        ys = [coords[i][1] for i in range(lo, hi)]
        result.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return result


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "LineString":
        return []
    return [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates", [])]


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(coords, coords[1:]))


def write_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_qml(path: Path) -> None:
    path.write_text(
        """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
  <renderer-v2 type="categorizedSymbol" attr="role" symbollevels="0" enableorderby="0" forceraster="0">
    <categories>
      <category value="mainline" symbol="0" label="mainline" render="true"/>
      <category value="support" symbol="1" label="support" render="true"/>
    </categories>
    <symbols>
      <symbol type="line" name="0" alpha="1">
        <layer class="SimpleLine" pass="0" locked="0" enabled="1">
          <Option type="Map">
            <Option name="line_color" value="255,40,220,255"/>
            <Option name="line_width" value="0.55"/>
            <Option name="line_width_unit" value="MM"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="line" name="1" alpha="1">
        <layer class="SimpleLine" pass="0" locked="0" enabled="1">
          <Option type="Map">
            <Option name="line_color" value="20,220,90,255"/>
            <Option name="line_width" value="0.42"/>
            <Option name="line_width_unit" value="MM"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <layerGeometryType>1</layerGeometryType>
</qgis>
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
