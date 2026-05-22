from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export centerline candidate GeoJSON to an ESRI Shapefile.")
    parser.add_argument("--input", required=True, help="Input GeoJSON path.")
    parser.add_argument("--out", required=True, help="Output .shp path.")
    parser.add_argument("--crs-raster", help="Optional georeferenced raster whose CRS is copied to the .prj file.")
    parser.add_argument("--epsg", default="", help="Optional EPSG code used when --crs-raster is not supplied.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    if output_path.suffix.lower() != ".shp":
        raise ValueError("--out must end with .shp")

    features = load_lines(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_shapefile(features, output_path)
    write_projection(output_path.with_suffix(".prj"), args.crs_raster, args.epsg)
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "feature_count": len(features),
                "sidecars": [
                    str(output_path.with_suffix(suffix))
                    for suffix in [".shx", ".dbf", ".prj", ".cpg"]
                    if output_path.with_suffix(suffix).exists()
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def load_lines(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        coords = geometry.get("coordinates") or []
        line = [(float(x), float(y)) for x, y, *_ in coords if x is not None and y is not None]
        if len(line) < 2:
            continue
        features.append({"properties": feature.get("properties") or {}, "line": line})
    return features


def write_shapefile(features: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import shapefile
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pyshp in the active virtual environment.") from exc

    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("id_text", "C", size=40)
    writer.field("role", "C", size=24)
    writer.field("image", "C", size=80)
    writer.field("cand_id", "N", decimal=0)
    writer.field("pts", "N", decimal=0)
    writer.field("gap_px", "F", decimal=3)
    writer.field("conf", "F", decimal=4)
    writer.field("space", "C", size=16)
    for feature in features:
        props = feature["properties"]
        id_text = str(props.get("candidate_id", props.get("chain_id", "")))[:40]
        role = str(props.get("role", ""))[:24]
        writer.line([feature["line"]])
        writer.record(
            id_text,
            role,
            str(props.get("image_name", ""))[:80],
            safe_int(props.get("candidate_id", props.get("chain_id", 0))),
            int(props.get("point_count", len(feature["line"]))),
            float(props.get("mean_gap_px", props.get("length_m", 0.0))),
            float(props.get("mean_confidence", 0.0)),
            str(props.get("coordinate_space", ""))[:16],
        )
    writer.close()


def safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def write_projection(prj_path: Path, crs_raster: str | None, epsg: str) -> None:
    wkt = ""
    if crs_raster:
        import rasterio

        with rasterio.open(Path(crs_raster).expanduser().resolve()) as dataset:
            if dataset.crs is not None:
                wkt = dataset.crs.to_wkt()
    elif epsg:
        import rasterio

        wkt = rasterio.crs.CRS.from_epsg(int(epsg)).to_wkt()
    if wkt:
        prj_path.write_text(wkt, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
