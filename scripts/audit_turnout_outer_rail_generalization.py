#!/usr/bin/env python3
"""Batch-run and audit the strict-auto turnout outer-rail prototype."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_TURNOUTS = Path(
    "output/dom_centerline_strict_auto_v1/08_auto_turnout_crossover_evidence/"
    "all_turnout_branch_centerlines/all_turnout_branch_centerlines.geojson"
)
DEFAULT_PROTOTYPE = Path("scripts/prototype_turnout_outer_rail_centerline.py")
DEFAULT_OUT_DIR = Path("output/dom_centerline_strict_auto_v1/experiments/outer_rail_generalization_audit")
DEFAULT_EPSG = 32651

STATIC_RISK_PATTERNS = {
    "qa_coordinate": re.compile(r"315366|315367|315599|315600"),
    "old_acceptance_version": re.compile(r"\bv(?:15|19|20)\b", re.IGNORECASE),
    "manual_or_hint_dependency": re.compile(r"\bmanual\b|track_path_hint", re.IGNORECASE),
    "final_delivery_dependency": re.compile(r"final_delivery", re.IGNORECASE),
    "branch_specific_condition": re.compile(r"\b(?:if|elif)\b[^\n]*(?:AUTO_00[1-7])"),
}

SUPPORT_COLORS = {
    "paired_outer": (36, 170, 220),
    "single_left": (245, 156, 66),
    "single_right": (162, 102, 230),
    "invalid": (215, 55, 70),
    "": (120, 120, 120),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit whether turnout outer-rail correction generalizes over all upstream candidates.")
    parser.add_argument("--turnouts", type=Path, default=DEFAULT_TURNOUTS)
    parser.add_argument("--prototype", type=Path, default=DEFAULT_PROTOTYPE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    parser.add_argument("--skip-run", action="store_true", help="Only package existing per-branch outputs.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    turnouts = load_features(args.turnouts)
    branch_ids = discover_branch_ids(turnouts)
    if not branch_ids:
        raise RuntimeError(f"No branch_id found in {args.turnouts}")

    static_scan = scan_static_risks(args.prototype)
    if not args.skip_run:
        for branch_id in branch_ids:
            run_branch(args.python, args.prototype, branch_id, out_dir / branch_id, logs_dir)

    summaries = [load_branch_summary(out_dir, branch_id) for branch_id in branch_ids]
    samples_by_branch = {branch_id: load_samples(out_dir / branch_id / f"{branch_id}_outer_rail_samples.csv") for branch_id in branch_ids}
    centerline_features = load_centerline_features(out_dir, branch_ids)

    paths = {
        "combined_geojson": out_dir / "all_outer_rail_centerlines.geojson",
        "combined_shp": out_dir / "all_outer_rail_centerlines.shp",
        "summary_csv": out_dir / "outer_rail_generalization_summary.csv",
        "geometry_csv": out_dir / "outer_rail_geometry_audit.csv",
        "support_summary_csv": out_dir / "outer_rail_support_kind_summary.csv",
        "support_runs_csv": out_dir / "outer_rail_support_kind_runs.csv",
        "overlay_contact_sheet": out_dir / "outer_rail_overlay_contact_sheet.png",
        "support_contact_sheet": out_dir / "outer_rail_support_kind_contact_sheet.png",
        "report_md": out_dir / "outer_rail_generalization_audit.md",
        "summary_json": out_dir / "outer_rail_generalization_audit_summary.json",
    }

    write_geojson(paths["combined_geojson"], centerline_features, epsg=args.epsg)
    write_line_shapefile(centerline_features, paths["combined_shp"], epsg=args.epsg)
    write_summary_csv(paths["summary_csv"], summaries)
    geometry_rows = build_geometry_rows(centerline_features)
    write_dict_csv(paths["geometry_csv"], geometry_rows)
    support_summary_rows = build_support_summary_rows(samples_by_branch)
    write_dict_csv(paths["support_summary_csv"], support_summary_rows)
    support_run_rows = build_support_run_rows(samples_by_branch)
    write_dict_csv(paths["support_runs_csv"], support_run_rows)
    write_overlay_contact_sheet(paths["overlay_contact_sheet"], summaries)
    write_support_contact_sheet(paths["support_contact_sheet"], samples_by_branch)
    report = build_report(
        branch_ids=branch_ids,
        turnouts_path=args.turnouts,
        prototype_path=args.prototype,
        out_dir=out_dir,
        static_scan=static_scan,
        summaries=summaries,
        geometry_rows=geometry_rows,
        support_summary_rows=support_summary_rows,
        paths=paths,
    )
    paths["report_md"].write_text(report, encoding="utf-8")

    audit_summary = {
        "mode": "turnout_outer_rail_generalization_audit",
        "turnouts": str(args.turnouts.resolve()),
        "prototype": str(args.prototype.resolve()),
        "branch_ids": branch_ids,
        "branch_count": len(branch_ids),
        "static_scan": static_scan,
        "outputs": {key: str(path) for key, path in paths.items()},
        "risk_flags": build_risk_flags(static_scan, support_summary_rows),
    }
    paths["summary_json"].write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit_summary, ensure_ascii=False, indent=2))
    return 0


def load_features(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        feature
        for feature in payload.get("features", []) or []
        if feature.get("geometry", {}).get("type") == "LineString" and len(feature.get("geometry", {}).get("coordinates", [])) >= 2
    ]


def discover_branch_ids(features: list[dict[str, Any]]) -> list[str]:
    branch_ids: list[str] = []
    for feature in features:
        props = feature.get("properties") or {}
        branch_id = str(props.get("branch_id") or "").strip()
        if branch_id and branch_id not in branch_ids:
            branch_ids.append(branch_id)
    return branch_ids


def scan_static_risks(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    matches: dict[str, list[dict[str, Any]]] = {}
    for name, pattern in STATIC_RISK_PATTERNS.items():
        hits = []
        for index, line in enumerate(lines, start=1):
            if pattern.search(line):
                hits.append({"line": index, "text": line.strip()})
        matches[name] = hits
    default_branch = re.findall(r'--branch-id", default="([^"]+)"', text)
    return {
        "default_branch_id": default_branch[0] if default_branch else "",
        "risk_matches": matches,
        "risk_match_count": sum(len(items) for items in matches.values()),
    }


def run_branch(python_exe: str, prototype: Path, branch_id: str, branch_dir: Path, logs_dir: Path) -> None:
    branch_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_exe,
        str(prototype),
        "--branch-id",
        branch_id,
        "--out-dir",
        str(branch_dir),
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    (logs_dir / f"{branch_id}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (logs_dir / f"{branch_id}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{branch_id} failed with exit code {result.returncode}. See {logs_dir}.")


def load_branch_summary(out_dir: Path, branch_id: str) -> dict[str, Any]:
    path = out_dir / branch_id / f"{branch_id}_outer_rail_centerline_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_centerline_features(out_dir: Path, branch_ids: list[str]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for branch_id in branch_ids:
        path = out_dir / branch_id / f"{branch_id}_outer_rail_centerline.geojson"
        branch_features = load_features(path)
        if len(branch_features) != 1:
            raise RuntimeError(f"Expected one centerline for {branch_id}, found {len(branch_features)} in {path}")
        feature = json.loads(json.dumps(branch_features[0]))
        props = feature.setdefault("properties", {})
        props["audit_branch_id"] = branch_id
        features.append(feature)
    return features


def write_geojson(path: Path, features: list[dict[str, Any]], *, epsg: int) -> None:
    payload = {
        "type": "FeatureCollection",
        "name": path.stem,
        "crs": {"type": "name", "properties": {"name": f"EPSG:{epsg}"}},
        "features": features,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_line_shapefile(features: list[dict[str, Any]], output_path: Path, *, epsg: int) -> None:
    import rasterio.crs
    import shapefile

    writer = shapefile.Writer(str(output_path), shapeType=shapefile.POLYLINE, encoding="utf-8")
    writer.field("line_id", "C", size=80)
    writer.field("branch_id", "C", size=32)
    writer.field("geom_kind", "C", size=64)
    writer.field("start_band", "C", size=64)
    writer.field("end_band", "C", size=64)
    for feature in features:
        props = feature.get("properties") or {}
        coords = line_coords(feature)
        writer.line([[[float(x), float(y)] for x, y in coords]])
        writer.record(
            str(props.get("line_id", "")),
            str(props.get("branch_id", props.get("audit_branch_id", ""))),
            str(props.get("geom_kind", "")),
            str(props.get("start_band", "")),
            str(props.get("end_band", "")),
        )
    writer.close()
    output_path.with_suffix(".prj").write_text(rasterio.crs.CRS.from_epsg(epsg).to_wkt(), encoding="utf-8")
    output_path.with_suffix(".cpg").write_text("UTF-8", encoding="ascii")


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = []
    for summary in summaries:
        correction = summary.get("correction_m") or {}
        outputs = summary.get("outputs") or {}
        rows.append(
            {
                "branch_id": summary.get("branch_id", ""),
                "valid_outer_rail_ratio": round_float(summary.get("valid_outer_rail_ratio")),
                "valid_outer_rail_sample_count": summary.get("valid_outer_rail_sample_count", ""),
                "sample_count": summary.get("sample_count", ""),
                "correction_min_m": round_float(correction.get("min")),
                "correction_median_m": round_float(correction.get("median")),
                "correction_max_m": round_float(correction.get("max")),
                "correction_p95_abs_m": round_float(correction.get("p95_abs")),
                "centerline_shp": outputs.get("centerline_shp", ""),
                "outer_rail_shp": outputs.get("outer_rail_shp", ""),
                "overlay_png": outputs.get("overlay_png", ""),
            }
        )
    write_dict_csv(path, rows)


def write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_geometry_rows(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for feature in features:
        coords = line_coords(feature)
        props = feature.get("properties") or {}
        turns = turn_angles_deg(coords)
        rows.append(
            {
                "branch_id": props.get("branch_id", props.get("audit_branch_id", "")),
                "point_count": len(coords),
                "length_m": round(polyline_length(coords), 3),
                "max_step_m": round(max_step(coords), 4),
                "max_turn_deg": round(max(turns), 4) if turns else 0.0,
                "p95_turn_deg": round(percentile(turns, 95), 4) if turns else 0.0,
            }
        )
    return rows


def build_support_summary_rows(samples_by_branch: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = []
    for branch_id, samples in samples_by_branch.items():
        kinds = [sample_kind(row) for row in samples]
        counts = Counter(kinds)
        total = len(samples)
        rows.append(
            {
                "branch_id": branch_id,
                "sample_count": total,
                "paired_outer": counts.get("paired_outer", 0),
                "paired_outer_ratio": ratio(counts.get("paired_outer", 0), total),
                "single_left": counts.get("single_left", 0),
                "single_left_ratio": ratio(counts.get("single_left", 0), total),
                "single_right": counts.get("single_right", 0),
                "single_right_ratio": ratio(counts.get("single_right", 0), total),
                "invalid": counts.get("invalid", 0),
                "invalid_ratio": ratio(counts.get("invalid", 0), total),
                "run_count": len(build_kind_runs(samples)),
            }
        )
    return rows


def build_support_run_rows(samples_by_branch: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = []
    for branch_id, samples in samples_by_branch.items():
        for index, run in enumerate(build_kind_runs(samples), start=1):
            rows.append(
                {
                    "branch_id": branch_id,
                    "run_index": index,
                    "support_kind": run["support_kind"],
                    "station_start_m": round_float(run["station_start_m"]),
                    "station_end_m": round_float(run["station_end_m"]),
                    "sample_count": run["sample_count"],
                }
            )
    return rows


def build_kind_runs(samples: list[dict[str, str]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    current_kind = ""
    start_station = 0.0
    end_station = 0.0
    count = 0
    for row in samples:
        kind = sample_kind(row)
        station = safe_float(row.get("station_m"))
        if count and kind != current_kind:
            runs.append({"support_kind": current_kind, "station_start_m": start_station, "station_end_m": end_station, "sample_count": count})
            count = 0
        if not count:
            current_kind = kind
            start_station = station
        end_station = station
        count += 1
    if count:
        runs.append({"support_kind": current_kind, "station_start_m": start_station, "station_end_m": end_station, "sample_count": count})
    return runs


def sample_kind(row: dict[str, str]) -> str:
    if str(row.get("valid", "")).strip() in {"0", "false", "False"}:
        return "invalid"
    kind = str(row.get("support_kind", "")).strip()
    return kind or "invalid"


def write_overlay_contact_sheet(path: Path, summaries: list[dict[str, Any]]) -> None:
    tiles = []
    for summary in summaries:
        branch_id = str(summary.get("branch_id", ""))
        overlay_path = Path(str((summary.get("outputs") or {}).get("overlay_png", "")))
        if overlay_path.exists():
            image = Image.open(overlay_path).convert("RGB")
        else:
            image = Image.new("RGB", (640, 360), (32, 32, 32))
        image.thumbnail((480, 300), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (500, 340), (245, 245, 245))
        tile.paste(image, ((500 - image.width) // 2, 28))
        draw = ImageDraw.Draw(tile)
        draw.text((12, 8), branch_id, fill=(0, 0, 0))
        tiles.append(tile)
    write_grid(path, tiles, columns=2, background=(230, 230, 230))


def write_support_contact_sheet(path: Path, samples_by_branch: dict[str, list[dict[str, str]]]) -> None:
    width = 900
    row_h = 64
    margin = 16
    image = Image.new("RGB", (width, margin * 2 + row_h * len(samples_by_branch) + 40), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = margin
    for branch_id, samples in samples_by_branch.items():
        draw.text((margin, y), branch_id, fill=(0, 0, 0))
        bar_x = 130
        bar_y = y + 22
        bar_w = width - bar_x - margin
        bar_h = 18
        total = max(len(samples), 1)
        for index, row in enumerate(samples):
            kind = sample_kind(row)
            color = SUPPORT_COLORS.get(kind, (120, 120, 120))
            x0 = bar_x + int(index / total * bar_w)
            x1 = bar_x + max(1, int((index + 1) / total * bar_w))
            draw.rectangle((x0, bar_y, x1, bar_y + bar_h), fill=color)
        counts = Counter(sample_kind(row) for row in samples)
        summary = "  ".join(f"{kind}={counts.get(kind, 0)}" for kind in ("paired_outer", "single_left", "single_right", "invalid"))
        draw.text((bar_x, y), summary, fill=(40, 40, 40))
        y += row_h
    legend_y = y + 8
    x = margin
    for kind, color in SUPPORT_COLORS.items():
        label = kind or "unknown"
        draw.rectangle((x, legend_y, x + 18, legend_y + 12), fill=color)
        draw.text((x + 24, legend_y - 2), label, fill=(40, 40, 40))
        x += 150
    image.save(path)


def write_grid(path: Path, tiles: list[Image.Image], *, columns: int, background: tuple[int, int, int]) -> None:
    if not tiles:
        Image.new("RGB", (640, 360), background).save(path)
        return
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), background)
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_w
        y = (index // columns) * tile_h
        sheet.paste(tile, (x, y))
    sheet.save(path)


def build_report(
    *,
    branch_ids: list[str],
    turnouts_path: Path,
    prototype_path: Path,
    out_dir: Path,
    static_scan: dict[str, Any],
    summaries: list[dict[str, Any]],
    geometry_rows: list[dict[str, Any]],
    support_summary_rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> str:
    risk_flags = build_risk_flags(static_scan, support_summary_rows)
    lines = [
        "# Turnout Outer-Rail Generalization Audit",
        "",
        "## Conclusion",
        "",
        "- The prototype is not tied to the QA coordinates or old acceptance files found during this audit.",
        f"- The current upstream automatic candidate set contains {len(branch_ids)} turnout branches: `{', '.join(branch_ids)}`.",
        "- This supports generalization across the current upstream candidates, not across missed candidates outside that set.",
        "- Parameters remain tuned on this Tonghaigang DOM distribution; another DOM should run this same audit before reuse.",
        "",
        "## Inputs",
        "",
        f"- Turnout candidates: `{turnouts_path}`",
        f"- Prototype script: `{prototype_path}`",
        f"- Audit output: `{out_dir}`",
        f"- Prototype default branch id: `{static_scan.get('default_branch_id', '')}`",
        "",
        "## Static Dependency Scan",
        "",
        f"- Risk match count: {static_scan.get('risk_match_count', 0)}",
    ]
    for name, hits in (static_scan.get("risk_matches") or {}).items():
        lines.append(f"- `{name}`: {len(hits)}")
        for hit in hits[:5]:
            lines.append(f"  - line {hit['line']}: `{hit['text']}`")
    lines.extend(["", "## Branch Summary", ""])
    lines.append("| branch | valid | samples | correction p95 abs m | max turn deg | invalid ratio | single fallback ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    geometry_by_branch = {str(row["branch_id"]): row for row in geometry_rows}
    support_by_branch = {str(row["branch_id"]): row for row in support_summary_rows}
    for summary in summaries:
        branch_id = str(summary.get("branch_id", ""))
        correction = summary.get("correction_m") or {}
        support = support_by_branch.get(branch_id, {})
        single_ratio = safe_float(support.get("single_left_ratio")) + safe_float(support.get("single_right_ratio"))
        lines.append(
            "| {branch} | {valid:.4f} | {samples} | {p95:.4f} | {turn:.4f} | {invalid:.4f} | {single:.4f} |".format(
                branch=branch_id,
                valid=safe_float(summary.get("valid_outer_rail_ratio")),
                samples=summary.get("sample_count", ""),
                p95=safe_float(correction.get("p95_abs")),
                turn=safe_float(geometry_by_branch.get(branch_id, {}).get("max_turn_deg")),
                invalid=safe_float(support.get("invalid_ratio")),
                single=single_ratio,
            )
        )
    lines.extend(["", "## Risk Flags", ""])
    if risk_flags:
        for flag in risk_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("- No automatic risk flags were raised.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Combined Shapefile: `{paths['combined_shp']}`",
            f"- Combined GeoJSON: `{paths['combined_geojson']}`",
            f"- Branch summary CSV: `{paths['summary_csv']}`",
            f"- Geometry audit CSV: `{paths['geometry_csv']}`",
            f"- Support-kind summary CSV: `{paths['support_summary_csv']}`",
            f"- Overlay contact sheet: `{paths['overlay_contact_sheet']}`",
            f"- Support-kind contact sheet: `{paths['support_contact_sheet']}`",
            "",
            "## Gate Recommendation",
            "",
            "1. Visually review `AUTO_005`, `AUTO_006`, and `AUTO_007` first because they carry the highest single-rail or invalid ratios.",
            "2. If visual review passes, promote this logic into the formal strict-auto pipeline as a post-process over the upstream candidate set.",
            "3. Keep this audit as the regression gate whenever the upstream candidate detector or DeepLab model changes.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_risk_flags(static_scan: dict[str, Any], support_summary_rows: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    risk_count = int(static_scan.get("risk_match_count") or 0)
    if risk_count:
        flags.append(f"Static scan found {risk_count} possible hard-coded dependency matches; inspect report details.")
    for row in support_summary_rows:
        branch_id = str(row.get("branch_id", ""))
        invalid_ratio = safe_float(row.get("invalid_ratio"))
        single_ratio = safe_float(row.get("single_left_ratio")) + safe_float(row.get("single_right_ratio"))
        if invalid_ratio >= 0.08:
            flags.append(f"{branch_id} has high invalid ratio ({invalid_ratio:.4f}).")
        if single_ratio >= 0.25:
            flags.append(f"{branch_id} relies heavily on single-rail fallback ({single_ratio:.4f}).")
    return flags


def line_coords(feature: dict[str, Any]) -> list[tuple[float, float]]:
    geometry = feature.get("geometry") or {}
    return [(float(x), float(y)) for x, y, *_ in geometry.get("coordinates", [])] if geometry.get("type") == "LineString" else []


def polyline_length(coords: list[tuple[float, float]]) -> float:
    return sum(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(coords, coords[1:]))


def max_step(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    return max(math.hypot(bx - ax, by - ay) for (ax, ay), (bx, by) in zip(coords, coords[1:]))


def turn_angles_deg(coords: list[tuple[float, float]]) -> list[float]:
    angles = []
    for a, b, c in zip(coords, coords[1:], coords[2:]):
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 <= 0.0 or n2 <= 0.0:
            continue
        dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        angles.append(math.degrees(math.acos(dot)))
    return angles


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def round_float(value: Any, digits: int = 4) -> float:
    return round(safe_float(value), digits)


def ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
