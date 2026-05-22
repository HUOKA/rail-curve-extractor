from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import compute_extent


def export_usda_basis_curves(path: Path, points: np.ndarray, width: float = 0.05) -> None:
    if len(points) < 2:
        raise ValueError("At least two points are required to export a curve.")

    path.parent.mkdir(parents=True, exist_ok=True)
    min_xyz, max_xyz = compute_extent(points)
    curve_vertex_count = len(points)
    points_literal = ",\n                ".join(_format_point(p) for p in points)
    extent_literal = f"[{_format_point(min_xyz)}, {_format_point(max_xyz)}]"
    content = f"""#usda 1.0
(
    defaultPrim = "RailTrack"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RailTrack"
{{
    def BasisCurves "Centerline"
    {{
        uniform token type = "linear"
        uniform token wrap = "nonperiodic"
        int[] curveVertexCounts = [{curve_vertex_count}]
        point3f[] points = [
                {points_literal}
        ]
        float[] widths = [{width:.6f}]
        float3[] extent = {extent_literal}
    }}
}}
"""
    path.write_text(content, encoding="utf-8")


def export_usda_multi_basis_curves(path: Path, curves: list[np.ndarray], width: float = 0.05) -> None:
    valid_curves = [curve for curve in curves if len(curve) >= 2]
    if not valid_curves:
        raise ValueError("At least one curve with two points is required to export curves.")

    path.parent.mkdir(parents=True, exist_ok=True)
    all_points = np.vstack(valid_curves)
    min_xyz, max_xyz = compute_extent(all_points)
    extent_literal = f"[{_format_point(min_xyz)}, {_format_point(max_xyz)}]"
    curve_blocks = []
    for index, points in enumerate(valid_curves, start=1):
        points_literal = ",\n                ".join(_format_point(p) for p in points)
        curve_blocks.append(
            f'''    def BasisCurves "Track_{index}_Centerline"
    {{
        uniform token type = "linear"
        uniform token wrap = "nonperiodic"
        int[] curveVertexCounts = [{len(points)}]
        point3f[] points = [
                {points_literal}
        ]
        float[] widths = [{width:.6f}]
    }}'''
        )

    content = f"""#usda 1.0
(
    defaultPrim = "RailTracks"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RailTracks"
{{
{chr(10).join(curve_blocks)}
    float3[] extent = {extent_literal}
}}
"""
    path.write_text(content, encoding="utf-8")


def _format_point(point: np.ndarray) -> str:
    x, y, z = [float(v) for v in point]
    return f"({x:.6f}, {y:.6f}, {z:.6f})"
